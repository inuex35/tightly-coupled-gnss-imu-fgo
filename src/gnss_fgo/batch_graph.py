"""Batch graph construction, RINEX frontend, and whole-run LM solver."""

import time

import cssrlib.gnss as gn
import cssrlib.rinex as rn
import gtsam
import numpy as np
from cssrlib.ephemeris import satposs
from cssrlib.gnss import ecef2pos, rSigRnx, sat2prn, uGNSS, uTYP
from cssrlib.rtk import rtkpos
from gtsam.symbol_shorthand import B, N, V, X

from ._extra_factors import (
    NhcFactorCalib,
    SingleDifferenceDopplerFactorArm,
)
from .utils import cauchy_noise, enu_frame, huber_noise, make_imu_params


def is_still_imu_window(acc: np.ndarray, gyro: np.ndarray,
                        bias_gyro_ref: np.ndarray | None = None) -> bool:
    """IMU stillness check for batch ZUPT."""
    if len(acc) < 5:
        return False
    acc_std = float(np.sqrt(np.mean(np.sum((acc - acc.mean(axis=0)) ** 2, axis=1))))
    gyro_std = float(np.sqrt(np.mean(np.sum((gyro - gyro.mean(axis=0)) ** 2, axis=1))))
    resid = gyro if bias_gyro_ref is None else (gyro - bias_gyro_ref)
    gyro_med = float(np.median(np.linalg.norm(resid, axis=1)))
    return acc_std <= 0.55 and gyro_std <= 0.030 and gyro_med <= 0.020


# ── Row layout for batch measurement arrays ───────────────────────────

EPOCH, FREQ, LAM, WEIGHT, ARC_REF, ARC_TGT = 0, 1, 2, 3, 4, 5
PR_RR, PR_BR, PR_RT, PR_BT = 6, 7, 8, 9          # pseudoranges [m]: rover/base x ref/tgt
CP_RR, CP_BR, CP_RT, CP_BT = 10, 11, 12, 13      # carriers [m]
SAT_RR, SAT_RT, SAT_BR, SAT_BT = (slice(14, 17), slice(17, 20),   # sat ECEF at rover
                                  slice(20, 23), slice(23, 26))   # / base signal time
SATNO_REF, SATNO_TGT = 26, 27
N_DD_COLS = 28

D_EPOCH, D_SAT, D_HZ, D_LAM, D_EL = 0, 1, 2, 3, 4
D_SATPOS, D_SATVEL = slice(5, 8), slice(8, 11)
N_DOP_COLS = 11


def sat_points(r):
    """(ref@rover, tgt@rover, ref@base, tgt@base) satellite positions."""
    return (gtsam.Point3(*r[SAT_RR]), gtsam.Point3(*r[SAT_RT]),
            gtsam.Point3(*r[SAT_BR]), gtsam.Point3(*r[SAT_BT]))


# ── Batch RINEX front end ─────────────────────────────────────────────

SYSTEMS = (uGNSS.GPS, uGNSS.GAL, uGNSS.QZS)
SIGS_ROVER = ["GC1C", "GC2W", "GL1C", "GL2W", "GS1C", "GS2W", "GD1C", "GD2W",
              "EC1C", "EC5Q", "EL1C", "EL5Q", "ES1C", "ES5Q", "ED1C", "ED5Q",
              "JC1C", "JC2L", "JL1C", "JL2L", "JS1C", "JS2L", "JD1C", "JD2L"]
SIGS_BASE = ["GC1C", "GC2W", "GL1C", "GL2W", "GS1C", "GS2W",
             "EC1X", "EC5X", "EL1X", "EL5X", "ES1X", "ES5X",
             "JC1C", "JC2X", "JL1C", "JL2X", "JS1C", "JS2X"]


def _up(pos_ecef):
    llh = gn.ecef2pos(np.asarray(pos_ecef, float))
    return np.array([np.cos(llh[0]) * np.cos(llh[1]),
                     np.cos(llh[0]) * np.sin(llh[1]), np.sin(llh[0])])


def _doppler_rows(out, k, obs, rs, vs, pos0, up, elmin):
    for i, s in enumerate(obs.sat):
        s = int(s)
        sys_ = sat2prn(s)[0]
        if sys_ not in SYSTEMS or not np.any(rs[i, :3]):
            continue
        los = rs[i, :3] - pos0
        el = np.arcsin(los @ up / np.linalg.norm(los))
        if el < elmin:
            continue
        for f in range(obs.D.shape[1]):
            if obs.D[i, f] != 0.0:
                lam = obs.sig[sys_][uTYP.L][f].wavelength()
                out.append((k, s, obs.D[i, f], lam, el, *rs[i, :3], *vs[i, :3]))
                break


def prepare(rover_obs, base_obs, nav_file, base_ecef=None, max_epochs=None,
            elmin_deg=15.0, gf_jump=0.05, sigs_rover=SIGS_ROVER, sigs_base=SIGS_BASE,
            log=print):
    """Decode the RINEX files and build the batch measurement arrays."""
    dec, decb = rn.rnxdec(), rn.rnxdec()
    dec.setSignals([rSigRnx(s) for s in sigs_rover])
    decb.setSignals([rSigRnx(s) for s in sigs_base])
    nav = gn.Nav()
    dec.decode_nav(nav_file, nav)
    dec.decode_obsh(rover_obs)
    decb.decode_obsh(base_obs)
    rb = np.asarray(base_ecef if base_ecef is not None else decb.pos, float)
    nav.rb = list(rb)
    nav.elmin = np.deg2rad(elmin_deg)
    rtk = rtkpos(nav, dec.pos)
    pos0 = np.array(dec.pos, float)
    nav.x[0:3] = pos0
    up = _up(pos0)

    rows, dop, tow = [], [], []
    arc_of, last_used, last_gf = {}, {}, {}
    last_tb, n_arc = None, 0
    t0 = time.perf_counter()
    for ne, (obs, obsb, _) in enumerate(rn.sync_obs_hold(dec, decb, maxage=nav.maxtdiff)):
        if max_epochs and ne >= max_epochs:
            break
        k = len(tow)
        tow.append(gn.time2gpst(obs.t)[1])
        dd = None
        if obsb is not None:
            dd = rtk.prepare_double_difference_measurements(obs, obsb, pos_pred=dec.pos)
        if dd is not None:
            rs, vs = dd.rs, dd.vs
        elif len(obs.sat):
            rs, vs, _, _, _ = satposs(obs, nav)
        else:
            continue
        _doppler_rows(dop, k, obs, rs, vs, pos0, up, nav.elmin)
        if dd is None:
            continue
        tb = gn.time2gpst(obsb.t)[1]
        fresh_base, last_tb = tb != last_tb, tb

        slip_gf = set()
        for kk, s in enumerate(dd.sat):
            s, iu = int(s), dd.iu[kk]
            sys_ = sat2prn(s)[0]
            if sys_ not in SYSTEMS or obs.L[iu, 0] == 0.0 or obs.L[iu, 1] == 0.0:
                continue
            gf = (obs.L[iu, 0] * obs.sig[sys_][uTYP.L][0].wavelength()
                  - obs.L[iu, 1] * obs.sig[sys_][uTYP.L][1].wavelength())
            prev = last_gf.get(s)
            if prev is not None and prev[0] == k - 1 and abs(gf - prev[1]) > gf_jump:
                slip_gf.add(s)
            last_gf[s] = (k, gf)

        def arc(s, f, iu, ir):
            nonlocal n_arc
            key = (s, f)
            slip = (last_used.get(key) != k - 1 or (obs.lli[iu, f] & 1)
                    or (fresh_base and (obsb.lli[ir, f] & 1)) or s in slip_gf)
            if key not in arc_of or slip:
                arc_of[key], n_arc = n_arc, n_arc + 1
            return arc_of[key]

        by_sys = {}
        for kk, s in enumerate(dd.sat):
            sys_ = sat2prn(int(s))[0]
            if sys_ in SYSTEMS and dd.el[kk] >= nav.elmin:
                by_sys.setdefault(sys_, []).append(kk)
        used_now = []
        for sys_, ks in by_sys.items():
            for f in range(nav.nf):
                lam = obs.sig[sys_][uTYP.L][f].wavelength()
                ok = [kk for kk in ks
                      if 0.0 not in (obs.P[dd.iu[kk], f], obsb.P[dd.ir[kk], f],
                                     obs.L[dd.iu[kk], f], obsb.L[dd.ir[kk], f])]
                if len(ok) < 2:
                    continue
                ridx = max(ok, key=lambda kk: dd.el[kk])
                sr, iur, irr = int(dd.sat[ridx]), dd.iu[ridx], dd.ir[ridx]
                a_ref = arc(sr, f, iur, irr)
                used_now.append((sr, f))
                for kk in ok:
                    if kk == ridx:
                        continue
                    st, iut, irt = int(dd.sat[kk]), dd.iu[kk], dd.ir[kk]
                    a_tgt = arc(st, f, iut, irt)
                    used_now.append((st, f))
                    w = 1.0 / max(np.sin(min(dd.el[kk], dd.el[ridx])), 0.1)
                    rows.append((
                        k, f, lam, w, a_ref, a_tgt,
                        obs.P[iur, f], obsb.P[irr, f], obs.P[iut, f], obsb.P[irt, f],
                        obs.L[iur, f] * lam, obsb.L[irr, f] * lam,
                        obs.L[iut, f] * lam, obsb.L[irt, f] * lam,
                        *dd.rs[iur, :3], *dd.rs[iut, :3],
                        *dd.rsb[irr, :3], *dd.rsb[irt, :3], sr, st))
        for key in used_now:
            last_used[key] = k
    out = dict(rows=np.array(rows, dtype=float).reshape(-1, N_DD_COLS),
               dop=np.array(dop, dtype=float).reshape(-1, N_DOP_COLS),
               tow=np.array(tow), rb=rb, pos0=pos0)
    log(f"front end: {len(tow)} epochs, {len(rows)} DD rows, {len(dop)} Doppler rows, "
        f"{n_arc} arcs ({time.perf_counter() - t0:.1f}s)")
    return out


def save(path, data):
    np.savez_compressed(path, **data)


def load(path):
    z = np.load(path)
    return {key: z[key] for key in ("rows", "dop", "tow", "rb", "pos0")}


# ── GNSS-only batch graph ─────────────────────────────────────────────

GNSS_DIMS = {"x": 3, "n": 1}


def add_ambiguities(graph, init, rows, sig_amb, seen=None):
    """N(a) per arc, seeded with SD carrier-minus-code, plus a weak prior."""
    seen = set() if seen is None else seen
    prior = gtsam.noiseModel.Isotropic.Sigma(1, sig_amb)
    for r in rows:
        lam = r[LAM]
        for a, cpr, cpb, prr, prb in (
                (int(r[ARC_REF]), r[CP_RR], r[CP_BR], r[PR_RR], r[PR_BR]),
                (int(r[ARC_TGT]), r[CP_RT], r[CP_BT], r[PR_RT], r[PR_BT])):
            if a not in seen:
                seen.add(a)
                sd = ((cpr - cpb) - (prr - prb)) / lam
                init.insert(N(a), float(sd))
                graph.add(gtsam.PriorFactorDouble(N(a), float(sd), prior))
    return seen


def build_gnss_graph(rows, tow, rb, pos0, cfg):
    graph, init = gtsam.NonlinearFactorGraph(), gtsam.Values()
    base = gtsam.Point3(*rb)
    for k in range(len(tow)):
        init.insert(X(k), gtsam.Point3(*pos0))
        if k > 0:
            dt = max(tow[k] - tow[k - 1], 0.2)
            graph.add(gtsam.BetweenFactorPoint3(
                X(k - 1), X(k), gtsam.Point3(0, 0, 0),
                gtsam.noiseModel.Isotropic.Sigma(3, cfg.sig_rw * np.sqrt(dt / 0.2))))
    graph.add(gtsam.PriorFactorPoint3(
        X(0), gtsam.Point3(*pos0), gtsam.noiseModel.Isotropic.Sigma(3, 100.0)))
    add_ambiguities(graph, init, rows, cfg.sig_amb)
    for r in rows:
        k, lam, w = int(r[EPOCH]), r[LAM], r[WEIGHT]
        sr, st, sbr, sbt = sat_points(r)
        graph.add(gtsam.DoubleDifferencePseudorangeFactor(
            X(k), r[PR_RR], r[PR_BR], r[PR_RT], r[PR_BT], sr, st, sbr, sbt, base,
            huber_noise(cfg.sig_pr * w, cfg.huber)))
        graph.add(gtsam.DoubleDifferenceCarrierPhaseFactor(
            X(k), N(int(r[ARC_REF])), N(int(r[ARC_TGT])),
            r[CP_RR], r[CP_BR], r[CP_RT], r[CP_BT], sr, st, sbr, sbt, base, lam,
            huber_noise(cfg.sig_cp * w, cfg.huber)))
    return graph, init


def add_fix(graph, accepted, sig_fix):
    """Copy of graph + one BetweenFactorDouble N(t) - N(r) = n per relation."""
    fgraph = gtsam.NonlinearFactorGraph(graph)
    noise = gtsam.noiseModel.Isotropic.Sigma(1, sig_fix)
    for r, t, n in accepted:
        fgraph.add(gtsam.BetweenFactorDouble(N(r), N(t), float(n), noise))
    return fgraph


def gnss_positions(values, n_ep):
    return np.array([values.atPoint3(X(k)) for k in range(n_ep)])


# ── Tightly-coupled batch graph ───────────────────────────────────────

TC_DIMS = {"x": 6, "v": 3, "b": 6, "n": 1, "m": 3}
MOUNT = gtsam.symbol("m", 0)


def imu_params(cfg):
    return make_imu_params(
        accel_noise=cfg.acc_noise,
        gyro_noise=cfg.gyro_noise,
        accel_bias_sigma=cfg.acc_bias_rw,
        gyro_bias_sigma=cfg.gyro_bias_rw,
        integ_cov=cfg.integ_cov,
        gravity=cfg.gravity,
    )


def initial_trajectory(ant_ecef, tow, R_enu2ecef, rb, lever):
    """Body poses / ENU velocities / speed from GNSS antenna positions."""
    ant = (ant_ecef - rb) @ R_enu2ecef
    vel = np.gradient(ant, tow, axis=0)
    kern = np.ones(5) / 5
    vel = np.column_stack([np.convolve(vel[:, i], kern, mode="same") for i in range(3)])
    speed = np.hypot(vel[:, 0], vel[:, 1])
    moving = speed > 1.0
    if not np.any(moving):
        raise ValueError(
            "Cannot initialize heading: no smoothed horizontal GNSS speed "
            "exceeds 1 m/s. Include a segment with sufficient motion."
        )
    yaw = np.full(len(tow), np.nan)
    yaw[moving] = np.arctan2(vel[moving, 1], vel[moving, 0])
    idx = np.where(moving, np.arange(len(tow)), -1)
    np.maximum.accumulate(idx, out=idx)
    idx[idx < 0] = np.argmax(moving)
    poses = []
    for k in range(len(tow)):
        Rk = gtsam.Rot3.Ypr(yaw[idx[k]], 0.0, 0.0)
        poses.append(gtsam.Pose3(Rk, gtsam.Point3(*(ant[k] - Rk.matrix() @ lever))))
    return poses, vel, speed


def doppler_speed(dop, n_ep, pos_ecef, fallback):
    """Per-epoch receiver speed from undifferenced Doppler least squares."""
    out = np.array(fallback, dtype=float)
    starts = np.searchsorted(dop[:, D_EPOCH].astype(int), np.arange(n_ep + 1))
    for k in range(n_ep):
        d = dop[starts[k]:starts[k + 1]]
        if len(d) < 5 or not np.all(np.isfinite(pos_ecef[k])):
            continue
        e = d[:, D_SATPOS] - pos_ecef[k]
        e /= np.linalg.norm(e, axis=1)[:, None]
        y = -d[:, D_LAM] * d[:, D_HZ] - np.sum(e * d[:, D_SATVEL], axis=1)
        A = np.column_stack([-e, np.ones(len(d))])
        use = np.ones(len(d), bool)
        for _ in range(2):
            x, *_ = np.linalg.lstsq(A[use], y[use], rcond=None)
            r = y - A @ x
            mad = 1.4826 * np.median(np.abs(r[use] - np.median(r[use]))) + 1e-3
            new = np.abs(r) < 4 * mad
            if new.sum() < 5 or np.array_equal(new, use):
                break
            use = new
        if use.sum() >= 5:
            out[k] = np.linalg.norm(x[:3])
    return out


def _trop_saast(llh, sin_el, humi=0.7):
    lat, hgt = llh[0], np.maximum(llh[2], 0.0)
    pres = 1013.25 * (1.0 - 2.2557e-5 * hgt) ** 5.2568
    temp = 15.0 - 6.5e-3 * hgt + 273.16
    e = 6.108 * humi * np.exp((17.15 * temp - 4684.0) / (temp - 38.45))
    zhd = 0.0022768 * pres / (1.0 - 0.00266 * np.cos(2.0 * lat) - 0.00028 * hgt / 1e3)
    zwd = 0.002277 * (1255.0 / temp + 0.05) * e
    return (zhd + zwd) / np.maximum(sin_el, 0.1)


def tropo_correct(rows, rover_ecef, rb):
    """Copy of the DD rows with modelled tropo removed from all eight PR/CP."""
    out = rows.copy()
    xr = rover_ecef[rows[:, EPOCH].astype(int)].copy()
    xr[~np.all(np.isfinite(xr), axis=1)] = rb
    llh_r = np.array([ecef2pos(p) for p in xr]).T
    llh_b = np.array(ecef2pos(rb))

    def delay(sat, x, llh):
        lat, lon = llh[0], llh[1]
        up = np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)
        los = sat - x
        return _trop_saast(llh, np.sum(los * up, axis=-1) / np.linalg.norm(los, axis=-1))

    t = (delay(rows[:, SAT_RR], xr, llh_r), delay(rows[:, SAT_BR], rb, llh_b),
         delay(rows[:, SAT_RT], xr, llh_r), delay(rows[:, SAT_BT], rb, llh_b))
    for cols in ((PR_RR, PR_BR, PR_RT, PR_BT), (CP_RR, CP_BR, CP_RT, CP_BT)):
        for c, d in zip(cols, t):
            out[:, c] -= d
    return out


def build_tc_graph(meas, imu, ant_ecef, cfg, use_cauchy=False):
    """Graph, initial values, ecef_T_nav and factor-index map."""
    gnss_noise = (lambda s: cauchy_noise(s, cfg.cauchy)) if use_cauchy else (
        lambda s: huber_noise(s, cfg.huber))
    dop_noise = (lambda s: cauchy_noise(s, cfg.cauchy)) if use_cauchy else (
        lambda s: huber_noise(s, cfg.huber_dop))
    rows, dop, tow, rb = meas["rows"], meas["dop"], meas["tow"], meas["rb"]
    R_enu2ecef, ecef_T_nav = enu_frame(rb)
    poses, vel, speed = initial_trajectory(ant_ecef, tow, R_enu2ecef, rb, cfg.lever)
    t_imu, acc, gyr = imu
    still = t_imu < tow[np.argmax(speed > 0.3)] - 1.0
    bg0 = gyr[still].mean(axis=0) if still.sum() > 100 else np.zeros(3)
    bias0 = gtsam.imuBias.ConstantBias(np.zeros(3), bg0)
    lever, base = gtsam.Point3(*cfg.lever), gtsam.Point3(*rb)
    params = imu_params(cfg)
    n_ep = len(tow)
    graph, init = gtsam.NonlinearFactorGraph(), gtsam.Values()
    nhc_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array(cfg.nhc_sigmas))

    cut = np.searchsorted(t_imu, tow + 1e-6, side="right")
    omega = np.zeros((n_ep, 3))
    for k in range(n_ep):
        init.insert(X(k), poses[k])
        init.insert(V(k), vel[k])
        init.insert(B(k), bias0)
        lo = cut[k - 1] if k > 0 else max(cut[0] - 20, 0)
        if cut[k] > lo:
            omega[k] = gyr[lo:cut[k]].mean(axis=0) - bg0
        if k == 0:
            continue
        pim = gtsam.PreintegratedCombinedMeasurements(params, bias0)
        tprev = tow[k - 1]
        for i in range(cut[k - 1], cut[k]):
            if t_imu[i] > tprev:
                pim.integrateMeasurement(acc[i], gyr[i], t_imu[i] - tprev)
            tprev = t_imu[i]
        if tow[k] > tprev and cut[k] > 0:
            pim.integrateMeasurement(acc[cut[k] - 1], gyr[cut[k] - 1], tow[k] - tprev)
        graph.add(gtsam.CombinedImuFactor(X(k - 1), V(k - 1), X(k), V(k), B(k - 1), B(k), pim))

    init.insert(MOUNT, gtsam.Rot3())
    graph.add(gtsam.PriorFactorRot3(MOUNT, gtsam.Rot3(), gtsam.noiseModel.Diagonal.Sigmas(
        np.array([0.01, 0.1, 0.1]))))
    for k in range(n_ep):
        graph.add(NhcFactorCalib(X(k), V(k), MOUNT, omega[k], np.zeros(3),
                                 nhc_noise))

    n_zupt = 0
    if cfg.zupt:
        gate = (doppler_speed(dop, n_ep, ant_ecef, speed)
                if cfg.zupt_speed_src == "doppler" else speed)
        zv = gtsam.noiseModel.Isotropic.Sigma(3, cfg.zupt_vel_sigma)
        zp = gtsam.noiseModel.Diagonal.Sigmas(np.array(
            [cfg.zupt_rot_sigma] * 3 + [cfg.zupt_pos_sigma] * 3))
        for k in range(1, n_ep):
            a, g = acc[cut[k - 1]:cut[k]], gyr[cut[k - 1]:cut[k]]
            if is_still_imu_window(a, g, bg0) and gate[k] < cfg.zupt_speed:
                graph.add(gtsam.PriorFactorVector(V(k), np.zeros(3), zv))
                graph.add(gtsam.BetweenFactorPose3(X(k - 1), X(k), gtsam.Pose3(), zp))
                n_zupt += 1

    graph.add(gtsam.PriorFactorPose3(X(0), poses[0], gtsam.noiseModel.Diagonal.Sigmas(
        np.array([0.05, 0.05, 0.5, 1.0, 1.0, 1.0]))))
    graph.add(gtsam.PriorFactorVector(V(0), vel[0], gtsam.noiseModel.Isotropic.Sigma(3, 0.5)))
    graph.add(gtsam.PriorFactorConstantBias(B(0), bias0, gtsam.noiseModel.Diagonal.Sigmas(
        np.array([0.05] * 3 + [0.005] * 3))))

    idx = {"pr": [], "cp": [], "dop": [], "dop_k": []}
    add_ambiguities(graph, init, rows, cfg.sig_amb)
    for r in rows:
        k, lam, w = int(r[EPOCH]), r[LAM], r[WEIGHT]
        sr, st, sbr, sbt = sat_points(r)
        idx["pr"].append(graph.size())
        graph.add(gtsam.DoubleDifferencePseudorangeFactorArm(
            X(k), r[PR_RR], r[PR_BR], r[PR_RT], r[PR_BT], sr, st, sbr, sbt, base,
            lever, ecef_T_nav, gnss_noise(cfg.sig_pr * w)))
        idx["cp"].append(graph.size())
        graph.add(gtsam.DoubleDifferenceCarrierPhaseFactorArm(
            X(k), N(int(r[ARC_REF])), N(int(r[ARC_TGT])),
            r[CP_RR], r[CP_BR], r[CP_RT], r[CP_BT], sr, st, sbr, sbt, base,
            lam, lever, ecef_T_nav, gnss_noise(cfg.sig_cp * w)))

    starts = np.searchsorted(dop[:, D_EPOCH].astype(int), np.arange(n_ep + 1))
    for k in range(n_ep):
        d = dop[starts[k]:starts[k + 1]]
        if len(d) < 2:
            continue
        rr = gtsam.Point3(*ant_ecef[k])
        ir = int(np.argmax(d[:, D_EL]))
        ref = d[ir]
        s_ref = cfg.dop_sigma0 / max(np.sin(ref[D_EL]), 0.1)
        for j, t in enumerate(d):
            if j == ir:
                continue
            s_t = cfg.dop_sigma0 / max(np.sin(t[D_EL]), 0.1)
            idx["dop"].append(graph.size())
            idx["dop_k"].append(k)
            graph.add(SingleDifferenceDopplerFactorArm(
                X(k), V(k), t[D_HZ], ref[D_HZ], t[D_LAM], ref[D_LAM],
                gtsam.Point3(*t[D_SATPOS]), gtsam.Point3(*t[D_SATVEL]),
                gtsam.Point3(*ref[D_SATPOS]), gtsam.Point3(*ref[D_SATVEL]),
                rr, lever, ecef_T_nav, gtsam.Point3(*omega[k]), 0.0, 0.0,
                dop_noise(np.hypot(s_t, s_ref))))
    idx = {key: np.array(val, dtype=np.int64) for key, val in idx.items()}
    idx["n_zupt"] = n_zupt
    return graph, init, ecef_T_nav, idx


def antenna_ecef(values, ecef_T_nav, n_ep, lever):
    lever = gtsam.Point3(*lever)
    return np.array([ecef_T_nav.transformFrom(values.atPose3(X(k)).transformFrom(lever))
                     for k in range(n_ep)])


def residuals(graph, values, idx):
    """Unwhitened residuals [m, m, m/s] of the DD PR / DD CP / SD Doppler factors."""
    def res(ids, dop=False):
        out = np.full(len(ids), np.nan)
        for j, i in enumerate(ids):
            f = graph.at(int(i))
            if f is None:
                continue
            if dop:
                k = int(idx["dop_k"][j])
                out[j] = f.evaluateError(values.atPose3(X(k)), values.atVector(V(k)))[0]
            else:
                out[j] = f.unwhitenedError(values)[0]
        return out
    return res(idx["pr"]), res(idx["cp"]), res(idx["dop"], dop=True)


# ── Whole-run LM solver ───────────────────────────────────────────────

def cuda_available():
    return hasattr(gtsam, "cuda")


def _lm_params(p, max_iters):
    p.setMaxIterations(max_iters)
    p.setRelativeErrorTol(1e-5)
    p.setAbsoluteErrorTol(1e-5)
    return p


def solve(graph, init, backend="auto", max_iters=200):
    """Optimize; returns (values, seconds, info string)."""
    if backend == "auto":
        backend = "cuda" if cuda_available() else "cpu"
    t = time.perf_counter()
    if backend == "cuda":
        cu = gtsam.cuda
        p = _lm_params(cu.SparseLevenbergMarquardtParams(), max_iters)
        lin = cu.LinearSolverOptions()
        lin.backend = cu.LinearSolverType.Cudss
        p.linear = lin
        p.fallbackOnUnsupported = False
        opt = cu.SparseLevenbergMarquardtOptimizer(graph, init, p)
        v = gtsam.Values(opt.optimize())
        r = opt.result()
        info = (f"cuda iters={r.iterations} "
                f"term={str(r.termination).split('.')[-1]}")
    else:
        p = _lm_params(gtsam.LevenbergMarquardtParams(), max_iters)
        p.setOrderingType("METIS")
        p.setDiagonalDamping(True)
        try:
            opt = gtsam.LevenbergMarquardtOptimizer(graph, init, p)
            v = gtsam.Values(opt.optimize())
        except (IndexError, RuntimeError):
            p.setOrderingType("COLAMD")
            opt = gtsam.LevenbergMarquardtOptimizer(graph, init, p)
            v = gtsam.Values(opt.optimize())
        info = f"cpu iters={opt.iterations()}"
    return v, time.perf_counter() - t, info
