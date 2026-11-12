"""Tightly-coupled GNSS RTK + IMU graph over a whole run.

Conventions of the sequential pipeline: local ENU navigation frame at the
base station, FLU body (raw IMU axes), lever arm in body FLU, gravity along
-Up. Per keyframe (every rover epoch) k:
  X(k) Pose3, V(k) velocity, B(k) IMU bias
  CombinedImuFactor between k-1 and k (IMU samples in (t[k-1], t[k]])
  DD pseudorange / carrier Arm factors (Huber) on X(k) and SD arcs N(a)
  single-differenced Doppler Arm factors (Huber) on X(k), V(k)
  NhcFactor (or NhcFactorCalib with a shared mounting angle M(0))
  ZUPT on stationary epochs: zero velocity + no-motion pose between factor
"""
import time

import numpy as np
import gtsam
from gtsam.symbol_shorthand import B, N, V, X

from . import rows as R
from .gnss_graph import add_ambiguities, huber_noise

DIMS = {"x": 6, "v": 3, "b": 6, "n": 1, "m": 3}
MOUNT = gtsam.symbol("m", 0)


def ecef2llh(p):
    a, e2 = 6378137.0, 6.69437999014e-3
    lon = np.arctan2(p[1], p[0])
    r = np.hypot(p[0], p[1])
    lat = np.arctan2(p[2], r * (1 - e2))
    for _ in range(5):
        nr = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
        h = r / np.cos(lat) - nr
        lat = np.arctan2(p[2], r * (1 - e2 * nr / (nr + h)))
    return lat, lon, h


def enu_frame(rb):
    """(R_enu2ecef, ecef_T_nav) for the ENU frame at the base."""
    lat, lon, _ = ecef2llh(rb)
    sl, cl, sn, cn = np.sin(lat), np.cos(lat), np.sin(lon), np.cos(lon)
    Rm = np.array([[-sn, -sl * cn, cl * cn], [cn, -sl * sn, cl * sn], [0, cl, sl]])
    return Rm, gtsam.Pose3(gtsam.Rot3(Rm), gtsam.Point3(*rb))


def imu_params(cfg):
    p = gtsam.PreintegrationCombinedParams.MakeSharedU(cfg.gravity)
    p.setAccelerometerCovariance(np.diag(cfg.acc_noise ** 2))
    p.setGyroscopeCovariance(np.diag(cfg.gyro_noise ** 2))
    p.setIntegrationCovariance(np.eye(3) * cfg.integ_cov)
    p.setBiasAccCovariance(np.eye(3) * cfg.acc_bias_rw ** 2)
    p.setBiasOmegaCovariance(np.eye(3) * cfg.gyro_bias_rw ** 2)
    return p


def initial_trajectory(ant_ecef, tow, R_enu2ecef, rb, lever):
    """Body poses / ENU velocities / speed from GNSS antenna positions
    (yaw from the course over ground, roll = pitch = 0)."""
    ant = (ant_ecef - rb) @ R_enu2ecef
    vel = np.gradient(ant, tow, axis=0)
    kern = np.ones(5) / 5
    vel = np.column_stack([np.convolve(vel[:, i], kern, mode="same") for i in range(3)])
    speed = np.hypot(vel[:, 0], vel[:, 1])
    moving = speed > 1.0
    yaw = np.full(len(tow), np.nan)
    yaw[moving] = np.arctan2(vel[moving, 1], vel[moving, 0])
    idx = np.where(moving, np.arange(len(tow)), -1)
    np.maximum.accumulate(idx, out=idx)          # forward fill
    idx[idx < 0] = np.argmax(moving)             # back fill before the first motion
    poses = []
    for k in range(len(tow)):
        Rk = gtsam.Rot3.Ypr(yaw[idx[k]], 0.0, 0.0)
        poses.append(gtsam.Pose3(Rk, gtsam.Point3(*(ant[k] - Rk.matrix() @ lever))))
    return poses, vel, speed


def doppler_speed(dop, n_ep, pos_ecef, fallback):
    """Per-epoch receiver speed from undifferenced Doppler least squares
    (velocity + clock drift, MAD rejection); fallback where < 5 satellites."""
    out = np.array(fallback, dtype=float)
    starts = np.searchsorted(dop[:, R.D_EPOCH].astype(int), np.arange(n_ep + 1))
    for k in range(n_ep):
        d = dop[starts[k]:starts[k + 1]]
        if len(d) < 5 or not np.all(np.isfinite(pos_ecef[k])):
            continue
        e = d[:, R.D_SATPOS] - pos_ecef[k]
        e /= np.linalg.norm(e, axis=1)[:, None]
        y = -d[:, R.D_LAM] * d[:, R.D_HZ] - np.sum(e * d[:, R.D_SATVEL], axis=1)
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
    """RTKLIB tropmodel (Saastamoinen, standard atmosphere), 1/sin(el) mapping."""
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
    xr = rover_ecef[rows[:, R.EPOCH].astype(int)].copy()
    xr[~np.all(np.isfinite(xr), axis=1)] = rb
    llh_r, llh_b = np.array(ecef2llh(xr.T)), np.array(ecef2llh(rb))

    def delay(sat, x, llh):
        lat, lon = llh[0], llh[1]
        up = np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)
        los = sat - x
        return _trop_saast(llh, np.sum(los * up, axis=-1) / np.linalg.norm(los, axis=-1))

    t = (delay(rows[:, R.SAT_RR], xr, llh_r), delay(rows[:, R.SAT_BR], rb, llh_b),
         delay(rows[:, R.SAT_RT], xr, llh_r), delay(rows[:, R.SAT_BT], rb, llh_b))
    for cols in ((R.PR_RR, R.PR_BR, R.PR_RT, R.PR_BT), (R.CP_RR, R.CP_BR, R.CP_RT, R.CP_BT)):
        for c, d in zip(cols, t):
            out[:, c] -= d
    return out


def build(meas, imu, ant_ecef, cfg):
    """Graph, initial values, ecef_T_nav and factor-index map (for FDE)."""
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

    cut = np.searchsorted(t_imu, tow + 1e-6, side="right")   # samples in (t[k-1], t[k]]
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
        if tow[k] > tprev and cut[k] > 0:          # hold the last sample to t[k]
            pim.integrateMeasurement(acc[cut[k] - 1], gyr[cut[k] - 1], tow[k] - tprev)
        graph.add(gtsam.CombinedImuFactor(X(k - 1), V(k - 1), X(k), V(k), B(k - 1), B(k), pim))

    if cfg.nhc_calib:
        # Shared mounting angle; its effect needs a forward speed, for which
        # the GNSS speed stands in (forward sigma stays loose). Roll about the
        # forward axis is unobservable from NHC -> tight prior.
        init.insert(MOUNT, np.zeros(3))
        graph.add(gtsam.PriorFactorVector(MOUNT, np.zeros(3), gtsam.noiseModel.Diagonal.Sigmas(
            np.array([0.01, 0.1, 0.1]))))
        for k in range(n_ep):
            graph.add(gtsam.NhcFactorCalib(X(k), V(k), MOUNT, omega[k], np.zeros(3),
                                           float(speed[k]), nhc_noise))
    else:
        for k in range(n_ep):
            graph.add(gtsam.NhcFactor(X(k), V(k), omega[k], np.zeros(3), 0.0, nhc_noise))

    # ZUPT: IMU stillness test (thresholds of the sequential pipeline) AND a
    # speed gate -- the IMU test alone also fires at constant speed.
    n_zupt = 0
    if cfg.zupt:
        gate = (doppler_speed(dop, n_ep, ant_ecef, speed)
                if cfg.zupt_speed_src == "doppler" else speed)
        zv = gtsam.noiseModel.Isotropic.Sigma(3, cfg.zupt_vel_sigma)
        zp = gtsam.noiseModel.Diagonal.Sigmas(np.array(
            [cfg.zupt_rot_sigma] * 3 + [cfg.zupt_pos_sigma] * 3))
        for k in range(1, n_ep):
            a, g = acc[cut[k - 1]:cut[k]], gyr[cut[k - 1]:cut[k]]
            if len(a) < 5:
                continue
            acc_std = np.sqrt(np.mean(np.sum((a - a.mean(axis=0)) ** 2, axis=1)))
            gyr_std = np.sqrt(np.mean(np.sum((g - g.mean(axis=0)) ** 2, axis=1)))
            gyr_med = np.median(np.linalg.norm(g - bg0, axis=1))
            if (acc_std <= 0.55 and gyr_std <= 0.030 and gyr_med <= 0.020
                    and gate[k] < cfg.zupt_speed):
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
        k, lam, w = int(r[R.EPOCH]), r[R.LAM], r[R.WEIGHT]
        sr, st, sbr, sbt = R.sat_points(r)
        idx["pr"].append(graph.size())
        graph.add(gtsam.DoubleDifferencePseudorangeFactorArm(
            X(k), r[R.PR_RR], r[R.PR_BR], r[R.PR_RT], r[R.PR_BT], sr, st, sbr, sbt, base,
            lever, ecef_T_nav, huber_noise(cfg.sig_pr * w, cfg.huber)))
        idx["cp"].append(graph.size())
        graph.add(gtsam.DoubleDifferenceCarrierPhaseFactorArm(
            X(k), N(int(r[R.ARC_REF])), N(int(r[R.ARC_TGT])),
            r[R.CP_RR], r[R.CP_BR], r[R.CP_RT], r[R.CP_BT], sr, st, sbr, sbt, base,
            lam, lever, ecef_T_nav, huber_noise(cfg.sig_cp * w, cfg.huber)))

    starts = np.searchsorted(dop[:, R.D_EPOCH].astype(int), np.arange(n_ep + 1))
    for k in range(n_ep):
        d = dop[starts[k]:starts[k + 1]]
        if len(d) < 2:
            continue
        rr = gtsam.Point3(*ant_ecef[k])
        ir = int(np.argmax(d[:, R.D_EL]))       # reference = highest elevation
        ref = d[ir]
        s_ref = cfg.dop_sigma0 / max(np.sin(ref[R.D_EL]), 0.1)
        for j, t in enumerate(d):
            if j == ir:
                continue
            s_t = cfg.dop_sigma0 / max(np.sin(t[R.D_EL]), 0.1)
            idx["dop"].append(graph.size())
            idx["dop_k"].append(k)
            graph.add(gtsam.SingleDifferenceDopplerFactorArm(
                X(k), V(k), t[R.D_HZ], ref[R.D_HZ], t[R.D_LAM], ref[R.D_LAM],
                gtsam.Point3(*t[R.D_SATPOS]), gtsam.Point3(*t[R.D_SATVEL]),
                gtsam.Point3(*ref[R.D_SATPOS]), gtsam.Point3(*ref[R.D_SATVEL]),
                rr, lever, ecef_T_nav, gtsam.Point3(*omega[k]), 0.0, 0.0,
                huber_noise(np.hypot(s_t, s_ref), cfg.huber_dop)))
    idx = {key: np.array(val, dtype=np.int64) for key, val in idx.items()}
    idx["n_zupt"] = n_zupt
    return graph, init, ecef_T_nav, idx


def antenna_ecef(values, ecef_T_nav, n_ep, lever):
    lever = gtsam.Point3(*lever)
    return np.array([ecef_T_nav.transformFrom(values.atPose3(X(k)).transformFrom(lever))
                     for k in range(n_ep)])


def residuals(graph, values, idx):
    """Unwhitened residuals [m, m, m/s] of the DD PR / DD CP / SD Doppler
    factors (NaN where a factor was removed)."""
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


def _flag(r, thresh, epoch, alive, max_frac):
    over = np.where(alive & (np.abs(r) > thresh))[0]
    out = []
    for k in np.unique(epoch[over]):
        cand = over[epoch[over] == k]
        cap = int(max_frac * np.sum(alive & (epoch == k)))
        out.extend(cand[np.argsort(-np.abs(r[cand]))][:cap])
    return np.array(out, dtype=np.int64)


def fde(graph, values, idx, rows, cfg, solve, log=print):
    """Iterative post-fit exclusion: remove (null) DD/Doppler factors over the
    thresholds in place and re-solve. Returns (values, carrier rows kept)."""
    ep = {"pr": rows[:, R.EPOCH].astype(int), "cp": rows[:, R.EPOCH].astype(int),
          "dop": idx["dop_k"]}
    alive = {key: np.ones(len(idx[key]), bool) for key in ("pr", "cp", "dop")}
    thresh = {"pr": cfg.fde_pr, "cp": cfg.fde_cp, "dop": cfg.fde_dop}
    for it in range(cfg.fde_iters):
        t0 = time.perf_counter()
        r = dict(zip(("pr", "cp", "dop"), residuals(graph, values, idx)))
        n_new = {}
        for key in ("pr", "cp", "dop"):
            bad = _flag(r[key], thresh[key], ep[key], alive[key], cfg.fde_max_frac)
            for j in bad:
                graph.remove(int(idx[key][j]))
            alive[key][bad] = False
            n_new[key] = len(bad)
        if sum(n_new.values()) == 0:
            log(f"FDE {it}: nothing over threshold")
            break
        values, dt, info = solve(graph, values)
        log(f"FDE {it}: removed PR {n_new['pr']} CP {n_new['cp']} Dop {n_new['dop']} "
            f"(residuals {time.perf_counter() - t0 - dt:.1f}s, re-solve {dt:.2f}s {info})")
    return values, alive["cp"]
