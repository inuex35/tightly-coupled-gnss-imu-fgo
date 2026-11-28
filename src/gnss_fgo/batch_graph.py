"""Batch graph construction and whole-run LM solver."""

import time

import gtsam
import numpy as np
from gtsam.symbol_shorthand import B, N, V, X

from ._extra_factors import (
    NhcFactorCalib,
    SingleDifferenceDopplerFactorArm,
)
from .batch_preparation import (
    ARC_REF,
    ARC_TGT,
    CP_BR,
    CP_BT,
    CP_RR,
    CP_RT,
    D_EPOCH,
    D_HZ,
    D_LAM,
    D_SATPOS,
    D_SATVEL,
    D_SNR,
    EPOCH,
    LAM,
    PR_BR,
    PR_BT,
    PR_RR,
    PR_RT,
    SAT_BR,
    SAT_BT,
    SAT_RR,
    SAT_RT,
    WEIGHT,
)
from .factors import sd_doppler_sat_sigma
from .utils import (
    cauchy_noise,
    enu_frame,
    huber_noise,
    integrate_imu_segment,
    make_imu_params,
)


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


def sat_points(r):
    """(ref@rover, tgt@rover, ref@base, tgt@base) satellite positions."""
    return (gtsam.Point3(*r[SAT_RR]), gtsam.Point3(*r[SAT_RT]),
            gtsam.Point3(*r[SAT_BR]), gtsam.Point3(*r[SAT_BT]))


# ── GNSS-only batch graph ─────────────────────────────────────────────


def add_ambiguities(graph, init, rows, sig_amb):
    """N(a) per arc, seeded with SD carrier-minus-code, plus a weak prior."""
    seen = set()
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
        pim, _, _ = integrate_imu_segment(
            params, bias0, t_imu, acc, gyr, cut[k - 1], cut[k],
            t_prev=tow[k - 1], t_end=tow[k])
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
        has_snr = d.shape[1] > D_SNR
        sigmas = [sd_doppler_sat_sigma(cfg.dop_sigma0, t[D_SNR] if has_snr else 0.0)
                  for t in d]
        ir = int(np.argmin(sigmas))
        ref, s_ref = d[ir], sigmas[ir]
        for j, (t, s_t) in enumerate(zip(d, sigmas)):
            if j == ir:
                continue
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
