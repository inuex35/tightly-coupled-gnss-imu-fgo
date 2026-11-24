"""Single-phase tightly-coupled epoch pipeline: bootstrap/IMU prediction -> slip check -> solve -> AR -> report."""

import gtsam
import numpy as np
from cssrlib.gnss import time2gpst

from . import ar as _tc_ar
from . import factors as _tc_factors
from .utils import (
    build_pim as _utils_build_pim,
    euler_to_R_body2enu,
    get_wavelengths as _get_wavelengths,
    sorted_amb_items,
)


def advance_epoch_and_pack(tc, sol, tag, nb, info, obs):
    """Advance tc.epoch / nav.t and pack the (sol, tag, nb, info) tuple."""
    tc.epoch += 1
    tc.nav.t = obs.t
    return sol, tag, nb, info


# ── Cycle-slip detection & outage arc expiry ──────────────────────────

def reset_slipped_ambiguities(tc, reset_keys):
    """End the arc for every slipped (sat, f) by clearing amb_key and bumping amb_gen."""
    n_reset = 0
    for key in reset_keys:
        sat_st = tc._sat_states.get(*key)
        if sat_st.amb_key is not None:
            sat_st.amb_key = None
            n_reset += 1
        sat_st.amb_gen += 1
    return n_reset


def _detslp_gf(tc, obs, row, s, lams, nf, reset_keys):
    """Geometry-free L1-L2 jump detector."""
    L1 = obs.L[row, 0]
    L2 = obs.L[row, 1]
    if L1 == 0 or L2 == 0 or lams[0] <= 0 or lams[1] <= 0:
        return
    gf = L1 * lams[0] - L2 * lams[1]
    prev_gf = tc._sat_states.gf
    if s in prev_gf and prev_gf[s] != 0:
        if abs(gf - prev_gf[s]) > tc.cfg.thres_slip:
            for f in range(nf):
                reset_keys.add((s, f))
    prev_gf[s] = gf


def detect_slips_and_reset_ambiguities(tc, obs, obs_sd, sat, iu):
    """LLI + GF slip detection plus outage expiry. Returns (n_reset, reset_keys)."""
    nf = tc.nav.nf
    ns = len(sat)
    reset_keys = set()
    current_sats = set()

    for i in range(ns):
        s = sat[i]
        lams = _get_wavelengths(obs_sd, s, glo_ch=tc.nav.glo_ch)
        for f in range(nf):
            current_sats.add((s, f))
            sat_state = tc._sat_states.get(s, f)
            sat_state.outc = 0
            if hasattr(obs, 'lli') and obs.lli[iu[i], f] == 1:
                reset_keys.add((s, f))
        if nf >= 2 and len(lams) >= 2:
            _detslp_gf(tc, obs, iu[i], s, lams, nf, reset_keys)

    maxout = tc._sat_states.maxout
    for key in list(tc._sat_states.keys()):
        if key not in current_sats:
            st = tc._sat_states.get(*key)
            st.outc += 1
            if st.outc > maxout:
                reset_keys.add(key)

    n_reset = reset_slipped_ambiguities(tc, reset_keys)
    return n_reset, reset_keys


# ── FixedLagSmoother update ───────────────────────────────────────────

_ISAM2_RELINEARIZE_SKIP = 1
_ISAM2_RELINEARIZE_THRESHOLD = 0.01


def make_isam2(lag, relinearize_skip=_ISAM2_RELINEARIZE_SKIP,
               relinearize_threshold=_ISAM2_RELINEARIZE_THRESHOLD):
    """Build IncrementalFixedLagSmoother with QR factorization and per-epoch relinearization."""
    params = gtsam.ISAM2Params()
    params.setFactorization("QR")
    params.setRelinearizeThreshold(float(relinearize_threshold))
    params.relinearizeSkip = int(relinearize_skip)
    return gtsam.IncrementalFixedLagSmoother(lag, params)


def fls_update(tc, graph, values, key_idx, keep_keys=()):
    """Apply an FLS update with current timestamps on pose/vel/bias and live N keys."""
    tc.tc_time += tc._epoch_dt
    ts = gtsam.FixedLagSmootherKeyTimestampMap()
    t = tc.tc_time
    ts[tc.Xpose(key_idx)] = t
    ts[tc.Vel(key_idx)] = t
    ts[tc.Bias(key_idx)] = t
    if key_idx > 0:
        ts[tc.Xpose(key_idx - 1)] = t
        ts[tc.Vel(key_idx - 1)] = t
        ts[tc.Bias(key_idx - 1)] = t
    for k in values.keys():
        if k not in ts:
            ts[k] = t
    for k in keep_keys:
        ts[k] = t
    tc.isam2.update(graph, values, ts)


def _solve_isam2(tc, graph, values, estimate, key_idx, R_enu2ecef, obs, info):
    """Update FixedLagSmoother and return (pose_tc, ecef_tc, early_result)."""
    try:
        live_amb_keys = [
            k for (_sf, k) in sorted_amb_items(tc._sat_states.amb_keys_dict())
            if estimate.exists(k) or values.exists(k)
        ]
        fls_update(tc, graph, values, key_idx, keep_keys=live_amb_keys)
        new_est = tc.isam2.calculateEstimate()
    except (RuntimeError, IndexError, ValueError) as ex:
        info['error'] = str(ex)
        return None, None, advance_epoch_and_pack(
            tc, np.array(tc.nav.x[0:3]), 'FLT', 0, info, obs)

    pose_tc = new_est.atPose3(tc.Xpose(key_idx))
    tc.tc_bias = new_est.atConstantBias(tc.Bias(key_idx))
    enu_tc = np.array(pose_tc.translation())
    ecef_tc = R_enu2ecef @ enu_tc + tc.base_ecef
    return pose_tc, ecef_tc, None


# ── Epoch pipeline ────────────────────────────────────────────────────

def _decide_fix_or_flt(tc, pose_tc, ecef_tc, nb, xa, info):
    """Pure FIX/FLT decision directly from LAMBDA acceptance (nav.smode == 4)."""
    pose_tc_antenna = tc._antenna_ecef(pose_tc, ecef_tc)
    if tc.nav.smode == 4 and nb > 0 and xa is not None:
        info['lambda_correction'] = float(np.linalg.norm(xa[0:3] - pose_tc_antenna))
        return xa[0:3], 'FIX', nb
    return pose_tc_antenna, 'FLT', nb


def _bootstrap_epoch0(tc, obs, obsb, rs, rsb, sat, el, iu, obs_sd, ir_map,
                      init_ecef, R_enu2ecef, info):
    """Initialize the single smoother at the first valid GNSS epoch."""
    _, tow_obs = time2gpst(obs.t)
    while tc.imu_idx < len(tc.imu_data) and tc.imu_data[tc.imu_idx]['tow'] <= tow_obs + 1e-6:
        tc.imu_idx += 1
    bias_acc, bias_gyro = tc._estimate_stationary_bias()
    bias0 = gtsam.imuBias.ConstantBias(bias_acc, bias_gyro)
    tc.tc_bias = bias0

    n_static = max(1, min(500, tc.imu_idx))
    acc_avg = np.mean([im['acc'] for im in tc.imu_data[:n_static]], axis=0)
    acc_tilt = np.asarray(acc_avg - bias_acc, dtype=float)
    pitch_rad = float(np.arctan2(
        acc_tilt[0], np.sqrt(acc_tilt[1] ** 2 + acc_tilt[2] ** 2)))
    roll_rad = float(np.arctan2(acc_tilt[1], acc_tilt[2]))
    rot0 = gtsam.Rot3(euler_to_R_body2enu(roll_rad, pitch_rad, 0.0))

    tc.imu_params = tc._make_imu_params()
    tc.lever_arm_tc = gtsam.Point3(*tc.lever_arm)
    enu0 = R_enu2ecef.T @ (init_ecef - tc.base_ecef)
    body_enu0 = enu0 - rot0.matrix() @ tc.lever_arm
    pose0 = gtsam.Pose3(rot0, gtsam.Point3(*body_enu0))

    tc.isam2 = tc._make_isam2(tc.cfg.fls_lag)
    tc.tc_time = 0.0
    tc.tc_epoch = 0
    info['tc_epoch'] = 0

    g = gtsam.NonlinearFactorGraph()
    v = gtsam.Values()
    v.insert(tc.Xpose(0), pose0)
    v.insert(tc.Vel(0), np.zeros(3))
    v.insert(tc.Bias(0), bias0)
    rot_sigma = np.deg2rad(2.0)
    g.addPriorPose3(tc.Xpose(0), pose0,
                    gtsam.noiseModel.Diagonal.Sigmas(
                        np.array([rot_sigma, rot_sigma, np.pi, 1e4, 1e4, 1e4])))
    g.addPriorVector(tc.Vel(0), np.zeros(3),
                     gtsam.noiseModel.Isotropic.Sigma(3, 1.0))
    g.addPriorConstantBias(tc.Bias(0), bias0,
                           gtsam.noiseModel.Isotropic.Sigma(6, 0.01))

    _tc_factors.build_dd_factors(
        tc, g, v, obs, obsb, obs_sd, rs, rsb, sat, el, iu, ir_map,
        tc.Xpose(0), tc.lever_arm_tc, tc._sat_states, existing_estimate=None)

    ts0 = gtsam.FixedLagSmootherKeyTimestampMap()
    tc.tc_time += tc._epoch_dt
    for k in v.keys():
        ts0[k] = tc.tc_time
    tc.isam2.update(g, v, ts0)
    for _ in range(3):
        tc.isam2.update(gtsam.NonlinearFactorGraph(), gtsam.Values(),
                        gtsam.FixedLagSmootherKeyTimestampMap())
    est = tc.isam2.calculateEstimate()
    pose_tc = est.atPose3(tc.Xpose(0))
    ecef_tc = R_enu2ecef @ np.array(pose_tc.translation()) + tc.base_ecef
    sol = tc._antenna_ecef(pose_tc, ecef_tc)
    tc.nav.x[0:3] = sol
    tc.nav.smode = 5
    return advance_epoch_and_pack(tc, sol, 'FLT', 0, info, obs)


def run_tc_epoch(tc, obs, obsb, rs, vs, rsb, sat, el, iu,
                 obs_sd, ir_map, info, ns, init_ecef, R_enu2ecef):
    """Single-phase IMU/GNSS TC epoch flow."""
    if tc.isam2 is None:
        if ns < 4:
            return advance_epoch_and_pack(
                tc, np.array(tc.nav.x[0:3]), 'FLT', 0, info, obs)
        return _bootstrap_epoch0(
            tc, obs, obsb, rs, rsb, sat, el, iu, obs_sd, ir_map,
            init_ecef, R_enu2ecef, info)

    # 1. IMU preintegration & state prediction
    tc.tc_epoch += 1
    key_idx = tc.tc_epoch
    info['tc_epoch'] = key_idx
    _, tow_obs = time2gpst(obs.t)
    pim, n_imu, gyro_mean, tc.imu_idx = _utils_build_pim(
        tc.imu_params, tc.tc_bias, tc.imu_data, tc.imu_idx, target_tow=tow_obs)
    info['n_imu'] = n_imu
    if n_imu == 0:
        return advance_epoch_and_pack(
            tc, np.array(tc.nav.x[0:3]), 'FLT', 0, info, obs)

    graph = gtsam.NonlinearFactorGraph()
    values = gtsam.Values()
    estimate = tc.isam2.calculateEstimate()

    if not estimate.exists(tc.Xpose(key_idx - 1)):
        info['prev_pose_missing'] = key_idx - 1
        return advance_epoch_and_pack(
            tc, np.array(tc.nav.x[0:3]), 'FLT', 0, info, obs)

    pose_p = estimate.atPose3(tc.Xpose(key_idx - 1))
    vel_prev = estimate.atVector(tc.Vel(key_idx - 1))
    bias_prev = estimate.atConstantBias(tc.Bias(key_idx - 1))
    pred_nav = pim.predict(gtsam.NavState(pose_p, vel_prev), bias_prev)
    values.insert(tc.Xpose(key_idx), pred_nav.pose())
    values.insert(tc.Vel(key_idx), pred_nav.velocity())
    values.insert(tc.Bias(key_idx), bias_prev)
    graph.add(gtsam.CombinedImuFactor(
        tc.Xpose(key_idx - 1), tc.Vel(key_idx - 1),
        tc.Xpose(key_idx), tc.Vel(key_idx),
        tc.Bias(key_idx - 1), tc.Bias(key_idx), pim))

    # 2. Cycle-slip detection & predicted antenna ECEF
    info['nsat'] = ns
    n_reset, _ = detect_slips_and_reset_ambiguities(tc, obs, obs_sd, sat, iu)
    info['n_slip'] = n_reset
    pred_enu = np.array(pred_nav.pose().translation())
    pred_ecef = tc._antenna_ecef(
        pred_nav.pose(), R_enu2ecef @ pred_enu + tc.base_ecef)

    # 3. Measurement factors (DDPR, DDCP, SD Doppler, NHC)
    nv, visible_sf, seeded_sf = _tc_factors.build_dd_factors(
        tc, graph, values, obs, obsb, obs_sd,
        rs, rsb, sat, el, iu, ir_map,
        tc.Xpose(key_idx), tc.lever_arm_tc, tc._sat_states,
        existing_estimate=estimate)
    info['n_dd'] = nv
    _tc_factors.add_sd_doppler_factors(
        tc, graph, key_idx, obs, obs_sd, rs, vs, sat, el, iu,
        pred_ecef, gyro_mean, info)
    if _tc_factors.add_nhc_factor(tc, graph, key_idx, gyro_mean_rh=gyro_mean):
        info['nhc'] = True

    # 4. Fixed-lag smoother update
    pose_tc, ecef_tc, early = _solve_isam2(
        tc, graph, values, estimate, key_idx, R_enu2ecef, obs, info)
    if early is not None:
        return early

    # 5. Continuous LAMBDA ambiguity resolution & solution output
    tc.nav.smode = 5
    tc.nav.x[0:3] = tc._antenna_ecef(pose_tc, ecef_tc)
    nb, xa = _tc_ar.run_ar(
        tc, sat, el, tc._sat_states.amb_keys_dict(),
        key_pose=tc.Xpose(key_idx),
        visible_sf=visible_sf, seeded_sf=seeded_sf)

    sol, tag, nb = _decide_fix_or_flt(tc, pose_tc, ecef_tc, nb, xa, info)
    info['bias_acc'] = tc.tc_bias.accelerometer()
    info['bias_gyro'] = tc.tc_bias.gyroscope()
    return advance_epoch_and_pack(tc, sol, tag, nb, info, obs)
