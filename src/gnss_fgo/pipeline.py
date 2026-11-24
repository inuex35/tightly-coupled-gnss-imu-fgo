"""Single-phase tightly-coupled epoch pipeline: bootstrap/IMU prediction -> slip check -> solve -> AR -> report."""

from types import SimpleNamespace

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


def fls_update(tc, graph, values, key_idx, keep_keys=(),
               advance_time=True, include_prev=True):
    """Apply an FLS update with current timestamps on pose/vel/bias and live N keys."""
    if advance_time:
        tc.tc_time += tc._epoch_dt
    ts = gtsam.FixedLagSmootherKeyTimestampMap()
    t = tc.tc_time
    ts[tc.Xpose(key_idx)] = t
    ts[tc.Vel(key_idx)] = t
    ts[tc.Bias(key_idx)] = t
    if include_prev and key_idx > 0:
        ts[tc.Xpose(key_idx - 1)] = t
        ts[tc.Vel(key_idx - 1)] = t
        ts[tc.Bias(key_idx - 1)] = t
    for k in values.keys():
        if k not in ts:
            ts[k] = t
    for k in keep_keys:
        ts[k] = t
    tc.isam2.update(graph, values, ts)


def _solve_isam2(tc, epoch):
    """Update FixedLagSmoother and snapshot the new estimate."""
    info = epoch.info
    try:
        live_amb_keys = [
            k for (_sf, k) in sorted_amb_items(tc._sat_states.amb_keys_dict())
            if epoch.estimate.exists(k) or epoch.values.exists(k)
        ]
        fls_update(tc, epoch.graph, epoch.values, epoch.key_idx,
                   keep_keys=live_amb_keys)
        epoch.estimate = tc.isam2.calculateEstimate()
    except (RuntimeError, IndexError, ValueError) as ex:
        info['error'] = str(ex)
        return advance_epoch_and_pack(
            tc, np.array(tc.nav.x[0:3]), 'FLT', 0, info, epoch.obs)

    epoch.pose_tc = epoch.estimate.atPose3(tc.Xpose(epoch.key_idx))
    tc.tc_bias = epoch.estimate.atConstantBias(tc.Bias(epoch.key_idx))
    enu_tc = np.array(epoch.pose_tc.translation())
    epoch.ecef_tc = epoch.R_enu2ecef @ enu_tc + tc.base_ecef
    return None


# ── Epoch pipeline ────────────────────────────────────────────────────

def _decide_fix_or_flt(tc, epoch):
    """Pure FIX/FLT decision directly from LAMBDA acceptance (nav.smode == 4)."""
    info = epoch.info
    pose_tc_antenna = tc._antenna_ecef(epoch.pose_tc, epoch.ecef_tc)
    if tc.nav.smode == 4 and epoch.nb > 0 and epoch.xa is not None:
        lc = float(np.linalg.norm(epoch.xa[0:3] - pose_tc_antenna))
        info['lambda_correction'] = lc
        return epoch.xa[0:3], 'FIX', epoch.nb
    return pose_tc_antenna, 'FLT', epoch.nb


def _bootstrap_epoch0(tc, obs, obsb, rs, rsb, sat, el, iu, obs_sd, ir_map,
                      init_ecef, R_enu2ecef, info):
    """Initialize the single smoother at the first valid GNSS epoch."""
    _, tow_obs = time2gpst(obs.t)
    while tc.imu_idx < len(tc.imu_data) and tc.imu_data[tc.imu_idx]['tow'] <= tow_obs + 1e-6:
        tc.imu_idx += 1
    bias_acc, bias_gyro = tc._estimate_stationary_bias()
    bias0 = gtsam.imuBias.ConstantBias(bias_acc, bias_gyro)
    tc.tc_bias = bias0
    tc.tc_bias_init = bias0

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

    tc.fls_lag = tc.cfg.fls_lag
    tc.isam2 = tc._make_isam2(tc.fls_lag)
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

    epoch = SimpleNamespace(
        obs=obs, obsb=obsb, rs=rs, vs=vs, rsb=rsb,
        sat=sat, el=el, iu=iu, obs_sd=obs_sd, ir_map=ir_map,
        info=info, ns=ns, init_ecef=init_ecef, R_enu2ecef=R_enu2ecef,
        key_idx=None, tow=None, imu_idx_prev=None, pim=None, n_imu=0,
        gyro_mean=None, graph=None, values=None, estimate=None,
        pose_p=None, vel_prev=None, bias_prev=None, pred_nav=None,
        pred_ecef=None, nv=0, pose_tc=None, ecef_tc=None,
        nb=0, xa=None, sol=None, tag=None)

    # 1. IMU preintegration & state prediction
    tc.tc_epoch += 1
    epoch.key_idx = tc.tc_epoch
    info['tc_epoch'] = epoch.key_idx
    epoch.imu_idx_prev = tc.imu_idx
    _, tow_obs = time2gpst(obs.t)
    epoch.tow = tow_obs
    epoch.pim, epoch.n_imu, epoch.gyro_mean, tc.imu_idx = _utils_build_pim(
        tc.imu_params, tc.tc_bias, tc.imu_data, tc.imu_idx, target_tow=tow_obs)
    info['n_imu'] = epoch.n_imu
    if epoch.n_imu == 0:
        return advance_epoch_and_pack(
            tc, np.array(tc.nav.x[0:3]), 'FLT', 0, info, obs)

    epoch.graph = gtsam.NonlinearFactorGraph()
    epoch.values = gtsam.Values()
    epoch.estimate = tc.isam2.calculateEstimate()

    if not epoch.estimate.exists(tc.Xpose(epoch.key_idx - 1)):
        info['prev_pose_missing'] = epoch.key_idx - 1
        return advance_epoch_and_pack(
            tc, np.array(tc.nav.x[0:3]), 'FLT', 0, info, obs)

    epoch.pose_p = epoch.estimate.atPose3(tc.Xpose(epoch.key_idx - 1))
    epoch.vel_prev = epoch.estimate.atVector(tc.Vel(epoch.key_idx - 1))
    epoch.bias_prev = epoch.estimate.atConstantBias(tc.Bias(epoch.key_idx - 1))
    epoch.pred_nav = epoch.pim.predict(
        gtsam.NavState(epoch.pose_p, epoch.vel_prev), epoch.bias_prev)
    epoch.values.insert(tc.Xpose(epoch.key_idx), epoch.pred_nav.pose())
    epoch.values.insert(tc.Vel(epoch.key_idx), epoch.pred_nav.velocity())
    epoch.values.insert(tc.Bias(epoch.key_idx), epoch.bias_prev)
    epoch.graph.add(gtsam.CombinedImuFactor(
        tc.Xpose(epoch.key_idx - 1), tc.Vel(epoch.key_idx - 1),
        tc.Xpose(epoch.key_idx), tc.Vel(epoch.key_idx),
        tc.Bias(epoch.key_idx - 1), tc.Bias(epoch.key_idx), epoch.pim))

    # 2. Cycle-slip detection & predicted antenna ECEF
    info['nsat'] = ns
    n_reset, _ = detect_slips_and_reset_ambiguities(tc, obs, obs_sd, sat, iu)
    info['n_slip'] = n_reset
    pred_enu = np.array(epoch.pred_nav.pose().translation())
    epoch.pred_ecef = tc._antenna_ecef(
        epoch.pred_nav.pose(), R_enu2ecef @ pred_enu + tc.base_ecef)

    # 3. Measurement factors (DDPR, DDCP, SD Doppler, NHC, ZUPT)
    epoch.nv = _tc_factors.build_dd_factors(
        tc, epoch.graph, epoch.values, obs, obsb, obs_sd,
        rs, rsb, sat, el, iu, ir_map,
        tc.Xpose(epoch.key_idx), tc.lever_arm_tc, tc._sat_states,
        existing_estimate=epoch.estimate)
    info['n_dd'] = epoch.nv
    _tc_factors.add_sd_doppler_factors(tc, epoch)
    if _tc_factors.add_nhc_factor(tc, epoch.graph, epoch.key_idx,
                                  gyro_mean_rh=epoch.gyro_mean):
        info['nhc'] = True
    _tc_factors.add_zupt_factors(
        tc, epoch.graph, epoch.key_idx,
        int(epoch.imu_idx_prev), int(epoch.n_imu), info)

    # 4. Fixed-lag smoother update
    early = _solve_isam2(tc, epoch)
    if early is not None:
        return early

    # 5. Continuous LAMBDA ambiguity resolution & solution output
    tc.nav.smode = 5
    tc.nav.x[0:3] = tc._antenna_ecef(epoch.pose_tc, epoch.ecef_tc)
    epoch.nb, epoch.xa = _tc_ar.run_ar(
        tc, sat, el, tc._sat_states.amb_keys_dict(),
        key_pose=tc.Xpose(epoch.key_idx))

    epoch.sol, epoch.tag, epoch.nb = _decide_fix_or_flt(tc, epoch)
    info['bias_acc'] = tc.tc_bias.accelerometer()
    info['bias_gyro'] = tc.tc_bias.gyroscope()
    return advance_epoch_and_pack(
        tc, epoch.sol, epoch.tag, epoch.nb, info, obs)
