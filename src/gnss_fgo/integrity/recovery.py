"""Minimal recovery and IMU-only propagation paths (no CP-hold or sanity heuristics)."""

import numpy as np
import gtsam

from cssrlib.gnss import time2gpst

from ..factors.epoch_context import make_epoch_diagnostics
from ..factors.nhc import add_nhc_factor as _add_nhc_factor
from ..factors.zupt import add_zupt_factors as _add_zupt_factor_inplace
from ..factors import imu_preintegration as _tc_pim
from ..pipeline import update_smoother as _tc_isam


def advance_epoch_and_pack(tc, sol, tag, nb, info, obs):
    """Advance tc.epoch / nav.t and pack the (sol, tag, nb, info) tuple."""
    tc.epoch += 1
    tc.nav.t = obs.t
    return sol, tag, nb, info


def warm_reset_phase2(tc, ecef_seed, rot_seed, vel_seed=None):
    """Rebuild Phase-2 smoother at a new ECEF position after a numerical failure."""
    R = tc.R_enu2ecef
    enu_seed = R.T @ (ecef_seed - tc.base_ecef)
    pose_seed = gtsam.Pose3(rot_seed, gtsam.Point3(*enu_seed))
    vel_seed = (np.zeros(3) if vel_seed is None
                else np.asarray(vel_seed, dtype=float).reshape(3))

    tc.isam2 = tc._make_isam2(tc.fls_lag)
    tc.tc_time = 0.0
    tc.tc_epoch = 0
    tc._held_dd_pairs.clear()

    for st in tc._sat_states.values():
        if st.amb_key is not None:
            st.amb_gen += 1
        st.amb_key = None

    bias0 = tc.tc_bias if tc.tc_bias is not None else gtsam.imuBias.ConstantBias()

    g = gtsam.NonlinearFactorGraph()
    v = gtsam.Values()
    v.insert(tc.Xpose(0), pose_seed)
    v.insert(tc.Vel(0), vel_seed)
    v.insert(tc.Bias(0), bias0)
    rot_sigma = np.deg2rad(1.0)
    g.addPriorPose3(tc.Xpose(0), pose_seed,
                    gtsam.noiseModel.Diagonal.Sigmas(
                        np.array([rot_sigma, rot_sigma, 0.3, 2.0, 2.0, 3.0])))
    g.addPriorVector(tc.Vel(0), vel_seed, gtsam.noiseModel.Isotropic.Sigma(3, 3.0))
    g.addPriorConstantBias(tc.Bias(0), bias0, gtsam.noiseModel.Isotropic.Sigma(6, 0.01))
    tc.tc_bias = bias0

    ts0 = gtsam.FixedLagSmootherKeyTimestampMap()
    tc.tc_time += tc._epoch_dt
    ts0[tc.Xpose(0)] = tc.tc_time
    ts0[tc.Vel(0)] = tc.tc_time
    ts0[tc.Bias(0)] = tc.tc_time
    tc.isam2.update(g, v, ts0)
    tc.nav.x[0:3] = tc._antenna_ecef(pose_seed, ecef_seed)


def try_ddpr_reset(tc, obs, obsb, obs_sd, rs, rsb, sat, el, iu, ir_map,
                   pose_init, rot_seed, info, reason, vel_seed=None):
    """Try DDPR-only LS + warm Phase-2 reset after a smoother exception."""
    ecef_ddpr, _, res_rms = tc._ddpr_only_position(
        obs, obsb, obs_sd, rs, rsb, sat, el, iu, ir_map, pose_init)
    if ecef_ddpr is None or not np.isfinite(res_rms):
        return None, False
    info[reason] = True
    warm_reset_phase2(tc, ecef_ddpr, rot_seed, vel_seed=vel_seed)
    enu_body = tc.R_enu2ecef.T @ (ecef_ddpr - tc.base_ecef)
    ant = tc._antenna_ecef(gtsam.Pose3(rot_seed, gtsam.Point3(*enu_body)), ecef_ddpr)
    return ant, True


def _outage_drain_imu(tc, tow_obs):
    while tc.imu_idx < len(tc.imu_data):
        if tc.imu_data[tc.imu_idx]['tow'] > tow_obs + 1e-6:
            break
        tc.imu_idx += 1


def _outage_tick_sat_outc(tc, info):
    n_skip_reset = 0
    maxout = tc._sat_states.maxout
    for st in tc._sat_states.values():
        st.outc += 1
        if st.outc > maxout and st.amb_key is not None:
            st.amb_key = None
            st.amb_gen += 1
            n_skip_reset += 1
    if n_skip_reset:
        info['skip_amb_reset'] = n_skip_reset


def process_imu_only(tc, obs):
    """Advance the Phase-2 graph by one epoch using IMU + NHC + ZUPT only."""
    tc._update_epoch_dt(obs)
    tc._reset_current_epoch()
    R = tc.R_enu2ecef
    info = make_epoch_diagnostics(tc, imu_only=True)
    info['gnss_skip'] = True
    _, tow_obs = time2gpst(obs.t)

    if tc.phase == 1:
        _outage_drain_imu(tc, tow_obs)
        return advance_epoch_and_pack(tc, np.array(tc.nav.x[0:3]), 'FLT', 0, info, obs)

    tc.tc_epoch += 1
    key_idx = tc.tc_epoch
    info['tc_epoch'] = key_idx
    _outage_tick_sat_outc(tc, info)

    imu_idx_prev = tc.imu_idx
    pim, n_imu, gyro_mean = _tc_pim.build_pim(tc, tc.tc_bias, target_tow=tow_obs)
    if n_imu == 0:
        return advance_epoch_and_pack(tc, np.array(tc.nav.x[0:3]), 'FLT', 0, info, obs)
    info['n_imu'] = n_imu

    graph = gtsam.NonlinearFactorGraph()
    values = gtsam.Values()
    estimate = tc.isam2.calculateEstimate()
    if not estimate.exists(tc.Xpose(key_idx - 1)):
        return advance_epoch_and_pack(tc, np.array(tc.nav.x[0:3]), 'FLT', 0, info, obs)

    pose_p = estimate.atPose3(tc.Xpose(key_idx - 1))
    vel_prev = estimate.atVector(tc.Vel(key_idx - 1))
    bias_prev = estimate.atConstantBias(tc.Bias(key_idx - 1))
    pred = pim.predict(gtsam.NavState(pose_p, vel_prev), bias_prev)
    values.insert(tc.Xpose(key_idx), pred.pose())
    values.insert(tc.Vel(key_idx), pred.velocity())
    values.insert(tc.Bias(key_idx), bias_prev)
    _tc_pim.add_imu_chain(tc, graph, values, key_idx, pim, pose_p, vel_prev, info)

    speed_for_nhc = float(np.linalg.norm(np.asarray(vel_prev, dtype=float)[:2]))
    if _add_nhc_factor(tc, graph, key_idx, speed_for_nhc, gyro_mean_rh=gyro_mean):
        info['nhc'] = True
    _add_zupt_factor_inplace(tc, graph, key_idx, imu_idx_prev, n_imu, info)

    try:
        _tc_isam.fls_update(tc, graph, values, key_idx,
                            keep_keys=tc._sat_states.amb_key_values())
        estimate = tc.isam2.calculateEstimate()
        pose_tc = estimate.atPose3(tc.Xpose(key_idx))
        tc.tc_bias = estimate.atConstantBias(tc.Bias(key_idx))
        ecef_tc = R @ np.array(pose_tc.translation()) + tc.base_ecef
        sol = tc._antenna_ecef(pose_tc, ecef_tc)
        tc.nav.x[0:3] = sol
    except (RuntimeError, IndexError, ValueError) as ex:
        info['error'] = str(ex)
        sol = np.array(tc.nav.x[0:3])

    tc.nav.smode = 5
    return advance_epoch_and_pack(tc, sol, 'FLT', 0, info, obs)


def handle_solve_exception(tc, ex, pred, bias_prev, key_idx, obs, obsb, obs_sd,
                           rs, rsb, sat, el, iu, ir_map, info):
    """Recover cleanly from an iSAM2 numerical exception."""
    info['error'] = str(ex)
    ecef_ddpr_fb, ok = try_ddpr_reset(
        tc, obs, obsb, obs_sd, rs, rsb, sat, el, iu, ir_map,
        pred.pose(), pred.pose().rotation(),
        info, 'ddpr_exception_recover', vel_seed=pred.velocity())
    if ok:
        return advance_epoch_and_pack(tc, ecef_ddpr_fb, 'FLT', 0, info, obs)
    return advance_epoch_and_pack(tc, np.array(tc.nav.x[0:3]), 'FLT', 0, info, obs)
