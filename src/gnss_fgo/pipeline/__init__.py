"""Single-phase tightly-coupled epoch pipeline: bootstrap/IMU prediction -> slip check -> solve -> AR -> report."""

import numpy as np
import gtsam
from cssrlib.gnss import time2gpst

from .. import ar as _tc_ar
from ..factors import doppler_sd as _tc_doppler_sd
from ..factors import factors as _tc_factors
from ..factors import imu_preintegration as _tc_pim
from ..factors import nhc as _tc_nhc
from ..factors import zupt as _tc_zupt
from ..integrity import recovery as _tc_recovery
from ..integrity import slip_detect as _tc_slip_detect
from ..state.epoch_data import EpochData
from ..utils import euler_to_R_body2enu, heading_from_pose, sensor_to_body_flu
from . import update_smoother as _tc_isam


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
    tc._collect_imu_samples(target_tow=tow_obs)
    bias_acc, bias_gyro = tc._estimate_stationary_bias()
    bias0 = gtsam.imuBias.ConstantBias(bias_acc, bias_gyro)
    tc.tc_bias = bias0
    tc.tc_bias_init = bias0

    n_static = max(1, min(500, tc.imu_idx))
    acc_avg = np.mean([im['acc'] for im in tc.imu_data[:n_static]], axis=0)
    acc_tilt = sensor_to_body_flu(np.array(acc_avg - bias_acc, dtype=float))
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
    return _tc_recovery.advance_epoch_and_pack(tc, sol, 'FLT', 0, info, obs)


def run_tc_epoch(tc, obs, obsb, rs, vs, rsb, sat, el, iu,
                 obs_sd, ir_map, info, ns, init_ecef, R_enu2ecef):
    """Single-phase IMU/GNSS TC epoch flow."""
    if tc.isam2 is None:
        return _bootstrap_epoch0(
            tc, obs, obsb, rs, rsb, sat, el, iu, obs_sd, ir_map,
            init_ecef, R_enu2ecef, info)

    epoch = EpochData(
        obs=obs, obsb=obsb, rs=rs, vs=vs, rsb=rsb,
        sat=sat, el=el, iu=iu, obs_sd=obs_sd, ir_map=ir_map,
        info=info, ns=ns, init_ecef=init_ecef, R_enu2ecef=R_enu2ecef)

    # 1. IMU preintegration & state prediction
    tc.tc_epoch += 1
    epoch.key_idx = tc.tc_epoch
    info['tc_epoch'] = epoch.key_idx
    epoch.imu_idx_prev = tc.imu_idx
    _, tow_obs = time2gpst(obs.t)
    epoch.tow = tow_obs
    epoch.pim, epoch.n_imu, epoch.gyro_mean = _tc_pim.build_pim(
        tc, tc.tc_bias, target_tow=tow_obs)
    info['n_imu'] = epoch.n_imu
    if epoch.n_imu == 0:
        return _tc_recovery.advance_epoch_and_pack(
            tc, np.array(tc.nav.x[0:3]), 'FLT', 0, info, obs)

    epoch.graph = gtsam.NonlinearFactorGraph()
    epoch.values = gtsam.Values()
    epoch.estimate = tc.isam2.calculateEstimate()

    if not epoch.estimate.exists(tc.Xpose(epoch.key_idx - 1)):
        info['prev_pose_missing'] = epoch.key_idx - 1
        return _tc_recovery.advance_epoch_and_pack(
            tc, np.array(tc.nav.x[0:3]), 'FLT', 0, info, obs)

    epoch.pose_p = epoch.estimate.atPose3(tc.Xpose(epoch.key_idx - 1))
    epoch.vel_prev = epoch.estimate.atVector(tc.Vel(epoch.key_idx - 1))
    epoch.bias_prev = epoch.estimate.atConstantBias(tc.Bias(epoch.key_idx - 1))
    epoch.pred_nav = epoch.pim.predict(
        gtsam.NavState(epoch.pose_p, epoch.vel_prev), epoch.bias_prev)
    info['pred_heading_deg'] = heading_from_pose(epoch.pred_nav.pose())
    epoch.values.insert(tc.Xpose(epoch.key_idx), epoch.pred_nav.pose())
    epoch.values.insert(tc.Vel(epoch.key_idx), epoch.pred_nav.velocity())
    epoch.values.insert(tc.Bias(epoch.key_idx), epoch.bias_prev)
    _tc_pim.add_imu_chain(tc, epoch.graph, epoch.values, epoch.key_idx,
                          epoch.pim, epoch.pose_p, epoch.vel_prev, info)

    # 2. Cycle-slip detection & predicted antenna ECEF
    info['nsat'] = ns
    n_reset, _ = _tc_slip_detect.detect_slips_and_reset_ambiguities(
        tc, obs, obs_sd, sat, iu)
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
    _tc_doppler_sd.add_sd_doppler_factors(tc, epoch)
    if _tc_nhc.add_nhc_factor(tc, epoch.graph, epoch.key_idx,
                              gyro_mean_rh=epoch.gyro_mean):
        info['nhc'] = True
    _tc_zupt.add_zupt_factors_for_stage(tc, epoch)

    # 4. Fixed-lag smoother update
    early = _tc_isam._solve_isam2(tc, epoch)
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
    return _tc_recovery.advance_epoch_and_pack(
        tc, epoch.sol, epoch.tag, epoch.nb, info, obs)
