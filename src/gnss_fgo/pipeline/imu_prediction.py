"""IMU prediction: preintegrate to the obs epoch, predict the state,
and attach that prediction as the process factor (CombinedImuFactor).
Measurement factors live in measurement_factors.py.

Bumps tc_epoch, integrates IMU up to the GNSS TOW, and seeds Xpose/Vel/
Bias(key_idx) at the predicted state. If the previous Xpose was already
marginalised out of the FLS window we attempt a DDPR warm-reset
through ``recovery.try_ddpr_reset`` rather than continue.
"""

import numpy as np
import gtsam

from cssrlib.gnss import time2gpst
from ..factors import imu_preintegration as _tc_pim
from ..utils import heading_from_pose
from ..integrity import recovery as _tc_recovery


# ── Phase-2 pipeline contract (see stage_contract.py) ──────────────
STAGE_READS = (
    'R_enu2ecef', 'bias_prev', 'el', 'estimate', 'graph', 'info', 'ir_map', 'iu', 'key_idx',
    'n_imu', 'obs', 'obs_sd', 'obsb', 'pim', 'pose_p', 'init_ecef', 'pred_nav',
    'rs', 'rsb', 'sat', 'values', 'vel_prev',
)
STAGE_WRITES = (
    'bias_prev', 'estimate', 'graph', 'gyro_mean', 'imu_idx_prev', 'is_recovery',
    'key_idx', 'n_imu', 'pim', 'pose_p', 'pred_nav', 'tow', 'values', 'vel_prev',
)


def run(tc, epoch):
    """Stage A: IMU preintegration + pose/vel prediction from ISAM2 prior.

    Populates epoch: key_idx, pim, n_imu, gyro_mean, is_recovery,
      graph, values, estimate, pose_p, vel_prev, bias_prev, pred.
    Early-return when n_imu==0 (no IMU samples) or prev pose is
    marginalized out (warm-reset via DDPR if possible).
    """
    info = epoch.info
    tc.tc_epoch += 1
    epoch.key_idx = tc.tc_epoch
    info['tc_epoch'] = epoch.key_idx

    # IMU preintegration: integrate up to current GNSS epoch TOW.
    # Relaxed PIM on recovery so stale pose(key_idx-1) doesn't tightly bind.
    epoch.is_recovery = tc.skip_count > 0
    epoch.imu_idx_prev = tc.imu_idx
    _, tow_obs = time2gpst(epoch.obs.t)
    epoch.tow = tow_obs
    epoch.pim, epoch.n_imu, epoch.gyro_mean = _tc_pim.build_pim(tc, 
        tc.tc_bias, target_tow=tow_obs)
    info['n_imu'] = epoch.n_imu
    # Fixed-dt integration audit: build_pim integrates every sample at a
    # nominal 0.01 s; if the CSV is gappy or phase-shifted this drifts
    # from the true obs interval (review finding #1 — measure first).
    info['pim_dt_mismatch'] = round(
        epoch.n_imu * 0.01 - float(tc._epoch_dt), 6)
    if epoch.n_imu == 0:
        return _tc_recovery.advance_epoch_and_pack(tc, 
            tc.nav.x[0:3], 'FLT', 0, info, epoch.obs)

    epoch.graph = gtsam.NonlinearFactorGraph()
    epoch.values = gtsam.Values()
    epoch.estimate = tc.isam2.calculateEstimate()

    if not epoch.estimate.exists(tc.Xpose(epoch.key_idx - 1)):
        info['prev_pose_missing'] = epoch.key_idx - 1
        dummy_pose = gtsam.Pose3(gtsam.Rot3.Identity(),
            gtsam.Point3(*(epoch.R_enu2ecef.T @ (epoch.init_ecef - tc.base_ecef))))
        ecef_ddpr_pm, ok = _tc_recovery.try_ddpr_reset(tc, 
            epoch.obs, epoch.obsb, epoch.obs_sd, epoch.rs, epoch.rsb,
            epoch.sat, epoch.el, epoch.iu, epoch.ir_map,
            dummy_pose, dummy_pose.rotation(), np.zeros(3),
            info, 'ddpr_prev_missing_recover')
        if ok:
            return _tc_recovery.advance_epoch_and_pack(tc, 
                ecef_ddpr_pm, 'FLT', 0, info, epoch.obs)
        return _tc_recovery.advance_epoch_and_pack(tc, 
            tc.nav.x[0:3], 'FLT', 0, info, epoch.obs)

    epoch.pose_p = epoch.estimate.atPose3(tc.Xpose(epoch.key_idx - 1))
    epoch.vel_prev = epoch.estimate.atVector(tc.Vel(epoch.key_idx - 1))
    epoch.bias_prev = epoch.estimate.atConstantBias(tc.Bias(epoch.key_idx - 1))
    epoch.pred_nav = epoch.pim.predict(
        gtsam.NavState(epoch.pose_p, epoch.vel_prev), epoch.bias_prev)
    info['pred_heading_deg'] = heading_from_pose(epoch.pred_nav.pose())
    epoch.values.insert(tc.Xpose(epoch.key_idx), epoch.pred_nav.pose())
    epoch.values.insert(tc.Vel(epoch.key_idx), epoch.pred_nav.velocity())
    epoch.values.insert(tc.Bias(epoch.key_idx), epoch.bias_prev)
    _tc_pim.add_imu_chain(tc, epoch.graph, epoch.values, epoch.key_idx, epoch.pim,
                        epoch.pose_p, epoch.vel_prev, info)
    return None
