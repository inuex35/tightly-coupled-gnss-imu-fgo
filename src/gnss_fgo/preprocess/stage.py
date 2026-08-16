"""Stage A — PIM build, IMU prediction, IMU chain attach.

Bumps tc_epoch, integrates IMU up to the GNSS TOW, and seeds Xpose/Vel/
Bias(kk) at the predicted state. If the previous Xpose was already
marginalised out of the FLS window we attempt a DDPR warm-reset
through ``recovery.try_ddpr_reset`` rather than continue.
"""

import numpy as np
import gtsam

from cssrlib.gnss import time2gpst
from ..buildfactor import imu_preintegration as _tc_pim
from ..utils import heading_from_pose
from .. import recovery as _tc_recovery


# ── Phase-2 pipeline contract (see stage_contract.py) ──────────────
STAGE_READS = (
    'R', 'bias_p', 'el', 'est2', 'g3', 'info', 'ir_map', 'iu', 'kk',
    'n_imu', 'obs', 'obs_sd', 'obsb', 'pim', 'pose_p', 'init_ecef', 'pred',
    'rs', 'rsb', 'sat', 'v3', 'vel_p',
)
STAGE_WRITES = (
    'bias_p', 'est2', 'g3', 'gyro_mean', 'imu_idx_prev', 'is_recovery',
    'kk', 'n_imu', 'pim', 'pose_p', 'pred', 'tow', 'v3', 'vel_p',
)


def run(tc, ed):
    """Stage A: IMU preintegration + pose/vel prediction from ISAM2 prior.

    Populates ed: kk, pim, n_imu, gyro_mean, is_recovery,
      g3, v3, est2, pose_p, vel_p, bias_p, pred.
    Early-return when n_imu==0 (no IMU samples) or prev pose is
    marginalized out (warm-reset via DDPR if possible).
    """
    info = ed.info
    tc.tc_epoch += 1
    ed.kk = tc.tc_epoch
    info['tc_epoch'] = ed.kk

    # IMU preintegration: integrate up to current GNSS epoch TOW.
    # Relaxed PIM on recovery so stale pose(kk-1) doesn't tightly bind.
    ed.is_recovery = getattr(tc, 'skip_count', 0) > 0
    ed.imu_idx_prev = tc.imu_idx
    _, tow_obs = time2gpst(ed.obs.t)
    ed.tow = tow_obs
    ed.pim, ed.n_imu, ed.gyro_mean = _tc_pim.build_pim(tc, 
        tc.tc_bias, target_tow=tow_obs)
    info['n_imu'] = ed.n_imu
    if ed.n_imu == 0:
        return _tc_recovery.finalize_epoch(tc, 
            tc.nav.x[0:3], 'FLT', 0, info, ed.obs)

    ed.g3 = gtsam.NonlinearFactorGraph()
    ed.v3 = gtsam.Values()
    ed.est2 = tc.isam2.calculateEstimate()

    if not ed.est2.exists(tc.Xpose(ed.kk - 1)):
        info['prev_pose_missing'] = ed.kk - 1
        dummy_pose = gtsam.Pose3(gtsam.Rot3.Identity(),
            gtsam.Point3(*(ed.R.T @ (ed.init_ecef - tc.base_ecef))))
        ecef_ddpr_pm, ok = _tc_recovery.try_ddpr_reset(tc, 
            ed.obs, ed.obsb, ed.obs_sd, ed.rs, ed.rsb,
            ed.sat, ed.el, ed.iu, ed.ir_map,
            dummy_pose, dummy_pose.rotation(), np.zeros(3),
            info, 'ddpr_prev_missing_recover')
        if ok:
            return _tc_recovery.finalize_epoch(tc, 
                ecef_ddpr_pm, 'FLT', 0, info, ed.obs)
        return _tc_recovery.finalize_epoch(tc, 
            tc.nav.x[0:3], 'FLT', 0, info, ed.obs)

    ed.pose_p = ed.est2.atPose3(tc.Xpose(ed.kk - 1))
    ed.vel_p = ed.est2.atVector(tc.Vel(ed.kk - 1))
    ed.bias_p = ed.est2.atConstantBias(tc.Bias(ed.kk - 1))
    ed.pred = ed.pim.predict(
        gtsam.NavState(ed.pose_p, ed.vel_p), ed.bias_p)
    info['pred_heading_deg'] = heading_from_pose(ed.pred.pose())
    ed.v3.insert(tc.Xpose(ed.kk), ed.pred.pose())
    ed.v3.insert(tc.Vel(ed.kk), ed.pred.velocity())
    ed.v3.insert(tc.Bias(ed.kk), ed.bias_p)
    _tc_pim.add_imu_chain(tc, ed.g3, ed.v3, ed.kk, ed.pim,
                        ed.pose_p, ed.vel_p, info)
    return None
