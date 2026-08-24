"""ZUPT / ZARU / streak-anchor pseudo-measurements.

GICI-style stationary detection on the current epoch's IMU window
drives up to three optional pseudo-measurements:

  * ``PriorFactorVector(Vel(key_idx), 0, σ)``                   — ZUPT
  * ``BetweenFactorPose3(X(key_idx-1), X(key_idx), I, σ_rot)``       — ZARU
  * ``PriorFactorPose3(X(key_idx), captured_pose, [σ_rot,σ_t])`` — streak anchor

Each is independently gated by its own σ knob (``cfg.zupt_*``).
``add_zupt_factors`` is callable from any code path
(the C1 factor stage and the recovery outage paths);
``add_zupt_factors_for_stage`` is its epoch wrapper for Stage C1.
"""

import numpy as np
import gtsam

from ..utils import compute_zupt_stats as _utils_compute_zupt_stats


def _clear_zupt_anchor(rec):
    """Drop the stored ZUPT anchor pose (stationary segment ended)."""
    if rec is not None:
        rec.zupt_anchor_pose = None
        rec.zupt_anchor_start_ep = None


def _zupt_should_fire(tc, n_imu, info, imu_idx_prev, vel_prev, gnss_available):
    """Run the ZUPT detection gates and return ``stats`` dict on hit,
    or ``None`` when any gate fails. ``info`` is populated with the
    diagnostic stats whenever they were computed."""
    cfg = tc.cfg
    if not int(cfg.zupt_enable or 0):
        return None
    if int(n_imu) < int(cfg.zupt_min_samples):
        return None
    # Velocity gate — only when GNSS is constraining the estimate.
    max_speed = float(cfg.zupt_max_speed)
    if max_speed > 0 and vel_prev is not None and gnss_available:
        sp = float(np.linalg.norm(np.asarray(vel_prev, dtype=np.float64)))
        info['zupt_speed_prev'] = sp
        if sp > max_speed:
            return None
    # Bias-init-subtracted IMU stats.
    bias_init = tc.tc_bias_init
    if bias_init is not None:
        ref_acc = np.asarray(bias_init.accelerometer(), dtype=np.float64)
        ref_gyro = np.asarray(bias_init.gyroscope(), dtype=np.float64)
    else:
        ref_acc = ref_gyro = None
    samples = tc.imu_data[int(imu_idx_prev):tc.imu_idx]
    stats = _utils_compute_zupt_stats(samples, ref_acc, ref_gyro)
    if stats is None:
        return None
    info['zupt_acc_std'] = stats['acc_std']
    info['zupt_gyro_std'] = stats['gyro_std']
    info['zupt_gyro_median'] = stats['gyro_median']
    g_dev = abs(stats['acc_norm_mean'] - 9.81)
    info['zupt_g_dev'] = g_dev
    if stats['acc_std'] > float(cfg.zupt_max_acc_std):
        return None
    if stats['gyro_std'] > float(cfg.zupt_max_gyro_std):
        return None
    if stats['gyro_median'] > float(cfg.zupt_max_gyro_median):
        return None
    g_dev_thr = float(cfg.zupt_g_dev_thr)
    if g_dev_thr > 0 and g_dev > g_dev_thr:
        return None
    return stats


def _add_zero_velocity_prior(tc, graph, key_idx, info, sigma):
    """Prior v=0 on this epoch's velocity key."""
    if sigma <= 0:
        return False
    graph.add(gtsam.PriorFactorVector(
        tc.Vel(key_idx),
        np.zeros(3, dtype=np.float64),
        gtsam.noiseModel.Isotropic.Sigma(3, sigma)))
    info['zupt'] = True
    return True


def _add_zaru_factor(tc, graph, key_idx, info, sigma_rot):
    """Zero-angular-rate Between on the pose pair (rotation tight, translation loose)."""
    if sigma_rot <= 0 or key_idx <= 0:
        return False
    sigmas_pose = np.array(
        [sigma_rot, sigma_rot, sigma_rot, 1e3, 1e3, 1e3])
    graph.add(gtsam.BetweenFactorPose3(
        tc.Xpose(key_idx - 1), tc.Xpose(key_idx),
        gtsam.Pose3(),
        gtsam.noiseModel.Diagonal.Sigmas(sigmas_pose)))
    info['zaru'] = True
    return True


def _maybe_capture_or_apply_anchor(tc, graph, key_idx, info, rec, pose_prev,
                                    sig_t, sig_r):
    """First stationary epoch captures the anchor pose; later ones get a prior to it."""
    if rec is None or sig_t <= 0 or sig_r <= 0:
        return False
    if rec.zupt_anchor_pose is None:
        if pose_prev is None:
            return False
        rec.zupt_anchor_pose = pose_prev
        rec.zupt_anchor_start_ep = int(key_idx)
        info['zupt_anchor_capture'] = int(key_idx)
        return True
    sigmas_anchor = np.array(
        [sig_r, sig_r, sig_r, sig_t, sig_t, sig_t])
    graph.add(gtsam.PriorFactorPose3(
        tc.Xpose(key_idx),
        rec.zupt_anchor_pose,
        gtsam.noiseModel.Diagonal.Sigmas(sigmas_anchor)))
    info['zupt_anchor'] = int(rec.zupt_anchor_start_ep or 0)
    return True


def add_zupt_factors(tc, graph, key_idx, imu_idx_prev, n_imu, info,
                             pose_prev=None, gnss_available=True,
                             vel_prev=None):
    """GICI-style ZUPT, callable from the C1 factor stage and the
    recovery outage paths.

    On a stationary detection (gates: acc_std / gyro_std / gyro_median
    against bias-init-subtracted residuals; gravity check optional;
    smoother-velocity gate when GNSS is available) up to three pseudo-
    measurements fire:

      * ``PriorFactorVector(Vel(key_idx), 0, σ)``   — vel = 0
      * ``BetweenFactorPose3(X(key_idx-1), X(key_idx), I, σ)`` — ZARU (off by
        default; pins inter-pose rotation)
      * ``PriorFactorPose3(X(key_idx), anchor, σ)`` — first-epoch capture
        + reuse for the rest of a contiguous stationary streak. Only
        active when ``gnss_available`` is False so DD-having epochs
        keep their direct pose constraint untouched.

    Each component is independently gated by its σ knob so the suite
    can be turned on / off piece-by-piece via env. Always writes the
    diagnostic stats into ``info`` — even when no factor fires.
    """
    rec = tc._recovery
    stats = _zupt_should_fire(tc, n_imu, info, int(imu_idx_prev),
                              vel_prev, gnss_available)
    if stats is None:
        _clear_zupt_anchor(rec)
        return False
    cfg = tc.cfg
    any_added = False
    any_added |= _add_zero_velocity_prior(
        tc, graph, key_idx, info, float(cfg.zupt_sigma_zero_velocity))
    any_added |= _add_zaru_factor(
        tc, graph, key_idx, info, float(cfg.zupt_sigma_zero_rotation))
    if not gnss_available:
        any_added |= _maybe_capture_or_apply_anchor(
            tc, graph, key_idx, info, rec, pose_prev,
            float(cfg.zupt_anchor_sigma_translation),
            float(cfg.zupt_anchor_sigma_rotation))
    return any_added


def add_zupt_factors_for_stage(tc, epoch):
    """Phase-2 optimize-stage wrapper around :func:`add_zupt_factors`.
    Passes ``gnss_available=False`` deliberately, so the stationary
    anchor stays unconditional (see the note at the call below).
    """
    return add_zupt_factors(
        tc, epoch.graph, epoch.key_idx,
        int(epoch.imu_idx_prev if epoch.imu_idx_prev is not None
            else tc.imu_idx),
        int(epoch.n_imu),
        epoch.info,
        pose_prev=epoch.pose_p,
        # Deliberately False: epoch.nb is not written until the AR
        # stage, so the original nb>0 read was always False here — and
        # the "corrected" gate was measured worse. The stationary anchor at
        # stops is load-bearing; keep it unconditional and say so.
        gnss_available=False,
        vel_prev=epoch.vel_prev)

