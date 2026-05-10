"""ZUPT / ZARU / streak-anchor pseudo-measurements.

GICI-style stationary detection on the current epoch's IMU window
drives up to three optional pseudo-measurements:

  * ``PriorFactorVector(Vel(kk), 0, σ)``                   — ZUPT
  * ``BetweenFactorPose3(X(kk-1), X(kk), I, σ_rot)``       — ZARU
  * ``PriorFactorPose3(X(kk), captured_pose, [σ_rot,σ_t])`` — streak anchor

Each is independently gated by its own σ knob in ``cfg.zupt.*``.
``add_zupt_factors`` is callable from any code path
(optimize.py and the recovery outage paths). ``add_zupt_factors_for_stage`` is
the optimize-stage ed wrapper.

Extracted from ``factors_support.py`` during the Phase 2 architectural
refactor; behaviour unchanged.
"""

import numpy as np
import gtsam

from ..utils import compute_zupt_stats as _utils_compute_zupt_stats


def _clear_zupt_anchor(rec):
    if rec is not None:
        rec.zupt_anchor_pose = None
        rec.zupt_anchor_start_ep = None


def _zupt_should_fire(tc, n_imu, info, vel_prev, gnss_available):
    """Run the ZUPT detection gates and return ``stats`` dict on hit,
    or ``None`` when any gate fails. ``info`` is populated with the
    diagnostic stats whenever they were computed."""
    cfg = tc.cfg
    if not int(getattr(cfg, 'zupt_enable', 0) or 0):
        return None
    if int(n_imu) < int(getattr(cfg, 'zupt_min_samples', 5)):
        return None
    # Velocity gate — only when GNSS is constraining the estimate.
    max_speed = float(getattr(cfg, 'zupt_max_speed', 0.0))
    if max_speed > 0 and vel_prev is not None and gnss_available:
        sp = float(np.linalg.norm(np.asarray(vel_prev, dtype=np.float64)))
        info['zupt_speed_prev'] = sp
        if sp > max_speed:
            return None
    # Bias-init-subtracted IMU stats.
    bias_init = getattr(tc, 'tc_bias_init', None)
    if bias_init is not None:
        ref_acc = np.asarray(bias_init.accelerometer(), dtype=np.float64)
        ref_gyro = np.asarray(bias_init.gyroscope(), dtype=np.float64)
    else:
        ref_acc = ref_gyro = None
    samples = tc.imu_data[int(info.get('_zupt_idx_prev', tc.imu_idx)):tc.imu_idx]
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


def _add_zero_velocity_prior(tc, g3, kk, info, sigma):
    if sigma <= 0:
        return False
    g3.add(gtsam.PriorFactorVector(
        tc.Vel(kk),
        np.zeros(3, dtype=np.float64),
        gtsam.noiseModel.Isotropic.Sigma(3, sigma)))
    info['zupt'] = True
    return True


def _add_zaru_factor(tc, g3, kk, info, sigma_rot):
    if sigma_rot <= 0 or kk <= 0:
        return False
    sigmas_pose = np.array(
        [sigma_rot, sigma_rot, sigma_rot, 1e3, 1e3, 1e3])
    g3.add(gtsam.BetweenFactorPose3(
        tc.Xpose(kk - 1), tc.Xpose(kk),
        gtsam.Pose3(),
        gtsam.noiseModel.Diagonal.Sigmas(sigmas_pose)))
    info['zaru'] = True
    return True


def _maybe_capture_or_apply_anchor(tc, g3, kk, info, rec, pose_prev,
                                    sig_t, sig_r):
    if rec is None or sig_t <= 0 or sig_r <= 0:
        return False
    if rec.zupt_anchor_pose is None:
        if pose_prev is None:
            return False
        rec.zupt_anchor_pose = pose_prev
        rec.zupt_anchor_start_ep = int(kk)
        info['zupt_anchor_capture'] = int(kk)
        return True
    sigmas_anchor = np.array(
        [sig_r, sig_r, sig_r, sig_t, sig_t, sig_t])
    g3.add(gtsam.PriorFactorPose3(
        tc.Xpose(kk),
        rec.zupt_anchor_pose,
        gtsam.noiseModel.Diagonal.Sigmas(sigmas_anchor)))
    info['zupt_anchor'] = int(rec.zupt_anchor_start_ep or 0)
    return True


def add_zupt_factors(tc, g3, kk, imu_idx_prev, n_imu, info,
                             pose_prev=None, gnss_available=True,
                             vel_prev=None):
    """GICI-style ZUPT, callable from optimize.py and the recovery
    outage paths.

    On a stationary detection (gates: acc_std / gyro_std / gyro_median
    against bias-init-subtracted residuals; gravity check optional;
    smoother-velocity gate when GNSS is available) up to three pseudo-
    measurements fire:

      * ``PriorFactorVector(Vel(kk), 0, σ)``   — vel = 0
      * ``BetweenFactorPose3(X(kk-1), X(kk), I, σ)`` — ZARU (off by
        default; pins inter-pose rotation)
      * ``PriorFactorPose3(X(kk), anchor, σ)`` — first-epoch capture
        + reuse for the rest of a contiguous stationary streak. Only
        active when ``gnss_available`` is False so DD-having epochs
        keep their direct pose constraint untouched.

    Each component is independently gated by its σ knob so the suite
    can be turned on / off piece-by-piece via env. Always writes the
    diagnostic stats into ``info`` — even when no factor fires.
    """
    rec = tc._recovery
    info['_zupt_idx_prev'] = int(imu_idx_prev)
    stats = _zupt_should_fire(tc, n_imu, info, vel_prev, gnss_available)
    info.pop('_zupt_idx_prev', None)
    if stats is None:
        _clear_zupt_anchor(rec)
        return False
    cfg = tc.cfg
    any_added = False
    any_added |= _add_zero_velocity_prior(
        tc, g3, kk, info, float(cfg.zupt_sigma_zero_velocity))
    any_added |= _add_zaru_factor(
        tc, g3, kk, info, float(getattr(cfg, 'zupt_sigma_zero_rotation', 0.0)))
    if not gnss_available:
        any_added |= _maybe_capture_or_apply_anchor(
            tc, g3, kk, info, rec, pose_prev,
            float(getattr(cfg, 'zupt_anchor_sigma_translation', 0.0)),
            float(getattr(cfg, 'zupt_anchor_sigma_rotation', 0.0)))
    return any_added


def add_zupt_factors_for_stage(tc, ed):
    """Phase-2 optimize-stage wrapper around :func:`add_zupt_factors`.
    Optimize stage is the DD-having path — flag ``gnss_available`` from
    ``ed.nb`` so the anchor branch defers to the recovery entry
    points where DD has not constrained pose this epoch.
    """
    return add_zupt_factors(
        tc, ed.g3, ed.kk,
        int(getattr(ed, 'imu_idx_prev', tc.imu_idx)),
        int(getattr(ed, 'n_imu', 0)),
        ed.info,
        pose_prev=getattr(ed, 'pose_p', None),
        gnss_available=int(getattr(ed, 'nb', 0)) > 0,
        vel_prev=getattr(ed, 'vel_p', None))

