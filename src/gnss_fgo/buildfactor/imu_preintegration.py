"""IMU preintegration + IMU chain factors.

Owns:
  * PIM construction (``build_pim`` / ``build_pim_from_idx``) with the
    per-epoch integration-covariance override that scales σ_pos by
    last-epoch DDPR residual when configured.
  * IMU chain wiring (``add_imu_chain``) that adds the
    ``CombinedImuFactor`` between consecutive epochs and the
    per-epoch ``BetweenFactorConstantBias``.
  * Bias σ helpers (``bias_between_noise`` / ``bias_prior_noise`` /
    ``bias_prior_anchor``).

Extracted from ``factors_support.py`` during the Phase 2 architectural
refactor; behaviour unchanged.
"""

import numpy as np
import gtsam

from ..utils import build_pim as _utils_build_pim


def bias_between_noise(tc):
    """Diagonal sigma for BetweenFactorConstantBias (acc / gyro split)."""
    ag = tc.cfg.bias_between_acc_sigma
    gy = tc.cfg.bias_between_gyro_sigma
    return gtsam.noiseModel.Diagonal.Sigmas(
        np.array([ag, ag, ag, gy, gy, gy]))


def bias_prior_noise(tc):
    """Diagonal sigma for the per-epoch bias anchor (acc / gyro split)."""
    ag = tc.cfg.bias_prior_acc_sigma
    gy = tc.cfg.bias_prior_gyro_sigma
    return gtsam.noiseModel.Diagonal.Sigmas(
        np.array([ag, ag, ag, gy, gy, gy]))


def bias_prior_anchor(tc, bias_prev):
    """Return the configured absolute bias anchor."""
    mode = int(tc.cfg.bias_prior_mode)
    if mode == 0:
        return None
    if mode == 1:
        return tc.tc_bias_init
    if bias_prev is not None:
        return bias_prev
    if tc.tc_bias is not None:
        return tc.tc_bias
    return tc.tc_bias_init


def _apply_mres_integ_cov_override(tc):
    """Per-epoch ``integ_eff = min(imu_integ_cov_max, max(imu_integ_cov, mres²/dt))``."""
    last_mres = float(tc._last_main_ddpr_res)
    last_ep = int(tc._last_main_ddpr_epoch)
    stale_max = int(tc.cfg.ddcp_res_weight_stale_max_epochs)
    is_stale = stale_max > 0 and (tc.epoch - last_ep) > stale_max
    default_cov = float(tc.cfg.imu_integ_cov)
    # Smooth inflation per epoch: integ_eff = max(default, mres²/dt),
    # capped by imu_integ_cov_max. No threshold / window cliffs — small
    # mres degrades naturally to default via the max(). Stale signals
    # (no recent DDPR) are skipped so a long IMU-only outage doesn't
    # carry old inflation.
    if is_stale:
        integ_eff = default_cov
    else:
        integ_eff = max(default_cov, last_mres ** 2 / max(tc._epoch_dt, 1e-3))
        cap = float(tc.cfg.imu_integ_cov_max)
        if cap > 0:
            integ_eff = min(integ_eff, cap)
    tc.imu_params.setIntegrationCovariance(integ_eff * np.eye(3))


def build_pim(tc, bias, target_tow=None):
    """Thin adapter — see utils.imu.build_pim. Advances tc.imu_idx."""
    pim, n, gyro_mean, tc.imu_idx = build_pim_from_idx(
        tc, bias, tc.imu_idx, target_tow=target_tow)
    return pim, n, gyro_mean


def build_pim_from_idx(tc, bias, imu_idx, target_tow=None):
    """Build a PIM from an explicit IMU cursor without mutating tc state."""
    _apply_mres_integ_cov_override(tc)
    return _utils_build_pim(
        tc.imu_params, bias, tc.imu_data, imu_idx,
        target_tow=target_tow)


def add_imu_chain(tc, graph, values, kk, pim, pose_p, vel_p, info):
    """Attach the IMU chain or break it after a reset."""
    if tc._pim_discontinuity:
        info['pim_discontinuity'] = True
        bias_for_pred = tc.tc_bias if tc.tc_bias is not None \
            else tc.tc_bias_init
        try:
            nav_p = gtsam.NavState(pose_p, vel_p)
            nav_pred = pim.predict(nav_p, bias_for_pred)
            seed_pose = nav_pred.pose()
            seed_vel = np.array(nav_pred.velocity())
        except RuntimeError:
            dt = float(tc._epoch_dt)
            seed_trans = np.array(pose_p.translation()) + vel_p * dt
            seed_pose = gtsam.Pose3(pose_p.rotation(),
                                     gtsam.Point3(*seed_trans))
            seed_vel = vel_p
        values.update(tc.Xpose(kk), seed_pose)
        values.update(tc.Vel(kk), seed_vel)
        trans_sig = float(tc.cfg.pim_break_trans_sigma)
        graph.addPriorPose3(tc.Xpose(kk), seed_pose,
            gtsam.noiseModel.Diagonal.Sigmas(
                np.array([0.1, 0.1, 0.3, trans_sig, trans_sig, trans_sig])))
        graph.addPriorVector(tc.Vel(kk), seed_vel,
            gtsam.noiseModel.Isotropic.Sigma(3, 2.0))
        bias_anchor = tc.tc_bias if tc.tc_bias is not None \
            else tc.tc_bias_init
        graph.addPriorConstantBias(tc.Bias(kk), bias_anchor,
            gtsam.noiseModel.Isotropic.Sigma(6, 0.01))
        tc._pim_discontinuity = False
        return

    graph.add(gtsam.CombinedImuFactor(
        tc.Xpose(kk - 1), tc.Vel(kk - 1),
        tc.Xpose(kk), tc.Vel(kk),
        tc.Bias(kk - 1), tc.Bias(kk), pim))
    graph.add(gtsam.BetweenFactorConstantBias(
        tc.Bias(kk - 1), tc.Bias(kk),
        gtsam.imuBias.ConstantBias(np.zeros(3), np.zeros(3)),
        bias_between_noise(tc)))
    bias_anchor = bias_prior_anchor(tc, tc.tc_bias)
    if bias_anchor is not None:
        graph.addPriorConstantBias(tc.Bias(kk), bias_anchor,
            bias_prior_noise(tc))
