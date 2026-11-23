"""ZUPT / ZARU stationary constraints (zero velocity + zero angular rate)."""

import numpy as np
import gtsam

from ..utils import compute_zupt_stats as _utils_compute_zupt_stats

_ZUPT_MIN_SAMPLES = 5  # minimum IMU samples for valid sample std


def _zupt_should_fire(tc, n_imu, info, imu_idx_prev):
    """Run IMU stillness check and return stats dict on hit, else None."""
    cfg = tc.cfg
    if (float(cfg.zupt_sigma_zero_velocity) <= 0
            and float(cfg.zupt_sigma_zero_rotation) <= 0):
        return None
    if int(n_imu) < _ZUPT_MIN_SAMPLES:
        return None
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
    if stats['acc_std'] > float(cfg.zupt_max_acc_std):
        return None
    if stats['gyro_std'] > float(cfg.zupt_max_gyro_std):
        return None
    if stats['gyro_median'] > float(cfg.zupt_max_gyro_median):
        return None
    return stats


def add_zupt_factors(tc, graph, key_idx, imu_idx_prev, n_imu, info):
    """Add ZUPT (vel=0) and ZARU (delta_R=I) factors when stationary."""
    stats = _zupt_should_fire(tc, n_imu, info, int(imu_idx_prev))
    if stats is None:
        return False
    cfg = tc.cfg
    any_added = False
    sig_v = float(cfg.zupt_sigma_zero_velocity)
    if sig_v > 0:
        graph.add(gtsam.PriorFactorVector(
            tc.Vel(key_idx),
            np.zeros(3, dtype=np.float64),
            gtsam.noiseModel.Isotropic.Sigma(3, sig_v)))
        info['zupt'] = True
        any_added = True
    sig_r = float(cfg.zupt_sigma_zero_rotation)
    if sig_r > 0:
        sigmas_bias = np.array([np.inf, np.inf, np.inf, sig_r, sig_r, sig_r])
        graph.add(gtsam.PriorFactorConstantBias(
            tc.Bias(key_idx),
            gtsam.imuBias.ConstantBias(np.zeros(3), stats['gyro_mean']),
            gtsam.noiseModel.Diagonal.Sigmas(sigmas_bias)))
        info['zaru'] = True
        any_added = True
    return any_added


def add_zupt_factors_for_stage(tc, epoch):
    """Phase-2 Stage C1 wrapper around add_zupt_factors."""
    return add_zupt_factors(
        tc, epoch.graph, epoch.key_idx,
        int(epoch.imu_idx_prev if epoch.imu_idx_prev is not None
            else tc.imu_idx),
        int(epoch.n_imu),
        epoch.info)
