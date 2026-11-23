"""ZUPT / ZARU stationary constraints (zero velocity + direct gyro-bias observation)."""

import numpy as np
import gtsam

from ..utils import compute_zupt_stats as _utils_compute_zupt_stats

_ZUPT_MIN_SAMPLES = 5         # minimum IMU samples for valid sample std
_ZUPT_MAX_ACC_STD = 0.55      # stationary accelerometer std gate [m/s²]
_ZUPT_MAX_GYRO_STD = 0.030    # stationary gyroscope std gate [rad/s]
_ZUPT_MAX_GYRO_MEDIAN = 0.020 # stationary gyroscope median gate [rad/s]
_ZUPT_SIGMA_VEL = 0.5         # zero-velocity prior σ [m/s]
_ZARU_SIGMA_GYRO_BIAS = 0.010 # stationary gyro-bias prior σ [rad/s]


def is_still_imu_window(acc: np.ndarray, gyro: np.ndarray,
                        bias_gyro_ref: np.ndarray | None = None) -> bool:
    """Shared IMU stillness check for both sequential and batch pipelines."""
    if len(acc) < _ZUPT_MIN_SAMPLES:
        return False
    acc_std = float(np.sqrt(np.mean(np.sum((acc - acc.mean(axis=0)) ** 2, axis=1))))
    gyro_std = float(np.sqrt(np.mean(np.sum((gyro - gyro.mean(axis=0)) ** 2, axis=1))))
    resid = gyro if bias_gyro_ref is None else (gyro - bias_gyro_ref)
    gyro_med = float(np.median(np.linalg.norm(resid, axis=1)))
    return (acc_std <= _ZUPT_MAX_ACC_STD
            and gyro_std <= _ZUPT_MAX_GYRO_STD
            and gyro_med <= _ZUPT_MAX_GYRO_MEDIAN)


def _zupt_should_fire(tc, n_imu, info, imu_idx_prev):
    """Run IMU stillness check and return stats dict on hit, else None."""
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
    if stats['acc_std'] > _ZUPT_MAX_ACC_STD:
        return None
    if stats['gyro_std'] > _ZUPT_MAX_GYRO_STD:
        return None
    if stats['gyro_median'] > _ZUPT_MAX_GYRO_MEDIAN:
        return None
    return stats


def add_zupt_factors(tc, graph, key_idx, imu_idx_prev, n_imu, info):
    """Add ZUPT (vel=0) and ZARU (gyro bias = gyro_mean) factors when stationary."""
    stats = _zupt_should_fire(tc, n_imu, info, int(imu_idx_prev))
    if stats is None:
        return False
    graph.add(gtsam.PriorFactorVector(
        tc.Vel(key_idx),
        np.zeros(3, dtype=np.float64),
        gtsam.noiseModel.Isotropic.Sigma(3, _ZUPT_SIGMA_VEL)))
    info['zupt'] = True

    sigmas_bias = np.array([np.inf, np.inf, np.inf,
                            _ZARU_SIGMA_GYRO_BIAS,
                            _ZARU_SIGMA_GYRO_BIAS,
                            _ZARU_SIGMA_GYRO_BIAS])
    graph.add(gtsam.PriorFactorConstantBias(
        tc.Bias(key_idx),
        gtsam.imuBias.ConstantBias(np.zeros(3), stats['gyro_mean']),
        gtsam.noiseModel.Diagonal.Sigmas(sigmas_bias)))
    info['zaru'] = True
    return True
