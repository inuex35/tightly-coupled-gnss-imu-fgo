"""IMU preintegration helpers (pure — no ImuGnssTc self-refs).

Sensor convention is fixed at FLU: the loader hands back IMU samples
already expressed in the body FLU frame.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, List

import numpy as np
import gtsam


# 1. Sensor datasheet specs (Tamagawa TAG300N1 / ADIS16505-2)
_SPEC_ACCEL_NOISE = 2.84e-4      # VRW 0.017 m/s/√hr  -> [m/s²/√Hz]
_SPEC_GYRO_NOISE = 4.01e-5       # ARW 0.138 °/√hr    -> [rad/s/√Hz]
_SPEC_ACCEL_BIAS_RW = 3.14e-4    # In-run bias stability 32 µg
_SPEC_GYRO_BIAS_RW = 9.70e-6     # In-run bias stability 2.0 °/hr

# 2. Measured vehicle chassis vibration PSD (idling / urban driving)
_VIB_ACCEL_NOISE = 2.83e-3       # [m/s²/√Hz]
_VIB_GYRO_NOISE = 4.01e-3        # [rad/s/√Hz]

# 3. Combined effective white noise PSD: σ_eff = √(σ_spec² + σ_vib²)
IMU_ACCEL_NOISE = math.hypot(_SPEC_ACCEL_NOISE, _VIB_ACCEL_NOISE)
IMU_GYRO_NOISE = math.hypot(_SPEC_GYRO_NOISE, _VIB_GYRO_NOISE)
IMU_ACCEL_BIAS_SIGMA = _SPEC_ACCEL_BIAS_RW
IMU_GYRO_BIAS_SIGMA = _SPEC_GYRO_BIAS_RW


def sensor_to_body_flu(vec_sensor: np.ndarray) -> np.ndarray:
    """Pass-through retained as an explicit boundary marker between the
    raw IMU sample and the body FLU frame used downstream."""
    return np.asarray(vec_sensor, dtype=np.float64)


def make_imu_params(
    accel_noise: float = IMU_ACCEL_NOISE,
    gyro_noise: float = IMU_GYRO_NOISE,
    accel_bias_sigma: float = IMU_ACCEL_BIAS_SIGMA,
    gyro_bias_sigma: float = IMU_GYRO_BIAS_SIGMA,
    integ_cov: float = 1e-3) -> gtsam.PreintegrationCombinedParams:
    """Build PreintegrationCombinedParams for GTSAM CombinedImuFactor."""
    # The TC state uses ENU navigation + FLU body. IMU samples are already
    # in body FLU (see sensor_to_body_flu) so body_P_sensor is identity.
    p = gtsam.PreintegrationCombinedParams.MakeSharedU(9.81)
    p.setBodyPSensor(gtsam.Pose3(gtsam.Rot3(np.eye(3)),
                                 gtsam.Point3(0.0, 0.0, 0.0)))
    p.setAccelerometerCovariance(accel_noise ** 2 * np.eye(3))
    p.setGyroscopeCovariance(gyro_noise ** 2 * np.eye(3))
    p.setIntegrationCovariance(integ_cov * np.eye(3))
    p.setBiasAccCovariance(accel_bias_sigma ** 2 * np.eye(3))
    p.setBiasOmegaCovariance(gyro_bias_sigma ** 2 * np.eye(3))
    return p


def build_pim(
    params: gtsam.PreintegrationCombinedParams,
    bias: gtsam.imuBias.ConstantBias,
    imu_data: List[dict],
    imu_idx: int,
    target_tow: Optional[float] = None,
    max_samples: int = 100,
    dt: float = 0.01) -> Tuple[gtsam.PreintegratedCombinedMeasurements, int, np.ndarray, int]:
    """Integrate IMU samples up to `target_tow` or `max_samples`.

    Returns (pim, n_integrated, gyro_mean_body_flu, new_imu_idx).
    """
    pim = gtsam.PreintegratedCombinedMeasurements(params, bias)
    n = 0
    gyro_sum = np.zeros(3)
    idx = imu_idx
    while idx < len(imu_data):
        im = imu_data[idx]
        if target_tow is not None:
            if im['tow'] > target_tow + 1e-6:
                break
        elif n >= max_samples:
            break
        gyro = sensor_to_body_flu(im['gyro'])
        # Real per-sample timestep from the IMU clock; rounded to 0.1 ms
        # so a perfectly-uniform log integrates with exactly the nominal
        # dt (float Δ of large tow values is 0.01±1e-11), while dropped
        # samples show up as 0.02, 0.03, ... instead of silently
        # compressing time. First sample (no predecessor): nominal dt.
        dt_i = dt
        if idx > 0:
            d = round(im['tow'] - imu_data[idx - 1]['tow'], 4)
            if 0.0 < d <= 10 * dt:
                dt_i = d
        pim.integrateMeasurement(im['acc'], gyro, dt_i)
        gyro_sum += gyro
        n += 1
        idx += 1
    gyro_mean = gyro_sum / n if n > 0 else np.zeros(3)
    return pim, n, gyro_mean, idx


def estimate_stationary_bias(
    imu_data: List[dict],
    n_max: int,
    n_min: int = 100,
    g_expected: float = 9.81):
    """Coarse stationary-IMU bias estimation over the first `n_max` samples.

    Model: measured specific force has magnitude ~g when stationary, and the
           residual from that magnitude is treated as coarse accel bias.
           gyro ≈ bias_gyro (stationary, body FLU frame)
    Falls back to zeros when fewer than `n_min` samples are available.

    Returns (bias_acc [3], bias_gyro [3]).
    """
    if n_max <= n_min:
        return np.zeros(3), np.zeros(3)
    acc = np.array([im['acc'] for im in imu_data[:n_max]])
    gyro = np.array([sensor_to_body_flu(im['gyro']) for im in imu_data[:n_max]])
    acc_mean = np.mean(acc, axis=0)
    gyro_mean = np.mean(gyro, axis=0)
    g_norm = np.linalg.norm(acc_mean)
    if g_norm > 1.0:
        bias_acc = acc_mean - g_expected * acc_mean / g_norm
    else:
        bias_acc = np.zeros(3)
    return bias_acc, gyro_mean


def compute_zupt_stats(
    imu_samples: List[dict],
    bias_acc_ref: Optional[np.ndarray] = None,
    bias_gyro_ref: Optional[np.ndarray] = None) -> Optional[dict]:
    """Stationarity stats over a window of IMU samples.

    ``acc_std`` / ``gyro_std`` are deviations from the window mean and
    are bias-immune by construction. ``gyro_median`` is the median of
    ``|gyro - bias_gyro_ref|`` so a steady bias is removed but a
    constant-rate rotation is **not** (the deviation form would also
    cancel constant rotation, which is the opposite of what we want).
    Pass the **stationary-init / Phase-1** bias as reference, not the
    smoother's running estimate (which absorbs lever-arm / scale /
    observation-fit errors and drifts during outages).

    Returns ``None`` when the sample list is empty. Otherwise a dict
    with ``n``, ``acc_std``, ``gyro_std``, ``gyro_median``,
    ``acc_norm_mean``.
    """
    n = len(imu_samples)
    if n == 0:
        return None
    acc = np.array([im['acc'] for im in imu_samples], dtype=np.float64)
    gyro = np.array([sensor_to_body_flu(im['gyro']) for im in imu_samples],
                    dtype=np.float64)
    acc_mean = acc.mean(axis=0)
    acc_dev = acc - acc_mean
    gyro_dev = gyro - gyro.mean(axis=0)
    acc_var = float((acc_dev * acc_dev).sum(axis=1).mean())
    gyro_var = float((gyro_dev * gyro_dev).sum(axis=1).mean())
    if bias_acc_ref is not None:
        acc_mean_corr = acc_mean - np.asarray(bias_acc_ref, dtype=np.float64)
    else:
        acc_mean_corr = acc_mean
    if bias_gyro_ref is not None:
        gyro_resid = gyro - np.asarray(bias_gyro_ref, dtype=np.float64)
    else:
        gyro_resid = gyro
    gyro_resid_norm = np.linalg.norm(gyro_resid, axis=1)
    return {
        'n': int(n),
        'acc_std': float(np.sqrt(max(acc_var, 0.0))),
        'gyro_std': float(np.sqrt(max(gyro_var, 0.0))),
        'gyro_median': float(np.median(gyro_resid_norm)),
        'acc_norm_mean': float(np.linalg.norm(acc_mean_corr)),
        'gyro_mean': gyro.mean(axis=0),
    }
