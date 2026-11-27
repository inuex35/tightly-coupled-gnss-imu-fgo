"""Shared helpers (geometry/rotations, IMU, RINEX signal autodetect, env parsing)."""

from __future__ import annotations

import csv
import math
import os
from typing import List, Optional, Tuple

import gtsam
import numpy as np
from cssrlib.gnss import ecef2pos, sat2prn, uGNSS, uTYP


# ── Env parsing ───────────────────────────────────────────────────────

def env_float(name: str, default) -> float:
    """Float env var with a default."""
    raw = os.environ.get(name, str(default))
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f'{name}={raw!r} is not a float') from None


def env_int(name: str, default) -> int:
    """Int env var with a default."""
    raw = os.environ.get(name, str(default))
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f'{name}={raw!r} is not an int') from None


def env_vector(name: str, default: str) -> np.ndarray:
    """Comma-separated float env var; one value is broadcast to 3 axes."""
    v = np.array([float(x) for x in os.environ.get(name, default).split(",")])
    return np.repeat(v, 3) if v.size == 1 else v


# ── Geometry & coordinate helpers ─────────────────────────────────────

_R_NED2ENU = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]]).T
_R_FLU2FRD = np.diag([1.0, -1.0, -1.0])


def enu_frame(rb: np.ndarray) -> Tuple[np.ndarray, gtsam.Pose3]:
    """(R_enu2ecef, ecef_T_nav) for the ENU frame at the base station."""
    lat, lon, _ = ecef2pos(rb)
    sl, cl = np.sin(lat), np.cos(lat)
    sn, cn = np.sin(lon), np.cos(lon)
    Rm = np.array([
        [-sn, -sl * cn, cl * cn],
        [cn, -sl * sn, cl * sn],
        [0, cl, sl],
    ])
    return Rm, gtsam.Pose3(gtsam.Rot3(Rm), gtsam.Point3(*rb))


def euler_to_R_body2enu(roll: float, pitch: float, heading: float) -> np.ndarray:
    """Reference NED/FRD Euler angles -> internal FLU body-to-ENU matrix."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    ch, sh = np.cos(heading), np.sin(heading)
    R_ned2body = np.array([
        [cp * ch, cp * sh, -sp],
        [sr * sp * ch - cr * sh, sr * sp * sh + cr * ch, sr * cp],
        [cr * sp * ch + sr * sh, cr * sp * sh - sr * ch, cr * cp],
    ])
    return _R_NED2ENU @ R_ned2body.T @ _R_FLU2FRD


def load_imu_csv(path: str, as_arrays: bool = False):
    """Load raw IMU CSV measurements in the body FLU frame.

    Returns list of {'tow', 'acc', 'gyro'} dicts by default, or
    (tow, acc [m/s^2], gyro [rad/s]) numpy arrays when as_arrays=True.
    """
    if as_arrays:
        m = np.loadtxt(path, delimiter=",", skiprows=1)
        return m[:, 0], m[:, 2:5], np.deg2rad(m[:, 5:8])
    data = []
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            acc = np.array([float(row[2]), float(row[3]), float(row[4])])
            data.append({
                'tow': float(row[0]),
                'acc': acc,
                'gyro': np.array([float(row[5]), float(row[6]), float(row[7])]) * np.pi / 180,
            })
    return data


# ── Ambiguity / DD-signal helpers ─────────────────────────────────────

def get_wavelengths(obs_sd, sat: int, glo_ch: dict | None = None) -> List[float]:
    """Carrier-phase wavelengths [m] for one satellite, indexed by frequency."""
    sys_i, _ = sat2prn(sat)
    if sys_i not in obs_sd.sig:
        return []
    if uTYP.L not in obs_sd.sig[sys_i]:
        return []
    sigs = obs_sd.sig[sys_i][uTYP.L]
    if sys_i == uGNSS.GLO:
        ch_map = glo_ch or {}
        return [s.wavelength(ch_map.get(sat, 0)) for s in sigs]
    return [s.wavelength() for s in sigs]


# ── Pipeline helpers ──────────────────────────────────────────────────

HUBER_K = 1.345  # 95% asymptotic relative efficiency under Gaussian noise
CAUCHY_K = 2.385  # 95% asymptotic relative efficiency under Gaussian noise (Holland & Welsch 1977)


def _robust_noise(sigma: float, k: float, estimator_cls):
    """Build 1-D Gaussian or GTSAM M-estimator robust noise model."""
    base = gtsam.noiseModel.Isotropic.Sigma(1, float(sigma))
    if float(k) <= 0.0:
        return base
    return gtsam.noiseModel.Robust.Create(
        estimator_cls.Create(float(k)), base)


def huber_noise(sigma: float, k: float = HUBER_K):
    """Build 1-D Gaussian or GTSAM Huber robust noise model."""
    return _robust_noise(sigma, k, gtsam.noiseModel.mEstimator.Huber)


def cauchy_noise(sigma: float, k: float = CAUCHY_K):
    """Build 1-D Gaussian or GTSAM Cauchy robust noise model."""
    return _robust_noise(sigma, k, gtsam.noiseModel.mEstimator.Cauchy)


def sorted_amb_items(amb_dict):
    """Deterministic ``(sat, freq) -> key`` iteration sorted by sat then freq."""
    return sorted(amb_dict.items(),
                  key=lambda item: (int(item[0][0]), int(item[0][1])))


# ── IMU preintegration helpers ────────────────────────────────────────

# 1. Sensor datasheet specs (Tamagawa TAG300N1 / ADIS16505-2)
_SPEC_ACCEL_NOISE = 2.84e-4      # VRW 0.017 m/s/√hr  -> [m/s²/√Hz]
_SPEC_GYRO_NOISE = 4.01e-5       # ARW 0.138 °/√hr    -> [rad/s/√Hz]
# In-run bias stability (32 µg = 3.14e-4 m/s²) is the Allan deviation minimum
# [m/s²], whereas GTSAM setBiasAccCovariance expects random-walk PSD [m/s²/√s]
# (σ_rw = σ_stab / √τ). With τ = 1000 s (typical tactical MEMS ADEV trough):
_SPEC_ACCEL_BIAS_STAB = 3.14e-4  # In-run bias stability 32 µg [m/s²]
_SPEC_BIAS_CORR_TIME_S = 1000.0  # Allan deviation correlation time τ [s]
_SPEC_ACCEL_BIAS_RW = _SPEC_ACCEL_BIAS_STAB / math.sqrt(_SPEC_BIAS_CORR_TIME_S)
_SPEC_GYRO_BIAS_RW = 9.70e-6     # 2.0 °/hr [rad/s] (kept as [rad/s/√s] for thermal drift)

# 2. Measured vehicle chassis vibration PSD (idling / urban driving)
_VIB_ACCEL_NOISE = 2.83e-3       # [m/s²/√Hz]
_VIB_GYRO_NOISE = 4.01e-3        # [rad/s/√Hz]

# 3. Combined effective white noise PSD: σ_eff = √(σ_spec² + σ_vib²)
IMU_ACCEL_NOISE = math.hypot(_SPEC_ACCEL_NOISE, _VIB_ACCEL_NOISE)
IMU_GYRO_NOISE = math.hypot(_SPEC_GYRO_NOISE, _VIB_GYRO_NOISE)
IMU_ACCEL_BIAS_SIGMA = _SPEC_ACCEL_BIAS_RW
IMU_GYRO_BIAS_SIGMA = _SPEC_GYRO_BIAS_RW


def make_imu_params(
    accel_noise: float | np.ndarray = IMU_ACCEL_NOISE,
    gyro_noise: float | np.ndarray = IMU_GYRO_NOISE,
    accel_bias_sigma: float = IMU_ACCEL_BIAS_SIGMA,
    gyro_bias_sigma: float = IMU_GYRO_BIAS_SIGMA,
    integ_cov: float = 1e-3,
    gravity: float = 9.81,
) -> gtsam.PreintegrationCombinedParams:
    """Build PreintegrationCombinedParams for GTSAM CombinedImuFactor."""
    p = gtsam.PreintegrationCombinedParams.MakeSharedU(gravity)
    p.setBodyPSensor(gtsam.Pose3(gtsam.Rot3(np.eye(3)),
                                 gtsam.Point3(0.0, 0.0, 0.0)))
    p.setAccelerometerCovariance(np.diag(np.broadcast_to(np.asarray(accel_noise, float), 3) ** 2))
    p.setGyroscopeCovariance(np.diag(np.broadcast_to(np.asarray(gyro_noise, float), 3) ** 2))
    p.setIntegrationCovariance(integ_cov * np.eye(3))
    p.setBiasAccCovariance(accel_bias_sigma ** 2 * np.eye(3))
    p.setBiasOmegaCovariance(gyro_bias_sigma ** 2 * np.eye(3))
    return p


def build_imu_preintegration(
    params: gtsam.PreintegrationCombinedParams,
    bias: gtsam.imuBias.ConstantBias,
    imu_data: List[dict],
    imu_idx: int,
    target_tow: Optional[float] = None,
    max_samples: int = 100,
    dt: float = 0.01,
) -> Tuple[gtsam.PreintegratedCombinedMeasurements, int, np.ndarray, int]:
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
        gyro = np.asarray(im['gyro'], dtype=np.float64)
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
    g_expected: float = 9.81,
):
    """Coarse stationary-IMU bias estimation over the first `n_max` samples."""
    if n_max <= n_min:
        return np.zeros(3), np.zeros(3)
    acc = np.array([im['acc'] for im in imu_data[:n_max]])
    gyro = np.array([im['gyro'] for im in imu_data[:n_max]], dtype=np.float64)
    acc_mean = np.mean(acc, axis=0)
    gyro_mean = np.mean(gyro, axis=0)
    g_norm = np.linalg.norm(acc_mean)
    if g_norm > 1.0:
        bias_acc = acc_mean - g_expected * acc_mean / g_norm
    else:
        bias_acc = np.zeros(3)
    return bias_acc, gyro_mean


# ── RINEX signal auto-detection ───────────────────────────────────────

_BAND_PRIORITY = (1, 2, 5, 7, 6, 8, 3, 4, 9)


def _group_by_band(sigs, typ):
    """Group rSigRnx values from a sig_map system by their frequency band."""
    out = {}
    for s in sigs.values():
        if s.typ != typ:
            continue
        band = int(s.sig) // 100
        out.setdefault(band, s)
    return out


def auto_detect_signals(
    sig_map_rov,
    sig_map_base,
    max_freq=2,
    required=(uTYP.C, uTYP.L, uTYP.S),
    systems=None,
    strict_freq=True,
):
    """Build rover/base signal lists with matching (sys, typ, band) coverage."""
    rov_systems = set(sig_map_rov.keys())
    base_systems = set(sig_map_base.keys())
    if systems is None:
        systems = rov_systems & base_systems
    else:
        systems = set(systems) & rov_systems & base_systems

    sigs, sigsb = [], []
    for sys in sorted(systems):
        rov_by_typ = {t: _group_by_band(sig_map_rov[sys], t) for t in required}
        base_by_typ = {t: _group_by_band(sig_map_base[sys], t) for t in required}
        common_bands = set(rov_by_typ[required[0]].keys())
        for t in required[1:]:
            common_bands &= rov_by_typ[t].keys()
        for t in required:
            common_bands &= base_by_typ[t].keys()
        if not common_bands:
            continue
        if strict_freq and len(common_bands) < max_freq:
            continue
        ordered = [b for b in _BAND_PRIORITY if b in common_bands]
        ordered += sorted(b for b in common_bands if b not in ordered)
        for band in ordered[:max_freq]:
            for t in required:
                sigs.append(rov_by_typ[t][band])
                sigsb.append(base_by_typ[t][band])
    return sigs, sigsb
