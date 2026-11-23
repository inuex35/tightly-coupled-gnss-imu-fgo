"""TcConfig — 25 grounded physical & statistical parameters for ImuGnssTc."""

import os
from dataclasses import dataclass

from .utils import env_f, env_i


IMU_PRESETS = {
    'tactical':   dict(accel_noise=2.84e-3, gyro_noise=4.01e-3,
                       accel_bias_sigma=3.14e-4, gyro_bias_sigma=9.70e-6),
    'consumer':   dict(accel_noise=2.0e-3,  gyro_noise=4.0e-4,
                       accel_bias_sigma=1.0e-3,  gyro_bias_sigma=5.0e-5),
    'industrial': dict(accel_noise=6.0e-4,  gyro_noise=1.0e-4,
                       accel_bias_sigma=5.0e-4,  gyro_bias_sigma=2.0e-5),
    'nav_grade':  dict(accel_noise=5.7e-5,  gyro_noise=3.5e-6,
                       accel_bias_sigma=2.0e-5,  gyro_bias_sigma=5.0e-7),
}


@dataclass
class TcConfig:
    """Minimal physical & statistical parameters for ImuGnssTc (env-overridable)."""

    # 1. GNSS observation model (7)
    err_a: float = 0.001           # base carrier-phase σ [m]
    err_b: float = 0.001           # elevation carrier-phase σ [m] (b/sin(el))
    err_eratio_pr: float = 100.0   # pseudorange / carrier-phase σ ratio
    elmin_deg: float = 15.0        # satellite elevation mask [deg]
    cnr_min_dbhz: float = 25.0     # minimum C/N0 [dB-Hz]
    huber_k: float = 1.345         # Huber M-estimator scale [σ] (0 = L2; 1.345 = 95% eff.)
    doppler_sd_sigma: float = 0.5  # SD Doppler zenith σ [m/s] (0 = off)

    # 2. IMU physical noise model (5)
    imu_grade: str = 'tactical'    # 'tactical'|'consumer'|'industrial'|'nav_grade'
    accel_noise: float = 2.84e-3   # [m/s²/√Hz]
    gyro_noise: float = 4.01e-3    # [rad/s/√Hz]
    accel_bias_sigma: float = 3.14e-4
    gyro_bias_sigma: float = 9.70e-6

    # 3. Non-holonomic vehicle constraint (3)
    nhc_sigma_lat: float = 0.3     # body lateral velocity σ [m/s] (0 = off)
    nhc_sigma_vert: float = 0.2    # body vertical velocity σ [m/s] (0 = off)
    nhc_lever: str = '0,0,0'       # IMU -> rear-axle lever arm [m]

    # 4. ZUPT / ZARU stationary constraints (5)
    zupt_max_acc_std: float = 0.55          # [m/s²]
    zupt_max_gyro_std: float = 0.030        # [rad/s]
    zupt_max_gyro_median: float = 0.020     # [rad/s]
    zupt_sigma_zero_velocity: float = 0.5   # [m/s] (0 = off)
    zupt_sigma_zero_rotation: float = 0.010 # [rad] (0 = off)

    # 5. Cycle slip, AR & sliding window (5)
    thres_slip: float = 0.15       # geometry-free cycle slip threshold [m]
    ar_mode: int = 3               # 0=off, 1=continuous, 3=C++ DD BetweenFactor hold
    ar_ratio: float = 3.0          # LAMBDA / FFRT upper ratio threshold
    ar_gdop_max: float = 8.0       # skip AR above this GDOP (0 = off)
    fls_lag: float = 1.0           # FixedLagSmoother window [s]

    def __post_init__(self):
        """Apply IMU grade preset if user hasn't explicitly set individual σ."""
        cls_fields = type(self).__dataclass_fields__
        is_default = all(
            getattr(self, k) == cls_fields[k].default
            for k in ('accel_noise', 'gyro_noise',
                      'accel_bias_sigma', 'gyro_bias_sigma'))
        if is_default:
            preset = IMU_PRESETS.get(self.imu_grade)
            if preset is None:
                raise ValueError(f"Unknown imu_grade: {self.imu_grade}. "
                                 f"Choose from {list(IMU_PRESETS)}")
            for k, v in preset.items():
                setattr(self, k, v)

    @classmethod
    def from_env(cls):
        """Build a TcConfig with every field overridable via its UPPER_CASE env var."""
        kw = {}
        for name, f in cls.__dataclass_fields__.items():
            if name == 'imu_grade':
                continue
            envname = name.upper()
            default = f.default
            if isinstance(default, str):
                kw[name] = os.environ.get(envname, default)
            elif isinstance(default, int) and not isinstance(default, bool):
                kw[name] = env_i(envname, default)
            else:
                kw[name] = env_f(envname, default)
        kw['imu_grade'] = os.environ.get('IMU_GRADE', cls.imu_grade)
        return cls(**kw)
