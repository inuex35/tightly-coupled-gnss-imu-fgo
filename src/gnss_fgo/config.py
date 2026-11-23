"""TcConfig — grounded physical & statistical parameters for ImuGnssTc."""

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
    """Core parameters for ImuGnssTc (loaded from env vars via from_env)."""

    # GNSS DD observation model (RTKLIB elevation model + GTSAM Robust kernel)
    err_a: float = 0.001           # base CP σ [m]
    err_b: float = 0.001           # elevation CP σ [m] (b/sin(el))
    err_eratio_pr: float = 100.0   # PR / CP σ ratio
    err_sclkstab: float = 5e-12    # receiver clock stability [s/s]
    el_mask_deg: float = 10.0      # observation elevation mask [deg]
    elmin_deg: float = 15.0        # AR elevation mask [deg]
    cnr_min_dbhz: float = 25.0     # minimum C/N0 [dB-Hz]
    snr_ref_dbhz: float = 42.0     # reference C/N0 for weighting [dB-Hz]
    dd_robust_kind: str = 'huber'  # 'none' | 'huber' | 'dcs' | 'cauchy'
    ddpr_robust_k: float = 1.345   # DDPR M-estimator scale (0 = L2)
    ddcp_robust_k: float = 1.345   # DDCP M-estimator scale (0 = L2)

    # Between-satellite single-differenced Doppler
    doppler_sd_sigma: float = 0.5  # [m/s] (0 = off)
    doppler_huber: float = 1.0     # [m/s] Huber width (0 = L2)
    doppler_fde_k: float = 4.0     # MAD outlier screen multiple (0 = off)

    # IMU physical noise model (CombinedImuFactor only)
    imu_grade: str = 'tactical'    # 'tactical'|'consumer'|'industrial'|'nav_grade'
    imu_integ_cov: float = 1e-3    # position integration covariance [m²/s]
    accel_noise: float = 2.84e-3   # [m/s²/√Hz]
    gyro_noise: float = 4.01e-3    # [rad/s/√Hz]
    accel_bias_sigma: float = 3.14e-4
    gyro_bias_sigma: float = 9.70e-6

    # Non-holonomic vehicle constraint (0 disables)
    nhc_sigma_lat: float = 0.3     # body lateral velocity σ [m/s]
    nhc_sigma_vert: float = 0.2    # body vertical velocity σ [m/s]
    nhc_lever: str = '0,0,0'       # IMU -> rear-axle lever arm [m]

    # ZUPT / ZARU stationary constraints (σ <= 0 disables each)
    zupt_max_acc_std: float = 0.55          # [m/s²]
    zupt_max_gyro_std: float = 0.030        # [rad/s]
    zupt_max_gyro_median: float = 0.020     # [rad/s]
    zupt_min_samples: int = 5               # samples per epoch
    zupt_sigma_zero_velocity: float = 0.5   # [m/s]
    zupt_sigma_zero_rotation: float = 0.010 # [rad]

    # Cycle slip, persistent ambiguity prior & LAMBDA AR
    thres_slip: float = 0.15       # geometry-free slip threshold [m]
    sigma_amb0: float = 30.0       # fresh arc ambiguity prior σ [cyc]
    ar_mode: int = 1               # 0=off, 1=continuous (no graph hold), 3=DD BetweenFactor hold
    ar_thresar_min: float = 1.5    # FFRT ratio lower clamp
    ar_thresar_max: float = 3.0    # FFRT ratio upper clamp
    ar_minfixsats: int = 4         # minimum satellites for AR
    ar_gdop_max: float = 8.0       # skip AR above this GDOP (0 = off)
    varholdamb: float = 0.001      # DD integer constraint variance [cyc²] (ar_mode=3)

    # Initialization & FixedLagSmoother / iSAM2
    vel_thresh: float = 1.0        # Phase 1 -> 2 motion threshold [m/s]
    n_collect: int = 5             # Phase 1 FIX epochs before Phase 2
    body_rot_std: float = 1.0      # initial heading uncertainty [deg]
    fls_lag: float = 1.0           # FixedLagSmoother window [s]
    isam2_relinearize_skip: int = 10
    isam2_relinearize_threshold: float = 0.05

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

    _ENV_OVERRIDES = {
        'sigma_amb0': 'SIG_AMB',
    }

    @classmethod
    def _env_name(cls, field_name):
        """Env-var name for a field: override table first, else UPPER_CASE."""
        return cls._ENV_OVERRIDES.get(field_name, field_name.upper())

    @classmethod
    def from_env(cls):
        """Build a TcConfig with every field overridable via its env var."""
        kw = {}
        for name, f in cls.__dataclass_fields__.items():
            if name == 'imu_grade':
                continue
            envname = cls._env_name(name)
            default = f.default
            if isinstance(f.default, str):
                kw[name] = os.environ.get(envname, default)
            elif isinstance(f.default, int) and not isinstance(f.default, bool):
                kw[name] = env_i(envname, default)
            else:
                kw[name] = env_f(envname, default)
        kw['imu_grade'] = os.environ.get('IMU_GRADE', cls.imu_grade)
        return cls(**kw)
