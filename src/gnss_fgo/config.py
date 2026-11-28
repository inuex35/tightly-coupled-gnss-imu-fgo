"""TcConfig and BatchConfig — physical & statistical parameters (env-overridable)."""

import os
from dataclasses import dataclass, field

import numpy as np

from .utils import (
    CAUCHY_K,
    HUBER_K,
    IMU_ACCEL_BIAS_SIGMA,
    IMU_ACCEL_NOISE,
    IMU_GYRO_BIAS_SIGMA,
    IMU_GYRO_NOISE,
    env_float,
    env_int,
    env_vector,
)


@dataclass
class TcConfig:
    """Minimal physical & statistical parameters for ImuGnssTc (env-overridable)."""

    # 1. GNSS observation model (4)
    err_a: float = 0.001           # carrier-phase σ scale [m] at 45 dB-Hz
    err_eratio_pr: float = 100.0   # pseudorange / carrier-phase σ ratio
    elmin_deg: float = 15.0        # satellite elevation mask [deg]
    cnr_min_dbhz: float = 25.0     # minimum C/N0 [dB-Hz]

    # 2. Non-holonomic vehicle constraint (3)
    nhc_sigma_lat: float = 0.3     # body lateral velocity σ [m/s] (0 = off)
    nhc_sigma_vert: float = 0.2    # body vertical velocity σ [m/s] (0 = off)
    nhc_lever: str = '0,0,0'       # IMU -> rear-axle lever arm [m]
    nhc_calib_rw: float = 1e-5     # mount Rot3 random walk σ [rad/epoch]

    # 3. Cycle slip, ambiguity random walk, AR & sliding window (4)
    thres_slip: float = 0.15       # geometry-free cycle slip threshold [m]
    amb_rw_sigma: float = 0.001    # carrier-phase ambiguity random walk σ [cyc/epoch] (> 0)
    armode: int = 2                # AR mode (0=off, 1=continuous, 2=fix-and-hold)
    fls_lag: float = 1.0           # FixedLagSmoother window [s]

    @classmethod
    def from_env(cls):
        """Build a TcConfig with every field overridable via its UPPER_CASE env var."""
        kw = {}
        for name, f in cls.__dataclass_fields__.items():
            envname = name.upper()
            default = f.default
            if isinstance(default, str):
                kw[name] = os.environ.get(envname, default)
            elif isinstance(default, int):
                kw[name] = env_int(envname, default)
            else:
                kw[name] = env_float(envname, default)
        return cls(**kw)


_TC_DEFAULTS = TcConfig()


def _env_field(envname: str, default, vec: bool = False):
    """Dataclass field whose default is read from ``envname`` at construction."""
    if vec:
        return field(default_factory=lambda: env_vector(envname, default))
    if isinstance(default, int):
        return field(default_factory=lambda: env_int(envname, default))
    if isinstance(default, float):
        return field(default_factory=lambda: env_float(envname, default))
    return field(default_factory=lambda: os.environ.get(envname, default))


@dataclass
class BatchConfig:
    """All tunables of gnss_fgo.batch. Loaded from env vars at construction."""

    # --- solver
    solver: str = _env_field("BATCH_SOLVER", "auto")
    #   auto: gtsam.cuda (cuDSS) when the gtsam build has it, else CPU LM
    max_iters: int = _env_field("BATCH_MAX_ITERS", 200)

    # --- GNSS measurement model (defaults shared with TcConfig)
    sig_pr: float = _env_field("BATCH_SIG_PR", _TC_DEFAULTS.err_a * _TC_DEFAULTS.err_eratio_pr)  # [m]
    sig_cp: float = _env_field("BATCH_SIG_CP", _TC_DEFAULTS.err_a)  # [m]
    huber: float = _env_field("BATCH_HUBER", HUBER_K)
    cauchy: float = _env_field("BATCH_CAUCHY", CAUCHY_K)
    sig_amb: float = 30.0          # [cyc] SD ambiguity gauge prior
    sig_rw: float = 5.0            # [m] GNSS-only position random walk per 0.2 s
    dop_sigma0: float = _env_field("DOP_SIGMA0", float(np.sqrt(2.0) * _TC_DEFAULTS.err_a / 0.02))  # [m/s]
    huber_dop: float = 1.0

    # --- IMU (datasheet + vehicle vibration PSD from utils)
    lever: np.ndarray = _env_field("LEVER_ARM", "0.31,0,0.55", vec=True)
    acc_noise: np.ndarray = _env_field("IMU_ACC_NOISE", f"{IMU_ACCEL_NOISE:.6g}", vec=True)
    gyro_noise: np.ndarray = _env_field("IMU_GYRO_NOISE", f"{IMU_GYRO_NOISE:.6g}", vec=True)
    acc_bias_rw: float = _env_field("IMU_ACC_BIAS_RW", IMU_ACCEL_BIAS_SIGMA)
    gyro_bias_rw: float = _env_field("IMU_GYRO_BIAS_RW", IMU_GYRO_BIAS_SIGMA)
    integ_cov: float = 1e-3
    gravity: float = _env_field("BATCH_GRAVITY", 9.81)
    nhc_sigmas: tuple = (_TC_DEFAULTS.nhc_sigma_lat, _TC_DEFAULTS.nhc_sigma_vert)

    # --- ambiguity resolution
    ratio: float | None = field(
        default_factory=lambda: env_float("AR_RATIO", 0.0) or None)  # None = FFRT
    sig_fix: float = 1e-3            # [cyc] integer DD constraint
    lambda_workers: int = _env_field("AR_LAMBDA_WORKERS", 0)  # 0 = all cores, 1 = serial

    # --- per-epoch FIX validation of the fixed solution
    fix_res: float = _env_field("FIX_RES", 0.05)          # [m]
    fix_min_pairs: int = _env_field("FIX_MIN_PAIRS", 4)
    fix_min_sats: int = _env_field("FIX_MIN_SATS", 4)
