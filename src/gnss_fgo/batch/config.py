"""BatchConfig — tunables of the batch (whole-run) pipeline, env-overridable."""
import os
from dataclasses import dataclass, field

import numpy as np

from ..utils import env_f, env_i


def _env_vec(name, default):
    """Comma-separated float env var; one value is broadcast to 3 axes."""
    v = np.array([float(x) for x in os.environ.get(name, default).split(",")])
    return np.repeat(v, 3) if v.size == 1 else v


@dataclass
class BatchConfig:
    """All tunables of gnss_fgo.batch. Loaded from env vars at construction."""

    # --- solver
    solver: str = field(default_factory=lambda: os.environ.get("BATCH_SOLVER", "auto"))
    #   auto: gtsam.cuda (cuDSS) when the gtsam build has it, else CPU LM
    max_iters: int = field(default_factory=lambda: env_i("BATCH_MAX_ITERS", 200))

    # --- GNSS measurement model (sigmas at zenith, scaled by 1/sin(el))
    sig_pr: float = field(default_factory=lambda: env_f("BATCH_SIG_PR", 0.5))     # [m]
    sig_cp: float = field(default_factory=lambda: env_f("BATCH_SIG_CP", 0.01))    # [m]
    huber: float = field(default_factory=lambda: env_f("BATCH_HUBER", 1.345))
    sig_amb: float = 30.0          # [cyc] SD ambiguity gauge prior
    sig_rw: float = 5.0            # [m] GNSS-only position random walk per 0.2 s
    # SD Doppler at zenith; residuals at the reference trajectory are
    # ~0.06-0.3 m/s, and 0.5 m/s let 20-30 s base outages drift by 35 m.
    dop_sigma0: float = field(default_factory=lambda: env_f("DOP_SIGMA0", 0.1))   # [m/s]
    huber_dop: float = 1.0
    tropo: int = field(default_factory=lambda: env_i("TROPO", 0))   # Saastamoinen DD tropo

    # --- IMU (ADIS16505-2 'tactical' preset of the sequential pipeline)
    lever: np.ndarray = field(default_factory=lambda: _env_vec("LEVER_ARM", "0.31,0,0.55"))
    acc_noise: np.ndarray = field(default_factory=lambda: _env_vec("IMU_ACC_NOISE", "2.84e-3"))
    gyro_noise: np.ndarray = field(default_factory=lambda: _env_vec("IMU_GYRO_NOISE", "4.01e-3"))
    acc_bias_rw: float = field(default_factory=lambda: env_f("IMU_ACC_BIAS_RW", 3.14e-4))
    gyro_bias_rw: float = field(default_factory=lambda: env_f("IMU_GYRO_BIAS_RW", 9.70e-6))
    integ_cov: float = 1e-3
    gravity: float = field(default_factory=lambda: env_f("BATCH_GRAVITY", 9.81))
    nhc_sigmas: tuple = (1e3, 0.3, 0.2)   # forward, lateral, vertical [m/s]
    nhc_calib: int = field(default_factory=lambda: env_i("NHC_CALIB", 0))   # mount angle state

    # --- ZUPT: IMU stillness test AND speed gate (IMU-only fires at speed)
    zupt: int = field(default_factory=lambda: env_i("ZUPT", 1))
    zupt_speed: float = field(default_factory=lambda: env_f("ZUPT_SPEED", 0.2))   # [m/s]
    zupt_speed_src: str = field(
        default_factory=lambda: os.environ.get("ZUPT_SPEED_SRC", "doppler"))   # doppler | gnss
    zupt_vel_sigma: float = 0.1     # [m/s]
    zupt_rot_sigma: float = 0.01    # [rad]
    zupt_pos_sigma: float = 0.05    # [m]

    # --- post-fit FDE on the float solution
    fde_iters: int = field(default_factory=lambda: env_i("FDE_ITERS", 2))
    fde_pr: float = field(default_factory=lambda: env_f("FDE_PR", 4.0))      # [m]
    fde_cp: float = field(default_factory=lambda: env_f("FDE_CP", 0.5))      # [m]
    fde_dop: float = field(default_factory=lambda: env_f("FDE_DOP", 2.0))    # [m/s]
    fde_max_frac: float = 0.5        # at most this share of an epoch's rows per pass

    # --- ambiguity resolution
    ratio: float = field(default_factory=lambda: env_f("AR_RATIO", 3.0))
    sig_fix: float = 1e-3            # [cyc] integer DD constraint
    lambda_backend: str = field(
        default_factory=lambda: os.environ.get("AR_LAMBDA", "auto"))   # auto | c | python

    # --- per-epoch FIX validation of the fixed solution
    fix_res: float = field(default_factory=lambda: env_f("FIX_RES", 0.05))          # [m]
    fix_min_pairs: int = field(default_factory=lambda: env_i("FIX_MIN_PAIRS", 4))
    # Distinct resolved target satellites: L1+L2 of one satellite pin the
    # same direction, so pair counting lets 3-satellite epochs through.
    fix_min_sats: int = field(default_factory=lambda: env_i("FIX_MIN_SATS", 4))
