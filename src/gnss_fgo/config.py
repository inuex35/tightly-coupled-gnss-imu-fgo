"""TcConfig — 12 grounded physical & statistical parameters for ImuGnssTc."""

import os
from dataclasses import dataclass

from .utils import env_f, env_i


@dataclass
class TcConfig:
    """Minimal physical & statistical parameters for ImuGnssTc (env-overridable)."""

    # 1. GNSS observation model (6)
    err_a: float = 0.001           # carrier-phase σ scale [m] (σ² = a²(1 + 1/sin²el))
    err_eratio_pr: float = 100.0   # pseudorange / carrier-phase σ ratio
    elmin_deg: float = 15.0        # satellite elevation mask [deg]
    cnr_min_dbhz: float = 25.0     # minimum C/N0 [dB-Hz]
    huber_k: float = 1.345         # Huber M-estimator scale [σ] (0 = L2; 1.345 = 95% eff.)
    doppler_sd_sigma: float = 0.5  # SD Doppler zenith σ [m/s] (0 = off)

    # 2. Non-holonomic vehicle constraint (3)
    nhc_sigma_lat: float = 0.3     # body lateral velocity σ [m/s] (0 = off)
    nhc_sigma_vert: float = 0.2    # body vertical velocity σ [m/s] (0 = off)
    nhc_lever: str = '0,0,0'       # IMU -> rear-axle lever arm [m]

    # 3. Cycle slip, ambiguity random walk, AR & sliding window (4)
    thres_slip: float = 0.15       # geometry-free cycle slip threshold [m]
    amb_rw_sigma: float = 0.001    # carrier-phase ambiguity random walk σ [cyc/epoch]
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
            elif isinstance(default, int) and not isinstance(default, bool):
                kw[name] = env_i(envname, default)
            else:
                kw[name] = env_f(envname, default)
        return cls(**kw)
