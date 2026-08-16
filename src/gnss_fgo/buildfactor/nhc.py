"""Non-Holonomic Constraint factor (C++ gtsam.NhcFactor).

Body lateral & vertical velocity ~ 0 at the rear-axle pivot:
  error = R^T v + gyro x lever - [wheelSpeed, 0, 0]
evaluated by ``gtsam.NhcFactor`` (custom/develop wheel). There is no
wheel odometry in this pipeline, so the forward component is released
with a huge sigma and only lateral/vertical constrain the solution.
When a non-zero ``cfg.nhc_lever`` is configured the constraint acts at
that lever-arm offset from the IMU origin via the gyro term.
"""

import numpy as np
import gtsam

from ..utils import parse_lever


def add_nhc_factor(tc, g3, kk, speed, gyro_mean_rh=None):
    """Non-Holonomic Constraint at rear-axle center in the FLU body frame."""
    if not tc.cfg.nhc_enable or speed < tc.cfg.nhc_min_speed:
        return False
    lever = parse_lever(tc.cfg.nhc_lever)
    if gyro_mean_rh is not None:
        bias_gyro = tc.tc_bias.gyroscope() if tc.tc_bias is not None \
            else np.zeros(3)
        omega = np.asarray(gyro_mean_rh - bias_gyro, dtype=float)
    else:
        omega = np.zeros(3)
    noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([1e3,
                  float(tc.cfg.nhc_sigma_lat),
                  float(tc.cfg.nhc_sigma_vert)]))
    g3.add(gtsam.NhcFactor(
        tc.Xpose(kk), tc.Vel(kk), omega,
        np.asarray(lever, dtype=float),
        float(speed), noise, np.zeros(3)))
    return True
