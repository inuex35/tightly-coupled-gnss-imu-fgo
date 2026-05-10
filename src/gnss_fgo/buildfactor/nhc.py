"""Non-Holonomic Constraint factor.

Body lateral & vertical velocity ≈ 0 at the rear-axle pivot. When a
non-zero ``cfg.nhc_lever`` is configured the constraint is evaluated
at that lever-arm offset from the IMU origin, with the per-epoch
gyro-driven offset ``ω × lever_xyz`` subtracted so the lateral / vertical
zero is enforced at the rear axle rather than at the IMU.

Extracted from ``factors_support.py`` during the Phase 2 architectural
refactor; behaviour unchanged.
"""

import numpy as np
import gtsam

from ..utils import parse_lever


def add_nhc_factor(tc, g3, kk, speed, gyro_mean_rh=None):
    """Non-Holonomic Constraint at rear-axle center in the FLU body frame."""
    if not tc.cfg.nhc_enable or speed < tc.cfg.nhc_min_speed:
        return False
    lever = parse_lever(tc.cfg.nhc_lever)
    if gyro_mean_rh is not None and np.linalg.norm(lever) > 0:
        bias_gyro = tc.tc_bias.gyroscope() if tc.tc_bias is not None \
            else np.zeros(3)
        omega = gyro_mean_rh - bias_gyro
        offset = np.cross(omega, lever)
    else:
        offset = np.zeros(3)
    noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([float(tc.cfg.nhc_sigma_lat),
                  float(tc.cfg.nhc_sigma_vert)]))

    def error_fn(this, values, jacobians):
        pose = values.atPose3(this.keys()[0])
        v = values.atVector(this.keys()[1])
        R = np.array(pose.rotation().matrix())
        v_body = R.T @ v + offset
        err = np.array([v_body[1], v_body[2]])
        if jacobians is not None:
            skew_v = np.array([[0, -v[2], v[1]],
                                [v[2], 0, -v[0]],
                                [-v[1], v[0], 0]])
            dR = (R.T @ skew_v)[1:3, :]
            jacobians[0] = np.hstack([dR, np.zeros((2, 3))])
            jacobians[1] = R.T[1:3, :]
        return err

    g3.add(gtsam.CustomFactor(
        noise, [tc.Xpose(kk), tc.Vel(kk)], error_fn))
    return True
