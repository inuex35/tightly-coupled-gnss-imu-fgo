"""Non-Holonomic Constraint factor (C++ gtsam.NhcFactor)."""

import numpy as np
import gtsam

from ..utils import parse_lever


def add_nhc_factor(tc, graph, key_idx, gyro_mean_rh=None):
    """Non-Holonomic Constraint at rear-axle pivot in the FLU body frame."""
    sig_lat = float(tc.cfg.nhc_sigma_lat)
    sig_vert = float(tc.cfg.nhc_sigma_vert)
    if sig_lat <= 0 or sig_vert <= 0:
        return False
    lever = tc._nhc_lever_cache
    if lever is None or tc._nhc_lever_src != tc.cfg.nhc_lever:
        lever = parse_lever(tc.cfg.nhc_lever)
        tc._nhc_lever_cache = lever
        tc._nhc_lever_src = tc.cfg.nhc_lever
    if gyro_mean_rh is not None:
        bias_gyro = (tc.tc_bias.gyroscope() if tc.tc_bias is not None
                     else np.zeros(3))
        omega = np.asarray(gyro_mean_rh - bias_gyro, dtype=float)
    else:
        omega = np.zeros(3)
    noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([sig_lat, sig_vert]))
    graph.add(gtsam.NhcFactor(
        tc.Xpose(key_idx), tc.Vel(key_idx), omega,
        np.asarray(lever, dtype=float), noise))
    return True
