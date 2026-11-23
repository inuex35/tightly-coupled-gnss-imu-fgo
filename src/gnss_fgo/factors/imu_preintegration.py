"""IMU preintegration and CombinedImuFactor wiring (no redundant bias priors)."""

import gtsam

from ..utils import build_pim as _utils_build_pim


def build_pim(tc, bias, target_tow=None):
    """Advance tc.imu_idx and return (pim, n_imu, gyro_mean)."""
    pim, n, gyro_mean, tc.imu_idx = build_pim_from_idx(
        tc, bias, tc.imu_idx, target_tow=target_tow)
    return pim, n, gyro_mean


def build_pim_from_idx(tc, bias, imu_idx, target_tow=None):
    """Build a PIM from an explicit IMU cursor without mutating tc state."""
    return _utils_build_pim(
        tc.imu_params, bias, tc.imu_data, imu_idx,
        target_tow=target_tow)


def add_imu_chain(tc, graph, values, key_idx, pim, pose_p, vel_prev, info):
    """Attach a single CombinedImuFactor between (key_idx - 1) and key_idx."""
    graph.add(gtsam.CombinedImuFactor(
        tc.Xpose(key_idx - 1), tc.Vel(key_idx - 1),
        tc.Xpose(key_idx), tc.Vel(key_idx),
        tc.Bias(key_idx - 1), tc.Bias(key_idx), pim))
