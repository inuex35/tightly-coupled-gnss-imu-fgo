"""Regression tests for course-based batch trajectory initialization."""
import gtsam
import numpy as np
import pytest

from gnss_fgo.batch.tc_graph import initial_trajectory


@pytest.mark.parametrize("speed", [0.0, 0.5, 1.0])
def test_initial_trajectory_rejects_unavailable_heading(speed):
    tow = np.arange(10, dtype=float)
    ant = np.column_stack([speed * tow, np.zeros((10, 2))])
    with pytest.raises(ValueError, match="Cannot initialize heading"):
        initial_trajectory(ant, tow, np.eye(3), np.zeros(3), np.zeros(3))


@pytest.mark.parametrize("stationary_prefix", [False, True])
@pytest.mark.parametrize("lever", [np.zeros(3), np.array([0.31, 0.0, 0.55])])
def test_initial_trajectory_motion_produces_finite_poses(stationary_prefix, lever):
    tow = np.arange(15, dtype=float)
    distance = 2.0 * (np.maximum(tow - 6.0, 0.0) if stationary_prefix else tow)
    # Northward motion exercises a nonzero yaw, including prefix backfill.
    ant = np.column_stack([np.zeros(15), distance, np.zeros(15)])
    poses, vel, speed = initial_trajectory(
        ant, tow, np.eye(3), np.zeros(3), lever
    )
    assert len(poses) == len(tow)
    assert np.all(np.isfinite(vel))
    assert np.all(np.isfinite(speed))
    for k, pose in enumerate(poses):
        assert np.all(np.isfinite(pose.matrix()))
        np.testing.assert_allclose(pose.rotation().yaw(), np.pi / 2)
        np.testing.assert_allclose(pose.transformFrom(lever), ant[k], atol=1e-12)
    values = gtsam.Values()
    values.insert(0, poses[0])
    prior = gtsam.PriorFactorPose3(0, poses[0], gtsam.noiseModel.Unit.Create(6))
    assert prior.error(values) == pytest.approx(0.0)
