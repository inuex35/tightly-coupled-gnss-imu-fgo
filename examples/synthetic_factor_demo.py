#!/usr/bin/env python3
"""Dataset-free DD carrier-phase factor smoke test, not an RTK benchmark.

Geometry and factor construction follow tests/test_jacobians.py. A pose
prior removes the single-observation position/ambiguity degeneracy; only
one float ambiguity is recovered. No IMU, LAMBDA or fix-and-hold is run.
"""
from pathlib import Path
import sys

import gtsam
import numpy as np
from cssrlib.gnss import ecef2pos, geodist

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from gnss_fgo.factors.factors import _make_ddcp_factor_with_held_n

BASE = np.array([-3961905.0, 3348994.0, 3698212.0])


def enu_frame(base):
    lat, lon, _ = ecef2pos(base)
    sl, cl = np.sin(lat), np.cos(lat)
    sn, cn = np.sin(lon), np.cos(lon)
    rotation = np.array([[-sn, -sl * cn, cl * cn],
                         [cn, -sl * sn, cl * sn],
                         [0, cl, sl]])
    return gtsam.Pose3(gtsam.Rot3(rotation), base)


def build_problem():
    """Return graph, DD factor, truth and a deliberately wrong initial state."""
    pose_key, ambiguity_key = 1, 2
    frame = enu_frame(BASE)
    pose = gtsam.Pose3(gtsam.Rot3.RzRyRx(0.05, -0.12, 2.2),
                       np.array([310.0, -220.0, 4.0]))
    lever = np.array([0.31, 0.0, 0.55])
    ref = np.array([-15200000.0, 12000000.0, 18300000.0])
    sat = np.array([4200000.0, 21000000.0, 15500000.0])
    ref_base, sat_base = ref + 30.0, sat - 25.0
    # Same held-reference/float-satellite model as the Jacobian test.
    offset_m, coeff_m = -34387.86, -0.1903
    ambiguity = -180705.3  # cycles; deliberately float, not an integer FIX
    antenna = frame.compose(pose).transformFrom(lever)
    # Independently synthesize a noiseless DD observation in metres,
    # including geodist's Earth-rotation correction on all four ranges.
    observation = (geodist(ref, antenna)[0] - geodist(sat, antenna)[0]
                   - geodist(ref_base, BASE)[0] + geodist(sat_base, BASE)[0]
                   + offset_m + coeff_m * ambiguity)
    factor = _make_ddcp_factor_with_held_n(
        pose_key, ambiguity_key, gtsam.noiseModel.Isotropic.Sigma(1, 0.01),
        ref, sat, ref_base, sat_base, BASE, observation, lever, frame,
        offset_m=offset_m, coeff_m=coeff_m)
    truth = gtsam.Values()
    truth.insert(pose_key, pose)
    truth.insert(ambiguity_key, ambiguity)
    initial = gtsam.Values()
    initial.insert(pose_key, pose)
    initial.insert(ambiguity_key, ambiguity + 10.0)
    graph = gtsam.NonlinearFactorGraph()
    graph.add(factor)
    graph.add(gtsam.PriorFactorPose3(
        pose_key, pose, gtsam.noiseModel.Isotropic.Sigma(6, 1e-3)))
    return graph, factor, truth, initial


def main():
    graph, factor, truth, initial = build_problem()
    result = gtsam.LevenbergMarquardtOptimizer(graph, initial).optimize()
    before = float(factor.unwhitenedError(initial)[0])
    after = float(factor.unwhitenedError(result)[0])
    cycle_error = abs(result.atDouble(2) - truth.atDouble(2))
    print('Synthetic DD carrier-phase factor (no dataset; no RTK FIX)')
    print(f'Initial residual: {before:.6f} m')
    print(f'Optimized residual: {after:.6f} m')
    print(f'Float ambiguity error: {cycle_error:.6f} cycles')
    # Explicit checks remain active even under python -O; CI runs this CLI.
    if not (np.isfinite(after) and abs(after) < 1e-6
            and cycle_error < 1e-5 and graph.error(result) < graph.error(initial)):
        raise RuntimeError('Synthetic factor smoke test failed')
    print('PASS: synthetic residual and float ambiguity checks')


if __name__ == '__main__':
    main()
