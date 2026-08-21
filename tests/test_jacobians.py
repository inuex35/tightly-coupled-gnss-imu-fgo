"""Every hand-written factor gets a numeric Jacobian check.

Add one test per new factor — the checker in factor_check handles any
variable layout. Tolerances: 2e-4 absorbs the first-order Sagnac term
the analytic GNSS Jacobians deliberately keep.
"""
import numpy as np
import gtsam

from cssrlib.gnss import ecef2pos
from factor_check import check_factor_jacobians

BASE = np.array([-3961905.0, 3348994.0, 3698212.0])


def enu_frame(base):
    lat, lon, _ = ecef2pos(base)
    sl, cl = np.sin(lat), np.cos(lat)
    sn, cn = np.sin(lon), np.cos(lon)
    R = np.array([[-sn, -sl * cn, cl * cn],
                  [cn, -sl * sn, cl * sn],
                  [0, cl, sl]])
    return gtsam.Pose3(gtsam.Rot3(R), gtsam.Point3(*base))


def test_nhc_factor():
    noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e3, 0.3, 0.1]))
    f = gtsam.NhcFactor(1, 2, np.array([0.02, -0.01, 0.3]),
                        np.array([0.5, 0.1, -0.3]), 8.0, noise, np.zeros(3))
    v = gtsam.Values()
    v.insert(1, gtsam.Pose3(gtsam.Rot3.RzRyRx(0.3, -0.2, 1.1),
                            gtsam.Point3(1, 2, 3)))
    v.insert(2, np.array([8.0, 0.5, -0.3]))
    check_factor_jacobians(f, v, atol=1e-5)


def test_held_ddcp_factor():
    from gnss_fgo.factors.factors import _make_ddcp_factor_with_held_n
    sat_ref = np.array([-15200000.0, 12000000.0, 18300000.0])
    sat_j = np.array([4200000.0, 21000000.0, 15500000.0])
    noise = gtsam.noiseModel.Isotropic.Sigma(1, 0.01)
    f = _make_ddcp_factor_with_held_n(
        1, 2, noise, sat_ref, sat_j, sat_ref + 30.0, sat_j - 25.0,
        BASE, 123.456, np.array([0.31, 0.0, 0.55]), enu_frame(BASE),
        offset_m=-34387.86, coeff_m=-0.1903)
    v = gtsam.Values()
    v.insert(1, gtsam.Pose3(gtsam.Rot3.RzRyRx(0.05, -0.12, 2.2),
                            gtsam.Point3(310.0, -220.0, 4.0)))
    v.insert(2, -180705.3)
    check_factor_jacobians(f, v, atol=2e-4)
