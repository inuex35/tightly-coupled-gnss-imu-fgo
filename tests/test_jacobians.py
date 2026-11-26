"""Every hand-written factor gets a numeric Jacobian check.

Add one test per new factor — the checker in factor_check handles any
variable layout. Tolerances: 2e-4 absorbs the first-order Sagnac term
the analytic GNSS Jacobians deliberately keep.
"""
import numpy as np
import gtsam

from cssrlib.gnss import ecef2pos
from factor_check import check_factor_jacobians
from gnss_fgo._extra_factors import NhcFactor

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
    noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.3, 0.1]))
    f = NhcFactor(1, 2, np.array([0.02, -0.01, 0.3]),
                  np.array([0.5, 0.1, -0.3]), noise)
    v = gtsam.Values()
    v.insert(1, gtsam.Pose3(gtsam.Rot3.RzRyRx(0.3, -0.2, 1.1),
                            gtsam.Point3(1, 2, 3)))
    v.insert(2, np.array([8.0, 0.5, -0.3]))
    check_factor_jacobians(f, v, atol=1e-5)


def test_cpp_ddcp_factor():
    sat_ref = gtsam.Point3(-15200000.0, 12000000.0, 18300000.0)
    sat_j = gtsam.Point3(4200000.0, 21000000.0, 15500000.0)
    sat_ref_b = gtsam.Point3(-15199970.0, 12000030.0, 18300030.0)
    sat_j_b = gtsam.Point3(4199975.0, 20999975.0, 15499975.0)
    noise = gtsam.noiseModel.Isotropic.Sigma(1, 0.01)
    f = gtsam.DoubleDifferenceCarrierPhaseFactorArm(
        1, 2, 3,
        21000000.1, 21000100.2, 22000000.3, 22000050.4,
        sat_ref, sat_j, sat_ref_b, sat_j_b,
        gtsam.Point3(*BASE), 0.1903,
        gtsam.Point3(0.31, 0.0, 0.55), enu_frame(BASE), noise)
    v = gtsam.Values()
    v.insert(1, gtsam.Pose3(gtsam.Rot3.RzRyRx(0.05, -0.12, 2.2),
                            gtsam.Point3(310.0, -220.0, 4.0)))
    v.insert(2, 100.0)
    v.insert(3, 88.0)
    check_factor_jacobians(f, v, atol=2e-4)
