"""Geometry & coordinate utilities (pure functions)."""

from __future__ import annotations

import csv
import os
import numpy as np
from cssrlib.gnss import sat2prn, ecef2pos


def env_f(name: str, default) -> float:
    return float(os.environ.get(name, str(default)))


def env_i(name: str, default) -> int:
    return int(os.environ.get(name, str(default)))


R_ENU2NED = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
R_NED2ENU = R_ENU2NED.T
R_FRD2FLU = np.diag([1.0, -1.0, -1.0])
R_FLU2FRD = R_FRD2FLU.T


def _euler_frd_to_R_body2ned_ref(
    roll: float, pitch: float, heading: float) -> np.ndarray:
    """Reference NED/FRD Euler angles (rad) -> FRD body-to-NED matrix.

    The input convention is:
    - navigation frame: NED
    - body frame: FRD
    - heading: 0=north, 90=east
    """
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    ch, sh = np.cos(heading), np.sin(heading)
    R_ned2body = np.array([
        [cp * ch, cp * sh, -sp],
        [sr * sp * ch - cr * sh, sr * sp * sh + cr * ch, sr * cp],
        [cr * sp * ch + sr * sh, cr * sp * sh - sr * ch, cr * cp]])
    return R_ned2body.T


def euler_to_R_body2ned(roll: float, pitch: float, heading: float) -> np.ndarray:
    """Reference NED/FRD Euler angles -> internal FLU body-to-NED matrix.

    Internally we use a FLU body frame, but the dataset/reference attitude
    convention is NED/FRD. This function is therefore the single bridge from
    the reference convention into the internal body frame.
    """
    return _euler_frd_to_R_body2ned_ref(roll, pitch, heading) @ R_FLU2FRD


def euler_to_R_body2enu(roll: float, pitch: float, heading: float) -> np.ndarray:
    """Reference NED/FRD Euler angles -> internal FLU body-to-ENU matrix."""
    return R_NED2ENU @ euler_to_R_body2ned(roll, pitch, heading)


def parse_lever(s: str) -> np.ndarray:
    """Parse 'x,y,z' CSV string → np.array."""
    return np.array([float(x) for x in s.split(',')])


def is_bds_geo(prn_sat: int) -> bool:
    """True for BeiDou GEO slots (C1-C5, C59-C63) — excluded as DD ref."""
    prn = sat2prn(prn_sat)[1]
    return prn <= 5 or 59 <= prn <= 63


def load_imu_csv(path: str) -> list:
    """Load raw IMU CSV measurements in the dataset's sensor FRD frame."""
    data = []
    flip_x = flip_y = flip_z = False
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            acc = np.array([float(row[2]), float(row[3]), float(row[4])])
            if flip_x:
                acc[0] *= -1.0
            if flip_y:
                acc[1] *= -1.0
            if flip_z:
                acc[2] *= -1.0
            data.append({
                'tow': float(row[0]),
                'acc': acc,
                'gyro': np.array([float(row[5]), float(row[6]), float(row[7])]) * np.pi / 180,
            })
    return data


def heading_from_vel(vel: np.ndarray, fallback: float,
                     disp_enu: np.ndarray | None = None,
                     vel_speed_min: float = 0.5,
                     disp_min: float = 0.01) -> float:
    """Heading [rad] from horizontal velocity (ENU X=E, Y=N).

    Falls back to displacement-based heading when velocity is too slow,
    then to `fallback` when displacement is also too small.
    """
    if np.linalg.norm(vel[:2]) > vel_speed_min:
        return float(np.arctan2(vel[0], vel[1]))
    if disp_enu is not None and np.linalg.norm(disp_enu[:2]) > disp_min:
        return float(np.arctan2(disp_enu[0], disp_enu[1]))
    return float(fallback)


def compute_gdop(pred_pose_trans_enu: np.ndarray, ns: int, rs: np.ndarray,
                 iu, R_enu2ecef: np.ndarray, base_ecef: np.ndarray) -> float:
    """GDOP at a predicted pose. Guards against NaN / absurd positions.

    pred_pose_trans_enu : 3-vec, pose.translation() in ENU (base-relative)
    rs : sat_positions ECEF array [N, >=3]
    iu : indices of observed sats into `rs`
    R_enu2ecef : 3x3 rotation (nav→ECEF)
    base_ecef : 3-vec base station ECEF
    """
    from cssrlib.gnss import dops, satazel
    if ns < 4:
        return 999.0
    pp_gate = R_enu2ecef @ np.asarray(pred_pose_trans_enu) + base_ecef
    if np.linalg.norm(pp_gate) > 1e7 or not np.all(np.isfinite(pp_gate)):
        return 999.0
    az_all = np.zeros(ns)
    el_all = np.zeros(ns)
    pos_geo = ecef2pos(pp_gate)
    for i in range(ns):
        diff = rs[iu[i], :3] - pp_gate
        norm = np.linalg.norm(diff)
        if norm < 1.0:
            return 999.0
        e_ij = diff / norm
        az_all[i], el_all[i] = satazel(pos_geo, e_ij)
    d = dops(az_all, el_all)
    return d[0] if d is not None else 999.0
