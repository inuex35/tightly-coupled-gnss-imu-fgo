"""Geometry & coordinate utilities (pure functions)."""

from __future__ import annotations

import csv
import os
import numpy as np
from cssrlib.gnss import sat2prn


def env_f(name: str, default) -> float:
    """Float env var with a default."""
    raw = os.environ.get(name, str(default))
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f'{name}={raw!r} is not a float') from None


def env_i(name: str, default) -> int:
    """Int env var with a default."""
    raw = os.environ.get(name, str(default))
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f'{name}={raw!r} is not an int') from None


_R_NED2ENU = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]]).T
_R_FLU2FRD = np.diag([1.0, -1.0, -1.0])


def euler_to_R_body2enu(roll: float, pitch: float, heading: float) -> np.ndarray:
    """Reference NED/FRD Euler angles -> internal FLU body-to-ENU matrix."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    ch, sh = np.cos(heading), np.sin(heading)
    R_ned2body = np.array([
        [cp * ch, cp * sh, -sp],
        [sr * sp * ch - cr * sh, sr * sp * sh + cr * ch, sr * cp],
        [cr * sp * ch + sr * sh, cr * sp * sh - sr * ch, cr * cp]])
    return _R_NED2ENU @ R_ned2body.T @ _R_FLU2FRD


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
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            acc = np.array([float(row[2]), float(row[3]), float(row[4])])
            data.append({
                'tow': float(row[0]),
                'acc': acc,
                'gyro': np.array([float(row[5]), float(row[6]), float(row[7])]) * np.pi / 180,
            })
    return data
