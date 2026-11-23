"""Build the ambiguity-resolution problem directly from the smoother's joint marginals."""

from dataclasses import dataclass, field

import gtsam
import numpy as np

from ..utils import sorted_amb_items


@dataclass
class ArProblem:
    """One epoch's float-ambiguity problem, self-contained."""

    keys: list
    values: dict
    cov: np.ndarray
    cross: np.ndarray
    elevations: dict = field(default_factory=dict)


def build(tc, sat_list, amb_dict, key_pose=None):
    """Assemble ArProblem from live smoother variables, or None if unposed."""
    smoother = tc.isam2
    if smoother is None:
        return None
    isam2 = smoother.getISAM2()
    if isam2 is None:
        return None
    est = smoother.calculateEstimate()

    if key_pose is not None:
        tc._ar_key_pose = key_pose
    key_pose = tc._ar_key_pose
    if key_pose is None or not est.exists(key_pose):
        return None

    key_of = dict(sorted_amb_items(amb_dict))
    present = {int(s) for s in sat_list}
    seeded = tc.current_epoch.seeded_amb_keys
    cp_visible = tc._ar_cp_visible_sf
    keys, values = [], {}
    for (s, f), k in sorted_amb_items(key_of):
        sf = (int(s), int(f))
        if (int(s) in present and est.exists(k)
                and (s, f) not in seeded
                and (not cp_visible or sf in cp_visible)):
            keys.append(sf)
            values[sf] = est.atDouble(k)
    if len(keys) < 2:
        return None

    kv = gtsam.KeyVector()
    kv.append(key_pose)
    for sf in keys:
        kv.append(key_of[sf])
    try:
        jm = isam2.jointMarginalCovariance(kv)
    except (RuntimeError, IndexError):
        return None
    m = len(keys)
    cov = np.empty((m, m))
    cross = np.empty((3, m))
    for i, a in enumerate(keys):
        ka = key_of[a]
        for j, b in enumerate(keys):
            cov[i, j] = jm.at(ka, key_of[b])[0, 0]
        cross[:, i] = jm.at(key_pose, ka)[3:6, 0]
    cross = est.atPose3(key_pose).rotation().matrix() @ cross

    if not (np.all(np.isfinite(cov)) and np.all(np.isfinite(cross))):
        return None

    el = {int(s): float(tc.nav.el[int(s) - 1]) for s, _ in keys}
    return ArProblem(keys=keys, values=values, cov=cov, cross=cross,
                     elevations=el)


def fixed_state(tc, problem, result):
    """Conditional position update x - Qab Qb^-1 (y_float - y_fixed) in DD space."""
    keys, cov, values = problem.keys, problem.cov, problem.values
    xa = tc.nav.x.copy()
    for sf, value in result.fixed.items():
        xa[tc.IB(sf[0], sf[1], tc.nav.na)] = value

    idx = {sf: i for i, sf in enumerate(keys)}
    D = np.zeros((len(result.pairs), len(keys)))
    for row, (ref, tgt) in enumerate(result.pairs):
        D[row, idx[ref]] = 1.0
        D[row, idx[tgt]] = -1.0
    x_float = np.array([values[sf] for sf in keys])
    x_fixed = np.array([result.fixed.get(sf, values[sf]) for sf in keys])
    Qb = D @ cov @ D.T
    Qab = problem.cross @ D.T
    try:
        gain = np.linalg.solve(Qb.T, Qab.T).T
    except np.linalg.LinAlgError:
        return xa
    d_enu = gain @ (D @ (x_float - x_fixed))
    xa[0:3] = tc.nav.x[0:3] - tc.R_enu2ecef @ d_enu
    return xa
