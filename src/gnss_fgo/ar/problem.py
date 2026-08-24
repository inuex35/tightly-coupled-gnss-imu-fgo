"""Build the ambiguity-resolution problem from the smoother.

This module only *reads*: the ISAM2 estimate and joint marginals, the
fix-and-hold table, and the selection state (``nav.vsat``, elevations). The
output is a plain :class:`ArProblem` -- everything LAMBDA needs, in one
value, with no reference back to the smoother or to ``nav``.

Selection contract (measured, not designed -- both halves are load-bearing):

* the satellite list is a **presence check** and ``nav.vsat`` does the
  selecting, exactly as in cssrlib's ``ddidx``. Using only the list drops
  the surviving bands of every excluded satellite; using only ``vsat`` lets
  an excluded satellite back in. Each mistake was measured on tokyo run2
  before this line was written the way it is.
* a **held** ambiguity contributes its pinned value with ``varholdamb``
  variance on the diagonal; its off-diagonal terms are zero unless the
  key still lives in the graph.
"""

from dataclasses import dataclass, field

import gtsam
import numpy as np

from ..utils import sorted_amb_items


@dataclass
class ArProblem:
    """One epoch's float-ambiguity problem, self-contained."""

    keys: list                       # ordered (sat, freq)
    values: dict                     # (sat, freq) -> float ambiguity [cycles]
    cov: np.ndarray                  # joint covariance over ``keys``
    cross: np.ndarray                # cov(position ENU, ambiguity), 3 x n
    held: dict = field(default_factory=dict)   # (sat, freq) -> pinned variance
    elevations: dict = field(default_factory=dict)   # sat -> el [rad]


def build(tc, sat_list, amb_dict):
    """Assemble the :class:`ArProblem`, or ``None`` when it cannot be posed.

    ``None`` means "no fix this epoch": a missing pose key, fewer than
    two candidates, a marginals failure, or non-finite covariance.
    """
    # Phase 1 estimates on its own short-lag smoother (tc.isam); tc.isam2
    # only exists from the Phase-2 transition on.
    smoother = tc.isam2 if tc.phase == 2 else tc.isam
    if smoother is None:
        return None
    isam2 = smoother.getISAM2()
    if isam2 is None:
        return None
    est = smoother.calculateEstimate()

    keys, values, held_var = [], {}, {}
    key_of = dict(sorted_amb_items(amb_dict))
    present = {int(s) for s in sat_list}
    for (s, f), value in tc._sat_states.held_items():
        sf = (int(s), int(f))
        if int(s) not in present or tc.nav.vsat[int(s) - 1, int(f)] != 1:
            continue
        keys.append(sf)
        values[sf] = float(value)
        held_var[sf] = max(float(tc.cfg.varholdamb), 1e-9)
    for (s, f), k in sorted_amb_items(key_of):
        sf = (int(s), int(f))
        if sf in values:
            continue
        if int(s) in present and est.exists(k) and tc.nav.vsat[s - 1, f] == 1:
            keys.append(sf)
            values[sf] = est.atDouble(k)
    if len(keys) < 2:
        return None

    in_graph = [sf for sf in keys if sf in key_of and est.exists(key_of[sf])]
    key_pose = tc._ar_key_pose
    if key_pose is None:
        return None

    # Retry/subset attempts resolve over key subsets of the first
    # attempt: slice its cached extraction instead of re-reading.
    cache = tc.current_epoch.ar_joint_cache
    if (cache is not None and cache['key_pose'] == key_pose
            and all(sf in cache['pos'] for sf in in_graph)):
        idx = [cache['pos'][sf] for sf in in_graph]
        cov_g = cache['cov'][np.ix_(idx, idx)]
        cross_g = cache['cross'][:, idx]
    else:
        kv = gtsam.KeyVector()
        kv.append(key_pose)
        for sf in in_graph:
            kv.append(key_of[sf])
        try:
            jm = isam2.jointMarginalCovariance(kv)
        except (RuntimeError, IndexError):
            return None
        m = len(in_graph)
        cov_g = np.empty((m, m))
        cross_g = np.empty((3, m))
        for i, a in enumerate(in_graph):
            ka = key_of[a]
            for j, b in enumerate(in_graph):
                cov_g[i, j] = jm.at(ka, key_of[b])[0, 0]
            cross_g[:, i] = jm.at(key_pose, ka)[3:6, 0]
        # Pose3 marginals live in the BODY tangent frame (right
        # retract); rotate the position rows into the nav frame.
        # Identity in Phase 1, so only Phase-2 fixes move.
        cross_g = est.atPose3(key_pose).rotation().matrix() @ cross_g
        if cache is None or len(in_graph) > len(cache['pos']):
            tc.current_epoch.ar_joint_cache = {
                'key_pose': key_pose,
                'pos': {sf: i for i, sf in enumerate(in_graph)},
                'cov': cov_g, 'cross': cross_g,
            }

    n = len(keys)
    cov = np.zeros((n, n))
    cross = np.zeros((3, n))
    gpos = {sf: i for i, sf in enumerate(in_graph)}
    for i, a in enumerate(keys):
        gi = gpos.get(a)
        if gi is not None:
            for j, b in enumerate(keys):
                gj = gpos.get(b)
                if gj is not None:
                    cov[i, j] = cov_g[gi, gj]
            cross[:, i] = cross_g[:, gi]
        if a in held_var:
            cov[i, i] = held_var[a]
    if not (np.all(np.isfinite(cov)) and np.all(np.isfinite(cross))):
        return None

    el = {int(s): float(tc.nav.el[int(s) - 1]) for s, _ in keys}
    return ArProblem(keys=keys, values=values, cov=cov, cross=cross,
                     held=held_var, elevations=el)


def fixed_state(tc, problem, result):
    """The full fixed state vector for an accepted resolution.

    Ambiguity slots take the fixed values; the position takes the standard
    conditional update ``x - Qab Qb^-1 (y_float - y_fixed)`` in DD space.
    Both the direction of that correction and the content of ``Qab`` are
    part of cssrlib's contract: the sign is a subtraction, and ``cross`` is
    already a covariance with position, so it is differenced, not multiplied
    by the covariance again. Each was wrong once and moved the fixed
    position enough for ``valpos`` to reach a different verdict from the
    same integers.

    Returns ``xa``.
    """
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
