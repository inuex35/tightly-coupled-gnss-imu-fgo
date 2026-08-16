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
  variance on the diagonal, and keeps the graph's off-diagonal terms --
  the same content ``write_marginals`` leaves in ``nav.P``.
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


def build(tc, sat_list):
    """Assemble the :class:`ArProblem`, or ``None`` when it cannot be posed.

    ``None`` means "let the caller fall back": Phase 1 (its own smoother, its
    own keys), a missing pose key, fewer than two candidates, a marginals
    failure, or non-finite covariance.
    """
    smoother = getattr(tc, 'isam2', None)
    if tc.phase != 2 or smoother is None:
        return None
    isam2 = smoother.getISAM2()
    if isam2 is None:
        return None
    est = smoother.calculateEstimate()

    keys, values, held_var = [], {}, {}
    key_of = dict(sorted_amb_items(tc._sat_states.amb_keys_dict()))
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
    key_pose = getattr(tc, '_ar_key_pose', None)
    if key_pose is None:
        return None
    kv = gtsam.KeyVector()
    kv.append(key_pose)
    for sf in in_graph:
        kv.append(key_of[sf])
    try:
        jm = isam2.jointMarginalCovariance(kv)
    except (RuntimeError, IndexError):
        return None

    n = len(keys)
    cov = np.zeros((n, n))
    cross = np.zeros((3, n))
    graph_set = set(in_graph)
    for i, a in enumerate(keys):
        if a not in graph_set:
            cov[i, i] = held_var.get(a, 0.0)
            continue
        for j, b in enumerate(keys):
            if b in graph_set:
                cov[i, j] = jm.at(key_of[a], key_of[b])[0, 0]
        cross[:, i] = jm.at(key_pose, key_of[a])[3:6, 0]
    for sf, var in held_var.items():
        cov[keys.index(sf)][keys.index(sf)] = var
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

    Returns ``(xa, Qb, Qab)``; ``Qb``/``Qab`` are handed on so the caller
    can update ``nav.Pa`` without rebuilding them.
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
        gain = Qab @ np.linalg.inv(Qb)
    except np.linalg.LinAlgError:
        return xa, Qb, None
    d_enu = gain @ (D @ (x_float - x_fixed))
    xa[0:3] = tc.nav.x[0:3] - tc.R_enu2ecef @ d_enu
    return xa, Qb, Qab
