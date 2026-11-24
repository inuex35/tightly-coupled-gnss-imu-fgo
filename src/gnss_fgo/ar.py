"""Integer ambiguity resolution (LAMBDA and partial AR)."""

from dataclasses import dataclass, field

import gtsam
import numpy as np
from cssrlib.core.mlambda import (
    estimILS,
    ldldecom,
    mlambda,
    reduction,
    sr_boost,
)
from cssrlib.gnss import sat2prn

from .utils import huber_noise, sorted_amb_items


_FFRT_MIN_RATIO = 1.5  # Teunissen FFRT lower clamp
_FFRT_MAX_RATIO = 3.0  # Teunissen FFRT upper clamp (P_f = 0.1%)
_SIGMA_HOLD_CYC = 0.01  # DD integer-hold factor sigma [cycles] (~2 mm)


# ── Sequential AR problem assembly & conditional update ───────────────

@dataclass
class ArProblem:
    """One epoch's float-ambiguity problem, self-contained."""

    keys: list
    values: dict
    cov: np.ndarray
    cross: np.ndarray
    elevations: dict = field(default_factory=dict)


def build_problem(tc, sat_list, amb_dict, key_pose,
                  el=None, visible_sat_freqs=None, seeded_sat_freqs=None):
    """Assemble ArProblem from live smoother variables, or None if unposed."""
    smoother = tc.isam2
    if smoother is None:
        return None
    isam2 = smoother.getISAM2()
    if isam2 is None:
        return None
    est = smoother.calculateEstimate()

    if key_pose is None or not est.exists(key_pose):
        return None

    key_of = dict(sorted_amb_items(amb_dict))
    present = {int(s) for s in sat_list}
    seeded = seeded_sat_freqs or set()
    cp_visible = visible_sat_freqs or set()
    keys, values = [], {}
    for (s, f), k in key_of.items():
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

    if el is not None and len(el) == len(sat_list):
        el_by_sat = {int(s): float(e) for s, e in zip(sat_list, el)}
        elev = {int(s): el_by_sat[int(s)] for s, _ in keys}
    else:
        elev = {int(s): float(tc.nav.el[int(s) - 1]) for s, _ in keys}
    return ArProblem(keys=keys, values=values, cov=cov, cross=cross,
                     elevations=elev)


def fixed_state(tc, problem, result):
    """Conditional position update x - Qab Qb^-1 (y_float - y_fixed) in DD space."""
    keys, cov, values = problem.keys, problem.cov, problem.values
    pos_float = tc.nav.x[0:3].copy()

    idx = {sf: i for i, sf in enumerate(keys)}
    D = np.zeros((len(result.pairs), len(keys)))
    for row, (ref, tgt) in enumerate(result.pairs):
        D[row, idx[ref]] = 1.0
        D[row, idx[tgt]] = -1.0
    y_float = D @ np.array([values[sf] for sf in keys])
    y_fixed = result.dd_fixed
    Qb = D @ cov @ D.T
    Qab = problem.cross @ D.T
    try:
        gain = np.linalg.solve(Qb.T, Qab.T).T
    except np.linalg.LinAlgError:
        return pos_float
    d_enu = gain @ (y_float - y_fixed)
    return pos_float - tc.R_enu2ecef @ d_enu


# ── LAMBDA resolver ───────────────────────────────────────────────────

@dataclass
class ResolverResult:
    """Outcome of one resolution attempt."""

    nb: int = 0
    fixed: dict = field(default_factory=dict)
    dd_fixed: np.ndarray | None = None
    s0: float = 0.0
    s1: float = 0.0
    pairs: list = field(default_factory=list)
    thres_used: float = 0.0
    dropped: int = 0


class AmbiguityResolver:
    """LAMBDA over a float ambiguity vector and its covariance."""

    min_fix = 12
    min_sats = 5
    par_p0 = 0.995
    par_trials = 8

    def __init__(self, thresar=3.0, min_pairs=2,
                 el_mask=0.0, thresar_min=0.0, thresar_max=0.0):
        self.thresar = float(thresar)
        self.min_pairs = int(min_pairs)
        self.el_mask = float(el_mask)
        self.thresar_min = float(thresar_min)
        self.thresar_max = float(thresar_max)

    _AR_POLY_COEFFS = (
        (-1.94058448e-01, -7.79023476e+00, 1.24231120e+02,
         -4.03126050e+02, 3.50413202e+02),
        (6.42237302e-01, -8.39813962e+00, 2.92107285e+01,
         -2.37577308e+01, -1.14307128e+00),
        (-2.22600390e-02, 3.23169103e-01, -1.39837429e+00,
         2.19282996e+00, -5.34583971e-02))

    def ratio_threshold(self, nb):
        """Ratio-test threshold for ``nb`` DD candidates."""
        if self.thresar_min == self.thresar_max:
            return self.thresar
        p0 = self.thresar
        nb1 = min(int(nb), 50)
        coeff = []
        for row in self._AR_POLY_COEFFS:
            c = row[0]
            for kj in range(1, 5):
                c = c * p0 + row[kj]
            coeff.append(c)
        th = coeff[0]
        for ki in range(1, 3):
            th = th / (nb1 + 1) + coeff[ki]
        return min(max(th, self.thresar_min), self.thresar_max)

    def double_difference(self, keys, elevations):
        """Pick a reference per (constellation, frequency) and pair the rest."""
        by_group = {}
        for sat, freq in keys:
            if elevations.get(int(sat), -np.inf) < self.el_mask:
                continue
            sys_i, _ = sat2prn(int(sat))
            by_group.setdefault((sys_i, int(freq)), []).append((int(sat), int(freq)))
        pairs = []
        for group in sorted(by_group):
            members = sorted(by_group[group])
            if len(members) < 2:
                continue
            ref = members[0]
            for key in members[1:]:
                pairs.append((ref, key))
        return pairs

    def resolve(self, float_values, covariance, keys, elevations,
                allow_partial=False):
        """Fix the ambiguities."""
        keys = list(keys)
        pairs = self.double_difference(keys, elevations)
        full_sats = {s for ref, tgt in pairs for s in (ref[0], tgt[0])}
        if (len(pairs) < max(self.min_pairs, self.min_fix)
                or len(full_sats) < self.min_sats):
            return ResolverResult(pairs=pairs)

        index = {k: i for i, k in enumerate(keys)}
        D = np.zeros((len(pairs), len(keys)))
        for row, (ref, tgt) in enumerate(pairs):
            D[row, index[ref]] = 1.0
            D[row, index[tgt]] = -1.0

        x = np.array([float_values[k] for k in keys], dtype=float)
        y = D @ x
        Q = D @ np.asarray(covariance, dtype=float) @ D.T

        candidates, sqnorms, nfix, _ = mlambda(y, Q)
        sqnorm_best = float(sqnorms[0]) if len(sqnorms) > 0 else 0.0
        sqnorm_second = float(sqnorms[1]) if len(sqnorms) > 1 else 0.0
        if sqnorm_best <= 1e-12 * max(sqnorm_second, 1.0):
            sqnorm_best = 0.0
        ratio = 0.0 if sqnorm_best <= 0.0 else sqnorm_second / sqnorm_best
        thres = self.ratio_threshold(len(pairs))
        result = ResolverResult(
            s0=sqnorm_best, s1=sqnorm_second, pairs=pairs, thres_used=thres)
        if nfix <= 0:
            return result
        if not (sqnorm_best <= 0.0 or ratio >= thres):
            par = self._partial(y, Q, pairs) if allow_partial else None
            if par is None:
                return result
            dd_fixed_vec, nb, sqnorm_best, sqnorm_second, thres, k = par
            result.s0, result.s1 = sqnorm_best, sqnorm_second
            result.thres_used = thres
            result.dropped = k
            result.nb = nb
            result.dd_fixed = dd_fixed_vec
            result.fixed = self._restore_sd(pairs, float_values, dd_fixed_vec)
            return result

        result.nb = len(pairs)
        result.dd_fixed = candidates[:, 0]
        result.fixed = self._restore_sd(pairs, float_values, candidates[:, 0])
        return result

    @staticmethod
    def _restore_sd(pairs, float_values, bvec):
        fixed = {}
        for row, (ref, tgt) in enumerate(pairs):
            fixed.setdefault(ref, float_values[ref])
            fixed[tgt] = fixed[ref] - float(bvec[row])
        return fixed

    def _partial(self, y, Q, pairs):
        n = len(y)
        if n <= self.min_fix:
            return None
        L, d = ldldecom(Q)
        L, d, Z = reduction(L, d)
        iZt = np.round(np.linalg.inv(Z.T))
        z = Z.T @ y
        Qz = L.T @ np.diag(d) @ L

        kmax = n - self.min_fix
        k = 1
        while k < kmax and sr_boost(d[k:]) < self.par_p0:
            k += 1
        for _ in range(self.par_trials):
            if k > kmax:
                return None
            int_rows = np.all(iZt[:, :k] == 0.0, axis=1)
            fixed_sats = {
                s for row, (ref, tgt) in enumerate(pairs)
                if int_rows[row] for s in (ref[0], tgt[0])
            }
            if len(fixed_sats) < self.min_sats:
                return None
            zpar, sq = estimILS(L[k:, k:], d[k:], z[k:], 2)
            sqnorm_best, sqnorm_second = float(sq[0]), float(sq[1])
            nb = n - k
            thres = self.ratio_threshold(nb)
            if sqnorm_best > 0.0 and sqnorm_second / sqnorm_best >= thres:
                QP = np.linalg.solve(Qz[k:, k:].T, Qz[:k, k:].T).T
                zc = z[:k] - QP @ (z[k:] - zpar[:, 0])
                dd_fixed_vec = iZt @ np.concatenate((zc, zpar[:, 0]))
                return dd_fixed_vec, nb, sqnorm_best, sqnorm_second, thres, k
            k += max(1, (kmax - k) // 4)
        return None


# ── Sequential AR entry point ─────────────────────────────────────────

def _hold_fixed_dd(tc, amb_dict, res):
    """Feed full-AR DD integer constraints back into the smoother."""
    key_of = dict(sorted_amb_items(amb_dict))
    est = tc.isam2.calculateEstimate()
    hold_graph = gtsam.NonlinearFactorGraph()
    noise = huber_noise(_SIGMA_HOLD_CYC, float(tc.cfg.huber_k))
    for (ref, tgt), b_val in zip(res.pairs, res.dd_fixed):
        kr, kt = key_of.get(ref), key_of.get(tgt)
        if kr is not None and kt is not None and est.exists(kr) and est.exists(kt):
            hold_graph.add(gtsam.BetweenFactorDouble(
                kt, kr, float(round(b_val)), noise))
    if hold_graph.size() > 0:
        tc.isam2.update(hold_graph, gtsam.Values(),
                        gtsam.FixedLagSmootherKeyTimestampMap())


def _resolve(tc, sat_list, amb_dict, key_pose, el=None,
             visible_sat_freqs=None, seeded_sat_freqs=None, allow_partial=False):
    """Run LAMBDA over the smoother's joint marginals. Returns (n_fixed, fixed_pos)."""
    problem = build_problem(
        tc, sat_list, amb_dict, key_pose,
        el=el, visible_sat_freqs=visible_sat_freqs,
        seeded_sat_freqs=seeded_sat_freqs)
    if problem is None:
        return 0, None
    resolver = AmbiguityResolver(
        thresar=_FFRT_MAX_RATIO,
        el_mask=float(tc.nav.elmaskar),
        thresar_min=_FFRT_MIN_RATIO,
        thresar_max=_FFRT_MAX_RATIO)
    res = resolver.resolve(problem.values, problem.cov, problem.keys,
                           problem.elevations, allow_partial=allow_partial)
    if res.nb <= 0:
        return 0, None
    fixed_pos = fixed_state(tc, problem, res)
    if tc.nav.armode >= 2 and res.dropped == 0:
        _hold_fixed_dd(tc, amb_dict, res)
    return res.nb, fixed_pos


def run_ar(tc, sat, el, amb_dict, key_pose=None,
           visible_sat_freqs=None, seeded_sat_freqs=None):
    """Run LAMBDA AR (full fix, then partial-AR fallback). Returns (n_fixed, fixed_pos)."""
    if tc.nav.armode == 0:
        return 0, None

    sat_list = [int(x) for x in sat]
    try:
        n_fixed, fixed_pos = _resolve(
            tc, sat_list, amb_dict, key_pose, el=el,
            visible_sat_freqs=visible_sat_freqs,
            seeded_sat_freqs=seeded_sat_freqs, allow_partial=False)
    except Exception:
        return 0, None

    if n_fixed <= 0:
        try:
            n_fixed, fixed_pos = _resolve(
                tc, sat_list, amb_dict, key_pose, el=el,
                visible_sat_freqs=visible_sat_freqs,
                seeded_sat_freqs=seeded_sat_freqs, allow_partial=True)
        except Exception:
            return 0, None

    if n_fixed <= 0 or fixed_pos is None:
        return 0, None

    tc.nav.smode = 4
    return n_fixed, fixed_pos
