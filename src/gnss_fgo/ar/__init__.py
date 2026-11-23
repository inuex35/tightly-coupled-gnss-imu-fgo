"""Integer ambiguity resolution (pure LAMBDA + optional partial AR)."""

import gtsam

from ..utils import huber_noise, sorted_amb_items
from . import problem as ar_problem
from .ambiguity_resolver import AmbiguityResolver

__all__ = ['ar_problem', 'AmbiguityResolver', 'run_ar']

_FFRT_MIN_RATIO = 1.5  # Teunissen FFRT lower clamp
_FFRT_MAX_RATIO = 3.0  # Teunissen FFRT upper clamp (P_f = 0.1%)
_SIGMA_HOLD_CYC = 0.01  # DD integer-hold factor sigma [cycles] (~2 mm)


def _hold_fixed_dd(tc, amb_dict, res):
    """Feed full-AR DD integer constraints back into the smoother."""
    key_of = dict(sorted_amb_items(amb_dict))
    est = tc.isam2.calculateEstimate()
    hold_graph = gtsam.NonlinearFactorGraph()
    noise = huber_noise(_SIGMA_HOLD_CYC, float(tc.cfg.huber_k))
    for ref, tgt in res.pairs:
        kr, kt = key_of.get(ref), key_of.get(tgt)
        if kr is not None and kt is not None and est.exists(kr) and est.exists(kt):
            b_int = float(round(res.fixed[ref] - res.fixed[tgt]))
            hold_graph.add(gtsam.BetweenFactorDouble(kt, kr, b_int, noise))
    if hold_graph.size() > 0:
        tc.isam2.update(hold_graph, gtsam.Values(),
                        gtsam.FixedLagSmootherKeyTimestampMap())


def _resolve(tc, sat_list, amb_dict, allow_partial=False):
    """Run LAMBDA over the smoother's joint marginals. Returns (nb, xa)."""
    problem = ar_problem.build(tc, sat_list, amb_dict)
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
    xa = ar_problem.fixed_state(tc, problem, res)
    if tc.nav.armode >= 2 and res.dropped == 0:
        _hold_fixed_dd(tc, amb_dict, res)
    return res.nb, xa


def run_ar(tc, sat, el, amb_dict, key_pose=None):
    """Run LAMBDA AR (full fix, then partial-AR fallback). Returns (nb, xa)."""
    if tc.nav.armode == 0:
        return 0, None
    if key_pose is not None:
        tc._ar_key_pose = key_pose

    sat_list = [int(x) for x in sat]
    try:
        nb, xa = _resolve(tc, sat_list, amb_dict, allow_partial=False)
    except Exception:
        return 0, None

    if nb <= 0:
        try:
            nb, xa = _resolve(tc, sat_list, amb_dict, allow_partial=True)
        except Exception:
            return 0, None

    if nb <= 0 or xa is None:
        return 0, None

    tc.nav.smode = 4
    return nb, xa
