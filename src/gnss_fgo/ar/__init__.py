"""Integer ambiguity resolution (pure LAMBDA + optional partial AR)."""

from . import problem as ar_problem
from .ambiguity_resolver import AmbiguityResolver

__all__ = ['ar_problem', 'AmbiguityResolver', 'run_ar']

_FFRT_MIN_RATIO = 1.5  # Teunissen FFRT lower clamp
_FFRT_MAX_RATIO = 3.0  # Teunissen FFRT upper clamp (P_f = 0.1%)


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
