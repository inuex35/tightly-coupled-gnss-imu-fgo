"""Integer ambiguity resolution (pure LAMBDA + optional partial AR)."""

from . import hold as ar_hold
from . import nav_bridge
from . import problem as ar_problem
from .ambiguity_resolver import AmbiguityResolver

__all__ = ['ar_hold', 'nav_bridge', 'ar_problem', 'AmbiguityResolver', 'run_ar']

_FFRT_MIN_RATIO = 1.5  # Teunissen FFRT lower clamp


def _resolve(tc, sat_list, amb_dict, allow_partial=False):
    """Run LAMBDA over the smoother's joint marginals. Returns (nb, xa, res, problem)."""
    problem = ar_problem.build(tc, sat_list, amb_dict)
    if problem is None:
        tc.ar_diag.outcome = 'problem_unposed'
        tc._last_s0 = 0.0
        tc._last_s1 = 0.0
        return 0, None, None, None
    ratio_max = float(tc.cfg.ar_ratio)
    resolver = AmbiguityResolver(
        thresar=ratio_max,
        el_mask=float(tc.nav.elmaskar),
        thresar_min=min(_FFRT_MIN_RATIO, ratio_max),
        thresar_max=ratio_max)
    res = resolver.resolve(problem.values, problem.cov, problem.keys,
                           problem.elevations, allow_partial=allow_partial)
    tc._last_ar_thres = float(res.thres_used) or ratio_max
    nav_bridge.publish_attempt(tc, sat_list, res)
    if res.nb <= 0:
        if 0 < len(res.pairs) < resolver.min_pairs:
            tc.ar_diag.outcome = 'min_pairs_declined'
        return 0, None, res, problem
    tc.ar_diag.partial_dropped = int(res.dropped)
    xa = ar_problem.fixed_state(tc, problem, res)
    return res.nb, xa, res, problem


def run_ar(tc, sat, el, amb_dict):
    """Run LAMBDA AR (full fix, then partial-AR fallback). Returns (nb, xa)."""
    tc._last_ar_result = None
    tc.ar_diag.outcome = 'not_called'
    if tc.nav.armode == 0:
        tc.ar_diag.outcome = 'armode_off'
        return 0, None

    sat_list = [int(x) for x in sat]
    sat_in_obs = set(sat_list)
    tc.ar_diag.amb_not_in_obs = sum(
        1 for (ss, _ff) in amb_dict.keys() if int(ss) not in sat_in_obs)
    tc.ar_diag.sat_in_obs_size = len(sat_in_obs)
    tc.ar_diag.outcome = 'entered'
    tc.ar_diag.resamb_raw_nb = -1

    try:
        nb, xa, res, _prob = _resolve(tc, sat_list, amb_dict, allow_partial=False)
    except Exception as ex:
        tc.ar_diag.outcome = 'lambda_exception'
        tc.ar_diag.exception = f'{type(ex).__name__}: {ex}'
        return 0, None

    tc.ar_diag.resamb_raw_nb = int(nb)
    if nb <= 0:
        try:
            nb, xa, res, _prob = _resolve(tc, sat_list, amb_dict, allow_partial=True)
        except Exception as ex:
            tc.ar_diag.outcome = 'lambda_exception'
            tc.ar_diag.exception = f'{type(ex).__name__}: {ex}'
            return 0, None

    if nb <= 0 or xa is None:
        if tc.ar_diag.outcome != 'min_pairs_declined':
            tc.ar_diag.outcome = 'lambda_zero'
        return 0, None

    tc._last_ar_result = res
    tc.ar_diag.outcome = 'success'
    tc.nav.smode = 4
    return nb, xa
