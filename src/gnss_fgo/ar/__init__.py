"""Integer ambiguity resolution.

    problem      graph -> ArProblem (reads only) + the fixed state vector
    ambiguity_resolver   LAMBDA over that problem (pure)
    subset       the ranked exclusion search (the only retry)
    gates        pre-checks and fix validation
    hold         fix-and-hold
    nav_bridge   every nav write, and the table of who reads each field

This package root wires them into the epoch flow (:func:`run_ar`).
"""

import numpy as np

from . import gates as ar_gates
from . import hold as ar_hold
from . import nav_bridge
from . import problem as ar_problem
from . import subset as ar_subset
from .ambiguity_resolver import AmbiguityResolver

__all__ = ['ar_gates', 'ar_hold', 'nav_bridge', 'ar_problem',
           'ar_subset', 'AmbiguityResolver', 'run_ar']



def _resolve(tc, sat_list, amb_dict, allow_partial=False):
    """AR straight off the smoother (the only resolver since the cssrlib
    resamb dispatch was retired).

    Three stages, one module each: :mod:`ar_problem` reads the smoother into
    a self-contained problem, :class:`AmbiguityResolver` fixes the integers,
    and :mod:`nav_bridge` publishes the side effects cssrlib's callers still
    read. Always returns ``(nb, x)``.
    """
    problem = ar_problem.build(tc, sat_list, amb_dict)
    if problem is None:
        # A no-fix result, never None; zero the stash or the subset
        # search reads a stale ratio.
        tc.ar_diag.outcome = 'problem_unposed'
        tc._last_s0 = 0.0
        tc._last_s1 = 0.0
        return 0, tc.nav.x.copy()
    resolver = AmbiguityResolver(
        thresar=float(tc.nav.thresar),
        el_mask=float(tc.nav.elmaskar),  # 20 deg, from cssrlib estimation config
        thresar_min=float(tc.cfg.ar_thresar_min),
        thresar_max=float(tc.cfg.ar_thresar_max))
    res = resolver.resolve(problem.values, problem.cov, problem.keys,
                           problem.elevations, allow_partial=allow_partial)
    tc._last_ar_thres = float(res.thres_used) or float(tc.nav.thresar)
    nav_bridge.publish_attempt(tc, sat_list, res)
    if res.nb <= 0:
        if 0 < len(res.pairs) < resolver.min_pairs:
            # A lone-pair decline is a candidate-stage verdict, not
            # ratio starvation: it must freeze the starvation counter
            # or every later purge fires early.
            # Zero-pair declines keep counting as starvation.
            tc.ar_diag.outcome = 'min_pairs_declined'
        return 0, tc.nav.x.copy()
    if res.s0 <= 0.0:
        # Exact fit: the ratio test could not run (held integers
        # re-entering the search); the fix rides on xvalidate/context.
        tc.ar_diag.exact_fit_accept = 1
    tc.ar_diag.partial_dropped = int(res.dropped)
    xa = ar_problem.fixed_state(tc, problem, res)
    return res.nb, xa


def _run_single_ar_attempt(tc, sat, amb_dict, sat_exclude=None,
                           restore_state=True):
    """Run one AR attempt, optionally excluding one or more satellites."""
    sat_list = [int(s) for s in sat]
    vsat_snapshot = tc.nav.vsat.copy()
    fix_snapshot = tc.nav.fix.copy()
    try:
        if sat_exclude is not None:
            if isinstance(sat_exclude, (int, np.integer)):
                excl = {int(sat_exclude)}
            else:
                excl = {int(x) for x in sat_exclude}
            for s in excl:
                tc.nav.vsat[s - 1, :] = 0
            sat_list = [s for s in sat_list if s not in excl]
        return _resolve(tc, sat_list, amb_dict)
    finally:
        if restore_state:
            tc.nav.vsat[:, :] = vsat_snapshot
            tc.nav.fix[:, :] = fix_snapshot


def _try_subset_ar(tc, sat, el, amb_dict):
    """Subset fallback over the same single-attempt entry point."""
    def attempt(t, s, sat_exclude=None, restore_state=True):
        return _run_single_ar_attempt(t, s, amb_dict,
                                      sat_exclude=sat_exclude,
                                      restore_state=restore_state)
    return ar_subset.try_subset_ar(tc, sat, el, amb_dict, attempt=attempt)


def run_ar(tc, sat, el, amb_dict):
    """LAMBDA AR with validation."""
    tc._ar_subset_debug = None
    tc.ar_diag.outcome = 'not_called'
    if tc.nav.armode == 0:
        tc.ar_diag.outcome = 'armode_off'
        return 0, None
    _record_amb_diagnostics(tc, sat, amb_dict)
    nb, xa = _run_lambda_attempts(tc, sat, el, amb_dict)
    if nb <= 0:
        return 0, None
    if not ar_gates.validate_fix(tc, nb):
        return 0, None
    # fix-and-hold is the caller's move, after its remaining verdicts.
    tc.nav.smode = 4
    return nb, xa


def _record_amb_diagnostics(tc, sat, amb_dict):
    """Phase A — count (sat, freq) entries in amb_dict / held that are missing from this-epoch sat list. cssrlib _ddidx_core silently skips them via not sat_present[sat_i]; surfacing the count makes "amb_size large yet nb=0" cases diagnosable."""
    sat_in_obs = {int(s) for s in sat}
    diag_amb_not_in_obs = sum(
        1 for (ss, ff) in amb_dict.keys() if int(ss) not in sat_in_obs)
    diag_held_not_in_obs = sum(
        1 for (ss, ff), _ in tc._sat_states.held_items()
        if int(ss) not in sat_in_obs)
    tc.ar_diag.amb_not_in_obs = diag_amb_not_in_obs
    tc.ar_diag.held_not_in_obs = diag_held_not_in_obs
    tc.ar_diag.sat_in_obs_size = len(sat_in_obs)
    tc.ar_diag.outcome = 'entered'


def _run_lambda_attempts(tc, sat, el, amb_dict):
    """Phase B — run the resolver, then the subset exclusion search on
    a decline, then the lambda_zero guard. Returns (nb, xa) or
    (0, None) on any rejection."""
    tc.ar_diag.resamb_raw_nb = -1
    try:
        # One resolver, both phases (no fallback — problem-unposed
        # epochs simply do not fix).
        sats = [int(x) for x in sat]
        nb, xa = _resolve(tc, sats, amb_dict)
    except Exception as ex:
        # mlambda raises LambdaError (a LinAlgError) when Qah is not
        # positive definite — a live path in the degenerate held-fit
        # regime, not a can't-happen guard.
        tc.ar_diag.outcome = 'lambda_exception'
        tc.ar_diag.exception = f'{type(ex).__name__}: {ex}'
        return 0, None

    tc.ar_diag.resamb_raw_nb = int(nb)
    if (nb <= 0 and bool(tc.cfg.subset_ar_enable)
            and len(amb_dict) >= int(tc.cfg.subset_ar_min_nb) + 1):
        try:
            nb, xa = _try_subset_ar(tc, sat, el, amb_dict)
        except Exception as ex:
            tc.ar_diag.exception = f'{type(ex).__name__}: {ex}'
            nb, xa = 0, None
    if nb <= 0:
        # Last resort: fix the well-determined z-subspace (published,
        # never held -- see fix_ambiguities).
        try:
            nb, xa = _resolve(tc, [int(x) for x in sat], amb_dict,
                              allow_partial=True)
        except Exception as ex:
            tc.ar_diag.outcome = 'lambda_exception'
            tc.ar_diag.exception = f'{type(ex).__name__}: {ex}'
            nb, xa = 0, None

    if nb <= 0:
        if tc.ar_diag.outcome != 'min_pairs_declined':
            tc.ar_diag.outcome = 'lambda_zero'
        return 0, None
    return nb, xa



