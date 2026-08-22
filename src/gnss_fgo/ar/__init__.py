"""Integer ambiguity resolution.

    problem      graph -> ArProblem (reads only) + the fixed state vector
    ambiguity_resolver   LAMBDA over that problem (pure)
    retry        demo5 single-satellite retry policy
    subset       this project's ranked subset fallback
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
from . import retry as ar_retry
from . import subset as ar_subset
from .ambiguity_resolver import AmbiguityResolver



def _resolve(tc, sat_list, amb_dict):
    """AR straight off the smoother (the only resolver since the cssrlib
    resamb dispatch was retired).

    Three stages, one module each: :mod:`ar_problem` reads the smoother into
    a self-contained problem, :class:`AmbiguityResolver` fixes the integers,
    and :mod:`nav_bridge` publishes the side effects cssrlib's callers still
    read. Always returns ``(nb, x)``.
    """
    problem = ar_problem.build(tc, sat_list, amb_dict)
    if problem is None:
        # A no-fix RESULT, never None: the retry wrapper's continuation
        # (exclusion pass, excsat/prev-ratio writes) must still run on
        # unposable epochs — losing it measurably moved the estimate.
        tc.ar_diag.outcome = 'problem_unposed'
        return 0, tc.nav.x.copy()
    resolver = AmbiguityResolver(
        thresar=float(tc.nav.thresar), parmode=int(tc.nav.parmode),
        par_p0=float(tc.nav.par_P0),
        el_mask=float(tc.nav.elmaskar))
    res = resolver.resolve(problem.values, problem.cov, problem.keys,
                           problem.elevations)
    nav_bridge.publish_attempt(tc, sat_list, res)
    if res.nb <= 0:
        if 0 < len(res.pairs) < resolver.min_pairs:
            # A lone-pair decline is a candidate-stage verdict, not
            # ratio starvation: it must freeze the starvation counter
            # (like fix_dres) or every later purge fires early.
            # Zero-pair declines keep counting as starvation.
            tc.ar_diag.outcome = 'min_pairs_declined'
        elif res.declined_partial:
            # Same freeze class: mlambda produced integers and the
            # partial-AR guard declined them.
            tc.ar_diag.outcome = 'partial_declined'
        return 0, tc.nav.x.copy()
    xa = ar_problem.fixed_state(tc, problem, res)
    return res.nb, xa


def _resolve_with_retry(tc, sat_list, amb_dict):
    """The demo5 retry policy around the resolver (see ar_retry)."""
    return ar_retry.run(
        tc, sat_list, lambda t, sl: _resolve(t, sl, amb_dict))


def _run_single_ar_attempt(tc, sat, amb_dict, sat_exclude=None,
                           restore_state=True):
    """Run one AR attempt, optionally excluding one or more satellites."""
    sat_list = [int(s) for s in sat]
    vsat_snapshot = tc.nav.vsat.copy()
    fix_snapshot = tc.nav.fix.copy()
    lock_snapshot = tc.nav.lock.copy()
    try:
        if sat_exclude is not None:
            if isinstance(sat_exclude, (int, np.integer)):
                excl = {int(sat_exclude)}
            else:
                excl = {int(x) for x in sat_exclude}
            for s in excl:
                if 1 <= s <= tc.nav.vsat.shape[0]:
                    tc.nav.vsat[s - 1, :] = 0
            sat_list = [s for s in sat_list if s not in excl]
        return (_resolve_with_retry(tc, sat_list, amb_dict)
                if tc.cfg.rtklib_mode
                else _resolve(tc, sat_list, amb_dict))
    finally:
        if restore_state:
            tc.nav.vsat[:, :] = vsat_snapshot
            tc.nav.fix[:, :] = fix_snapshot
            # Probing attempts must not age the lock counters.
            tc.nav.lock[:, :] = lock_snapshot


def _try_subset_ar(tc, sat, el, amb_dict):
    """Subset fallback over the same single-attempt entry point."""
    def attempt(t, s, sat_exclude=None, restore_state=True):
        return _run_single_ar_attempt(t, s, amb_dict,
                                      sat_exclude=sat_exclude,
                                      restore_state=restore_state)
    return ar_subset.try_subset_ar(tc, sat, el, amb_dict, attempt=attempt)


def run_ar(tc, obs, rs, vs, dts, sat, el, iu, estimate,
              key_pose, amb_dict, graph=None):
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
    if not ar_gates.validate_fix(tc, obs, rs, vs, dts, sat, el, iu, xa, nb,
                         estimate=estimate, key_pose=key_pose, graph=graph):
        return 0, None
    if tc.nav.armode == 3:
        if not ar_hold.apply_fix_and_hold(tc, key_pose, amb_dict, xa):
            return 0, None
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
    """Phase B — run the resolver (with the demo5 retry under rtklib_mode) plus optional subset retry, then guard with lambda_zero / min_nb_gate. Returns (nb, xa) or (0, None) on any rejection."""
    tc.ar_diag.resamb_raw_nb = -1
    try:
        # One resolver, both phases (no fallback — problem-unposed
        # epochs simply do not fix).
        sats = [int(x) for x in sat]
        nb, xa = (_resolve_with_retry(tc, sats, amb_dict)
                  if tc.cfg.rtklib_mode
                  else _resolve(tc, sats, amb_dict))
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
        if tc.ar_diag.outcome not in ('partial_declined',
                                      'min_pairs_declined'):
            tc.ar_diag.outcome = 'lambda_zero'
        return 0, None
    return nb, xa



