"""Integer ambiguity resolution.

    problem      graph -> ArProblem (reads only) + the fixed state vector
    ambiguity_resolver   LAMBDA over that problem (pure)
    retry        demo5 single-satellite retry policy
    subset       this project's ranked subset fallback
    gates        pre-checks and fix validation
    hold         fix-and-hold
    nav_bridge   every nav write, and the table of who reads each field

This package root wires them into the epoch flow (:func:`run_ar`) and keeps
the monolith-era names importable -- callers and long-lived probes address
``ar._resolve_native`` and friends unchanged.
"""

import numpy as np

from . import gates as ar_gates
from . import hold as ar_hold
from . import nav_bridge
from . import problem as ar_problem
from . import retry as ar_retry
from . import subset as ar_subset
from .ambiguity_resolver import AmbiguityResolver



def _resolve_native(tc, sat_list, amb_dict):
    """AR straight off the smoother, without the cssrlib nav round-trip.

    Three stages, one module each: :mod:`ar_problem` reads the smoother into
    a self-contained problem, :class:`AmbiguityResolver` fixes the integers,
    and :mod:`nav_bridge` publishes the side effects cssrlib's callers still
    read. ``None`` means "fall back to the cssrlib path".

    Equivalence with that path is measured, not assumed: shadowed in both
    directions over tokyo run2 (4361 + 2422 calls) with identical nb and
    ratio, and sequential 3000-epoch runs are line-identical.
    """
    problem = ar_problem.build(tc, sat_list, amb_dict)
    if problem is None:
        # Return a no-fix RESULT, never None: bubbling None out of the
        # retry wrapper killed its continuation (the exclusion pass and
        # its excsat/prev-ratio writes), and that bookkeeping is what
        # the retired cssrlib dispatch was actually contributing on
        # unposable epochs — its absence shifted the estimate from
        # ep4794 (21.35 -> 23.72). Ablated to this minimal form: the
        # dispatch's other side effects (double lock update, ddidx
        # marks, stash zeroing) were all measured non-essential
        # (line-identical without them).
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
            # The resolver declines a problem with a lone DD pair; the
            # cssrlib path instead manufactures a degenerate candidate
            # from it (exact-fit residual, no ratio) and lets a fix
            # gate kill it downstream — which freezes the starvation
            # counter. The decline must freeze it the same way, or the
            # counter drifts one count per lone-pair epoch and every
            # later starvation purge fires early. Zero-pair declines
            # keep counting: the cssrlib path is empty-handed there
            # too and counts them as starvation.
            tc.ar_diag.outcome = 'min_pairs_declined'
        elif res.declined_partial:
            # A candidate-stage verdict, not ratio starvation: mlambda
            # produced integers and the partial-AR guard declined them.
            # Classified apart from lambda_zero so the starvation
            # counter freezes, exactly as it does when the cssrlib path
            # over-reports the same candidate and a downstream fix gate
            # rejects it.
            tc.ar_diag.outcome = 'partial_declined'
        return 0, tc.nav.x.copy()
    xa, Qb, Qab = ar_problem.fixed_state(tc, problem, res)
    nav_bridge.publish_fix(tc, xa, Qb, Qab)
    return res.nb, xa


def _resolve_native_retry(tc, sat_list, amb_dict):
    """The demo5 retry policy around the native resolver (see ar_retry)."""
    return ar_retry.run(
        tc, sat_list, lambda t, sl: _resolve_native(t, sl, amb_dict))


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
                if 1 <= s <= tc.nav.vsat.shape[0]:
                    tc.nav.vsat[s - 1, :] = 0
            sat_list = [s for s in sat_list if s not in excl]
        if tc.phase != 2:
            # Phase 1 resolves through cssrlib's resamb as the PARITY
            # implementation, not a fallback: the native construction
            # agrees to ~9 digits from the first attempt (probed), but
            # the shared-key bootstrap problem is ULP-chaotic — the
            # assembly-order difference alone forks the decision stream
            # and was measured at 21.35 -> 35.49 AllRMS. Bit-parity
            # requires cssrlib's exact arithmetic, which is this call.
            if tc.cfg.rtklib_mode and hasattr(tc, 'resamb_lambda_rtklib'):
                return tc.resamb_lambda_rtklib(sat_list)
            return tc.resamb_lambda(sat_list, tc.nav.parmode,
                                    tc.nav.par_P0)
        native = (_resolve_native_retry(tc, sat_list, amb_dict)
                  if tc.cfg.rtklib_mode
                  else _resolve_native(tc, sat_list, amb_dict))
        return native
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
    """Phase B — call resamb_lambda (rtklib subset / rtklib / vanilla) with optional subset retry, then guard with lambda_zero / min_nb_gate. Returns (nb, xa) or (0, None) on any rejection."""
    tc.ar_diag.resamb_raw_nb = -1
    try:
        # Phase 2: native only (no fallback — problem-unposed epochs
        # simply do not fix). Phase 1: cssrlib as the parity
        # implementation, see _run_single_ar_attempt.
        if tc.phase != 2:
            if tc.cfg.rtklib_mode and hasattr(tc, 'resamb_lambda_rtklib'):
                nb, xa = tc.resamb_lambda_rtklib(sat)
            else:
                nb, xa = tc.resamb_lambda(sat, tc.nav.parmode,
                                          tc.nav.par_P0)
        else:
            sats = [int(x) for x in sat]
            native = (_resolve_native_retry(tc, sats, amb_dict)
                      if tc.cfg.rtklib_mode
                      else _resolve_native(tc, sats, amb_dict))
            nb, xa = native
    except (Exception, SystemExit) as ex:
        # cssrlib mlambda raises SystemExit when Qah is not positive definite
        tc.ar_diag.outcome = 'lambda_exception'
        tc.ar_diag.exception = f'{type(ex).__name__}: {ex}'
        return 0, None

    tc.ar_diag.resamb_raw_nb = int(nb)
    if (nb <= 0 and bool(tc.cfg.subset_ar_enable)
            and len(amb_dict) >= int(tc.cfg.subset_ar_min_nb) + 1):
        try:
            nb, xa = _try_subset_ar(tc, sat, el, amb_dict)
        except (Exception, SystemExit) as ex:
            tc.ar_diag.exception = f'{type(ex).__name__}: {ex}'
            nb, xa = 0, None

    if nb <= 0:
        if tc.ar_diag.outcome not in ('partial_declined',
                                      'min_pairs_declined'):
            tc.ar_diag.outcome = 'lambda_zero'
        return 0, None
    return nb, xa



