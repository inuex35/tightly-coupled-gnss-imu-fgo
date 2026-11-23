"""Stage C4 — Continuous LAMBDA ambiguity resolution."""

from .. import ar as _tc_ar


_AR_OUTCOME_CODES = {
    'not_called': 0, 'armode_off': 1, 'entered': 2,
    'lambda_exception': 3, 'lambda_zero': 4,
    'success': 8, 'gdop_gate': 12, 'problem_unposed': 13,
    'min_pairs_declined': 15,
}


def _run_lambda_ar(tc, epoch):
    """Stage C4 — publish marginals and run continuous LAMBDA AR."""
    tc.nav.smode = 5
    epoch.nb = 0
    epoch.xa = None

    info = epoch.info
    tc.nav.x[0:3] = tc._antenna_ecef(epoch.pose_tc, epoch.ecef_tc)
    amb_snapshot = tc._sat_states.amb_keys_dict()
    _tc_ar.nav_bridge.publish_marginals(
        tc, epoch.estimate, tc.Xpose(epoch.key_idx), amb_snapshot)

    epoch.nb, epoch.xa = _tc_ar.run_ar(tc, epoch.sat, epoch.el, amb_snapshot)

    if tc.ar_diag.exception:
        info['ar_exception'] = tc.ar_diag.exception
    if tc.ar_diag.outcome:
        info['ar_outcome_code'] = _AR_OUTCOME_CODES.get(tc.ar_diag.outcome, -1)
    if tc.ar_diag.partial_dropped:
        info['ar_partial_dropped'] = int(tc.ar_diag.partial_dropped)
    return None
