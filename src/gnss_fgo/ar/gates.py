"""Gates around a LAMBDA answer: when not to believe it.

Two related judgements, both reading the same context (post-fit DDPR
residuals, CP-hold and ddpr-bad streaks):

* :func:`validate_fix` -- RTKLIB valpos on the fixed solution, then the
  graph-objective delta test, then :func:`context_reject`;
* :func:`context_reject` -- a fix that is small (nb <= ar_context_nb_max)
  in a burst-like context is more likely wrong than lucky.

These are policy, not resolution: nothing here touches LAMBDA's inputs.
"""


from ..pipeline import residuals as _tc_residuals


def context_reject(tc, nb):
    """Reject fragile AR fixes in burst-like contexts before hold/anchor."""
    nb = int(nb)
    if nb <= 0:
        return False, None
    main_res = float(tc._cached_ddpr_res_pre
                     or tc._mres_signals.last_res or 0.0)
    per_sat = tc._mres_signals.per_sat or {}
    worst_res = float(max(per_sat.values())) if per_sat else 0.0
    cp_hold_active = int(tc._recov_cp_hold or 0) > 0
    ddpr_bad_active = int(tc._ddpr_bad_count or 0) > 0

    burst_like = False
    if (bool(tc.cfg.ar_context_reject_during_cp_hold)
            and cp_hold_active):
        burst_like = True
    if (bool(tc.cfg.ar_context_reject_during_ddpr_bad)
            and ddpr_bad_active):
        burst_like = True

    if main_res > float(tc.cfg.ar_context_main_ddpr_max):
        burst_like = True
    if worst_res > float(tc.cfg.ar_context_worst_sat_max):
        burst_like = True

    if burst_like and nb <= int(tc.cfg.ar_context_nb_max):
        return True, {
            'nb': nb,
            'main_ddpr_res': main_res,
            'worst_sat_res': worst_res,
            'cp_hold_active': cp_hold_active,
            'ddpr_bad_active': ddpr_bad_active,
        }
    return False, None


def validate_fix(tc, obs, rs, vs, dts, sat, el, iu, xa, nb,
                  estimate=None, key_pose=None, graph=None):
    """Phase C — valpos, then the fix_dres objective-delta gate, then ar_context_reject. True iff the fix survives all three."""
    # xa[0:3] is already antenna position (nav.x[0:3] was set to antenna pos)
    fix_antenna = xa[0:3]

    yu, eu, _ = tc.zdres(obs, None, None, rs, vs, dts, fix_antenna)
    v_fix, _, R_fix = tc.sdres(obs, xa, yu[iu], eu[iu], sat, el)

    if not tc.valpos(v_fix, R_fix):
        tc.ar_diag.outcome = 'valpos_failed'
        return False

    # Likelihood-ratio gate in the graph's OWN objective (pre-hold):
    # Δres = DDPR RMS with the pose moved to the fixed solution xa,
    # minus the same RMS at the float solution. A wrong-integer basin
    # is phase-self-consistent but the epoch's code factors protest —
    # the DELTA isolates that protest from the NLOS noise floor that
    # defeats absolute thresholds. Evaluated BEFORE fix-and-hold, so a
    # wrong basin is rejected before holds can lock it (once holds drag
    # the float into the basin the delta vanishes — timing matters).
    dres_thr = float(tc.cfg.ar_fix_dres_max)
    if dres_thr > 0.0 and graph is not None and estimate is not None \
            and key_pose is not None:
        res_pre = tc._cached_ddpr_res_pre
        res_xa = _tc_residuals.ddpr_res_at_fixed_pose(
            tc, graph, estimate, key_pose, xa)
        if res_pre is not None and res_xa is not None:
            fix_dres = float(res_xa) - float(res_pre)
            if fix_dres > dres_thr:
                tc.ar_diag.outcome = 'fix_dres'
                return False

    reject_ctx, reject_detail = context_reject(tc, nb)
    if reject_ctx:
        tc._ar_context_reject = reject_detail
        tc.ar_diag.outcome = 'ar_context_reject'
        return False
    tc._ar_context_reject = None
    tc.ar_diag.outcome = 'success'

    return True


