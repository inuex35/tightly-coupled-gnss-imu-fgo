"""Gates around a LAMBDA answer: when not to believe it.

One judgement: :func:`context_reject` -- a fix that is small
(nb <= ar_context_nb_max) in a burst-like context (elevated DDPR
residuals) is more likely wrong than lucky. :func:`validate_fix` is
its epoch-flow wrapper.

These are policy, not resolution: nothing here touches LAMBDA's inputs.
"""




def context_reject(tc, nb):
    """Reject fragile AR fixes in burst-like contexts before hold/anchor."""
    nb = int(nb)
    if nb <= 0:
        return False, None
    main_res = float(tc._cached_ddpr_res_pre
                     or tc._mres_signals.last_res or 0.0)
    per_sat = tc._mres_signals.per_sat or {}
    worst_res = float(max(per_sat.values())) if per_sat else 0.0
    burst_like = False
    if main_res > float(tc.cfg.ar_context_main_ddpr_max):
        burst_like = True
    if worst_res > float(tc.cfg.ar_context_worst_sat_max):
        burst_like = True

    if burst_like and nb <= int(tc.cfg.ar_context_nb_max):
        return True, {
            'nb': nb,
            'main_ddpr_res': main_res,
            'worst_sat_res': worst_res,
        }
    return False, None


def validate_fix(tc, nb):
    """Phase C — ar_context_reject on the accepted integers."""
    reject_ctx, reject_detail = context_reject(tc, nb)
    if reject_ctx:
        tc._ar_context_reject = reject_detail
        tc.ar_diag.outcome = 'ar_context_reject'
        return False
    tc._ar_context_reject = None
    tc.ar_diag.outcome = 'success'

    return True


