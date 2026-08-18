"""Gates around a LAMBDA answer: when not to ask, and when not to believe.

Three related judgements, all reading the same context (post-fit DDPR
residuals, CP-hold and ddpr-bad streaks):

* :func:`should_skip_ar_precheck` -- do not even attempt AR this epoch;
* :func:`validate_fix` -- RTKLIB valpos on the fixed solution, then the
  graph-objective delta test, then :func:`context_reject`;
* :func:`context_reject` -- a fix that is small (nb <= ar_context_nb_max)
  in a burst-like context is more likely wrong than lucky.

These are policy, not resolution: nothing here touches LAMBDA's inputs.
"""

import numpy as np
import gtsam

from ..validation import residuals as _tc_residuals


def should_skip_ar_precheck(tc):
    """Pre-AR fast skip — same context as ``_ar_context_reject`` minus"""
    main_res = float(tc._cached_ddpr_res_pre
                     or tc._last_main_ddpr_res or 0.0)
    per_sat = tc._last_main_ddpr_per_sat or {}
    worst_res = float(max(per_sat.values())) if per_sat else 0.0
    cp_hold_active = int(tc._recov_cp_hold or 0) > 0
    ddpr_bad_active = int(tc._ddpr_bad_count or 0) > 0

    skip = False
    if (bool(tc.cfg.ar_context_reject_during_cp_hold)
            and cp_hold_active):
        skip = True
    if (bool(tc.cfg.ar_context_reject_during_ddpr_bad)
            and ddpr_bad_active):
        skip = True
    if main_res > float(tc.cfg.ar_context_main_ddpr_max):
        skip = True
    if worst_res > float(tc.cfg.ar_context_worst_sat_max):
        skip = True
    if not skip:
        return False, None
    return True, {
        'main_ddpr_res': main_res,
        'worst_sat_res': worst_res,
        'cp_hold_active': cp_hold_active,
        'ddpr_bad_active': ddpr_bad_active,
    }


def context_reject(tc, nb):
    """Reject fragile AR fixes in burst-like contexts before hold/anchor."""
    nb = int(nb)
    if nb <= 0:
        return False, None
    main_res = float(tc._cached_ddpr_res_pre
                     or tc._last_main_ddpr_res or 0.0)
    per_sat = tc._last_main_ddpr_per_sat or {}
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
                  estimate=None, key_pose=None):
    """Phase C — RTKLIB-style post-fit valpos + project-specific ar_context_reject. Returns True when the fix is accepted, False when either gate trips."""
    # xa[0:3] is already antenna position (nav.x[0:3] was set to antenna pos)
    fix_antenna = xa[0:3]

    yu, eu, _ = tc.zdres(obs, None, None, rs, vs, dts, fix_antenna)
    v_fix, _, R_fix = tc.sdres(obs, xa, yu[iu], eu[iu], sat, el)

    if not tc.valpos(v_fix, R_fix):
        tc._last_ar_outcome = 'valpos_failed'
        return False

    # Likelihood-ratio gate in the graph's OWN objective (pre-hold):
    # Δres = DDPR RMS with the pose moved to the fixed solution xa,
    # minus the same RMS at the float solution. A wrong-integer basin
    # is phase-self-consistent but the epoch's code factors protest —
    # the DELTA isolates that protest from the NLOS noise floor that
    # defeats absolute thresholds. Evaluated BEFORE fix-and-hold, so a
    # wrong basin is rejected before holds can lock it (once holds drag
    # the float into the basin the delta vanishes — timing matters).
    dres_thr = float(getattr(tc.cfg, 'ar_fix_dres_max', 0.0) or 0.0)
    ed = getattr(tc, '_cur_ed', None)
    if dres_thr > 0.0 and ed is not None and estimate is not None \
            and key_pose is not None:
        res_pre = tc._cached_ddpr_res_pre
        res_xa = None
        try:
            cur_pose = estimate.atPose3(key_pose)
            R_be = tc.ecef_T_nav.compose(cur_pose).rotation().matrix()
            lever_arr = (np.array(tc.lever_arm_tc)
                         if getattr(tc, 'lever_arm_tc', None) is not None
                         else np.zeros(3))
            body_xa = np.asarray(xa[0:3], dtype=float) - R_be @ lever_arr
            body_nav = tc.ecef_T_nav.transformTo(gtsam.Point3(*body_xa))
            v_xa = gtsam.Values()
            v_xa.insert(key_pose,
                        gtsam.Pose3(cur_pose.rotation(), body_nav))
            res_xa, _ = _tc_residuals.main_ddpr_residuals(tc, ed.graph, v_xa)
        except (RuntimeError, ValueError, IndexError):
            res_xa = None
        if res_pre is not None and res_xa is not None:
            tc._last_fix_dres = float(res_xa) - float(res_pre)
            if tc._last_fix_dres > dres_thr:
                tc._last_ar_outcome = 'fix_dres'
                return False

    reject_ctx, reject_detail = context_reject(tc, nb)
    if reject_ctx:
        tc._ar_context_reject = reject_detail
        tc._last_ar_outcome = 'ar_context_reject'
        return False
    tc._ar_context_reject = None
    tc._last_ar_outcome = 'success'

    return True


