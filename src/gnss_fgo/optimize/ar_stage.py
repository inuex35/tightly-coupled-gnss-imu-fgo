"""Stage C3 -- the AR pass over the fresh estimate.

Eligibility (which needs the epoch healthy and the geometry meaningful),
marginals publication, the LAMBDA attempt through gnss_fgo.ar, outcome
diagnostics, and the starvation reset that clears a hopeless ambiguity set.
"""


import numpy as np
import gtsam

from .. import ar as _tc_ar
from ..validation import residuals as _tc_residuals
from .. import recovery as _tc_recovery


_AR_OUTCOME_CODES = {
    'not_called': 0, 'armode_off': 1, 'entered': 2,
    'lambda_exception': 3, 'lambda_zero': 4, 'min_nb_gate': 5,
    'valpos_failed': 6, 'ar_context_reject': 7, 'success': 8,
    'fix_dres': 11, 'gdop_gate': 12,
}

_AR_DIAG_ATTRS = (
    ('_last_orphan_cp_count', 'orphan_cp_count'),
    ('_last_amb_dict_size', 'amb_dict_size'),
    ('_last_held_size', 'held_size'),
    ('_last_cp_visible_size', 'cp_visible_size'),
    ('_last_amb_estimate_missing', 'amb_estimate_missing'),
    ('_last_amb_vsat1', 'amb_vsat1'),
    ('_last_amb_vsat0_young', 'amb_vsat0_young'),
    ('_last_amb_vsat0_held_bad', 'amb_vsat0_held_bad'),
    ('_last_amb_age_median', 'amb_age_median'),
    ('_last_amb_age_min', 'amb_age_min'),
    ('_last_amb_not_in_obs', 'amb_not_in_obs'),
    ('_last_held_not_in_obs', 'held_not_in_obs'),
    ('_last_sat_in_obs_size', 'sat_in_obs_size'),
    ('_last_resamb_raw_nb', 'resamb_raw_nb'),
    ('_last_amb_el_min_deg', 'amb_el_min_deg'),
    ('_last_amb_el_median_deg', 'amb_el_median_deg'),
    ('_last_amb_el_above15', 'amb_el_above15'),
    ('_last_amb_el_above25', 'amb_el_above25'),
)


def _ar_eligibility(tc, epoch):
    """Return True iff this epoch should run LAMBDA AR."""
    if epoch.key_idx < tc.cfg.n_collect + 3:
        return False
    if tc._recov_cp_hold > 0:
        return False
    if tc.ar_max_frac < 0.5:
        max_frac = tc._compute_max_dd_frac(
            epoch.estimate, epoch.obs_sd, epoch.sat, epoch.ns)
        epoch.info['max_frac'] = max_frac
        if max_frac > tc.ar_max_frac:
            return False
    return True


def _run_ar_with_marginals(tc, epoch):
    """Pre-check, write_marginals + per_sat gate + run_ar; populate epoch.nb, epoch.xa, info[ar_skipped*]."""
    info = epoch.info
    if tc.cfg.ar_precheck_skip:
        skip_ar, skip_detail = _tc_ar.should_skip_ar_precheck(tc)
        if skip_ar:
            info['ar_skipped_precheck'] = True
            info.update({f'ar_skipped_{k}': v
                         for k, v in skip_detail.items()})
            return
    tc._cur_ed = epoch                 # for the fix-vs-LS gate in run_ar
    tc.nav.x[0:3] = tc._antenna_ecef(epoch.pose_tc, epoch.ecef_tc)
    amb_snapshot = tc._sat_states.amb_keys_dict()
    _tc_ar.write_marginals(tc,
        tc.isam2.getFactors(), epoch.estimate,
        tc.Xpose(epoch.key_idx), amb_snapshot)
    # AR-only geometry gate (demo5 arthres1 spirit): when the DOP says
    # the geometry cannot support an integer decision, do not attempt
    # AR. At 7 sats / GDOP~10 a 9 m vertical basin costs only ~1.7 m of
    # code residual — no residual test can see it, and the marginal
    # covariance can't be used (it depends on the very holds the gate
    # controls: measured feedback death, fix 46%->0.6%). GDOP is pure
    # geometry: no loop. Measured run1 separation: correct fixes GDOP
    # p50 3.3 / p99 7.6, basin wrong fixes p50 9.5.
    ar_gdop = float(getattr(tc.cfg, 'ar_gdop_max', 0.0) or 0.0)
    gdop_now = float(info.get('gdop', 0.0) or 0.0)
    if ar_gdop > 0.0 and gdop_now > ar_gdop:
        info['ar_gdop_skip'] = True
        tc._last_ar_outcome = 'gdop_gate'
        return
    epoch.nb, epoch.xa = _tc_ar.run_ar(tc,
        epoch.obs, epoch.rs, epoch.vs, epoch.dts,
        epoch.sat, epoch.el, epoch.iu, epoch.estimate,
        tc.Xpose(epoch.key_idx), amb_snapshot)
    xv_thr = float(tc.cfg.ar_ddpr_xvalidate_thresh or 0.0)
    xv_delta = float(tc.cfg.ar_ddpr_xvalidate_delta_thresh or 0.0)
    if (xv_thr > 0.0 or xv_delta > 0.0) and epoch.nb > 0 and epoch.xa is not None:
        try:
            cur_pose = epoch.estimate.atPose3(tc.Xpose(epoch.key_idx))
            R_body_to_ecef = tc.ecef_T_nav.compose(
                cur_pose).rotation().matrix()
            lever_arr = (np.array(tc.lever_arm_tc)
                         if getattr(tc, 'lever_arm_tc', None) is not None
                         else np.zeros(3))
            body_ecef_xa = np.asarray(epoch.xa[0:3]) - R_body_to_ecef @ lever_arr
            body_nav_xa = tc.ecef_T_nav.transformTo(
                gtsam.Point3(*body_ecef_xa))
            xa_pose = gtsam.Pose3(cur_pose.rotation(), body_nav_xa)
            v_xa = gtsam.Values()
            v_xa.insert(tc.Xpose(epoch.key_idx), xa_pose)
            res_xa, _ = _tc_residuals.main_ddpr_residuals(tc, epoch.graph, v_xa)
            info['ar_ddpr_xvalidate_res_at_xa'] = float(res_xa)
            res_pre = tc._cached_ddpr_res_pre
            if res_pre is not None:
                info['ar_ddpr_xvalidate_delta'] = float(res_xa - res_pre)
            reject = False
            if xv_thr > 0.0 and float(res_xa) > xv_thr:
                reject = True
            if (xv_delta > 0.0 and res_pre is not None
                    and float(res_xa) - float(res_pre) > xv_delta):
                reject = True
            if reject:
                info['ar_ddpr_xvalidate_reject'] = True
                epoch.nb = 0
                epoch.xa = None
                tc.nav.smode = 5
        except (RuntimeError, ValueError):
            pass


def _record_ar_diagnostics(tc, info):
    """Copy tc._last_* counters → info + outcome code + ar_context_reject + ar_subset_dbg."""
    for diag_attr, info_key in _AR_DIAG_ATTRS:
        v = getattr(tc, diag_attr, None)
        if v is not None:
            info[info_key] = int(v)
    outcome = tc._last_ar_outcome
    if outcome:
        info['ar_outcome_code'] = _AR_OUTCOME_CODES.get(outcome, -1)
    ar_ctx_reject = tc._ar_context_reject
    if ar_ctx_reject:
        info['ar_context_reject'] = True
        info['ar_context_reject_nb'] = int(ar_ctx_reject.get('nb', 0))
        info['ar_context_reject_main_ddpr_res'] = float(
            ar_ctx_reject.get('main_ddpr_res', 0.0))
        info['ar_context_reject_worst_sat_res'] = float(
            ar_ctx_reject.get('worst_sat_res', 0.0))
    ar_subset_dbg = tc._ar_subset_debug
    if ar_subset_dbg:
        info['ar_subset_candidates'] = int(ar_subset_dbg.get('candidates', 0))
        if ar_subset_dbg.get('used'):
            info['ar_subset_used'] = True
            info['ar_subset_drop_sat'] = int(ar_subset_dbg.get('drop_sat', 0))
            info['ar_subset_nb'] = int(ar_subset_dbg.get('nb', 0))
            info['ar_subset_ratio'] = float(ar_subset_dbg.get('ratio', 0.0))


def _ar_starvation_reset(tc, epoch):
    """Purge cold-biased float arcs after prolonged ratio starvation.

    A float that settled into a biased-but-smooth basin (e.g. after a
    mass hold loss under TDCP-class constraints) keeps every residual
    small and every AR attempt ratio-starved: none of the existing
    alarms (innovation, residual spike) can fire. Detect it as N
    consecutive lambda_zero outcomes while the DDPR residual stays
    quiet, and run the same arc purge the residual-spike recovery uses;
    fresh arcs re-anchor on DDPR and plain ILS re-fixes within epochs.
    """
    n_max = int(tc.cfg.ar_starve_reset)
    if n_max <= 0:
        return
    outcome = tc._last_ar_outcome
    if outcome == 'success':
        tc._ar_starve_streak = 0
        return
    if outcome != 'lambda_zero':
        return
    tc._ar_starve_streak = int(getattr(tc, '_ar_starve_streak', 0) or 0) + 1
    epoch.info['ar_starve_streak'] = tc._ar_starve_streak
    if tc._ar_starve_streak < n_max or tc._recov_cp_hold > 0:
        return
    res_pre = tc._cached_ddpr_res_pre
    if res_pre is not None and float(res_pre) > float(tc.cfg.ar_starve_max_res):
        return                      # NLOS storm — arcs are load-bearing
    n_removed = _tc_recovery.reset_ambiguities_with_cp_hold(tc)
    epoch.info['ar_starve_reset'] = n_removed
    tc._ar_starve_streak = 0


def _run_lambda_ar(tc, epoch):
    """Stage C3 — pre-AR gate + write_marginals + LAMBDA AR + AR-outcome diagnostics. Always returns None."""
    # LAMBDA AR — uses the FDE-cleaned float solution.
    tc.ar_max_frac = tc.cfg.ar_max_frac
    tc.nav.smode = 5
    epoch.nb = 0
    epoch.xa = None
    if not _ar_eligibility(tc, epoch):
        return None
    _run_ar_with_marginals(tc, epoch)
    _ar_starvation_reset(tc, epoch)
    _record_ar_diagnostics(tc, epoch.info)
    return None


