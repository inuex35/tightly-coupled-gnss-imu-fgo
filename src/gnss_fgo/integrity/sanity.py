"""DDPR sanity escalation ladder — the wrong-basin recovery policy.

Stateful escalation over consecutive epochs (trigger -> persist ->
diagnostic anchor -> reset). Lives next to recovery.py because
every rung ends in a recovery action; validation/residuals.py stays a
pure residual-computation library.
"""

import numpy as np
import gtsam

from . import recovery as _tc_recovery
from ..pipeline import residuals as _tc_residuals
from . import recovery as _tc_state


def _ddpr_multipath_dominated(tc, info):
    """Return True when the per-sat DDPR residual distribution looks"""
    ratio_thr = float(tc.cfg.sanity_max_median_ratio)
    if ratio_thr <= 0:
        return False
    per_sat = info.get('main_ddpr_per_sat') or {}
    n = len(per_sat)
    if n < int(tc.cfg.sanity_max_median_min_sats):
        return False
    vals = sorted(float(v) for v in per_sat.values())
    median = vals[n // 2]
    max_v = vals[-1]
    if median <= 1e-3:
        return False
    ratio = max_v / median
    if ratio > ratio_thr:
        info['sanity_skipped_multipath_ratio'] = ratio
        return True
    return False


def run_ddpr_sanity(tc, graph, estimate, pose_tc, ecef_tc, pred, obs, obsb, obs_sd,
                     rs, rsb, sat, el, iu, ir_map, key_idx, info, nb=0):
    """Trigger warm reset when main-graph DDPR residuals say TC pose is"""
    main_res = info.get('main_ddpr_res', 0.0)
    if not _ddpr_sanity_trigger(tc, main_res, info):
        return None
    if _ddpr_multipath_dominated(tc, info):
        return None
    pred_res = _compute_res_at_pred(tc, graph, pred, key_idx, info)
    fast = _ddpr_sanity_fast_path(
        tc, main_res, pose_tc, pred, pred_res, obs, info, nb=nb)
    if fast is not None:
        return fast
    if not _ddpr_sanity_persist(tc, main_res, info):
        return None
    _ddpr_sanity_anchor_diagnostics(
        tc, obs, obsb, obs_sd, rs, rsb, sat, el, iu, ir_map,
        pose_tc, ecef_tc, pred, info)
    return _apply_sanity_reset(tc, pose_tc, pred, pred_res, info, obs)


def _compute_res_at_pred(tc, graph, pred, key_idx, info):
    """DDPR residual evaluated at the IMU-predicted pose."""
    try:
        v_pred = gtsam.Values()
        v_pred.insert(tc.Xpose(key_idx), pred.pose())
        res, _ = _tc_residuals.main_ddpr_residuals(tc, graph, v_pred)
        info['ddpr_res_at_pred'] = res
        return float(res)
    except RuntimeError:
        return float('inf')


def _sanity_report_translation(tc, pose_tc, pred, pred_res, info):
    """Pose translation to report when sanity recovery fires."""
    tc_t = np.array(pose_tc.translation())
    thr = float(tc.cfg.sanity_pose_replace_thresh)
    if thr <= 0 or pred is None:
        return tc_t
    if pred_res is None or pred_res > thr:
        info['sanity_pose_replace_pred_dirty'] = (
            pred_res if pred_res is not None else -1.0)
        return tc_t
    try:
        pred_t = np.array(pred.pose().translation())
    except (RuntimeError, AttributeError):
        return tc_t
    gap = float(np.linalg.norm(tc_t - pred_t))
    info['sanity_pose_gap'] = gap
    if gap > thr:
        info['sanity_pose_replaced'] = 1
        return pred_t
    return tc_t


def _apply_sanity_reset(tc, pose_tc, pred, pred_res, info, obs):
    """Sanity-recovery graph surgery shared between the normal"""
    info['ddpr_recover'] = tc._ddpr_bad_count
    n_removed = _tc_recovery.reset_ambiguities_with_cp_hold(tc)
    info['sanity_dd_removed'] = n_removed
    tc._ddpr_bad_count = 0
    if int(tc.cfg.sanity_break_pim):
        tc._pim_discontinuity = True
    report_t = _sanity_report_translation(tc, pose_tc, pred, pred_res, info)
    ecef_tc_now = tc.R_enu2ecef @ report_t + tc.base_ecef
    return _tc_recovery.advance_epoch_and_pack(tc, ecef_tc_now, 'FLT', 0, info, obs)


def _ddpr_sanity_fast_path(tc, main_res, pose_tc, pred, pred_res, obs, info, nb=0):
    """Fast path for catastrophic residual spikes."""
    if main_res <= tc.cfg.main_ddpr_res_catastrophic:
        return None
    if int(nb) > 0:
        return None
    worst_sat_res = 0.0
    worst_pair = info.get('main_ddpr_sat_worst')
    if worst_pair is not None:
        try:
            _, worst_sat_res = worst_pair
            worst_sat_res = float(worst_sat_res)
        except (ValueError, TypeError):
            worst_sat_res = 0.0
    if worst_sat_res < float(tc.cfg.ddpr_fast_worst_sat_min):
        return None
    info['ddpr_bad'] = tc._ddpr_bad_count + 1
    info['ddpr_fast_recover'] = main_res
    info['ddpr_fast_worst_sat_res'] = worst_sat_res
    n_removed = _tc_recovery.reset_ambiguities_with_cp_hold(tc)
    info['sanity_dd_removed'] = n_removed
    tc._ddpr_bad_count = 0
    if int(tc.cfg.sanity_break_pim):
        tc._pim_discontinuity = True
    report_t = _sanity_report_translation(tc, pose_tc, pred, pred_res, info)
    ecef_tc_now = tc.R_enu2ecef @ report_t + tc.base_ecef
    return _tc_recovery.advance_epoch_and_pack(tc, ecef_tc_now, 'FLT', 0, info, obs)


def _ddpr_sanity_trigger(tc, main_res, info):
    """Escalation step 1: clean residual signal → reset bad-count, return False."""
    rms_bad = main_res > tc.cfg.main_ddpr_res_thresh
    if not rms_bad:
        tc._ddpr_bad_count = 0
        return False
    return True


def _ddpr_sanity_persist(tc, main_res, info):
    """Escalation step 2: count consecutive bad epochs, fire CP-hold each one,"""
    tc._ddpr_bad_count = tc._ddpr_bad_count + 1
    info['ddpr_bad'] = tc._ddpr_bad_count
    _tc_state.trigger_cp_hold(tc, 'ddpr_main_res', info, value=main_res)
    return tc._ddpr_bad_count >= tc.cfg.ddpr_sanity_persist


def _ddpr_sanity_anchor_diagnostics(tc, obs, obsb, obs_sd, rs, rsb, sat, el,
                                    iu, ir_map, pose_tc, ecef_tc, pred, info):
    """Forensics for the reset that is about to happen (no decision).

    The original ladder ran this DDPR-only LS anchor as escalation
    steps 3-5, but every rung ended in the same _apply_sanity_reset —
    the anchor result never steered the action, in any public revision.
    It is now explicitly diagnostics-only: the anchor position, its
    innovation against the TC pose, and its gap to the IMU prediction
    land in info for post-run analysis. diag_sanity_anchor=0 skips the
    extra solve.
    """
    if not tc.cfg.diag_sanity_anchor:
        return
    ecef_ddpr, n_ddpr, res_rms = tc._ddpr_only_position(
        obs, obsb, obs_sd, rs, rsb, sat, el, iu, ir_map, pose_tc)
    info['ddpr_nv'] = n_ddpr
    info['ddpr_res'] = res_rms
    if ecef_ddpr is not None:
        info['ecef_ddpr'] = ecef_ddpr
        info['ddpr_innov'] = float(np.linalg.norm(ecef_tc - ecef_ddpr))
        ecef_pred = (tc.R_enu2ecef @ np.array(pred.pose().translation())
                     + tc.base_ecef)
        info['anchor_imu_gap'] = float(np.linalg.norm(ecef_ddpr - ecef_pred))
    if ecef_ddpr is None or res_rms > tc.cfg.ddpr_max_res:
        info['ddpr_anchor_untrusted'] = res_rms
