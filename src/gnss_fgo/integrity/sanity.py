"""DDPR sanity escalation ladder — the wrong-basin recovery policy.

Stateful escalation over consecutive epochs (trigger -> persist ->
anchor -> anchor-vs-IMU -> reset). Lives next to recovery.py because
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
    if not _ddpr_sanity_gdop_ok(tc, info):
        return None
    anchor = _ddpr_sanity_fetch_anchor(
        tc, obs, obsb, obs_sd, rs, rsb, sat, el, iu, ir_map,
        pose_tc, ecef_tc, info)
    if anchor is None:
        return _ddpr_sanity_anchor_fallback(
            tc, pose_tc, pred, pred_res, info, obs)
    if not _ddpr_sanity_anchor_vs_imu(tc, anchor, main_res, pred, info):
        return _ddpr_sanity_anchor_fallback(
            tc, pose_tc, pred, pred_res, info, obs)
    return _ddpr_sanity_apply_reset(
        tc, anchor, estimate, pose_tc, pred, pred_res, graph, key_idx, info, obs)


def _ddpr_sanity_gdop_ok(tc, info):
    """Escalation step 4: abort sanity when geometry is too weak to trust the"""
    if tc.cfg.sanity_max_gdop <= 0:
        return True
    cur_gdop = info.get('gdop', 0.0)
    if cur_gdop > tc.cfg.sanity_max_gdop:
        info['sanity_skipped_gdop'] = cur_gdop
        return False
    return True


def _ddpr_sanity_anchor_fallback(tc, pose_tc, pred, pred_res, info, obs):
    """Stages 5/6 fallback: ``_apply_sanity_reset`` is graph surgery and"""
    info['sanity_anchor_fallback'] = 1
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
    per_sat_bad = False
    psat_thr = float(tc.cfg.main_ddpr_per_sat_thresh)
    if psat_thr > 0:
        per_sat = info.get('main_ddpr_per_sat') or {}
        if per_sat:
            per_sat_max = max(per_sat.values())
            per_sat_bad = per_sat_max > psat_thr
            if per_sat_bad:
                info['sanity_trig_per_sat'] = per_sat_max
    if not (rms_bad or per_sat_bad):
        tc._ddpr_bad_count = 0
        return False
    return True


def _ddpr_sanity_persist(tc, main_res, info):
    """Escalation step 2: count consecutive bad epochs, fire CP-hold each one,"""
    tc._ddpr_bad_count = tc._ddpr_bad_count + 1
    info['ddpr_bad'] = tc._ddpr_bad_count
    _tc_state.trigger_cp_hold(tc, 'ddpr_main_res', info, value=main_res)
    return tc._ddpr_bad_count >= tc.cfg.ddpr_sanity_persist


def _ddpr_sanity_fetch_anchor(tc, obs, obsb, obs_sd, rs, rsb, sat, el, iu,
                               ir_map, pose_tc, ecef_tc, info):
    """Escalation step 3: DDPR-only LS anchor. Returns (ecef, res_rms) or None"""
    ecef_ddpr, n_ddpr, res_rms = tc._ddpr_only_position(
        obs, obsb, obs_sd, rs, rsb, sat, el, iu, ir_map, pose_tc)
    info['ddpr_nv'] = n_ddpr
    info['ddpr_res'] = res_rms
    if ecef_ddpr is not None:
        info['ecef_ddpr'] = ecef_ddpr
        info['ddpr_innov'] = float(np.linalg.norm(ecef_tc - ecef_ddpr))
    if ecef_ddpr is None or res_rms > tc.cfg.ddpr_max_res:
        info['ddpr_anchor_untrusted'] = res_rms
        return None
    return (ecef_ddpr, res_rms)


def _ddpr_sanity_anchor_vs_imu(tc, anchor, main_res, pred, info):
    """Escalation step 4: anchor must agree with IMU-predicted position (sub-metre"""
    ecef_ddpr, res_rms = anchor
    R = tc.R_enu2ecef
    ecef_pred = R @ np.array(pred.pose().translation()) + tc.base_ecef
    anchor_imu_gap = float(np.linalg.norm(ecef_ddpr - ecef_pred))
    info['anchor_imu_gap'] = anchor_imu_gap
    clean_anchor = (res_rms < tc.cfg.anchor_imu_clean_res
                    and main_res > tc.cfg.anchor_imu_clean_main_res)
    if (anchor_imu_gap > tc.cfg.anchor_imu_hard_max
            and not clean_anchor):
        info['ddpr_anchor_vs_imu_bad'] = anchor_imu_gap
        return False
    catastrophic = main_res > tc.cfg.main_ddpr_res_catastrophic
    persistent_bad = (tc._ddpr_bad_count
                      >= tc.cfg.ddpr_bad_persist_override
                      and res_rms < tc.cfg.ddpr_clean_res)
    if (anchor_imu_gap > tc.cfg.anchor_imu_max_gap
            and not catastrophic and not persistent_bad):
        info['ddpr_anchor_vs_imu_bad'] = anchor_imu_gap
        return False
    return True


def _ddpr_sanity_apply_reset(tc, anchor, estimate, pose_tc, pred, pred_res,
                              graph, key_idx, info, obs):
    """Stage 5: recover from a wrong-basin lock via DDCP removal + N"""
    return _apply_sanity_reset(tc, pose_tc, pred, pred_res, info, obs)
