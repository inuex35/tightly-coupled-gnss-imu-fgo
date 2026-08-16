"""Fix-and-hold: turn an accepted fix into priors the graph keeps.

RTKLIB's armode==3, adapted to the two-smoother layout: Phase 1 holds each
ambiguity with a PriorDouble at sigma = sqrt(varholdamb); Phase 2 anchors the
pose at the fixed antenna position (optional) and gates the whole hold batch
with the GICI-style post-AR cost test -- if the post-fit DDPR RMS rises by
more than ``post_ar_cost_thresh`` after the holds go in, the batch is
removed again. Accepted Phase-2 holds are then copied onto the per-satellite
hold state and ``nav.x`` (that value re-enters AR as a pinned input via
ar_problem, and write_marginals mirrors it into ``nav.P``).

Entry point: :func:`apply_fix_and_hold`.
"""

import numpy as np
import gtsam

from ..utils import sorted_amb_items
from ..validation import residuals as _tc_postfit


def _collect_held_sat_freq_keys(tc, amb_dict):
    """Return [(s, f), ...] for ambiguities that holdamb_flags() just promoted to fix=3."""
    return [(s, f) for (s, f), _k in sorted_amb_items(amb_dict)
            if tc.nav.fix[s - 1, f] == 3]


def _add_phase1_hold_priors(tc, hg, hold_keys, amb_dict, xa):
    """Add per-held-N PriorDouble factors with σ = √varholdamb (RTKLIB-style, in cycles)."""
    hold_sigma = float(np.sqrt(tc.cfg.varholdamb))
    hold_noise = tc._noise1(hold_sigma)
    for s, f in hold_keys:
        hg.addPriorDouble(
            amb_dict[(s, f)], xa[tc.IB(s, f, tc.nav.na)], hold_noise)


def _add_fix_pose_anchor_factor(tc, hg, estimate, key_pose, xa):
    """Add a PriorPose3 at the LAMBDA-fixed antenna position; skipped (returns False) on any failure."""
    anchor_sigma = float(tc.cfg.fix_pose_anchor_sigma)
    try:
        cur_pose = estimate.atPose3(key_pose)
        R_body_to_ecef = tc.ecef_T_nav.compose(cur_pose).rotation().matrix()
        lever_arr = (np.array(tc.lever_arm_tc)
                     if getattr(tc, 'lever_arm_tc', None) is not None
                     else np.zeros(3))
        body_ecef_target = xa[0:3] - R_body_to_ecef @ lever_arr
        body_nav_target = tc.ecef_T_nav.transformTo(
            gtsam.Point3(*body_ecef_target))
        target_pose = gtsam.Pose3(cur_pose.rotation(), body_nav_target)
        # 1e6 rad on rotation = unconstrained; translation σ = anchor_sigma m.
        sigmas = np.array([1e6, 1e6, 1e6,
                           anchor_sigma, anchor_sigma, anchor_sigma])
        anchor_noise = gtsam.noiseModel.Diagonal.Sigmas(sigmas)
        hg.addPriorPose3(key_pose, target_pose, anchor_noise)
        return True
    except RuntimeError:
        return False


def _apply_holds_phase2_with_gate(tc, hg, key_pose, anchor_added):
    """Phase 2: ISAM2.update with the GICI-style post-AR cost gate."""
    isam = tc.isam2
    full_graph = isam.getFactors()
    res_pre = tc._cached_ddpr_res_pre
    ts_h = gtsam.FixedLagSmootherKeyTimestampMap()
    if anchor_added:
        ts_h[key_pose] = tc.tc_time
    base_idx_undo = tc.total_factor_count
    n_added = hg.size()
    try:
        isam.update(hg, gtsam.Values(), ts_h)
        tc.total_factor_count += n_added
        res_post = None
        if res_pre is not None:
            try:
                est_post = isam.calculateEstimate()
                res_post, _ = _tc_postfit.main_ddpr_residuals(
                    tc, full_graph, est_post)
            except (RuntimeError, IndexError):
                res_post = None
        if (res_pre is not None and res_post is not None
                and (res_post - res_pre) > tc.cfg.post_ar_cost_thresh):
            # Reject: remove hold-prior factors.
            try:
                isam.update(
                    gtsam.NonlinearFactorGraph(),
                    gtsam.Values(),
                    gtsam.FixedLagSmootherKeyTimestampMap(),
                    list(range(base_idx_undo, base_idx_undo + n_added)))
            except (RuntimeError, IndexError):
                pass
            return False
    except (RuntimeError, IndexError):
        pass
    return True


def _apply_holds_phase1(tc, hg, hold_keys, amb_dict):
    """Phase 1: IncrementalFixedLagSmoother.update with held-N timestamps + amb_factor_indices tracking."""
    isam = tc.isam
    ts_h1 = gtsam.FixedLagSmootherKeyTimestampMap()
    t_p1 = getattr(tc, 'phase1_t', 0.0)
    for sf in hold_keys:
        ts_h1[amb_dict[sf]] = t_p1
    try:
        isam.update(hg, gtsam.Values(), ts_h1)
        base_idx = tc.total_factor_count
        for i, key_id in enumerate(hold_keys):
            tc._sat_states.get(*key_id).amb_factor_indices.append(
                base_idx + i)
        tc.total_factor_count += hg.size()
    except (RuntimeError, IndexError):
        pass


def _activate_phase2_hold_states(tc, hold_keys, xa):
    """Phase 2: copy held N → sat_state hold + nav.x; clear amb_key / amb_factor_indices."""
    for s, f in hold_keys:
        held_value = float(xa[tc.IB(s, f, tc.nav.na)])
        sat_st = tc._sat_states.get(s, f)
        sat_st.activate_hold(held_value)
        sat_st.amb_key = None
        sat_st.amb_factor_indices = []
        tc.nav.x[tc.IB(s, f, tc.nav.na)] = held_value


def apply_fix_and_hold(tc, estimate, key_pose, amb_dict, xa):
    """Phase D — fix-and-hold (armode==3): mark held flags, build hold-prior factors, optional fix_pose_anchor, run ISAM2.update with post-AR cost gate, then activate hold on sat_states. Returns True on accept, False when the post-AR cost gate rejects the fix."""
    tc.holdamb_flags()
    hold_keys = _collect_held_sat_freq_keys(tc, amb_dict)
    hg = gtsam.NonlinearFactorGraph()
    if tc.phase != 2:
        _add_phase1_hold_priors(tc, hg, hold_keys, amb_dict, xa)
    anchor_added = False
    if tc.phase == 2 and float(tc.cfg.fix_pose_anchor_sigma) > 0:
        anchor_added = _add_fix_pose_anchor_factor(
            tc, hg, estimate, key_pose, xa)
    if hg.size() > 0:
        if tc.phase == 2:
            if not _apply_holds_phase2_with_gate(
                    tc, hg, key_pose, anchor_added):
                return False
        else:
            _apply_holds_phase1(tc, hg, hold_keys, amb_dict)
    if tc.phase == 2:
        _activate_phase2_hold_states(tc, hold_keys, xa)
    return True


