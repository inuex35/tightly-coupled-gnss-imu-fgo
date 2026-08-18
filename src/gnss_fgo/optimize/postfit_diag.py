"""Stage C4 -- post-fit diagnostics on the solved epoch.

Main DDPR residuals, per-satellite bookkeeping (persist-bad holds,
observation quality), the post-fit FDE re-solve, and the pose snapshot the
output stage reports.
"""


import numpy as np
import gtsam

from .. import sat_quality as _satq
from ..utils import heading_from_pose
from ..validation import residuals as _tc_residuals


def _compute_postfit_diagnostics(tc, ed):
    """Stage C4 — main DDPR + factor-residual diagnostics, persist-bad / observation-quality bookkeeping, post-fit FDE re-solve, and pose snapshot."""
    info = ed.info
    sq = _satq.get_sat_quality(tc)
    if tc.cfg.diag_main_ddpr_res:
        main_res_pre_fde, per_sat_res, pair_rows = _tc_residuals.main_ddpr_residuals(tc, 
            ed.graph, ed.estimate, with_pairs=True)
        info['main_ddpr_res'] = main_res_pre_fde
        info['main_ddpr_per_sat'] = per_sat_res
        info['main_ddpr_pairs'] = pair_rows
        info['ref_sats'] = dict(getattr(tc, 'ref_sats', {}) or {})
        tc._cached_ddpr_res_pre = main_res_pre_fde
        tc._mres_signals.update(
            last_res=main_res_pre_fde,
            per_sat=dict(per_sat_res) if per_sat_res else {},
            epoch=int(tc.epoch))
    else:
        main_res_pre_fde = 0.0
        per_sat_res = {}
        tc._cached_ddpr_res_pre = None
        tc._mres_signals.reset()
    if tc.cfg.diag_factor_residuals:
        all_res = _tc_residuals.all_factor_residuals(tc, ed.graph, ed.estimate)
        for tag, (rms, n) in all_res.items():
            info[f'fres_{tag}'] = rms
            info[f'fcnt_{tag}'] = n

    if per_sat_res:
        worst_sat = max(per_sat_res, key=per_sat_res.get)
        info['main_ddpr_sat_worst'] = (worst_sat,
                                         per_sat_res[worst_sat])
    if tc.cfg.ar_persist_bad_enable and getattr(tc, 'phase', 1) >= 2:
        sq = _satq.get_sat_quality(tc)
        thr = float(tc.cfg.ar_persist_bad_res_thresh)
        streak_need = max(1, int(tc.cfg.ar_persist_bad_streak))
        hold_len = max(1, int(tc.cfg.ar_persist_bad_hold))
        seen = set()
        for s, rmax in (per_sat_res or {}).items():
            s = int(s)
            seen.add(s)
            if rmax > thr:
                st = sq.persist_bad_streak.get(s, 0) + 1
                sq.persist_bad_streak[s] = st
                if st >= streak_need:
                    sq.persist_bad_hold[s] = max(
                        int(sq.persist_bad_hold.get(s, 0)), hold_len)
                    for f in range(tc.nav.nf):
                        key = (s, f)
                        sat_st = tc._sat_states.get(*key)
                        sat_st.amb_gen += 1
                        sat_st.rejc_cp_pr = 0
                        sat_st.fix_streak = 0
            else:
                sq.persist_bad_streak[s] = 0
        for s in list(sq.persist_bad_streak.keys()):
            if s not in seen:
                sq.persist_bad_streak[s] = 0

    if getattr(tc, 'phase', 1) >= 2:
        worst_sat_id = int(worst_sat) if per_sat_res else None
        cppr_sat = info.get('sat_cppr_sat', {}) or {}
        sq = _satq.get_sat_quality(tc)
        sq.update_observation_quality(
            tc.cfg, per_sat_res, worst_sat=worst_sat_id, cppr_sat=cppr_sat,
            sat_el_deg=info.get('sat_el_deg'),
            sat_snr_dbhz=info.get('sat_snr_dbhz'))

    if tc.cfg.fde_enable:
        ed.estimate = _tc_residuals.apply_fde(tc, 
            ed.graph, ed.kk, ed.nv, ed.estimate, info)

    # Pose after FDE re-solve
    ed.pose_tc = ed.estimate.atPose3(tc.Xpose(ed.kk))
    info['post_heading_deg'] = heading_from_pose(ed.pose_tc)
    tc.tc_bias = ed.estimate.atConstantBias(tc.Bias(ed.kk))
    enu_tc = np.array(ed.pose_tc.translation())
    ed.ecef_tc = ed.R @ enu_tc + tc.base_ecef

    ref = getattr(ed, 'ref_ecef', None)
    if ref is not None and tc.cfg.diag_truth_residual:
        try:
            R_e2n = tc.R_enu2ecef.T
            lever_arr = np.array(tc.lever_arm_tc) \
                if getattr(tc, 'lever_arm_tc', None) is not None \
                else np.zeros(3)
            R_body = np.array(
                tc.ecef_T_nav.compose(ed.pose_tc).rotation().matrix())
            truth_body_ecef = np.asarray(ref) - R_body @ lever_arr
            truth_body_enu = R_e2n @ (truth_body_ecef - tc.base_ecef)
            v_truth = gtsam.Values()
            v_truth.insert(tc.Xpose(ed.kk),
                           gtsam.Pose3(ed.pose_tc.rotation(),
                                        gtsam.Point3(*truth_body_enu)))
            truth_res, truth_per_sat, truth_pair_rows = _tc_residuals.main_ddpr_residuals(tc, 
                ed.graph, v_truth, with_pairs=True)
            info['ddpr_res_at_truth'] = float(truth_res)
            info['ddpr_per_sat_at_truth'] = (
                dict(truth_per_sat) if truth_per_sat else {}
            )
            info['ddpr_pairs_at_truth'] = truth_pair_rows
            info['truth_offset'] = float(np.linalg.norm(
                np.array(ed.pose_tc.translation())
                - np.asarray(truth_body_enu)))
        except (RuntimeError, ValueError):
            pass
