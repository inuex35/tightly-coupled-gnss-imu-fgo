"""Stage C3 -- post-fit diagnostics on the solved epoch.

Main DDPR residuals, per-satellite bookkeeping (persist-bad holds,
observation quality), the post-fit FDE re-solve, and the pose snapshot the
output stage reports.
"""


import numpy as np
import gtsam

from ..utils import heading_from_pose
from ..pipeline import residuals as _tc_residuals


def _compute_postfit_diagnostics(tc, epoch):
    """Stage C3 — main DDPR + factor-residual diagnostics, post-fit FDE re-solve, and pose snapshot."""
    info = epoch.info
    if tc.cfg.diag_main_ddpr_res:
        main_res_pre_fde, per_sat_res, pair_rows = _tc_residuals.main_ddpr_residuals(tc, 
            epoch.graph, epoch.estimate, with_pairs=True)
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
        all_res = _tc_residuals.all_factor_residuals(tc, epoch.graph, epoch.estimate)
        for tag, (rms, n) in all_res.items():
            info[f'fres_{tag}'] = rms
            info[f'fcnt_{tag}'] = n

    if per_sat_res:
        worst_sat = max(per_sat_res, key=per_sat_res.get)
        info['main_ddpr_sat_worst'] = (worst_sat,
                                         per_sat_res[worst_sat])
    if tc.cfg.fde_enable:
        epoch.estimate = _tc_residuals.apply_fde(tc, 
            epoch.graph, epoch.key_idx, epoch.nv, epoch.estimate, info)

    # Pose after FDE re-solve
    epoch.pose_tc = epoch.estimate.atPose3(tc.Xpose(epoch.key_idx))
    info['post_heading_deg'] = heading_from_pose(epoch.pose_tc)
    tc.tc_bias = epoch.estimate.atConstantBias(tc.Bias(epoch.key_idx))
    enu_tc = np.array(epoch.pose_tc.translation())
    epoch.ecef_tc = epoch.R_enu2ecef @ enu_tc + tc.base_ecef

    ref = getattr(epoch, 'ref_ecef', None)
    if ref is not None and tc.cfg.diag_truth_residual:
        try:
            R_e2n = tc.R_enu2ecef.T
            lever_arr = np.array(tc.lever_arm_tc) \
                if getattr(tc, 'lever_arm_tc', None) is not None \
                else np.zeros(3)
            R_body = np.array(
                tc.ecef_T_nav.compose(epoch.pose_tc).rotation().matrix())
            truth_body_ecef = np.asarray(ref) - R_body @ lever_arr
            truth_body_enu = R_e2n @ (truth_body_ecef - tc.base_ecef)
            v_truth = gtsam.Values()
            v_truth.insert(tc.Xpose(epoch.key_idx),
                           gtsam.Pose3(epoch.pose_tc.rotation(),
                                        gtsam.Point3(*truth_body_enu)))
            truth_res, truth_per_sat, truth_pair_rows = _tc_residuals.main_ddpr_residuals(tc, 
                epoch.graph, v_truth, with_pairs=True)
            info['ddpr_res_at_truth'] = float(truth_res)
            info['ddpr_per_sat_at_truth'] = (
                dict(truth_per_sat) if truth_per_sat else {}
            )
            info['ddpr_pairs_at_truth'] = truth_pair_rows
            info['truth_offset'] = float(np.linalg.norm(
                np.array(epoch.pose_tc.translation())
                - np.asarray(truth_body_enu)))
        except (RuntimeError, ValueError):
            pass
