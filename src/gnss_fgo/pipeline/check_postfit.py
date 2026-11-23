"""Stage C3 — extract solved pose/bias and record diagnostic DDPR residuals."""

import numpy as np

from ..utils import heading_from_pose
from . import residuals as _tc_residuals


def _compute_postfit_diagnostics(tc, epoch):
    """Stage C3 — snapshot solved pose/bias and record diagnostic residuals (no manual FDE)."""
    info = epoch.info
    main_res, per_sat_res, pair_rows = _tc_residuals.main_ddpr_residuals(
        tc, epoch.graph, epoch.estimate, with_pairs=True)
    info['main_ddpr_res'] = main_res
    info['main_ddpr_per_sat'] = per_sat_res
    info['main_ddpr_pairs'] = pair_rows
    info['ref_sats'] = dict(tc.current_epoch.ref_sats)
    if per_sat_res:
        worst_sat = max(per_sat_res, key=per_sat_res.get)
        info['main_ddpr_sat_worst'] = (worst_sat, per_sat_res[worst_sat])

    epoch.pose_tc = epoch.estimate.atPose3(tc.Xpose(epoch.key_idx))
    info['post_heading_deg'] = heading_from_pose(epoch.pose_tc)
    tc.tc_bias = epoch.estimate.atConstantBias(tc.Bias(epoch.key_idx))
    enu_tc = np.array(epoch.pose_tc.translation())
    epoch.ecef_tc = epoch.R_enu2ecef @ enu_tc + tc.base_ecef
