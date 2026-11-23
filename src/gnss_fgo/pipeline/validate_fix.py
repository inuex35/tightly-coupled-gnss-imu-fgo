"""Stage D — emit FIX or FLT solution from LAMBDA result (no heuristic post-gates)."""

import numpy as np


STAGE_READS = (
    'R_enu2ecef', 'ecef_tc', 'info', 'nb', 'pose_tc', 'pred_nav', 'xa',
)
STAGE_WRITES = (
    'nb', 'sol', 'tag',
)


def _decide_fix_or_flt(tc, epoch):
    """Pure FIX/FLT decision directly from LAMBDA acceptance (nav.smode == 4)."""
    info = epoch.info
    pose_tc_antenna = tc._antenna_ecef(epoch.pose_tc, epoch.ecef_tc)
    if tc.nav.smode == 4 and epoch.nb > 0 and epoch.xa is not None:
        lc = float(np.linalg.norm(epoch.xa[0:3] - pose_tc_antenna))
        info['lambda_correction'] = lc
        return epoch.xa[0:3], 'FIX', epoch.nb
    return pose_tc_antenna, 'FLT', epoch.nb


def run(tc, epoch):
    """Stage D: record innovation diagnostic and select FIX/FLT output."""
    pred_ecef = (epoch.R_enu2ecef @ np.array(epoch.pred_nav.pose().translation())
                 + tc.base_ecef)
    epoch.info['innovation'] = float(np.linalg.norm(epoch.ecef_tc - pred_ecef))
    epoch.sol, epoch.tag, epoch.nb = _decide_fix_or_flt(tc, epoch)
    return None
