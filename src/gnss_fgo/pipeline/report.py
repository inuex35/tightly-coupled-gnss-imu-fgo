"""Stage E — emit final tuple + bookkeeping.

Pure formatting / diagnostics; no accuracy policy here. All tuning
decisions live in Stages B (gate) and D (postprocess).
"""

import numpy as np

from ..integrity import recovery as _tc_recovery


# ── Phase-2 pipeline contract (see stage_contract.py) ──────────────
STAGE_READS = ('R_enu2ecef', 'estimate', 'info', 'key_idx', 'nb', 'obs', 'sol', 'tag')
STAGE_WRITES = ()


def run(tc, epoch):
    """Stage E: emit final (sol, tag, nb, info) tuple and book-keep.

    Pure formatting / diagnostics — no accuracy policy here. All
    tuning decisions are made in Stages B (gate) and D (postprocess).
    """
    info = epoch.info
    # Diagnostic: FLS re-optimization drift of pose(key_idx-1)
    try:
        if epoch.key_idx > 0:
            prev_pose_now = epoch.estimate.atPose3(tc.Xpose(epoch.key_idx - 1))
            prev_enu_now = np.array(prev_pose_now.translation())
            prev_ecef_now = epoch.R_enu2ecef @ prev_enu_now + tc.base_ecef
            prev_ant_now = tc._antenna_ecef(
                prev_pose_now, prev_ecef_now)
            last_sol = tc._last_sol_ecef
            if last_sol is not None:
                info['prev_pose_drift'] = float(
                    np.linalg.norm(prev_ant_now - last_sol))
    except RuntimeError:
        pass
    tc._last_sol_ecef = np.array(epoch.sol)
    info['bias_acc'] = tc.tc_bias.accelerometer()
    info['bias_gyro'] = tc.tc_bias.gyroscope()
    if tc.cfg.fls_update_timing:
        info['fls_update_ms'] = float(
            tc._fls_update_last_ms)
    tc._last_per_sat_res = info.get('main_ddpr_per_sat', {})
    return _tc_recovery.advance_epoch_and_pack(tc, 
        epoch.sol, epoch.tag, epoch.nb, info, epoch.obs)
