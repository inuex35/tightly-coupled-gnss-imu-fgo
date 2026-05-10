"""Stage E — emit final tuple + bookkeeping.

Pure formatting / diagnostics; no accuracy policy here. All tuning
decisions live in Stages B (gate) and D (postprocess).
"""

import numpy as np

from . import recovery as _tc_recovery


# ── Phase-2 pipeline contract (see stage_contract.py) ──────────────
STAGE_READS = ('R', 'est2', 'info', 'kk', 'nb', 'obs', 'sol', 'tag')
STAGE_WRITES = ()


def run(tc, ed):
    """Stage E: emit final (sol, tag, nb, info) tuple and book-keep.

    Pure formatting / diagnostics — no accuracy policy here. All
    tuning decisions are made in Stages B (gate) and D (postprocess).
    """
    info = ed.info
    # Diagnostic: FLS re-optimization drift of pose(kk-1)
    try:
        if ed.kk > 0:
            prev_pose_now = ed.est2.atPose3(tc.Xpose(ed.kk - 1))
            prev_enu_now = np.array(prev_pose_now.translation())
            prev_ecef_now = ed.R @ prev_enu_now + tc.base_ecef
            prev_ant_now = tc._antenna_ecef(
                prev_pose_now, prev_ecef_now)
            last_sol = tc._last_sol_ecef
            if last_sol is not None:
                info['prev_pose_drift'] = float(
                    np.linalg.norm(prev_ant_now - last_sol))
    except RuntimeError:
        pass
    tc._last_sol_ecef = np.array(ed.sol)
    info['bias_acc'] = tc.tc_bias.accelerometer()
    info['bias_gyro'] = tc.tc_bias.gyroscope()
    if tc.cfg.fls_update_timing:
        info['fls_update_ms'] = float(
            tc._fls_update_last_ms)
    tc._last_per_sat_res = info.get('main_ddpr_per_sat', {})
    return _tc_recovery.finalize_epoch(tc, 
        ed.sol, ed.tag, ed.nb, info, ed.obs)
