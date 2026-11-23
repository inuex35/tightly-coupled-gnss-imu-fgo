"""Stage D — select FIX/FLT solution and emit final tuple + telemetry."""

import numpy as np

from ..integrity import recovery as _tc_recovery


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
    """Stage D: select FIX/FLT output and emit final (sol, tag, nb, info) tuple."""
    info = epoch.info
    epoch.sol, epoch.tag, epoch.nb = _decide_fix_or_flt(tc, epoch)
    info['bias_acc'] = tc.tc_bias.accelerometer()
    info['bias_gyro'] = tc.tc_bias.gyroscope()
    return _tc_recovery.advance_epoch_and_pack(
        tc, epoch.sol, epoch.tag, epoch.nb, info, epoch.obs)
