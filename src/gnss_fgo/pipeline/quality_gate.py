"""Stage B — cycle-slip detection and predicted antenna position."""

import numpy as np

from ..integrity import slip_detect as _tc_slip_detect


def run(tc, epoch):
    """Stage B: detect cycle slips and predict antenna ECEF."""
    info = epoch.info
    info['nsat'] = epoch.ns

    n_reset, _ = _tc_slip_detect.detect_slips_and_reset_ambiguities(
        tc, epoch.obs, epoch.obs_sd, epoch.sat, epoch.iu)
    info['n_slip'] = n_reset

    pred_enu = np.array(epoch.pred_nav.pose().translation())
    pred_body_ecef = epoch.R_enu2ecef @ pred_enu + tc.base_ecef
    epoch.pred_ecef = tc._antenna_ecef(epoch.pred_nav.pose(), pred_body_ecef)
    return None
