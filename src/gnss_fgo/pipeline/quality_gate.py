"""Stage B — geometry telemetry, cycle-slip detection, and predicted antenna position."""

import numpy as np

from ..integrity import slip_detect as _tc_slip_detect


STAGE_READS = (
    'R_enu2ecef', 'el', 'info', 'iu', 'ns', 'obs', 'obs_sd', 'pred_nav',
    'rs', 'sat',
)
STAGE_WRITES = (
    'pred_ecef',
)


def run(tc, epoch):
    """Stage B: record GDOP, detect cycle slips, and predict antenna ECEF."""
    info = epoch.info
    info['gdop'] = tc._compute_gdop(
        epoch.pred_nav, epoch.ns, epoch.rs, epoch.iu, epoch.R_enu2ecef)
    info['nsat'] = epoch.ns

    n_reset, _ = _tc_slip_detect.detect_slips_and_reset_ambiguities(
        tc, epoch.obs, epoch.obs_sd, epoch.sat, epoch.iu)
    info['n_slip'] = n_reset

    info['sat_el_deg'] = {
        int(epoch.sat[i]): float(np.degrees(epoch.el[i]))
        for i in range(len(epoch.sat))
    }
    if hasattr(epoch.obs, 'S'):
        sat_snr = {}
        for i, s in enumerate(epoch.sat):
            try:
                vals = np.asarray(epoch.obs.S[epoch.iu[i]], float)
                vals = vals[np.isfinite(vals)]
                if vals.size:
                    sat_snr[int(s)] = float(np.max(vals))
            except (ValueError, TypeError, IndexError):
                pass
        if sat_snr:
            info['sat_snr_dbhz'] = sat_snr

    pred_enu = np.array(epoch.pred_nav.pose().translation())
    pred_body_ecef = epoch.R_enu2ecef @ pred_enu + tc.base_ecef
    epoch.pred_ecef = tc._antenna_ecef(epoch.pred_nav.pose(), pred_body_ecef)
    return None
