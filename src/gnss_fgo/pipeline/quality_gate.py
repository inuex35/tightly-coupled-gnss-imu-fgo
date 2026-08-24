"""Stage B — quality gate + slip / CP-hold decisions."""

import numpy as np

from ..integrity import slip_detect as _tc_slip_detect
from ..utils import sorted_amb_items


# ── Phase-2 pipeline contract (see stage_contract.py) ──────────────
STAGE_READS = (
    'R_enu2ecef', 'el', 'estimate', 'graph', 'gyro_mean', 'imu_idx_prev', 'info',
    'ir_map', 'iu',
    'key_idx', 'ns', 'obs', 'obs_sd', 'obsb', 'pred_nav',
    'prev_amb_values', 'rs', 'sat', 'skip_cp_now', 'vel_prev',
    'values',
)
STAGE_WRITES = (
    'el[*]', 'iu[*]', 'pred_ecef', 'prev_amb_values',
    'prev_amb_values[*]', 'sat[*]', 'skip_cp_now',
)


def run(tc, epoch):
    """Stage B: quality gating + slip / CP-hold decisions."""
    _record_geometry(tc, epoch)

    epoch.skip_cp_now = _collect_telemetry_and_tick_holds(tc, epoch)

    epoch.prev_amb_values = _carry_prev_amb_and_rotate_keys(tc, epoch)
    epoch.pred_ecef = _predict_antenna_position(tc, epoch)
    return None


def _record_geometry(tc, epoch):
    """Step 1 — record GDOP / nsat for the AR gates and diagnostics.

    Every epoch gets solved: the old GDOP/nsat gate that skipped GNSS
    wholesale measured −12.6% total AllRMS when opened — the degraded
    epochs carry real steering information (deep canyons, the static
    basin), and per-observation weighting plus the AR gates own the
    quality question now.
    """
    info = epoch.info
    info['gdop'] = tc._compute_gdop(
        epoch.pred_nav, epoch.ns, epoch.rs, epoch.iu, epoch.R_enu2ecef)
    info['nsat'] = epoch.ns


def _collect_telemetry_and_tick_holds(tc, epoch):
    """Steps 2-4 — slip detection, per-sat telemetry (el / SNR / cppr), and the global CP-hold countdown/release decision."""
    info = epoch.info
    # Cycle slip detection + CMC multipath detection
    n_reset, _slip_keys = \
        _tc_slip_detect.detect_slips_and_reset_ambiguities(tc,
            epoch.obs, epoch.obs_sd, epoch.sat, epoch.iu)
    info['n_slip'] = n_reset

    # skip_cp_now reflects active global CP-hold (any trigger source).
    skip_cp_now = tc._recov_cp_hold > 0
    info['sat_el_deg'] = {
        int(epoch.sat[i]): float(np.degrees(epoch.el[i]))
        for i in range(len(epoch.sat))
    }
    if hasattr(epoch.obs, 'S'):
        sat_snr = {}
        for i, s in enumerate(epoch.sat):
            try:
                row = epoch.iu[i]
                vals = np.asarray(epoch.obs.S[row], float)
                vals = vals[np.isfinite(vals)]
                if vals.size:
                    sat_snr[int(s)] = float(np.max(vals))
            except (ValueError, TypeError, IndexError):
                pass
        if sat_snr:
            info['sat_snr_dbhz'] = sat_snr
    if skip_cp_now:
        tc._recovery.tick_cp_hold(info)
    return skip_cp_now




def _carry_prev_amb_and_rotate_keys(tc, epoch):
    """Step 5 — copy prev-epoch N values onto ``epoch.prev_amb_values`` for the BetweenN chain AND clear every ``amb_key`` (key rotation for the new epoch); skipped during whole-epoch CP-hold."""
    # Collect prev-epoch amb values for BetweenFactor chain (unless hold).
    prev_amb_values = {}
    if epoch.skip_cp_now:
        for (s, f), k in sorted_amb_items(tc._sat_states.amb_keys_dict()):
            tc._sat_states.get(s, f).amb_gen += 1
    else:
        for (s, f), k in sorted_amb_items(tc._sat_states.amb_keys_dict()):
            if epoch.estimate.exists(k):
                prev_amb_values[(s, f)] = (k, epoch.estimate.atDouble(k))
    for st in tc._sat_states.values():
        st.amb_key = None
    return prev_amb_values



def _predict_antenna_position(tc, epoch):
    """Step 6 — pred_ecef from the IMU-predicted pose and the antenna lever arm. Pure; the caller applies."""
    pred_enu = np.array(epoch.pred_nav.pose().translation())
    pred_body_ecef = epoch.R_enu2ecef @ pred_enu + tc.base_ecef
    return tc._antenna_ecef(epoch.pred_nav.pose(), pred_body_ecef)


