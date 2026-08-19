"""Stage B — quality gate + slip / CP-hold decisions."""

import numpy as np

from .. import sat_quality as _satq
from . import slip_detect as _tc_slip_detect
from ..utils import sorted_amb_items
from .. import recovery as _tc_recovery


# ── Phase-2 pipeline contract (see stage_contract.py) ──────────────
STAGE_READS = (
    'R_enu2ecef', 'el', 'estimate', 'graph', 'imu_idx_prev', 'info', 'ir_map', 'iu',
    'key_idx', 'ns', 'obs', 'obs_sd', 'obsb', 'pred_enu', 'pred_nav',
    'prev_amb_values', 'remove_indices', 'rs', 'sat', 'skip_cp_now',
    'slip_keys', 'values',
)
STAGE_WRITES = (
    'el[*]', 'iu[*]', 'pred_ecef', 'pred_enu', 'prev_amb_values',
    'prev_amb_values[*]', 'sat[*]', 'skip_cp_now',
    'slip_keys',
)


def run(tc, epoch):
    """Stage B: quality gating + slip / CP-hold decisions."""
    info = epoch.info
    early = _gdop_gate_and_skip(tc, epoch)
    if early is not None:
        return early
    tc.skip_count = 0

    fresh_amb_bootstrap = int(
        tc._tc_fresh_amb_epochs or 0) > 0
    if fresh_amb_bootstrap:
        info['fresh_amb_bootstrap'] = int(tc._tc_fresh_amb_epochs)
    sq = _satq.get_sat_quality(tc)

    forced_hold = _collect_telemetry_and_tick_holds(tc, epoch, sq)
    sq.forced_hold_per_sat = forced_hold

    _carry_prev_amb_and_rotate_keys(tc, epoch, fresh_amb_bootstrap, forced_hold)
    _update_pred_ecef(tc, epoch)
    return None


def _gdop_gate_and_skip(tc, epoch):
    """Step 1 — GDOP / nsat gate; on failure PROCESSES the epoch via recovery.process_gdop_skip and returns its tuple (else None)."""
    info = epoch.info
    gdop_val = tc._compute_gdop(epoch.pred_nav, epoch.ns, epoch.rs, epoch.iu, epoch.R_enu2ecef)
    info['gdop'] = gdop_val
    info['nsat'] = epoch.ns
    if not (gdop_val < tc.cfg.gdop_max
            and epoch.ns >= tc.cfg.nsat_min):
        _update_pred_ecef(tc, epoch)
        return _tc_recovery.process_gdop_skip(tc,
            epoch.obs, epoch.key_idx, epoch.graph, epoch.values, epoch.R_enu2ecef, info,
            imu_idx_prev=epoch.imu_idx_prev,
            gyro_mean=getattr(epoch, 'gyro_mean', None),
            vel_prev=getattr(epoch, 'vel_prev', None), epoch=epoch)



def _collect_telemetry_and_tick_holds(tc, epoch, sq):
    """Steps 2-4 — slip detection, per-sat telemetry (el / SNR / cppr), forced-hold tick, CP-lock update, and the global CP-hold countdown/release decision. Returns the ``forced_hold`` set."""
    info = epoch.info
    # Cycle slip detection + CMC multipath detection
    n_reset, epoch.remove_indices, n_cmc, epoch.slip_keys = \
        _tc_slip_detect.detect_slips_and_manage_amb(tc, 
            epoch.obs, epoch.obs_sd, epoch.sat, epoch.iu,
            obsb=epoch.obsb, ir_map=epoch.ir_map)
    info['n_slip'] = n_reset
    if n_cmc > 0:
        info['cp_slip'] = n_cmc

    # Slip burst CP-hold trigger disabled — pure-form pipeline.

    # skip_cp_now reflects active global CP-hold (any trigger source).
    epoch.skip_cp_now = tc._recov_cp_hold > 0
    info['sat_el_deg'] = {
        int(epoch.sat[i]): float(np.degrees(epoch.el[i]))
        for i in range(len(epoch.sat))
    }
    if hasattr(epoch, 'obs') and hasattr(epoch.obs, 'S'):
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
    sat_cppr = {}
    for s in epoch.sat:
        s = int(s)
        cpprs = [int(tc._sat_states.at(s, f).rejc_cp_pr)
                 for f in range(tc.nav.nf)]
        sat_cppr[s] = max(cpprs) if cpprs else 0
    info['sat_cppr_sat'] = sat_cppr
    forced_hold = sq.tick(tc._sat_states.amb_keys_dict(), info)
    visible_keys = {
        (int(s), int(f))
        for s in epoch.sat
        for f in range(tc.nav.nf)
    }
    sq.update_cp_lock(visible_keys, slip_keys=epoch.slip_keys, forced_hold=forced_hold)
    if epoch.skip_cp_now:
        tc._recov_cp_hold -= 1
        thr = float(tc.cfg.recov_cp_release_thresh)
        if thr > 0:
            last_res = float(tc._last_main_ddpr_res)
            if last_res > 0 and last_res <= thr:
                tc._recov_cp_release_streak = (
                    tc._recov_cp_release_streak + 1)
            else:
                tc._recov_cp_release_streak = 0
            need = int(tc.cfg.recov_cp_release_count)
            if (tc._recov_cp_hold <= 0
                    and tc._recov_cp_release_streak < need):
                tc._recov_cp_hold = 1
                info['recov_cp_release_wait'] = last_res
        info['recov_cp_hold'] = tc._recov_cp_hold + 1
    return forced_hold




def _carry_prev_amb_and_rotate_keys(tc, epoch, fresh_amb_bootstrap, forced_hold):
    """Step 6 — copy prev-epoch N values onto ``epoch.prev_amb_values`` for the BetweenN chain AND clear every ``amb_key`` (key rotation for the new epoch); skips forced-hold sats and whole-epoch CP-hold."""
    # Collect prev-epoch amb values for BetweenFactor chain (unless hold).
    epoch.prev_amb_values = {}
    if fresh_amb_bootstrap:
        pass
    elif epoch.skip_cp_now:
        for (s, f), k in sorted_amb_items(tc._sat_states.amb_keys_dict()):
            tc._sat_states.get(s, f).amb_gen += 1
    else:
        for (s, f), k in sorted_amb_items(tc._sat_states.amb_keys_dict()):
            if (s, f) in forced_hold:
                tc._sat_states.get(s, f).amb_gen += 1
                continue
            if epoch.estimate.exists(k):
                epoch.prev_amb_values[(s, f)] = (k, epoch.estimate.atDouble(k))
    for st in tc._sat_states.values():
        st.amb_key = None
    if fresh_amb_bootstrap:
        tc._tc_fresh_amb_epochs = max(
            0, int(tc._tc_fresh_amb_epochs) - 1)



def _update_pred_ecef(tc, epoch):
    """Step 7 — write ``epoch.pred_enu`` and ``epoch.pred_ecef`` from the IMU-predicted pose, using the antenna lever arm."""
    epoch.pred_enu = np.array(epoch.pred_nav.pose().translation())
    pred_body_ecef = epoch.R_enu2ecef @ epoch.pred_enu + tc.base_ecef
    epoch.pred_ecef = tc._antenna_ecef(epoch.pred_nav.pose(), pred_body_ecef)
    return None


