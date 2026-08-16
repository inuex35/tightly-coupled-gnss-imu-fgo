"""Stage B — quality gate + slip / CP-hold decisions."""

import numpy as np

from . import sat_quality as _satq
from ..state import effective_cp_hold_epochs
from . import slip_detect as _tc_slip_detect
from ..utils import sorted_amb_items
from .. import recovery as _tc_recovery
from ..buildfactor import doppler_sd as _tc_doppler_sd


# ── Phase-2 pipeline contract (see stage_contract.py) ──────────────
STAGE_READS = (
    'R', 'el', 'est2', 'g3', 'imu_idx_prev', 'info', 'ir_map', 'iu',
    'kk', 'ns', 'obs', 'obs_sd', 'obsb', 'pred_enu', 'pred',
    'prev_amb_tc', 'remove_indices', 'rs', 'sat', 'skip_cp_now',
    'slip_keys', 'v3',
)
STAGE_WRITES = (
    'el[*]', 'iu[*]', 'pred_ecef', 'pred_enu', 'prev_amb_tc',
    'prev_amb_tc[*]', 'sat[*]', 'skip_cp_now',
    'slip_keys',
)


def run(tc, ed):
    """Stage B: quality gating + slip / CP-hold decisions."""
    info = ed.info
    early = _gdop_gate_and_skip(tc, ed)
    if early is not None:
        return early
    tc.skip_count = 0

    fresh_amb_bootstrap = int(
        tc._tc_fresh_amb_epochs or 0) > 0
    if fresh_amb_bootstrap:
        info['fresh_amb_bootstrap'] = int(tc._tc_fresh_amb_epochs)
    sq = _satq.get_sat_quality(tc)

    forced_hold = _collect_sat_telemetry_and_holds(tc, ed, sq)
    sq.forced_hold_per_sat = forced_hold

    _carry_prev_amb(tc, ed, fresh_amb_bootstrap, forced_hold)
    _update_pred_ecef(tc, ed)
    return None


def _gdop_gate_and_skip(tc, ed):
    """Step 1 — GDOP / nsat gate; on failure PROCESSES the epoch via recovery.process_gdop_skip and returns its tuple (else None)."""
    info = ed.info
    gdop_val = tc._compute_gdop(ed.pred, ed.ns, ed.rs, ed.iu, ed.R)
    info['gdop'] = gdop_val
    info['nsat'] = ed.ns
    if not (gdop_val < tc.cfg.gdop_max
            and ed.ns >= tc.cfg.nsat_min):
        if tc.cfg.doppler_skip_aid and tc.cfg.doppler_sd_sigma > 0:
            # The skipped epoch still has 4-6 tracked satellites whose
            # Doppler bounds the velocity (NHC leaves vertical free and
            # the canyon drift is mostly U). Factors land in ed.g3,
            # which process_gdop_skip solves.
            _update_pred_ecef(tc, ed)
            _tc_doppler_sd.add_sd_doppler_factors(tc, ed, in_outage=True)
        return _tc_recovery.process_gdop_skip(tc,
            ed.obs, ed.kk, ed.g3, ed.v3, ed.R, info,
            imu_idx_prev=ed.imu_idx_prev,
            gyro_mean=getattr(ed, 'gyro_mean', None),
            vel_prev=getattr(ed, 'vel_p', None))



def _collect_sat_telemetry_and_holds(tc, ed, sq):
    """Steps 2-4 — slip detection, per-sat telemetry (el / SNR / cppr), forced-hold tick + CP-lock update + residual-based hold extension. Returns ``forced_hold`` set populated by ``sq.tick``."""
    info = ed.info
    # Cycle slip detection + CMC multipath detection
    n_reset, ed.remove_indices, n_cmc, ed.slip_keys = \
        _tc_slip_detect.detect_slips_and_manage_amb(tc, 
            ed.obs, ed.obs_sd, ed.sat, ed.iu,
            obsb=ed.obsb, ir_map=ed.ir_map)
    info['n_slip'] = n_reset
    if n_cmc > 0:
        info['cp_slip'] = n_cmc

    # Slip burst CP-hold trigger disabled — pure-form pipeline.

    # skip_cp_now reflects active global CP-hold (any trigger source).
    ed.skip_cp_now = tc._recov_cp_hold > 0
    info['sat_el_deg'] = {
        int(ed.sat[i]): float(np.degrees(ed.el[i]))
        for i in range(len(ed.sat))
    }
    if hasattr(ed, 'obs') and hasattr(ed.obs, 'S'):
        sat_snr = {}
        for i, s in enumerate(ed.sat):
            try:
                row = ed.iu[i]
                vals = np.asarray(ed.obs.S[row], float)
                vals = vals[np.isfinite(vals)]
                if vals.size:
                    sat_snr[int(s)] = float(np.max(vals))
            except (ValueError, TypeError, IndexError):
                pass
        if sat_snr:
            info['sat_snr_dbhz'] = sat_snr
    sat_cppr = {}
    for s in ed.sat:
        s = int(s)
        cpprs = [int(tc._sat_states.at(s, f).rejc_cp_pr)
                 for f in range(tc.nav.nf)]
        sat_cppr[s] = max(cpprs) if cpprs else 0
    info['sat_cppr_sat'] = sat_cppr
    forced_hold = sq.tick(tc._sat_states.amb_keys_dict(), info)
    visible_keys = {
        (int(s), int(f))
        for s in ed.sat
        for f in range(tc.nav.nf)
    }
    sq.update_cp_lock(visible_keys, slip_keys=ed.slip_keys, forced_hold=forced_hold)
    if ed.skip_cp_now:
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




def _carry_prev_amb(tc, ed, fresh_amb_bootstrap, forced_hold):
    """Step 6 — copy previous-epoch ambiguity values onto ``ed.prev_amb_tc`` for the BetweenN chain, skipping forced-hold sats and entire-epoch CP-hold."""
    # Collect prev-epoch amb values for BetweenFactor chain (unless hold).
    ed.prev_amb_tc = {}
    if fresh_amb_bootstrap:
        pass
    elif ed.skip_cp_now:
        for (s, f), k in sorted_amb_items(tc._sat_states.amb_keys_dict()):
            tc._sat_states.get(s, f).amb_gen += 1
    else:
        for (s, f), k in sorted_amb_items(tc._sat_states.amb_keys_dict()):
            if (s, f) in forced_hold:
                tc._sat_states.get(s, f).amb_gen += 1
                continue
            if ed.est2.exists(k):
                ed.prev_amb_tc[(s, f)] = (k, ed.est2.atDouble(k))
    for st in tc._sat_states.values():
        st.amb_key = None
    if fresh_amb_bootstrap:
        tc._tc_fresh_amb_epochs = max(
            0, int(tc._tc_fresh_amb_epochs) - 1)



def _update_pred_ecef(tc, ed):
    """Step 7 — write ``ed.pred_enu`` and ``ed.pred_ecef`` from the IMU-predicted pose, using the antenna lever arm."""
    ed.pred_enu = np.array(ed.pred.pose().translation())
    pred_body_ecef = ed.R @ ed.pred_enu + tc.base_ecef
    ed.pred_ecef = tc._antenna_ecef(ed.pred.pose(), pred_body_ecef)
    return None


