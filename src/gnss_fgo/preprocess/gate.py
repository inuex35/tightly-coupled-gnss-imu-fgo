"""Stage B — quality gate + slip / CP-hold decisions."""

import numpy as np

from . import sat_quality as _satq
from ..state import effective_cp_hold_epochs
from . import slip_detect as _tc_slip_detect
from ..utils import sorted_amb_items
from ..validation import recovery as _tc_recovery


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
    early = _check_gdop_nsat_gate(tc, ed)
    if early is not None:
        return early
    tc.skip_count = 0

    fresh_amb_bootstrap = int(
        tc._tc_fresh_amb_epochs or 0) > 0
    if fresh_amb_bootstrap:
        info['fresh_amb_bootstrap'] = int(tc._tc_fresh_amb_epochs)
    sq = _satq.get_sat_quality(tc)

    forced_hold = _collect_sat_telemetry_and_holds(tc, ed, sq)
    penalized_hold = _apply_dirty_sat_reset(tc, ed, sq, forced_hold)
    sq.forced_hold_per_sat = forced_hold
    sq.penalized_per_sat = penalized_hold

    _carry_prev_amb(tc, ed, fresh_amb_bootstrap, forced_hold)
    _update_pred_ecef(tc, ed)
    return None


def _check_gdop_nsat_gate(tc, ed):
    """Phase 1 — GDOP / nsat gate. Returns the recovery early-return tuple when the gate fails, else None."""
    info = ed.info
    gdop_val = tc._compute_gdop(ed.pred, ed.ns, ed.rs, ed.iu, ed.R)
    info['gdop'] = gdop_val
    info['nsat'] = ed.ns
    if not (gdop_val < tc.cfg.gdop_max
            and ed.ns >= tc.cfg.nsat_min):
        return _tc_recovery.process_gdop_skip(tc, 
            ed.obs, ed.kk, ed.g3, ed.v3, ed.R, info,
            imu_idx_prev=ed.imu_idx_prev,
            gyro_mean=getattr(ed, 'gyro_mean', None),
            vel_prev=getattr(ed, 'vel_p', None))



def _collect_sat_telemetry_and_holds(tc, ed, sq):
    """Phase 2-4 — slip detection, per-sat telemetry (el / SNR / cppr), forced-hold tick + CP-lock update + residual-based hold extension. Returns ``forced_hold`` set populated by ``sq.tick``."""
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


def _apply_dirty_sat_reset(tc, ed, sq, forced_hold):
    """Phase 5 — dirty-sat immediate reset (cp_hold_dirty_reset_enable). Bumps amb_gen for sats whose post-fit DDPR residual stayed above ``recov_cp_release_thresh`` for the suspect-streak threshold; mutates ``forced_hold`` in place. Returns the ``penalized_hold`` set for sats currently in the pre-reset stage."""
    info = ed.info
    penalized_hold = set()
    q_map = sq.hold_quarantine
    cooldown_map = sq.dirty_cooldown
    suspect_map = sq.dirty_suspect
    n_dirty_reset = 0
    dirty_reset_detail = []
    dirty_penalized_detail = []
    if (bool(tc.cfg.cp_hold_dirty_reset_enable)
            and tc.cfg.recov_cp_release_thresh > 0):
        thr = float(tc.cfg.recov_cp_release_thresh)
        hold_n = int(tc.cfg.cp_hold_dirty_reset_hold)
        if hold_n <= 0:
            hold_n = effective_cp_hold_epochs(tc)
        suspect_need = max(
            1, int(tc.cfg.cp_hold_dirty_reset_suspect_count))
        cooldown_n = max(
            0, int(tc.cfg.cp_hold_dirty_reset_cooldown))
        per_sat = tc._mres_signals.per_sat
        for (s, f), _amb in list(tc._sat_states.amb_items()):
            key = (s, f)
            if key in forced_hold:
                continue
            res_s = float(per_sat.get(s, 0.0))
            if res_s <= thr:
                suspect_map.pop(key, None)
                continue
            cppr_count = int(tc._sat_states.at(s, f).rejc_cp_pr)
            if cppr_count <= 0:
                suspect_map.pop(key, None)
                continue
            streak = suspect_map.get(key, 0) + 1
            suspect_map[key] = streak
            if cooldown_map.get(key, 0) > 0:
                penalized_hold.add(key)
                dirty_penalized_detail.append({
                    'sat': int(s),
                    'freq': int(f),
                    'ddpr_res': res_s,
                    'cp_pr_reject': cppr_count,
                    'suspect_streak': int(streak),
                    'cooldown_epochs': int(cooldown_map[key]),
                })
                continue
            if streak < suspect_need:
                penalized_hold.add(key)
                dirty_penalized_detail.append({
                    'sat': int(s),
                    'freq': int(f),
                    'ddpr_res': res_s,
                    'cp_pr_reject': cppr_count,
                    'suspect_streak': int(streak),
                    'cooldown_epochs': 0,
                })
                continue
            _sat_st = tc._sat_states.get(s, f)
            _sat_st.amb_gen += 1
            _sat_st.clear_hold()
            _sat_st.rejc_cp_pr = 0
            _sat_st.rejc_post_ddpr = 0
            _sat_st.fix_streak = 0
            q_map[(s, f)] = hold_n
            if cooldown_n > 0:
                cooldown_map[key] = cooldown_n
            suspect_map.pop(key, None)
            forced_hold.add(key)
            n_dirty_reset += 1
            dirty_reset_detail.append({
                'sat': int(s),
                'freq': int(f),
                'ddpr_res': res_s,
                'cp_pr_reject': cppr_count,
                'hold_epochs': int(hold_n),
                'suspect_streak': int(streak),
            })
        if n_dirty_reset:
            sq.reset_cp_lock(forced_hold)
        if n_dirty_reset:
            info['dirty_sat_reset'] = n_dirty_reset
            info['dirty_sat_reset_detail'] = dirty_reset_detail
    if dirty_penalized_detail:
        info['dirty_sat_penalized'] = len(dirty_penalized_detail)
        info['dirty_sat_penalized_detail'] = dirty_penalized_detail
    return penalized_hold


def _carry_prev_amb(tc, ed, fresh_amb_bootstrap, forced_hold):
    """Phase 6 — copy previous-epoch ambiguity values onto ``ed.prev_amb_tc`` for the BetweenN chain, skipping forced-hold sats and entire-epoch CP-hold."""
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
    """Phase 7 — write ``ed.pred_enu`` and ``ed.pred_ecef`` from the IMU-predicted pose, using the antenna lever arm."""
    ed.pred_enu = np.array(ed.pred.pose().translation())
    pred_body_ecef = ed.R @ ed.pred_enu + tc.base_ecef
    ed.pred_ecef = tc._antenna_ecef(ed.pred.pose(), pred_body_ecef)
    return None


