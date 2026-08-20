"""Stage D — post-solve accuracy policy."""

import numpy as np

from ..integrity import sanity as _tc_sanity


# ── Phase-2 pipeline contract (see stage_contract.py) ──────────────
STAGE_READS = (
    'R_enu2ecef', 'ecef_tc', 'el', 'estimate', 'graph', 'info', 'ir_map', 'iu', 'key_idx',
    'nb', 'obs', 'obs_sd', 'obsb', 'pose_tc', 'pred_nav', 'rs', 'rsb',
    'sat', 'tag', 'xa',
)
STAGE_WRITES = (
    'nb', 'sol', 'tag', 'xa[*]',
)


def _release_suspicious_held_on_flt(tc, info):
    """Release the single most suspicious externally-held ambiguity."""
    per_sat = info.get('main_ddpr_per_sat') or {}
    cppr_sat = info.get('sat_cppr_sat') or {}
    worst_pair = info.get('main_ddpr_sat_worst')
    worst_sat = None
    worst_res = 0.0
    if isinstance(worst_pair, (tuple, list)) and len(worst_pair) >= 2:
        try:
            worst_sat = int(worst_pair[0])
            worst_res = float(worst_pair[1])
        except (ValueError, TypeError):
            worst_sat = None
            worst_res = 0.0
    res_thr = max(2.0, 0.5 * float(getattr(tc.cfg, 'ar_context_worst_sat_max', 0.0)))
    candidates = []
    for (s, f), _held_value in tc._sat_states.held_items():
        s_i = int(s)
        f_i = int(f)
        sat_res = float(per_sat.get(s_i, 0.0) or 0.0)
        cppr = max(
            int(cppr_sat.get(s_i, 0) or 0),
            int(tc._sat_states.at(s_i, f_i).rejc_cp_pr))
        score = 0.0
        if sat_res >= res_thr:
            score += sat_res
        if worst_sat is not None and s_i == worst_sat and worst_res >= res_thr:
            score += max(1.0, 0.25 * worst_res)
        if cppr > 0:
            score += 10.0 + float(cppr)
        if score > 0.0:
            candidates.append((score, sat_res, cppr, s_i, f_i))
    if not candidates:
        return 0
    candidates.sort(reverse=True)
    score, sat_res, cppr, s_i, f_i = candidates[0]
    sat_st = tc._sat_states.get(s_i, f_i)
    sat_st.release_hold(seed=True)
    sat_st.fix_streak = 0
    info['held_release_flt_count'] = 1
    info['held_release_flt_sat'] = s_i
    info['held_release_flt_freq'] = f_i
    info['held_release_flt_score'] = float(score)
    info['held_release_flt_res'] = float(sat_res)
    info['held_release_flt_cppr'] = int(cppr)
    return 1


def run(tc, epoch):
    """Stage D: post-solve accuracy policy."""
    _record_innovation(tc, epoch)
    sanity_result = _maybe_run_ddpr_sanity(tc, epoch)
    if sanity_result is not None:
        return sanity_result
    epoch.sol, epoch.tag, epoch.nb = _decide_fix_or_flt(tc, epoch)
    _update_streaks_and_post_hooks(tc, epoch)
    return None


def _record_innovation(tc, epoch):
    """Phase D-1 — record |pose_tc - IMU-predicted pose| as the innovation diagnostic."""
    info = epoch.info
    # Innovation vs IMU prediction (diagnostic + next-epoch CP-hold trigger)
    pred_ecef = (epoch.R_enu2ecef @ np.array(epoch.pred_nav.pose().translation())
                 + tc.base_ecef)
    innov = np.linalg.norm(epoch.ecef_tc - pred_ecef)
    info['innovation'] = innov
    # Innovation CP-hold trigger disabled — pure-form pipeline.



def _maybe_run_ddpr_sanity(tc, epoch):
    """Phase D-2 — opt-in DDPR sanity warm-reset (cfg.ddpr_sanity_enable). Returns the recovery early-return tuple when sanity fires, else None."""
    info = epoch.info
    if not tc.cfg.ddpr_sanity_enable:
        return None
    return _tc_sanity.run_ddpr_sanity(tc,
        epoch.graph, epoch.estimate, epoch.pose_tc, epoch.ecef_tc, epoch.pred_nav,
        epoch.obs, epoch.obsb, epoch.obs_sd, epoch.rs, epoch.rsb,
        epoch.sat, epoch.el, epoch.iu, epoch.ir_map, epoch.key_idx, info,
        nb=epoch.nb)


def _decide_fix_or_flt(tc, epoch):
    """Stage D step 3 — lambda_correction / weak-fix / low-nb gates.

    Pure decision: returns ``(sol, tag, nb)``; the caller applies it.
    """
    info = epoch.info
    # FIX / FLT tag decision + FLT DDPR-LS fallback
    pose_tc_antenna = tc._antenna_ecef(epoch.pose_tc, epoch.ecef_tc)
    if tc.nav.smode == 4 and epoch.nb > 0:
        lc = float(np.linalg.norm(epoch.xa[0:3] - pose_tc_antenna))
        info['lambda_correction'] = lc
        prev_was_flt = int(info.get('prev_smode', 0)) == 5
        prev_fix_streak_max = max(
            (st.fix_streak for st in tc._sat_states.values()), default=0)
        info['prev_fix_streak_max'] = prev_fix_streak_max
        weak_fix_fresh = prev_was_flt
        if int(tc.cfg.weak_fix_reject_max_prev_fix_streak) > 0:
            weak_fix_fresh = (
                weak_fix_fresh
                or prev_fix_streak_max
                <= int(tc.cfg.weak_fix_reject_max_prev_fix_streak))
        low_nb_fresh = prev_was_flt
        if int(tc.cfg.low_nb_fix_reject_max_prev_fix_streak) > 0:
            low_nb_fresh = (
                low_nb_fresh
                or prev_fix_streak_max
                <= int(tc.cfg.low_nb_fix_reject_max_prev_fix_streak))
        main_res = float(info.get('main_ddpr_res', 0.0) or 0.0)
        if (tc.cfg.lambda_corr_hard_max > 0
                and lc > tc.cfg.lambda_corr_hard_max):
            info['lambda_corr_hard_reject'] = lc
            return pose_tc_antenna, 'FLT', 0
        elif (tc.cfg.low_nb_fix_reject_nb_max > 0
                and epoch.nb <= tc.cfg.low_nb_fix_reject_nb_max
                and (not tc.cfg.low_nb_fix_only_after_flt or low_nb_fresh)):
            info['weak_fix_reject'] = True
            info['weak_fix_reject_nb'] = epoch.nb
            info['weak_fix_reject_lc'] = lc
            info['weak_fix_reject_main_ddpr_res'] = main_res
            return pose_tc_antenna, 'FLT', 0
        elif (tc.cfg.weak_fix_nb_max > 0
                and epoch.nb <= tc.cfg.weak_fix_nb_max
                and (not tc.cfg.weak_fix_only_after_flt or weak_fix_fresh)
                and ((tc.cfg.weak_fix_lambda_corr_max > 0
                      and lc > tc.cfg.weak_fix_lambda_corr_max)
                     or (tc.cfg.weak_fix_main_ddpr_res_max > 0
                         and main_res > tc.cfg.weak_fix_main_ddpr_res_max))):
            info['weak_fix_reject'] = True
            info['weak_fix_reject_nb'] = epoch.nb
            info['weak_fix_reject_lc'] = lc
            info['weak_fix_reject_main_ddpr_res'] = main_res
            return pose_tc_antenna, 'FLT', 0
        else:
            return epoch.xa[0:3], 'FIX', epoch.nb
    else:
        return pose_tc_antenna, 'FLT', epoch.nb



def _update_streaks_and_post_hooks(tc, epoch):
    """Phase D-4 — per-sat fix-streak update."""
    info = epoch.info
    if epoch.tag == 'FIX':
        for (s, f), _ in tc._sat_states.amb_items():
            try:
                held = (1 <= s <= tc.nav.fix.shape[0]
                        and 0 <= f < tc.nav.fix.shape[1]
                        and int(tc.nav.fix[s - 1, f]) == 3)
            except (ValueError, TypeError, IndexError):
                held = False
            sat_st = tc._sat_states.get(s, f)
            sat_st.fix_streak = sat_st.fix_streak + 1 if held else 0
    else:
        for st in tc._sat_states.values():
            st.fix_streak = 0
        _release_suspicious_held_on_flt(tc, info)



