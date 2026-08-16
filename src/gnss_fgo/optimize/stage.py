"""Stage C — solve side: DD factor build, ISAM2 update, FDE, LAMBDA AR."""

import os
import numpy as np
import gtsam

from .. import ar as _tc_ar
from ..buildfactor import clock as _tc_clock
from ..buildfactor import doppler as _tc_doppler
from ..buildfactor import doppler_sd as _tc_doppler_sd
from ..buildfactor import tdcp as _tc_tdcp
from ..buildfactor import factors as _tc_factors
from ..buildfactor import nhc as _tc_nhc
from ..buildfactor import zupt as _tc_zupt
from ..preprocess import prefit as _tc_prefit
from ..preprocess import sat_quality as _satq
from ..utils import heading_from_pose, sorted_amb_items
from ..validation import postfit as _tc_postfit
from ..validation import recovery as _tc_recovery
from . import solver as _tc_solver


# ── Phase-2 pipeline contract (see stage_contract.py) ──────────────
STAGE_READS = (
    'R', 'bias_p', 'dts', 'ecef_tc', 'el', 'est2', 'g3', 'gyro_mean',
    'info', 'ir_map', 'iu', 'kk', 'nb', 'ns', 'nv', 'obs', 'obs_sd',
    'obsb', 'pose_tc', 'pred_ecef', 'pred', 'prev_amb_tc',
    'remove_indices', 'rs', 'rsb', 'sat', 'skip_cp_now', 'slip_keys',
    'v3', 'vs',
)
STAGE_WRITES = (
    'ecef_tc', 'est2', 'nb', 'nv', 'pose_tc', 'prev_amb_tc[*]', 'xa',
)


def run(tc, ed):
    """Stage C: solve (DD factors → ISAM2 → FDE → LAMBDA AR)."""
    prev_smode = int(getattr(tc.nav, 'smode', 0))
    _build_factor_block(tc, ed, prev_smode)        # Layer 3
    early = _solve_isam2(tc, ed)                   # Layer 4
    if early is not None:
        return early
    _compute_postfit_diagnostics(tc, ed)           # Layer 6
    return _run_lambda_ar(tc, ed)                  # Layer 5


def _build_factor_block(tc, ed, prev_smode):
    """Layer 3 — DD factors + BetweenN chain + propagate-prior fallback + (Doppler / NHC / ZUPT / bootstrap-DDPR) priors."""
    info = ed.info
    # DD factor construction
    ed.nv = _tc_factors.build_dd_factors(tc, 
        ed.g3, ed.v3, ed.obs, ed.obsb, ed.obs_sd,
        ed.rs, ed.rsb, ed.sat, ed.el, ed.iu, ed.ir_map,
        ed.pred_ecef, tc.Xpose(ed.kk), tc.lever_arm_tc,
        tc.amb_keys_tc,
        track_indices=True, dd_epoch=ed.kk,
        prev_amb_values=ed.prev_amb_tc,
        skip_cp=ed.skip_cp_now, slip_keys=ed.slip_keys)
    sq = _satq.get_sat_quality(tc)
    sat_lock_age = {}
    for s in ed.sat:
        s = int(s)
        ages = [
            int(sq.cp_lock_streak.get((s, f), 0))
            for f in range(tc.nav.nf)
            if (s, f) in sq.cp_lock_streak
        ]
        sat_lock_age[s] = (int(max(ages)) if ages else np.nan)
    info['sat_lock_age'] = sat_lock_age

    last_flt = (prev_smode == 5)
    info['prev_smode'] = prev_smode
    sig_between_flt = tc.cfg.sigma_n_between_flt
    sig_between_fix = tc.cfg.sigma_n_between
    warmup = max(0, int(tc.cfg.sigma_n_between_warmup))
    streak_map = tc._fix_streak
    n_between = 0
    if not ed.skip_cp_now and tc.cfg.betweenn_enable:
        for (s, f), k_new in sorted_amb_items(tc._sat_states.amb_keys_dict()):
            if (s, f) in ed.prev_amb_tc:
                k_old, _ = ed.prev_amb_tc[(s, f)]
                if last_flt:
                    sig_between = sig_between_flt
                elif warmup > 0 and streak_map is not None:
                    streak = streak_map.get((s, f), 0)
                    sig_between = (sig_between_fix if streak >= warmup
                                   else sig_between_flt)
                else:
                    sig_between = sig_between_fix
                ed.g3.add(gtsam.BetweenFactorDouble(
                    k_old, k_new, 0.0,
                    tc._noise1(sig_between)))
                n_between += 1
    info['n_dd'] = ed.nv
    cp_pr_rej = tc._last_cp_pr_reject
    rejc_wipe = tc._last_rejc_wipe
    if cp_pr_rej:
        info['cp_pr_reject'] = cp_pr_rej
    if rejc_wipe:
        info['rejc_wipe'] = rejc_wipe
    tc._last_cp_pr_reject = 0
    tc._last_rejc_wipe = 0

    if ed.nv < tc.cfg.min_dd_for_solve:
        info['propagate_prior'] = ed.nv
        ed.g3.addPriorPose3(
            tc.Xpose(ed.kk), ed.pred.pose(),
            gtsam.noiseModel.Isotropic.Sigma(
                6, tc.cfg.propagate_pose_sigma))
        ed.g3.addPriorVector(
            tc.Vel(ed.kk), ed.pred.velocity(),
            gtsam.noiseModel.Isotropic.Sigma(
                3, tc.cfg.propagate_vel_sigma))
        ed.g3.addPriorConstantBias(
            tc.Bias(ed.kk), ed.bias_p,
            gtsam.noiseModel.Isotropic.Sigma(
                6, tc.cfg.propagate_bias_sigma))
        if not ed.skip_cp_now:
            n_anchored = 0
            amb_noise = tc._noise1(tc.cfg.propagate_amb_sigma)
            for (s, f), k_new in sorted_amb_items(tc._sat_states.amb_keys_dict()):
                if (s, f) in ed.prev_amb_tc:
                    _, n_prev = ed.prev_amb_tc[(s, f)]
                    ed.g3.add(gtsam.PriorFactorDouble(
                        k_new, n_prev, amb_noise))
                    n_anchored += 1
            info['n_anchored'] = n_anchored
    info['n_between'] = n_between

    # Raw per-satellite Doppler factors (opt-in via cfg.doppler_sigma)
    _tc_doppler.add_doppler_factors(tc, ed)

    # Undifferenced pseudoranges pinning the clock chain the Doppler needs
    _tc_clock.add_clock_pr_factors(tc, ed)

    # Between-satellite differenced Doppler (clock-free; cfg.doppler_sd_sigma)
    _tc_doppler_sd.add_sd_doppler_factors(tc, ed)

    # TDCP relative-displacement constraints (rover-only carrier deltas)
    _tc_tdcp.add_tdcp_factors(tc, ed)

    try:
        speed_for_nhc = float(np.linalg.norm(
            np.array(ed.est2.atVector(tc.Vel(ed.kk - 1)))[:2]))
    except RuntimeError:
        speed_for_nhc = float(np.linalg.norm(
            np.array(ed.pred.velocity())[:2]))
    if _tc_nhc.add_nhc_factor(tc, ed.g3, ed.kk, speed_for_nhc,
                            gyro_mean_rh=ed.gyro_mean):
        info['nhc'] = True

    _tc_zupt.add_zupt_factors_for_stage(tc, ed)

    bootstrap_ddpr_epochs = int(
        tc._tc_bootstrap_ddpr_epochs or 0)
    if bootstrap_ddpr_epochs > 0:
        try:
            ecef_ls, n_ls, res_ls = tc._ddpr_only_position(
                ed.obs, ed.obsb, ed.obs_sd, ed.rs, ed.rsb,
                ed.sat, ed.el, ed.iu, ed.ir_map, ed.pred.pose())
            if ecef_ls is not None and n_ls >= 4:
                body_enu_ls = ed.R.T @ (np.asarray(ecef_ls) - tc.base_ecef)
                pose_ls = gtsam.Pose3(
                    ed.pred.pose().rotation(),
                    gtsam.Point3(*body_enu_ls))
                boot_sigma = float(os.environ.get('BOOT_DDPR_SIGMA', '0.5'))
                sigmas = np.array([1e6, 1e6, 1e6,
                                   boot_sigma, boot_sigma, boot_sigma])
                ed.g3.addPriorPose3(
                    tc.Xpose(ed.kk), pose_ls,
                    gtsam.noiseModel.Diagonal.Sigmas(sigmas))
                info['bootstrap_ddpr_prior_nv'] = int(n_ls)
                info['bootstrap_ddpr_prior_res'] = float(res_ls)
                info['bootstrap_ddpr_prior_sigma'] = float(boot_sigma)
        except (RuntimeError, ValueError):
            pass
        tc._tc_bootstrap_ddpr_epochs = max(0, bootstrap_ddpr_epochs - 1)

    # ────────────────────────────────────────────────────────────────


def _solve_isam2(tc, ed):
    """Layer 4 — gather kept keys, run FLS update, snapshot the new estimate. Returns the recovery early-return tuple on solve failure, else None."""
    info = ed.info
    try:
        extra = [k for (_sf, k) in sorted_amb_items(tc._sat_states.amb_keys_dict())]
        for sf, (k_old, _) in sorted_amb_items(ed.prev_amb_tc):
            if tc._sat_states.at(*sf).amb_key is not None:
                extra.append(k_old)
        extra.extend(tc._doppler_keep_keys)
        _tc_solver.fls_update(tc, ed.g3, ed.v3, ed.kk, keep_keys=extra,
                         remove_indices=ed.remove_indices)
        ed.est2 = tc.isam2.calculateEstimate()
    except (RuntimeError, IndexError, ValueError) as ex:
        # ValueError: ISAM2 marginalization raises it ("Asking to remove
        # variables from the variable index that are not unused") when a
        # prior purge left the FLS bookkeeping inconsistent — exactly
        # the smoother-broke case the warm reset exists for.
        return _tc_recovery.handle_solve_exception(tc,
            ex, ed.pred, ed.bias_p, ed.kk,
            ed.obs, ed.obsb, ed.obs_sd, ed.rs, ed.rsb,
            ed.sat, ed.el, ed.iu, ed.ir_map, info)

    # ────────────────────────────────────────────────────────────────


def _compute_postfit_diagnostics(tc, ed):
    """Layer 6 — main DDPR + factor-residual diagnostics, persist-bad / observation-quality bookkeeping, post-fit FDE re-solve, and pose snapshot."""
    info = ed.info
    sq = _satq.get_sat_quality(tc)
    if tc.cfg.diag_main_ddpr_res:
        main_res_pre_fde, per_sat_res, pair_rows = _tc_postfit.main_ddpr_residuals(tc, 
            ed.g3, ed.est2, with_pairs=True)
        info['main_ddpr_res'] = main_res_pre_fde
        info['main_ddpr_per_sat'] = per_sat_res
        info['main_ddpr_pairs'] = pair_rows
        info['ref_sats'] = dict(getattr(tc, 'ref_sats', {}) or {})
        tc._cached_ddpr_res_pre = main_res_pre_fde
        tc._mres_signals.update(
            last_res=main_res_pre_fde,
            per_sat=dict(per_sat_res) if per_sat_res else {},
            epoch=int(tc.epoch))
    else:
        main_res_pre_fde = 0.0
        per_sat_res = {}
        tc._cached_ddpr_res_pre = None
        tc._mres_signals.reset()
    if tc.cfg.diag_factor_residuals:
        all_res = _tc_postfit.all_factor_residuals(tc, ed.g3, ed.est2)
        for tag, (rms, n) in all_res.items():
            info[f'fres_{tag}'] = rms
            info[f'fcnt_{tag}'] = n

    if per_sat_res:
        worst_sat = max(per_sat_res, key=per_sat_res.get)
        info['main_ddpr_sat_worst'] = (worst_sat,
                                         per_sat_res[worst_sat])
    if tc.cfg.ar_persist_bad_enable and getattr(tc, 'phase', 1) >= 2:
        sq = _satq.get_sat_quality(tc)
        thr = float(tc.cfg.ar_persist_bad_res_thresh)
        streak_need = max(1, int(tc.cfg.ar_persist_bad_streak))
        hold_len = max(1, int(tc.cfg.ar_persist_bad_hold))
        seen = set()
        for s, rmax in (per_sat_res or {}).items():
            s = int(s)
            seen.add(s)
            if rmax > thr:
                st = sq.persist_bad_streak.get(s, 0) + 1
                sq.persist_bad_streak[s] = st
                if st >= streak_need:
                    sq.persist_bad_hold[s] = max(
                        int(sq.persist_bad_hold.get(s, 0)), hold_len)
                    for f in range(tc.nav.nf):
                        key = (s, f)
                        sat_st = tc._sat_states.get(*key)
                        sat_st.amb_gen += 1
                        sat_st.rejc_cp_pr = 0
                        sat_st.fix_streak = 0
            else:
                sq.persist_bad_streak[s] = 0
        for s in list(sq.persist_bad_streak.keys()):
            if s not in seen:
                sq.persist_bad_streak[s] = 0

    if getattr(tc, 'phase', 1) >= 2:
        worst_sat_id = int(worst_sat) if per_sat_res else None
        cppr_sat = info.get('sat_cppr_sat', {}) or {}
        sq = _satq.get_sat_quality(tc)
        sq.update_observation_quality(
            tc.cfg, per_sat_res, worst_sat=worst_sat_id, cppr_sat=cppr_sat,
            sat_el_deg=info.get('sat_el_deg'),
            sat_snr_dbhz=info.get('sat_snr_dbhz'))

    if tc.cfg.fde_enable:
        ed.est2 = _tc_postfit.apply_fde(tc, 
            ed.g3, ed.kk, ed.nv, ed.est2, info)

    # Pose after FDE re-solve
    ed.pose_tc = ed.est2.atPose3(tc.Xpose(ed.kk))
    info['post_heading_deg'] = heading_from_pose(ed.pose_tc)
    tc.tc_bias = ed.est2.atConstantBias(tc.Bias(ed.kk))
    enu_tc = np.array(ed.pose_tc.translation())
    ed.ecef_tc = ed.R @ enu_tc + tc.base_ecef

    ref = getattr(ed, 'ref_ecef', None)
    if ref is not None and tc.cfg.diag_truth_residual:
        try:
            R_e2n = tc.R_enu2ecef.T
            lever_arr = np.array(tc.lever_arm_tc) \
                if getattr(tc, 'lever_arm_tc', None) is not None \
                else np.zeros(3)
            R_body = np.array(
                tc.ecef_T_nav.compose(ed.pose_tc).rotation().matrix())
            truth_body_ecef = np.asarray(ref) - R_body @ lever_arr
            truth_body_enu = R_e2n @ (truth_body_ecef - tc.base_ecef)
            v_truth = gtsam.Values()
            v_truth.insert(tc.Xpose(ed.kk),
                           gtsam.Pose3(ed.pose_tc.rotation(),
                                        gtsam.Point3(*truth_body_enu)))
            truth_res, truth_per_sat, truth_pair_rows = _tc_postfit.main_ddpr_residuals(tc, 
                ed.g3, v_truth, with_pairs=True)
            info['ddpr_res_at_truth'] = float(truth_res)
            info['ddpr_per_sat_at_truth'] = (
                dict(truth_per_sat) if truth_per_sat else {}
            )
            info['ddpr_pairs_at_truth'] = truth_pair_rows
            info['truth_offset'] = float(np.linalg.norm(
                np.array(ed.pose_tc.translation())
                - np.asarray(truth_body_enu)))
        except (RuntimeError, ValueError):
            pass


_AR_OUTCOME_CODES = {
    'not_called': 0, 'armode_off': 1, 'entered': 2,
    'lambda_exception': 3, 'lambda_zero': 4, 'min_nb_gate': 5,
    'valpos_failed': 6, 'ar_context_reject': 7, 'success': 8,
    'fix_dres': 11, 'gdop_gate': 12,
}

_AR_DIAG_ATTRS = (
    ('_last_orphan_cp_count', 'orphan_cp_count'),
    ('_last_amb_dict_size', 'amb_dict_size'),
    ('_last_held_size', 'held_size'),
    ('_last_cp_visible_size', 'cp_visible_size'),
    ('_last_amb_estimate_missing', 'amb_estimate_missing'),
    ('_last_amb_vsat1', 'amb_vsat1'),
    ('_last_amb_vsat0_young', 'amb_vsat0_young'),
    ('_last_amb_vsat0_held_bad', 'amb_vsat0_held_bad'),
    ('_last_amb_age_median', 'amb_age_median'),
    ('_last_amb_age_min', 'amb_age_min'),
    ('_last_amb_not_in_obs', 'amb_not_in_obs'),
    ('_last_held_not_in_obs', 'held_not_in_obs'),
    ('_last_sat_in_obs_size', 'sat_in_obs_size'),
    ('_last_resamb_raw_nb', 'resamb_raw_nb'),
    ('_last_amb_el_min_deg', 'amb_el_min_deg'),
    ('_last_amb_el_median_deg', 'amb_el_median_deg'),
    ('_last_amb_el_above15', 'amb_el_above15'),
    ('_last_amb_el_above25', 'amb_el_above25'),
)


def _ar_eligibility(tc, ed):
    """Return True iff this epoch should run LAMBDA AR."""
    if ed.kk < tc.cfg.n_collect + 3:
        return False
    if tc._recov_cp_hold > 0:
        return False
    if tc.ar_max_frac < 0.5:
        max_frac = tc._compute_max_dd_frac(
            ed.est2, ed.obs_sd, ed.sat, ed.ns)
        ed.info['max_frac'] = max_frac
        if max_frac > tc.ar_max_frac:
            return False
    return True


def _run_ar_with_marginals(tc, ed):
    """Pre-check, write_marginals + per_sat gate + run_ar; populate ed.nb, ed.xa, info[ar_skipped*]."""
    info = ed.info
    if tc.cfg.ar_precheck_skip:
        skip_ar, skip_detail = _tc_ar.should_skip_ar_precheck(tc)
        if skip_ar:
            info['ar_skipped_precheck'] = True
            info.update({f'ar_skipped_{k}': v
                         for k, v in skip_detail.items()})
            return
    tc._cur_ed = ed                 # for the fix-vs-LS gate in run_ar
    tc.nav.x[0:3] = tc._antenna_ecef(ed.pose_tc, ed.ecef_tc)
    amb_snapshot = tc._sat_states.amb_keys_dict()
    _tc_ar.write_marginals(tc,
        tc.isam2.getFactors(), ed.est2,
        tc.Xpose(ed.kk), amb_snapshot)
    # AR-only geometry gate (demo5 arthres1 spirit): when the DOP says
    # the geometry cannot support an integer decision, do not attempt
    # AR. At 7 sats / GDOP~10 a 9 m vertical basin costs only ~1.7 m of
    # code residual — no residual test can see it, and the marginal
    # covariance can't be used (it depends on the very holds the gate
    # controls: measured feedback death, fix 46%->0.6%). GDOP is pure
    # geometry: no loop. Measured run1 separation: correct fixes GDOP
    # p50 3.3 / p99 7.6, basin wrong fixes p50 9.5.
    ar_gdop = float(getattr(tc.cfg, 'ar_gdop_max', 0.0) or 0.0)
    gdop_now = float(info.get('gdop', 0.0) or 0.0)
    if ar_gdop > 0.0 and gdop_now > ar_gdop:
        info['ar_gdop_skip'] = True
        tc._last_ar_outcome = 'gdop_gate'
        return
    ed.nb, ed.xa = _tc_ar.run_ar(tc,
        ed.obs, ed.rs, ed.vs, ed.dts,
        ed.sat, ed.el, ed.iu, ed.est2,
        tc.Xpose(ed.kk), amb_snapshot)
    xv_thr = float(tc.cfg.ar_ddpr_xvalidate_thresh or 0.0)
    xv_delta = float(tc.cfg.ar_ddpr_xvalidate_delta_thresh or 0.0)
    if (xv_thr > 0.0 or xv_delta > 0.0) and ed.nb > 0 and ed.xa is not None:
        try:
            cur_pose = ed.est2.atPose3(tc.Xpose(ed.kk))
            R_body_to_ecef = tc.ecef_T_nav.compose(
                cur_pose).rotation().matrix()
            lever_arr = (np.array(tc.lever_arm_tc)
                         if getattr(tc, 'lever_arm_tc', None) is not None
                         else np.zeros(3))
            body_ecef_xa = np.asarray(ed.xa[0:3]) - R_body_to_ecef @ lever_arr
            body_nav_xa = tc.ecef_T_nav.transformTo(
                gtsam.Point3(*body_ecef_xa))
            xa_pose = gtsam.Pose3(cur_pose.rotation(), body_nav_xa)
            v_xa = gtsam.Values()
            v_xa.insert(tc.Xpose(ed.kk), xa_pose)
            res_xa, _ = _tc_postfit.main_ddpr_residuals(tc, ed.g3, v_xa)
            info['ar_ddpr_xvalidate_res_at_xa'] = float(res_xa)
            res_pre = tc._cached_ddpr_res_pre
            if res_pre is not None:
                info['ar_ddpr_xvalidate_delta'] = float(res_xa - res_pre)
            reject = False
            if xv_thr > 0.0 and float(res_xa) > xv_thr:
                reject = True
            if (xv_delta > 0.0 and res_pre is not None
                    and float(res_xa) - float(res_pre) > xv_delta):
                reject = True
            if reject:
                info['ar_ddpr_xvalidate_reject'] = True
                ed.nb = 0
                ed.xa = None
                tc.nav.smode = 5
        except (RuntimeError, ValueError):
            pass


def _record_ar_diagnostics(tc, info):
    """Copy tc._last_* counters → info + outcome code + ar_context_reject + ar_subset_dbg."""
    for diag_attr, info_key in _AR_DIAG_ATTRS:
        v = getattr(tc, diag_attr, None)
        if v is not None:
            info[info_key] = int(v)
    outcome = tc._last_ar_outcome
    if outcome:
        info['ar_outcome_code'] = _AR_OUTCOME_CODES.get(outcome, -1)
    ar_ctx_reject = tc._ar_context_reject
    if ar_ctx_reject:
        info['ar_context_reject'] = True
        info['ar_context_reject_nb'] = int(ar_ctx_reject.get('nb', 0))
        info['ar_context_reject_main_ddpr_res'] = float(
            ar_ctx_reject.get('main_ddpr_res', 0.0))
        info['ar_context_reject_worst_sat_res'] = float(
            ar_ctx_reject.get('worst_sat_res', 0.0))
    ar_subset_dbg = tc._ar_subset_debug
    if ar_subset_dbg:
        info['ar_subset_candidates'] = int(ar_subset_dbg.get('candidates', 0))
        if ar_subset_dbg.get('used'):
            info['ar_subset_used'] = True
            info['ar_subset_drop_sat'] = int(ar_subset_dbg.get('drop_sat', 0))
            info['ar_subset_nb'] = int(ar_subset_dbg.get('nb', 0))
            info['ar_subset_ratio'] = float(ar_subset_dbg.get('ratio', 0.0))


def _ar_starvation_reset(tc, ed):
    """Purge cold-biased float arcs after prolonged ratio starvation.

    A float that settled into a biased-but-smooth basin (e.g. after a
    mass hold loss under TDCP-class constraints) keeps every residual
    small and every AR attempt ratio-starved: none of the existing
    alarms (innovation, residual spike) can fire. Detect it as N
    consecutive lambda_zero outcomes while the DDPR residual stays
    quiet, and run the same arc purge the residual-spike recovery uses;
    fresh arcs re-anchor on DDPR and plain ILS re-fixes within epochs.
    """
    n_max = int(tc.cfg.ar_starve_reset)
    if n_max <= 0:
        return
    outcome = tc._last_ar_outcome
    if outcome == 'success':
        tc._ar_starve_streak = 0
        return
    if outcome != 'lambda_zero':
        return
    tc._ar_starve_streak = int(getattr(tc, '_ar_starve_streak', 0) or 0) + 1
    ed.info['ar_starve_streak'] = tc._ar_starve_streak
    if tc._ar_starve_streak < n_max or tc._recov_cp_hold > 0:
        return
    res_pre = tc._cached_ddpr_res_pre
    if res_pre is not None and float(res_pre) > float(tc.cfg.ar_starve_max_res):
        return                      # NLOS storm — arcs are load-bearing
    n_removed = _tc_recovery.reset_ambiguities_with_cp_hold(tc)
    ed.info['ar_starve_reset'] = n_removed
    tc._ar_starve_streak = 0


def _run_lambda_ar(tc, ed):
    """Layer 5 — pre-AR gate + write_marginals + LAMBDA AR + AR-outcome diagnostics. Always returns None."""
    # LAMBDA AR — uses the FDE-cleaned float solution.
    tc.ar_max_frac = tc.cfg.ar_max_frac
    tc.nav.smode = 5
    ed.nb = 0
    ed.xa = None
    if not _ar_eligibility(tc, ed):
        return None
    _run_ar_with_marginals(tc, ed)
    _ar_starvation_reset(tc, ed)
    _record_ar_diagnostics(tc, ed.info)
    return None


