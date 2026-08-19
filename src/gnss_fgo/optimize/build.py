"""Stage C1 -- assemble this epoch's factor block.

DD code/phase factors, the BetweenFactor chain on continuing ambiguities,
the propagate-prior fallback when the DD set is too thin to solve, and the
optional measurement families (Doppler raw/SD, undifferenced clock PR,
TDCP, NHC, ZUPT, bootstrap DDPR prior). Everything lands in ``epoch.graph`` /
``epoch.values``; nothing here talks to the smoother.
"""


import numpy as np
import gtsam

from ..buildfactor import clock as _tc_clock
from ..buildfactor import doppler as _tc_doppler
from ..buildfactor import doppler_sd as _tc_doppler_sd
from ..buildfactor import tdcp as _tc_tdcp
from ..buildfactor import factors as _tc_factors
from ..buildfactor import nhc as _tc_nhc
from ..buildfactor import zupt as _tc_zupt
from .. import sat_quality as _satq
from ..utils import sorted_amb_items


def _build_factor_block(tc, epoch, prev_smode):
    """Stage C1 — DD factors + BetweenN chain + propagate-prior fallback + (Doppler / NHC / ZUPT / bootstrap-DDPR) priors."""
    info = epoch.info
    # DD factor construction
    nv = _tc_factors.build_dd_factors(tc, 
        epoch.graph, epoch.values, epoch.obs, epoch.obsb, epoch.obs_sd,
        epoch.rs, epoch.rsb, epoch.sat, epoch.el, epoch.iu, epoch.ir_map,
        epoch.pred_ecef, tc.Xpose(epoch.key_idx), tc.lever_arm_tc,
        tc.amb_keys_tc,
        track_indices=True, dd_epoch=epoch.key_idx,
        prev_amb_values=epoch.prev_amb_values,
        skip_cp=epoch.skip_cp_now, slip_keys=epoch.slip_keys)
    epoch.nv = nv
    sq = _satq.get_sat_quality(tc)
    sat_lock_age = {}
    for s in epoch.sat:
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
    if not epoch.skip_cp_now and tc.cfg.betweenn_enable:
        for (s, f), k_new in sorted_amb_items(tc._sat_states.amb_keys_dict()):
            if (s, f) in epoch.prev_amb_values:
                k_old, _ = epoch.prev_amb_values[(s, f)]
                if last_flt:
                    sig_between = sig_between_flt
                elif warmup > 0 and streak_map is not None:
                    streak = streak_map.get((s, f), 0)
                    sig_between = (sig_between_fix if streak >= warmup
                                   else sig_between_flt)
                else:
                    sig_between = sig_between_fix
                epoch.graph.add(gtsam.BetweenFactorDouble(
                    k_old, k_new, 0.0,
                    tc._noise1(sig_between)))
                n_between += 1
    info['n_dd'] = epoch.nv
    if tc._last_hold_gauge_rel:
        info['hold_gauge_rel'] = list(tc._last_hold_gauge_rel)
        tc._last_hold_gauge_rel = []
    cp_pr_rej = tc._last_cp_pr_reject
    rejc_wipe = tc._last_rejc_wipe
    if cp_pr_rej:
        info['cp_pr_reject'] = cp_pr_rej
    if rejc_wipe:
        info['rejc_wipe'] = rejc_wipe
    tc._last_cp_pr_reject = 0
    tc._last_rejc_wipe = 0

    if epoch.nv < tc.cfg.min_dd_for_solve:
        info['propagate_prior'] = epoch.nv
        epoch.graph.addPriorPose3(
            tc.Xpose(epoch.key_idx), epoch.pred_nav.pose(),
            gtsam.noiseModel.Isotropic.Sigma(
                6, tc.cfg.propagate_pose_sigma))
        epoch.graph.addPriorVector(
            tc.Vel(epoch.key_idx), epoch.pred_nav.velocity(),
            gtsam.noiseModel.Isotropic.Sigma(
                3, tc.cfg.propagate_vel_sigma))
        epoch.graph.addPriorConstantBias(
            tc.Bias(epoch.key_idx), epoch.bias_prev,
            gtsam.noiseModel.Isotropic.Sigma(
                6, tc.cfg.propagate_bias_sigma))
        if not epoch.skip_cp_now:
            n_anchored = 0
            amb_noise = tc._noise1(tc.cfg.propagate_amb_sigma)
            for (s, f), k_new in sorted_amb_items(tc._sat_states.amb_keys_dict()):
                if (s, f) in epoch.prev_amb_values:
                    _, n_prev = epoch.prev_amb_values[(s, f)]
                    epoch.graph.add(gtsam.PriorFactorDouble(
                        k_new, n_prev, amb_noise))
                    n_anchored += 1
            info['n_anchored'] = n_anchored
    info['n_between'] = n_between

    # Raw per-satellite Doppler factors (opt-in via cfg.doppler_sigma)
    _tc_doppler.add_doppler_factors(tc, epoch)

    # Undifferenced pseudoranges pinning the clock chain the Doppler needs
    _tc_clock.add_clock_pr_factors(tc, epoch)

    # Between-satellite differenced Doppler (clock-free; cfg.doppler_sd_sigma)
    _tc_doppler_sd.add_sd_doppler_factors(tc, epoch)

    # TDCP relative-displacement constraints (rover-only carrier deltas)
    _tc_tdcp.add_tdcp_factors(tc, epoch)

    try:
        speed_for_nhc = float(np.linalg.norm(
            np.array(epoch.estimate.atVector(tc.Vel(epoch.key_idx - 1)))[:2]))
    except RuntimeError:
        speed_for_nhc = float(np.linalg.norm(
            np.array(epoch.pred_nav.velocity())[:2]))
    if _tc_nhc.add_nhc_factor(tc, epoch.graph, epoch.key_idx, speed_for_nhc,
                            gyro_mean_rh=epoch.gyro_mean):
        info['nhc'] = True

    _tc_zupt.add_zupt_factors_for_stage(tc, epoch)

    bootstrap_ddpr_epochs = int(
        tc._tc_bootstrap_ddpr_epochs or 0)
    if bootstrap_ddpr_epochs > 0:
        try:
            ecef_ls, n_ls, res_ls = tc._ddpr_only_position(
                epoch.obs, epoch.obsb, epoch.obs_sd, epoch.rs, epoch.rsb,
                epoch.sat, epoch.el, epoch.iu, epoch.ir_map, epoch.pred_nav.pose())
            if ecef_ls is not None and n_ls >= 4:
                body_enu_ls = epoch.R_enu2ecef.T @ (np.asarray(ecef_ls) - tc.base_ecef)
                pose_ls = gtsam.Pose3(
                    epoch.pred_nav.pose().rotation(),
                    gtsam.Point3(*body_enu_ls))
                boot_sigma = float(tc.cfg.boot_ddpr_sigma)
                sigmas = np.array([1e6, 1e6, 1e6,
                                   boot_sigma, boot_sigma, boot_sigma])
                epoch.graph.addPriorPose3(
                    tc.Xpose(epoch.key_idx), pose_ls,
                    gtsam.noiseModel.Diagonal.Sigmas(sigmas))
                info['bootstrap_ddpr_prior_nv'] = int(n_ls)
                info['bootstrap_ddpr_prior_res'] = float(res_ls)
                info['bootstrap_ddpr_prior_sigma'] = float(boot_sigma)
        except (RuntimeError, ValueError):
            pass
        tc._tc_bootstrap_ddpr_epochs = max(0, bootstrap_ddpr_epochs - 1)

    # ────────────────────────────────────────────────────────────────


