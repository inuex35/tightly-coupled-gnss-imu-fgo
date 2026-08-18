"""Stage C1 -- assemble this epoch's factor block.

DD code/phase factors, the BetweenFactor chain on continuing ambiguities,
the propagate-prior fallback when the DD set is too thin to solve, and the
optional measurement families (Doppler raw/SD, undifferenced clock PR,
TDCP, NHC, ZUPT, bootstrap DDPR prior). Everything lands in ``ed.graph`` /
``ed.values``; nothing here talks to the smoother.
"""

import os

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


def _build_factor_block(tc, ed, prev_smode):
    """Stage C1 — DD factors + BetweenN chain + propagate-prior fallback + (Doppler / NHC / ZUPT / bootstrap-DDPR) priors."""
    info = ed.info
    # DD factor construction
    ed.nv = _tc_factors.build_dd_factors(tc, 
        ed.graph, ed.values, ed.obs, ed.obsb, ed.obs_sd,
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
                ed.graph.add(gtsam.BetweenFactorDouble(
                    k_old, k_new, 0.0,
                    tc._noise1(sig_between)))
                n_between += 1
    info['n_dd'] = ed.nv
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

    if ed.nv < tc.cfg.min_dd_for_solve:
        info['propagate_prior'] = ed.nv
        ed.graph.addPriorPose3(
            tc.Xpose(ed.kk), ed.pred.pose(),
            gtsam.noiseModel.Isotropic.Sigma(
                6, tc.cfg.propagate_pose_sigma))
        ed.graph.addPriorVector(
            tc.Vel(ed.kk), ed.pred.velocity(),
            gtsam.noiseModel.Isotropic.Sigma(
                3, tc.cfg.propagate_vel_sigma))
        ed.graph.addPriorConstantBias(
            tc.Bias(ed.kk), ed.bias_p,
            gtsam.noiseModel.Isotropic.Sigma(
                6, tc.cfg.propagate_bias_sigma))
        if not ed.skip_cp_now:
            n_anchored = 0
            amb_noise = tc._noise1(tc.cfg.propagate_amb_sigma)
            for (s, f), k_new in sorted_amb_items(tc._sat_states.amb_keys_dict()):
                if (s, f) in ed.prev_amb_tc:
                    _, n_prev = ed.prev_amb_tc[(s, f)]
                    ed.graph.add(gtsam.PriorFactorDouble(
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
            np.array(ed.estimate.atVector(tc.Vel(ed.kk - 1)))[:2]))
    except RuntimeError:
        speed_for_nhc = float(np.linalg.norm(
            np.array(ed.pred.velocity())[:2]))
    if _tc_nhc.add_nhc_factor(tc, ed.graph, ed.kk, speed_for_nhc,
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
                ed.graph.addPriorPose3(
                    tc.Xpose(ed.kk), pose_ls,
                    gtsam.noiseModel.Diagonal.Sigmas(sigmas))
                info['bootstrap_ddpr_prior_nv'] = int(n_ls)
                info['bootstrap_ddpr_prior_res'] = float(res_ls)
                info['bootstrap_ddpr_prior_sigma'] = float(boot_sigma)
        except (RuntimeError, ValueError):
            pass
        tc._tc_bootstrap_ddpr_epochs = max(0, bootstrap_ddpr_epochs - 1)

    # ────────────────────────────────────────────────────────────────


