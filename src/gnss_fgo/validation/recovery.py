"""Recovery actions — paths the runner takes when the normal optimize"""

import os
import numpy as np
import gtsam

from cssrlib.gnss import time2gpst

from ..buildfactor.epoch import make_epoch_diagnostics
from ..buildfactor.nhc import add_nhc_factor as _add_nhc_factor
from ..buildfactor.zupt import add_zupt_factors as _add_zupt_factor_inplace
from ..preprocess import sat_quality as _satq
from ..state import effective_cp_hold_epochs
from ..buildfactor import imu_preintegration as _tc_pim
from ..optimize import solver as _tc_solver


def finalize_epoch(tc, sol, tag, nb, info, obs):
    """Increment epoch counter, update nav.t, and return the process() tuple."""
    tc.epoch += 1
    tc.nav.t = obs.t
    return sol, tag, nb, info




def warm_reset_phase2(tc, ecef_seed, rot_seed, vel_seed=None,
                      break_pim=True):
    """In-place Phase 2 reset at a new ECEF position (from DDPR)."""
    R = tc.R_enu2ecef
    enu_seed = R.T @ (ecef_seed - tc.base_ecef)
    pose_seed = gtsam.Pose3(rot_seed, gtsam.Point3(*enu_seed))
    if vel_seed is None:
        vel_seed = np.zeros(3)
    if break_pim:
        vel_seed = np.zeros(3)

    tc.isam2 = tc._make_isam2(tc.fls_lag,
                                tc.cfg.isam2_relinearize_skip,
                                tc.cfg.isam2_relinearize_threshold)
    tc.tc_time = 0.0
    tc.tc_epoch = 0

    # Clear ambiguity / factor state (fresh DD init next epoch)
    for st in tc._sat_states.values():
        if st.amb_key is not None:
            st.amb_gen += 1
        st.clear_hold()
        st.amb_key = None
        st.amb_factor_indices = []
    tc.total_factor_count = 0

    bias0 = tc.tc_bias if tc.tc_bias is not None \
        else gtsam.imuBias.ConstantBias()

    g = gtsam.NonlinearFactorGraph()
    v = gtsam.Values()
    v.insert(tc.Xpose(0), pose_seed)
    v.insert(tc.Vel(0), vel_seed)
    v.insert(tc.Bias(0), bias0)
    rot_sigma = tc.body_rot_std * np.pi / 180
    g.addPriorPose3(tc.Xpose(0), pose_seed,
        gtsam.noiseModel.Diagonal.Sigmas(
            np.array([rot_sigma, rot_sigma, 0.3, 2.0, 2.0, 3.0])))
    g.addPriorVector(tc.Vel(0), vel_seed,
        gtsam.noiseModel.Isotropic.Sigma(3, 3.0))
    g.addPriorConstantBias(tc.Bias(0), bias0,
        gtsam.noiseModel.Isotropic.Sigma(6, 0.01))
    tc.tc_bias = bias0

    ts0 = gtsam.FixedLagSmootherKeyTimestampMap()
    tc.tc_time += tc._epoch_dt
    ts0[tc.Xpose(0)] = tc.tc_time
    ts0[tc.Vel(0)] = tc.tc_time
    ts0[tc.Bias(0)] = tc.tc_time
    tc.isam2.update(g, v, ts0)
    tc.total_factor_count += g.size()

    tc._recov_cp_hold = effective_cp_hold_epochs(tc)
    tc._recov_cp_release_streak = 0
    _satq.get_sat_quality(tc).clear()
    tc.nav.x[0:3] = ecef_seed.copy()
    tc.skip_count = 0
    # Conditionally break IMU preintegration chain. See docstring.
    tc._pim_discontinuity = bool(break_pim)
    for st in tc._sat_states.track.values():
        st.prev_phase = None


def reset_ambiguities_with_cp_hold(tc):
    """Wrong-basin recovery by N reset + N-dependent factor removal."""
    n_key_set = set(tc._sat_states.amb_key_values())
    remove_safe = []
    if n_key_set:
        try:
            facs = tc.isam2.getFactors()
            for i in range(facs.size()):
                fac = facs.at(i)
                if fac is None:
                    continue
                # any factor that names one of the current N keys
                fac_keys = fac.keys()
                for j in range(len(fac_keys)):
                    if fac_keys[j] in n_key_set:
                        remove_safe.append(i)
                        break
        except RuntimeError:
            remove_safe = []
    # Bump amb_gen and clear bookkeeping so the next epoch fresh-inits.
    for st in tc._sat_states.values():
        if st.amb_key is not None:
            st.amb_gen += 1
        st.clear_hold()
        st.amb_key = None
        st.amb_factor_indices = []
    tc._recov_cp_hold = effective_cp_hold_epochs(tc)
    tc._recov_cp_release_streak = 0
    _satq.get_sat_quality(tc).clear()
    if remove_safe:
        ts = gtsam.FixedLagSmootherKeyTimestampMap()
        try:
            tc.isam2.update(gtsam.NonlinearFactorGraph(),
                             gtsam.Values(), ts, remove_safe)
        except (RuntimeError, IndexError, ValueError):
            pass
    return len(remove_safe)


def try_ddpr_reset(tc, obs, obsb, obs_sd, rs, rsb, sat, el, iu, ir_map,
                    pose_init, rot_seed, vel_seed, info, reason):
    """Try DDPR-only solve + warm Phase 2 reset. Returns (ecef_ddpr, ok)."""
    ecef_ddpr, _, res_rms = tc._ddpr_only_position(
        obs, obsb, obs_sd, rs, rsb, sat, el, iu, ir_map, pose_init)
    if ecef_ddpr is None:
        return None, False
    if res_rms > tc.cfg.ddpr_max_res:
        info[reason + '_untrusted'] = res_rms
        return None, False
    info[reason] = True
    warm_reset_phase2(tc, ecef_ddpr, rot_seed, vel_seed)
    tc._ddpr_bad_count = 0
    return ecef_ddpr, True


def _outage_advance_skip_count(tc, info, source):
    """Increment the unified outage skip counter and emit ``gnss_skip``."""
    info['gnss_skip'] = True
    info['outage_source'] = source
    tc.skip_count += 1


def _outage_drain_imu(tc, tow_obs):
    """Advance the IMU sample cursor up to ``tow_obs`` without integrating."""
    while tc.imu_idx < len(tc.imu_data):
        if tc.imu_data[tc.imu_idx]['tow'] > tow_obs + 1e-6:
            break
        tc.imu_idx += 1


def _outage_tick_sat_outc(tc, info):
    """Tick sat_outc on every currently-tracked ambiguity key and expire"""
    skip_remove_indices = []
    n_skip_reset = 0
    maxout = tc._sat_states.maxout
    for st in tc._sat_states.values():
        st.outc += 1
        if st.outc > maxout:
            if st.amb_factor_indices:
                skip_remove_indices.extend(st.amb_factor_indices)
                st.amb_factor_indices = []
            st.clear_hold()
            if st.amb_key is not None:
                st.amb_key = None
                st.amb_gen += 1
                n_skip_reset += 1
    if n_skip_reset:
        info['skip_amb_reset'] = n_skip_reset
    return skip_remove_indices


def _outage_add_pseudo_measurements(tc, g3, kk, info, imu_idx_prev,
                                      pose_prev, vel_prev, gyro_mean):
    """Outage-path pseudo-measurements (NHC + ZUPT/ZARU/anchor)."""
    speed_for_nhc = (
        float(np.linalg.norm(np.asarray(vel_prev, dtype=float)[:2]))
        if vel_prev is not None else 0.0)
    if _add_nhc_factor(tc, g3, kk, speed_for_nhc, gyro_mean_rh=gyro_mean):
        info['nhc'] = True
    if imu_idx_prev is None:
        return
    n_imu = int(info.get('n_imu', tc.imu_idx - imu_idx_prev) or 0)
    _add_zupt_factor_inplace(
        tc, g3, kk, imu_idx_prev, n_imu, info,
        pose_prev=pose_prev, gnss_available=False, vel_prev=vel_prev)


def _outage_anchor_bias_prior(tc, g3, kk):
    """Add the SKIP-only tight bias prior to ``g3`` at epoch ``kk``."""
    legacy = os.environ.get('SKIP_BIAS_PRIOR_SIGMA')
    legacy_default = float(legacy) if legacy is not None else None
    sig_acc_default = legacy_default if legacy_default is not None else 1e-4
    sig_gyro_default = legacy_default if legacy_default is not None else 3e-6
    sig_acc = float(os.environ.get('SKIP_BIAS_PRIOR_SIGMA_ACC', sig_acc_default))
    sig_gyro = float(os.environ.get('SKIP_BIAS_PRIOR_SIGMA_GYRO', sig_gyro_default))
    if sig_acc <= 0 and sig_gyro <= 0:
        return
    if sig_acc <= 0:
        sig_acc = 1e9
    if sig_gyro <= 0:
        sig_gyro = 1e9
    bias_anchor = tc.tc_bias if tc.tc_bias is not None else tc.tc_bias_init
    sigmas = np.array([sig_acc, sig_acc, sig_acc,
                       sig_gyro, sig_gyro, sig_gyro], dtype=np.float64)
    g3.addPriorConstantBias(
        tc.Bias(kk), bias_anchor,
        gtsam.noiseModel.Diagonal.Sigmas(sigmas))


def process_gdop_skip(tc, obs, kk, g3, v3, R_enu2ecef, info,
                      imu_idx_prev=None, gyro_mean=None, vel_prev=None):
    """Bad GNSS geometry: IMU-only epoch. Advances state + keeps amb keys alive."""
    _outage_advance_skip_count(tc, info, source='gdop')
    skip_remove_indices = _outage_tick_sat_outc(tc, info)
    _outage_anchor_bias_prior(tc, g3, kk)
    try:
        est_now = tc.isam2.calculateEstimate()
        gdop_pose_prev = est_now.atPose3(tc.Xpose(kk - 1))
        gdop_vel_prev = np.array(est_now.atVector(tc.Vel(kk - 1)))
    except (RuntimeError, IndexError, ValueError):
        gdop_pose_prev = None
        gdop_vel_prev = vel_prev
    _outage_add_pseudo_measurements(
        tc, g3, kk, info, imu_idx_prev,
        gdop_pose_prev, gdop_vel_prev, gyro_mean)
    # Gauge anchor: with GNSS skipped the epoch graph is relative-only
    # (IMU between + NHC + bias prior) and consecutive skips leave the
    # pose gauge numerically unconstrained — measured divergence x3-7
    # per epoch up to 1591 km over a 16-epoch skip streak. Pin the
    # PREVIOUS pose/vel at their current estimates with the same
    # propagate sigmas the thin-epoch path uses; the IMU factor then
    # moves the new epoch freely on a bounded leash.
    if gdop_pose_prev is not None:
        g3.addPriorPose3(
            tc.Xpose(kk - 1), gdop_pose_prev,
            gtsam.noiseModel.Isotropic.Sigma(
                6, tc.cfg.propagate_pose_sigma))
    if gdop_vel_prev is not None:
        g3.addPriorVector(
            tc.Vel(kk - 1), np.asarray(gdop_vel_prev, dtype=float),
            gtsam.noiseModel.Isotropic.Sigma(
                3, tc.cfg.propagate_vel_sigma))
    try:
        _tc_solver.fls_update(tc, g3, v3, kk,
                         keep_keys=tc._sat_states.amb_key_values(),
                         remove_indices=skip_remove_indices or None)
        est2 = tc.isam2.calculateEstimate()
        pose_tc = est2.atPose3(tc.Xpose(kk))
        tc.tc_bias = est2.atConstantBias(tc.Bias(kk))
        ecef_tc = R_enu2ecef @ np.array(pose_tc.translation()) + tc.base_ecef
        tc.nav.x[0:3] = tc._antenna_ecef(pose_tc, ecef_tc)
    except (RuntimeError, IndexError, ValueError):
        pass
    tc.nav.smode = 5
    info['bias_acc'] = tc.tc_bias.accelerometer()
    info['bias_gyro'] = tc.tc_bias.gyroscope()
    return finalize_epoch(tc, tc.nav.x[0:3], 'FLT', 0, info, obs)


def process_imu_only(tc, obs):
    """Advance the Phase 2 graph by one epoch using IMU only (no GNSS)."""
    tc._update_epoch_dt(obs)
    R = tc.R_enu2ecef
    info = make_epoch_diagnostics(tc, gnss_skip=True, imu_only=True)
    _, tow_obs = time2gpst(obs.t)

    if tc.phase == 1:
        # Drain IMU to keep sample counter aligned with real time
        _outage_drain_imu(tc, tow_obs)
        return finalize_epoch(
            tc, tc.nav.x[0:3], 'FLT', 0, info, obs)

    _outage_advance_skip_count(tc, info, source='imu_only')

    # Phase 2: advance graph with IMU factor only
    tc.tc_epoch += 1
    kk = tc.tc_epoch
    info['tc_epoch'] = kk

    skip_remove_indices = _outage_tick_sat_outc(tc, info)

    imu_idx_prev = tc.imu_idx
    pim, n_imu, gyro_mean = _tc_pim.build_pim(tc, 
        tc.tc_bias, target_tow=tow_obs)
    if n_imu == 0:
        return finalize_epoch(
            tc, tc.nav.x[0:3], 'FLT', 0, info, obs)
    info['n_imu'] = n_imu

    g3 = gtsam.NonlinearFactorGraph()
    v3 = gtsam.Values()
    est2 = tc.isam2.calculateEstimate()
    if not est2.exists(tc.Xpose(kk - 1)):
        # Previous pose marginalised — can't build IMU factor.
        return finalize_epoch(
            tc, tc.nav.x[0:3], 'FLT', 0, info, obs)

    pose_p = est2.atPose3(tc.Xpose(kk - 1))
    vel_p = est2.atVector(tc.Vel(kk - 1))
    bias_p = est2.atConstantBias(tc.Bias(kk - 1))
    pred = pim.predict(gtsam.NavState(pose_p, vel_p), bias_p)
    v3.insert(tc.Xpose(kk), pred.pose())
    v3.insert(tc.Vel(kk), pred.velocity())
    v3.insert(tc.Bias(kk), bias_p)
    _tc_pim.add_imu_chain(tc, g3, v3, kk, pim, pose_p, vel_p, info)
    _outage_anchor_bias_prior(tc, g3, kk)
    _outage_add_pseudo_measurements(
        tc, g3, kk, info, imu_idx_prev, pose_p, vel_p, gyro_mean)

    try:
        _tc_solver.fls_update(tc, g3, v3, kk,
                         keep_keys=tc._sat_states.amb_key_values(),
                         remove_indices=skip_remove_indices or None)
        est2 = tc.isam2.calculateEstimate()
        pose_tc = est2.atPose3(tc.Xpose(kk))
        tc.tc_bias = est2.atConstantBias(tc.Bias(kk))
        ecef_tc = R @ np.array(pose_tc.translation()) + tc.base_ecef
        sol = tc._antenna_ecef(pose_tc, ecef_tc)
        tc.nav.x[0:3] = sol
    except (RuntimeError, IndexError, ValueError) as ex:
        info['error'] = str(ex)
        sol = tc.nav.x[0:3]

    tc.nav.smode = 5
    return finalize_epoch(tc, sol, 'FLT', 0, info, obs)


def handle_solve_exception(tc, ex, pred, bias_p, kk, obs, obsb, obs_sd,
                            rs, rsb, sat, el, iu, ir_map, info):
    """Main solve failed (numerical). Try DDPR warm-reset; else IMU prior fallback."""
    info['error'] = str(ex)
    ecef_ddpr_fb, ok = try_ddpr_reset(
        tc, obs, obsb, obs_sd, rs, rsb, sat, el, iu, ir_map,
        pred.pose(), pred.pose().rotation(), pred.velocity(),
        info, 'ddpr_exception_recover')
    if ok:
        return finalize_epoch(tc, ecef_ddpr_fb, 'FLT', 0, info, obs)
    try:
        g_fb = gtsam.NonlinearFactorGraph()
        v_fb = gtsam.Values()
        v_fb.insert(tc.Xpose(kk), pred.pose())
        v_fb.insert(tc.Vel(kk), pred.velocity())
        v_fb.insert(tc.Bias(kk), bias_p)
        g_fb.addPriorPose3(tc.Xpose(kk), pred.pose(),
            gtsam.noiseModel.Isotropic.Sigma(6, 1.0))
        g_fb.addPriorVector(tc.Vel(kk), pred.velocity(),
            gtsam.noiseModel.Isotropic.Sigma(3, 1.0))
        g_fb.addPriorConstantBias(tc.Bias(kk), bias_p,
            gtsam.noiseModel.Isotropic.Sigma(6, 0.1))
        _tc_solver.fls_update(tc, g_fb, v_fb, kk)
        tc.tc_bias = bias_p
    except (RuntimeError, IndexError, ValueError):
        pass
    return finalize_epoch(tc, tc.nav.x[0:3], 'FLT', 0, info, obs)
