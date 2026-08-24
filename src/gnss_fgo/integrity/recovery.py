"""Recovery actions — the paths the runner takes when the normal
optimize/AR epoch cannot run or must be unwound: GDOP skip, IMU-only
epochs, outage adoption, warm reset, and the global CP-hold trigger."""

import numpy as np
import gtsam

from cssrlib.gnss import time2gpst

from ..factors.epoch_context import make_epoch_diagnostics
from ..factors.nhc import add_nhc_factor as _add_nhc_factor
from ..factors.zupt import add_zupt_factors as _add_zupt_factor_inplace
from ..factors import imu_preintegration as _tc_pim
from ..pipeline import update_smoother as _tc_isam


def advance_epoch_and_pack(tc, sol, tag, nb, info, obs):
    """Advance tc.epoch / nav.t and pack the (sol, tag, nb, info) result tuple."""
    tc.epoch += 1
    tc.nav.t = obs.t
    return sol, tag, nb, info




def warm_reset_phase2(tc, ecef_seed, rot_seed, break_pim=True):
    """In-place Phase 2 reset at a new ECEF position (from DDPR).

    The velocity restarts at zero: with ``break_pim`` (every current
    caller) a carried velocity seed belongs to the broken chain.
    """
    R = tc.R_enu2ecef
    enu_seed = R.T @ (ecef_seed - tc.base_ecef)
    pose_seed = gtsam.Pose3(rot_seed, gtsam.Point3(*enu_seed))
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

    tc._recov_cp_hold = effective_cp_hold_epochs(tc)
    tc.nav.x[0:3] = ecef_seed.copy()
    # Conditionally break IMU preintegration chain. See docstring.
    tc._pim_discontinuity = bool(break_pim)


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
    tc._recov_cp_hold = effective_cp_hold_epochs(tc)
    if remove_safe:
        ts = gtsam.FixedLagSmootherKeyTimestampMap()
        try:
            tc.isam2.update(gtsam.NonlinearFactorGraph(),
                             gtsam.Values(), ts, remove_safe)
        except (RuntimeError, IndexError, ValueError):
            return 0    # nothing actually left the graph
    return len(remove_safe)


def try_ddpr_reset(tc, obs, obsb, obs_sd, rs, rsb, sat, el, iu, ir_map,
                    pose_init, rot_seed, info, reason):
    """Try DDPR-only solve + warm Phase 2 reset. Returns (ecef_ddpr, ok)."""
    ecef_ddpr, _, res_rms = tc._ddpr_only_position(
        obs, obsb, obs_sd, rs, rsb, sat, el, iu, ir_map, pose_init)
    if ecef_ddpr is None:
        return None, False
    if res_rms > tc.cfg.ddpr_max_res:
        info[reason + '_untrusted'] = res_rms
        return None, False
    info[reason] = True
    warm_reset_phase2(tc, ecef_ddpr, rot_seed)
    tc._ddpr_bad_count = 0
    return ecef_ddpr, True


def _outage_mark_skip(tc, info, source):
    """Emit the ``gnss_skip`` / ``outage_source`` telemetry."""
    info['gnss_skip'] = True
    info['outage_source'] = source


def _outage_drain_imu(tc, tow_obs):
    """Advance the IMU sample cursor up to ``tow_obs`` without integrating."""
    while tc.imu_idx < len(tc.imu_data):
        if tc.imu_data[tc.imu_idx]['tow'] > tow_obs + 1e-6:
            break
        tc.imu_idx += 1


def _outage_tick_sat_outc(tc, info):
    """Tick sat_outc on every currently-tracked ambiguity key and
    expire arcs past maxout (release holds, drop keys)."""
    n_skip_reset = 0
    maxout = tc._sat_states.maxout
    for st in tc._sat_states.values():
        st.outc += 1
        if st.outc > maxout:
            st.clear_hold()
            if st.amb_key is not None:
                st.amb_key = None
                st.amb_gen += 1
                n_skip_reset += 1
    if n_skip_reset:
        info['skip_amb_reset'] = n_skip_reset


def _outage_add_pseudo_measurements(tc, graph, key_idx, info, imu_idx_prev,
                                      pose_prev, vel_prev, gyro_mean):
    """Outage-path pseudo-measurements (NHC + ZUPT/ZARU/anchor)."""
    speed_for_nhc = (
        float(np.linalg.norm(np.asarray(vel_prev, dtype=float)[:2]))
        if vel_prev is not None else 0.0)
    if _add_nhc_factor(tc, graph, key_idx, speed_for_nhc, gyro_mean_rh=gyro_mean):
        info['nhc'] = True
    if imu_idx_prev is None:
        return
    n_imu = int(info.get('n_imu', tc.imu_idx - imu_idx_prev) or 0)
    _add_zupt_factor_inplace(
        tc, graph, key_idx, imu_idx_prev, n_imu, info,
        pose_prev=pose_prev, gnss_available=False, vel_prev=vel_prev)


def _outage_anchor_bias_prior(tc, graph, key_idx):
    """Add the SKIP-only tight bias prior to ``graph`` at epoch ``key_idx``."""
    sig_acc, sig_gyro = 1e-4, 3e-6   # outage bias-anchor sigmas (measured)
    bias_anchor = tc.tc_bias if tc.tc_bias is not None else tc.tc_bias_init
    sigmas = np.array([sig_acc, sig_acc, sig_acc,
                       sig_gyro, sig_gyro, sig_gyro], dtype=np.float64)
    graph.addPriorConstantBias(
        tc.Bias(key_idx), bias_anchor,
        gtsam.noiseModel.Diagonal.Sigmas(sigmas))


def _outage_solve_and_adopt(tc, graph, values, key_idx,
                            R_enu2ecef, info, record_error=False):
    """Shared outage-epoch tail: FLS update, adopt the solved pose into
    nav.x, refresh tc.tc_bias. Returns the antenna ECEF (or the previous
    nav.x[0:3] when the solve fails)."""
    try:
        _tc_isam.fls_update(tc, graph, values, key_idx,
                         keep_keys=tc._sat_states.amb_key_values())
        estimate = tc.isam2.calculateEstimate()
        pose_tc = estimate.atPose3(tc.Xpose(key_idx))
        tc.tc_bias = estimate.atConstantBias(tc.Bias(key_idx))
        ecef_tc = R_enu2ecef @ np.array(pose_tc.translation()) + tc.base_ecef
        sol = tc._antenna_ecef(pose_tc, ecef_tc)
        tc.nav.x[0:3] = sol
        return sol
    except (RuntimeError, IndexError, ValueError) as ex:
        if record_error:
            info['error'] = str(ex)
        return tc.nav.x[0:3]


def process_imu_only(tc, obs):
    """Advance the Phase 2 graph by one epoch using IMU only (no GNSS)."""
    tc._update_epoch_dt(obs)
    tc._reset_current_epoch()
    R = tc.R_enu2ecef
    info = make_epoch_diagnostics(tc, gnss_skip=True, imu_only=True)
    _, tow_obs = time2gpst(obs.t)

    if tc.phase == 1:
        # Drain IMU to keep sample counter aligned with real time
        _outage_drain_imu(tc, tow_obs)
        return advance_epoch_and_pack(
            tc, tc.nav.x[0:3], 'FLT', 0, info, obs)

    _outage_mark_skip(tc, info, source='imu_only')

    # Phase 2: advance graph with IMU factor only
    tc.tc_epoch += 1
    key_idx = tc.tc_epoch
    info['tc_epoch'] = key_idx

    _outage_tick_sat_outc(tc, info)

    imu_idx_prev = tc.imu_idx
    pim, n_imu, gyro_mean = _tc_pim.build_pim(tc, 
        tc.tc_bias, target_tow=tow_obs)
    if n_imu == 0:
        return advance_epoch_and_pack(
            tc, tc.nav.x[0:3], 'FLT', 0, info, obs)
    info['n_imu'] = n_imu

    graph = gtsam.NonlinearFactorGraph()
    values = gtsam.Values()
    estimate = tc.isam2.calculateEstimate()
    if not estimate.exists(tc.Xpose(key_idx - 1)):
        # Previous pose marginalised — can't build IMU factor.
        return advance_epoch_and_pack(
            tc, tc.nav.x[0:3], 'FLT', 0, info, obs)

    pose_p = estimate.atPose3(tc.Xpose(key_idx - 1))
    vel_prev = estimate.atVector(tc.Vel(key_idx - 1))
    bias_prev = estimate.atConstantBias(tc.Bias(key_idx - 1))
    pred = pim.predict(gtsam.NavState(pose_p, vel_prev), bias_prev)
    values.insert(tc.Xpose(key_idx), pred.pose())
    values.insert(tc.Vel(key_idx), pred.velocity())
    values.insert(tc.Bias(key_idx), bias_prev)
    _tc_pim.add_imu_chain(tc, graph, values, key_idx, pim, pose_p, vel_prev, info)
    _outage_anchor_bias_prior(tc, graph, key_idx)
    _outage_add_pseudo_measurements(
        tc, graph, key_idx, info, imu_idx_prev, pose_p, vel_prev, gyro_mean)

    sol = _outage_solve_and_adopt(tc, graph, values, key_idx, R, info,
                                  record_error=True)

    tc.nav.smode = 5
    return advance_epoch_and_pack(tc, sol, 'FLT', 0, info, obs)


def handle_solve_exception(tc, ex, pred, bias_prev, key_idx, obs, obsb, obs_sd,
                            rs, rsb, sat, el, iu, ir_map, info):
    """Main solve failed (numerical). Try DDPR warm-reset; else IMU prior fallback."""
    info['error'] = str(ex)
    ecef_ddpr_fb, ok = try_ddpr_reset(
        tc, obs, obsb, obs_sd, rs, rsb, sat, el, iu, ir_map,
        pred.pose(), pred.pose().rotation(),
        info, 'ddpr_exception_recover')
    if ok:
        return advance_epoch_and_pack(tc, ecef_ddpr_fb, 'FLT', 0, info, obs)
    try:
        g_fb = gtsam.NonlinearFactorGraph()
        v_fb = gtsam.Values()
        v_fb.insert(tc.Xpose(key_idx), pred.pose())
        v_fb.insert(tc.Vel(key_idx), pred.velocity())
        v_fb.insert(tc.Bias(key_idx), bias_prev)
        g_fb.addPriorPose3(tc.Xpose(key_idx), pred.pose(),
            gtsam.noiseModel.Isotropic.Sigma(6, 1.0))
        g_fb.addPriorVector(tc.Vel(key_idx), pred.velocity(),
            gtsam.noiseModel.Isotropic.Sigma(3, 1.0))
        g_fb.addPriorConstantBias(tc.Bias(key_idx), bias_prev,
            gtsam.noiseModel.Isotropic.Sigma(6, 0.1))
        _tc_isam.fls_update(tc, g_fb, v_fb, key_idx)
        tc.tc_bias = bias_prev
    except (RuntimeError, IndexError, ValueError) as ex:
        info['fallback_error'] = f'{type(ex).__name__}: {ex}'
    return advance_epoch_and_pack(tc, tc.nav.x[0:3], 'FLT', 0, info, obs)


# ── CP-hold triggering (formerly state.py) ─────────────────────────

def trigger_cp_hold(tc, reason, info, value=None, skip_if_active=False):
    """Engage global CP-hold for RECOV_CP_HOLD epochs.
    Triggers: ddpr_main_res (sanity escalation) and fde_safeguard
    (runaway FDE).

    skip_if_active=True prevents re-trigger during active hold (fde/innovation
    would fire every epoch during recovery, creating infinite loop).
    """
    fired = tc._recovery.start_cp_hold(
        effective_cp_hold_epochs(tc), reason, info, value=value,
        skip_if_active=skip_if_active)
    return fired


def effective_cp_hold_epochs(tc) -> int:
    """Configured cp-hold length."""
    return int(tc.cfg.recov_cp_hold)
