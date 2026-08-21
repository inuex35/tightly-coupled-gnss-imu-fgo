"""Phase 1 — stationary GNSS-only Pose3 RTK."""

import numpy as np
import gtsam

from .utils import euler_to_R_body2enu, sensor_to_body_flu

from cssrlib.gnss import time2gpst
from . import ar as _tc_ar
from .integrity import recovery as _tc_recovery
from .factors import factors as _tc_factors


def run_init_epoch(tc, obs, obsb, rs, vs, dts, rsb, sat, el, iu,
                    obs_sd, ir_map, info, init_ecef, R):
    """Phase 1: GNSS-only Pose3 RTK on shared ambiguity keys."""
    est = _p1_build_and_solve(tc, obs, obsb, obs_sd, rs, rsb,
                              sat, el, iu, ir_map, init_ecef, R)
    sol, tag, nb, _xa = _p1_emit_and_run_ar(tc, est, obs, rs, vs, dts,
                                            sat, el, iu, R)
    _p1_collect_and_maybe_transition(tc, obs, obsb, obs_sd, rs, vs, dts,
                                       rsb, sat, el, iu, ir_map,
                                       sol, info, R)
    tc._last_sol_ecef = np.array(sol)
    return _tc_recovery.advance_epoch_and_pack(tc, sol, tag, nb, info, obs)


def _p1_fresh_restart(tc):
    """Rebuild the Phase-1 smoother from scratch (numerical-failure recovery)."""
    tc.isam = tc._make_isam2(tc.cfg.phase1_fls_lag,
                              tc.cfg.isam2_relinearize_skip,
                              tc.cfg.isam2_relinearize_threshold)
    tc.amb_keys.clear()
    tc._isam_p1_inserted = set()
    for st in tc._sat_states.values():
        st.amb_gen += 1
        st.amb_key = None


def _p1_build_and_solve(tc, obs, obsb, obs_sd, rs, rsb, sat, el, iu,
                          ir_map, init_ecef, R):
    """Phase 1A — build the per-epoch DD graph + Values, run an FLS update with three follow-up iterations, and return the smoother estimate."""
    g = gtsam.NonlinearFactorGraph()
    v = gtsam.Values()
    ep = tc.epoch

    enu_pp = R.T @ (init_ecef - tc.base_ecef)
    pose0 = gtsam.Pose3(gtsam.Rot3.Identity(), gtsam.Point3(*enu_pp))
    prev_missing = (ep > 0 and tc.Xp(ep - 1) not in tc._isam_p1_inserted)
    fresh_start = ep == 0 or prev_missing
    if fresh_start:
        # The RINEX header position can be arbitrarily stale (km-level on
        # some rovers); anchoring Phase 1 there with a 3 m prior would
        # never converge. Seed the anchor from a code-only DD LS instead —
        # its linearization is insensitive to km-level initial error.
        try:
            ecef_ls, n_ls, res_ls = tc._ddpr_only_position(
                obs, obsb, obs_sd, rs, rsb, sat, el, iu, ir_map, pose0)
            if ecef_ls is not None and n_ls >= 4 and np.isfinite(res_ls):
                # Re-anchor everything that depends on the a-priori
                # position: the pose prior AND the geometry used to seed
                # the ambiguities (a km-wrong init_ecef puts thousands of
                # cycles of error into the N inits, whose priors then drag
                # the pose away from the measurements).
                init_ecef = np.asarray(ecef_ls, dtype=float)
                tc.nav.x[0:3] = init_ecef
                enu_pp = R.T @ (init_ecef - tc.base_ecef)
                pose0 = gtsam.Pose3(gtsam.Rot3.Identity(),
                                    gtsam.Point3(*enu_pp))
        except (RuntimeError, ValueError):
            pass
    if fresh_start:
        g.addPriorPose3(tc.Xp(ep), pose0,
            gtsam.noiseModel.Diagonal.Sigmas(
                np.array([0.01, 0.01, 0.01, 3, 3, 3])))
    else:
        g.add(gtsam.BetweenFactorPose3(
            tc.Xp(ep - 1), tc.Xp(ep),
            gtsam.Pose3.Identity(),
            gtsam.noiseModel.Diagonal.Sigmas(
                np.array([0.001, 0.001, 0.001, 0.1, 0.1, 0.1]))))
    v.insert(tc.Xp(ep), pose0)

    lever = gtsam.Point3(0, 0, 0)
    _tc_factors.build_dd_factors(tc,
        g, v, obs, obsb, obs_sd, rs, rsb, sat, el, iu, ir_map,
        tc.Xp(ep), lever, tc.amb_keys,
        dd_epoch=0)

    for k in list(v.keys()):
        if k in tc._isam_p1_inserted:
            v.erase(k)
    tc.phase1_t += tc._epoch_dt
    ts = gtsam.FixedLagSmootherKeyTimestampMap()
    ts[tc.Xp(ep)] = tc.phase1_t
    if not fresh_start:
        ts[tc.Xp(ep - 1)] = tc.phase1_t
    for k in v.keys():
        ts[k] = tc.phase1_t
    for k in tc.amb_keys.values():
        ts[k] = tc.phase1_t
    try:
        tc.isam.update(g, v, ts)
    except RuntimeError:
        # Numerical failure in the Phase-1 smoother (e.g. indeterminate
        # system from a degenerate ambiguity/pose combination). A GNSS-only
        # bootstrap holds no irreplaceable state — degrade gracefully by
        # restarting Phase 1 fresh instead of crashing the run.
        _p1_fresh_restart(tc)
        g = gtsam.NonlinearFactorGraph()
        v = gtsam.Values()
        g.addPriorPose3(tc.Xp(ep), pose0,
            gtsam.noiseModel.Diagonal.Sigmas(
                np.array([0.01, 0.01, 0.01, 3, 3, 3])))
        v.insert(tc.Xp(ep), pose0)
        _tc_factors.build_dd_factors(tc,
            g, v, obs, obsb, obs_sd, rs, rsb, sat, el, iu, ir_map,
            tc.Xp(ep), lever, tc.amb_keys,
            dd_epoch=0)
        tc.phase1_t += tc._epoch_dt
        ts = gtsam.FixedLagSmootherKeyTimestampMap()
        for k in v.keys():
            ts[k] = tc.phase1_t
        try:
            tc.isam.update(g, v, ts)
        except (RuntimeError, IndexError):
            # A second consecutive failure used to take the whole run
            # down (r5 #11); skip the epoch instead.
            return None
    # Record everything we just inserted so the next epoch can skip.
    tc._isam_p1_inserted.add(tc.Xp(ep))
    for k in v.keys():
        tc._isam_p1_inserted.add(k)
    for _ in range(3):
        tc.isam.update(gtsam.NonlinearFactorGraph(), gtsam.Values(),
                        gtsam.FixedLagSmootherKeyTimestampMap())
    est = tc.isam.calculateEstimate()

    # Drop marginalized variables from the inserted-set: a satellite that
    # left amb_keys (ref switch) gets marginalized out of the smoother;
    # when it reappears under the same key, the stale record would erase
    # its fresh Values insert and the new CP factor would reference a
    # variable the smoother no longer has (IndeterminantLinearSystem).
    tc._isam_p1_inserted = {
        k for k in tc._isam_p1_inserted if est.exists(k)}

    return est


def _p1_emit_and_run_ar(tc, est, obs, rs, vs, dts, sat, el, iu, R):
    """Phase 1B — write nav.x from the Phase-1 estimate, populate marginals, run LAMBDA AR (only after the warm-up window), and emit the (sol, tag, nb, xa) tuple."""
    ep = tc.epoch
    enu_e = np.array(est.atPose3(tc.Xp(ep)).translation())
    tc.nav.x[0:3] = R @ enu_e + tc.base_ecef

    _tc_ar.nav_bridge.publish_marginals(tc, 
        tc.isam.getFactors(), est, tc.Xp(ep), tc.amb_keys)

    tc.nav.smode = 5
    nb = 0
    xa = None
    if ep >= 5:
        nb, xa = _tc_ar.run_ar(tc, 
            obs, rs, vs, dts, sat, el, iu, est,
            tc.Xp(ep), tc.amb_keys)

    sol = xa[0:3] if nb > 0 and tc.nav.smode == 4 else tc.nav.x[0:3]
    tag = 'FIX' if tc.nav.smode == 4 else 'FLT'

    return sol, tag, nb, xa


def _p1_collect_and_maybe_transition(tc, obs, obsb, obs_sd, rs, vs, dts,
                                       rsb, sat, el, iu, ir_map,
                                       sol, info, R):
    """Phase 1C — collect IMU samples + estimate velocity from a 1.5-second sliding Fix-position window, accumulate fixes once we are moving, and trigger Phase-2 init when ``n_collect`` fixes are in hand."""
    # Collect IMU samples up to current GNSS epoch
    _, _tow_obs = time2gpst(obs.t)
    imu_samples = tc._collect_imu_samples(target_tow=_tow_obs)

    if tc.nav.smode == 4:
        tc._sol_hist.append((_tow_obs, sol.copy()))
        while tc._sol_hist and _tow_obs - tc._sol_hist[0][0] > 1.5:
            tc._sol_hist.pop(0)

    est_vel_enu = np.zeros(3)
    if len(tc._sol_hist) >= 2:
        t0, s0 = tc._sol_hist[0]
        t1, s1 = tc._sol_hist[-1]
        dt = t1 - t0
        if dt >= 0.8:
            est_vel_enu = R.T @ (s1 - s0) / dt

    # Start collecting when moving + Fix
    vel_mag = np.linalg.norm(est_vel_enu[:2])
    if not tc.collecting and vel_mag > tc.cfg.vel_thresh and tc.nav.smode == 4:
        tc.collecting = True

    # Accumulate Fix positions during collection
    if tc.collecting and tc.nav.smode == 4:
        tc.collected_fixes.append({
            'ecef': sol.copy(),
            'vel': est_vel_enu.copy(),
            'imu': imu_samples,
            'gnss': {
                'obs': obs, 'obsb': obsb, 'obs_sd': obs_sd,
                'rs': rs.copy(), 'vs': vs.copy(), 'dts': dts.copy(),
                'rsb': rsb.copy(),
                'sat': sat.copy(), 'el': el.copy(),
                'iu': iu.copy(), 'ir_map': dict(ir_map),
            }
        })
        info['n_collected'] = len(tc.collected_fixes)

        # Check if we have enough
        if len(tc.collected_fixes) >= tc.cfg.n_collect:
            p, r, h, ba0, bg0 = transition_to_tc(tc, tc.collected_fixes)
            info['transition'] = True
            info['pitch'] = np.degrees(p)
            info['roll'] = np.degrees(r)
            info['heading'] = np.degrees(h)
            info['bias_acc_init'] = ba0
            info['bias_gyro_init'] = bg0
            info['n_amb_tc'] = tc._sat_states.amb_count()
    elif tc.collecting and tc.nav.smode != 4:
        # Lost Fix during collection — reset
        tc.collecting = False
        tc.collected_fixes = []


# ── Phase-1 → Phase-2 handoff ────────────────────────────────────
# The transition seeds the TC graph from the collected Phase-1 fixes;
# it belongs to initialization, not to the Phase-2 epoch loop.

def _init_tc_heading_series(tc, collected_fixes):
    """Smooth per-fix heading seed from the collected fix path in ENU."""
    R = tc.R_enu2ecef
    enu = np.array([R.T @ (fix['ecef'] - tc.base_ecef) for fix in collected_fixes])
    vel_mean = np.mean(
        [fix.get('vel', np.zeros(3)) for fix in collected_fixes], axis=0)
    total_disp = enu[-1] - enu[0]
    if np.linalg.norm(total_disp[:2]) > 0.01:
        heading_seed = float(np.arctan2(total_disp[0], total_disp[1]))
    else:
        heading_seed = tc._heading_from_vel(
            vel_mean, fallback=0.0, disp_enu=total_disp)

    heading_series = []
    prev_h = heading_seed
    n = len(collected_fixes)
    for i in range(n):
        if n == 1:
            disp = total_disp
        elif i == 0:
            disp = enu[1] - enu[0]
        elif i == n - 1:
            disp = enu[-1] - enu[-2]
        else:
            disp = enu[i + 1] - enu[i - 1]
        if np.linalg.norm(disp[:2]) > 0.01:
            h_i = float(np.arctan2(disp[0], disp[1]))
        else:
            h_i = prev_h
        heading_series.append(h_i)
        prev_h = h_i
    return heading_seed, heading_series


def add_tc_seed_epoch(tc, g, v, i, fix, bias0,
                           roll_rad, pitch_rad, heading_i,
                           body_rot_std_rad, lever_arr):
    """Add pose/vel/bias values and factors for one collected-fix epoch."""
    R = tc.R_enu2ecef
    fix_ecef = fix['ecef']
    fix_enu = R.T @ (fix_ecef - tc.base_ecef)
    vel_i = fix.get('vel', np.zeros(3))
    speed_i = np.linalg.norm(vel_i[:2])

    R_b2e = euler_to_R_body2enu(roll_rad, pitch_rad, heading_i)
    body_enu = np.asarray(fix_enu) - R_b2e @ np.asarray(lever_arr)
    pose_i = gtsam.Pose3(gtsam.Rot3(R_b2e), gtsam.Point3(*body_enu))

    v.insert(tc.Xpose(i), pose_i)
    v.insert(tc.Vel(i), vel_i)
    v.insert(tc.Bias(i), bias0)

    if i == 0:
        # GICI-style: roll/pitch tight, yaw σ from velocity
        std_yaw = np.sqrt(body_rot_std_rad**2 + (0.1 / speed_i)**2) \
            if speed_i > 0.1 else 0.5
        g.addPriorPose3(tc.Xpose(0), pose_i,
            gtsam.noiseModel.Diagonal.Sigmas(
                np.array([0.02, 0.02, std_yaw, 0.05, 0.05, 0.1])))
        g.addPriorVector(tc.Vel(0), vel_i,
            gtsam.noiseModel.Isotropic.Sigma(3, 0.5))
        g.addPriorConstantBias(tc.Bias(0), bias0,
            gtsam.noiseModel.Isotropic.Sigma(6, 0.01))
    else:
        std_yaw = np.sqrt(body_rot_std_rad**2 + (0.2 / max(speed_i, 0.2))**2)
        g.addPriorPose3(
            tc.Xpose(i), pose_i,
            gtsam.noiseModel.Diagonal.Sigmas(
                np.array([0.01, 0.01, std_yaw, 0.10, 0.10, 0.10])))
        g.addPriorVector(tc.Vel(i), vel_i,
            gtsam.noiseModel.Isotropic.Sigma(3, 0.5))
        g.addPriorConstantBias(tc.Bias(i), bias0,
            gtsam.noiseModel.Isotropic.Sigma(6, 0.01))
        pim, _, _ = tc._build_pim_from_samples(bias0, fix['imu'])
        g.add(gtsam.CombinedImuFactor(
            tc.Xpose(i-1), tc.Vel(i-1),
            tc.Xpose(i), tc.Vel(i),
            tc.Bias(i-1), tc.Bias(i), pim))

    g.add(gtsam.GPSFactorArm(
        tc.Xpose(i), fix_enu, lever_arr,
        gtsam.noiseModel.Isotropic.Sigma(3, 0.02)))


def transition_to_tc(tc, collected_fixes):
    """Initialize Phase 2 with accumulated Fix positions + IMU."""
    R = tc.R_enu2ecef
    n = len(collected_fixes)

    # IMU bias from stationary pre-motion window
    bias_acc, bias_gyro = tc._estimate_stationary_bias()
    bias0 = gtsam.imuBias.ConstantBias(bias_acc, bias_gyro)
    tc.tc_bias = bias0
    tc.tc_bias_init = bias0  # anchor for weak prior

    n_static = min(500, len(tc.imu_data))
    acc_avg = np.mean([im['acc'] for im in tc.imu_data[:n_static]], axis=0)
    acc_tilt = sensor_to_body_flu(np.array(acc_avg - bias_acc, dtype=float))
    pitch_rad = np.arctan2(
        acc_tilt[0], np.sqrt(acc_tilt[1]**2 + acc_tilt[2]**2))
    roll_rad = np.arctan2(acc_tilt[1], acc_tilt[2])
    if np.isfinite(tc.cfg.init_pitch_deg):
        pitch_rad = np.deg2rad(float(tc.cfg.init_pitch_deg))

    body_rot_std_rad = tc.body_rot_std * np.pi / 180
    heading_rad, heading_series = _init_tc_heading_series(
        tc, collected_fixes)

    # IMU PIM params
    tc.imu_params = tc._make_imu_params()

    # Lever arm: enabled from start
    tc.lever_arm_tc = gtsam.Point3(*tc.lever_arm)
    lever_arr = tc.lever_arm

    # FixedLagSmoother with epoch-specific N
    tc.fls_lag = tc.cfg.fls_lag
    tc.isam2 = tc._make_isam2(tc.fls_lag,
                                tc.cfg.isam2_relinearize_skip,
                                tc.cfg.isam2_relinearize_threshold)
    tc.tc_time = 0.0

    # Build initial graph with all collected epochs
    g = gtsam.NonlinearFactorGraph()
    v = gtsam.Values()

    for i in range(n):
        add_tc_seed_epoch(
            tc, g, v, i, collected_fixes[i], bias0,
            roll_rad, pitch_rad, heading_series[i],
            body_rot_std_rad, lever_arr)

    tc.amb_keys_tc = {}
    for i in range(n):
        fix = collected_fixes[i]
        if 'gnss' in fix:
            gd = fix['gnss']
            _tc_factors.build_dd_factors(tc, 
                g, v, gd['obs'], gd['obsb'], gd['obs_sd'],
                gd['rs'], gd['rsb'], gd['sat'], gd['el'],
                gd['iu'], gd['ir_map'],
                tc.Xpose(i), tc.lever_arm_tc, tc.amb_keys_tc,
                dd_epoch=0)  # shared key (dd_epoch=0, no prev)

    # Timestamps: spread within lag window (real seconds, dt apart)
    dt = tc._epoch_dt
    ts = gtsam.FixedLagSmootherKeyTimestampMap()
    for i in range(n):
        t_i = float(i) * dt
        ts[tc.Xpose(i)] = t_i
        ts[tc.Vel(i)] = t_i
        ts[tc.Bias(i)] = t_i
    # All variables in v need timestamps
    for key in v.keys():
        if key not in ts:
            ts[key] = float(n - 1) * dt  # amb keys at last epoch
    tc.tc_time = float(n - 1) * dt
    # Ensure lag covers all init epochs
    if tc.fls_lag < float(n) * dt:
        tc.fls_lag = float(n) * dt + dt
        tc.isam2 = tc._make_isam2(tc.fls_lag,
                                tc.cfg.isam2_relinearize_skip,
                                tc.cfg.isam2_relinearize_threshold)
    tc.isam2.update(g, v, ts)

    # AR on initial graph (using last epoch's GNSS data)
    last = collected_fixes[-1]
    if 'gnss' in last:
        est_init = tc.isam2.calculateEstimate()

        # Write marginals for LAMBDA
        _tc_ar.nav_bridge.publish_marginals(tc, 
            tc.isam2.getFactors(), est_init,
            tc.Xpose(n - 1), tc._sat_states.amb_keys_dict())

        # Write antenna position to nav.x
        pose_last = est_init.atPose3(tc.Xpose(n - 1))
        enu_last = np.array(pose_last.translation())
        ecef_last = R @ enu_last + tc.base_ecef
        tc.nav.x[0:3] = tc._antenna_ecef(pose_last, ecef_last)

        tc.nav.smode = 5

    tc.tc_epoch = n - 1
    tc.phase = 2
    tc._tc_fresh_amb_epochs = 0
    boot_ddpr_epochs = int(
        tc.cfg.boot_ddpr_epochs)
    tc._tc_bootstrap_ddpr_epochs = max(0, boot_ddpr_epochs)

    return pitch_rad, roll_rad, heading_rad, bias_acc, bias_gyro
