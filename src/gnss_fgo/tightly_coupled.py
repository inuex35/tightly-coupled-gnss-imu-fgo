"""Phase 2 — moving CombinedImuFactor + DDFactorArm pipeline."""

import os
import numpy as np
import gtsam

from .buildfactor.epoch import make_epoch_data
from .utils import euler_to_R_body2enu, sensor_to_body_flu
from .preprocess import gate, stage as preprocess
from .optimize import stage as optimize
from .validation import output, postprocess
from .optimize import ar as _tc_ar
from .buildfactor import factors as _tc_factors


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
                gd['iu'], gd['ir_map'], fix['ecef'],
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
    tc.total_factor_count = g.size()

    # AR on initial graph (using last epoch's GNSS data)
    last = collected_fixes[-1]
    if 'gnss' in last:
        est_init = tc.isam2.calculateEstimate()

        # Write marginals for LAMBDA
        _tc_ar.write_marginals(tc, 
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
        os.environ.get('BOOT_DDPR_EPOCHS', '20'))
    tc._tc_bootstrap_ddpr_epochs = max(0, boot_ddpr_epochs)

    return pitch_rad, roll_rad, heading_rad, bias_acc, bias_gyro


def run_tc_epoch(tc, obs, obsb, rs, vs, dts, rsb, sat, el, iu,
                    obs_sd, ir_map, ref_vel, ref_ecef, info, ns, init_ecef, R):
    """Phase 2: IMU/GNSS TC pipeline."""
    ed = make_epoch_data(
        obs, obsb, rs, vs, dts, rsb, sat, el, iu, obs_sd, ir_map,
        ref_vel, ref_ecef, info, ns, init_ecef, R)
    for stage in (preprocess.run, gate.run,
                  optimize.run, postprocess.run,
                  output.run):
        result = stage(tc, ed)
        if result is not None:
            return result
    raise RuntimeError("tightly-coupled pipeline did not terminate")
