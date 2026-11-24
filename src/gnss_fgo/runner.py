"""ImuGnssTc — single-phase IMU/GNSS tightly-coupled FGO runner & epoch flow."""

import gtsam
import numpy as np
from cssrlib.gnss import time2gpst, timediff
from cssrlib.rtk import rtkpos

from . import ar as _tc_ar
from . import factors as _tc_factors
from .config import TcConfig
from .factors import SatStateMap
from .utils import (
    build_imu_preintegration,
    enu_frame,
    estimate_stationary_bias,
    euler_to_R_body2enu,
    make_imu_params,
    sorted_amb_items,
)


# ── FixedLagSmoother helpers ──────────────────────────────────────────

def update_smoother(tc, graph, values, key_idx, keep_keys=()):
    """Update FLS while keeping the previous pose/vel/bias and live N keys alive."""
    tc.tc_time += tc._epoch_dt
    ts = gtsam.FixedLagSmootherKeyTimestampMap()
    t = tc.tc_time
    ts[tc.Pose(key_idx)] = t
    ts[tc.Vel(key_idx)] = t
    ts[tc.Bias(key_idx)] = t
    if key_idx > 0:
        ts[tc.Pose(key_idx - 1)] = t
        ts[tc.Vel(key_idx - 1)] = t
        ts[tc.Bias(key_idx - 1)] = t
    for k in values.keys():
        if k not in ts:
            ts[k] = t
    for k in keep_keys:
        ts[k] = t
    tc.isam2.update(graph, values, ts)


def _solve_isam2(tc, graph, values, estimate, key_idx, R_enu2ecef, obs, info):
    """Update FixedLagSmoother and return (pose_tc, ecef_tc, early_result)."""
    try:
        live_amb_keys = [
            k for (_sf, k) in sorted_amb_items(tc._sat_states.amb_keys_dict())
            if estimate.exists(k) or values.exists(k)
        ]
        update_smoother(tc, graph, values, key_idx, keep_keys=live_amb_keys)
        new_est = tc.isam2.calculateEstimate()
    except (RuntimeError, IndexError, ValueError) as ex:
        info['error'] = str(ex)
        return None, None, (np.array(tc.nav.x[0:3]), 'FLT', 0, info)

    pose_tc = new_est.atPose3(tc.Pose(key_idx))
    tc.tc_bias = new_est.atConstantBias(tc.Bias(key_idx))
    enu_tc = np.array(pose_tc.translation())
    ecef_tc = R_enu2ecef @ enu_tc + tc.base_ecef
    return pose_tc, ecef_tc, None


# ── Epoch pipeline ────────────────────────────────────────────────────

def _decide_fix_or_flt(tc, float_antenna_ecef, n_fixed, fixed_pos, info):
    """Pure FIX/FLT decision directly from LAMBDA acceptance (nav.smode == 4)."""
    if tc.nav.smode == 4 and n_fixed > 0 and fixed_pos is not None:
        info['lambda_correction'] = float(np.linalg.norm(fixed_pos[0:3] - float_antenna_ecef))
        return fixed_pos[0:3], 'FIX', n_fixed
    return float_antenna_ecef, 'FLT', n_fixed


def _init_pose_and_bias_epoch0(tc, obs, init_ecef, R_enu2ecef):
    """Estimate stationary IMU bias & tilt from pre-GNSS samples and build pose0."""
    _, tow_obs = time2gpst(obs.t)
    while tc.imu_idx < len(tc.imu_data) and tc.imu_data[tc.imu_idx]['tow'] <= tow_obs + 1e-6:
        tc.imu_idx += 1
    bias_acc, bias_gyro = estimate_stationary_bias(
        tc.imu_data, n_max=min(3000, tc.imu_idx))
    bias0 = gtsam.imuBias.ConstantBias(bias_acc, bias_gyro)
    tc.tc_bias = bias0

    n_static = max(1, min(500, tc.imu_idx))
    acc_avg = np.mean([im['acc'] for im in tc.imu_data[:n_static]], axis=0)
    acc_tilt = np.asarray(acc_avg - bias_acc, dtype=float)
    pitch_rad = float(np.arctan2(
        acc_tilt[0], np.sqrt(acc_tilt[1] ** 2 + acc_tilt[2] ** 2)))
    roll_rad = float(np.arctan2(acc_tilt[1], acc_tilt[2]))
    rot0 = gtsam.Rot3(euler_to_R_body2enu(roll_rad, pitch_rad, 0.0))

    enu0 = R_enu2ecef.T @ (init_ecef - tc.base_ecef)
    body_enu0 = enu0 - rot0.matrix() @ tc.lever_arm
    pose0 = gtsam.Pose3(rot0, gtsam.Point3(*body_enu0))
    return pose0, bias0


def _bootstrap_epoch0(tc, obs, obsb, rs, rsb, sat, el, iu, obs_sd, ir_map,
                      init_ecef, R_enu2ecef, info):
    """Initialize the single smoother at the first valid GNSS epoch."""
    # 1. Stationary IMU alignment (bias, roll/pitch, lever-arm-compensated pose0)
    pose0, bias0 = _init_pose_and_bias_epoch0(tc, obs, init_ecef, R_enu2ecef)
    tc.imu_params = make_imu_params()
    tc.lever_arm_tc = gtsam.Point3(*tc.lever_arm)

    # 2. FixedLagSmoother setup and epoch-0 priors + DD factors
    params = gtsam.ISAM2Params()
    params.setFactorization("QR")
    params.setRelinearizeThreshold(0.01)
    params.relinearizeSkip = 1
    tc.isam2 = gtsam.IncrementalFixedLagSmoother(tc.cfg.fls_lag, params)
    tc.tc_time = 0.0
    tc.smoother_node_idx = 0
    info['tc_epoch'] = 0

    g = gtsam.NonlinearFactorGraph()
    v = gtsam.Values()
    v.insert(tc.Pose(0), pose0)
    v.insert(tc.Vel(0), np.zeros(3))
    v.insert(tc.Bias(0), bias0)
    rot_sigma = np.deg2rad(2.0)
    g.addPriorPose3(tc.Pose(0), pose0,
                    gtsam.noiseModel.Diagonal.Sigmas(
                        np.array([rot_sigma, rot_sigma, np.pi, 1e4, 1e4, 1e4])))
    g.addPriorVector(tc.Vel(0), np.zeros(3),
                     gtsam.noiseModel.Isotropic.Sigma(3, 1.0))
    g.addPriorConstantBias(tc.Bias(0), bias0,
                           gtsam.noiseModel.Isotropic.Sigma(6, 0.01))

    _tc_factors.build_dd_factors(
        tc, g, v, obs, obsb, obs_sd, rs, rsb, sat, el, iu, ir_map,
        tc.Pose(0), tc.lever_arm_tc, existing_estimate=None)

    # 3. Solve + extra relinearization passes to settle unobservably-yawed pose0
    update_smoother(tc, g, v, 0)
    for _ in range(3):
        tc.isam2.update(gtsam.NonlinearFactorGraph(), gtsam.Values(),
                        gtsam.FixedLagSmootherKeyTimestampMap())
    est = tc.isam2.calculateEstimate()
    pose_tc = est.atPose3(tc.Pose(0))
    ecef_tc = R_enu2ecef @ np.array(pose_tc.translation()) + tc.base_ecef
    sol = tc._antenna_ecef(pose_tc, ecef_tc)
    tc.nav.x[0:3] = sol
    tc.nav.smode = 5
    return sol, 'FLT', 0, info


def run_tc_epoch(tc, obs, obsb, rs, vs, rsb, sat, el, iu,
                 obs_sd, ir_map, info, ns, init_ecef, R_enu2ecef):
    """Single-phase IMU/GNSS TC epoch flow."""
    if tc.isam2 is None:
        if ns < 4:
            return np.array(tc.nav.x[0:3]), 'FLT', 0, info
        return _bootstrap_epoch0(
            tc, obs, obsb, rs, rsb, sat, el, iu, obs_sd, ir_map,
            init_ecef, R_enu2ecef, info)

    # 1. IMU preintegration & state prediction
    tc.smoother_node_idx += 1
    key_idx = tc.smoother_node_idx
    info['tc_epoch'] = key_idx
    _, tow_obs = time2gpst(obs.t)
    pim, n_imu, gyro_mean, tc.imu_idx = build_imu_preintegration(
        tc.imu_params, tc.tc_bias, tc.imu_data, tc.imu_idx, target_tow=tow_obs)
    info['n_imu'] = n_imu
    if n_imu == 0:
        return np.array(tc.nav.x[0:3]), 'FLT', 0, info

    graph = gtsam.NonlinearFactorGraph()
    values = gtsam.Values()
    estimate = tc.isam2.calculateEstimate()

    if not estimate.exists(tc.Pose(key_idx - 1)):
        info['prev_pose_missing'] = key_idx - 1
        return np.array(tc.nav.x[0:3]), 'FLT', 0, info

    pose_prev = estimate.atPose3(tc.Pose(key_idx - 1))
    vel_prev = estimate.atVector(tc.Vel(key_idx - 1))
    bias_prev = estimate.atConstantBias(tc.Bias(key_idx - 1))
    pred_nav = pim.predict(gtsam.NavState(pose_prev, vel_prev), bias_prev)
    values.insert(tc.Pose(key_idx), pred_nav.pose())
    values.insert(tc.Vel(key_idx), pred_nav.velocity())
    values.insert(tc.Bias(key_idx), bias_prev)
    graph.add(gtsam.CombinedImuFactor(
        tc.Pose(key_idx - 1), tc.Vel(key_idx - 1),
        tc.Pose(key_idx), tc.Vel(key_idx),
        tc.Bias(key_idx - 1), tc.Bias(key_idx), pim))

    # 2. Cycle-slip detection & predicted antenna ECEF
    info['nsat'] = ns
    n_reset, _ = _tc_factors.detect_slips_and_reset_ambiguities(
        tc, obs, obs_sd, sat, iu)
    info['n_slip'] = n_reset
    pred_enu = np.array(pred_nav.pose().translation())
    pred_ecef = tc._antenna_ecef(
        pred_nav.pose(), R_enu2ecef @ pred_enu + tc.base_ecef)

    # 3. Measurement factors (DDPR, DDCP, SD Doppler, NHC)
    n_dd, visible_sat_freqs, seeded_sat_freqs = _tc_factors.build_dd_factors(
        tc, graph, values, obs, obsb, obs_sd,
        rs, rsb, sat, el, iu, ir_map,
        tc.Pose(key_idx), tc.lever_arm_tc,
        existing_estimate=estimate)
    info['n_dd'] = n_dd
    _tc_factors.add_sd_doppler_factors(
        tc, graph, key_idx, obs, obs_sd, rs, vs, sat, el, iu,
        pred_ecef, gyro_mean, info)
    if _tc_factors.add_nhc_factor(tc, graph, key_idx, gyro_mean=gyro_mean):
        info['nhc'] = True

    # 4. Fixed-lag smoother update
    pose_tc, ecef_tc, early = _solve_isam2(
        tc, graph, values, estimate, key_idx, R_enu2ecef, obs, info)
    if early is not None:
        return early

    # 5. Continuous LAMBDA ambiguity resolution & solution output
    tc.nav.smode = 5
    float_antenna_ecef = tc._antenna_ecef(pose_tc, ecef_tc)
    tc.nav.x[0:3] = float_antenna_ecef
    n_fixed, fixed_pos = _tc_ar.run_ar(
        tc, sat, el, tc._sat_states.amb_keys_dict(),
        key_pose=tc.Pose(key_idx),
        visible_sat_freqs=visible_sat_freqs,
        seeded_sat_freqs=seeded_sat_freqs)

    sol, tag, n_fixed = _decide_fix_or_flt(
        tc, float_antenna_ecef, n_fixed, fixed_pos, info)
    info['bias_acc'] = tc.tc_bias.accelerometer()
    info['bias_gyro'] = tc.tc_bias.gyroscope()
    return sol, tag, n_fixed, info


class ImuGnssTc:
    """Single-phase IMU/GNSS tightly-coupled factor-graph processor."""

    Pose = staticmethod(lambda i: gtsam.symbol('P', i))
    Vel = staticmethod(lambda i: gtsam.symbol('v', i))
    Bias = staticmethod(lambda i: gtsam.symbol('b', i))
    N = staticmethod(lambda s, f, gen=0: gtsam.symbol(
        'n', int(gen) * 1000000 + int(s) * 10 + int(f)))

    @staticmethod
    def _noise1(sigma):
        return gtsam.noiseModel.Isotropic.Sigma(1, float(sigma))

    def __init__(self, nav, pos0, base_ecef, imu_data,
                 lever_arm=np.zeros(3), logfile=None, cfg=None):
        self.engine = rtkpos(nav, pos0, logfile)
        self.nav = self.engine.nav

        self.base_ecef = np.array(base_ecef)
        self.imu_data = imu_data
        self.lever_arm = np.array(lever_arm, dtype=float)

        self.R_enu2ecef, self.ecef_T_nav = enu_frame(self.base_ecef)
        self.base_pt = gtsam.Point3(*self.base_ecef)

        self.cfg = cfg if cfg is not None else TcConfig.from_env()
        self._apply_nav_config()
        self._init_runtime_state()

    def _apply_nav_config(self):
        self.nav.armode = int(self.cfg.armode)
        self.nav.elmin = np.deg2rad(float(self.cfg.elmin_deg))
        self.nav.cnr_min = float(self.cfg.cnr_min_dbhz)
        self.nav.sat_band_plan = True

    def _init_runtime_state(self):
        self.epoch = 0
        self.imu_idx = 0
        self.smoother_node_idx = 0

        self._sat_states = SatStateMap()

        self.isam2 = None
        self.imu_params = None
        self.tc_bias = None
        self.lever_arm_tc = gtsam.Point3(0, 0, 0)
        self._nhc_lever = np.array(
            [float(x) for x in self.cfg.nhc_lever.split(',')], dtype=float)

        self._last_obs_t = None
        self._epoch_dt = 0.2

    def _update_epoch_dt(self, obs):
        if self._last_obs_t is not None:
            try:
                dt = float(timediff(obs.t, self._last_obs_t))
                if dt > 0:
                    self._epoch_dt = dt
            except (TypeError, ValueError):
                pass
        self._last_obs_t = obs.t

    def _antenna_ecef(self, pose, ecef_body):
        lever_arr = np.array(self.lever_arm_tc)
        if np.linalg.norm(lever_arr) == 0:
            return ecef_body
        R_body = self.ecef_T_nav.compose(pose).rotation().matrix()
        return ecef_body + R_body @ lever_arr

    def _prepare_dd_measurements(self, obs, obsb):
        if self.isam2 is None:
            self.nav.t = obs.t
        prep = None
        if obsb is not None:
            prep = self.engine.prepare_double_difference_measurements(
                obs, obsb, pos_pred=self.nav.x[0:3].copy(),
                dd_only=True, compute_zdres=False)
        if prep is not None and len(prep['iu']) >= 4:
            return (prep['rs'], prep['vs'], prep['rsb'], prep['sat'],
                    prep['el'], prep['iu'], prep['obs_sd'],
                    {s: i for i, s in enumerate(obsb.sat)})
        return (None, None, None, np.empty(0, dtype=int),
                np.empty(0, dtype=float), np.empty(0, dtype=int), None, {})

    def process(self, obs, obsb=None, rs=None, vs=None, rsb=None,
                sat=None, el=None, iu=None, obs_sd=None, ir_map=None):
        self._update_epoch_dt(obs)
        if rs is None:
            rs, vs, rsb, sat, el, iu, obs_sd, ir_map = (
                self._prepare_dd_measurements(obs, obsb))

        R, ns, init_ecef = self.R_enu2ecef, len(sat), self.nav.x[0:3].copy()
        info = {'tc_epoch': self.smoother_node_idx}
        res = run_tc_epoch(
            self, obs, obsb, rs, vs, rsb, sat, el, iu, obs_sd, ir_map,
            info, ns, init_ecef, R)
        self.epoch += 1
        self.nav.t = obs.t
        return res
