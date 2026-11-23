"""IMU-only outage propagation path."""

import numpy as np
import gtsam

from cssrlib.gnss import time2gpst

from ..factors.nhc import add_nhc_factor as _add_nhc_factor
from ..factors.zupt import add_zupt_factors as _add_zupt_factor_inplace
from ..factors import imu_preintegration as _tc_pim
from ..pipeline import update_smoother as _tc_isam


def _make_imu_only_diagnostics(tc):
    return {
        'phase': 2,
        'tc_epoch': tc.tc_epoch,
        'imu_only': True,
        'gnss_skip': True,
    }


def advance_epoch_and_pack(tc, sol, tag, nb, info, obs):
    """Advance tc.epoch / nav.t and pack the (sol, tag, nb, info) tuple."""
    tc.epoch += 1
    tc.nav.t = obs.t
    return sol, tag, nb, info


def _outage_drain_imu(tc, tow_obs):
    while tc.imu_idx < len(tc.imu_data):
        if tc.imu_data[tc.imu_idx]['tow'] > tow_obs + 1e-6:
            break
        tc.imu_idx += 1


def _outage_tick_sat_outc(tc, info):
    n_skip_reset = 0
    maxout = tc._sat_states.maxout
    for st in tc._sat_states.values():
        st.outc += 1
        if st.outc > maxout and st.amb_key is not None:
            st.amb_key = None
            st.amb_gen += 1
            n_skip_reset += 1
    if n_skip_reset:
        info['skip_amb_reset'] = n_skip_reset


def process_imu_only(tc, obs):
    """Advance the graph by one epoch using IMU + NHC + ZUPT only."""
    tc._update_epoch_dt(obs)
    tc._reset_current_epoch()
    R = tc.R_enu2ecef
    info = _make_imu_only_diagnostics(tc)
    _, tow_obs = time2gpst(obs.t)

    if tc.isam2 is None:
        _outage_drain_imu(tc, tow_obs)
        return advance_epoch_and_pack(tc, np.array(tc.nav.x[0:3]), 'FLT', 0, info, obs)

    tc.tc_epoch += 1
    key_idx = tc.tc_epoch
    info['tc_epoch'] = key_idx
    _outage_tick_sat_outc(tc, info)

    imu_idx_prev = tc.imu_idx
    pim, n_imu, gyro_mean = _tc_pim.build_pim(tc, tc.tc_bias, target_tow=tow_obs)
    if n_imu == 0:
        return advance_epoch_and_pack(tc, np.array(tc.nav.x[0:3]), 'FLT', 0, info, obs)
    info['n_imu'] = n_imu

    graph = gtsam.NonlinearFactorGraph()
    values = gtsam.Values()
    estimate = tc.isam2.calculateEstimate()
    if not estimate.exists(tc.Xpose(key_idx - 1)):
        return advance_epoch_and_pack(tc, np.array(tc.nav.x[0:3]), 'FLT', 0, info, obs)

    pose_p = estimate.atPose3(tc.Xpose(key_idx - 1))
    vel_prev = estimate.atVector(tc.Vel(key_idx - 1))
    bias_prev = estimate.atConstantBias(tc.Bias(key_idx - 1))
    pred = pim.predict(gtsam.NavState(pose_p, vel_prev), bias_prev)
    values.insert(tc.Xpose(key_idx), pred.pose())
    values.insert(tc.Vel(key_idx), pred.velocity())
    values.insert(tc.Bias(key_idx), bias_prev)
    _tc_pim.add_imu_chain(tc, graph, values, key_idx, pim, pose_p, vel_prev, info)

    if _add_nhc_factor(tc, graph, key_idx, gyro_mean_rh=gyro_mean):
        info['nhc'] = True
    _add_zupt_factor_inplace(tc, graph, key_idx, imu_idx_prev, n_imu, info)

    try:
        _tc_isam.fls_update(tc, graph, values, key_idx,
                            keep_keys=tc._sat_states.amb_key_values())
        estimate = tc.isam2.calculateEstimate()
        pose_tc = estimate.atPose3(tc.Xpose(key_idx))
        tc.tc_bias = estimate.atConstantBias(tc.Bias(key_idx))
        ecef_tc = R @ np.array(pose_tc.translation()) + tc.base_ecef
        sol = tc._antenna_ecef(pose_tc, ecef_tc)
        tc.nav.x[0:3] = sol
    except (RuntimeError, IndexError, ValueError) as ex:
        info['error'] = str(ex)
        sol = np.array(tc.nav.x[0:3])

    tc.nav.smode = 5
    return advance_epoch_and_pack(tc, sol, 'FLT', 0, info, obs)
