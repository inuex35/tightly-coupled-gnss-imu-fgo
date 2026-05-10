"""Phase 1 — stationary GNSS-only Pose3 RTK."""

import numpy as np
import gtsam

from cssrlib.gnss import time2gpst
from .optimize import ar as _tc_ar
from . import tightly_coupled as _tightly_coupled
from .validation import recovery as _tc_recovery
from .buildfactor import factors as _tc_factors


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
    return _tc_recovery.finalize_epoch(tc, sol, tag, nb, info, obs)


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
        init_ecef, tc.Xp(ep), lever, tc.amb_keys,
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
    tc.isam.update(g, v, ts)
    # Record everything we just inserted so the next epoch can skip.
    tc._isam_p1_inserted.add(tc.Xp(ep))
    for k in v.keys():
        tc._isam_p1_inserted.add(k)
    for _ in range(3):
        tc.isam.update(gtsam.NonlinearFactorGraph(), gtsam.Values(),
                        gtsam.FixedLagSmootherKeyTimestampMap())
    est = tc.isam.calculateEstimate()

    return est


def _p1_emit_and_run_ar(tc, est, obs, rs, vs, dts, sat, el, iu, R):
    """Phase 1B — write nav.x from the Phase-1 estimate, populate marginals, run LAMBDA AR (only after the warm-up window), and emit the (sol, tag, nb, xa) tuple."""
    ep = tc.epoch
    enu_e = np.array(est.atPose3(tc.Xp(ep)).translation())
    tc.nav.x[0:3] = R @ enu_e + tc.base_ecef

    _tc_ar.write_marginals(tc, 
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
    """Phase 1C — collect IMU samples + estimate velocity from a 1-second sliding Fix-position window, accumulate fixes once we are moving, and trigger Phase-2 init when ``n_collect`` fixes are in hand."""
    # Collect IMU samples up to current GNSS epoch
    _, _tow_obs = time2gpst(obs.t)
    imu_samples = tc._collect_imu_samples(target_tow=_tow_obs)

    if not hasattr(tc, '_sol_hist'):
        tc._sol_hist = []  # [(tow, sol_ecef), ...]
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
            p, r, h, ba0, bg0 = _tightly_coupled.transition_to_tc(tc, tc.collected_fixes)
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



