"""Stage C2 -- the smoother update (ISAM2/FLS glue + one epoch).

Gathers the keys that must survive this update (live ambiguities, their
predecessors, the Doppler clock pair), applies the update with the epoch's
removals, and snapshots the new estimate. A solve failure is handed to
validation/recovery, which owns the warm-reset decision.
"""


import gtsam

from ..utils import sorted_amb_items

from ..integrity import recovery as _tc_recovery


def _solve_isam2(tc, epoch):
    """Stage C2 — gather kept keys, run FLS update, snapshot the new estimate. Returns the recovery early-return tuple on solve failure, else None."""
    info = epoch.info
    try:
        extra = [k for (_sf, k) in sorted_amb_items(tc._sat_states.amb_keys_dict())]
        for sf, (k_old, _) in sorted_amb_items(epoch.prev_amb_values):
            if tc._sat_states.at(*sf).amb_key is not None:
                extra.append(k_old)
        extra.extend(tc._doppler_keep_keys)
        fls_update(tc, epoch.graph, epoch.values, epoch.key_idx,
                   keep_keys=extra)
        epoch.estimate = tc.isam2.calculateEstimate()
    except (RuntimeError, IndexError, ValueError) as ex:
        # ValueError: ISAM2 marginalization raises it ("Asking to remove
        # variables from the variable index that are not unused") when a
        # prior purge left the FLS bookkeeping inconsistent — exactly
        # the smoother-broke case the warm reset exists for.
        return _tc_recovery.handle_solve_exception(tc,
            ex, epoch.pred_nav, epoch.bias_prev, epoch.key_idx,
            epoch.obs, epoch.obsb, epoch.obs_sd, epoch.rs, epoch.rsb,
            epoch.sat, epoch.el, epoch.iu, epoch.ir_map, info)

    # ────────────────────────────────────────────────────────────────




# ── ISAM2 / FixedLagSmoother glue (was optimize/solver.py) ──────────


def make_isam2(lag, relinearize_skip=1, relinearize_threshold=0.01):
    """Build IncrementalFixedLagSmoother with project-standard ISAM2 params."""
    params = gtsam.ISAM2Params()
    params.setRelinearizeThreshold(float(relinearize_threshold))
    params.relinearizeSkip = int(relinearize_skip)
    return gtsam.IncrementalFixedLagSmoother(lag, params)


def fls_update(tc, graph, values, key_idx, keep_keys=(),
               advance_time=True, include_prev=True):
    """Apply an isam2 update with a timestamp map for Xpose/Vel/Bias(key_idx)."""
    if advance_time:
        tc.tc_time += tc._epoch_dt
    ts = gtsam.FixedLagSmootherKeyTimestampMap()
    t = tc.tc_time
    ts[tc.Xpose(key_idx)] = t
    ts[tc.Vel(key_idx)] = t
    ts[tc.Bias(key_idx)] = t
    if include_prev and key_idx > 0:
        ts[tc.Xpose(key_idx - 1)] = t
        ts[tc.Vel(key_idx - 1)] = t
        ts[tc.Bias(key_idx - 1)] = t
    for k in values.keys():
        if k not in ts:
            ts[k] = t
    for k in keep_keys:
        ts[k] = t
    timing_on = bool(getattr(tc.cfg, 'fls_update_timing', False))
    if timing_on:
        import time as _time
        _t0 = _time.perf_counter()
    tc.isam2.update(graph, values, ts)
    if timing_on:
        _dt = _time.perf_counter() - _t0
        tc._fls_update_time_total = (
            tc._fls_update_time_total + _dt)
        tc._fls_update_calls = tc._fls_update_calls + 1
        tc._fls_update_last_ms = _dt * 1000.0
    extra = int(tc.cfg.cp_hold_isam_iters)
    if extra > 0 and tc._recov_cp_hold > 0:
        empty_g = gtsam.NonlinearFactorGraph()
        empty_v = gtsam.Values()
        empty_ts = gtsam.FixedLagSmootherKeyTimestampMap()
        for _ in range(extra):
            tc.isam2.update(empty_g, empty_v, empty_ts)
    tc.total_factor_count += graph.size()
