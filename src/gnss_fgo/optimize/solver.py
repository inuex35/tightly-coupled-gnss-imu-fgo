"""ISAM2 / FixedLagSmoother glue."""

import gtsam


def make_isam2(lag, relinearize_skip=1, relinearize_threshold=0.01):
    """Build IncrementalFixedLagSmoother with project-standard ISAM2 params."""
    params = gtsam.ISAM2Params()
    params.setRelinearizeThreshold(float(relinearize_threshold))
    params.relinearizeSkip = int(relinearize_skip)
    return gtsam.IncrementalFixedLagSmoother(lag, params)


def filter_removable_indices(tc, indices, keep_cp=True, keep_hold=True):
    """Filter stale factor indices (already marginalized out of isam2)."""
    if not indices:
        return []
    facs = tc.isam2.getFactors()
    n = facs.size()
    valid = []
    for i in indices:
        if i is None or i < 0 or i >= n:
            continue
        try:
            fac = facs.at(i)
            if fac is None:
                continue
            tname = type(fac).__name__
            if keep_cp and 'CarrierPhase' in tname:
                continue
            if keep_hold and 'Prior' in tname:
                continue
        except RuntimeError:
            continue
        valid.append(i)
    return valid


def fls_update(tc, graph, values, kk, keep_keys=(), remove_indices=None,
               advance_time=True, include_prev=True):
    """Apply an isam2 update with a timestamp map for Xpose/Vel/Bias(kk)."""
    if advance_time:
        tc.tc_time += tc._epoch_dt
    ts = gtsam.FixedLagSmootherKeyTimestampMap()
    t = tc.tc_time
    ts[tc.Xpose(kk)] = t
    ts[tc.Vel(kk)] = t
    ts[tc.Bias(kk)] = t
    if include_prev and kk > 0:
        ts[tc.Xpose(kk - 1)] = t
        ts[tc.Vel(kk - 1)] = t
        ts[tc.Bias(kk - 1)] = t
    for k in values.keys():
        if k not in ts:
            ts[k] = t
    for k in keep_keys:
        ts[k] = t
    remove_safe = filter_removable_indices(tc, remove_indices)
    timing_on = bool(getattr(tc.cfg, 'fls_update_timing', False))
    if timing_on:
        import time as _time
        _t0 = _time.perf_counter()
    if remove_safe:
        tc.isam2.update(graph, values, ts, remove_safe)
    else:
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
