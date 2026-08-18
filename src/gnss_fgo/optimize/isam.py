"""Stage C2 -- the smoother update (ISAM2/FLS glue + one epoch).

Gathers the keys that must survive this update (live ambiguities, their
predecessors, the Doppler clock pair), applies the update with the epoch's
removals, and snapshots the new estimate. A solve failure is handed to
validation/recovery, which owns the warm-reset decision.
"""


import gtsam

from ..utils import sorted_amb_items

from .. import recovery as _tc_recovery


def _solve_isam2(tc, ed):
    """Stage C2 — gather kept keys, run FLS update, snapshot the new estimate. Returns the recovery early-return tuple on solve failure, else None."""
    info = ed.info
    try:
        extra = [k for (_sf, k) in sorted_amb_items(tc._sat_states.amb_keys_dict())]
        for sf, (k_old, _) in sorted_amb_items(ed.prev_amb_tc):
            if tc._sat_states.at(*sf).amb_key is not None:
                extra.append(k_old)
        extra.extend(tc._doppler_keep_keys)
        fls_update(tc, ed.graph, ed.values, ed.kk, keep_keys=extra,
                         remove_indices=ed.remove_indices)
        ed.estimate = tc.isam2.calculateEstimate()
    except (RuntimeError, IndexError, ValueError) as ex:
        # ValueError: ISAM2 marginalization raises it ("Asking to remove
        # variables from the variable index that are not unused") when a
        # prior purge left the FLS bookkeeping inconsistent — exactly
        # the smoother-broke case the warm reset exists for.
        return _tc_recovery.handle_solve_exception(tc,
            ex, ed.pred, ed.bias_p, ed.kk,
            ed.obs, ed.obsb, ed.obs_sd, ed.rs, ed.rsb,
            ed.sat, ed.el, ed.iu, ed.ir_map, info)

    # ────────────────────────────────────────────────────────────────




# ── ISAM2 / FixedLagSmoother glue (was optimize/solver.py) ──────────


def make_isam2(lag, relinearize_skip=1, relinearize_threshold=0.01):
    """Build IncrementalFixedLagSmoother with project-standard ISAM2 params."""
    params = gtsam.ISAM2Params()
    params.setRelinearizeThreshold(float(relinearize_threshold))
    params.relinearizeSkip = int(relinearize_skip)
    return gtsam.IncrementalFixedLagSmoother(lag, params)


def filter_removable_indices(tc, indices, keep_cp=True, keep_hold=True):
    """Filter stale factor indices (already marginalized out of isam2).

    Identity guard: every index in these lists was recorded as "a factor
    of some ambiguity", so the slot must currently hold a factor that
    references an 'n' key. The FLS reuses freed slots
    (findUnusedFactorSlots), so a remembered index can point at an
    unrelated factor — removing that would corrupt the graph.
    """
    if not indices:
        return []
    facs = tc.isam2.getFactors()
    n = facs.size()
    valid = []
    n_chr = ord('n')
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
            # Never break the BetweenN chain or pull out custom factors
            # mid-window: with keep_cp/keep_hold those are the only
            # things an amb_factor_indices entry can still name, and
            # removing them leaves chained N keys factorless
            # ("variables that are not unused" on the next
            # marginalization). Arcs die via gen-bump + natural
            # marginalization instead — which is also what the historic
            # stale-slot indices effectively did (no-ops).
            if 'Between' in tname or tname == 'CustomFactor':
                continue
            if not any(gtsam.Symbol(k).chr() == n_chr
                       for k in fac.keys()):
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
