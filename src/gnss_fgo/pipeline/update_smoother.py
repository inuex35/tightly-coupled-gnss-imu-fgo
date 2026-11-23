"""Stage C2 — IncrementalFixedLagSmoother update keeping live arc ambiguities alive."""

import gtsam

from ..utils import sorted_amb_items
from ..integrity import recovery as _tc_recovery


def _solve_isam2(tc, epoch):
    """Stage C2 — update FixedLagSmoother and snapshot the new estimate."""
    info = epoch.info
    try:
        # Keep only currently-visible or active arc ambiguity keys stamped at t.
        live_amb_keys = [
            k for (_sf, k) in sorted_amb_items(tc._sat_states.amb_keys_dict())
            if epoch.estimate.exists(k) or epoch.values.exists(k)
        ]
        fls_update(tc, epoch.graph, epoch.values, epoch.key_idx,
                   keep_keys=live_amb_keys)
        epoch.estimate = tc.isam2.calculateEstimate()
    except (RuntimeError, IndexError, ValueError) as ex:
        return _tc_recovery.handle_solve_exception(
            tc, ex, epoch.pred_nav, epoch.bias_prev, epoch.key_idx,
            epoch.obs, epoch.obsb, epoch.obs_sd, epoch.rs, epoch.rsb,
            epoch.sat, epoch.el, epoch.iu, epoch.ir_map, info)
    return None


def make_isam2(lag, relinearize_skip=1, relinearize_threshold=0.01):
    """Build IncrementalFixedLagSmoother with standard ISAM2 parameters."""
    params = gtsam.ISAM2Params()
    params.setRelinearizeThreshold(float(relinearize_threshold))
    params.relinearizeSkip = int(relinearize_skip)
    return gtsam.IncrementalFixedLagSmoother(lag, params)


def fls_update(tc, graph, values, key_idx, keep_keys=(),
               advance_time=True, include_prev=True):
    """Apply an FLS update with current timestamps on pose/vel/bias and live N keys."""
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
    tc.isam2.update(graph, values, ts)
