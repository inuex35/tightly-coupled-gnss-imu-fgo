"""Stage C2 — IncrementalFixedLagSmoother update keeping live arc ambiguities alive."""

import numpy as np
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
        info['error'] = str(ex)
        return _tc_recovery.advance_epoch_and_pack(
            tc, np.array(tc.nav.x[0:3]), 'FLT', 0, info, epoch.obs)

    epoch.pose_tc = epoch.estimate.atPose3(tc.Xpose(epoch.key_idx))
    tc.tc_bias = epoch.estimate.atConstantBias(tc.Bias(epoch.key_idx))
    enu_tc = np.array(epoch.pose_tc.translation())
    epoch.ecef_tc = epoch.R_enu2ecef @ enu_tc + tc.base_ecef
    return None


_ISAM2_RELINEARIZE_SKIP = 1
_ISAM2_RELINEARIZE_THRESHOLD = 0.01


def make_isam2(lag, relinearize_skip=_ISAM2_RELINEARIZE_SKIP,
               relinearize_threshold=_ISAM2_RELINEARIZE_THRESHOLD):
    """Build IncrementalFixedLagSmoother with QR factorization and per-epoch relinearization."""
    params = gtsam.ISAM2Params()
    params.setFactorization("QR")
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
