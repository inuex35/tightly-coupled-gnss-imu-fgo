"""Layer 4 -- one FixedLagSmoother update.

Gathers the keys that must survive this update (live ambiguities, their
predecessors, the Doppler clock pair), applies the update with the epoch's
removals, and snapshots the new estimate. A solve failure is handed to
validation/recovery, which owns the warm-reset decision.
"""

import os

import numpy as np
import gtsam

from .. import ar as _tc_ar
from ..buildfactor import clock as _tc_clock
from ..buildfactor import doppler as _tc_doppler
from ..buildfactor import doppler_sd as _tc_doppler_sd
from ..buildfactor import tdcp as _tc_tdcp
from ..buildfactor import factors as _tc_factors
from ..buildfactor import nhc as _tc_nhc
from ..buildfactor import zupt as _tc_zupt
from ..preprocess import sat_quality as _satq
from ..utils import heading_from_pose, sorted_amb_items
from ..validation import postfit as _tc_postfit
from ..validation import recovery as _tc_recovery
from . import solver as _tc_solver


def _solve_isam2(tc, ed):
    """Layer 4 — gather kept keys, run FLS update, snapshot the new estimate. Returns the recovery early-return tuple on solve failure, else None."""
    info = ed.info
    try:
        extra = [k for (_sf, k) in sorted_amb_items(tc._sat_states.amb_keys_dict())]
        for sf, (k_old, _) in sorted_amb_items(ed.prev_amb_tc):
            if tc._sat_states.at(*sf).amb_key is not None:
                extra.append(k_old)
        extra.extend(tc._doppler_keep_keys)
        _tc_solver.fls_update(tc, ed.g3, ed.v3, ed.kk, keep_keys=extra,
                         remove_indices=ed.remove_indices)
        ed.est2 = tc.isam2.calculateEstimate()
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


