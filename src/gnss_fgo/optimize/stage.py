"""Stage C — solve side: DD factor build, ISAM2 update, FDE, LAMBDA AR."""

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
from ..preprocess import prefit as _tc_prefit
from ..preprocess import sat_quality as _satq
from ..utils import heading_from_pose, sorted_amb_items
from ..validation import residuals as _tc_residuals
from .. import recovery as _tc_recovery
from . import isam as _tc_isam


# ── Phase-2 pipeline contract (see stage_contract.py) ──────────────
STAGE_READS = (
    'R', 'bias_p', 'dts', 'ecef_tc', 'el', 'estimate', 'graph', 'gyro_mean',
    'info', 'ir_map', 'iu', 'kk', 'nb', 'ns', 'nv', 'obs', 'obs_sd',
    'obsb', 'pose_tc', 'pred_ecef', 'pred', 'prev_amb_tc',
    'remove_indices', 'rs', 'rsb', 'sat', 'skip_cp_now', 'slip_keys',
    'values', 'vs',
)
STAGE_WRITES = (
    'ecef_tc', 'estimate', 'nb', 'nv', 'pose_tc', 'prev_amb_tc[*]', 'xa',
)



from .build import _build_factor_block
from .isam import _solve_isam2
from .postfit_diag import _compute_postfit_diagnostics
from .ar_stage import (
    _ar_eligibility, _ar_starvation_reset, _record_ar_diagnostics,
    _run_ar_with_marginals, _run_lambda_ar,
)

def run(tc, ed):
    """Stage C: solve (DD factors → ISAM2 → FDE → LAMBDA AR)."""
    prev_smode = int(getattr(tc.nav, 'smode', 0))
    _build_factor_block(tc, ed, prev_smode)        # C1
    early = _solve_isam2(tc, ed)                   # C2
    if early is not None:
        return early
    _compute_postfit_diagnostics(tc, ed)           # C4
    return _run_lambda_ar(tc, ed)                  # C3


