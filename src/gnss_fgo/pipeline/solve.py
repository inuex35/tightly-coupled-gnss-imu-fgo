"""Stage C — solve side: DD factor build, ISAM2 update, post-fit snapshot, LAMBDA AR."""

STAGE_READS = (
    'R_enu2ecef', 'bias_prev', 'ecef_tc', 'el', 'estimate', 'graph', 'gyro_mean',
    'info', 'ir_map', 'iu', 'key_idx', 'nb', 'nv', 'obs', 'obs_sd',
    'obsb', 'pose_tc', 'pred_ecef', 'pred_nav',
    'rs', 'rsb', 'sat', 'values', 'vs',
)
STAGE_WRITES = (
    'ecef_tc', 'estimate', 'nb', 'nv', 'pose_tc', 'xa',
)

from .measurement_factors import _build_factor_block
from .update_smoother import _solve_isam2
from .check_postfit import _compute_postfit_diagnostics
from .fix_ambiguities import _run_lambda_ar


def run(tc, epoch):
    """Stage C: solve (DD factors -> ISAM2 -> snapshot -> LAMBDA AR)."""
    _build_factor_block(tc, epoch)          # C1
    early = _solve_isam2(tc, epoch)         # C2
    if early is not None:
        return early
    _compute_postfit_diagnostics(tc, epoch) # C3
    return _run_lambda_ar(tc, epoch)        # C4
