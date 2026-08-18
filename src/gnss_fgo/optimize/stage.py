"""Stage C — solve side: DD factor build, ISAM2 update, FDE, LAMBDA AR."""




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
from .ar_stage import _run_lambda_ar

def run(tc, ed):
    """Stage C: solve (DD factors → ISAM2 → FDE → LAMBDA AR)."""
    prev_smode = int(getattr(tc.nav, 'smode', 0))
    _build_factor_block(tc, ed, prev_smode)        # C1
    early = _solve_isam2(tc, ed)                   # C2
    if early is not None:
        return early
    _compute_postfit_diagnostics(tc, ed)           # C4
    return _run_lambda_ar(tc, ed)                  # C3


