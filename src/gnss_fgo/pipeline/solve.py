"""Stage C — solve side: DD factor build, ISAM2 update, LAMBDA AR."""

from .measurement_factors import _build_factor_block
from .update_smoother import _solve_isam2
from .fix_ambiguities import _run_lambda_ar


def run(tc, epoch):
    """Stage C: solve (DD factors -> ISAM2 -> LAMBDA AR)."""
    _build_factor_block(tc, epoch)          # C1
    early = _solve_isam2(tc, epoch)         # C2
    if early is not None:
        return early
    return _run_lambda_ar(tc, epoch)        # C3
