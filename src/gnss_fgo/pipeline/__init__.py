"""The per-epoch flow, in execution order:

    imu_prediction -> quality_gate
    -> solve (measurement_factors, update_smoother, fix_ambiguities)
    -> report
"""

from ..factors.epoch_context import make_epoch_data
from . import imu_prediction, quality_gate, solve, report


def run_tc_epoch(tc, obs, obsb, rs, vs, rsb, sat, el, iu,
                    obs_sd, ir_map, info, ns, init_ecef,
                    R_enu2ecef):
    """Phase 2: IMU/GNSS TC pipeline."""
    epoch = make_epoch_data(
        obs, obsb, rs, vs, rsb, sat, el, iu, obs_sd, ir_map,
        info, ns, init_ecef, R_enu2ecef)
    for stage in (imu_prediction.run, quality_gate.run,
                  solve.run, report.run):
        result = stage(tc, epoch)
        if result is not None:
            return result
    raise RuntimeError("tightly-coupled pipeline did not terminate")
