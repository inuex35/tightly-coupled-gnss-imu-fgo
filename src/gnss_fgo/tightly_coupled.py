"""Phase 2 — moving CombinedImuFactor + DDFactorArm pipeline."""

from .buildfactor.epoch_context import make_epoch_data
from .preprocess import gate, stage as preprocess
from .optimize import stage as optimize
from .validation import output, postprocess


def run_tc_epoch(tc, obs, obsb, rs, vs, dts, rsb, sat, el, iu,
                    obs_sd, ir_map, ref_vel, ref_ecef, info, ns, init_ecef,
                    R_enu2ecef):
    """Phase 2: IMU/GNSS TC pipeline."""
    epoch = make_epoch_data(
        obs, obsb, rs, vs, dts, rsb, sat, el, iu, obs_sd, ir_map,
        ref_vel, ref_ecef, info, ns, init_ecef, R_enu2ecef)
    for stage in (preprocess.run, gate.run,
                  optimize.run, postprocess.run,
                  output.run):
        result = stage(tc, epoch)
        if result is not None:
            return result
    raise RuntimeError("tightly-coupled pipeline did not terminate")
