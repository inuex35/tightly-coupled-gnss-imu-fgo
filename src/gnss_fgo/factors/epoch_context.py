"""Epoch-level factories shared by runner, tightly_coupled, and recovery."""

from ..state.epoch_data import EpochData


def prepare_process_epoch(tc, obs, sat, obs_sd):
    """Refresh per-epoch runner state and return common process inputs."""
    tc._update_epoch_dt(obs)
    tc._reset_current_epoch()
    return tc.R_enu2ecef, len(sat), tc.nav.x[0:3].copy()


def make_epoch_diagnostics(tc, **extra):
    """Create the default diagnostics payload for a new process epoch."""
    info = {
        'phase': tc.phase,
        'tc_epoch': 0,
        'collecting': tc.collecting,
        'n_collected': len(tc.collected_fixes),
    }
    info.update(extra)
    return info


def make_epoch_data(
        obs, obsb, rs, vs, dts, rsb, sat, el, iu, obs_sd, ir_map,
        ref_ecef, info, ns, init_ecef, R_enu2ecef):
    """Construct the per-epoch mutable context consumed by Phase 2 stages."""
    return EpochData(
        obs=obs, obsb=obsb, rs=rs, vs=vs, dts=dts, rsb=rsb,
        sat=sat, el=el, iu=iu, obs_sd=obs_sd, ir_map=ir_map,
        ref_ecef=ref_ecef,
        info=info, ns=ns, init_ecef=init_ecef,
        R_enu2ecef=R_enu2ecef)
