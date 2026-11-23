"""Stage C1 — assemble measurement factors (DDPR, DDCP, SD Doppler, NHC, ZUPT)."""

from ..factors import doppler_sd as _tc_doppler_sd
from ..factors import factors as _tc_factors
from ..factors import nhc as _tc_nhc
from ..factors import zupt as _tc_zupt


def _build_factor_block(tc, epoch):
    """Stage C1 — add pure C++ GNSS + vehicle constraint factors to epoch.graph."""
    info = epoch.info
    epoch.nv = _tc_factors.build_dd_factors(
        tc, epoch.graph, epoch.values, epoch.obs, epoch.obsb, epoch.obs_sd,
        epoch.rs, epoch.rsb, epoch.sat, epoch.el, epoch.iu, epoch.ir_map,
        tc.Xpose(epoch.key_idx), tc.lever_arm_tc, tc._sat_states,
        existing_estimate=epoch.estimate)
    info['n_dd'] = epoch.nv

    _tc_doppler_sd.add_sd_doppler_factors(tc, epoch)

    if _tc_nhc.add_nhc_factor(tc, epoch.graph, epoch.key_idx,
                              gyro_mean_rh=epoch.gyro_mean):
        info['nhc'] = True

    _tc_zupt.add_zupt_factors_for_stage(tc, epoch)
