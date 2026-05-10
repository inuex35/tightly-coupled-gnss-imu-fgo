"""Per-epoch Doppler velocity prior."""

import gtsam
from ..preprocess import prefit as _tc_prefit


def add_doppler_vel_prior(tc, ed):
    """Add ``PriorFactorVector(Vel(kk), v_doppler_enu, σ)`` from rover Doppler-LS when ``cfg.doppler_vel_sigma > 0``."""
    sigma = float(tc.cfg.doppler_vel_sigma)
    if sigma <= 0:
        return
    info = ed.info
    v_dop_ecef, clkdr_mps, dop_ok, dop_res, dop_n = _tc_prefit.doppler_velocity_ls(
        tc, ed.obs, ed.obs_sd, ed.rs, ed.vs, ed.iu, ed.sat,
        ed.pred_ecef, ed.pred.velocity())
    info['doppler_n'] = dop_n
    dop_accept = bool(dop_ok)
    if dop_ok:
        info['doppler_res'] = dop_res
        info['doppler_clkdr_mps'] = float(clkdr_mps)
    if not dop_ok or dop_res >= float(tc.cfg.doppler_max_res):
        dop_accept = False
    info['doppler_accept'] = bool(dop_accept)
    if dop_ok and not dop_accept:
        info['doppler_res_rejected'] = dop_res
    if not dop_accept:
        return
    v_dop_enu = ed.R.T @ v_dop_ecef
    info['doppler_vel_enu'] = v_dop_enu
    ed.g3.add(gtsam.PriorFactorVector(
        tc.Vel(ed.kk), v_dop_enu,
        gtsam.noiseModel.Isotropic.Sigma(3, sigma)))
