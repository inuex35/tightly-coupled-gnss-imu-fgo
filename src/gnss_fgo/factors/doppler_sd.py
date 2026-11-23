"""Between-satellite single-differenced Doppler (clock-free, C++ factor)."""

import numpy as np
import gtsam

from cssrlib.gnss import sat2prn, uGNSS

from ..utils import get_wavelengths, huber_noise


def _doppler_rows(tc, epoch):
    """Collect (sat, sat_pos, sat_vel, doppler_hz, wavelength_m, el_rad)."""
    rows = []
    for si, i_obs in enumerate(epoch.iu):
        s = int(epoch.sat[si])
        if sat2prn(s)[0] == uGNSS.GLO:
            continue
        if i_obs >= epoch.obs.D.shape[0]:
            continue
        lams = None
        for f in range(min(tc.nav.nf, epoch.obs.D.shape[1])):
            d_obs = epoch.obs.D[i_obs, f]
            if d_obs == 0.0:
                continue
            if lams is None:
                lams = get_wavelengths(epoch.obs_sd, s, glo_ch=tc.nav.glo_ch)
            if f < len(lams) and lams[f] > 0:
                rows.append((s, epoch.rs[i_obs, :3], epoch.vs[i_obs, :3],
                             float(d_obs), float(lams[f]),
                             float(epoch.el[si])))
                break
    return rows


def add_sd_doppler_factors(tc, epoch):
    """Add clock-free SD Doppler factors against the highest-elevation reference sat."""
    sigma = float(tc.cfg.doppler_sd_sigma)
    if sigma <= 0 or epoch.key_idx is None:
        return

    rows = _doppler_rows(tc, epoch)
    if len(rows) < 2:
        return

    sigmas = [sigma / max(np.sin(r[5]), 0.1) for r in rows]
    i_ref = int(np.argmin(sigmas))
    ref, sigma_ref = rows[i_ref], sigmas[i_ref]

    omega = np.zeros(3)
    if epoch.gyro_mean is not None:
        bias_gyro = (tc.tc_bias.gyroscope() if tc.tc_bias is not None
                     else np.zeros(3))
        omega = np.asarray(epoch.gyro_mean, dtype=float) - bias_gyro

    key_idx = int(epoch.key_idx)
    lever = np.asarray(tc.lever_arm_tc, dtype=float)
    rr = np.asarray(epoch.pred_ecef, dtype=float)
    n = 0
    huber = float(tc.cfg.huber_k)
    for row, sig in zip(rows, sigmas):
        if row is ref:
            continue
        noise = huber_noise(np.hypot(sig, sigma_ref), huber)
        epoch.graph.add(gtsam.SingleDifferenceDopplerFactorArm(
            tc.Xpose(key_idx), tc.Vel(key_idx),
            row[3], ref[3],
            row[4], ref[4],
            np.asarray(row[1], dtype=float), np.asarray(row[2], dtype=float),
            np.asarray(ref[1], dtype=float), np.asarray(ref[2], dtype=float),
            rr, lever, tc.ecef_T_nav, omega,
            0.0, 0.0,
            noise))
        n += 1
    if n:
        epoch.info['doppler_sd_n'] = n
        epoch.info['doppler_sd_ref'] = int(ref[0])
