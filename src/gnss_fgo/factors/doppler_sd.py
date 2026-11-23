"""Between-satellite single-differenced Doppler (clock-free, C++ factor)."""

import numpy as np
import gtsam

from cssrlib.gnss import sat2prn, uGNSS

from ..utils import get_wavelengths


_DOPPLER_MAD_K = 4.0     # standard 4σ MAD outlier screen
_SNR_REF_DBHZ = 42.0     # open-sky C/N0 reference [dB-Hz]


def _doppler_rows(tc, epoch):
    """Collect (sat, sat_pos, sat_vel, doppler_hz, wavelength_m, el_rad, snr)."""
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
                snr = (float(epoch.obs.S[i_obs, f])
                       if f < epoch.obs.S.shape[1] else 0.0)
                rows.append((s, epoch.rs[i_obs, :3], epoch.vs[i_obs, :3],
                             float(d_obs), float(lams[f]),
                             float(epoch.el[si]), snr))
                break
    return rows


def screen_rows(tc, epoch, rows):
    """Robust 4σ MAD screen on epoch Doppler residuals; returns (kept_rows, scale)."""
    if len(rows) < 5:
        return rows, 0.0
    v_pred = np.asarray(epoch.R_enu2ecef, dtype=float) @ np.asarray(
        epoch.pred_nav.velocity(), dtype=float)
    p_r = np.asarray(epoch.pred_ecef, dtype=float)
    A, b, keep = [], [], []
    for row in rows:
        p_sat, v_sat, d_obs, lam = row[1], row[2], row[3], row[4]
        d_vec = np.asarray(p_sat, dtype=float) - p_r
        rho = float(np.linalg.norm(d_vec))
        if rho < 1.0:
            continue
        e = d_vec / rho
        pred = float(e @ (np.asarray(v_sat, dtype=float) - v_pred))
        A.append([-e[0], -e[1], -e[2], 1.0])
        b.append(-lam * d_obs - pred)
        keep.append(row)
    if len(b) < 5:
        return rows, 0.0
    A, b = np.asarray(A), np.asarray(b)
    try:
        x, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return rows, 0.0
    r = A @ x - b
    scale = 1.4826 * float(np.median(np.abs(r - np.median(r))))
    if scale > 0:
        kept = [row for row, ri in zip(keep, r) if abs(ri) <= _DOPPLER_MAD_K * scale]
        if len(kept) >= 4:
            n_drop = len(keep) - len(kept)
            if n_drop:
                epoch.info['doppler_fde'] = n_drop
            keep = kept
    epoch.info['doppler_scale'] = scale
    return keep, scale


def add_sd_doppler_factors(tc, epoch):
    """Add clock-free SD Doppler factors against the best-weighted reference sat."""
    sigma = float(tc.cfg.doppler_sd_sigma)
    if sigma <= 0 or epoch.key_idx is None:
        return

    rows, scale = screen_rows(tc, epoch, _doppler_rows(tc, epoch))
    if len(rows) < 2:
        return

    def _sigma_of(el, snr):
        s = max(sigma, scale)
        s /= max(np.sin(el), 0.1)
        if snr > 0:
            s *= float(np.clip(10.0 ** ((_SNR_REF_DBHZ - snr) / 20.0), 1.0, 10.0))
        return s

    sigmas = [_sigma_of(r[5], r[6]) for r in rows]
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
        noise = gtsam.noiseModel.Isotropic.Sigma(
            1, float(np.hypot(sig, sigma_ref)))
        if huber > 0:
            noise = gtsam.noiseModel.Robust.Create(
                gtsam.noiseModel.mEstimator.Huber.Create(huber), noise)
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
