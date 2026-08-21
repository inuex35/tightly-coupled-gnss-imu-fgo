"""Between-satellite single-differenced Doppler.

Every raw Doppler carries the receiver clock drift, which is why
``gtsam.DopplerFactor`` keys on two clock-bias states. In a double-difference
graph nothing else observes those states, and then they cost more than the
measurement brings: reachable only through the Dopplers themselves, the drift
and the velocity along the mean line of sight go nearly degenerate under a
narrow sky and the estimate splits the error between them. Measured over the
full tokyo run2, the undifferenced form gives 0.48 m Fix-RMS against 0.09 m
for the difference, and a 99th percentile of 10.5 m against 2.2 m.

Differencing two satellites of one epoch removes the clock before it reaches
the graph -- the same reason this pipeline double-differences code and phase.
``gtsam.SingleDifferenceDopplerFactorArm`` holds the model; this module picks
the reference satellite, screens the epoch and sets the weights.
"""

import numpy as np
import gtsam

from cssrlib.gnss import sat2prn, uGNSS

from ..utils import get_wavelengths



def _doppler_rows(tc, epoch):
    """(sat, sat ECEF pos/vel, Doppler [Hz], wavelength [m], elevation [rad]).

    Mirrors the constellation and band choice the LS rows used: GLONASS is
    skipped (its FDMA channel makes the wavelength satellite-specific and the
    DD-only core never resolves it), and each satellite contributes the first
    band that actually carries a Doppler observation.
    """
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
    """Robust epoch screen: drop outliers, return a residual scale [m/s].

    A quick least squares for (velocity correction, clock drift) over the
    epoch's own Doppler rows, then the robust scale of what it cannot fit.
    Two things come out of it. Satellites whose residual is far outside that
    scale are dropped, and the scale itself is handed back so the factor
    sigmas can follow the measurements: entering the tokyo canyon the
    truth-referenced Doppler error goes 0.02 -> 0.14 m/s with the elevations
    and C/N0 barely moving, and a sigma that ignores that lets a degraded
    epoch pull the velocity as hard as a clean one.
    """
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
    k = float(tc.cfg.doppler_fde_k)
    if k > 0 and scale > 0:
        kept = [row for row, ri in zip(keep, r) if abs(ri) <= k * scale]
        if len(kept) >= 4:
            n_drop = len(keep) - len(kept)
            if n_drop:
                epoch.info['doppler_fde'] = n_drop
            keep = kept
    epoch.info['doppler_scale'] = scale
    return keep, scale


def add_sd_doppler_factors(tc, epoch, in_outage=False):
    """Add one SD Doppler factor per satellite, against the epoch reference.

    ``in_outage=True`` is the GDOP-skip path: no DD set exists by
    definition, so the require_dd/gdop gates don't apply — Doppler is
    the only velocity observation the epoch has, exactly where it's
    worth the most (the canyon drift is what it bounds).
    """
    sigma = float(tc.cfg.doppler_sd_sigma)
    if sigma <= 0 or epoch.key_idx is None:
        return
    if not in_outage:
        gdop_max = float(tc.cfg.doppler_gdop_max)
        if gdop_max > 0 and float(epoch.info.get('gdop', 0.0) or 0.0) > gdop_max:
            epoch.info['doppler_sd_skipped'] = 'gdop'
            return
        if tc.cfg.doppler_require_dd and epoch.nv < tc.cfg.min_dd_for_solve:
            epoch.info['doppler_sd_skipped'] = int(epoch.nv)
            return

    rows, scale = screen_rows(tc, epoch, _doppler_rows(tc, epoch))
    if len(rows) < 2:
        return

    snr_ref_db = float(tc.cfg.snr_ref_dbhz)

    def _sigma_of(el, snr):
        s = max(sigma, scale if tc.cfg.doppler_adaptive_sigma else 0.0)
        s /= max(np.sin(el), 0.1)
        if tc.cfg.doppler_snr_weight and snr > 0:
            s *= float(np.clip(10.0 ** ((snr_ref_db - snr) / 20.0), 1.0, 10.0))
        return s

    sigmas = [_sigma_of(r[5], r[6]) for r in rows]
    i_ref = int(np.argmin(sigmas))                     # best-weighted satellite
    ref, sigma_ref = rows[i_ref], sigmas[i_ref]

    omega = np.zeros(3)
    if epoch.gyro_mean is not None:
        bias_gyro = (tc.tc_bias.gyroscope() if tc.tc_bias is not None
                     else np.zeros(3))
        omega = np.asarray(epoch.gyro_mean, dtype=float) - bias_gyro

    key_idx = int(epoch.key_idx)
    lever = np.asarray(tc.lever_arm, dtype=float)
    rr = np.asarray(epoch.pred_ecef, dtype=float)
    n = 0
    huber = float(tc.cfg.doppler_huber)
    for row, sig in zip(rows, sigmas):
        if row is ref:
            continue
        # Differences share the reference satellite and are therefore
        # correlated; this keeps the diagonal approximation the DD factors use.
        noise = gtsam.noiseModel.Isotropic.Sigma(
            1, float(np.hypot(sig, sigma_ref)))
        if huber > 0:
            # Bounded influence. screen_rows references the predicted
            # velocity, so once the velocity state degrades, NLOS rows
            # survive the screen and drag it further — the kernel caps
            # what any surviving row can pull (run1 tunnel approach:
            # 2152 m blackout drift without it, 25 m with it).
            noise = gtsam.noiseModel.Robust.Create(
                gtsam.noiseModel.mEstimator.Huber.Create(huber), noise)
        epoch.graph.add(gtsam.SingleDifferenceDopplerFactorArm(
            tc.Xpose(key_idx), tc.Vel(key_idx),
            row[3], ref[3],                 # measured Doppler [Hz]
            row[4], ref[4],                 # wavelength [m/cycle]
            np.asarray(row[1], dtype=float), np.asarray(row[2], dtype=float),
            np.asarray(ref[1], dtype=float), np.asarray(ref[2], dtype=float),
            rr, lever, tc.ecef_T_nav, omega,
            0.0, 0.0,                       # satellite clock drift [s/s]
            noise))
        n += 1
    if n:
        epoch.info['doppler_sd_n'] = n
        epoch.info['doppler_sd_ref'] = int(ref[0])
