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

from .doppler import _doppler_rows, screen_rows


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
