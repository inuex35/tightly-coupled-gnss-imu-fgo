"""Per-satellite raw Doppler (range-rate) factors.

Each rover Doppler observation becomes one ``gtsam.DopplerFactorArm`` on
[Xpose(kk), Vel(kk), Clk(kk-1), Clk(kk)]: the range rate constrains the
antenna velocity along the line of sight, and the receiver clock drift it
also contains is carried by the difference of two clock-bias states rather
than by a state of its own.

This replaces the earlier least-squares route (solve rover velocity + clock
drift from all Dopplers, then feed the result in as one PriorFactorVector).
That route measured as a no-op on tokyo run2 — 0.1572 m vs 0.1573 m 3D RMS
over 400 epochs with it switched off — because one isotropic 3-vector prior
throws away the per-satellite geometry.

Keeping the measurements separate lets the robust kernel act per satellite,
lets an epoch with three usable Dopplers contribute what it has instead of
being dropped for rank, and couples the measurement to attitude through the
lever arm (v_antenna = v_body + omega x lever).

Clock states are in seconds, as gtsam's GNSS factors define them
----------------------------------------------------------------
The whitened Jacobian on a clock state is ``C_LIGHT / dt / σ ~ 1e9`` against
~1 for pose and velocity (``PseudorangeFactor`` is the same, 3e8 at σ=1 m),
which looks alarming, and rescaling the state to metres of range by passing
``dt * C_LIGHT`` is an exact reparameterization -- ``dt`` appears nowhere
else in the factor (DopplerFactor.cpp:83-97, 226-248). Measured on tokyo
run2, 400 epochs, everything else identical: seconds gives 0.1582 m 3D RMS
with 353 fixes and no solver complaints, metres gives 0.1826 m with 335 and
four indeterminate-system events. Seconds it is.

What actually breaks the solve is a badly started clock chain, not its
units: initialize the state at zero and every satellite opens with a ~150
m/s residual, and pin every state with an absolute prior and the drift the
factors are there to measure is pinned along with it. The chain below is
started from a robust drift estimate and pinned only at its head.
"""

import numpy as np
import gtsam
from cssrlib.gnss import rCST, sat2prn, uGNSS

from ..utils import get_wavelengths


def _doppler_rows(tc, ed):
    """(sat, sat ECEF pos/vel, Doppler [Hz], wavelength [m], elevation [rad]).

    Mirrors the constellation and band choice the LS rows used: GLONASS is
    skipped (its FDMA channel makes the wavelength satellite-specific and the
    DD-only core never resolves it), and each satellite contributes the first
    band that actually carries a Doppler observation.
    """
    rows = []
    for si, i_obs in enumerate(ed.iu):
        s = int(ed.sat[si])
        if sat2prn(s)[0] == uGNSS.GLO:
            continue
        if i_obs >= ed.obs.D.shape[0]:
            continue
        lams = None
        for f in range(min(tc.nav.nf, ed.obs.D.shape[1])):
            d_obs = ed.obs.D[i_obs, f]
            if d_obs == 0.0:
                continue
            if lams is None:
                lams = get_wavelengths(ed.obs_sd, s, glo_ch=tc.nav.glo_ch)
            if f < len(lams) and lams[f] > 0:
                rows.append((s, ed.rs[i_obs, :3], ed.vs[i_obs, :3],
                             float(d_obs), float(lams[f]),
                             float(ed.el[si])))
                break
    return rows


def _estimate_clock_drift(tc, ed, rows):
    """Median of (measured range rate − geometric rate) over the rows [m/s].

    The receiver clock drift is common to every satellite, so the median of
    what the geometry cannot explain is a robust estimate of it. Used only to
    initialize the clock state — the graph re-estimates it.
    """
    if not rows:
        return 0.0
    v_r = np.asarray(ed.R, dtype=float) @ np.asarray(
        ed.pred.velocity(), dtype=float)
    p_r = np.asarray(ed.pred_ecef, dtype=float)
    resid = []
    for _s, p_sat, v_sat, d_obs, lam, _el in rows:
        d_vec = np.asarray(p_sat, dtype=float) - p_r
        rho = np.linalg.norm(d_vec)
        if rho < 1.0:
            continue
        e = d_vec / rho
        geom = float(e @ (np.asarray(v_sat, dtype=float) - v_r))
        resid.append(-lam * d_obs - geom)
    return float(np.median(resid)) if resid else 0.0


def add_doppler_factors(tc, ed):
    """Add one DopplerFactorArm per rover Doppler when ``cfg.doppler_sigma > 0``.

    Maintains the clock-bias chain the factors difference: the head of a chain
    is pinned by a prior (range rates observe the difference only, so the
    level is free), and consecutive states are tied by a random walk so the
    chain stays determined through epochs that carry no usable Doppler.
    """
    sigma = float(tc.cfg.doppler_sigma)
    tc._doppler_keep_keys = []
    if sigma <= 0 or ed.kk is None:
        return

    kk = int(ed.kk)
    dt = float(tc._epoch_dt)
    chain_prev = tc._doppler_clk_last
    rows = _doppler_rows(tc, ed)

    cb_prev = 0.0
    if chain_prev == kk - 1 and ed.est2 is not None:
        try:
            cb_prev = float(ed.est2.atDouble(tc.Clk(kk - 1)))
        except RuntimeError:
            cb_prev = 0.0
    drift_mps = _estimate_clock_drift(tc, ed, rows)
    cb_init = cb_prev + drift_mps * dt / rCST.CLIGHT     # [s]
    ed.v3.insert(tc.Clk(kk), cb_init)
    ed.info['doppler_drift_mps'] = drift_mps

    if chain_prev != kk - 1:
        # No usable predecessor (first Doppler epoch, or the chain was cut by
        # a warm reset): pin this end and start differencing next epoch. Only
        # the head of a chain is pinned -- range rates observe the difference
        # only, so the level is free to be anything, but pinning every state
        # would pin the difference too and leave the Doppler nothing to say.
        ed.g3.add(gtsam.PriorFactorDouble(
            tc.Clk(kk), cb_init,
            tc._noise1(tc.cfg.doppler_clk_anchor_sigma)))
        tc._doppler_clk_last = kk
        ed.info['doppler_chain'] = 'anchored'
        return

    ed.g3.add(gtsam.BetweenFactorDouble(
        tc.Clk(kk - 1), tc.Clk(kk), cb_init - cb_prev,
        tc._noise1(float(tc.cfg.doppler_clk_rw) * dt / rCST.CLIGHT)))
    tc._doppler_clk_last = kk
    # The factors reach back one epoch, so that key must survive the
    # fixed-lag window this update (fls_lag is one epoch wide by default).
    tc._doppler_keep_keys = [tc.Clk(kk - 1)]

    huber = float(tc.cfg.doppler_huber)

    def _model(el_rad):
        # Elevation weighting as in gtsam's DopplerVelocityExample notebook:
        # the zenith sigma divided by sin(el), floored so a horizon satellite
        # is downweighted by 10x rather than by an unbounded factor.
        base = gtsam.noiseModel.Isotropic.Sigma(
            1, sigma / max(np.sin(el_rad), 0.1))
        if huber <= 0:
            return base
        return gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber.Create(huber), base)

    omega = np.zeros(3)
    if ed.gyro_mean is not None:
        bias_gyro = (tc.tc_bias.gyroscope() if tc.tc_bias is not None
                     else np.zeros(3))
        omega = np.asarray(ed.gyro_mean, dtype=float) - bias_gyro

    lever = np.asarray(tc.lever_arm, dtype=float)
    rr = np.asarray(ed.pred_ecef, dtype=float)
    n = 0
    for s, p_sat, v_sat, d_obs, lam, el in rows:
        ed.g3.add(gtsam.DopplerFactorArm(
            tc.Xpose(kk), tc.Vel(kk), tc.Clk(kk - 1), tc.Clk(kk),
            d_obs, lam,
            np.asarray(p_sat, dtype=float), np.asarray(v_sat, dtype=float),
            rr, lever, tc.ecef_T_nav, omega, dt, 0.0, _model(el)))
        n += 1
    ed.info['doppler_n'] = n
