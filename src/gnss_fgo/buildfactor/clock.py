"""Undifferenced pseudoranges that make the receiver clock chain observable.

The Doppler factors carry the receiver clock drift as the difference of two
clock-bias states. In a double-difference graph nothing else uses those
states, so the drift is estimated from the Dopplers alone -- and when the
sky narrows to a handful of satellites in one cone, the clock-drift
direction and the velocity along the mean line of sight are nearly
degenerate. The solver then splits the error between them: measured on
tokyo run2 at the epochs where the DD solution drops out, the clock drift
came out 1.4 m/s off the truth-derived value and the velocity 1 m/s off,
with per-satellite Doppler residuals still at 0.03 m/s. The velocity error
integrates into position and the float excursion grows to 46 m where a run
with no Doppler at all drifts 6.6 m.

Undifferenced pseudoranges on the same clock states break that degeneracy:
they observe the clock *level* against the code solution, so an error can no
longer hide in the clock. They are deliberately weak (metres, against the
centimetres the DD factors give) -- the position is already pinned by the
DD factors, so what these factors carry is the clock.

Kept simple on purpose:

* GPS only. A single clock series across constellations would absorb the
  inter-system biases, which are tens of nanoseconds; modelling those needs
  one clock state per system.
* Ionosphere-free L1/L2 combination rather than a broadcast ionosphere
  model, which also puts the measurement on the same reference as the
  broadcast satellite clock (no TGD term).
* Saastamoinen zenith delay with a Black-Eisner mapping for the troposphere.
"""

import numpy as np
import gtsam
from cssrlib.gnss import ecef2pos, rCST, sat2prn, tropmodel, uGNSS, uTYP


def _tropo_delay(obs_t, pos, el_rad):
    """Saastamoinen zenith delay mapped to the line of sight [m]."""
    hs, wet, _ = tropmodel(obs_t, pos)
    m = 1.001 / np.sqrt(0.002001 + np.sin(max(el_rad, np.deg2rad(3.0)))**2)
    return (hs + wet) * m


def _iono_free_rows(tc, ed):
    """(satellite, ECEF position, IF pseudorange [m], sat clock [s], el [rad])."""
    pos = ecef2pos(np.asarray(ed.pred_ecef, dtype=float))
    rows = []
    for si, i_obs in enumerate(ed.iu):
        s = int(ed.sat[si])
        sys_i = sat2prn(s)[0]
        if sys_i != uGNSS.GPS:
            continue
        sigs = ed.obs_sd.sig.get(sys_i, {}).get(uTYP.C, [])
        if len(sigs) < 2:
            continue
        p1, p2 = ed.obs.P[i_obs, 0], ed.obs.P[i_obs, 1]
        if p1 == 0.0 or p2 == 0.0:
            continue
        f1, f2 = sigs[0].frequency(), sigs[1].frequency()
        if f1 <= 0 or f2 <= 0 or abs(f1 - f2) < 1.0:
            continue
        g = (f1 * f1) / (f1 * f1 - f2 * f2)
        pr_if = g * p1 - (g - 1.0) * p2
        el = float(ed.el[si])
        rows.append((s, np.asarray(ed.rs[i_obs, :3], dtype=float),
                     pr_if - _tropo_delay(ed.obs.t, pos, el),
                     float(ed.dts[i_obs]), el))
    return rows


def estimate_clock_bias(tc, ed):
    """Receiver clock bias from the code solution [s], or None.

    Median over satellites of ``(PR - range)/c + dts``. The chain's level is
    arbitrary as far as the Doppler factors are concerned, but as soon as
    pseudoranges observe it, it has to start at the value they mean.
    """
    rows = _iono_free_rows(tc, ed)
    if not rows:
        return None
    p_r = np.asarray(ed.pred_ecef, dtype=float)
    vals = [(pr - float(np.linalg.norm(np.asarray(p_sat) - p_r)))
            / rCST.CLIGHT + dts for _s, p_sat, pr, dts, _el in rows]
    return float(np.median(vals))


def add_clock_pr_factors(tc, ed):
    """Add PseudorangeFactorArm on [Xpose(kk), Clk(kk)] when enabled.

    Requires the Doppler builder to have created Clk(kk) for this epoch --
    the clock states exist for the Doppler factors, and these pseudoranges
    are here to make them observable, not the other way round.
    """
    sigma = float(tc.cfg.clock_pr_sigma)
    if sigma <= 0 or ed.kk is None or tc._doppler_clk_last != int(ed.kk):
        return

    lever = np.asarray(tc.lever_arm, dtype=float)
    n = 0
    for _s, p_sat, pr, dts, el in _iono_free_rows(tc, ed):
        model = gtsam.noiseModel.Isotropic.Sigma(
            1, sigma / max(np.sin(el), 0.1))
        ed.g3.add(gtsam.PseudorangeFactorArm(
            tc.Xpose(int(ed.kk)), tc.Clk(int(ed.kk)), pr, p_sat, lever,
            tc.ecef_T_nav, dts, model))
        n += 1
    if n:
        ed.info['clock_pr_n'] = n
