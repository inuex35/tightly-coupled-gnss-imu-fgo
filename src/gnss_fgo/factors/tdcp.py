"""Time-differenced carrier phase (TDCP) relative-displacement factors.

Between two consecutive epochs the carrier phase of a continuously
tracked satellite changes by the change in geometric range plus a
receiver-clock drift term common to all satellites:

    dPhi_s = [rho_s(x_k) - rho_s(x_{k-1})] + ddelta + eps_s

The ambiguity and any slowly varying bias (NLOS offset, troposphere,
orbit error) cancel in the time difference, so TDCP stays accurate in
exactly the deep-urban stretches where the pseudoranges are poisoned —
it is the measurement that bounds float drift there.

The common receiver-clock delta is estimated as one scalar variable per
epoch pair (undifferenced form) instead of differencing across
satellites: differencing would inject the reference satellite's phase
error into every pair as fully correlated noise while the isotropic
model treats it as independent — overweighting it by sqrt(N). With the
clock variable each satellite's residual is genuinely independent.
Satellite clock drift is corrected with the broadcast dts at both
epochs; cycle slips are excluded via the per-epoch slip set and a
Huber kernel guards residual reflection-switch outliers.
"""

import numpy as np
import gtsam
from cssrlib.gnss import geodist, rCST

from ..utils.robust import maybe_robust as _maybe_robust


def _make_tdcp_factor(key_prev, key_curr, key_clk,
                      sat_prev_xyz, sat_curr_xyz, dphi_m,
                      lever_arr, ecef_T_nav, noise):
    """CustomFactor on (pose_{k-1}, pose_k, ddelta) for one satellite."""
    err_arr = np.empty(1, dtype=float)
    lever_local = np.asarray(lever_arr, dtype=float)
    one = np.array([[1.0]])

    def error_fn(this, values, jacobians):
        pose_p = values.atPose3(this.keys()[0])
        pose_c = values.atPose3(this.keys()[1])
        clk = values.atDouble(this.keys()[2])
        if ecef_T_nav is not None:
            pose_p = ecef_T_nav.compose(pose_p)
            pose_c = ecef_T_nav.compose(pose_c)
        Rp = pose_p.rotation().matrix()
        Rc = pose_c.rotation().matrix()
        ant_p = pose_p.translation() + Rp @ lever_local
        ant_c = pose_c.translation() + Rc @ lever_local

        rho_c, e_c = geodist(sat_curr_xyz, ant_c)
        rho_p, e_p = geodist(sat_prev_xyz, ant_p)

        err = (rho_c - rho_p) + clk - dphi_m

        if jacobians is not None:
            hc = -np.asarray(e_c)   # d err / d ant_c
            hp = np.asarray(e_p)    # d err / d ant_p
            Hc = np.empty((1, 6))
            hRc = hc @ Rc
            Hc[0, 0] = lever_local[1] * hRc[2] - lever_local[2] * hRc[1]
            Hc[0, 1] = lever_local[2] * hRc[0] - lever_local[0] * hRc[2]
            Hc[0, 2] = lever_local[0] * hRc[1] - lever_local[1] * hRc[0]
            Hc[0, 3:6] = hRc
            Hp = np.empty((1, 6))
            hRp = hp @ Rp
            Hp[0, 0] = lever_local[1] * hRp[2] - lever_local[2] * hRp[1]
            Hp[0, 1] = lever_local[2] * hRp[0] - lever_local[0] * hRp[2]
            Hp[0, 2] = lever_local[0] * hRp[1] - lever_local[1] * hRp[0]
            Hp[0, 3:6] = hRp
            jacobians[0] = Hp
            jacobians[1] = Hc
            jacobians[2] = one.copy()
        err_arr[0] = err
        return err_arr.copy()

    return gtsam.CustomFactor(noise, [key_prev, key_curr, key_clk], error_fn)


def add_tdcp_factors(tc, epoch):
    """Add undifferenced TDCP factors between Xpose(key_idx-1) and Xpose(key_idx)."""
    sigma = float(getattr(tc.cfg, 'tdcp_sigma', 0.0) or 0.0)
    prev = getattr(tc, '_tdcp_prev', None)
    # Snapshot the current epoch for the next one.
    snap = {}
    obs = epoch.obs
    f = 0
    for idx_pos, k in enumerate(epoch.iu):
        s = int(epoch.sat[idx_pos])
        L = float(obs.L[k, f]) if f < obs.L.shape[1] else 0.0
        lam = float(tc._sat_states.at(s, f).amb_lam or 0.0)
        if L == 0.0 or lam <= 0.0:
            continue
        if (s, f) in (getattr(epoch, 'slip_keys', None) or ()):
            continue
        rs_xyz = np.asarray(epoch.rs[k, :3], dtype=float)
        if not np.isfinite(rs_xyz).all() or np.linalg.norm(rs_xyz) < 1e3:
            continue
        dts = float(epoch.dts[k]) if epoch.dts is not None else 0.0
        snap[s] = (L * lam, rs_xyz, dts)
    tc._tdcp_prev = {'key_idx': epoch.key_idx, 'sats': snap}

    if sigma <= 0 or prev is None or prev.get('key_idx') != epoch.key_idx - 1:
        return 0
    prev_sats = prev['sats']
    common = sorted(s for s in snap if s in prev_sats)
    if len(common) < 4:
        return 0
    base_noise = gtsam.noiseModel.Isotropic.Sigma(1, sigma)
    noise = _maybe_robust(base_noise, 4.0, kind='tukey')  # redescending:
    # gross outliers (undetected mass slips at reacquisition) get ~zero
    # influence instead of Huber's bounded-but-nonzero pull
    lever_arr = (np.array(tc.lever_arm_tc)
                 if getattr(tc, 'lever_arm_tc', None) is not None
                 else np.zeros(3))
    key_prev = tc.Xpose(epoch.key_idx - 1)
    key_curr = tc.Xpose(epoch.key_idx)
    key_clk = gtsam.symbol('d', epoch.key_idx)
    epoch.values.insert(key_clk, 0.0)
    # Weak prior keeps the clock-delta variable determinate even if all
    # TDCP factors get FDE-removed later.
    epoch.graph.addPriorDouble(key_clk, 0.0, tc._noise1(1.0e4))
    n = 0
    for s in common:
        dphi = snap[s][0] - prev_sats[s][0]
        # Satellite clock advanced by (dts_k - dts_{k-1}); the phase moved
        # with it — correct the observable back to pure geometry+rx clock.
        dphi += rCST.CLIGHT * (snap[s][2] - prev_sats[s][2])
        epoch.graph.add(_make_tdcp_factor(
            key_prev, key_curr, key_clk,
            prev_sats[s][1], snap[s][1],
            dphi, lever_arr, tc.ecef_T_nav, noise))
        n += 1
    epoch.info['n_tdcp'] = n
    return n
