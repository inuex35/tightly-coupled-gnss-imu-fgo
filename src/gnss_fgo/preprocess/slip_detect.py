"""Stage 1 — slip detection on raw observations."""

import numpy as np

from cssrlib.gnss import time2gpst
from ..buildfactor import factors as _tc_factors


def detect_slips_and_manage_amb(tc, obs, obs_sd, sat, iu,
                                obsb=None, ir_map=None):
    """Detect cycle slips and remove affected ambiguities."""
    nf = tc.nav.nf
    ns = len(sat)
    reset_keys = set()
    cmc_exclude = set()  # CMC jump → ambiguity reset
    cmc_level_exclude = set()  # sustained multipath → DD exclude this epoch
    current_sats = set()

    for i in range(ns):
        s = sat[i]
        lams = _tc_factors.get_wavelengths(tc, obs_sd, s)

        for f in range(nf):
            current_sats.add((s, f))
            sat_state = tc._sat_states.get(s, f)
            sat_state.outc = 0

            # LLI check
            if hasattr(obs, 'lli') and obs.lli[iu[i], f] == 1:
                reset_keys.add((s, f))

            if (obsb is not None and ir_map is not None and
                    s in ir_map and f < len(lams) and lams[f] > 0 and
                    tc.cmc_thresh > 0):
                pr_rov = obs.P[iu[i], f]
                cp_rov = obs.L[iu[i], f]
                pr_bas = obsb.P[ir_map[s], f]
                cp_bas = obsb.L[ir_map[s], f]
                if pr_rov != 0 and cp_rov != 0 and pr_bas != 0 and cp_bas != 0:
                    cmc = (pr_rov - pr_bas) - (cp_rov - cp_bas) * lams[f]
                    # Jump check (legacy)
                    prev = sat_state.cmc
                    if prev is not None and abs(cmc - prev) > tc.cmc_thresh:
                        cmc_exclude.add((s, f))
                    sat_state.cmc = cmc
                    # Level check (sustained multipath)
                    level_thr = tc.cfg.cmc_level_thresh
                    if level_thr > 0:
                        warmup = tc.cfg.cmc_warmup_epochs
                        baseline = sat_state.cmc_baseline
                        count = sat_state.cmc_count
                        if baseline is None:
                            sat_state.cmc_baseline = cmc
                            sat_state.cmc_count = 1
                        elif count < warmup:
                            # Average over warmup window
                            sat_state.cmc_baseline = (
                                (baseline * count + cmc) / (count + 1))
                            sat_state.cmc_count = count + 1
                        else:
                            # Steady-state: detect deviation, slow-update mean
                            if abs(cmc - baseline) > level_thr:
                                cmc_level_exclude.add((s, f))
                            else:
                                a = tc.cfg.cmc_alpha
                                sat_state.cmc_baseline = (
                                    (1 - a) * baseline + a * cmc)

        if nf >= 2 and len(lams) >= 2:
            L1 = obs.L[iu[i], 0]
            L2 = obs.L[iu[i], 1]
            if L1 != 0 and L2 != 0 and lams[0] > 0 and lams[1] > 0:
                gf = L1 * lams[0] - L2 * lams[1]
                use_avg = bool(tc.cfg.gf_avg_enable)
                if use_avg:
                    state = tc._gf_state.get(s)
                    if state is None:
                        tc._gf_state[s] = [float(gf), 1, float(gf)]
                    else:
                        mean = state[2]
                        if abs(gf - mean) > tc.thresslip:
                            for f in range(nf):
                                reset_keys.add((s, f))
                            tc._gf_state[s] = [float(gf), 1, float(gf)]
                        else:
                            state[0] += float(gf)
                            state[1] += 1
                            state[2] = state[0] / state[1]
                else:
                    prev_gf = tc._sat_states.gf
                    if s in prev_gf and prev_gf[s] != 0:
                        if abs(gf - prev_gf[s]) > tc.thresslip:
                            for f in range(nf):
                                reset_keys.add((s, f))
                    prev_gf[s] = gf

    if bool(tc.cfg.gf_avg_enable):
        gf_state = tc._gf_state
        if gf_state is not None:
            seen_now = {sat[i] for i in range(ns)}
            for s_key in list(gf_state.keys()):
                if s_key not in seen_now:
                    del gf_state[s_key]

    # CMCで検出された衛星もリセット（マルチパスはNを壊す）
    reset_keys.update(cmc_exclude)
    tc._sat_states.cmc_skip_dd = cmc_level_exclude

    if tc.cfg.thresdop > 0 and hasattr(obs, 'D') and obs.D.size > 0:
        _detslp_dop(tc, obs, sat, iu, reset_keys)

    if tc.cfg.mw_thresh > 0 and nf >= 2:
        _detslp_mw(tc, obs, sat, iu, reset_keys)

    # Outage counter: increment for satellites NOT seen this epoch
    maxout = tc._sat_states.maxout
    for key in list(tc._sat_states.keys()):
        if key not in current_sats:
            st = tc._sat_states.get(*key)
            st.outc += 1
            if st.outc > maxout:
                reset_keys.add(key)

    # Collect factor indices to remove, increment generation, clear amb keys
    remove_indices = []
    n_reset = 0
    for key in reset_keys:
        sat_st = tc._sat_states.get(*key)
        if sat_st.amb_key is not None:
            sat_st.amb_key = None
            n_reset += 1
        sat_st.clear_hold()
        if sat_st.amb_factor_indices:
            remove_indices.extend(sat_st.amb_factor_indices)
            sat_st.amb_factor_indices = []
        # Increment generation → next _build_dd will create new GTSAM variable
        sat_st.amb_gen += 1

    return n_reset, remove_indices, len(cmc_exclude), reset_keys


def _detslp_dop(tc, obs, sat, iu, reset_keys):
    """Inner helper for the Doppler-based slip detector — RTKLIB-demo5"""
    nf = tc.nav.nf
    ns = len(sat)
    thr = tc.cfg.thresdop
    _, tow_now = time2gpst(obs.t)

    dopdif = {}     # (sat, f) → cycles/s residual
    inliers = []
    for i in range(ns):
        s = sat[i]
        for f in range(nf):
            if f >= obs.L.shape[1] or f >= obs.D.shape[1]:
                continue
            L = obs.L[iu[i], f]
            D = obs.D[iu[i], f]
            if L == 0.0 or D == 0.0:
                continue
            _prev_st = tc._sat_states.track.get((s, f))
            prev = _prev_st.prev_phase if _prev_st is not None else None
            if prev is None:
                continue
            L_prev, t_prev = prev
            dt = tow_now - t_prev
            if dt <= 0.0:
                continue
            dph = (L - L_prev) / dt
            dpt = -D
            dd = dph - dpt
            dopdif[(s, f)] = dd
            if abs(dd) < 3 * thr:
                inliers.append(dd)

    if inliers:
        mean_dop = float(np.mean(inliers))
        for (s, f), dd in dopdif.items():
            if abs(dd - mean_dop) > thr:
                reset_keys.add((s, f))

    for st in tc._sat_states.track.values():
        st.prev_phase = None
    for i in range(ns):
        s = sat[i]
        for f in range(nf):
            if f >= obs.L.shape[1] or f >= obs.D.shape[1]:
                continue
            L = obs.L[iu[i], f]
            D = obs.D[iu[i], f]
            if L != 0.0 and D != 0.0:
                tc._sat_states.get(s, f).prev_phase = (L, tow_now)


def _compute_mw_n_wl(tc, obs, iu_idx, s, f):
    """Compute Melbourne-Wübbena N_WL [cycles] for sat s on (L1, Lf) pair; return None when L1/Lf/P1/Pf / λ are unusable."""
    if (iu_idx >= obs.L.shape[0] or obs.L.shape[1] <= f
            or obs.P.shape[1] <= f or f < 1):
        return None
    L1 = obs.L[iu_idx, 0]
    Lf = obs.L[iu_idx, f]
    P1 = obs.P[iu_idx, 0]
    Pf = obs.P[iu_idx, f]
    if L1 == 0.0 or Lf == 0.0 or P1 == 0.0 or Pf == 0.0:
        return None
    lams = _tc_factors.get_wavelengths(tc, obs, s)
    if len(lams) <= f or lams[0] <= 0 or lams[f] <= 0 or lams[0] == lams[f]:
        return None
    f1 = 1.0 / lams[0]      # cycles/m equivalent (c cancels out below)
    f2 = 1.0 / lams[f]
    lam_wl = 1.0 / (f1 - f2)
    if lam_wl <= 0:
        return None
    phi_wl = L1 - Lf                                  # cycles
    p_nl = (f1 * P1 + f2 * Pf) / (f1 + f2)            # m
    return phi_wl - p_nl / lam_wl                     # cycles


def _check_slip_mw_avg(tc, s, f, n_wl, thr, reset_keys):
    """Time-averaged MW per-(s,f): compare n_wl against the running mean; on |Δ|>thr flag (s,0) and (s,f) for reset and reset accumulator."""
    key = (s, f)
    state = tc._mw_state.get(key)
    if state is None:
        tc._mw_state[key] = [float(n_wl), 1, float(n_wl)]
        return
    mean = state[2]
    if abs(n_wl - mean) > thr:
        # MW(L1, Lf) jump can be L1 or Lf slip — reset both.
        reset_keys.add((s, 0))
        reset_keys.add(key)
        tc._mw_state[key] = [float(n_wl), 1, float(n_wl)]
    else:
        state[0] += float(n_wl)
        state[1] += 1
        state[2] = state[0] / state[1]


def _check_slip_mw_legacy(tc, s, f, n_wl, thr, reset_keys, new_mw):
    """Single-epoch MW per-(s,f): compare n_wl against previous epoch's value; on |Δ|>thr flag (s,0) and (s,f)."""
    key = (s, f)
    prev = tc._sat_states.mw.get(key)
    if prev is not None and abs(n_wl - prev) > thr:
        reset_keys.add((s, 0))
        reset_keys.add(key)
    new_mw[key] = n_wl


def _purge_unseen_mw_state(tc, seen_keys, use_avg, new_mw):
    """Drop state for (s,f) keys not seen this epoch so re-acquisition starts a fresh accumulator (avoids stale-mean mis-compare across the gap)."""
    if use_avg:
        for key in list(tc._mw_state.keys()):
            if key not in seen_keys:
                del tc._mw_state[key]
    else:
        tc._sat_states.mw = new_mw


def _detslp_mw(tc, obs, sat, iu, reset_keys):
    """Melbourne-Wübbena cycle-slip detector (rover-only, per (sat, freq))."""
    nf = tc.nav.nf
    if nf < 2:
        return
    ns = len(sat)
    thr = tc.cfg.mw_thresh
    use_avg = bool(tc.cfg.mw_avg_enable)
    new_mw = {}
    if not hasattr(tc, '_mw_state'):
        tc._mw_state = {}
    seen_keys = set()

    for i in range(ns):
        s = sat[i]
        for f in range(1, nf):
            n_wl = _compute_mw_n_wl(tc, obs, iu[i], s, f)
            if n_wl is None:
                continue
            seen_keys.add((s, f))
            if use_avg:
                _check_slip_mw_avg(tc, s, f, n_wl, thr, reset_keys)
            else:
                _check_slip_mw_legacy(tc, s, f, n_wl, thr, reset_keys, new_mw)

    _purge_unseen_mw_state(tc, seen_keys, use_avg, new_mw)
