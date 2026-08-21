"""Slip detection on raw observations (Stage B, step 2)."""


from ..factors.factors_support import get_wavelengths as _get_wavelengths


def detect_slips_and_reset_ambiguities(tc, obs, obs_sd, sat, iu,
                                obsb=None, ir_map=None):
    """Run the four slip/multipath detectors (LLI, CMC, GF, Doppler)
    plus the outage expiry, then reset every flagged ambiguity.
    Returns (n_reset, n_cmc_jumps, slip_keys)."""
    nf = tc.nav.nf
    ns = len(sat)
    reset_keys = set()
    cmc_exclude = set()  # CMC jump → ambiguity reset
    current_sats = set()

    for i in range(ns):
        s = sat[i]
        lams = _get_wavelengths(tc, obs_sd, s)

        for f in range(nf):
            current_sats.add((s, f))
            sat_state = tc._sat_states.get(s, f)
            sat_state.outc = 0

            # LLI check
            if hasattr(obs, 'lli') and obs.lli[iu[i], f] == 1:
                reset_keys.add((s, f))

            if (obsb is not None and ir_map is not None and
                    s in ir_map and f < len(lams) and lams[f] > 0 and
                    tc.cfg.cmc_thresh > 0):
                _detslp_cmc(tc, sat_state, obs, obsb, iu[i], ir_map[s],
                            s, f, lams[f], cmc_exclude)

        if nf >= 2 and len(lams) >= 2:
            _detslp_gf(tc, obs, iu[i], s, lams, nf, reset_keys)

    # CMCで検出された衛星もリセット（マルチパスはNを壊す）
    reset_keys.update(cmc_exclude)


    # Outage counter: increment for satellites NOT seen this epoch
    maxout = tc._sat_states.maxout
    for key in list(tc._sat_states.keys()):
        if key not in current_sats:
            st = tc._sat_states.get(*key)
            st.outc += 1
            if st.outc > maxout:
                reset_keys.add(key)

    n_reset = reset_slipped_ambiguities(tc, reset_keys)
    return n_reset, len(cmc_exclude), reset_keys


def reset_slipped_ambiguities(tc, reset_keys):
    """Kill the ambiguity of every slipped (sat, f): drop the key, clear
    holds, bump the generation (next build creates a fresh N variable)."""
    n_reset = 0
    for key in reset_keys:
        sat_st = tc._sat_states.get(*key)
        if sat_st.amb_key is not None:
            sat_st.amb_key = None
            n_reset += 1
        sat_st.clear_hold()
        sat_st.amb_gen += 1
    return n_reset


def _detslp_cmc(tc, sat_state, obs, obsb, row, brow, s, f, lam,
                cmc_exclude):
    """Code-minus-carrier: jump -> slip flag; sustained level -> DD skip."""
    pr_rov = obs.P[row, f]
    cp_rov = obs.L[row, f]
    pr_bas = obsb.P[brow, f]
    cp_bas = obsb.L[brow, f]
    if pr_rov == 0 or cp_rov == 0 or pr_bas == 0 or cp_bas == 0:
        return
    cmc = (pr_rov - pr_bas) - (cp_rov - cp_bas) * lam
    prev = sat_state.cmc
    if prev is not None and abs(cmc - prev) > tc.cfg.cmc_thresh:
        cmc_exclude.add((s, f))
    sat_state.cmc = cmc


def _detslp_gf(tc, obs, row, s, lams, nf, reset_keys):
    """Geometry-free L1-L2 combination: a jump means a slip on some band."""
    L1 = obs.L[row, 0]
    L2 = obs.L[row, 1]
    if L1 == 0 or L2 == 0 or lams[0] <= 0 or lams[1] <= 0:
        return
    gf = L1 * lams[0] - L2 * lams[1]
    prev_gf = tc._sat_states.gf
    if s in prev_gf and prev_gf[s] != 0:
        if abs(gf - prev_gf[s]) > tc.cfg.thres_slip:
            for f in range(nf):
                reset_keys.add((s, f))
    prev_gf[s] = gf
