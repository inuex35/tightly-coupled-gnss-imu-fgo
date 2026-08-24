"""Slip detection on raw observations (Stage B, step 2)."""


from ..factors.factors_support import get_wavelengths as _get_wavelengths


def detect_slips_and_reset_ambiguities(tc, obs, obs_sd, sat, iu):
    """LLI + GF slip detection plus the outage expiry, then reset
    every flagged ambiguity. Returns (n_reset, slip_keys)."""
    nf = tc.nav.nf
    ns = len(sat)
    reset_keys = set()
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

        if nf >= 2 and len(lams) >= 2:
            _detslp_gf(tc, obs, iu[i], s, lams, nf, reset_keys)

    # Outage counter: increment for satellites NOT seen this epoch
    maxout = tc._sat_states.maxout
    for key in list(tc._sat_states.keys()):
        if key not in current_sats:
            st = tc._sat_states.get(*key)
            st.outc += 1
            if st.outc > maxout:
                reset_keys.add(key)

    n_reset = reset_slipped_ambiguities(tc, reset_keys)
    return n_reset, reset_keys


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


def detslp_cmc(sat_state, obs, obsb, row, brow, f, lam, thresh):
    """Code-minus-carrier jump detector. Not wired — kept for
    receivers with poorer LLI. Returns True on a jump."""
    pr_rov = obs.P[row, f]
    cp_rov = obs.L[row, f]
    pr_bas = obsb.P[brow, f]
    cp_bas = obsb.L[brow, f]
    if pr_rov == 0 or cp_rov == 0 or pr_bas == 0 or cp_bas == 0:
        return False
    cmc = (pr_rov - pr_bas) - (cp_rov - cp_bas) * lam
    prev = getattr(sat_state, 'cmc', None)
    sat_state.cmc = cmc
    return prev is not None and abs(cmc - prev) > thresh


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
