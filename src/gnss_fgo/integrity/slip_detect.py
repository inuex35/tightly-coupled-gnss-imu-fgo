"""Cycle slip detection (LLI + geometry-free) and outage arc expiry."""

from ..factors.factors_support import get_wavelengths as _get_wavelengths


def detect_slips_and_reset_ambiguities(tc, obs, obs_sd, sat, iu):
    """LLI + GF slip detection plus outage expiry. Returns (n_reset, reset_keys)."""
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
            if hasattr(obs, 'lli') and obs.lli[iu[i], f] == 1:
                reset_keys.add((s, f))
        if nf >= 2 and len(lams) >= 2:
            _detslp_gf(tc, obs, iu[i], s, lams, nf, reset_keys)

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
    """End the arc for every slipped (sat, f) by clearing amb_key and bumping amb_gen."""
    n_reset = 0
    for key in reset_keys:
        sat_st = tc._sat_states.get(*key)
        if sat_st.amb_key is not None:
            sat_st.amb_key = None
            n_reset += 1
        sat_st.amb_gen += 1
    return n_reset


def _detslp_gf(tc, obs, row, s, lams, nf, reset_keys):
    """Geometry-free L1-L2 jump detector."""
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
