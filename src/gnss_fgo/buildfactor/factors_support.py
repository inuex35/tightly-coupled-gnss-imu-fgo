"""Cross-cutting helpers shared between the DD factor builder and the
DDCP build policy. The substantial factor builders live in their own
modules (``imu_preintegration.py``, ``nhc.py``, ``doppler.py``,
``zupt.py``) and are imported directly by their consumers.
"""

from ..utils import get_wavelengths as _utils_get_wavelengths


def get_wavelengths(tc, obs, sat):
    """Thin adapter — see utils.ambiguity.get_wavelengths."""
    return _utils_get_wavelengths(obs, sat, glo_ch=tc.nav.glo_ch)


def _hold_penalty_decision(ref_in_hold, j_in_hold, penalty):
    """Apply the hold-penalty rule to a DDCP pair: σ ×= ``penalty²`` when both sats are in hold (compounds), else ``penalty``. Returns ``(cp_allowed, cp_sigma_mult)``; ``penalty <= 0`` disables CP for the pair entirely."""
    if penalty <= 0:
        return False, 1.0
    cp_sigma_mult = penalty * penalty if (ref_in_hold and j_in_hold) else penalty
    return True, cp_sigma_mult


def compute_cp_build_policy(tc, sq_state, ref_sat, j_sat, freq, skip_cp):
    """Return ``(cp_allowed, cp_sigma_mult)`` for the DDCP pair.

    The policy is intentionally isolated from ``factors.py`` because it
    depends on several sat-quality states. Tier ordering (each returns
    on first match):

      1. forced hold (sq.tick + cp_hold_dirty_reset_enable)
      2. penalized hold (dirty-reset pre-stage)
      3. release probation (just-released held N)
      4. pair_bad (recent CP-vs-PR or post-fit fail count)
    """
    forced_hold = sq_state.forced_hold_per_sat
    penalized_hold = sq_state.penalized_per_sat
    probation_hold = sq_state.release_probation
    pair_bad_map = sq_state.recent_pair_bad

    ref_forced = bool(forced_hold and (ref_sat, freq) in forced_hold)
    j_forced = bool(forced_hold and (j_sat, freq) in forced_hold)
    ref_pen = bool(penalized_hold and (ref_sat, freq) in penalized_hold)
    j_pen = bool(penalized_hold and (j_sat, freq) in penalized_hold)
    ref_prob = bool(probation_hold and (ref_sat, freq) in probation_hold)
    j_prob = bool(probation_hold and (j_sat, freq) in probation_hold)

    if ref_forced or j_forced:
        return _hold_penalty_decision(
            ref_forced, j_forced, float(tc.cfg.cp_hold_sigma_penalty))
    if ref_pen or j_pen:
        # Dirty-reset penalty falls back to the forced-hold penalty
        # when its own knob is unset.
        penalty = float(tc.cfg.cp_hold_dirty_reset_penalty)
        if penalty <= 0:
            penalty = float(tc.cfg.cp_hold_sigma_penalty)
        return _hold_penalty_decision(ref_pen, j_pen, penalty)
    if ref_prob or j_prob:
        return _hold_penalty_decision(
            ref_prob, j_prob, float(tc.cfg.cp_release_probation_penalty))

    pair_bad_thr = float(tc.cfg.pair_bad_cp_hold_thresh or 0.0)
    if pair_bad_thr > 0.0:
        pair_bad = float(pair_bad_map.get((ref_sat, j_sat, freq), 0.0) or 0.0)
        if pair_bad >= pair_bad_thr:
            pair_bad_penalty = float(tc.cfg.pair_bad_cp_hold_penalty or 0.0)
            if pair_bad_penalty > 0.0:
                return True, pair_bad_penalty
            return False, 1.0

    return not skip_cp, 1.0

