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

      1. forced hold (sq.tick — post-recovery CP distrust window)
    """
    forced_hold = sq_state.forced_hold_per_sat

    ref_forced = bool(forced_hold and (ref_sat, freq) in forced_hold)
    j_forced = bool(forced_hold and (j_sat, freq) in forced_hold)

    if ref_forced or j_forced:
        return _hold_penalty_decision(
            ref_forced, j_forced, float(tc.cfg.cp_hold_sigma_penalty))

    return not skip_cp, 1.0

