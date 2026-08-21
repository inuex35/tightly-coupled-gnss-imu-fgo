"""Fix-and-hold: turn an accepted fix into priors the graph keeps.

RTKLIB's armode==3, adapted to the two-smoother layout: Phase 1 holds each
ambiguity with a PriorDouble at sigma = sqrt(varholdamb). Phase 2 adds no
graph factors at all — the held integers live on the per-satellite hold
state and ``nav.x`` (the value re-enters AR as a pinned input via
ar_problem, write_marginals mirrors it into ``nav.P``, and the DDCP
builder folds it into the factor offset).

Entry point: :func:`apply_fix_and_hold`.
"""

import numpy as np
import gtsam

from ..utils import sorted_amb_items


def _collect_held_sat_freq_keys(tc, amb_dict):
    """Return [(s, f), ...] for ambiguities that holdamb_flags() just promoted to fix=3."""
    return [(s, f) for (s, f), _k in sorted_amb_items(amb_dict)
            if tc.nav.fix[s - 1, f] == 3]


def _add_phase1_hold_priors(tc, hg, hold_keys, amb_dict, xa):
    """Add per-held-N PriorDouble factors with σ = √varholdamb (RTKLIB-style, in cycles)."""
    hold_sigma = float(np.sqrt(tc.cfg.varholdamb))
    hold_noise = tc._noise1(hold_sigma)
    for s, f in hold_keys:
        hg.addPriorDouble(
            amb_dict[(s, f)], xa[tc.IB(s, f, tc.nav.na)], hold_noise)


def _apply_holds_phase1(tc, hg, hold_keys, amb_dict):
    """Phase 1: IncrementalFixedLagSmoother.update with held-N timestamps."""
    isam = tc.isam
    ts_h1 = gtsam.FixedLagSmootherKeyTimestampMap()
    t_p1 = tc.phase1_t
    for sf in hold_keys:
        ts_h1[amb_dict[sf]] = t_p1
    try:
        isam.update(hg, gtsam.Values(), ts_h1)
        tc.total_factor_count += hg.size()
    except (RuntimeError, IndexError):
        pass


def _activate_phase2_hold_states(tc, hold_keys, xa):
    """Phase 2: copy held N → sat_state hold + nav.x; clear amb_key."""
    for s, f in hold_keys:
        held_value = float(xa[tc.IB(s, f, tc.nav.na)])
        sat_st = tc._sat_states.get(s, f)
        sat_st.activate_hold(held_value)
        # Count the FIX streak HERE, before amb_key is cleared: the
        # Stage-D streak loop only sees (s, f) still carrying amb_key,
        # so held sats never counted and prev_fix_streak_max was 0 on
        # every epoch (review r5 #1) — the low-nb established-fix
        # guard could never pass.
        sat_st.fix_streak += 1
        sat_st.amb_key = None
        tc.nav.x[tc.IB(s, f, tc.nav.na)] = held_value


def apply_fix_and_hold(tc, key_pose, amb_dict, xa):
    """Phase D — fix-and-hold (armode==3): mark held flags, then Phase 1 adds hold-prior factors while Phase 2 activates the hold on sat_states / nav.x (no graph factors). Always returns True; the bool return survives for the Phase-1 call shape."""
    tc.holdamb_flags()
    hold_keys = _collect_held_sat_freq_keys(tc, amb_dict)
    if tc.phase != 2:
        hg = gtsam.NonlinearFactorGraph()
        _add_phase1_hold_priors(tc, hg, hold_keys, amb_dict, xa)
        if hg.size() > 0:
            _apply_holds_phase1(tc, hg, hold_keys, amb_dict)
    else:
        _activate_phase2_hold_states(tc, hold_keys, xa)
    return True


