"""Every write the AR path makes to cssrlib's ``nav`` state, in one place.

The graph is the estimator; ``nav`` survives as the interface to cssrlib's
validation (``zdres``/``sdres``/``valpos``) and to this project's own gates.
These writes *are* the contract
between the two worlds, and every one of them has been broken at least once
by being implicit -- the table says who reads each field, so the next change
knows what it is touching.

    field                    written by              read by
    -----                    ----------              -------
    nav.x[0:3]               stage/postfit           zdres/sdres/valpos, gates
    nav.x[na:]               publish_marginals       ddidx (nonzero = has an
                                                     estimate), sdres trace
    nav.P (held diag)        publish_marginals       sdres optional file trace
    nav.vsat                 publish_marginals       ddidx selection, retries
    nav.el                   qcedit (cssrlib)        ddidx mask, weights
    nav.fix                  ddidx                   hold policy
    nav.lock                 retry, every call       next epoch's arfilter
    nav.excsat               retry outcome           next epoch's round-robin
    nav.prev_ratio2          retry outcome           next epoch's arfilter
    tc._last_s0/_last_s1     every resolution        ratio gates, diagnostics

Contract: satellite ids are 1..MAXSAT (cssrlib guarantees the range),
so ``nav.*[s - 1, f]`` indexing is unguarded everywhere by design.

"""


import numpy as np

from ..utils import sorted_amb_items



def publish_attempt(tc, sat_list, result):
    """Side effects of one resolution attempt, accepted or not.

    ``nav.fix`` comes from ``ddidx`` itself rather than a re-implementation:
    its marks are quirky and hand-written reproductions have drifted.
    """
    tc.ddidx(tc.nav, sat_list)
    tc._last_s0, tc._last_s1 = result.s0, result.s1


def update_lock_counters(tc, sat_list):
    """RTKLIB ``ssat[].lock`` semantics, run on every retry-path call.

    Incremented for satellites valid this epoch, reset for the rest. Next
    epoch's ``arfilter`` reads ``lock == 1`` as "freshly acquired" -- leave
    this out and the retry excludes a different satellite from then on.
    """
    valid = {int(s) for s in sat_list}
    for i in range(tc.nav.lock.shape[0]):
        sv = i + 1
        for f in range(tc.nav.nf):
            if sv in valid and tc.nav.vsat[i, f] != 0:
                tc.nav.lock[i, f] += 1
            else:
                tc.nav.lock[i, f] = 0


def publish_retry_success(tc, ratio):
    """First-pass fix: prev_ratio2 follows it, the cursor resets.

    A degenerate exact-fit success (ratio 0) measured nothing: keep the
    last real ratio instead of overwriting the heuristic's memory.
    """
    if ratio > 0.0:
        tc.nav.prev_ratio2 = ratio
    tc.nav.excsat = 0


def publish_retry_outcome(tc, fixed, ratio, excluded_sat):
    """Second pass done: remember the exclusion only when it worked."""
    if fixed:
        if ratio > 0.0:
            tc.nav.prev_ratio2 = ratio
        tc.nav.excsat = excluded_sat
    else:
        tc.nav.excsat = 0


def publish_marginals(tc, estimate, key_pose, amb_dict):
    """Publish the AR-selection state: float/held ambiguities, vsat, key_pose.

    The covariance readback that used to fill nav.P from joint marginals
    was measured dead (its only reader is sdres's optional file trace)
    and removing it is line-identical and several times faster; only the
    cheap held-variance diagonal is still written.
    """
    # The resolver reads the marginals straight from ISAM2 and needs the
    # very pose they are taken against, not tc.tc_epoch, which still
    # points at the previous epoch while AR runs.
    tc._ar_key_pose = key_pose
    tc.nav.P[:, :] = 0
    tc.nav.vsat[:, :] = 0
    tc.nav.x[tc.nav.na:] = 0
    cp_visible_sf = set(tc._ar_cp_visible_sf)
    _publish_float_ambiguities(tc, estimate, amb_dict)
    _publish_held_ambiguities(tc, cp_visible_sf, amb_dict)


def _publish_float_ambiguities(tc, estimate, amb_dict):
    """nav.x gets the float N values; vsat marks AR eligibility
    (converged age, not persist-bad). Diagnostics land on tc._last_*."""
    diag_estimate_missing = 0
    diag_vsat1 = 0
    diag_vsat0_young = 0
    diag_amb_el_deg = []  # elevation [deg] of vsat=1 amb sats
    for (s, f), k in sorted_amb_items(amb_dict):
        if estimate.exists(k):
            tc.nav.x[tc.IB(s, f, tc.nav.na)] = estimate.atDouble(k)
            # Exclude ambiguities (re)seeded THIS epoch. Making the
            # wait span real epochs was measured worse; one epoch is the spec.
            if (s, f) not in tc.current_epoch.seeded_amb_keys:
                tc.nav.vsat[s - 1, f] = 1
                diag_vsat1 += 1
                el_idx = int(s) - 1
                if 0 <= el_idx < tc.nav.el.shape[0]:
                    diag_amb_el_deg.append(float(np.degrees(tc.nav.el[el_idx])))
            else:
                tc.nav.vsat[s - 1, f] = 0  # exclude from LAMBDA
                diag_vsat0_young += 1
        else:
            diag_estimate_missing += 1
    tc.ar_diag.amb_el_min_deg = (int(round(min(diag_amb_el_deg)))
                                 if diag_amb_el_deg else -1)
    tc.ar_diag.amb_el_median_deg = (int(round(float(np.median(diag_amb_el_deg))))
                                    if diag_amb_el_deg else -1)
    tc.ar_diag.amb_el_above15 = sum(1 for e in diag_amb_el_deg if e >= 15)
    tc.ar_diag.amb_el_above25 = sum(1 for e in diag_amb_el_deg if e >= 25)
    tc.ar_diag.amb_estimate_missing = diag_estimate_missing
    tc.ar_diag.amb_vsat1 = diag_vsat1
    tc.ar_diag.amb_vsat0_young = diag_vsat0_young


def _publish_held_ambiguities(tc, cp_visible_sf, amb_dict):
    """Held integers enter nav.x at varholdamb variance; vsat only
    while the sat stays CP-visible. Also counts orphan CP signals
    (visible but neither held nor float)."""
    held_var = max(float(tc.cfg.varholdamb), 1e-6)
    for (s, f), held_value in tc._sat_states.held_items():
        tc.nav.x[tc.IB(s, f, tc.nav.na)] = float(held_value)
        tc.nav.P[tc.IB(s, f, tc.nav.na), tc.IB(s, f, tc.nav.na)] = held_var
        is_visible = (s, f) in cp_visible_sf
        tc.nav.vsat[s - 1, f] = (1 if is_visible else 0)

    held_sf = {(int(s), int(f)) for (s, f), _ in tc._sat_states.held_items()}
    amb_sf = {(int(s), int(f)) for (s, f) in amb_dict.keys()}
    orphan = [(s, f) for (s, f) in cp_visible_sf
              if (s, f) not in held_sf and (s, f) not in amb_sf]
    tc.ar_diag.orphan_cp_count = len(orphan)
    tc.ar_diag.amb_dict_size = len(amb_sf)
    tc.ar_diag.held_size = len(held_sf)
    tc.ar_diag.cp_visible_size = len(cp_visible_sf)
