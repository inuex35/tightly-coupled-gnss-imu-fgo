"""Every write the AR path makes to cssrlib's ``nav`` state, in one place.

The graph is the estimator; ``nav`` survives as the interface to cssrlib's
validation (``zdres``/``sdres``/``valpos``), to LAMBDA when the cssrlib path
runs it, and to this project's own gates. These writes *are* the contract
between the two worlds, and every one of them has been broken at least once
by being implicit -- the table says who reads each field, so the next change
knows what it is touching.

    field                    written by              read by
    -----                    ----------              -------
    nav.x[0:3]               stage/postfit           zdres/sdres/valpos, gates
    nav.x[na:]               publish_marginals       resamb_lambda, ddidx
    nav.P                    publish_marginals       resamb_lambda (Qb, Qab)
    nav.vsat                 publish_marginals       ddidx selection, retries
    nav.el                   qcedit (cssrlib)        ddidx mask, weights
    nav.fix                  ddidx (both paths)      restamb, hold policy
    nav.xa, nav.Pa           accepted fix (both)     hold policy, output
    nav.lock                 retry, every call       next epoch's arfilter
    nav.excsat               retry outcome           next epoch's round-robin
    nav.prev_ratio1/2        retry outcome           next epoch's arfilter
    tc._last_s0/_last_s1     every resolution        ratio gates, diagnostics

"""


import numpy as np
import gtsam

from ..utils import sorted_amb_items



def publish_attempt(tc, sat_list, result):
    """Side effects of one resolution attempt, accepted or not.

    ``nav.fix`` comes from ``ddidx`` itself rather than a re-implementation:
    its marks are quirky (the reference of a singleton group is flagged even
    though no difference is formed; below-mask satellites are flagged only
    when they precede the reference in PRN order) and reproducing them by
    hand is how the two paths once drifted apart.
    """
    tc.ddidx(tc.nav, sat_list)
    tc._last_s0, tc._last_s1 = result.s0, result.s1


def publish_fix(tc, xa, Qb, Qab):
    """Side effects of an accepted fix: ``nav.xa`` and ``nav.Pa``."""
    tc.nav.xa = tc.nav.x.copy()
    tc.nav.xa[0:tc.nav.na] = xa[0:tc.nav.na]
    na = tc.nav.na
    Pa = tc.nav.P[0:na, 0:na].copy()
    if Qab is not None:
        Qab_full = np.zeros((na, Qb.shape[0]))
        Qab_full[0:3, :] = tc.R_enu2ecef @ Qab
        try:
            Pa = Pa - Qab_full @ np.linalg.inv(Qb) @ Qab_full.T
        except np.linalg.LinAlgError:
            pass
    tc.nav.Pa = Pa


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
    """First-pass fix: both previous ratios follow it, the cursor resets."""
    tc.nav.prev_ratio1 = ratio
    tc.nav.prev_ratio2 = ratio
    tc.nav.excsat = 0


def publish_retry_first_failure(tc, ratio):
    """First pass failed: only prev_ratio1 tracks it."""
    tc.nav.prev_ratio1 = ratio


def publish_retry_outcome(tc, fixed, ratio, excluded_sat):
    """Second pass done: remember the exclusion only when it worked."""
    if fixed:
        tc.nav.prev_ratio2 = ratio
        tc.nav.excsat = excluded_sat
    else:
        tc.nav.excsat = 0


def publish_marginals(tc, factors, estimate, key_pose, amb_dict):
    """Write GTSAM Marginals to nav.P with ENU->ECEF rotation."""
    # The native resolver reads the same marginals straight from ISAM2 and
    # needs the very pose these were taken against, not tc.tc_epoch, which
    # still points at the previous epoch while AR runs.
    tc._ar_key_pose = key_pose
    R = tc.R_enu2ecef
    tc.nav.P[:, :] = 0
    tc.nav.vsat[:, :] = 0
    tc.nav.x[tc.nav.na:] = 0
    cp_visible_sf = set(tc._ar_cp_visible_sf)
    _publish_float_ambiguities(tc, estimate, amb_dict)
    _publish_held_ambiguities(tc, cp_visible_sf, amb_dict)
    _publish_covariances(tc, factors, estimate, key_pose, amb_dict, R)


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
            # Exclude ambiguities (re)seeded THIS epoch. amb_init_epoch
            # is cleared by the per-epoch scratch reset, so a non-None
            # value can only mean amb_seed wrote it this epoch. Making
            # the wait span real epochs was measured worse (A-1 A/B:
            # AllRMS 21.35 -> 21.66), so one epoch is the spec.
            seeded_now = (
                tc._sat_states.at(s, f).amb_init_epoch is not None)
            if not seeded_now:
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
    tc._last_amb_el_min_deg = (int(round(min(diag_amb_el_deg)))
                                 if diag_amb_el_deg else -1)
    tc._last_amb_el_median_deg = (int(round(float(np.median(diag_amb_el_deg))))
                                    if diag_amb_el_deg else -1)
    tc._last_amb_el_above15 = sum(1 for e in diag_amb_el_deg if e >= 15)
    tc._last_amb_el_above25 = sum(1 for e in diag_amb_el_deg if e >= 25)
    tc._last_amb_estimate_missing = diag_estimate_missing
    tc._last_amb_vsat1 = diag_vsat1
    tc._last_amb_vsat0_young = diag_vsat0_young


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
    tc._last_orphan_cp_count = len(orphan)
    tc._last_amb_dict_size = len(amb_sf)
    tc._last_held_size = len(held_sf)
    tc._last_cp_visible_size = len(cp_visible_sf)


def _publish_covariances(tc, factors, estimate, key_pose, amb_dict, R):
    """nav.P from the smoother's cached Bayes tree (pose block, N
    diagonals, pose-N and N-N cross terms), rotated ENU->ECEF."""
    # Pull marginals straight from the smoother's Bayes tree; constructing
    # a fresh gtsam.Marginals(factors, estimate) re-linearizes the entire
    # graph (including every Python CustomFactor) and dominates the AR
    # stage. ISAM2 already has the cached factorization. FLS exposes
    # marginalCovariance but joint marginals must come from getISAM2().
    smoother = tc.isam2
    isam2 = smoother.getISAM2() if smoother is not None else None
    try:
        if isam2 is not None:
            P_pose = isam2.marginalCovariance(key_pose)
        else:
            mg = gtsam.Marginals(factors, estimate)
            P_pose = mg.marginalCovariance(key_pose)
        tc.nav.P[0:3, 0:3] = R @ P_pose[3:6, 3:6] @ R.T

        active = [(s, f, k) for (s, f), k in sorted_amb_items(amb_dict)
                  if estimate.exists(k)
                  and tc.nav.vsat[s - 1, f] == 1]
        if active:
            keys = gtsam.KeyVector()
            keys.append(key_pose)
            for s, f, k in active:
                keys.append(k)
            jm = (isam2.jointMarginalCovariance(keys) if isam2 is not None
                  else mg.jointMarginalCovariance(keys))
            for s, f, k in active:
                idx = tc.IB(s, f, tc.nav.na)
                tc.nav.P[idx, idx] = jm.at(k, k)[0, 0]
                Pxn = jm.at(key_pose, k)
                pxn_ecef = R @ Pxn[3:6, 0]
                tc.nav.P[0:3, idx] = pxn_ecef
                tc.nav.P[idx, 0:3] = pxn_ecef
            for i, (s1, f1, k1) in enumerate(active):
                i1 = tc.IB(s1, f1, tc.nav.na)
                for j, (s2, f2, k2) in enumerate(active):
                    if i >= j:
                        continue
                    i2 = tc.IB(s2, f2, tc.nav.na)
                    c = jm.at(k1, k2)[0, 0]
                    tc.nav.P[i1, i2] = c
                    tc.nav.P[i2, i1] = c
    except (RuntimeError, IndexError):
        # IndexError: gtsam raises it when key_pose (or an amb key) is not
        # in the BayesTree yet, e.g. the epoch right after a warm reset.
        pass
    bad = ~np.isfinite(tc.nav.P)
    if bad.any():
        tc.nav.P[bad] = 0.0
        diag_idx = np.where(np.diag(bad))[0]
        if len(diag_idx):
            tc.nav.P[diag_idx, diag_idx] = 1e10
