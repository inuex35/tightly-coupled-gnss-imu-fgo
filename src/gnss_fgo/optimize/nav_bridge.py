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

``publish_marginals`` (the historical ``write_marginals``) stays in
``ar.py`` for now -- it is scheduled to move here once its diagnostics
side-band is untangled -- but the native path's writes all live below.
"""

import numpy as np


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
