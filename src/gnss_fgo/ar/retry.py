"""RTKLIB demo5 retry policy over any single-shot resolver.

A transliteration of cssrlib's ``resamb_lambda_rtklib`` with the resolution
step abstracted out: pass 1 over the full set, and -- only when it fails
with enough satellites left -- one retry with a single satellite excluded.
The exclusion is chosen round-robin from ``nav.excsat``, except that
``arfilter`` prefers a freshly-acquired satellite (``lock == 1``) when the
ratio just dropped below threshold relative to ``prev_ratio2``.

All persistent state (lock counters, cursor, previous ratios) lives on
``nav`` through :mod:`nav_bridge` -- next epoch's decisions read it there,
and keeping any of it on private attributes instead measurably changed
which exclusions later epochs chose.
"""

from . import nav_bridge


def run(tc, sat_list, solve):
    """Apply the retry policy around ``solve(tc, sat_list) -> (nb, xa)|None``.

    Mirrors ``resamb_lambda_rtklib`` line for line, including the order of
    every ``nav`` write. Returns ``None`` when ``solve`` declines (the caller
    falls back to the cssrlib path), else ``(nb, xa)``.
    """
    nav_bridge.update_lock_counters(tc, sat_list)

    out = solve(tc, sat_list)
    if out is None:
        return None
    nb, xa = out
    ratio = 0.0 if tc._last_s0 <= 0.0 else tc._last_s1 / tc._last_s0
    if nb > 0:
        nav_bridge.publish_retry_success(tc, ratio)
        return nb, xa

    if len(sat_list) < tc.nav.minfixsats:
        return 0, xa

    sat_arr = [int(s) for s in sat_list]
    try:
        start = sat_arr.index(tc.nav.excsat) + 1
    except ValueError:
        start = 0
    order = sat_arr[start:] + sat_arr[:start]

    exc = 0
    if (tc.nav.arfilter and ratio < tc.nav.thresar
            and tc.nav.prev_ratio2 > 0.0
            and ratio < 1.1 * tc.nav.prev_ratio2):
        for s_ in order:
            if any(0 < tc.nav.lock[s_ - 1, f] <= 1
                   for f in range(tc.nav.nf)):
                exc = s_
                break
    if exc == 0:
        for s_ in order:
            if any(tc.nav.vsat[s_ - 1, f] != 0
                   for f in range(tc.nav.nf)):
                exc = s_
                break
    if exc == 0:
        return 0, xa

    # Exclude by zeroing vsat for one epoch; selection reads vsat.
    vsat_row = tc.nav.vsat[exc - 1, :].copy()
    tc.nav.vsat[exc - 1, :] = 0
    try:
        out2 = solve(tc, [s for s in sat_list if s != exc])
    finally:
        tc.nav.vsat[exc - 1, :] = vsat_row
    if out2 is None:
        return 0, xa
    nb2, xa2 = out2

    ratio2 = 0.0 if tc._last_s0 <= 0.0 else tc._last_s1 / tc._last_s0
    nav_bridge.publish_retry_outcome(tc, nb2 > 0, ratio2, exc)
    return (nb2, xa2) if nb2 > 0 else (0, xa)
