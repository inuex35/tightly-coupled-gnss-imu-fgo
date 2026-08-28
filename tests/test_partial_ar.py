"""Partial AR: a full-set decline with a clean core still fixes."""
import numpy as np

from gnss_fgo.ar.ambiguity_resolver import AmbiguityResolver


def _problem(n_clean=6, n_noisy=3, seed=3):
    rng = np.random.default_rng(seed)
    n = n_clean + n_noisy
    keys = [(s + 1, 0) for s in range(n + 1)]  # ref + n targets
    truth = rng.integers(-20, 20, size=n + 1).astype(float)
    var = np.full(n + 1, 1e-4)
    if n_noisy:
        var[-n_noisy:] = 4.0  # cycles^2: hopeless components
    cov = np.diag(var)
    vals = {k: truth[i] + rng.normal(0, np.sqrt(var[i]) * 0.3)
            for i, k in enumerate(keys)}
    el = {s + 1: np.radians(60.0) for s in range(n + 1)}
    return keys, vals, cov, el, truth


def test_partial_fix_accepts_the_clean_core():
    keys, vals, cov, el, truth = _problem()
    r = AmbiguityResolver(thresar=3.0, thresar_min=1.5, thresar_max=3.0)
    r.min_fix = 4  # synthetic problem is small
    res = r.resolve(vals, cov, keys, el, allow_partial=True)
    assert res.nb > 0
    assert res.dropped > 0
    # the clean targets come back at the true integer offsets
    ref = keys[0]
    for i, k in enumerate(keys[1:7], start=1):
        dd = res.fixed[ref] - res.fixed[k]
        assert abs(dd - (truth[0] - truth[i])) < 0.35


def test_full_fix_reports_dropped_zero():
    keys, vals, cov, el, _ = _problem(n_clean=8, n_noisy=0)
    r = AmbiguityResolver(thresar=3.0)
    res = r.resolve(vals, cov, keys, el)
    assert res.nb == len(res.pairs) and res.dropped == 0
