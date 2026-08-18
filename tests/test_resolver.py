"""AmbiguityResolver: synthetic problems with known integer answers."""
import numpy as np
from gnss_fgo.ar.ambiguity_resolver import AmbiguityResolver

KEYS = [(1, 0), (2, 0), (3, 0), (4, 0)]     # four GPS sats, one band
EL = {1: 1.2, 2: 0.9, 3: 0.7, 4: 0.5}
TRUTH = {(1, 0): 100.0, (2, 0): -250.0, (3, 0): 731.0, (4, 0): 12.0}


def test_clean_fix_recovers_integers():
    rng = np.random.default_rng(7)
    floats = {k: TRUTH[k] + rng.normal(0, 0.01) for k in KEYS}
    r = AmbiguityResolver(thresar=3.0).resolve(
        floats, np.eye(4) * 1e-4, KEYS, EL)
    assert r.nb > 0
    for ref, tgt in r.pairs:
        dd_fixed = r.fixed[ref] - r.fixed[tgt]
        assert abs(dd_fixed - round(TRUTH[ref] - TRUTH[tgt])) < 1e-9


def test_noisy_float_is_not_blindly_fixed():
    rng = np.random.default_rng(3)
    floats = {k: TRUTH[k] + rng.normal(0, 0.45) for k in KEYS}
    r = AmbiguityResolver(thresar=3.0).resolve(
        floats, np.eye(4) * 0.45 ** 2, KEYS, EL)
    assert r.nb == 0 or r.ratio >= 3.0


def test_reference_is_shared_within_group():
    pairs = AmbiguityResolver().double_difference(KEYS, EL)
    assert len({ref for ref, _ in pairs}) == 1 and len(pairs) == 3


def test_elevation_mask_drops_low_sats():
    pairs = AmbiguityResolver(el_mask=0.6).double_difference(KEYS, EL)
    assert 4 not in {s for pair in pairs for (s, _) in pair}
