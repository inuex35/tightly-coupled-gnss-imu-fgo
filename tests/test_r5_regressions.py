"""Regression tests: varerr scale and N-symbol decode."""
import math
import types

import gtsam
import pytest

from gnss_fgo.factors.factors import varerr_dd_sigma


def _cfg():
    return types.SimpleNamespace(
        err_eratio_pr=100.0, err_a=0.001, amb_rw_sigma=0.001)


def test_varerr_zenith_values():
    tc = types.SimpleNamespace(cfg=_cfg())
    sig_pr = varerr_dd_sigma(tc, code=1, el_rad=math.pi / 2, dt_s=0.2)
    sig_cp = varerr_dd_sigma(tc, code=0, el_rad=math.pi / 2, dt_s=0.2)
    assert sig_pr == pytest.approx(0.2828, abs=2e-3)
    assert sig_cp == pytest.approx(0.002860, abs=2e-5)
    sig_pr_low = varerr_dd_sigma(tc, code=1, el_rad=math.radians(10), dt_s=0.2)
    assert sig_pr_low / sig_pr == pytest.approx(
        math.sqrt((1 + 1 / math.sin(math.radians(10)) ** 2) / 2), rel=1e-3)


def test_n_symbol_roundtrip():
    for s, f, gen in [(1, 0, 0), (63, 2, 7), (255, 9, 1_500_007)]:
        key = gtsam.symbol('n', gen * 1_000_000 + s * 10 + f)
        sym = gtsam.Symbol(key)
        assert sym.chr() == ord('n')
        idx = sym.index() % 100000
        assert (idx // 10, idx % 10) == (s, f)

