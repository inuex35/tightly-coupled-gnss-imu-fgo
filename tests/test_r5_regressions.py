"""Regression tests: varerr scale, N-symbol decode, and C++ DD BetweenFactorDouble hold."""
import math
import types

import numpy as np
import gtsam
import pytest

from gnss_fgo.factors.prefit import varerr_dd_sigma
from gnss_fgo.ar.hold import apply_fix_and_hold


def _cfg():
    return types.SimpleNamespace(
        err_eratio_pr=100.0, err_a=0.001, err_b=0.001, err_sclkstab=5e-12,
        varholdamb=0.001)


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


def test_dd_hold_enforces_ref_minus_tgt_integer():
    """BetweenFactorDouble(k_tgt, k_ref, z_dd) must enforce (N_ref - N_tgt) == z_dd."""
    k_ref = gtsam.symbol('n', 10)
    k_tgt = gtsam.symbol('n', 20)
    smoother = gtsam.IncrementalFixedLagSmoother(1.0, gtsam.ISAM2Params())
    g = gtsam.NonlinearFactorGraph()
    v = gtsam.Values()
    v.insert(k_ref, 100.3)
    v.insert(k_tgt, 87.8)
    g.addPriorDouble(k_ref, 100.3, gtsam.noiseModel.Isotropic.Sigma(1, 5.0))
    g.addPriorDouble(k_tgt, 87.8, gtsam.noiseModel.Isotropic.Sigma(1, 5.0))
    ts = gtsam.FixedLagSmootherKeyTimestampMap()
    ts[k_ref] = 1.0
    ts[k_tgt] = 1.0
    smoother.update(g, v, ts)

    tc = types.SimpleNamespace(
        cfg=_cfg(),
        phase=2,
        isam2=smoother,
        tc_time=1.0,
        _held_dd_pairs=set(),
        _last_ar_result=types.SimpleNamespace(
            dropped=0,
            pairs=[((1, 0), (2, 0))],
            fixed={(1, 0): 100.0, (2, 0): 88.0},
        ),
        ar_diag=types.SimpleNamespace(exception=None),
        _noise1=lambda s: gtsam.noiseModel.Isotropic.Sigma(1, float(s)),
    )
    ok = apply_fix_and_hold(tc, {(1, 0): k_ref, (2, 0): k_tgt}, xa=None)
    assert ok is True
    est = smoother.calculateEstimate()
    assert (est.atDouble(k_ref) - est.atDouble(k_tgt)) == pytest.approx(12.0, abs=1e-2)
    # Second call on the same persistent keys is a no-op (no double-counting).
    assert apply_fix_and_hold(tc, {(1, 0): k_ref, (2, 0): k_tgt}, xa=None) is False
