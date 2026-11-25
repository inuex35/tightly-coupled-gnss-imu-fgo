"""Regression tests: varerr scale and N-symbol decode."""
import math
import types

import gtsam
import numpy as np
import pytest

from gnss_fgo.factors import _make_sd_doppler_custom_factor, dd_obs_sigma


def _cfg():
    return types.SimpleNamespace(err_eratio_pr=100.0, err_a=0.001)


def test_varerr_zenith_values():
    tc = types.SimpleNamespace(cfg=_cfg())
    ref_snrs = (45.0, 45.0, 45.0, 45.0)
    sig_pr = dd_obs_sigma(tc, is_pseudorange=True, snrs=ref_snrs, dt_s=0.2)
    sig_cp = dd_obs_sigma(tc, is_pseudorange=False, snrs=ref_snrs, dt_s=0.2)
    assert sig_pr == pytest.approx(0.2828, abs=2e-3)
    assert sig_cp == pytest.approx(0.002860, abs=2e-5)
    low_snrs = (45.0, 45.0, 35.0, 45.0)
    sig_pr_low = dd_obs_sigma(tc, is_pseudorange=True, snrs=low_snrs, dt_s=0.2)
    assert sig_pr_low / sig_pr == pytest.approx(math.sqrt((3.0 + 10.0) / 4.0), rel=1e-3)


def test_n_symbol_roundtrip():
    for s, f, gen in [(1, 0, 0), (63, 2, 7), (255, 9, 1_500_007)]:
        key = gtsam.symbol('n', gen * 1_000_000 + s * 10 + f)
        sym = gtsam.Symbol(key)
        assert sym.chr() == ord('n')
        idx = sym.index() % 100000
        assert (idx // 10, idx % 10) == (s, f)


def test_sd_doppler_custom_matches_cpp():
    p_key, v_key = gtsam.symbol('x', 0), gtsam.symbol('v', 0)
    rec = np.array([-3950e3, 3350e3, 3700e3])
    ecef_T_nav = gtsam.Pose3(gtsam.Rot3.RzRyRx(0.4, 0.5, -0.1), rec)
    vals = gtsam.Values()
    vals.insert(p_key, gtsam.Pose3(gtsam.Rot3.RzRyRx(0.1, -0.2, 0.3), np.array([1.0, 2.0, 3.0])))
    vals.insert(v_key, np.array([5.0, -2.0, 0.4]))
    sat_pt, sat_vt = np.array([20000e3, 12000e3, 10000e3]), np.array([1500.0, -1200.0, 800.0])
    ref_pt, ref_vt = np.array([-15000e3, 18000e3, 12000e3]), np.array([-1100.0, 900.0, 1400.0])
    lever, omega = np.array([0.31, 0.0, 0.55]), np.array([0.05, -0.02, 0.3])
    noise = gtsam.noiseModel.Isotropic.Sigma(1, 0.1)

    f_py = _make_sd_doppler_custom_factor(
        p_key, v_key, 1234.5, -987.6, 0.1903, 0.1903,
        sat_pt, sat_vt, ref_pt, ref_vt, rec, lever, ecef_T_nav, omega, noise)
    sd_cpp = getattr(gtsam, 'SingleDifferenceDopplerFactorArm', None)
    if sd_cpp is not None:
        f_cpp = sd_cpp(
            p_key, v_key, 1234.5, -987.6, 0.1903, 0.1903,
            sat_pt, sat_vt, ref_pt, ref_vt, rec, lever, ecef_T_nav, omega, 0.0, 0.0, noise)
        assert f_py.error(vals) == pytest.approx(f_cpp.error(vals), abs=1e-12)
        g_py, g_cpp = f_py.linearize(vals).jacobian(), f_cpp.linearize(vals).jacobian()
        assert np.allclose(g_py[0], g_cpp[0], atol=1e-12)
        assert np.allclose(g_py[1], g_cpp[1], atol=1e-12)

