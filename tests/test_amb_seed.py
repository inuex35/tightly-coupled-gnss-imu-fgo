"""Ambiguity seeding: the clock-free property that run1 ep1619 proved vital."""
import numpy as np
import gtsam
import pytest

from gnss_fgo.ar.ambiguity_resolver import AmbiguityResolver  # noqa: F401 (import check)
from gnss_fgo.buildfactor import amb_seed


class _Cfg:
    sigma_cont = 1.0
    sigma_amb0 = 30.0
    hold_gauge_gate_m = 0.0


class _SatState:
    def __init__(self):
        self.amb_lam = 0.0
        self.held_value = None
        self.last_held_value = None
        self.release_seed_pending = False
        self.amb_init_epoch = None

    def clear_hold(self):
        self.held_value = None
        self.last_held_value = None
        self.release_seed_pending = False


class _SatMap(dict):
    def get(self, sat, freq):
        return self.setdefault((sat, freq), _SatState())


class _Tc:
    phase = 2
    epoch = 100

    def __init__(self):
        self.cfg = _Cfg()
        self._sat_states = _SatMap()
        self._last_hold_gauge_rel = []

    @staticmethod
    def _noise1(sigma):
        return gtsam.noiseModel.Isotropic.Sigma(1, float(sigma))


LAM = 0.1903


def _seed(tc, cp_r, cp_b, pr_r, pr_b, prev=None):
    graph, values, amb, new = gtsam.NonlinearFactorGraph(), gtsam.Values(), {}, {}
    amb_seed.init_dd_ambiguity_priors(
        tc, graph, values, amb, new, prev, 0, LAM,
        ((7, gtsam.symbol('n', 7), cp_r, cp_b, pr_r, pr_b),))
    return values.atDouble(gtsam.symbol('n', 7))


def test_seed_is_clock_free():
    """Adding a receiver clock term to BOTH phase and code must not move N."""
    tc = _Tc()
    base = _seed(tc, cp_r=1000.0, cp_b=400.0, pr_r=990.0, pr_b=395.0)
    for clk_m in (150.0, -3000.0, 45678.9):
        tc2 = _Tc()
        shifted = _seed(tc2, cp_r=1000.0 + clk_m, cp_b=400.0,
                        pr_r=990.0 + clk_m, pr_b=395.0)
        assert shifted == pytest.approx(base, abs=1e-9)


def test_held_sat_is_skipped():
    tc = _Tc()
    st = tc._sat_states.get(7, 0)
    st.held_value = -1234.0
    graph, values, amb, new = gtsam.NonlinearFactorGraph(), gtsam.Values(), {}, {}
    amb_seed.init_dd_ambiguity_priors(
        tc, graph, values, amb, new, None, 0, LAM,
        ((7, gtsam.symbol('n', 7), 1000.0, 400.0, 990.0, 395.0),))
    assert not values.exists(gtsam.symbol('n', 7))
    assert st.held_value == -1234.0


def test_gauge_gate_releases_distant_hold():
    tc = _Tc()
    tc.cfg.hold_gauge_gate_m = 300.0
    st = tc._sat_states.get(7, 0)
    st.held_value = 1e6          # absurdly far from the fresh seed
    n0 = _seed(tc, cp_r=1000.0, cp_b=400.0, pr_r=990.0, pr_b=395.0)
    assert st.held_value is None
    assert n0 == pytest.approx(((1000.0 - 400.0) - (990.0 - 395.0)) / LAM)
    assert tc._last_hold_gauge_rel and tc._last_hold_gauge_rel[0][0] == 7


def test_release_reseeds_at_held_integer():
    tc = _Tc()
    st = tc._sat_states.get(7, 0)
    st.release_seed_pending = True
    st.last_held_value = -777.0
    n0 = _seed(tc, cp_r=1000.0, cp_b=400.0, pr_r=990.0, pr_b=395.0)
    assert n0 == pytest.approx(-777.0)
    assert st.release_seed_pending is False


def test_continuing_prior_reuses_previous_value():
    tc = _Tc()
    prev = {(7, 0): (None, 55.25)}
    n0 = _seed(tc, cp_r=1000.0, cp_b=400.0, pr_r=990.0, pr_b=395.0, prev=prev)
    assert n0 == pytest.approx(55.25)
