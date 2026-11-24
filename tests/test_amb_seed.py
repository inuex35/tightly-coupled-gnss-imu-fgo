"""Clock-free persistent-arc ambiguity seeding tests."""
import gtsam
import pytest

from gnss_fgo import factors as amb_seed


class _Tc:
    epoch = 100

    def __init__(self):
        from gnss_fgo.state import CurrentEpochState
        self.current_epoch = CurrentEpochState()

    @staticmethod
    def _noise1(sigma):
        return gtsam.noiseModel.Isotropic.Sigma(1, float(sigma))


LAM = 0.1903


def _seed(tc, cp_r, cp_b, pr_r, pr_b, existing_estimate=None):
    graph, values, amb, new = gtsam.NonlinearFactorGraph(), gtsam.Values(), {}, {}
    amb_seed.init_dd_ambiguity_priors(
        tc, graph, values, amb, new, existing_estimate, 0, LAM,
        ((7, gtsam.symbol('n', 7), cp_r, cp_b, pr_r, pr_b),))
    return graph, values, new


def test_seed_is_clock_free():
    """Adding a receiver clock term to BOTH phase and code must not move N."""
    tc = _Tc()
    _, v0, _ = _seed(tc, cp_r=1000.0, cp_b=400.0, pr_r=990.0, pr_b=395.0)
    base = v0.atDouble(gtsam.symbol('n', 7))
    for clk_m in (150.0, -3000.0, 45678.9):
        tc2 = _Tc()
        _, v2, _ = _seed(tc2, cp_r=1000.0 + clk_m, cp_b=400.0,
                         pr_r=990.0 + clk_m, pr_b=395.0)
        assert v2.atDouble(gtsam.symbol('n', 7)) == pytest.approx(base, abs=1e-9)


def test_existing_smoother_key_is_reused_without_reprior():
    """A persistent arc key already in the smoother must not be re-inserted or re-priored."""
    tc = _Tc()
    existing = gtsam.Values()
    existing.insert(gtsam.symbol('n', 7), 55.25)
    graph, values, new = _seed(
        tc, cp_r=1000.0, cp_b=400.0, pr_r=990.0, pr_b=395.0,
        existing_estimate=existing)
    assert graph.size() == 0
    assert not values.exists(gtsam.symbol('n', 7))
    assert new[(7, 0)] == gtsam.symbol('n', 7)
    assert (7, 0) not in tc.current_epoch.seeded_amb_keys
