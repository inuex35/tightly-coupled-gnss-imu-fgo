"""Round-5 regression pins: varerr scale, N-symbol decode, hold FSM,
and the fix_streak-counts-held-fixes repair (#1)."""
import math
import types

import numpy as np
import gtsam
import pytest

from gnss_fgo.factors.prefit import varerr_dd_sigma
from gnss_fgo.state.runtime_state import SatState, SatStateMap
from gnss_fgo.ar.hold import _activate_phase2_hold_states


def _cfg():
    return types.SimpleNamespace(
        err_eratio_pr=100.0, err_a=0.001, err_b=0.001, err_sclkstab=5e-12)


def test_varerr_zenith_values():
    """Anchor for the r5 #4 unit discussion: the zenith DD sigmas under
    the default error model. PR: 2*eratio*err_a = 0.283 m (dt term is
    negligible at 0.2 s); CP: 0.00283 m."""
    tc = types.SimpleNamespace(cfg=_cfg())
    sig_pr = varerr_dd_sigma(tc, code=1, el_rad=math.pi / 2, dt_s=0.2)
    sig_cp = varerr_dd_sigma(tc, code=0, el_rad=math.pi / 2, dt_s=0.2)
    assert sig_pr == pytest.approx(0.2828, abs=2e-3)
    assert sig_cp == pytest.approx(0.002860, abs=2e-5)   # clock-stability term is visible at CP scale
    # 10-deg elevation inflates by b/sin(el): ~4.1x at these defaults
    sig_pr_low = varerr_dd_sigma(tc, code=1, el_rad=math.radians(10), dt_s=0.2)
    assert sig_pr_low / sig_pr == pytest.approx(
        math.sqrt((1 + 1 / math.sin(math.radians(10)) ** 2) / 2), rel=1e-3)


def test_n_symbol_roundtrip():
    """N(s, f, gen) encodes gen*1e6 + s*10 + f; the FDE decode takes
    idx % 100000 -> (s, f). Must survive large gens (dd_epoch*100+g)."""
    for s, f, gen in [(1, 0, 0), (63, 2, 7), (255, 9, 1_500_007)]:
        key = gtsam.symbol('n', gen * 1_000_000 + s * 10 + f)
        sym = gtsam.Symbol(key)
        assert sym.chr() == ord('n')
        idx = sym.index() % 100000
        assert (idx // 10, idx % 10) == (s, f)


def test_hold_state_machine():
    st = SatState()
    st.activate_hold(42.0)
    assert (st.held_value, st.last_held_value, st.release_seed_pending) \
        == (42.0, 42.0, False)
    st.release_hold(seed=True)
    assert st.held_value is None
    assert st.last_held_value == 42.0
    assert st.release_seed_pending is True
    st.clear_hold()
    assert (st.held_value, st.last_held_value, st.release_seed_pending) \
        == (None, None, False)


def test_hold_activation_counts_fix_streak():
    """r5 #1: prev_fix_streak_max was 0 forever because the streak was
    counted only for (s,f) still holding amb_key with fix==3, but hold
    activation clears amb_key first. Pin the repair."""
    m = SatStateMap()
    m.get(7, 0).amb_key = 123
    tc = types.SimpleNamespace(
        _sat_states=m,
        nav=types.SimpleNamespace(na=3, x=np.zeros(64)),
        IB=lambda s, f, na: 3 + s,
    )
    _activate_phase2_hold_states(tc, [(7, 0)], xa=np.arange(64.0))
    st = m.get(7, 0)
    assert st.amb_key is None and st.held_value is not None
    assert st.fix_streak == 1
    assert max((s.fix_streak for s in m.values()), default=0) > 0
