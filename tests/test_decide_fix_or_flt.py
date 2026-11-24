"""Pure Stage-D FIX/FLT decision tests (no heuristic post-gates)."""
import types
import numpy as np

from gnss_fgo.runner import _decide_fix_or_flt


def _tc(smode):
    return types.SimpleNamespace(nav=types.SimpleNamespace(smode=smode))


def test_float_mode_passes_through():
    info = {}
    sol, tag, nb = _decide_fix_or_flt(_tc(smode=5), np.zeros(3), 0, None, info)
    assert (tag, nb) == ('FLT', 0)


def test_clean_fix_accepted():
    info = {}
    sol, tag, nb = _decide_fix_or_flt(
        _tc(smode=4), np.zeros(3), 12, np.array([0.01, 0.0, 0.0]), info)
    assert (tag, nb) == ('FIX', 12)
    assert 'lambda_correction' in info
