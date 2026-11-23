"""Pure Stage-D FIX/FLT decision tests (no heuristic post-gates)."""
import types
import numpy as np

from gnss_fgo.pipeline.validate_fix import _decide_fix_or_flt


def _tc(smode):
    tc = types.SimpleNamespace(nav=types.SimpleNamespace(smode=smode))
    tc._antenna_ecef = lambda pose, ecef: np.zeros(3)
    return tc


def _epoch(nb, xa):
    return types.SimpleNamespace(
        nb=nb, xa=xa, pose_tc=None, ecef_tc=np.zeros(3), info={})


def test_float_mode_passes_through():
    ep = _epoch(nb=0, xa=None)
    sol, tag, nb = _decide_fix_or_flt(_tc(smode=5), ep)
    assert (tag, nb) == ('FLT', 0)


def test_clean_fix_accepted():
    ep = _epoch(nb=12, xa=np.array([0.01, 0.0, 0.0]))
    sol, tag, nb = _decide_fix_or_flt(_tc(smode=4), ep)
    assert (tag, nb) == ('FIX', 12)
    assert 'lambda_correction' in ep.info
