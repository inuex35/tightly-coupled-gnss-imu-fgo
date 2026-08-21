"""Branch table for the Stage-D FIX/FLT decision (review round 4, A-4).

Pure-function test through stubs: no gtsam objects are exercised, only
the decision logic in _decide_fix_or_flt.
"""
import numpy as np
import types

from gnss_fgo.pipeline.validate_fix import _decide_fix_or_flt


class _SatStates(dict):
    def values(self):
        return [types.SimpleNamespace(fix_streak=v) for v in super().values()]


def _tc(smode, streaks=(), hard_max=1.0, low_nb_max=6, only_after_flt=1):
    cfg = types.SimpleNamespace(
        lambda_corr_hard_max=hard_max,
        low_nb_fix_reject_nb_max=low_nb_max,
        low_nb_fix_only_after_flt=only_after_flt,
        low_nb_fix_reject_max_prev_fix_streak=2,
    )
    tc = types.SimpleNamespace(cfg=cfg, nav=types.SimpleNamespace(smode=smode))
    tc._sat_states = _SatStates({i: s for i, s in enumerate(streaks)})
    tc._antenna_ecef = lambda pose, ecef: np.zeros(3)
    return tc


def _epoch(nb, xa, prev_smode=4, main_res=0.0):
    return types.SimpleNamespace(
        nb=nb, xa=xa, pose_tc=None, ecef_tc=np.zeros(3),
        info={'prev_smode': prev_smode, 'main_ddpr_res': main_res})


def test_float_mode_passes_through():
    ep = _epoch(nb=0, xa=None, prev_smode=5)
    sol, tag, nb = _decide_fix_or_flt(_tc(smode=5), ep)
    assert tag == 'FLT'


def test_hard_lambda_corr_rejects():
    ep = _epoch(nb=10, xa=np.array([5.0, 0, 0]))
    sol, tag, nb = _decide_fix_or_flt(_tc(smode=4, streaks=(9, 9)), ep)
    assert (tag, nb) == ('FLT', 0)
    assert 'lambda_corr_hard_reject' in ep.info


def test_low_nb_fresh_fix_rejected():
    ep = _epoch(nb=4, xa=np.array([0.01, 0, 0]), prev_smode=5)
    sol, tag, nb = _decide_fix_or_flt(_tc(smode=4, streaks=(0,)), ep)
    assert (tag, nb) == ('FLT', 0)
    assert ep.info.get('low_nb_fix_reject') is True
    assert ep.info['low_nb_fix_reject_nb'] == 4


def test_low_nb_established_fix_survives():
    # same nb, but the previous epoch was FIX with a long streak
    ep = _epoch(nb=4, xa=np.array([0.01, 0, 0]), prev_smode=4)
    sol, tag, nb = _decide_fix_or_flt(_tc(smode=4, streaks=(9, 9)), ep)
    assert (tag, nb) == ('FIX', 4)


def test_clean_fix_accepted():
    ep = _epoch(nb=12, xa=np.array([0.01, 0, 0]))
    sol, tag, nb = _decide_fix_or_flt(_tc(smode=4, streaks=(9,)), ep)
    assert (tag, nb) == ('FIX', 12)
