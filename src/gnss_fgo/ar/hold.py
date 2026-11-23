"""In-graph DD integer constraints (ar_mode == 3) via pure C++ BetweenFactorDouble."""

import math
import gtsam

_HOLD_SIGMA = math.sqrt(0.001)  # RTKLIB standard varholdamb = 0.001 cyc²


def apply_fix_and_hold(tc, amb_dict, xa):
    """Add once-per-arc C++ DD integer BetweenFactorDouble(k_tgt, k_ref, z_int) with Huber."""
    res = getattr(tc, '_last_ar_result', None)
    if res is None or res.dropped or not res.pairs:
        return False

    smoother = tc.isam2 if tc.phase == 2 else tc.isam
    if smoother is None:
        return False

    base_noise = tc._noise1(_HOLD_SIGMA)
    # Quadratic within ±0.5 cycles (snaps normal float ambiguities in one step),
    # Huber-bounded beyond 0.5 cycles (caps wrong-integer conflicts).
    hold_noise = gtsam.noiseModel.Robust.Create(
        gtsam.noiseModel.mEstimator.Huber.Create(0.5 / _HOLD_SIGMA), base_noise)

    hg = gtsam.NonlinearFactorGraph()
    ts = gtsam.FixedLagSmootherKeyTimestampMap()
    t_now = tc.tc_time if tc.phase == 2 else tc.phase1_t
    held_pairs = tc._held_dd_pairs

    for ref_sf, tgt_sf in res.pairs:
        k_ref = amb_dict.get(ref_sf)
        k_tgt = amb_dict.get(tgt_sf)
        if k_ref is None or k_tgt is None:
            continue
        pair_key = (int(k_tgt), int(k_ref))
        if pair_key in held_pairs:
            continue
        v_ref = res.fixed.get(ref_sf)
        v_tgt = res.fixed.get(tgt_sf)
        if v_ref is None or v_tgt is None:
            continue
        z_dd = float(round(v_ref - v_tgt))
        hg.add(gtsam.BetweenFactorDouble(k_tgt, k_ref, z_dd, hold_noise))
        ts[k_tgt] = t_now
        ts[k_ref] = t_now
        held_pairs.add(pair_key)

    if hg.size() == 0:
        return False
    try:
        smoother.update(hg, gtsam.Values(), ts)
        return True
    except (RuntimeError, IndexError) as ex:
        tc.ar_diag.exception = f'dd_hold: {type(ex).__name__}: {ex}'
        return False
