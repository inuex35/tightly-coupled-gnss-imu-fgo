"""Subset AR: retry LAMBDA with the likeliest bad satellites removed.

This is the project's own fallback (distinct from the demo5 single-satellite
round-robin in :mod:`ar_retry`): rank satellites by post-fit residual, then
low elevation, and try dropping combinations of up to ``subset_ar_max_drop``
of them, keeping the best ratio. A dirty-satellite cap keeps it away from
hopeless epochs (a spread-out residual field means the pose, not one
satellite, is the problem).

The actual attempt is injected (``attempt(tc, sat, sat_exclude=...)``) so
this policy is indifferent to which resolver runs underneath.
"""

from itertools import combinations

from ..utils import sorted_amb_items


def ratio_from_last_lambda(tc):
    """Ratio from the stashed s0/s1 of the last LAMBDA call (0 when unformed)."""
    s0 = float(tc._last_s0)
    s1 = float(tc._last_s1)
    if s0 <= 0.0:
        return 0.0
    return s1 / s0


def _rank_subset_drop_sats(tc, sat, el, amb_dict):
    """Rank candidate satellites to drop for subset AR fallback."""
    seen = set()
    per_sat = tc._mres_signals.per_sat or {}
    sat_el = {}
    for i, s in enumerate(sat):
        sat_el[int(s)] = max(sat_el.get(int(s), 0.0), float(el[i]))
    rows = []
    for (s, _f), _k in sorted_amb_items(amb_dict):
        s = int(s)
        if s in seen:
            continue
        seen.add(s)
        rows.append((
            float(per_sat.get(s, 0.0)),
            -float(sat_el.get(s, 0.0)),
            s))
    rows.sort(reverse=True)
    max_candidates = max(0, int(tc.cfg.subset_ar_max_candidates))
    return [s for *_rest, s in rows[:max_candidates]]


def try_subset_ar(tc, sat, el, amb_dict, attempt):
    """Retry AR with one (or more) candidate bad satellites removed."""
    dirty_max = int(tc.cfg.subset_ar_max_dirty_sats)
    if dirty_max > 0:
        per_sat = tc._mres_signals.per_sat or {}
        dirty_thr = float(tc.cfg.subset_ar_dirty_sat_res_m)
        dirty_n = sum(1 for v in per_sat.values()
                      if float(v or 0.0) > dirty_thr)
        if dirty_n > dirty_max:
            tc._ar_subset_debug = {
                'candidates': 0, 'used': False,
                'skip_reason': 'dirty_gate', 'dirty_n': dirty_n,
            }
            return 0, None
    min_nb = max(1, int(tc.cfg.subset_ar_min_nb))
    max_drop = max(1, int(tc.cfg.subset_ar_max_drop))
    best = None
    candidates = _rank_subset_drop_sats(tc, sat, el, amb_dict)
    tc._ar_subset_debug = {
        'candidates': len(candidates),
        'max_drop': max_drop,
        'used': False,
    }
    for k in range(1, max_drop + 1):
        if k > len(candidates):
            break
        for drop_combo in combinations(candidates, k):
            drop_combo = tuple(int(x) for x in drop_combo)
            nb, xa = attempt(tc, sat, sat_exclude=drop_combo)
            ratio = ratio_from_last_lambda(tc)
            if nb < min_nb or xa is None:
                continue
            # Quantize the ratio so ULP noise cannot outrank the
            # deliberate smaller-k tie-break: a no-op exclusion perturbs
            # the summation order enough to shift the ratio in its last
            # digits and prefer a 2-sat drop over the equal 1-sat drop.
            score = (round(float(ratio), 6), int(nb), -k)
            if best is None or score > best['score']:
                best = {
                    'drop_sats': drop_combo,
                    'nb': int(nb),
                    'xa': xa.copy() if hasattr(xa, 'copy') else xa,
                    'ratio': float(ratio),
                    'score': score,
                }
    if best is None:
        return 0, None
    nb, xa = attempt(tc, sat, sat_exclude=best['drop_sats'],
                     restore_state=False)
    if nb < min_nb or xa is None:
        tc._ar_subset_debug = {
            'candidates': len(candidates),
            'max_drop': max_drop,
            'used': False,
        }
        return 0, None
    tc._ar_subset_debug = {
        'candidates': len(candidates),
        'max_drop': max_drop,
        'used': True,
        'drop_sats': list(best['drop_sats']),
        'nb': int(nb),
        'ratio': ratio_from_last_lambda(tc),
    }
    return nb, xa


