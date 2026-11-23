"""Pure diagnostic post-fit residual evaluation (normalized whitened z-scores)."""

import numpy as np


def main_ddpr_residuals(tc, graph, estimate, with_pairs=False):
    """Compute dimensionless normalized DDPR residuals sqrt(2 * factor.error(est))."""
    res_sq = []
    per_sat = {}
    pair_rows = [] if with_pairs else None
    tag_map = {idx: (ref, j, f)
               for (idx, ref, j, f) in tc._last_ddpr_sat_tags}
    g_size = graph.size()
    if tag_map and max(tag_map.keys()) < g_size:
        iter_indices = sorted(tag_map.keys())
        check_type = False
    else:
        iter_indices = range(g_size)
        check_type = True

    for i in iter_indices:
        fac = graph.at(i)
        if fac is None:
            continue
        if check_type and 'Pseudorange' not in type(fac).__name__:
            continue
        try:
            err = float(fac.error(estimate))
        except RuntimeError:
            continue
        z = float(np.sqrt(2.0 * max(err, 0.0)))
        res_sq.append(z * z)
        tag = tag_map.get(i)
        if tag is not None:
            ref, j, freq = tag
            if z > per_sat.get(ref, 0.0):
                per_sat[ref] = z
            if z > per_sat.get(j, 0.0):
                per_sat[j] = z
            if with_pairs:
                pair_rows.append({
                    'ref': int(ref),
                    'sat': int(j),
                    'freq': int(freq),
                    'res': z,
                })
    rms_all = float(np.sqrt(np.mean(res_sq))) if res_sq else 0.0
    if with_pairs:
        return rms_all, per_sat, pair_rows
    return rms_all, per_sat
