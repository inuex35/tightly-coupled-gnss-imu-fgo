"""GNSS elevation-dependent error model and reference-satellite selection."""

import numpy as np

from cssrlib.gnss import uGNSS

from ..utils import is_bds_geo as _utils_is_bds_geo


_CLIGHT = 299792458.0
_SCLKSTAB = 5e-12  # standard receiver clock stability [s/s] (RTKLIB default)


def varerr_dd_sigma(tc, code, el_rad, dt_s):
    """RTKLIB varerr formula returning DD σ in metres."""
    cfg = tc.cfg
    fact = cfg.err_eratio_pr if code else 1.0
    a = fact * cfg.err_a
    b = fact * cfg.err_b
    d = _CLIGHT * _SCLKSTAB * dt_s
    sinel = max(np.sin(el_rad), 0.05)
    var_sd = 2.0 * (a * a + b * b / (sinel * sinel)) + d * d
    return float(np.sqrt(2.0 * var_sd))


def pick_ref_sat_idx(tc, sys_id, idx_sys, sat, el):
    """Select the highest-elevation non-BDS-GEO reference satellite."""
    prev_ref = tc.current_epoch.ref_sats.get(sys_id)
    sats_in_sys = [sat[i] for i in idx_sys]
    prev_is_geo = (sys_id == uGNSS.BDS and prev_ref is not None
                   and _utils_is_bds_geo(prev_ref))
    if prev_ref is not None and prev_ref in sats_in_sys and not prev_is_geo:
        return idx_sys[sats_in_sys.index(prev_ref)], prev_ref

    pool = [i for i in idx_sys
            if not (sys_id == uGNSS.BDS and _utils_is_bds_geo(sat[i]))]
    if not pool:
        pool = idx_sys
    pool_arr = np.array(pool)
    ref_idx = int(pool_arr[np.argmax(el[pool_arr])])
    return ref_idx, sat[ref_idx]
