"""Pre-fit gates and reference-satellite pick (Stage B/C support).

Operates between input preprocessing and graph optimization to keep
multipath-contaminated satellites out of the LAMBDA tree and to
expose Doppler velocity as a coarse pre-check signal.

Future hooks: Mahalanobis χ² per-sat gate, two-stage
Doppler×IMU velocity outlier test (see refactor_plan.md).
"""

import numpy as np

from cssrlib.gnss import uGNSS

from ..utils import is_bds_geo as _utils_is_bds_geo


_CLIGHT = 299792458.0


def varerr_dd_sigma(tc, code, el_rad, dt_s):
    """RTKLIB-demo5 varerr formula port (rtkpos.c:402), returning the
    σ in metres to feed the DD factor's noise model.

    SD-level variance:

        var_SD = 2·(a² + b²/sin²el) + (CLIGHT·sclkstab·dt)²
        a = fact · err_a   b = fact · err_b
        fact = err_eratio_pr (PR) | 1 (CP)

    DD residual is the difference of two SDs (rover-base for ref and
    target sat), so var_DD = 2 · var_SD and σ_DD = √(2 · var_SD).
    """
    cfg = tc.cfg
    fact = cfg.err_eratio_pr if code else 1.0
    a = fact * cfg.err_a
    b = fact * cfg.err_b
    d = _CLIGHT * cfg.err_sclkstab * dt_s
    sinel = max(np.sin(el_rad), 0.05)  # cap below 3° to avoid blow-up
    var_sd = 2.0 * (a*a + b*b / (sinel*sinel)) + d*d
    return float(np.sqrt(2.0 * var_sd))



def pick_ref_sat_idx(tc, sys_id, idx_sys, sat, el):
    """Select DD reference satellite index from idx_sys.

    The reference ledger lives on the epoch scratch (cross-epoch
    continuity was measured worse — A-2 A/B: AllRMS 21.35 -> 93.03),
    so the prev-ref preference works WITHIN an epoch only: the main
    build writes it, and later same-epoch DD solves (LS
    fallback, sanity anchor, FDE re-solve) pick the same reference so
    their residuals stay comparable. First pick each epoch is
    highest-elevation, excluding BeiDou GEO and sats whose prev-epoch
    DDPR residual exceeded per_sat_res_thresh (likely multipath).
    Returns (ref_idx, ref_sat).
    """
    prev_ref = tc.current_epoch.ref_sats.get(sys_id)
    sats_in_sys = [sat[i] for i in idx_sys]
    prev_is_geo = (sys_id == uGNSS.BDS and prev_ref is not None
                   and _utils_is_bds_geo(prev_ref))
    # Reject prev_ref if it was flagged as multipath-contaminated last epoch
    prev_res = tc._last_per_sat_res.get(prev_ref, 0.0) \
        if prev_ref is not None else 0.0
    prev_multipath = prev_res > tc.cfg.per_sat_res_thresh
    if (prev_ref is not None and prev_ref in sats_in_sys
            and not prev_is_geo and not prev_multipath):
        ref_idx = idx_sys[sats_in_sys.index(prev_ref)]
        return ref_idx, prev_ref
    # Fresh pick: highest-elevation, excluding BDS GEO and high-residual sats
    last_res = tc._last_per_sat_res
    thresh = tc.cfg.per_sat_res_thresh
    def is_ref_candidate(i):
        s = sat[i]
        if sys_id == uGNSS.BDS and _utils_is_bds_geo(s):
            return False
        if last_res.get(s, 0.0) > thresh:
            return False
        return True
    pool = [i for i in idx_sys if is_ref_candidate(i)]
    if not pool:
        # Fallback: drop multipath filter (keep only GEO filter)
        pool = [i for i in idx_sys
                if not (sys_id == uGNSS.BDS and _utils_is_bds_geo(sat[i]))]
    if not pool:
        pool = idx_sys
    pool_arr = np.array(pool)
    ref_idx = int(pool_arr[np.argmax(el[pool_arr])])
    return ref_idx, sat[ref_idx]


