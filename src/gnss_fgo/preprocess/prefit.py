"""Stage 2 — pre-fit gates.

Operates between input preprocessing and graph optimization to keep
multipath-contaminated satellites out of the LAMBDA tree and to
expose Doppler velocity as a coarse pre-check signal.

Future hooks: Mahalanobis χ² per-sat gate, two-stage
Doppler×IMU velocity outlier test (see refactor_plan.md).
"""

import os
import numpy as np

from cssrlib.gnss import uGNSS
from ..utils import get_wavelengths as _utils_get_wavelengths

from ..utils import doppler_velocity_ls as _utils_doppler_velocity_ls
from ..utils import is_bds_geo as _utils_is_bds_geo


_CLIGHT = 299792458.0


def varerr_dd_sigma(tc, code, freq, el_rad, dt_s):
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


def apply_per_sat_residual_gate(tc, info):
    """Zero nav.vsat[s,f] for sats whose main-graph DDPR residual
    exceeded per_sat_res_thresh — keeps multipath-biased satellites
    out of LAMBDA so the integer fix isn't pulled onto a wrong tree.
    """
    per_sat_cur = info.get('main_ddpr_per_sat', {})
    if not per_sat_cur:
        return
    thresh_sat = tc.cfg.per_sat_res_thresh
    n_drop = 0
    for s, rmax in per_sat_cur.items():
        if rmax > thresh_sat and 1 <= s <= tc.nav.vsat.shape[0]:
            for f in range(tc.nav.nf):
                if tc.nav.vsat[s - 1, f] == 1:
                    tc.nav.vsat[s - 1, f] = 0
                    n_drop += 1
    if n_drop:
        info['ar_vsat_drop'] = n_drop


def pick_ref_sat_idx(tc, sys_id, idx_sys, sat, el):
    """Select DD reference satellite index from idx_sys.

    Prefers the previously-locked reference (tc.ref_sats[sys_id]) for
    continuity; on first use or loss, picks highest-elevation (excluding
    BeiDou GEO and sats whose prev-epoch DDPR residual exceeded
    per_sat_res_thresh — they're likely multipath-contaminated).
    Returns (ref_idx, ref_sat).
    """
    prev_ref = tc.ref_sats.get(sys_id)
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
    recent_ref_bad = getattr(tc._sat_quality, 'recent_ref_bad', {}) or {}
    ref_bad_thr = float(tc.cfg.ref_bad_reject_thresh or 0.0)
    def ok(i):
        s = sat[i]
        if sys_id == uGNSS.BDS and _utils_is_bds_geo(s):
            return False
        if last_res.get(s, 0.0) > thresh:
            return False
        if ref_bad_thr > 0.0 and float(recent_ref_bad.get(int(s), 0.0) or 0.0) >= ref_bad_thr:
            return False
        return True
    pool = [i for i in idx_sys if ok(i)]
    if not pool:
        # Fallback: drop multipath filter (keep only GEO filter)
        pool = [i for i in idx_sys
                if not (sys_id == uGNSS.BDS and _utils_is_bds_geo(sat[i]))]
    if not pool:
        pool = idx_sys
    pool_arr = np.array(pool)
    ref_idx = int(pool_arr[np.argmax(el[pool_arr])])
    return ref_idx, sat[ref_idx]


def doppler_velocity_ls(tc, obs, obs_sd, rs, vs, iu, sat, pos_ecef,
                        vel_pred_ecef=None):
    """GICI-style Doppler → rover velocity LS. See utils.ls_solvers."""
    outlier = float(os.environ.get('DOP_OUTLIER_M_S', '3.0'))
    return _utils_doppler_velocity_ls(
        obs, obs_sd, rs, vs, iu, sat, pos_ecef,
        nav_nf=tc.nav.nf,
        get_wavelengths=lambda obs, sat: _utils_get_wavelengths(
            obs, sat, glo_ch=tc.nav.glo_ch),
        vel_pred_ecef=vel_pred_ecef,
        outlier_thresh_m_s=outlier)
