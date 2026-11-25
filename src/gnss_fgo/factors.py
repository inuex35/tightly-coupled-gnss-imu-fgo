"""Factor builders, arc bookkeeping, and cycle-slip detection."""

from dataclasses import dataclass, field
from typing import Optional

import gtsam
import numpy as np
from cssrlib.gnss import SAT_SYS_ARR, sat2prn, uGNSS

from .utils import get_wavelengths, huber_noise


# ── Satellite / arc state bookkeeping ─────────────────────────────────

@dataclass
class SatState:
    """Bookkeeping for a single (sat, freq) continuous carrier-phase arc."""

    outage_count: int = 0            # epochs since last seen
    amb_key: Optional[int] = None    # persistent GTSAM symbol for this arc's N
    amb_gen: int = 0                 # monotonic arc generation (++ on slip / reset)


@dataclass
class SatStateMap:
    """Per-(sat, freq) ``SatState`` map, per-sat GF slip state, and DD ref sats."""

    track: dict = field(default_factory=dict)            # {(sat,f): SatState}
    prev_gf_m: dict = field(default_factory=dict)        # {sat: geometry-free phase [m]}
    ref_sats: dict = field(default_factory=dict)         # {sys_id: sat}
    maxout: int = 5

    def get(self, sat: int, freq: int) -> SatState:
        key = (int(sat), int(freq))
        st = self.track.get(key)
        if st is None:
            st = SatState()
            self.track[key] = st
        return st

    def keys(self):
        return self.track.keys()

    def amb_keys_dict(self) -> dict:
        return {k: st.amb_key for k, st in self.track.items()
                if st.amb_key is not None}


# ── Cycle-slip detection & outage arc expiry ──────────────────────────

def _detect_gf_slip(tc, obs, row, s, lams, nf, reset_keys):
    """Geometry-free L1-L2 jump detector."""
    L1 = obs.L[row, 0]
    L2 = obs.L[row, 1]
    if L1 == 0 or L2 == 0 or lams[0] <= 0 or lams[1] <= 0:
        return
    gf = L1 * lams[0] - L2 * lams[1]
    prev_gf = tc._sat_states.prev_gf_m
    if s in prev_gf and prev_gf[s] != 0:
        if abs(gf - prev_gf[s]) > tc.cfg.thres_slip:
            for f in range(nf):
                reset_keys.add((s, f))
    prev_gf[s] = gf


def detect_slips_and_reset_ambiguities(tc, obs, obs_sd, sat, iu):
    """LLI + GF slip detection plus outage expiry. Returns (n_reset, reset_keys)."""
    nf = tc.nav.nf
    ns = len(sat)
    reset_keys = set()
    current_sats = set()

    for i in range(ns):
        s = sat[i]
        lams = get_wavelengths(obs_sd, s, glo_ch=tc.nav.glo_ch)
        for f in range(nf):
            current_sats.add((s, f))
            sat_state = tc._sat_states.get(s, f)
            sat_state.outage_count = 0
            if hasattr(obs, 'lli') and obs.lli[iu[i], f] == 1:
                reset_keys.add((s, f))
        if nf >= 2 and len(lams) >= 2:
            _detect_gf_slip(tc, obs, iu[i], s, lams, nf, reset_keys)

    maxout = tc._sat_states.maxout
    for key in list(tc._sat_states.keys()):
        if key not in current_sats:
            st = tc._sat_states.get(*key)
            st.outage_count += 1
            if st.outage_count > maxout:
                reset_keys.add(key)

    n_reset = 0
    for key in reset_keys:
        sat_st = tc._sat_states.get(*key)
        if sat_st.amb_key is not None:
            sat_st.amb_key = None
            n_reset += 1
        sat_st.amb_gen += 1
    return n_reset, reset_keys

# ── Ambiguity seeding ─────────────────────────────────────────────────

_SIGMA_AMB0_GAUGE = 30.0  # loose gauge prior [cyc] so initial code multipath cannot pull N


def init_dd_ambiguity_priors(tc, graph, values, new_amb,
                             existing_estimate, freq, lam, pair_sat_info,
                             seeded_keys=None):
    """Reuse existing smoother arc keys untouched; seed new arcs once clock-free."""
    for sat_id, key_n, cp_rover, cp_base, pr_rover, pr_base in pair_sat_info:
        key_id = (sat_id, freq)
        if key_id in new_amb or values.exists(key_n):
            continue
        if existing_estimate is not None and existing_estimate.exists(key_n):
            new_amb[key_id] = key_n
            continue
        n0_seed = ((cp_rover - cp_base) - (pr_rover - pr_base)) / lam
        values.insert(key_n, n0_seed)
        graph.addPriorDouble(key_n, n0_seed, tc._noise1(_SIGMA_AMB0_GAUGE))
        if seeded_keys is not None:
            seeded_keys.add(key_id)
        new_amb[key_id] = key_n


# ── DD pseudorange / carrier-phase factors ────────────────────────────

_CLIGHT = 299792458.0
_SCLKSTAB = 5e-12  # standard receiver clock stability [s/s] (RTKLIB default)


def dd_obs_sigma(tc, is_pseudorange, el_rad, dt_s):
    """RTKLIB varerr formula returning DD σ in metres."""
    cfg = tc.cfg
    fact = cfg.err_eratio_pr if is_pseudorange else 1.0
    a = fact * cfg.err_a
    d = _CLIGHT * _SCLKSTAB * dt_s
    sinel = max(np.sin(el_rad), 0.05)
    var_sd = 2.0 * (a * a * (1.0 + 1.0 / (sinel * sinel))) + d * d
    return float(np.sqrt(2.0 * var_sd))


def _pick_ref_sat_idx(tc, sys_id, idx_sys, sat, el):
    """Select the highest-elevation non-BDS-GEO reference satellite."""
    prev_ref = tc._sat_states.ref_sats.get(sys_id)
    sats_in_sys = [sat[i] for i in idx_sys]
    prev_is_geo = (sys_id == uGNSS.BDS and prev_ref is not None
                   and _is_bds_geo(prev_ref))
    if prev_ref is not None and prev_ref in sats_in_sys and not prev_is_geo:
        return idx_sys[sats_in_sys.index(prev_ref)], prev_ref

    pool = [i for i in idx_sys
            if not (sys_id == uGNSS.BDS and _is_bds_geo(sat[i]))]
    if not pool:
        pool = idx_sys
    pool_arr = np.array(pool)
    ref_idx = int(pool_arr[np.argmax(el[pool_arr])])
    return ref_idx, sat[ref_idx]


def _is_bds_geo(prn_sat: int) -> bool:
    """True for BeiDou GEO slots (C1-C5, C59-C63) — excluded as DD ref."""
    prn = sat2prn(prn_sat)[1]
    return prn <= 5 or 59 <= prn <= 63


def _resolve_arc_amb_key(tc, graph, values, new_amb, existing_estimate, sat_id, freq):
    """Return the active ambiguity symbol for (sat_id, freq).

    Reuses the existing symbol while the arc stays in the smoother, chains a
    random-walk ``BetweenFactorDouble`` step when ``amb_rw_sigma > 0``, or
    allocates a fresh symbol if the previous arc was marginalized out.
    """
    sf = (int(sat_id), int(freq))
    if sf in new_amb:
        return new_amb[sf]
    st = tc._sat_states.get(sat_id, freq)
    rw = float(tc.cfg.amb_rw_sigma)
    if (rw > 0.0 and st.amb_key is not None
            and existing_estimate is not None
            and existing_estimate.exists(st.amb_key)):
        k_old = st.amb_key
        st.amb_gen += 1
        k_new = tc.N(sat_id, freq, st.amb_gen)
        values.insert(k_new, existing_estimate.atDouble(k_old))
        graph.add(gtsam.BetweenFactorDouble(k_old, k_new, 0.0, tc._noise1(rw)))
        new_amb[sf] = k_new
        return k_new
    if st.amb_key is not None:
        if existing_estimate is None or existing_estimate.exists(st.amb_key):
            return st.amb_key
        st.amb_key = None
        st.amb_gen += 1
    return tc.N(sat_id, freq, st.amb_gen)


def _build_cp_pair(tc, graph, values, obs, obsb, iu, ir_map, key_pose, lever,
                   new_amb, existing_estimate,
                   visible_sat_freqs, seeded_sat_freqs,
                   sys_id, ref_idx, j_idx, ref_sat, j_sat, f,
                   lams, sat_pts, pr_obs, sigma):
    if sys_id == uGNSS.GLO or f >= len(lams) or lams[f] <= 0:
        return 0
    lam = lams[f]
    cp_ref_r = obs.L[iu[ref_idx], f] * lam
    cp_ref_b = obsb.L[ir_map[ref_sat], f] * lam if ref_sat in ir_map else 0
    cp_j_r = obs.L[iu[j_idx], f] * lam
    cp_j_b = obsb.L[ir_map[j_sat], f] * lam if j_sat in ir_map else 0
    if cp_ref_r == 0 or cp_ref_b == 0 or cp_j_r == 0 or cp_j_b == 0:
        return 0

    visible_sat_freqs.add((int(ref_sat), int(f)))
    visible_sat_freqs.add((int(j_sat), int(f)))

    key_n_ref = _resolve_arc_amb_key(
        tc, graph, values, new_amb, existing_estimate, ref_sat, f)
    key_n_j = _resolve_arc_amb_key(
        tc, graph, values, new_amb, existing_estimate, j_sat, f)
    pr_ref_r, pr_ref_b, pr_j_r, pr_j_b = pr_obs
    init_dd_ambiguity_priors(
        tc, graph, values, new_amb,
        existing_estimate, f, lam,
        ((ref_sat, key_n_ref, cp_ref_r, cp_ref_b, pr_ref_r, pr_ref_b),
         (j_sat, key_n_j, cp_j_r, cp_j_b, pr_j_r, pr_j_b)),
        seeded_keys=seeded_sat_freqs)

    ref_pt, j_pt, ref_base_pt, j_base_pt = sat_pts
    graph.add(gtsam.DoubleDifferenceCarrierPhaseFactorArm(
        key_pose, key_n_ref, key_n_j,
        cp_ref_r, cp_ref_b, cp_j_r, cp_j_b,
        ref_pt, j_pt, ref_base_pt, j_base_pt, tc.base_pt, lam,
        lever, tc.ecef_T_nav, huber_noise(sigma, float(tc.cfg.huber_k))))
    return 1


def build_dd_factors(tc, graph, values, obs, obsb, obs_sd,
                     rs, rsb, sat, el, iu, ir_map,
                     key_pose, lever, existing_estimate=None):
    """Build pure C++ DDPR + DDCP factors; return (n_dd, visible_sat_freqs, seeded_sat_freqs)."""
    ns = len(sat)
    if ns == 0 or obsb is None or obs_sd is None:
        return 0, set(), set()
    nf = tc.nav.nf
    el_min_rad = np.radians(max(1.0, tc.cfg.elmin_deg))
    dt_s = float(tc._epoch_dt)
    visible_sat_freqs, seeded_sat_freqs, new_amb = set(), set(), {}
    n_dd = 0
    sat_sys_for_obs = SAT_SYS_ARR[np.asarray(sat[:ns], dtype=np.int32)]

    for sys_id in sorted(obs_sd.sig.keys(), key=int):
        idx_sys = np.where(sat_sys_for_obs == int(sys_id))[0].tolist()
        if len(idx_sys) < 2:
            continue
        ref_idx, ref_sat = _pick_ref_sat_idx(tc, sys_id, idx_sys, sat, el)
        tc._sat_states.ref_sats[sys_id] = ref_sat
        lams = get_wavelengths(obs_sd, ref_sat, glo_ch=tc.nav.glo_ch)
        ref_pt = gtsam.Point3(*rs[iu[ref_idx], :3])
        ref_base_pt = (gtsam.Point3(*rsb[ir_map[ref_sat], :3])
                       if ref_sat in ir_map else ref_pt)
        for j_idx in idx_sys:
            if j_idx == ref_idx:
                continue
            j_sat = sat[j_idx]
            if SAT_SYS_ARR[j_sat] == uGNSS.BDS and _is_bds_geo(j_sat):
                continue
            j_pt = gtsam.Point3(*rs[iu[j_idx], :3])
            j_base_pt = (gtsam.Point3(*rsb[ir_map[j_sat], :3])
                         if j_sat in ir_map else j_pt)
            sat_pts = (ref_pt, j_pt, ref_base_pt, j_base_pt)
            el_pair = max(min(el[ref_idx], el[j_idx]), el_min_rad)
            sigma_pr = dd_obs_sigma(tc, True, el_pair, dt_s)
            sigma_cp = dd_obs_sigma(tc, False, el_pair, dt_s)
            for f in range(nf):
                if obs_sd.P[ref_idx, f] == 0 or obs_sd.P[j_idx, f] == 0:
                    continue
                pr_obs = (
                    obs.P[iu[ref_idx], f],
                    obsb.P[ir_map[ref_sat], f] if ref_sat in ir_map else 0,
                    obs.P[iu[j_idx], f],
                    obsb.P[ir_map[j_sat], f] if j_sat in ir_map else 0,
                )
                if 0 in pr_obs:
                    continue
                graph.add(gtsam.DoubleDifferencePseudorangeFactorArm(
                    key_pose, *pr_obs, *sat_pts, tc.base_pt, lever,
                    tc.ecef_T_nav, huber_noise(sigma_pr, float(tc.cfg.huber_k))))
                n_dd += 1
                if obs_sd.L[ref_idx, f] == 0 or obs_sd.L[j_idx, f] == 0:
                    continue
                n_dd += _build_cp_pair(
                    tc, graph, values, obs, obsb, iu, ir_map, key_pose, lever,
                    new_amb, existing_estimate,
                    visible_sat_freqs, seeded_sat_freqs,
                    sys_id, ref_idx, j_idx, ref_sat, j_sat, f,
                    lams, sat_pts, pr_obs, sigma_cp)
    for (s, f), k in new_amb.items():
        tc._sat_states.get(s, f).amb_key = k
    return n_dd, visible_sat_freqs, seeded_sat_freqs


# ── Single-differenced Doppler factors ────────────────────────────────

def _doppler_rows(tc, obs, obs_sd, rs, vs, sat, el, iu):
    """Collect (sat, sat_pos, sat_vel, doppler_hz, wavelength_m, el_rad)."""
    if obs_sd is None or len(iu) == 0:
        return []
    rows = []
    for si, i_obs in enumerate(iu):
        s = int(sat[si])
        if sat2prn(s)[0] == uGNSS.GLO:
            continue
        if i_obs >= obs.D.shape[0]:
            continue
        lams = None
        # Use the first valid frequency per satellite (SD Doppler is geometry-based).
        for f in range(min(tc.nav.nf, obs.D.shape[1])):
            d_obs = obs.D[i_obs, f]
            if d_obs == 0.0:
                continue
            if lams is None:
                lams = get_wavelengths(obs_sd, s, glo_ch=tc.nav.glo_ch)
            if f < len(lams) and lams[f] > 0:
                rows.append((s, rs[i_obs, :3], vs[i_obs, :3],
                             float(d_obs), float(lams[f]), float(el[si])))
                break
    return rows


def add_sd_doppler_factors(tc, graph, key_idx, obs, obs_sd, rs, vs, sat, el,
                           iu, pred_ecef, gyro_mean, info):
    """Add clock-free SD Doppler factors against the highest-elevation reference sat."""
    sigma = float(tc.cfg.doppler_sd_sigma)
    if sigma <= 0 or key_idx is None:
        return

    rows = _doppler_rows(tc, obs, obs_sd, rs, vs, sat, el, iu)
    if len(rows) < 2:
        return

    sigmas = [sigma / max(np.sin(r[5]), 0.1) for r in rows]
    i_ref = int(np.argmin(sigmas))
    ref, sigma_ref = rows[i_ref], sigmas[i_ref]

    omega = np.zeros(3)
    if gyro_mean is not None:
        bias_gyro = (tc.tc_bias.gyroscope() if tc.tc_bias is not None
                     else np.zeros(3))
        omega = np.asarray(gyro_mean, dtype=float) - bias_gyro

    key_idx = int(key_idx)
    lever = np.asarray(tc.lever_arm_tc, dtype=float)
    rr = np.asarray(pred_ecef, dtype=float)
    n = 0
    huber = float(tc.cfg.huber_k)
    for row, sig in zip(rows, sigmas):
        if row is ref:
            continue
        noise = huber_noise(np.hypot(sig, sigma_ref), huber)
        graph.add(gtsam.SingleDifferenceDopplerFactorArm(
            tc.Pose(key_idx), tc.Vel(key_idx),
            row[3], ref[3],
            row[4], ref[4],
            np.asarray(row[1], dtype=float), np.asarray(row[2], dtype=float),
            np.asarray(ref[1], dtype=float), np.asarray(ref[2], dtype=float),
            rr, lever, tc.ecef_T_nav, omega,
            0.0, 0.0,
            noise))
        n += 1
    if n:
        info['doppler_sd_n'] = n
        info['doppler_sd_ref'] = int(ref[0])


# ── Non-Holonomic Constraint (NHC) ────────────────────────────────────

def add_nhc_factor(tc, graph, key_idx, gyro_mean=None):
    """Non-Holonomic Constraint at rear-axle pivot in the FLU body frame."""
    sig_lat = float(tc.cfg.nhc_sigma_lat)
    sig_vert = float(tc.cfg.nhc_sigma_vert)
    if sig_lat <= 0 or sig_vert <= 0:
        return False
    lever = getattr(tc, '_nhc_lever', None)
    if lever is None:
        lever = np.array([float(x) for x in tc.cfg.nhc_lever.split(',')], dtype=float)
    if gyro_mean is not None:
        bias_gyro = (tc.tc_bias.gyroscope() if tc.tc_bias is not None
                     else np.zeros(3))
        omega = np.asarray(gyro_mean - bias_gyro, dtype=float)
    else:
        omega = np.zeros(3)
    noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([sig_lat, sig_vert]))
    graph.add(gtsam.NhcFactor(
        tc.Pose(key_idx), tc.Vel(key_idx), omega,
        np.asarray(lever, dtype=float), noise))
    return True
