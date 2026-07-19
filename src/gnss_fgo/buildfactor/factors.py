"""DD pseudorange / carrier-phase factor builder for the TC graph."""

import os
import numpy as np
import gtsam
from cssrlib.gnss import uGNSS, geodist, SAT_SYS_ARR, rCST
from ..utils.geometry import is_bds_geo as _is_bds_geo
from ..utils.robust import maybe_robust as _maybe_robust

from ..preprocess import sat_quality as _satq
from ..preprocess import prefit as _tc_prefit
from .factors_support import compute_cp_build_policy, get_wavelengths
from ..utils import sorted_amb_keys, sorted_sys_ids


def _seed_one_amb_prior(tc, graph, values, sat_st, key_n, n0_seed,
                          prev_amb_values, key_id):
    """Insert N value + add prior with the right σ for one sat. Three modes:"""
    if prev_amb_values is not None and key_id in prev_amb_values:
        n0 = prev_amb_values[key_id][1]
        values.insert(key_n, n0)
        graph.addPriorDouble(key_n, n0, tc._noise1(tc.cfg.sigma_cont))
        return
    if (sat_st.release_seed_pending
            and sat_st.last_held_value is not None):
        n0 = sat_st.last_held_value
        values.insert(key_n, n0)
        graph.addPriorDouble(key_n, n0, tc._noise1(0.1))
        sat_st.amb_init_epoch = tc.epoch
        sat_st.release_seed_pending = False
        return
    # Phase 2: σ=sigma_amb0 (cssrlib sig_n0); Phase 1: σ=3 cyc.
    sig = tc.cfg.sigma_amb0 if tc.phase == 2 else 3.0
    values.insert(key_n, n0_seed)
    graph.addPriorDouble(key_n, n0_seed, tc._noise1(sig))
    sat_st.amb_init_epoch = tc.epoch


def _init_dd_ambiguity_priors(tc, graph, values, amb_dict, new_amb,
                                prev_amb_values, freq, lam,
                                pair_sat_info, pos_ecef, rb):
    """Add Prior factors + initial values for the two N ambiguities of one"""
    for sat_id, key_n, cp_rover, cp_base, sat_xyz in pair_sat_info:
        key_id = (sat_id, freq)
        sat_st = tc._sat_states.get(*key_id)
        sat_st.amb_lam = lam
        if sat_st.held_value is not None:
            continue
        if key_id in amb_dict or key_id in new_amb:
            continue
        if values.exists(key_n):
            continue
        rv, _ = geodist(sat_xyz, pos_ecef)
        bv, _ = geodist(sat_xyz, rb)
        n0_seed = ((cp_rover - cp_base) - (rv - bv)) / lam
        _seed_one_amb_prior(tc, graph, values, sat_st, key_n, n0_seed,
                              prev_amb_values, key_id)
        new_amb[key_id] = key_n


def _add_ddpr_factor(tc, graph, key_pose, lever,
                      pr_obs, sat_pts, pair_id,
                      pair_sigma_base):
    """Add one DDPR factor for a (ref, j, freq) triple to ``graph``."""
    pr_ref_r, pr_ref_b, pr_j_r, pr_j_b = pr_obs
    ref_pt, j_pt, ref_base_pt, j_base_pt = sat_pts
    ref_sat, j_sat, freq = pair_id
    pr_base = tc._noise1(pair_sigma_base)
    pr_noise = (pr_base if tc.cfg.huber_pr <= 0
                else _maybe_robust(
                    pr_base, tc.cfg.huber_pr,
                    kind=tc.cfg.pr_robust_kind))
    tc._last_ddpr_sat_tags.append(
        (graph.size(), ref_sat, j_sat, freq))
    graph.add(gtsam.DoubleDifferencePseudorangeFactorArm(
        key_pose, pr_ref_r, pr_ref_b, pr_j_r, pr_j_b,
        ref_pt, j_pt, ref_base_pt, j_base_pt,
        tc.base_pt, lever, tc.ecef_T_nav, pr_noise))



def _compute_cp_sigma(pair_sigma_base, cp_sigma_mult):
    """Compute σ for one DDCP pair."""
    return pair_sigma_base * cp_sigma_mult


def _make_ddcp_factor_with_held_n(key_pose, key_float, noise,
                                  sat_ref_rov, sat_j_rov,
                                  sat_ref_base, sat_j_base,
                                  base_ecef, lam, dd_obs_cp,
                                  lever_arr, ecef_T_nav,
                                  offset_m, coeff_m):
    """Custom DDCP factor with 0/1 N-variable arity (held N folded into ``offset_m``)."""
    keys = [key_pose] if key_float is None else [key_pose, key_float]
    has_float = key_float is not None

    # Per-factor constants — fold base→sat ranges + offset_m + dd_obs_cp into
    # a single scalar so error_fn does only the rover-side geodist work.
    rho_base_ref, _ = geodist(sat_ref_base, base_ecef)
    rho_base_j, _ = geodist(sat_j_base, base_ecef)
    err_const = offset_m - (rho_base_ref - rho_base_j) - dd_obs_cp

    # Earth-rotation correction terms for each rover satellite (constants).
    omc = rCST.OMGE / rCST.CLIGHT
    sr0, sr1 = float(sat_ref_rov[0]), float(sat_ref_rov[1])
    sj0, sj1 = float(sat_j_rov[0]), float(sat_j_rov[1])
    omc_diff_x = omc * (sr1 - sj1)
    omc_diff_y = omc * (sr0 - sj0)

    lx, ly, lz = float(lever_arr[0]), float(lever_arr[1]), float(lever_arr[2])
    lever_arr_local = lever_arr  # closure capture

    err_arr = np.empty(1, dtype=float)
    coeff_jac = np.array([[coeff_m]], dtype=float) if has_float else None

    def error_fn(this, values, jacobians):
        pose = values.atPose3(this.keys()[0])
        pose_ecef = ecef_T_nav.compose(pose) if ecef_T_nav is not None else pose
        R = pose_ecef.rotation().matrix()
        t = pose_ecef.translation()
        antenna_ecef = t + R @ lever_arr_local

        rho_rov_ref, e_ref = geodist(sat_ref_rov, antenna_ecef)
        rho_rov_j, e_j = geodist(sat_j_rov, antenna_ecef)

        err = (rho_rov_ref - rho_rov_j) + err_const
        if has_float:
            err += coeff_m * values.atDouble(this.keys()[1])

        if jacobians is not None:
            # H_ant = D(e_ref, sat_ref_rov) - D(e_j, sat_j_rov), inlined.
            h0 = -(e_ref[0] - e_j[0]) - omc_diff_x
            h1 = -(e_ref[1] - e_j[1]) + omc_diff_y
            h2 = -(e_ref[2] - e_j[2])

            # h_R = H_ant @ R  (rotation-frame antenna sensitivity).
            h_R0 = h0 * R[0, 0] + h1 * R[1, 0] + h2 * R[2, 0]
            h_R1 = h0 * R[0, 1] + h1 * R[1, 1] + h2 * R[2, 1]
            h_R2 = h0 * R[0, 2] + h1 * R[1, 2] + h2 * R[2, 2]

            # H_rot = H_ant @ (-R @ skew(lever)) = lever × h_R  (cross product).
            H_pose = np.empty((1, 6))
            H_pose[0, 0] = ly * h_R2 - lz * h_R1
            H_pose[0, 1] = lz * h_R0 - lx * h_R2
            H_pose[0, 2] = lx * h_R1 - ly * h_R0
            H_pose[0, 3] = h_R0
            H_pose[0, 4] = h_R1
            H_pose[0, 5] = h_R2

            jacobians[0] = H_pose
            if has_float:
                jacobians[1] = coeff_jac

        err_arr[0] = err
        return err_arr.copy()

    return gtsam.CustomFactor(noise, keys, error_fn)


def _emit_held_ddcp_factor(tc, graph, fi_cp, pair_id, key_pose, key_float,
                             cp_noise, sat_xyz, rb, lam, dd_obs_cp,
                             lever_arr, offset_m, coeff_m):
    """Add a held-N variant DDCP factor (``_make_ddcp_factor_with_held_n``) and stamp it into ``tc._last_custom_ddcp_local`` / ``tc._last_custom_ddcp_global`` so the post-fit FDE evaluates it via the custom-factor accessor instead of reading aN()/cN()."""
    ref_xyz, j_xyz, ref_base_xyz, j_base_xyz = sat_xyz
    graph.add(_make_ddcp_factor_with_held_n(
        key_pose, key_float, cp_noise,
        ref_xyz, j_xyz, ref_base_xyz, j_base_xyz,
        rb, lam, dd_obs_cp, lever_arr, tc.ecef_T_nav,
        offset_m=offset_m, coeff_m=coeff_m))
    tc._last_custom_ddcp_local.add(graph.size() - 1)
    # Keyed by the factor's key tuple, NOT its slot index: the FLS
    # reuses freed slots (findUnusedFactorSlots) and per-epoch counter
    # arithmetic cannot name a slot reliably. When both ambiguities are
    # held the factor is pose-only and its key tuple is not unique
    # (every such factor shares the pose key) — leave those out of the
    # FDE bookkeeping rather than guess.
    if key_float is not None:
        tc._last_custom_ddcp_global[
            (int(key_pose), int(key_float))] = pair_id


def _add_ddcp_factor(tc, graph, key_pose, cp_noise, dd_obs_cp, lam,
                      sat_pts, sat_xyz, cp_obs,
                      lever, lever_arr, rb,
                      keys_n, held, states,
                      pair_id, fi_cp, track_indices):
    """Add the DDCP factor for one (ref, j, freq) triple to ``graph``,"""
    key_n_ref, key_n_j = keys_n
    ref_held_value, j_held_value = held
    ref_state, j_state = states
    held_kw = dict(
        tc=tc, graph=graph, fi_cp=fi_cp, pair_id=pair_id,
        key_pose=key_pose, cp_noise=cp_noise, sat_xyz=sat_xyz,
        rb=rb, lam=lam, dd_obs_cp=dd_obs_cp, lever_arr=lever_arr)
    if ref_held_value is not None and j_held_value is not None:
        _emit_held_ddcp_factor(
            **held_kw, key_float=None,
            offset_m=lam * (ref_held_value - j_held_value), coeff_m=0.0)
    elif ref_held_value is not None:
        _emit_held_ddcp_factor(
            **held_kw, key_float=key_n_j,
            offset_m=lam * ref_held_value, coeff_m=-lam)
        if track_indices:
            j_state.amb_factor_indices.append(fi_cp)
    elif j_held_value is not None:
        _emit_held_ddcp_factor(
            **held_kw, key_float=key_n_ref,
            offset_m=-lam * j_held_value, coeff_m=lam)
        if track_indices:
            ref_state.amb_factor_indices.append(fi_cp)
    else:
        ref_pt, j_pt, ref_base_pt, j_base_pt = sat_pts
        cp_ref_r, cp_ref_b, cp_j_r, cp_j_b = cp_obs
        graph.add(gtsam.DoubleDifferenceCarrierPhaseFactorArm(
            key_pose, key_n_ref, key_n_j,
            cp_ref_r, cp_ref_b, cp_j_r, cp_j_b,
            ref_pt, j_pt, ref_base_pt, j_base_pt, tc.base_pt, lam,
            lever, tc.ecef_T_nav, cp_noise))
        if track_indices:
            ref_state.amb_factor_indices.append(fi_cp)
            j_state.amb_factor_indices.append(fi_cp)



class DdFactorBuilder:
    """Per-call state bag for ``build_dd_factors``."""

    def __init__(self, tc, graph, values, obs, obsb, obs_sd,
                 rs, rsb, sat, el, iu, ir_map, pos_ecef,
                 key_pose, lever, amb_dict, track_indices=False,
                 dd_epoch=0, prev_amb_values=None,
                 skip_cp=False, slip_keys=None):
        # Inputs
        self.tc = tc
        self.graph = graph
        self.values = values
        self.obs = obs
        self.obsb = obsb
        self.obs_sd = obs_sd
        self.rs = rs
        self.rsb = rsb
        self.sat = sat
        self.el = el
        self.iu = iu
        self.ir_map = ir_map
        self.pos_ecef = pos_ecef
        self.key_pose = key_pose
        self.lever = lever
        self.amb_dict = amb_dict
        self.track_indices = track_indices
        self.dd_epoch = dd_epoch
        self.prev_amb_values = prev_amb_values
        self.skip_cp = skip_cp
        self.slip_keys = slip_keys

        # Derived constants
        self.nf = tc.nav.nf
        self.ns = len(sat)
        self.rb = np.array(tc.nav.rb)
        self.lever_arr = np.asarray(lever, dtype=float)

        # Reset per-epoch DDPR / DDCP tags so consumers see this epoch only.
        tc._last_ddpr_sat_tags = []
        tc._last_custom_ddcp_local = set()
        tc._last_custom_ddcp_global = {}
        tc._ar_cp_visible_sf = set()

        self._prev_keys = (set(prev_amb_values.keys())
                            if prev_amb_values else set())

        tc._last_cp_pr_reject = 0

        # Elevation + SNR / cfg constants used by pair_sigma + bad scaling
        self.el_min_rad = np.radians(max(1.0, tc.cfg.el_mask_deg))
        self.dt_s = float(tc._epoch_dt)
        self.use_varerr = bool(tc.cfg.varerr_enable)

        self.sq = _satq.get_sat_quality(tc)

        # Mutable accumulators
        self.nv = 0
        self.new_amb = {}

    # --- closures-as-methods ---

    def has_prev_or_held(self, sat_id, freq):
        return ((sat_id, freq) in self._prev_keys
                or self.tc._sat_states.at(sat_id, freq).held_value is not None)

    def is_fresh_pair(self, ref_s, j_s, freq):
        return (not self.has_prev_or_held(ref_s, freq)
                or not self.has_prev_or_held(j_s, freq))


    def _select_ref_for_system(self, sys_id, idx_sys, sat, el,
                                amb_dict, slip_keys):
        """Pick reference sat for system; reset SD ambiguities only when the previous ref actually slipped (not on geometry-driven re-pick)."""
        tc = self.tc
        prev_ref = tc.ref_sats.get(sys_id, None)
        ref_idx, ref_sat = _tc_prefit.pick_ref_sat_idx(
            tc, sys_id, idx_sys, sat, el)
        if prev_ref is not None and prev_ref != ref_sat:
            reset_on_ref_switch = bool(
                os.environ.get('REF_SWITCH_RESET_ALWAYS', '0') != '0')
            if reset_on_ref_switch or slip_keys is None:
                reset_sys = [(s, f) for (s, f) in sorted_amb_keys(amb_dict)
                             if SAT_SYS_ARR[s] == sys_id]
                for key in reset_sys:
                    del amb_dict[key]
                    _st = tc._sat_states.track.get(key)
                    if _st is not None:
                        _st.amb_factor_indices = []
        tc.ref_sats[sys_id] = ref_sat
        return ref_idx, ref_sat

    def _compute_ref_geometry(self, ref_idx, ref_sat):
        """Reference-sat ``Point3`` + xyz hoisted outside the j loop (pybind ``Point3()`` isn't free). Returns ``(ref_pt, ref_base_pt, ref_xyz, ref_base_xyz)``."""
        rs = self.rs
        rsb = self.rsb
        iu = self.iu
        ir_map = self.ir_map
        ref_pt = gtsam.Point3(*rs[iu[ref_idx], :3])
        ref_base_pt = (gtsam.Point3(*rsb[ir_map[ref_sat], :3])
                       if ref_sat in ir_map else ref_pt)
        ref_xyz = np.asarray(rs[iu[ref_idx], :3], dtype=float)
        ref_base_xyz = (np.asarray(rsb[ir_map[ref_sat], :3], dtype=float)
                        if ref_sat in ir_map else ref_xyz)
        return ref_pt, ref_base_pt, ref_xyz, ref_base_xyz

    def _build_pr_for_pair(self, ref_idx, j_idx, ref_sat, j_sat, f,
                            sat_pts):
        """Build the DDPR factor for one (ref, j, f) and return the 4-tuple of PR observations on success, ``None`` when any of the four PR values is zero (skip)."""
        obs = self.obs
        obsb = self.obsb
        iu = self.iu
        ir_map = self.ir_map
        pr_ref_r = obs.P[iu[ref_idx], f]
        pr_ref_b = obsb.P[ir_map[ref_sat], f] if ref_sat in ir_map else 0
        pr_j_r = obs.P[iu[j_idx], f]
        pr_j_b = obsb.P[ir_map[j_sat], f] if j_sat in ir_map else 0
        if pr_ref_r == 0 or pr_ref_b == 0 or pr_j_r == 0 or pr_j_b == 0:
            return None
        _add_ddpr_factor(
            self.tc, self.graph, self.key_pose, self.lever,
            pr_obs=(pr_ref_r, pr_ref_b, pr_j_r, pr_j_b),
            sat_pts=sat_pts,
            pair_id=(ref_sat, j_sat, f),
            pair_sigma_base=self.pair_sigma(
                1, f, self.el[ref_idx], self.el[j_idx]),
            )
        return pr_ref_r, pr_ref_b, pr_j_r, pr_j_b

    def _build_cp_for_pair(self, sys_id, ref_idx, j_idx, ref_sat, j_sat, f,
                            lams, sat_pts, sat_xyz, pr_obs,
                            fresh_pair,
                            amb_dict, new_amb, dd_epoch):
        """Build the DDCP factor for one (ref, j, f) — handles N init, CP-vs-PR consistency gate, σ computation, and the actual ``_add_ddcp_factor`` call. Returns ``1`` on success, ``0`` when skipped (GLO PR-only / missing λ / zero CP / cp_allowed=False / consistency gate reject)."""
        tc = self.tc
        if sys_id == uGNSS.GLO:
            return 0
        if f >= len(lams) or lams[f] <= 0:
            return 0
        lam = lams[f]
        obs = self.obs
        obsb = self.obsb
        iu = self.iu
        ir_map = self.ir_map
        cp_ref_r = obs.L[iu[ref_idx], f] * lam
        cp_ref_b = (obsb.L[ir_map[ref_sat], f] * lam
                    if ref_sat in ir_map else 0)
        cp_j_r = obs.L[iu[j_idx], f] * lam
        cp_j_b = (obsb.L[ir_map[j_sat], f] * lam
                  if j_sat in ir_map else 0)
        if cp_ref_r == 0 or cp_ref_b == 0 or cp_j_r == 0 or cp_j_b == 0:
            return 0
        tc._ar_cp_visible_sf.add((int(ref_sat), int(f)))
        tc._ar_cp_visible_sf.add((int(j_sat), int(f)))

        ref_state = tc._sat_states.get(ref_sat, f)
        j_state = tc._sat_states.get(j_sat, f)
        ref_held_value = ref_state.held_value
        j_held_value = j_state.held_value
        key_n_ref = tc.N(ref_sat, f, dd_epoch * 100 + ref_state.amb_gen)
        key_n_j = tc.N(j_sat, f, dd_epoch * 100 + j_state.amb_gen)
        # N初期化: hybrid (continuing prior / release-seed / fresh)
        _init_dd_ambiguity_priors(
            tc, self.graph, self.values, amb_dict, new_amb,
            self.prev_amb_values, f, lam,
            ((ref_sat, key_n_ref, cp_ref_r, cp_ref_b,
              self.rs[iu[ref_idx], :3]),
             (j_sat, key_n_j, cp_j_r, cp_j_b,
              self.rs[iu[j_idx], :3])),
            self.pos_ecef, self.rb)

        sq_state = _satq.get_sat_quality(tc)
        cp_allowed, cp_sigma_mult = compute_cp_build_policy(
            tc, sq_state, ref_sat, j_sat, f, self.skip_cp)
        if not cp_allowed:
            return 0
        # Track CP factor index for both ref and target satellites
        fi_cp = tc.total_factor_count + self.graph.size()
        cp_sigma = _compute_cp_sigma(
            self.pair_sigma(0, f, self.el[ref_idx], self.el[j_idx]),
            cp_sigma_mult)
        cp_noise = tc._noise1(cp_sigma)
        dd_obs_cp = (cp_ref_r - cp_j_r) - (cp_ref_b - cp_j_b)
        _add_ddcp_factor(
            tc, self.graph, self.key_pose, cp_noise, dd_obs_cp, lam,
            sat_pts=sat_pts, sat_xyz=sat_xyz,
            cp_obs=(cp_ref_r, cp_ref_b, cp_j_r, cp_j_b),
            lever=self.lever, lever_arr=self.lever_arr, rb=self.rb,
            keys_n=(key_n_ref, key_n_j),
            held=(ref_held_value, j_held_value),
            states=(ref_state, j_state),
            pair_id=(ref_sat, j_sat, f),
            fi_cp=fi_cp, track_indices=self.track_indices)
        return 1

    def _compute_pair_geometry(self, j_idx, j_sat):
        """j-sat SD geometry: rover/base ``Point3`` + xyz arrays. Returns ``(j_pt, j_base_pt, j_xyz, j_base_xyz)``."""
        rs = self.rs
        rsb = self.rsb
        iu = self.iu
        ir_map = self.ir_map
        j_pt = gtsam.Point3(*rs[iu[j_idx], :3])
        j_base_pt = (gtsam.Point3(*rsb[ir_map[j_sat], :3])
                     if j_sat in ir_map else j_pt)
        j_xyz = np.asarray(rs[iu[j_idx], :3], dtype=float)
        j_base_xyz = (np.asarray(rsb[ir_map[j_sat], :3], dtype=float)
                      if j_sat in ir_map else j_xyz)
        return j_pt, j_base_pt, j_xyz, j_base_xyz

    def pair_sigma(self, code, freq, el_ref_rad, el_j_rad):
        """DD σ for the (code, freq, ref/j) pair — RTKLIB-demo5 ``varerr``"""
        if self.use_varerr:
            el_pair = max(min(el_ref_rad, el_j_rad), self.el_min_rad)
            return _tc_prefit.varerr_dd_sigma(
                self.tc, code, freq, el_pair, self.dt_s)
        sigma_base = self.tc.cfg.sigma_pr if code else self.tc.cfg.sigma_cp
        return sigma_base * np.sqrt(2)

    def run(self):
        """Build all DD factors for this call and return the factor count"""
        tc = self.tc
        obs = self.obs
        obs_sd = self.obs_sd
        sat = self.sat
        el = self.el
        iu = self.iu
        amb_dict = self.amb_dict
        dd_epoch = self.dd_epoch
        slip_keys = self.slip_keys
        nf = self.nf
        ns = self.ns
        new_amb = self.new_amb
        _is_fresh_pair = self.is_fresh_pair
        nv = 0

        sat_sys_for_obs = SAT_SYS_ARR[np.asarray(sat[:ns], dtype=np.int32)]
        for sys_id in sorted_sys_ids(obs_sd.sig):
            idx_sys = np.where(sat_sys_for_obs == int(sys_id))[0].tolist()
            if len(idx_sys) < 2:
                continue
            ref_idx, ref_sat = self._select_ref_for_system(
                sys_id, idx_sys, sat, el, amb_dict, slip_keys)
            lams = get_wavelengths(tc, obs_sd, ref_sat)
            (ref_pt, ref_base_pt, ref_xyz,
             ref_base_xyz) = self._compute_ref_geometry(ref_idx, ref_sat)

            cmc_skip = tc._sat_states.cmc_skip_dd
            skip_bds_geo = bool(tc.cfg.exclude_bds_geo)
            for j_idx in idx_sys:
                if j_idx == ref_idx:
                    continue
                j_sat = sat[j_idx]
                if (skip_bds_geo
                        and SAT_SYS_ARR[j_sat] == uGNSS.BDS
                        and _is_bds_geo(j_sat)):
                    continue
                (j_pt, j_base_pt, j_xyz,
                 j_base_xyz) = self._compute_pair_geometry(j_idx, j_sat)

                sat_pts = (ref_pt, j_pt, ref_base_pt, j_base_pt)
                sat_xyz = (ref_xyz, j_xyz, ref_base_xyz, j_base_xyz)
                for f in range(nf):
                    if (j_sat, f) in cmc_skip:
                        continue
                    # Pair-level features used by both PR and CP build.
                    fresh_pair = _is_fresh_pair(ref_sat, j_sat, f)
                    # PR factor: CPなし (L信号不在) でも追加する
                    pr_obs = self._build_pr_for_pair(
                        ref_idx, j_idx, ref_sat, j_sat, f,
                        sat_pts)
                    if pr_obs is None:
                        continue
                    nv += 1
                    nv += self._build_cp_for_pair(
                        sys_id, ref_idx, j_idx, ref_sat, j_sat, f, lams,
                        sat_pts, sat_xyz, pr_obs,
                        fresh_pair,
                        amb_dict, new_amb, dd_epoch)

        amb_dict.update(new_amb)
        self.nv = nv
        return nv


def build_dd_factors(tc, graph, values, obs, obsb, obs_sd,
                          rs, rsb, sat, el, iu, ir_map, pos_ecef,
                          key_pose, lever, amb_dict, track_indices=False,
                          dd_epoch=0, prev_amb_values=None,
                          skip_cp=False, slip_keys=None):
    """Build DD pseudorange + carrier phase factors (Arm version)."""
    builder = DdFactorBuilder(
        tc, graph, values, obs, obsb, obs_sd, rs, rsb, sat, el, iu,
        ir_map, pos_ecef, key_pose, lever, amb_dict, track_indices,
        dd_epoch, prev_amb_values, skip_cp, slip_keys)
    return builder.run()
