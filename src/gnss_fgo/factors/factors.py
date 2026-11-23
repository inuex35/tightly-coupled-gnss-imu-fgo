"""Pure C++ DD pseudorange / carrier-phase factor builder with persistent arcs and GTSAM Robust noise."""

import numpy as np
import gtsam
from cssrlib.gnss import uGNSS, SAT_SYS_ARR

from ..utils.geometry import is_bds_geo as _is_bds_geo
from ..utils import sorted_sys_ids
from . import prefit as _tc_prefit
from .factors_support import get_wavelengths
from .amb_seed import init_dd_ambiguity_priors as _init_dd_ambiguity_priors


def _dd_noise(tc, sigma, code):
    """Build 1-D Gaussian or GTSAM native Robust noise model for DDPR/DDCP."""
    base = tc._noise1(sigma)
    kind = str(tc.cfg.dd_robust_kind or 'none').lower()
    k = float(tc.cfg.ddpr_robust_k if code else tc.cfg.ddcp_robust_k)
    if kind == 'none' or k <= 0.0:
        return base
    if kind == 'huber':
        est = gtsam.noiseModel.mEstimator.Huber.Create(k)
    elif kind == 'dcs':
        est = gtsam.noiseModel.mEstimator.DCS.Create(k)
    elif kind == 'cauchy':
        est = gtsam.noiseModel.mEstimator.Cauchy.Create(k)
    else:
        raise ValueError(f"Unknown dd_robust_kind: {kind!r}")
    return gtsam.noiseModel.Robust.Create(est, base)


class DdFactorBuilder:
    """Build all C++ DDPR + DDCP factors for one epoch using persistent arc keys."""

    def __init__(self, tc, graph, values, obs, obsb, obs_sd,
                 rs, rsb, sat, el, iu, ir_map,
                 key_pose, lever, amb_dict,
                 existing_estimate=None):
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
        self.key_pose = key_pose
        self.lever = lever
        self.amb_dict = amb_dict
        self.existing_estimate = existing_estimate

        self.nf = tc.nav.nf
        self.ns = len(sat)
        self.el_min_rad = np.radians(max(1.0, tc.cfg.el_mask_deg))
        self.dt_s = float(tc._epoch_dt)

        tc._last_ddpr_sat_tags = []
        tc._ar_cp_visible_sf = set()

        self.nv = 0
        self.new_amb = {}

    def _select_ref_for_system(self, sys_id, idx_sys):
        ref_idx, ref_sat = _tc_prefit.pick_ref_sat_idx(
            self.tc, sys_id, idx_sys, self.sat, self.el)
        self.tc.current_epoch.ref_sats[sys_id] = ref_sat
        return ref_idx, ref_sat

    def _sat_pts(self, idx, sat_id):
        pt = gtsam.Point3(*self.rs[self.iu[idx], :3])
        base_pt = (gtsam.Point3(*self.rsb[self.ir_map[sat_id], :3])
                   if sat_id in self.ir_map else pt)
        return pt, base_pt

    def _pair_sigma(self, code, el_ref_rad, el_j_rad):
        el_pair = max(min(el_ref_rad, el_j_rad), self.el_min_rad)
        return _tc_prefit.varerr_dd_sigma(self.tc, code, el_pair, self.dt_s)

    def _persistent_key(self, sat_id, freq):
        """Return persistent arc key for (sat_id, freq); bump gen if marginalized out."""
        tc = self.tc
        st = tc._sat_states.get(sat_id, freq)
        if st.amb_key is not None:
            if (self.existing_estimate is None
                    or self.existing_estimate.exists(st.amb_key)):
                return st.amb_key
            # Marginalized out of FLS window: start a fresh arc generation.
            st.amb_key = None
            st.amb_gen += 1
        return tc.N(sat_id, freq, st.amb_gen)

    def _build_pr_for_pair(self, ref_idx, j_idx, ref_sat, j_sat, f, sat_pts):
        obs, obsb, iu, ir_map = self.obs, self.obsb, self.iu, self.ir_map
        pr_ref_r = obs.P[iu[ref_idx], f]
        pr_ref_b = obsb.P[ir_map[ref_sat], f] if ref_sat in ir_map else 0
        pr_j_r = obs.P[iu[j_idx], f]
        pr_j_b = obsb.P[ir_map[j_sat], f] if j_sat in ir_map else 0
        if pr_ref_r == 0 or pr_ref_b == 0 or pr_j_r == 0 or pr_j_b == 0:
            return None
        sigma = self._pair_sigma(1, self.el[ref_idx], self.el[j_idx])
        pr_noise = _dd_noise(self.tc, sigma, code=1)
        ref_pt, j_pt, ref_base_pt, j_base_pt = sat_pts
        self.tc._last_ddpr_sat_tags.append(
            (self.graph.size(), ref_sat, j_sat, f))
        self.graph.add(gtsam.DoubleDifferencePseudorangeFactorArm(
            self.key_pose, pr_ref_r, pr_ref_b, pr_j_r, pr_j_b,
            ref_pt, j_pt, ref_base_pt, j_base_pt,
            self.tc.base_pt, self.lever, self.tc.ecef_T_nav, pr_noise))
        return pr_ref_r, pr_ref_b, pr_j_r, pr_j_b

    def _build_cp_for_pair(self, sys_id, ref_idx, j_idx, ref_sat, j_sat, f,
                           lams, sat_pts, pr_obs):
        if sys_id == uGNSS.GLO or f >= len(lams) or lams[f] <= 0:
            return 0
        lam = lams[f]
        obs, obsb, iu, ir_map = self.obs, self.obsb, self.iu, self.ir_map
        cp_ref_r = obs.L[iu[ref_idx], f] * lam
        cp_ref_b = obsb.L[ir_map[ref_sat], f] * lam if ref_sat in ir_map else 0
        cp_j_r = obs.L[iu[j_idx], f] * lam
        cp_j_b = obsb.L[ir_map[j_sat], f] * lam if j_sat in ir_map else 0
        if cp_ref_r == 0 or cp_ref_b == 0 or cp_j_r == 0 or cp_j_b == 0:
            return 0

        tc = self.tc
        tc._ar_cp_visible_sf.add((int(ref_sat), int(f)))
        tc._ar_cp_visible_sf.add((int(j_sat), int(f)))

        key_n_ref = self._persistent_key(ref_sat, f)
        key_n_j = self._persistent_key(j_sat, f)
        pr_ref_r, pr_ref_b, pr_j_r, pr_j_b = pr_obs
        _init_dd_ambiguity_priors(
            tc, self.graph, self.values, self.amb_dict, self.new_amb,
            self.existing_estimate, f, lam,
            ((ref_sat, key_n_ref, cp_ref_r, cp_ref_b, pr_ref_r, pr_ref_b),
             (j_sat, key_n_j, cp_j_r, cp_j_b, pr_j_r, pr_j_b)))

        sigma = self._pair_sigma(0, self.el[ref_idx], self.el[j_idx])
        cp_noise = _dd_noise(tc, sigma, code=0)
        ref_pt, j_pt, ref_base_pt, j_base_pt = sat_pts
        self.graph.add(gtsam.DoubleDifferenceCarrierPhaseFactorArm(
            self.key_pose, key_n_ref, key_n_j,
            cp_ref_r, cp_ref_b, cp_j_r, cp_j_b,
            ref_pt, j_pt, ref_base_pt, j_base_pt, tc.base_pt, lam,
            self.lever, tc.ecef_T_nav, cp_noise))
        return 1

    def run(self):
        nv = 0
        sat_sys_for_obs = SAT_SYS_ARR[
            np.asarray(self.sat[:self.ns], dtype=np.int32)]
        for sys_id in sorted_sys_ids(self.obs_sd.sig):
            idx_sys = np.where(sat_sys_for_obs == int(sys_id))[0].tolist()
            if len(idx_sys) < 2:
                continue
            ref_idx, ref_sat = self._select_ref_for_system(sys_id, idx_sys)
            lams = get_wavelengths(self.tc, self.obs_sd, ref_sat)
            ref_pt, ref_base_pt = self._sat_pts(ref_idx, ref_sat)
            for j_idx in idx_sys:
                if j_idx == ref_idx:
                    continue
                j_sat = self.sat[j_idx]
                if SAT_SYS_ARR[j_sat] == uGNSS.BDS and _is_bds_geo(j_sat):
                    continue
                j_pt, j_base_pt = self._sat_pts(j_idx, j_sat)
                sat_pts = (ref_pt, j_pt, ref_base_pt, j_base_pt)
                for f in range(self.nf):
                    pr_obs = self._build_pr_for_pair(
                        ref_idx, j_idx, ref_sat, j_sat, f, sat_pts)
                    if pr_obs is None:
                        continue
                    nv += 1
                    nv += self._build_cp_for_pair(
                        sys_id, ref_idx, j_idx, ref_sat, j_sat, f,
                        lams, sat_pts, pr_obs)
        self.amb_dict.update(self.new_amb)
        self.nv = nv
        return nv


def build_dd_factors(tc, graph, values, obs, obsb, obs_sd,
                     rs, rsb, sat, el, iu, ir_map,
                     key_pose, lever, amb_dict,
                     existing_estimate=None):
    """Build pure C++ DDPR + DDCP factors on persistent ambiguity arc keys."""
    builder = DdFactorBuilder(
        tc, graph, values, obs, obsb, obs_sd, rs, rsb, sat, el, iu,
        ir_map, key_pose, lever, amb_dict,
        existing_estimate=existing_estimate)
    return builder.run()
