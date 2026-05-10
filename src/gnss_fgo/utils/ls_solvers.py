"""Stand-alone least-squares solvers over GNSS observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import gtsam

from cssrlib.gnss import rCST, sat2prn, uGNSS

from .pipeline_helpers import sorted_sys_ids
from .robust import maybe_robust


OMGE = 7.2921151467e-5  # Earth rotation rate [rad/s] (WGS-84)



def _doppler_build_rows(obs, obs_sd, rs, vs, iu, sat, pos_ecef,
                         nav_nf, get_wavelengths, v_pred):
    """Build the per-sat (e_los | -1) rows and the corresponding measurement"""
    CLIGHT = rCST.CLIGHT
    rows = []
    rhs = []
    for si, i_obs in enumerate(iu):
        s = sat[si]
        sys_id = sat2prn(s)[0]
        if sys_id == uGNSS.GLO:
            continue
        # Pick first freq with Doppler observation
        lam = None
        D_obs = None
        for f in range(nav_nf):
            if f >= obs.D.shape[1]:
                break
            if obs.D[i_obs, f] != 0.0:
                lams = get_wavelengths(obs_sd, s)
                if f < len(lams) and lams[f] > 0:
                    lam = lams[f]
                    D_obs = obs.D[i_obs, f]
                    break
        if D_obs is None:
            continue
        p_sat = rs[i_obs, :3]
        v_sat = vs[i_obs, :3] if vs.shape[1] >= 3 else np.zeros(3)
        d_vec = p_sat - pos_ecef
        rho = np.linalg.norm(d_vec)
        if rho < 1.0:
            continue
        e_los = d_vec / rho
        # Earth-rotation (Sagnac-like for range rate) correction.
        omge_corr = OMGE / CLIGHT * (
            v_sat[1] * pos_ecef[0] + p_sat[1] * v_pred[0]
            - v_sat[0] * pos_ecef[1] - p_sat[0] * v_pred[1])
        meas_m_s = lam * D_obs + float(np.dot(e_los, v_sat)) - omge_corr
        rows.append([e_los[0], e_los[1], e_los[2], -1.0])
        rhs.append(meas_m_s)
    return np.asarray(rows), np.asarray(rhs)


def _doppler_iterative_huber(A, b, outlier_thresh_m_s):
    """Iterative single-largest-outlier reject around a 3D + clock-drift LS,"""
    try:
        x, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None, None
    keep = np.ones(len(b), dtype=bool)
    for _ in range(3):
        r = A[keep] @ x - b[keep]
        mad = float(np.median(np.abs(r))) if len(r) else 0.0
        thresh = max(outlier_thresh_m_s, 3.0 * mad)
        idx_keep = np.where(keep)[0]
        worst_i = np.argmax(np.abs(r))
        if abs(r[worst_i]) <= thresh or keep.sum() <= 5:
            break
        keep[idx_keep[worst_i]] = False
        try:
            x, *_ = np.linalg.lstsq(A[keep], b[keep], rcond=None)
        except np.linalg.LinAlgError:
            return None, None
    return x, keep


def doppler_velocity_ls(
    obs,
    obs_sd,
    rs: np.ndarray,
    vs: np.ndarray,
    iu,
    sat,
    pos_ecef: np.ndarray,
    nav_nf: int,
    get_wavelengths: Callable,
    vel_pred_ecef: Optional[np.ndarray] = None,
    outlier_thresh_m_s: float = 3.0):
    """GICI-style Doppler → rover velocity LS (ECEF)."""
    v_pred = (np.asarray(vel_pred_ecef) if vel_pred_ecef is not None
              else np.zeros(3))
    A, b = _doppler_build_rows(obs, obs_sd, rs, vs, iu, sat, pos_ecef,
                                nav_nf, get_wavelengths, v_pred)
    n_rows = len(b)
    if n_rows < 4:
        return None, None, False, 0.0, n_rows
    x, keep = _doppler_iterative_huber(A, b, outlier_thresh_m_s)
    if x is None:
        return None, None, False, 0.0, n_rows
    r_final = A[keep] @ x - b[keep]
    res_rms = float(np.sqrt(np.mean(r_final ** 2))) if len(r_final) else 0.0
    return x[:3], float(x[3]), True, res_rms, int(keep.sum())



@dataclass
class DDPRContext:
    """Immutable-ish bundle of constants + callables for DDPR LS solve."""
    R_enu2ecef: np.ndarray
    base_ecef: np.ndarray
    base_pt: gtsam.Point3
    ecef_T_nav: gtsam.Pose3
    lever: gtsam.Point3
    nav_nf: int
    sigma_pr: float
    huber_pr: float
    pr_robust_kind: str
    fde_pr: float
    pick_ref_sat_idx: Callable  # (sys_id, idx_sys, sat, el) -> (ri, ref_sat)


def _ddpr_build_specs(obs, obsb, obs_sd, rs, rsb, sat, el, iu, ir_map, ctx):
    """Iterate every (sys, ref-sat, j-sat, freq) triple with all four PR"""
    ns = len(sat)
    specs = []        # (a1, a2, a3, a4, sr, st, srb, stb)
    spec_sats = []    # (j_sat, f)
    for sys_id in sorted_sys_ids(obs_sd.sig):
        idx_sys = [i for i in range(ns)
                   if sat2prn(sat[i])[0] == sys_id]
        if len(idx_sys) < 2:
            continue
        ri, ref_sat = ctx.pick_ref_sat_idx(sys_id, idx_sys, sat, el)
        for ji in idx_sys:
            if ji == ri:
                continue
            j_sat = sat[ji]
            sr = gtsam.Point3(*rs[iu[ri], :3])
            st = gtsam.Point3(*rs[iu[ji], :3])
            srb = (gtsam.Point3(*rsb[ir_map[ref_sat], :3])
                   if ref_sat in ir_map else sr)
            stb = (gtsam.Point3(*rsb[ir_map[j_sat], :3])
                   if j_sat in ir_map else st)
            for f in range(ctx.nav_nf):
                a1 = obs.P[iu[ri], f]
                a2 = (obsb.P[ir_map[ref_sat], f]
                      if ref_sat in ir_map else 0)
                a3 = obs.P[iu[ji], f]
                a4 = obsb.P[ir_map[j_sat], f] if j_sat in ir_map else 0
                if a1 == 0 or a2 == 0 or a3 == 0 or a4 == 0:
                    continue
                specs.append((a1, a2, a3, a4, sr, st, srb, stb))
                spec_sats.append((j_sat, f))
    return specs, spec_sats


def _ddpr_solve_with_fde(specs, spec_sats, key, pose_init, ctx):
    """Iterative LM solve with single-pass FDE outlier rejection (max 3"""
    sigma_pr_m = ctx.sigma_pr * np.sqrt(2)
    pr_base = gtsam.noiseModel.Isotropic.Sigma(1, sigma_pr_m)
    pr_noise = maybe_robust(pr_base, ctx.huber_pr, kind=ctx.pr_robust_kind)
    pose_prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([0.05, 0.05, 0.1, 50.0, 50.0, 50.0]))

    active = list(range(len(specs)))
    est = None
    per_sat_max = {}
    res_kept = []
    for _fde_iter in range(3):
        g = gtsam.NonlinearFactorGraph()
        v = gtsam.Values()
        v.insert(key, pose_init)
        g.addPriorPose3(key, pose_init, pose_prior_noise)
        for idx in active:
            a1, a2, a3, a4, sr, st, srb, stb = specs[idx]
            g.add(gtsam.DoubleDifferencePseudorangeFactorArm(
                key, a1, a2, a3, a4,
                sr, st, srb, stb, ctx.base_pt, ctx.lever,
                ctx.ecef_T_nav, pr_noise))
        try:
            params = gtsam.LevenbergMarquardtParams()
            params.setMaxIterations(10)
            est = gtsam.LevenbergMarquardtOptimizer(g, v, params).optimize()
        except RuntimeError:
            return None, active, {}, []

        # Rebuild with non-robust noise to compute raw residuals
        g_eval = gtsam.NonlinearFactorGraph()
        for idx in active:
            a1, a2, a3, a4, sr, st, srb, stb = specs[idx]
            g_eval.add(gtsam.DoubleDifferencePseudorangeFactorArm(
                key, a1, a2, a3, a4,
                sr, st, srb, stb, ctx.base_pt, ctx.lever,
                ctx.ecef_T_nav, pr_base))

        new_active = []
        dropped = 0
        res_kept = []
        per_sat_max = {}
        for j, idx in enumerate(active):
            err = g_eval.at(j).error(est)
            res_m = np.sqrt(max(err, 0) * 2.0) * sigma_pr_m
            if res_m > ctx.fde_pr and len(active) - dropped > 4:
                dropped += 1
                continue
            new_active.append(idx)
            res_kept.append(res_m)
            s, _f = spec_sats[idx]
            if res_m > per_sat_max.get(s, 0.0):
                per_sat_max[s] = res_m
        if dropped == 0:
            break
        active = new_active
    return est, active, per_sat_max, res_kept


def ddpr_only_position(obs, obsb, obs_sd, rs, rsb, sat, el, iu, ir_map,
                        pose_init: gtsam.Pose3,
                        ctx: DDPRContext):
    """Standalone DDPR-only LS solve with iterative outlier rejection."""
    specs, spec_sats = _ddpr_build_specs(
        obs, obsb, obs_sd, rs, rsb, sat, el, iu, ir_map, ctx)
    if len(specs) < 4:
        return None, 0, float('inf'), {}, set()

    key = gtsam.symbol('z', 0)
    est, active, per_sat_max, res_kept = _ddpr_solve_with_fde(
        specs, spec_sats, key, pose_init, ctx)

    rejected_sats = {spec_sats[idx][0]
                     for idx in range(len(specs))
                     if idx not in active}

    if est is None or len(active) < 4:
        return None, len(active), float('inf'), per_sat_max, rejected_sats
    pose_out = est.atPose3(key)
    ecef_out = ctx.R_enu2ecef @ np.array(pose_out.translation()) + ctx.base_ecef
    res_rms = (float(np.sqrt(np.mean(np.square(res_kept))))
               if res_kept else 0.0)
    return ecef_out, len(active), res_rms, per_sat_max, rejected_sats
