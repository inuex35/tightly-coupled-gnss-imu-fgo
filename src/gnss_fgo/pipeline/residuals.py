"""Post-fit residual tests + DDPR sanity escalation (Stage C/D support)."""

import numpy as np
import gtsam
from ..integrity import recovery as _tc_state


def all_factor_residuals(tc, graph, estimate):
    """RMS of each factor type in the live graph evaluated at estimate."""
    # Buckets keyed by factor class substring → (sum_sq, count)
    buckets = {
        'DDPR':       [0.0, 0],   # DoubleDifferencePseudorange
        'DDCP':       [0.0, 0],   # DoubleDifferenceCarrierPhase
        'CombinedImu':[0.0, 0],
        'BetweenBias':[0.0, 0],
        'BetweenN':   [0.0, 0],   # BetweenFactorDouble (N chain)
        'PriorBias':  [0.0, 0],
        'PriorVel':   [0.0, 0],
        'PriorPose':  [0.0, 0],
        'PriorN':     [0.0, 0],   # PriorFactorDouble (ambiguity priors)
        'NHC':        [0.0, 0],   # via PriorFactorVector on Vel — distinguished by key
        'Other':      [0.0, 0],
    }
    custom_cp_local = set(tc._last_custom_ddcp_local)
    for i in range(graph.size()):
        fac = graph.at(i)
        if fac is None:
            continue
        try:
            err = float(fac.error(estimate))
        except RuntimeError:
            continue
        tname = type(fac).__name__
        if 'DoubleDifferencePseudorange' in tname:
            tag = 'DDPR'
        elif 'DoubleDifferenceCarrierPhase' in tname or i in custom_cp_local:
            tag = 'DDCP'
        elif 'CombinedImuFactor' in tname or 'ImuFactor' in tname:
            tag = 'CombinedImu'
        elif 'BetweenFactorConstantBias' in tname:
            tag = 'BetweenBias'
        elif 'BetweenFactorDouble' in tname:
            tag = 'BetweenN'
        elif 'PriorFactorConstantBias' in tname:
            tag = 'PriorBias'
        elif 'PriorFactorVector' in tname:
            tag = 'PriorVel'
        elif 'PriorFactorPose3' in tname:
            tag = 'PriorPose'
        elif 'PriorFactorDouble' in tname:
            tag = 'PriorN'
        else:
            tag = 'Other'
        buckets[tag][0] += err
        buckets[tag][1] += 1
    out = {}
    for tag, (sse, n) in buckets.items():
        if n == 0: continue
        out[tag] = (float(np.sqrt(2 * sse / n)), n)
    return out


def _choose_ddpr_iter_indices(tag_map, graph):
    """Decide whether to iterate just the tagged DDPR indices (per-epoch ``graph``) or the full graph. Returns ``(iter_indices, skip_type_check)``."""
    g3_size = graph.size()
    if not tag_map or max(tag_map.keys()) >= g3_size:
        return range(g3_size), False
    try:
        probe = graph.at(next(iter(tag_map.keys())))
        if probe is None or 'Pseudorange' not in type(probe).__name__:
            return range(g3_size), False
    except (RuntimeError, StopIteration):
        return range(g3_size), False
    return sorted(tag_map.keys()), True


def _ddpr_factor_error(fac, estimate):
    """Return the DDPR factor's chi-squared error at ``estimate``."""
    try:
        return fac.error(estimate)
    except RuntimeError:
        return None


def ddpr_res_at_fixed_pose(tc, graph, estimate, key_pose, xa):
    """Main-graph DDPR RMS with the pose moved to the LAMBDA-fixed
    antenna position ``xa[0:3]`` (rotation kept from ``estimate``).
    Returns None when the evaluation is impossible.
    """
    try:
        cur_pose = estimate.atPose3(key_pose)
        R_be = tc.ecef_T_nav.compose(cur_pose).rotation().matrix()
        body_ecef = (np.asarray(xa[0:3], dtype=float)
                     - R_be @ np.array(tc.lever_arm_tc))
        body_nav = tc.ecef_T_nav.transformTo(gtsam.Point3(*body_ecef))
        v_xa = gtsam.Values()
        v_xa.insert(key_pose, gtsam.Pose3(cur_pose.rotation(), body_nav))
        res, _ = main_ddpr_residuals(tc, graph, v_xa)
        return float(res)
    except (RuntimeError, ValueError, IndexError):
        return None


def main_ddpr_residuals(tc, graph, estimate, with_pairs=False):
    """DDPR residuals in the main graph at ``estimate``.

    UNIT CAVEAT (r5 #4): res = sqrt(2*err)*sigma_pr*sqrt(2) rescales
    the factor-whitened residual by the FLAT sigma_pr — but with
    varerr_enable=1 the factor sigma is elevation-dependent, so these
    are elevation-NORMALIZED residuals in pseudo-metres (zenith ~1.5x
    the true residual, 10-deg elevation ~0.36x), and every threshold
    documented as [m] (main_ddpr_res_thresh, per_sat_res_thresh,
    fde_pr/fde_cp, ...) actually cuts on that normalized scale. The
    elevation-weighted rejection is deliberate; the unit labels were
    not."""
    sigma_pr_m = tc.cfg.sigma_pr * np.sqrt(2)
    res_sq = []
    per_sat = {}
    pair_rows = [] if with_pairs else None
    # Build a fast lookup from factor-index to (ref_sat, j_sat, f).
    tag_map = {idx: (ref, j, f)
               for (idx, ref, j, f) in tc._last_ddpr_sat_tags}
    iter_indices, skip_type_check = _choose_ddpr_iter_indices(tag_map, graph)
    for i in iter_indices:
        fac = graph.at(i)
        if fac is None:
            continue
        if not skip_type_check and 'Pseudorange' not in type(fac).__name__:
            continue
        err = _ddpr_factor_error(fac, estimate)
        if err is None:
            continue
        res_m = float(np.sqrt(2.0 * max(err, 0.0)) * sigma_pr_m)
        res_sq.append(res_m * res_m)
        tag = tag_map.get(i)
        if tag is not None:
            ref, j, _ = tag
            # Attributing the pair MAX to the reference makes the ref
            # the per_sat argmax on every epoch — deliberate by
            # measurement, not an accident: it doubles as an
            # any-bad-pair-swaps-the-ref policy, and the honest
            # median attribution measured catastrophically worse.
            if res_m > per_sat.get(ref, 0.0):
                per_sat[ref] = res_m
            if res_m > per_sat.get(j, 0.0):
                per_sat[j] = res_m
            if with_pairs:
                pair_rows.append({
                    'ref': int(ref),
                    'sat': int(j),
                    'freq': int(tag[2]),
                    'res': float(res_m),
                })
    rms_all = float(np.sqrt(np.mean(res_sq))) if res_sq else 0.0
    if with_pairs:
        return rms_all, per_sat, pair_rows
    return rms_all, per_sat


def _fde_collect_residuals(tc, factors_all, fi_start, nf_total, estimate):
    """Helper: collect (fi, residual_in_meters) for current-epoch DD factors."""
    pr_entries = []
    cp_entries = []
    custom_cp_keys = set((tc._last_custom_ddcp_global or {}).keys())
    for fi in range(fi_start, nf_total):
        fac = factors_all.at(fi)
        if fac is None:
            continue
        fname = type(fac).__name__
        is_custom_cp = (
            fname == 'CustomFactor' and custom_cp_keys
            and tuple(int(k) for k in fac.keys()) in custom_cp_keys)
        try:
            err = fac.error(estimate)
        except RuntimeError:
            continue
        if 'Pseudorange' in fname:
            pr_entries.append(
                (fi, np.sqrt(2.0 * err) * tc.cfg.sigma_pr * np.sqrt(2)))
        elif 'CarrierPhase' in fname or is_custom_cp:
            cp_entries.append(
                (fi, np.sqrt(2.0 * err) * tc.cfg.sigma_cp * np.sqrt(2)))
    return pr_entries, cp_entries


def _fde_pick_rejects_iterative(tc, pr_entries, cp_entries):
    """Iterative FDE: pick the SINGLE largest outlier across PR and CP.
    No centering — sign-less magnitudes made it measured-worse."""
    best_d = 0.0
    best_fi = None
    for fi, res in pr_entries:
        d = abs(res)
        if d > tc.cfg.fde_pr and d > best_d:
            best_d, best_fi = d, fi
    for fi, res in cp_entries:
        d = abs(res)
        if d > tc.cfg.fde_cp and d > best_d:
            best_d, best_fi = d, fi
    return [best_fi] if best_fi is not None else []


def _fde_pick_rejects_single_pass(tc, pr_entries, cp_entries):
    """Single-pass FDE: collect every PR + CP entry above its threshold."""
    reject_fi = []
    for fi, res in pr_entries:
        if abs(res) > tc.cfg.fde_pr:
            reject_fi.append(fi)
    for fi, res in cp_entries:
        if abs(res) > tc.cfg.fde_cp:
            reject_fi.append(fi)
    return reject_fi


def _fde_reset_rejected_amb(tc, factors_all, reject_fi):
    """Purge the arcs behind every rejected CP factor: release holds
    (as seeds) and discard the float arcs so the next epoch re-seeds.

    NOT slip handling, despite the treatment being the same — one bad
    residual says nothing about integer continuity (the slip detectors
    own that call). The arc discard earns its keep differently: the
    float value is an accumulation of the same measurements FDE just
    distrusted, and keeping such arcs alive was measured well worse
    on the AR-heavy run. The purge is history hygiene, not a verdict
    on the integer."""
    custom_cp_meta = tc._last_custom_ddcp_global or {}
    for fi in reject_fi:
        fac = factors_all.at(fi)
        if fac is None:
            continue
        kt = (tuple(int(k) for k in fac.keys())
              if type(fac).__name__ == 'CustomFactor' else None)
        is_cp = ('CarrierPhase' in type(fac).__name__
                 or (kt is not None and kt in custom_cp_meta))
        if not is_cp:
            continue
        if kt is not None and kt in custom_cp_meta:
            ref_sat, j_sat, freq = custom_cp_meta[kt]
            for key in ((ref_sat, freq), (j_sat, freq)):
                st_hold = tc._sat_states.track.get(key)
                if st_hold is not None and st_hold.held_value is not None:
                    st_hold.release_hold(seed=True)
        # The float arcs stay: a rejected residual is evidence about
        # THIS epoch's measurement (already excluded above), not about
        # the integer's continuity — the slip detectors own that call.
        # Held integers were already handed over as seeds above.


def apply_fde(tc, graph, key_idx, nv, estimate, info, fi_start=None):
    """GICI-style Fault Detection and Exclusion.

    ``fi_start`` is the smoother's factor count before this epoch's
    insert: the epoch's factors occupy [fi_start, fi_start + G). The
    old ``nf_total - G`` derivation skipped the first M factors of the
    epoch whenever the same update appended M marginal containers.
    """
    max_iter = max(1, tc.cfg.fde_max_iter)
    iterative = max_iter > 1
    total_rejected = 0
    for _it in range(max_iter):
        factors_all = tc.isam2.getFactors()
        nf_total = factors_all.size()
        # Current epoch's factors only: rejecting history
        # un-anchors the trajectory (measured km-scale divergence).
        if fi_start is None:
            fi_start = max(0, nf_total - graph.size())
        info['fde_slot_shift'] = (nf_total - graph.size()) - fi_start
        fi_end = min(nf_total, fi_start + graph.size())
        pr_entries, cp_entries = _fde_collect_residuals(
            tc, factors_all, fi_start, fi_end, estimate)
        if iterative:
            reject_fi = _fde_pick_rejects_iterative(tc, pr_entries, cp_entries)
            if not reject_fi:
                break
            # Same over-rejection safeguard as single-pass.
            if total_rejected + 1 > tc.cfg.fde_max_frac * max(1, nv):
                info['fde_skipped'] = total_rejected + 1
                _tc_state.trigger_cp_hold(tc, 'fde_safeguard', info,
                                     value=info['fde_skipped'],
                                     skip_if_active=True)
                break
        else:
            reject_fi = _fde_pick_rejects_single_pass(tc, pr_entries, cp_entries)
            if not reject_fi:
                break
            if len(reject_fi) > tc.cfg.fde_max_frac * max(1, nv):
                info['fde_skipped'] = len(reject_fi)
                _tc_state.trigger_cp_hold(tc, 'fde_safeguard', info,
                                     value=info['fde_skipped'],
                                     skip_if_active=True)
                return estimate

        _fde_reset_rejected_amb(tc, factors_all, reject_fi)
        total_rejected += len(reject_fi)
        try:
            tc.isam2.update(
                gtsam.NonlinearFactorGraph(), gtsam.Values(),
                gtsam.FixedLagSmootherKeyTimestampMap(), reject_fi)
            est_fde = tc.isam2.calculateEstimate()
            if est_fde.exists(tc.Xpose(key_idx)):
                estimate = est_fde
        except (RuntimeError, IndexError):
            break
        # Single-pass: one removal batch, done.
        if not iterative:
            info['fde_reject'] = total_rejected
            return estimate
    if total_rejected:
        info['fde_reject'] = total_rejected
    return estimate
