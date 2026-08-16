"""LAMBDA ambiguity resolution for the TC graph."""

import os

import numpy as np
import gtsam

from .ambiguity_resolver import AmbiguityResolver

from ..preprocess import sat_quality as _satq
from ..utils import sorted_amb_items
from ..validation import postfit as _tc_postfit


def _ratio_from_last_lambda(tc):
    s0 = float(tc._last_s0)
    s1 = float(tc._last_s1)
    if s0 <= 0.0:
        return 0.0
    return s1 / s0


def _rank_subset_drop_sats(tc, sat, el, amb_dict):
    """Rank candidate satellites to drop for subset AR fallback."""
    seen = set()
    cppr = getattr(tc, 'rejc_cp_pr', {}) or {}
    per_sat = tc._last_main_ddpr_per_sat or {}
    sat_el = {}
    for i, s in enumerate(sat):
        sat_el[int(s)] = max(sat_el.get(int(s), 0.0), float(el[i]))
    rows = []
    for (s, _f), _k in sorted_amb_items(amb_dict):
        s = int(s)
        if s in seen:
            continue
        seen.add(s)
        cppr_max = 0
        for f in range(tc.nav.nf):
            cppr_max = max(cppr_max, int(cppr.get((s, f), 0)))
        rows.append((
            float(per_sat.get(s, 0.0)),
            cppr_max,
            -float(sat_el.get(s, 0.0)),
            s))
    rows.sort(reverse=True)
    max_candidates = max(0, int(tc.cfg.subset_ar_max_candidates))
    return [s for *_rest, s in rows[:max_candidates]]


def _resolve_native(tc, sat_list):
    """AR straight off the smoother, without the cssrlib nav round-trip.

    Assembles the float ambiguities and their joint covariance from ISAM2 --
    the same numbers ``write_marginals`` transcribes into ``nav.x`` / ``nav.P``
    -- and hands them to :class:`AmbiguityResolver`. Held ambiguities keep the
    pinned value and ``varholdamb`` variance the fix-and-hold policy gives
    them; that policy stays with the caller, the resolver only fixes integers.

    Returns the ``(nb, xa)`` pair the rest of the AR path expects, with ``xa``
    carrying the fixed ambiguities and the position corrected by the usual
    Kalman gain from the (pose, ambiguity) cross-covariance.
    """
    # Phase 1 runs on its own smoother with its own ambiguity keys; this
    # resolver reads the Phase-2 graph, so leave Phase 1 to cssrlib.
    smoother = getattr(tc, 'isam2', None)
    if tc.phase != 2 or smoother is None:
        return None
    isam2 = smoother.getISAM2()
    if isam2 is None:
        return None
    est = smoother.calculateEstimate()

    keys, values, held_var = [], {}, {}
    key_of = dict(sorted_amb_items(tc._sat_states.amb_keys_dict()))
    # Held ambiguities first: fix-and-hold pins them at varholdamb and that
    # pinning wins over the smoother's own spread, exactly as write_marginals
    # overwrites nav.P for them. Taking the graph covariance instead leaves
    # the double differences far looser than cssrlib sees them -- measured on
    # tokyo, the best residual came out at 890 against 610 and the ratio fell
    # short of the threshold that the cssrlib path cleared.
    for (s, f), value in tc._sat_states.held_items():
        sf = (int(s), int(f))
        if tc.nav.vsat[int(s) - 1, int(f)] != 1:
            continue
        keys.append(sf)
        values[sf] = float(value)
        held_var[sf] = max(float(tc.cfg.varholdamb), 1e-9)
    # Selection is nav.vsat's job, exactly as in ddidx: the satellite list is
    # a presence check there, not a filter. Intersecting the two drops every
    # (sat, band) whose satellite left the list -- on a subset retry that took
    # the double differences from fifteen down to four and turned fixes the
    # cssrlib path accepted at ratio 12.8 into no fix at all.
    for (s, f), k in sorted_amb_items(key_of):
        sf = (int(s), int(f))
        if sf in values:
            continue
        if est.exists(k) and tc.nav.vsat[s - 1, f] == 1:
            keys.append(sf)
            values[sf] = est.atDouble(k)
    if len(keys) < 2:
        return None

    free = [sf for sf in keys if sf not in held_var]
    key_pose = getattr(tc, '_ar_key_pose', None)
    if key_pose is None:
        return None
    kv = gtsam.KeyVector()
    kv.append(key_pose)
    for sf in free:
        kv.append(key_of[sf])
    try:
        jm = isam2.jointMarginalCovariance(kv)
    except (RuntimeError, IndexError):
        return None

    n = len(keys)
    cov = np.zeros((n, n))
    cross = np.zeros((3, n))          # (position, ambiguity), ENU
    for i, a in enumerate(keys):
        if a in held_var:
            cov[i, i] = held_var[a]
            continue
        for j, b in enumerate(keys):
            if b in held_var:
                continue
            cov[i, j] = jm.at(key_of[a], key_of[b])[0, 0]
        cross[:, i] = jm.at(key_pose, key_of[a])[3:6, 0]
    if not (np.all(np.isfinite(cov)) and np.all(np.isfinite(cross))):
        return None

    resolver = AmbiguityResolver(
        thresar=float(tc.nav.thresar), parmode=int(tc.nav.parmode),
        par_p0=float(tc.nav.par_P0),
        el_mask=float(getattr(tc.nav, 'elmaskar', 0.0)))
    el = {int(s): float(tc.nav.el[int(s) - 1]) for s, _ in keys}
    res = resolver.resolve(values, cov, keys, el)
    # Diagnostics elsewhere still read the stashed ratio pair.
    tc._last_s0, tc._last_s1 = res.s0, res.s1
    if res.nb <= 0:
        return 0, tc.nav.x.copy()

    # Downstream reads state that cssrlib's resamb_lambda leaves behind:
    # nav.fix marks which satellites entered the double differences (the hold
    # policy and restamb both consult it) and nav.xa carries the fixed
    # non-ambiguity state. Set them here or the two paths diverge a few
    # epochs later even when LAMBDA agreed.
    tc.nav.fix[:, :] = 0
    for ref, tgt in res.pairs:
        tc.nav.fix[ref[0] - 1, ref[1]] = 2
        tc.nav.fix[tgt[0] - 1, tgt[1]] = 2

    xa = tc.nav.x.copy()
    for sf, value in res.fixed.items():
        xa[tc.IB(sf[0], sf[1], tc.nav.na)] = value
    # Position update: x_fix = x - Qab Qb^-1 (y - b), in the DD space.
    idx = {sf: i for i, sf in enumerate(keys)}
    D = np.zeros((len(res.pairs), n))
    for row, (ref, tgt) in enumerate(res.pairs):
        D[row, idx[ref]] = 1.0
        D[row, idx[tgt]] = -1.0
    x_float = np.array([values[sf] for sf in keys])
    x_fixed = np.array([res.fixed.get(sf, values[sf]) for sf in keys])
    Qb = D @ cov @ D.T
    Qab = cross @ cov @ D.T
    try:
        gain = Qab @ np.linalg.inv(Qb)
    except np.linalg.LinAlgError:
        return res.nb, xa
    d_enu = gain @ (D @ (x_float - x_fixed))
    xa[0:3] = tc.nav.x[0:3] + tc.R_enu2ecef @ d_enu
    tc.nav.xa = tc.nav.x.copy()
    tc.nav.xa[0:tc.nav.na] = xa[0:tc.nav.na]
    return res.nb, xa


def _resolve_native_retry(tc, sat_list):
    """Native AR with the one-satellite retry cssrlib does inside resamb_lambda.

    RTKLIB (and cssrlib after it) retries a failed fix once with a single
    satellite excluded, chosen round-robin so a different one is tried each
    epoch, and prefers a freshly-acquired satellite when its arrival is what
    dropped the ratio. Both live inside ``resamb_lambda`` there, reached by
    zeroing ``nav.vsat`` and calling back in; here the resolver takes an
    explicit satellite list, so a retry is just a shorter list.
    """
    out = _resolve_native(tc, sat_list)
    if out is None:
        return None
    nb, xa = out
    ratio = 0.0 if tc._last_s0 <= 0 else tc._last_s1 / tc._last_s0
    if nb > 0:
        tc._native_prev_ratio = ratio
        tc._native_excsat = 0
        return nb, xa
    if len(sat_list) < int(tc.nav.minfixsats) + 1:
        return 0, xa

    order = [int(s) for s in sat_list]
    try:
        start = order.index(int(getattr(tc, '_native_excsat', 0) or 0)) + 1
    except ValueError:
        start = 0
    order = order[start:] + order[:start]

    exc = 0
    prev_ratio = float(getattr(tc, '_native_prev_ratio', 0.0) or 0.0)
    if (tc.nav.arfilter and ratio < tc.nav.thresar and prev_ratio > 0.0
            and ratio < 1.1 * prev_ratio):
        # A satellite that has only just been locked is the likely culprit.
        for s in order:
            lock = tc.nav.lock[s - 1, :] if hasattr(tc.nav, 'lock') else None
            if lock is not None and any(0 < lock[f] <= 1
                                        for f in range(tc.nav.nf)):
                exc = s
                break
    if exc == 0:
        exc = order[0] if order else 0
    if exc == 0:
        return 0, xa

    out2 = _resolve_native(tc, [s for s in sat_list if s != exc])
    if out2 is None:
        return 0, xa
    nb2, xa2 = out2
    tc._native_prev_ratio = (0.0 if tc._last_s0 <= 0
                             else tc._last_s1 / tc._last_s0)
    tc._native_excsat = exc if nb2 > 0 else 0
    return (nb2, xa2) if nb2 > 0 else (0, xa)


def _run_single_ar_attempt(tc, sat, sat_exclude=None, restore_state=True):
    """Run one AR attempt, optionally excluding one or more satellites."""
    sat_list = [int(s) for s in sat]
    vsat_snapshot = tc.nav.vsat.copy()
    fix_snapshot = tc.nav.fix.copy()
    try:
        if sat_exclude is not None:
            if isinstance(sat_exclude, (int, np.integer)):
                excl = {int(sat_exclude)}
            else:
                excl = {int(x) for x in sat_exclude}
            for s in excl:
                if 1 <= s <= tc.nav.vsat.shape[0]:
                    tc.nav.vsat[s - 1, :] = 0
            sat_list = [s for s in sat_list if s not in excl]
        if tc.cfg.ar_native_resolver:
            native = (_resolve_native_retry(tc, sat_list) if tc.cfg.rtklib_mode
                      else _resolve_native(tc, sat_list))
            if native is not None:
                return native
        if tc.cfg.rtklib_mode and hasattr(tc, 'resamb_lambda_rtklib'):
            return tc.resamb_lambda_rtklib(sat_list)
        return tc.resamb_lambda(sat_list, tc.nav.parmode, tc.nav.par_P0)
    finally:
        if restore_state:
            tc.nav.vsat[:, :] = vsat_snapshot
            tc.nav.fix[:, :] = fix_snapshot


def _try_subset_ar(tc, sat, el, amb_dict):
    """Retry AR with one (or more) candidate bad satellites removed."""
    from itertools import combinations
    mres_max = float(getattr(tc.cfg, 'subset_ar_max_mres_m', 0.0) or 0.0)
    if mres_max > 0.0:
        last_mres = float(tc._last_main_ddpr_res or 0.0)
        if last_mres > mres_max:
            tc._ar_subset_debug = {
                'candidates': 0, 'used': False,
                'skip_reason': 'mres_gate', 'mres': last_mres,
            }
            return 0, None
    dirty_max = int(getattr(tc.cfg, 'subset_ar_max_dirty_sats', 0) or 0)
    if dirty_max > 0:
        per_sat = tc._last_main_ddpr_per_sat or {}
        dirty_thr = float(getattr(tc.cfg, 'subset_ar_dirty_sat_res_m', 1.0))
        dirty_n = sum(1 for v in per_sat.values()
                      if float(v or 0.0) > dirty_thr)
        if dirty_n > dirty_max:
            tc._ar_subset_debug = {
                'candidates': 0, 'used': False,
                'skip_reason': 'dirty_gate', 'dirty_n': dirty_n,
            }
            return 0, None
    min_nb = max(1, int(tc.cfg.subset_ar_min_nb))
    max_drop = max(1, int(getattr(tc.cfg, 'subset_ar_max_drop', 1) or 1))
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
            nb, xa = _run_single_ar_attempt(
                tc, sat, sat_exclude=drop_combo)
            ratio = _ratio_from_last_lambda(tc)
            if nb < min_nb or xa is None:
                continue
            score = (float(ratio), int(nb), -k)  # smaller k preferred on ties
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
    nb, xa = _run_single_ar_attempt(
        tc, sat, sat_exclude=best['drop_sats'], restore_state=False)
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
        'ratio': _ratio_from_last_lambda(tc),
    }
    return nb, xa


def should_skip_ar_precheck(tc):
    """Pre-AR fast skip — same context as ``_ar_context_reject`` minus"""
    main_res = float(tc._cached_ddpr_res_pre
                     or tc._last_main_ddpr_res or 0.0)
    per_sat = tc._last_main_ddpr_per_sat or {}
    worst_res = float(max(per_sat.values())) if per_sat else 0.0
    cp_hold_active = int(tc._recov_cp_hold or 0) > 0
    ddpr_bad_active = int(tc._ddpr_bad_count or 0) > 0

    skip = False
    if (bool(tc.cfg.ar_context_reject_during_cp_hold)
            and cp_hold_active):
        skip = True
    if (bool(tc.cfg.ar_context_reject_during_ddpr_bad)
            and ddpr_bad_active):
        skip = True
    if main_res > float(tc.cfg.ar_context_main_ddpr_max):
        skip = True
    if worst_res > float(tc.cfg.ar_context_worst_sat_max):
        skip = True
    if not skip:
        return False, None
    return True, {
        'main_ddpr_res': main_res,
        'worst_sat_res': worst_res,
        'cp_hold_active': cp_hold_active,
        'ddpr_bad_active': ddpr_bad_active,
    }


def _ar_context_reject(tc, nb):
    """Reject fragile AR fixes in burst-like contexts before hold/anchor."""
    nb = int(nb)
    if nb <= 0:
        return False, None
    main_res = float(tc._cached_ddpr_res_pre
                     or tc._last_main_ddpr_res or 0.0)
    per_sat = tc._last_main_ddpr_per_sat or {}
    worst_res = float(max(per_sat.values())) if per_sat else 0.0
    cp_hold_active = int(tc._recov_cp_hold or 0) > 0
    ddpr_bad_active = int(tc._ddpr_bad_count or 0) > 0

    burst_like = False
    if (bool(tc.cfg.ar_context_reject_during_cp_hold)
            and cp_hold_active):
        burst_like = True
    if (bool(tc.cfg.ar_context_reject_during_ddpr_bad)
            and ddpr_bad_active):
        burst_like = True

    if main_res > float(tc.cfg.ar_context_main_ddpr_max):
        burst_like = True
    if worst_res > float(tc.cfg.ar_context_worst_sat_max):
        burst_like = True

    if burst_like and nb <= int(tc.cfg.ar_context_nb_max):
        return True, {
            'nb': nb,
            'main_ddpr_res': main_res,
            'worst_sat_res': worst_res,
            'cp_hold_active': cp_hold_active,
            'ddpr_bad_active': ddpr_bad_active,
        }
    return False, None


def write_marginals(tc, factors, estimate, key_pose, amb_dict):
    """Write GTSAM Marginals to nav.P with ENU->ECEF rotation."""
    # The native resolver reads the same marginals straight from ISAM2 and
    # needs the very pose these were taken against, not tc.tc_epoch, which
    # still points at the previous epoch while AR runs.
    tc._ar_key_pose = key_pose
    R = tc.R_enu2ecef
    tc.nav.P[:, :] = 0
    tc.nav.vsat[:, :] = 0
    tc.nav.x[tc.nav.na:] = 0
    cp_visible_sf = set(tc._ar_cp_visible_sf)
    hold_epochs = int(os.environ.get('HELD_VSAT_HOLD_EPOCHS', '0') or 0)
    last_visible = tc._cp_visible_sf_last_ep
    if last_visible is None:
        last_visible = {}
        tc._cp_visible_sf_last_ep = last_visible
    for sf in cp_visible_sf:
        last_visible[sf] = tc.epoch

    diag_estimate_missing = 0
    diag_vsat1 = 0
    diag_vsat0_young = 0
    diag_vsat0_held_bad = 0
    diag_ages = []
    diag_amb_el_deg = []  # elevation [deg] of vsat=1 amb sats
    for (s, f), k in sorted_amb_items(amb_dict):
        if estimate.exists(k):
            tc.nav.x[tc.IB(s, f, tc.nav.na)] = estimate.atDouble(k)
            # Exclude new ambiguities from AR until converged
            init_ep = tc._sat_states.at(s, f).amb_init_epoch
            age = tc.epoch - (init_ep if init_ep is not None else 0)
            diag_ages.append(int(age))
            held_bad = int(_satq.get_sat_quality(tc).persist_bad_hold.get(int(s), 0)) > 0
            if age >= tc.ar_wait_new and not held_bad:
                tc.nav.vsat[s - 1, f] = 1
                diag_vsat1 += 1
                el_idx = int(s) - 1
                if 0 <= el_idx < tc.nav.el.shape[0]:
                    diag_amb_el_deg.append(float(np.degrees(tc.nav.el[el_idx])))
            else:
                tc.nav.vsat[s - 1, f] = 0  # exclude from LAMBDA
                if held_bad:
                    diag_vsat0_held_bad += 1
                else:
                    diag_vsat0_young += 1
        else:
            diag_estimate_missing += 1
    tc._last_amb_el_min_deg = (int(round(min(diag_amb_el_deg)))
                                 if diag_amb_el_deg else -1)
    tc._last_amb_el_median_deg = (int(round(float(np.median(diag_amb_el_deg))))
                                    if diag_amb_el_deg else -1)
    tc._last_amb_el_above15 = sum(1 for e in diag_amb_el_deg if e >= 15)
    tc._last_amb_el_above25 = sum(1 for e in diag_amb_el_deg if e >= 25)
    tc._last_amb_estimate_missing = diag_estimate_missing
    tc._last_amb_vsat1 = diag_vsat1
    tc._last_amb_vsat0_young = diag_vsat0_young
    tc._last_amb_vsat0_held_bad = diag_vsat0_held_bad
    tc._last_amb_age_median = int(np.median(diag_ages)) if diag_ages else -1
    tc._last_amb_age_min = int(min(diag_ages)) if diag_ages else -1

    held_var = max(float(tc.cfg.varholdamb), 1e-6)
    for (s, f), held_value in tc._sat_states.held_items():
        tc.nav.x[tc.IB(s, f, tc.nav.na)] = float(held_value)
        tc.nav.P[tc.IB(s, f, tc.nav.na), tc.IB(s, f, tc.nav.na)] = held_var
        held_bad = int(_satq.get_sat_quality(tc).persist_bad_hold.get(int(s), 0)) > 0
        is_visible = (s, f) in cp_visible_sf
        if not is_visible and hold_epochs > 0:
            last_ep = last_visible.get((s, f), -10**9)
            is_visible = (tc.epoch - last_ep) <= hold_epochs
        tc.nav.vsat[s - 1, f] = (1 if is_visible and not held_bad else 0)

    held_sf = {(int(s), int(f)) for (s, f), _ in tc._sat_states.held_items()}
    amb_sf = {(int(s), int(f)) for (s, f) in amb_dict.keys()}
    orphan = [(s, f) for (s, f) in cp_visible_sf
              if (s, f) not in held_sf and (s, f) not in amb_sf]
    tc._last_orphan_cp_count = len(orphan)
    tc._last_amb_dict_size = len(amb_sf)
    tc._last_held_size = len(held_sf)
    tc._last_cp_visible_size = len(cp_visible_sf)

    # Pull marginals straight from the smoother's Bayes tree; constructing
    # a fresh gtsam.Marginals(factors, estimate) re-linearizes the entire
    # graph (including every Python CustomFactor) and dominates the AR
    # stage. ISAM2 already has the cached factorization. FLS exposes
    # marginalCovariance but joint marginals must come from getISAM2().
    smoother = getattr(tc, 'isam2', None)
    isam2 = smoother.getISAM2() if smoother is not None else None
    try:
        if isam2 is not None:
            P_pose = isam2.marginalCovariance(key_pose)
        else:
            mg = gtsam.Marginals(factors, estimate)
            P_pose = mg.marginalCovariance(key_pose)
        tc.nav.P[0:3, 0:3] = R @ P_pose[3:6, 3:6] @ R.T

        active = [(s, f, k) for (s, f), k in sorted_amb_items(amb_dict)
                  if estimate.exists(k)
                  and tc.nav.vsat[s - 1, f] == 1]
        if active:
            keys = gtsam.KeyVector()
            keys.append(key_pose)
            for s, f, k in active:
                keys.append(k)
            jm = (isam2.jointMarginalCovariance(keys) if isam2 is not None
                  else mg.jointMarginalCovariance(keys))
            for s, f, k in active:
                idx = tc.IB(s, f, tc.nav.na)
                tc.nav.P[idx, idx] = jm.at(k, k)[0, 0]
                Pxn = jm.at(key_pose, k)
                pxn_ecef = R @ Pxn[3:6, 0]
                tc.nav.P[0:3, idx] = pxn_ecef
                tc.nav.P[idx, 0:3] = pxn_ecef
            for i, (s1, f1, k1) in enumerate(active):
                i1 = tc.IB(s1, f1, tc.nav.na)
                for j, (s2, f2, k2) in enumerate(active):
                    if i >= j:
                        continue
                    i2 = tc.IB(s2, f2, tc.nav.na)
                    c = jm.at(k1, k2)[0, 0]
                    tc.nav.P[i1, i2] = c
                    tc.nav.P[i2, i1] = c
    except (RuntimeError, IndexError):
        # IndexError: gtsam raises it when key_pose (or an amb key) is not
        # in the BayesTree yet, e.g. the epoch right after a warm reset.
        pass
    bad = ~np.isfinite(tc.nav.P)
    if bad.any():
        tc.nav.P[bad] = 0.0
        diag_idx = np.where(np.diag(bad))[0]
        if len(diag_idx):
            tc.nav.P[diag_idx, diag_idx] = 1e10


def run_ar(tc, obs, rs, vs, dts, sat, el, iu, estimate,
              key_pose, amb_dict):
    """LAMBDA AR with validation."""
    tc._ar_subset_debug = None
    tc._last_ar_outcome = 'not_called'
    if tc.nav.armode == 0:
        tc._last_ar_outcome = 'armode_off'
        return 0, None
    _record_amb_diagnostics(tc, sat, amb_dict)
    nb, xa = _run_lambda_attempts(tc, sat, el, amb_dict)
    if nb <= 0:
        return 0, None
    if not _validate_fix(tc, obs, rs, vs, dts, sat, el, iu, xa, nb,
                         estimate=estimate, key_pose=key_pose):
        return 0, None
    if tc.nav.armode == 3:
        if not _apply_fix_and_hold(tc, estimate, key_pose, amb_dict, xa):
            return 0, None
    tc.nav.smode = 4
    return nb, xa


def _record_amb_diagnostics(tc, sat, amb_dict):
    """Phase A — count (sat, freq) entries in amb_dict / held that are missing from this-epoch sat list. cssrlib _ddidx_core silently skips them via not sat_present[sat_i]; surfacing the count makes "amb_size large yet nb=0" cases diagnosable."""
    sat_in_obs = {int(s) for s in sat}
    diag_amb_not_in_obs = sum(
        1 for (ss, ff) in amb_dict.keys() if int(ss) not in sat_in_obs)
    diag_held_not_in_obs = sum(
        1 for (ss, ff), _ in tc._sat_states.held_items()
        if int(ss) not in sat_in_obs)
    tc._last_amb_not_in_obs = diag_amb_not_in_obs
    tc._last_held_not_in_obs = diag_held_not_in_obs
    tc._last_sat_in_obs_size = len(sat_in_obs)
    tc._last_ar_outcome = 'entered'


def _run_lambda_attempts(tc, sat, el, amb_dict):
    """Phase B — call resamb_lambda (rtklib subset / rtklib / vanilla) with optional subset retry, then guard with lambda_zero / min_nb_gate. Returns (nb, xa) or (0, None) on any rejection."""
    tc._last_resamb_raw_nb = -1
    try:
        # Match the path being replaced: the round-robin retry belongs to
        # resamb_lambda_rtklib, and the default path calls plain resamb_lambda.
        native = None
        if tc.cfg.ar_native_resolver:
            sats = [int(x) for x in sat]
            native = (_resolve_native_retry(tc, sats) if tc.cfg.rtklib_mode
                      else _resolve_native(tc, sats))
        if native is not None:
            nb, xa = native
        elif tc.cfg.rtklib_mode and hasattr(tc, 'resamb_lambda_rtklib'):
            nb, xa = tc.resamb_lambda_rtklib(sat)
        else:
            nb, xa = tc.resamb_lambda(sat, tc.nav.parmode, tc.nav.par_P0)
    except (Exception, SystemExit):
        # cssrlib mlambda raises SystemExit when Qah is not positive definite
        tc._last_ar_outcome = 'lambda_exception'
        return 0, None

    tc._last_resamb_raw_nb = int(nb)
    if (nb <= 0 and bool(tc.cfg.subset_ar_enable)
            and len(amb_dict) >= int(tc.cfg.subset_ar_min_nb) + 1):
        try:
            nb, xa = _try_subset_ar(tc, sat, el, amb_dict)
        except (Exception, SystemExit):
            nb, xa = 0, None

    if nb <= 0:
        tc._last_ar_outcome = 'lambda_zero'
        return 0, None
    if (not tc.cfg.rtklib_mode) and nb < tc.cfg.ar_min_nb:
        tc._last_ar_outcome = 'min_nb_gate'
        return 0, None
    return nb, xa



def _validate_fix(tc, obs, rs, vs, dts, sat, el, iu, xa, nb,
                  estimate=None, key_pose=None):
    """Phase C — RTKLIB-style post-fit valpos + project-specific ar_context_reject. Returns True when the fix is accepted, False when either gate trips."""
    # xa[0:3] is already antenna position (nav.x[0:3] was set to antenna pos)
    fix_antenna = xa[0:3]

    yu, eu, _ = tc.zdres(obs, None, None, rs, vs, dts, fix_antenna)
    v_fix, _, R_fix = tc.sdres(obs, xa, yu[iu], eu[iu], sat, el)

    if not tc.valpos(v_fix, R_fix):
        tc._last_ar_outcome = 'valpos_failed'
        return False

    # Likelihood-ratio gate in the graph's OWN objective (pre-hold):
    # Δres = DDPR RMS with the pose moved to the fixed solution xa,
    # minus the same RMS at the float solution. A wrong-integer basin
    # is phase-self-consistent but the epoch's code factors protest —
    # the DELTA isolates that protest from the NLOS noise floor that
    # defeats absolute thresholds. Evaluated BEFORE fix-and-hold, so a
    # wrong basin is rejected before holds can lock it (once holds drag
    # the float into the basin the delta vanishes — timing matters).
    dres_thr = float(getattr(tc.cfg, 'ar_fix_dres_max', 0.0) or 0.0)
    ed = getattr(tc, '_cur_ed', None)
    if dres_thr > 0.0 and ed is not None and estimate is not None \
            and key_pose is not None:
        res_pre = tc._cached_ddpr_res_pre
        res_xa = None
        try:
            cur_pose = estimate.atPose3(key_pose)
            R_be = tc.ecef_T_nav.compose(cur_pose).rotation().matrix()
            lever_arr = (np.array(tc.lever_arm_tc)
                         if getattr(tc, 'lever_arm_tc', None) is not None
                         else np.zeros(3))
            body_xa = np.asarray(xa[0:3], dtype=float) - R_be @ lever_arr
            body_nav = tc.ecef_T_nav.transformTo(gtsam.Point3(*body_xa))
            v_xa = gtsam.Values()
            v_xa.insert(key_pose,
                        gtsam.Pose3(cur_pose.rotation(), body_nav))
            res_xa, _ = _tc_postfit.main_ddpr_residuals(tc, ed.g3, v_xa)
        except (RuntimeError, ValueError, IndexError):
            res_xa = None
        if res_pre is not None and res_xa is not None:
            tc._last_fix_dres = float(res_xa) - float(res_pre)
            if tc._last_fix_dres > dres_thr:
                tc._last_ar_outcome = 'fix_dres'
                return False

    reject_ctx, reject_detail = _ar_context_reject(tc, nb)
    if reject_ctx:
        tc._ar_context_reject = reject_detail
        tc._last_ar_outcome = 'ar_context_reject'
        return False
    tc._ar_context_reject = None
    tc._last_ar_outcome = 'success'

    return True


def _collect_held_sat_freq_keys(tc, amb_dict):
    """Return [(s, f), ...] for ambiguities that holdamb_flags() just promoted to fix=3."""
    return [(s, f) for (s, f), _k in sorted_amb_items(amb_dict)
            if tc.nav.fix[s - 1, f] == 3]


def _add_phase1_hold_priors(tc, hg, hold_keys, amb_dict, xa):
    """Add per-held-N PriorDouble factors with σ = √varholdamb (RTKLIB-style, in cycles)."""
    hold_sigma = float(np.sqrt(tc.cfg.varholdamb))
    hold_noise = tc._noise1(hold_sigma)
    for s, f in hold_keys:
        hg.addPriorDouble(
            amb_dict[(s, f)], xa[tc.IB(s, f, tc.nav.na)], hold_noise)


def _add_fix_pose_anchor_factor(tc, hg, estimate, key_pose, xa):
    """Add a PriorPose3 at the LAMBDA-fixed antenna position; skipped (returns False) on any failure."""
    anchor_sigma = float(tc.cfg.fix_pose_anchor_sigma)
    try:
        cur_pose = estimate.atPose3(key_pose)
        R_body_to_ecef = tc.ecef_T_nav.compose(cur_pose).rotation().matrix()
        lever_arr = (np.array(tc.lever_arm_tc)
                     if getattr(tc, 'lever_arm_tc', None) is not None
                     else np.zeros(3))
        body_ecef_target = xa[0:3] - R_body_to_ecef @ lever_arr
        body_nav_target = tc.ecef_T_nav.transformTo(
            gtsam.Point3(*body_ecef_target))
        target_pose = gtsam.Pose3(cur_pose.rotation(), body_nav_target)
        # 1e6 rad on rotation = unconstrained; translation σ = anchor_sigma m.
        sigmas = np.array([1e6, 1e6, 1e6,
                           anchor_sigma, anchor_sigma, anchor_sigma])
        anchor_noise = gtsam.noiseModel.Diagonal.Sigmas(sigmas)
        hg.addPriorPose3(key_pose, target_pose, anchor_noise)
        return True
    except RuntimeError:
        return False


def _apply_holds_phase2_with_gate(tc, hg, key_pose, anchor_added):
    """Phase 2: ISAM2.update with the GICI-style post-AR cost gate."""
    isam = tc.isam2
    full_graph = isam.getFactors()
    res_pre = tc._cached_ddpr_res_pre
    ts_h = gtsam.FixedLagSmootherKeyTimestampMap()
    if anchor_added:
        ts_h[key_pose] = tc.tc_time
    base_idx_undo = tc.total_factor_count
    n_added = hg.size()
    try:
        isam.update(hg, gtsam.Values(), ts_h)
        tc.total_factor_count += n_added
        res_post = None
        if res_pre is not None:
            try:
                est_post = isam.calculateEstimate()
                res_post, _ = _tc_postfit.main_ddpr_residuals(
                    tc, full_graph, est_post)
            except (RuntimeError, IndexError):
                res_post = None
        if (res_pre is not None and res_post is not None
                and (res_post - res_pre) > tc.cfg.post_ar_cost_thresh):
            # Reject: remove hold-prior factors.
            try:
                isam.update(
                    gtsam.NonlinearFactorGraph(),
                    gtsam.Values(),
                    gtsam.FixedLagSmootherKeyTimestampMap(),
                    list(range(base_idx_undo, base_idx_undo + n_added)))
            except (RuntimeError, IndexError):
                pass
            return False
    except (RuntimeError, IndexError):
        pass
    return True


def _apply_holds_phase1(tc, hg, hold_keys, amb_dict):
    """Phase 1: IncrementalFixedLagSmoother.update with held-N timestamps + amb_factor_indices tracking."""
    isam = tc.isam
    ts_h1 = gtsam.FixedLagSmootherKeyTimestampMap()
    t_p1 = getattr(tc, 'phase1_t', 0.0)
    for sf in hold_keys:
        ts_h1[amb_dict[sf]] = t_p1
    try:
        isam.update(hg, gtsam.Values(), ts_h1)
        base_idx = tc.total_factor_count
        for i, key_id in enumerate(hold_keys):
            tc._sat_states.get(*key_id).amb_factor_indices.append(
                base_idx + i)
        tc.total_factor_count += hg.size()
    except (RuntimeError, IndexError):
        pass


def _activate_phase2_hold_states(tc, hold_keys, xa):
    """Phase 2: copy held N → sat_state hold + nav.x; clear amb_key / amb_factor_indices."""
    for s, f in hold_keys:
        held_value = float(xa[tc.IB(s, f, tc.nav.na)])
        sat_st = tc._sat_states.get(s, f)
        sat_st.activate_hold(held_value)
        sat_st.amb_key = None
        sat_st.amb_factor_indices = []
        tc.nav.x[tc.IB(s, f, tc.nav.na)] = held_value


def _apply_fix_and_hold(tc, estimate, key_pose, amb_dict, xa):
    """Phase D — fix-and-hold (armode==3): mark held flags, build hold-prior factors, optional fix_pose_anchor, run ISAM2.update with post-AR cost gate, then activate hold on sat_states. Returns True on accept, False when the post-AR cost gate rejects the fix."""
    tc.holdamb_flags()
    hold_keys = _collect_held_sat_freq_keys(tc, amb_dict)
    hg = gtsam.NonlinearFactorGraph()
    if tc.phase != 2:
        _add_phase1_hold_priors(tc, hg, hold_keys, amb_dict, xa)
    anchor_added = False
    if tc.phase == 2 and float(tc.cfg.fix_pose_anchor_sigma) > 0:
        anchor_added = _add_fix_pose_anchor_factor(
            tc, hg, estimate, key_pose, xa)
    if hg.size() > 0:
        if tc.phase == 2:
            if not _apply_holds_phase2_with_gate(
                    tc, hg, key_pose, anchor_added):
                return False
        else:
            _apply_holds_phase1(tc, hg, hold_keys, amb_dict)
    if tc.phase == 2:
        _activate_phase2_hold_states(tc, hold_keys, xa)
    return True


