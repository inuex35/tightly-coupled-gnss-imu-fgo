#!/usr/bin/env python3
"""Two-phase IMU/GNSS tight coupling example.

Phase 1: GNSS-only Pose3 RTK (stationary, DDFactorArm lever=0)
Phase 2: CombinedImuFactor + DDFactorArm (moving, progressive lever arm)

Usage:
  python run_imu_gnss_tc.py <rover.obs> <base.obs> <nav_file> <imu.csv> <reference.csv>

Environment variables:
  MAX_EP       : max epochs (default: end of rover.obs)
  LEVER_ARM    : lever arm x,y,z in body FLU (default "0.31,0,0.55")
  SAVE_NPZ     : write per-epoch diagnostics to this path
  (pipeline knobs: every TcConfig field is an env var — see config.py)
"""

import sys
import os
import csv
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import cssrlib.rinex as rn
import cssrlib.gnss as gn
from cssrlib.gnss import uTYP, ecef2pos, sat2prn
from gnss_fgo import ImuGnssTc, load_imu_csv
from gnss_fgo.utils import R_ENU2NED, R_FRD2FLU
from gnss_fgo.utils.sig_autodetect import auto_detect_signals


def parse_args():
    if len(sys.argv) < 6:
        print(__doc__)
        sys.exit(1)
    return (sys.argv[1], sys.argv[2], sys.argv[3],
            sys.argv[4], sys.argv[5])


def _format_sat_freq(sat, freq):
    sys_i, prn = sat2prn(int(sat))
    sys_ch = {0: 'G', 1: 'E', 2: 'J', 3: 'C', 4: 'R'}.get(sys_i, '?')
    return f"{sys_ch}{prn:02d}f{int(freq)}"


def _pose_rph_deg(tc):
    """Current Phase-2 pose attitude in the reference NED/FRD convention."""
    if tc.phase != 2 or tc.isam2 is None or tc.tc_epoch < 0:
        return (np.nan, np.nan, np.nan)
    try:
        est = tc.isam2.calculateEstimate()
        pose = est.atPose3(tc.Xpose(tc.tc_epoch))
        r_body2enu = pose.rotation().matrix()
        r_body2ned = R_ENU2NED @ r_body2enu @ R_FRD2FLU
        roll = np.degrees(np.arctan2(r_body2ned[2, 1], r_body2ned[2, 2]))
        pitch = np.degrees(np.arcsin(np.clip(-r_body2ned[2, 0], -1.0, 1.0)))
        heading = np.degrees(np.arctan2(r_body2ned[1, 0], r_body2ned[0, 0]))
        return (float(roll), float(pitch), float(heading))
    except Exception:
        return (np.nan, np.nan, np.nan)


def _pose_axis_headings_deg(tc):
    """Current Phase-2 body-axis headings from the internal FLU pose."""
    if tc.phase != 2 or tc.isam2 is None or tc.tc_epoch < 0:
        return (np.nan, np.nan, np.nan)
    try:
        est = tc.isam2.calculateEstimate()
        pose = est.atPose3(tc.Xpose(tc.tc_epoch))
        r_body2enu = pose.rotation().matrix()
        h_fwd = np.degrees(np.arctan2(r_body2enu[0, 0], r_body2enu[1, 0]))
        h_left = np.degrees(np.arctan2(r_body2enu[0, 1], r_body2enu[1, 1]))
        h_up = np.degrees(np.arctan2(r_body2enu[0, 2], r_body2enu[1, 2]))
        return (float(h_fwd), float(h_left), float(h_up))
    except Exception:
        return (np.nan, np.nan, np.nan)


def main():
    obsfile, basefile, navfile, imufile, reffile = parse_args()

    # Auto-detect signals from rover/base RINEX headers. max_freq=3 picks
    # the lowest 3 bands by priority (L1/L2/L5 family); systems missing a
    # band are dropped (strict_freq).
    nav_nf = 3
    dec = rn.rnxdec()
    decb = rn.rnxdec()
    dec.decode_obsh(obsfile)
    decb.decode_obsh(basefile)
    # GLONASS is excluded: FDMA inter-channel biases do not cancel in the
    # double difference, so the DD-only core cannot use it (its biased
    # pseudoranges would poison the float solution).
    sigs, sigsb = auto_detect_signals(
        dec.sig_map, decb.sig_map, max_freq=nav_nf,
        required=(uTYP.C, uTYP.L, uTYP.S),
        systems=[gn.uGNSS.GPS, gn.uGNSS.GAL, gn.uGNSS.QZS, gn.uGNSS.BDS],
        strict_freq=False,
    )
    # Add rover Doppler on the same picked bands (base has no Doppler,
    # detslp_dop is rover-only); skip systems where rover lacks Doppler.
    rov_picks_by_sys = {}
    for s in sigs:
        rov_picks_by_sys.setdefault(s.sys, set()).add(int(s.sig) // 100)
    for sys_id, bands in rov_picks_by_sys.items():
        rov_typ_d = {int(s.sig) // 100: s for s in dec.sig_map.get(sys_id, {}).values()
                     if s.typ == uTYP.D}
        for band in bands:
            if band in rov_typ_d:
                sigs.append(rov_typ_d[band])
    dec.setSignals(sigs)
    decb.setSignals(sigsb)
    nav = gn.Nav(nf=nav_nf)
    dec.decode_nav(navfile, nav)

    base_ecef = np.array(list(decb.pos))
    if np.linalg.norm(base_ecef) == 0:
        print("Error: base position not found in RINEX header")
        sys.exit(1)

    # Load reference (5Hz CSV)
    ref = []
    with open(reffile) as f:
        for row in csv.DictReader(f):
            ref.append({
                'tow': float(row['GPS TOW (s)']),
                'ecef': np.array([float(row['ECEF X (m)']),
                                  float(row['ECEF Y (m)']),
                                  float(row['ECEF Z (m)'])]),
                'vel': np.array([float(row['East Velocity (m/s)']),
                                 float(row['North Velocity (m/s)']),
                                 float(row['Up Velocity (m/s)'])]),
            })
    # Lookup ref by TOW (rover may be 1Hz or 5Hz; ref is 5Hz)
    ref_tows = np.array([r['tow'] for r in ref])
    # Load IMU
    imu_data = load_imu_csv(imufile)
    print(f"Loaded {len(imu_data)} IMU samples, {len(ref)} reference points")

    # Lever arm from env, user-facing FLU.
    lever_str = os.environ.get('LEVER_ARM', '0.31,0,0.55')
    lever_arm = np.array([float(x) for x in lever_str.split(',')])

    # Setup
    nav.rb = base_ecef.tolist()
    nav.pmode = 1
    nav.ephopt = 0
    pos0 = np.array(dec.pos) if np.linalg.norm(dec.pos) > 0 else base_ecef.copy()

    tc = ImuGnssTc(nav, pos0, base_ecef, imu_data,
                   lever_arm=lever_arm)

    nep_env = os.environ.get('MAX_EP', '').strip()
    if not nep_env or nep_env.lower() in ('all', 'auto', '0'):
        # Default: process to end of rover.obs (StopIteration breaks the loop).
        nep = 10**9
        nep_label = 'all (until rover end)'
    else:
        nep = int(nep_env)
        nep_label = str(nep)
    gnss_cut_from = int(os.environ.get('GNSS_CUT_FROM', '-1'))
    print(f"Base: {base_ecef}")
    print(f"Lever arm (FLU input): {lever_arm}")
    print(f"Max epochs: {nep_label}")
    if gnss_cut_from >= 0:
        print(f"Forced GNSS cut from epoch: {gnss_cut_from}")
    print()

    results = []
    maxage = float(os.environ.get('BASE_MAXAGE', '30.0'))
    sync_gen = rn.sync_obs_hold(dec, decb, maxage=maxage)
    for ne in range(nep):
        try:
            obs, obsb, dt_sync = next(sync_gen)
        except StopIteration:
            break
        forced_gnss_cut = gnss_cut_from >= 0 and ne >= gnss_cut_from
        if forced_gnss_cut:
            obsb = None
        if obsb is None:
            # Base missing: advance graph by IMU only (keeps chain coherent).
            sol, tag, nb, info = tc.process_imu_only(obs)
            roll_deg, pitch_deg, heading_deg = _pose_rph_deg(tc)
            axis_heading_fwd_deg, axis_heading_right_deg, axis_heading_down_deg = _pose_axis_headings_deg(tc)
            _, tow_obs = gn.time2gpst(obs.t)
            ri_ref = int(np.argmin(np.abs(ref_tows - tow_obs)))
            ref_ecef = ref[ri_ref]['ecef']
            enu_err = gn.ecef2enu(ecef2pos(ref_ecef), sol - ref_ecef)
            err_3d = float(np.linalg.norm(sol - ref_ecef))
            reason = "FORCED_GNSS_CUT" if forced_gnss_cut else "SKIP_NOBASE"
            print(f"Ep {ne:4d} IMU: {tag} E={enu_err[0]:+.4f} "
                  f"N={enu_err[1]:+.4f} U={enu_err[2]:+.4f} "
                  f"3D={err_3d:.4f}m {reason}")
            results.append({'ne': ne, 'tag': tag, 'phase': tc.phase,
                            'enu': enu_err, 'err': err_3d,
                            'nb': nb, 'smode': tc.nav.smode,
                            'sol_xyz': np.array(sol, copy=True),
                            'roll_deg': roll_deg,
                            'pitch_deg': pitch_deg,
                            'heading_deg': heading_deg,
                            'axis_heading_fwd_deg': axis_heading_fwd_deg,
                            'axis_heading_right_deg': axis_heading_right_deg,
                            'axis_heading_down_deg': axis_heading_down_deg,
                            'pred_heading_deg': info.get('pred_heading_deg', np.nan),
                            'post_heading_deg': info.get('post_heading_deg', np.nan),
                            'forced_gnss_cut': forced_gnss_cut,
                            'truth_xyz': np.asarray(ref_ecef)})
            continue
        if ne == 0:
            nav.t = obs.t

        prep = tc.prepare_double_difference_measurements(
            obs, obsb, pos_pred=nav.x[0:3].copy(), dd_only=True,
            compute_zdres=False,
        )
        ns = 0 if prep is None else len(prep['iu'])
        if ns < 4:
            # Too few sats for DD: advance graph by IMU only.
            sol, tag, nb, info = tc.process_imu_only(obs)
            _, tow_obs = gn.time2gpst(obs.t)
            ri_ref = int(np.argmin(np.abs(ref_tows - tow_obs)))
            ref_ecef = ref[ri_ref]['ecef']
            enu_err = gn.ecef2enu(ecef2pos(ref_ecef), sol - ref_ecef)
            err_3d = float(np.linalg.norm(sol - ref_ecef))
            print(f"Ep {ne:4d} IMU: {tag} E={enu_err[0]:+.4f} "
                  f"N={enu_err[1]:+.4f} U={enu_err[2]:+.4f} "
                  f"3D={err_3d:.4f}m SKIP_FEWSATS(ns={ns})")
            results.append({'ne': ne, 'tag': tag, 'phase': tc.phase,
                            'enu': enu_err, 'err': err_3d,
                            'nb': nb, 'smode': tc.nav.smode,
                            'sol_xyz': np.array(sol, copy=True),
                            'truth_xyz': np.asarray(ref_ecef)})
            continue

        rs = prep['rs']
        vs = prep['vs']
        dts = prep['dts']
        rsb = prep['rsb']
        sat = prep['sat']
        el = prep['el']
        iu = prep['iu']
        obs_sd = prep['obs_sd']
        ir_map = {s: i for i, s in enumerate(obsb.sat)}

        _, tow_obs = gn.time2gpst(obs.t)
        ri_ref = int(np.argmin(np.abs(ref_tows - tow_obs)))
        ref_vel = ref[ri_ref]['vel']
        ref_ecef = ref[ri_ref]['ecef']

        sol, tag, nb, info = tc.process(
            obs, obsb, rs, vs, dts, rsb, sat, el, iu, obs_sd, ir_map,
            ref_ecef=ref_ecef)
        roll_deg, pitch_deg, heading_deg = _pose_rph_deg(tc)
        axis_heading_fwd_deg, axis_heading_right_deg, axis_heading_down_deg = _pose_axis_headings_deg(tc)

        enu_err = gn.ecef2enu(ecef2pos(ref_ecef), sol - ref_ecef)
        err_3d = np.linalg.norm(sol - ref_ecef)
        rec = {
            'ep': ne, 'tag': tag, 'nb': nb, 'err': err_3d,
            'enu': enu_err, 'phase': info['phase'],
            'sol_xyz': np.array(sol, copy=True),
            'roll_deg': roll_deg,
            'pitch_deg': pitch_deg,
            'heading_deg': heading_deg,
            'axis_heading_fwd_deg': axis_heading_fwd_deg,
            'axis_heading_right_deg': axis_heading_right_deg,
            'axis_heading_down_deg': axis_heading_down_deg,
            'pred_heading_deg': info.get('pred_heading_deg', np.nan),
            'post_heading_deg': info.get('post_heading_deg', np.nan),
            'forced_gnss_cut': forced_gnss_cut,
            'truth_xyz': np.asarray(ref_ecef),
            'ref_vel': np.asarray(ref_vel),
            'smode': tc.nav.smode,
        }
        rec.update({k: v for k, v in info.items() if k != 'transition'})
        results.append(rec)

        if 'transition' in info:
            print(f"\n--- Phase 2 transition at epoch {ne} ---")
            print(f"    pitch={info['pitch']:.1f} roll={info['roll']:.1f} "
                  f"heading={info['heading']:.1f} deg")
            print(f"    {info['n_amb_tc']} ambiguities (fresh init, DD at all {info.get('n_collected',0)} collected epochs)")
            print(f"    lever_arm={lever_arm.tolist()}")
            if 'bias_acc_init' in info:
                ba0 = info['bias_acc_init']
                bg0 = info['bias_gyro_init']
                print(f"    bias_acc0=[{ba0[0]:+.4f},{ba0[1]:+.4f},{ba0[2]:+.4f}]")
                print(f"    bias_gyro0=[{np.degrees(bg0[0]):+.4f},{np.degrees(bg0[1]):+.4f},{np.degrees(bg0[2]):+.4f}] deg/s")
            print()

        vel_mag = np.linalg.norm(ref_vel[:2])
        phase_tag = f"P{info['phase']}"
        if info.get('collecting') and info['phase'] == 1:
            phase_tag = f"P1c{info['n_collected']}"
        extra = ""
        if info.get('ddpr_recover'):
            extra += f" DDPR_RECOVER(n={info['ddpr_recover']})"
        if 'ddpr_fast_recover' in info:
            extra += (
                f" DDPR_FAST({info['ddpr_fast_recover']:.2f},"
                f"worst={float(info.get('ddpr_fast_worst_sat_res', 0.0)):.1f})"
            )
        if 'prev_pose_missing' in info:
            extra += f" PREV_MISSING(k={info['prev_pose_missing']})"
        if info.get('error'):
            # Keep log single-line & greppable; show up to 200 chars
            e_msg = info['error'].replace('\n', ' | ').replace('(', '[').replace(')', ']')
            extra += f" ERR[{e_msg[:200]}]"
        if info['phase'] == 2 and 'bias_acc' in info:
            slip = info.get('n_slip', 0)
            mf = info.get('max_frac', 0)
            extra += f" frac={mf:.2f}"
            if slip > 0:
                extra += f" slip={slip}"
            cp_slip = info.get('cp_slip', 0)
            if cp_slip > 0:
                extra += f" cp_slip={cp_slip}"
            fde = info.get('fde_reject', 0)
            if fde > 0:
                extra += f" FDE={fde}"
            if info.get('gnss_skip'):
                extra += f" SKIP(gdop={info.get('gdop',0):.1f},ns={info.get('nsat',0)})"
            if 'ddpr_innov' in info:
                extra += f" ddpr={info['ddpr_innov']:.2f}"
                if 'ecef_ddpr' in info:
                    ddpr_err = np.linalg.norm(info['ecef_ddpr'] - ref_ecef)
                    extra += f"(truth={ddpr_err:.2f})"
                if 'ddpr_res' in info:
                    extra += f" res={info['ddpr_res']:.2f}"
            if info.get('nhc'):
                extra += " NHC"
            if 'lambda_correction' in info:
                extra += f" λcorr={info['lambda_correction']:.3f}"
            if info.get('ar_subset_used'):
                drop_sat = int(info.get('ar_subset_drop_sat', 0))
                sys_i, prn = sat2prn(drop_sat)
                sys_ch = {0:'G',1:'E',2:'J',3:'C',4:'R'}.get(sys_i, '?')
                extra += (
                    f" SUBSET_AR({sys_ch}{prn:02d}"
                    f",nb={int(info.get('ar_subset_nb', 0))}"
                    f",r={float(info.get('ar_subset_ratio', 0.0)):.2f})"
                )
            if 'lambda_corr_hard_reject' in info:
                extra += f" LCHR({info['lambda_corr_hard_reject']:.2f})"
            if info.get('weak_fix_reject'):
                extra += (
                    f" WEAK_FIX_REJ(nb={int(info.get('weak_fix_reject_nb', 0))}"
                    f",lc={float(info.get('weak_fix_reject_lc', 0.0)):.2f}"
                    f",mres={float(info.get('weak_fix_reject_main_ddpr_res', 0.0)):.2f})"
                )
            if 'prev_pose_drift' in info:
                extra += f" prev_drift={info['prev_pose_drift']:.3f}"
            if info.get('pim_discontinuity'):
                extra += " PIM_BREAK"
            if 'main_ddpr_res' in info:
                extra += f" mres={info['main_ddpr_res']:.2f}"
                if 'main_ddpr_sat_worst' in info:
                    sw, rw = info['main_ddpr_sat_worst']
                    sys_i, prn = sat2prn(sw)
                    sys_ch = {0:'G',1:'E',2:'J',3:'C',4:'R'}.get(sys_i, '?')
                    extra += f" worst={sys_ch}{prn:02d}:{rw:.1f}"
            if int(info.get('held_release_flt_count', 0) or 0) > 0:
                s_rel = int(info.get('held_release_flt_sat', 0) or 0)
                f_rel = int(info.get('held_release_flt_freq', 0) or 0)
                score_rel = float(info.get('held_release_flt_score', 0.0) or 0.0)
                res_rel = float(info.get('held_release_flt_res', 0.0) or 0.0)
                cppr_rel = int(info.get('held_release_flt_cppr', 0) or 0)
                extra += (
                    f" HREL({_format_sat_freq(s_rel, f_rel)}"
                    f",score={score_rel:.1f},res={res_rel:.1f},cppr={cppr_rel})"
                )
            gauge_rel = info.get('hold_gauge_rel')
            if gauge_rel:
                worst = max(gauge_rel, key=lambda t: t[2])
                extra += (
                    f" HGAUGE(n={len(gauge_rel)},"
                    f"worst={_format_sat_freq(worst[0], worst[1])}"
                    f":{worst[2]:.0f}m)"
                )

        if True:  # (kept indent; printing is unconditional)
            print(f"Ep {ne:3d} {phase_tag}: {tag} E={enu_err[0]:+.4f} "
                  f"N={enu_err[1]:+.4f} U={enu_err[2]:+.4f} "
                  f"3D={err_3d:.4f}m nb={nb} vel={vel_mag:.2f}{extra}")

    dec.fobs.close()
    decb.fobs.close()

    # Summary
    print(f"\n{'='*60}")
    print(f"IMU/GNSS TC Results: {len(results)} epochs")
    print(f"{'='*60}")

    if not results:
        return

    p1 = [r for r in results if r['phase'] == 1]
    p2 = [r for r in results if r['phase'] == 2]

    for label, subset in [("Phase 1", p1), ("Phase 2", p2), ("All", results)]:
        if not subset:
            continue
        fix = [r for r in subset if r['tag'] == 'FIX']
        flt = [r for r in subset if r['tag'] == 'FLT']
        enu_all = np.array([r['enu'] for r in subset])
        err_all = np.array([r['err'] for r in subset])
        print(f"\n{label} ({len(subset)} ep, {len(fix)} fix, {len(flt)} flt):")
        print(f"  3D RMS: {np.sqrt(np.mean(err_all**2)):.4f}m, "
              f"median: {np.median(err_all):.4f}m")
        print(f"  E RMS:  {np.sqrt(np.mean(enu_all[:,0]**2)):.4f}m")
        print(f"  N RMS:  {np.sqrt(np.mean(enu_all[:,1]**2)):.4f}m")
        print(f"  U RMS:  {np.sqrt(np.mean(enu_all[:,2]**2)):.4f}m")
        if fix:
            e3f = np.array([r['err'] for r in fix])
            print(f"  Fix 3D RMS: {np.sqrt(np.mean(e3f**2)):.4f}m")

    savefile = os.environ.get('SAVE_NPZ')
    if savefile:
        n = len(results)
        # Always-present scalars / vectors
        out = {
            'ne': np.array([r.get('ep', r.get('ne', -1)) for r in results]),
            'enu': np.array([r['enu'] for r in results]),
            'err3d': np.array([r['err'] for r in results]),
            'smode': np.array([4 if r['tag'] == 'FIX' else
                               5 if r['tag'] == 'FLT' else 0
                               for r in results]),
            'phase': np.array([r['phase'] for r in results]),
            'nb': np.array([r.get('nb', 0) for r in results]),
            'sol_xyz': np.array([r.get('sol_xyz', np.full(3, np.nan))
                                 for r in results]),
            'truth_xyz': np.array([r.get('truth_xyz', np.full(3, np.nan))
                                   for r in results]),
            'ref_vel': np.array([r.get('ref_vel', np.full(3, np.nan))
                                 for r in results]),
            'base_xyz': np.asarray(base_ecef),
        }
        # Discover all info-dict keys across epochs and pack each as a
        # dense (n,) or (n,3) array with NaN where absent.
        all_keys = set()
        for r in results:
            all_keys.update(r.keys())
        skip = set(out.keys()) | {'ep', 'ne', 'tag', 'err', 'phase',
                                  'enu', 'sol_xyz', 'truth_xyz', 'ref_vel',
                                  'smode', 'main_ddpr_per_sat',
                                  'main_ddpr_sat_worst', 'error'}
        for k in sorted(all_keys - skip):
            samples = [r[k] for r in results if k in r and r[k] is not None]
            if not samples:
                continue
            # Guess dtype
            s0 = samples[0]
            if isinstance(s0, (bool, np.bool_)):
                arr = np.zeros(n, dtype=bool)
                for i, r in enumerate(results):
                    if r.get(k) is True: arr[i] = True
            elif isinstance(s0, (int, float, np.integer, np.floating)):
                arr = np.full(n, np.nan, dtype=float)
                for i, r in enumerate(results):
                    if k in r and r[k] is not None:
                        try: arr[i] = float(r[k])
                        except (TypeError, ValueError): pass
            elif hasattr(s0, '__len__') and len(s0) == 3:
                arr = np.full((n, 3), np.nan)
                for i, r in enumerate(results):
                    if k in r and r[k] is not None:
                        try: arr[i] = np.asarray(r[k], float)
                        except Exception: pass
            else:
                continue  # skip non-trivial types
            out[k] = arr
        np.savez(savefile, **out)
        print(f"\nSaved {len(out)} variables to {savefile}: "
              f"{', '.join(sorted(out.keys()))}")

    persatfile = os.environ.get('SAVE_PER_SAT')
    if persatfile:
        import pickle
        per_sat_dump = []
        per_sat_truth_dump = []
        sat_el_dump = []
        sat_snr_dump = []
        sat_lock_age_dump = []
        sat_cppr_dump = []
        pair_main_dump = []
        pair_truth_dump = []
        ref_sats_dump = []
        for r in results:
            d = r.get('main_ddpr_per_sat', None)
            per_sat_dump.append(dict(d) if d else None)
            dt = r.get('ddpr_per_sat_at_truth', None)
            per_sat_truth_dump.append(dict(dt) if dt else None)
            se = r.get('sat_el_deg', None)
            sat_el_dump.append(dict(se) if se else None)
            ss = r.get('sat_snr_dbhz', None)
            sat_snr_dump.append(dict(ss) if ss else None)
            sl = r.get('sat_lock_age', None)
            sat_lock_age_dump.append(dict(sl) if sl else None)
            sc = r.get('sat_cppr_sat', None)
            sat_cppr_dump.append(dict(sc) if sc else None)
            pm = r.get('main_ddpr_pairs', None)
            pair_main_dump.append(list(pm) if pm else None)
            pt = r.get('ddpr_pairs_at_truth', None)
            pair_truth_dump.append(list(pt) if pt else None)
            rs = r.get('ref_sats', None)
            ref_sats_dump.append(dict(rs) if rs else None)
        with open(persatfile, 'wb') as f:
            pickle.dump({'per_sat': per_sat_dump,
                         'per_sat_truth': per_sat_truth_dump,
                         'sat_el_deg': sat_el_dump,
                         'sat_snr_dbhz': sat_snr_dump,
                         'sat_lock_age': sat_lock_age_dump,
                         'sat_cppr_sat': sat_cppr_dump,
                         'pair_main': pair_main_dump,
                         'pair_truth': pair_truth_dump,
                         'ref_sats': ref_sats_dump,
                          'err3d': out.get('err3d') if 'err3d' in out else None,
                          'smode': out.get('smode') if 'smode' in out else None}, f)
        print(f"Saved per-sat residuals to {persatfile}")


if __name__ == '__main__':
    main()
