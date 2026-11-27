#!/usr/bin/env python3
"""IMU/GNSS tight coupling example.

CombinedImuFactor + DDFactorArm + SD Doppler + NHC.

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
from cssrlib.gnss import uTYP, ecef2pos
from gnss_fgo import ImuGnssTc, load_imu_csv
from gnss_fgo.utils import auto_detect_signals


def parse_args():
    if len(sys.argv) < 6:
        print(__doc__)
        sys.exit(1)
    return (sys.argv[1], sys.argv[2], sys.argv[3],
            sys.argv[4], sys.argv[5])


def main():
    obsfile, basefile, navfile, imufile, reffile = parse_args()

    # Auto-detect signals from rover/base RINEX headers. max_freq=3 picks
    # the lowest 3 bands by priority (L1/L2/L5 family); systems missing a
    # band are kept (strict_freq=False).
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
        if s.typ != uTYP.L:
            continue
        band = int(s.sig) // 100
        bands = rov_picks_by_sys.setdefault(s.sys, [])
        if band not in bands:
            bands.append(band)
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
    print(f"Base: {base_ecef}")
    print(f"Lever arm (FLU input): {lever_arm}")
    print(f"Max epochs: {nep_label}")
    print()

    results = []
    ref_match_dt_max = 0.0
    maxage = float(os.environ.get('BASE_MAXAGE', '30.0'))
    sync_gen = rn.sync_obs_hold(dec, decb, maxage=maxage, dt_fill=0.2)
    for ne in range(nep):
        try:
            obs, obsb, dt_sync = next(sync_gen)
        except StopIteration:
            break

        _, tow_obs = gn.time2gpst(obs.t)
        ri_ref = int(np.argmin(np.abs(ref_tows - tow_obs)))
        ref_match_dt_max = max(ref_match_dt_max,
                               abs(float(ref_tows[ri_ref]) - float(tow_obs)))
        ref_vel = ref[ri_ref]['vel']
        ref_ecef = ref[ri_ref]['ecef']

        sol, tag, nb, info = tc.process(obs, obsb)

        enu_err = gn.ecef2enu(ecef2pos(ref_ecef), sol - ref_ecef)
        err_3d = np.linalg.norm(sol - ref_ecef)
        rec = {
            'ep': ne, 'tag': tag, 'nb': nb, 'err': err_3d,
            'enu': enu_err,
            'sol_xyz': np.array(sol, copy=True),
            'truth_xyz': np.asarray(ref_ecef),
            'ref_vel': np.asarray(ref_vel),
            'smode': tc.nav.smode,
        }
        rec.update(info)
        results.append(rec)

        vel_mag = np.linalg.norm(ref_vel[:2])
        extra = ""
        if 'prev_pose_missing' in info:
            extra += f" PREV_MISSING(k={info['prev_pose_missing']})"
        if info.get('error'):
            e_msg = info['error'].replace('\n', ' | ').replace('(', '[').replace(')', ']')
            extra += f" ERR[{e_msg[:200]}]"
        if 'bias_acc' in info:
            slip = info.get('n_slip', 0)
            if slip > 0:
                extra += f" slip={slip}"
            if info.get('n_dd', 0) == 0:
                extra += f" SKIP(ns={info.get('nsat', 0)})"
            if info.get('nhc'):
                extra += " NHC"
            if 'lambda_correction' in info:
                extra += f" λcorr={info['lambda_correction']:.3f}"

        print(f"Ep {ne:3d}: {tag} E={enu_err[0]:+.4f} "
              f"N={enu_err[1]:+.4f} U={enu_err[2]:+.4f} "
              f"3D={err_3d:.4f}m nb={nb} vel={vel_mag:.2f}{extra}")

    dec.fobs.close()
    decb.fobs.close()

    # Summary
    print(f"\n{'='*60}")
    print(f"IMU/GNSS TC Results: {len(results)} epochs")
    print(f"{'='*60}")
    print(f"  ref match max|dt|: {ref_match_dt_max:.3f}s")

    if not results:
        return

    fix = [r for r in results if r['tag'] == 'FIX']
    flt = [r for r in results if r['tag'] == 'FLT']
    enu_all = np.array([r['enu'] for r in results])
    err_all = np.array([r['err'] for r in results])
    print(f"\nAll ({len(results)} ep, {len(fix)} fix ({100.0*len(fix)/len(results):.1f}%), {len(flt)} flt):")
    print(f"  3D RMS: {np.sqrt(np.mean(err_all**2)):.4f}m, "
          f"median: {np.median(err_all):.4f}m, "
          f"<50cm: {100.0*np.mean(err_all < 0.5):.1f}%")
    print(f"  E RMS:  {np.sqrt(np.mean(enu_all[:,0]**2)):.4f}m")
    print(f"  N RMS:  {np.sqrt(np.mean(enu_all[:,1]**2)):.4f}m")
    print(f"  U RMS:  {np.sqrt(np.mean(enu_all[:,2]**2)):.4f}m")
    if fix:
        e3f = np.array([r['err'] for r in fix])
        print(f"  Fix 3D RMS: {np.sqrt(np.mean(e3f**2)):.4f}m")

    savefile = os.environ.get('SAVE_NPZ')
    out = {}
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
        skip = set(out.keys()) | {'ep', 'ne', 'tag', 'err',
                                  'enu', 'sol_xyz', 'truth_xyz', 'ref_vel',
                                  'smode', 'error'}
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


if __name__ == '__main__':
    main()
