"""Batch front end: RINEX -> DD rows + rover Doppler rows (see batch.rows).

Every rover epoch is a keyframe. Epochs with a base observation (held up to
nav.maxtdiff) emit one DD row per (system, frequency, target satellite), with
the per-epoch reference = highest elevation. Each row carries the rover/base
pseudoranges and carriers of reference and target, the four satellite
positions (rover-time and base-time), the SD ambiguity arc ids and the
satellite numbers. Every epoch also emits rover Doppler rows.

A new SD ambiguity arc starts when a satellite/frequency was not used in the
previous epoch, when LLI bit 0 is set (rover, or base on a fresh base epoch),
or when the rover L1-L2 geometry-free combination jumps by more than gf_jump.
"""
import time

import numpy as np
import cssrlib.gnss as gn
import cssrlib.rinex as rn
from cssrlib.ephemeris import satposs
from cssrlib.gnss import rSigRnx, sat2prn, uGNSS, uTYP
from cssrlib.rtk import rtkpos

from . import rows as R

SYSTEMS = (uGNSS.GPS, uGNSS.GAL, uGNSS.QZS)
# PPC-Dataset: Septentrio mosaic-X5 rover, Trimble base (Galileo/QZSS track
# different signal components at the two receivers; they cancel in the DD).
SIGS_ROVER = ["GC1C", "GC2W", "GL1C", "GL2W", "GS1C", "GS2W", "GD1C", "GD2W",
              "EC1C", "EC5Q", "EL1C", "EL5Q", "ES1C", "ES5Q", "ED1C", "ED5Q",
              "JC1C", "JC2L", "JL1C", "JL2L", "JS1C", "JS2L", "JD1C", "JD2L"]
SIGS_BASE = ["GC1C", "GC2W", "GL1C", "GL2W", "GS1C", "GS2W",
             "EC1X", "EC5X", "EL1X", "EL5X", "ES1X", "ES5X",
             "JC1C", "JC2X", "JL1C", "JL2X", "JS1C", "JS2X"]


def _up(pos_ecef):
    llh = gn.ecef2pos(np.asarray(pos_ecef, float))
    return np.array([np.cos(llh[0]) * np.cos(llh[1]),
                     np.cos(llh[0]) * np.sin(llh[1]), np.sin(llh[0])])


def _doppler_rows(out, k, obs, rs, vs, pos0, up, elmin):
    for i, s in enumerate(obs.sat):
        s = int(s)
        sys_ = sat2prn(s)[0]
        if sys_ not in SYSTEMS or not np.any(rs[i, :3]):
            continue
        los = rs[i, :3] - pos0
        el = np.arcsin(los @ up / np.linalg.norm(los))
        if el < elmin:
            continue
        for f in range(obs.D.shape[1]):
            if obs.D[i, f] != 0.0:
                lam = obs.sig[sys_][uTYP.L][f].wavelength()
                out.append((k, s, obs.D[i, f], lam, el, *rs[i, :3], *vs[i, :3]))
                break


def prepare(rover_obs, base_obs, nav_file, base_ecef=None, max_epochs=None,
            elmin_deg=15.0, gf_jump=0.05, sigs_rover=SIGS_ROVER, sigs_base=SIGS_BASE,
            log=print):
    """Decode the RINEX files and build the batch measurement arrays.

    base_ecef: base antenna ECEF [m]; default = base.obs header position.
    Returns dict(rows, dop, tow, rb, pos0).
    """
    dec, decb = rn.rnxdec(), rn.rnxdec()
    dec.setSignals([rSigRnx(s) for s in sigs_rover])
    decb.setSignals([rSigRnx(s) for s in sigs_base])
    nav = gn.Nav()
    dec.decode_nav(nav_file, nav)
    dec.decode_obsh(rover_obs)
    decb.decode_obsh(base_obs)
    rb = np.asarray(base_ecef if base_ecef is not None else decb.pos, float)
    nav.rb = list(rb)
    nav.elmin = np.deg2rad(elmin_deg)
    rtk = rtkpos(nav, dec.pos)
    pos0 = np.array(dec.pos, float)
    nav.x[0:3] = pos0
    up = _up(pos0)

    rows, dop, tow = [], [], []
    arc_of, last_used, last_gf = {}, {}, {}
    last_tb, n_arc = None, 0
    t0 = time.perf_counter()
    for ne, (obs, obsb, _) in enumerate(rn.sync_obs_hold(dec, decb, maxage=nav.maxtdiff)):
        if max_epochs and ne >= max_epochs:
            break
        k = len(tow)
        tow.append(gn.time2gpst(obs.t)[1])
        dd = None
        if obsb is not None:
            dd = rtk.prepare_double_difference_measurements(obs, obsb, pos_pred=dec.pos)
        if dd is not None:
            rs, vs = dd.rs, dd.vs
        elif len(obs.sat):
            rs, vs, _, _, _ = satposs(obs, nav)
        else:
            continue
        _doppler_rows(dop, k, obs, rs, vs, pos0, up, nav.elmin)
        if dd is None:
            continue
        tb = gn.time2gpst(obsb.t)[1]
        fresh_base, last_tb = tb != last_tb, tb

        slip_gf = set()   # rover L1-L2 geometry-free jump
        for kk, s in enumerate(dd.sat):
            s, iu = int(s), dd.iu[kk]
            sys_ = sat2prn(s)[0]
            if sys_ not in SYSTEMS or obs.L[iu, 0] == 0.0 or obs.L[iu, 1] == 0.0:
                continue
            gf = (obs.L[iu, 0] * obs.sig[sys_][uTYP.L][0].wavelength()
                  - obs.L[iu, 1] * obs.sig[sys_][uTYP.L][1].wavelength())
            prev = last_gf.get(s)
            if prev is not None and prev[0] == k - 1 and abs(gf - prev[1]) > gf_jump:
                slip_gf.add(s)
            last_gf[s] = (k, gf)

        def arc(s, f, iu, ir):
            nonlocal n_arc
            key = (s, f)
            slip = (last_used.get(key) != k - 1 or (obs.lli[iu, f] & 1)
                    or (fresh_base and (obsb.lli[ir, f] & 1)) or s in slip_gf)
            if key not in arc_of or slip:
                arc_of[key], n_arc = n_arc, n_arc + 1
            return arc_of[key]

        by_sys = {}
        for kk, s in enumerate(dd.sat):
            sys_ = sat2prn(int(s))[0]
            if sys_ in SYSTEMS and dd.el[kk] >= nav.elmin:
                by_sys.setdefault(sys_, []).append(kk)
        used_now = []
        for sys_, ks in by_sys.items():
            for f in range(nav.nf):
                lam = obs.sig[sys_][uTYP.L][f].wavelength()
                ok = [kk for kk in ks
                      if 0.0 not in (obs.P[dd.iu[kk], f], obsb.P[dd.ir[kk], f],
                                     obs.L[dd.iu[kk], f], obsb.L[dd.ir[kk], f])]
                if len(ok) < 2:
                    continue
                ridx = max(ok, key=lambda kk: dd.el[kk])
                sr, iur, irr = int(dd.sat[ridx]), dd.iu[ridx], dd.ir[ridx]
                a_ref = arc(sr, f, iur, irr)
                used_now.append((sr, f))
                for kk in ok:
                    if kk == ridx:
                        continue
                    st, iut, irt = int(dd.sat[kk]), dd.iu[kk], dd.ir[kk]
                    a_tgt = arc(st, f, iut, irt)
                    used_now.append((st, f))
                    w = 1.0 / max(np.sin(min(dd.el[kk], dd.el[ridx])), 0.1)
                    rows.append((
                        k, f, lam, w, a_ref, a_tgt,
                        obs.P[iur, f], obsb.P[irr, f], obs.P[iut, f], obsb.P[irt, f],
                        obs.L[iur, f] * lam, obsb.L[irr, f] * lam,
                        obs.L[iut, f] * lam, obsb.L[irt, f] * lam,
                        *dd.rs[iur, :3], *dd.rs[iut, :3],
                        *dd.rsb[irr, :3], *dd.rsb[irt, :3], sr, st))
        for key in used_now:
            last_used[key] = k
    out = dict(rows=np.array(rows, dtype=float).reshape(-1, R.N_DD_COLS),
               dop=np.array(dop, dtype=float).reshape(-1, R.N_DOP_COLS),
               tow=np.array(tow), rb=rb, pos0=pos0)
    log(f"front end: {len(tow)} epochs, {len(rows)} DD rows, {len(dop)} Doppler rows, "
        f"{n_arc} arcs ({time.perf_counter() - t0:.1f}s)")
    return out


def save(path, data):
    np.savez_compressed(path, **data)


def load(path):
    z = np.load(path)
    return {key: z[key] for key in ("rows", "dop", "tow", "rb", "pos0")}
