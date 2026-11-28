"""Batch RINEX front end, NPZ serialization, and tropospheric correction."""

import time

import cssrlib.gnss as gn
import cssrlib.rinex as rn
import numpy as np
from cssrlib.ephemeris import satposs
from cssrlib.gnss import ecef2pos, rSigRnx, sat2prn, uGNSS, uTYP
from cssrlib.rtk import rtkpos

from .config import _TC_DEFAULTS
from .factors import _is_bds_geo, dd_snr_weight
from .slip import detect_dop_slip, detect_gf_slip
from .utils import auto_detect_signals


# ── Row layout for batch measurement arrays ───────────────────────────

EPOCH, FREQ, LAM, WEIGHT, ARC_REF, ARC_TGT = 0, 1, 2, 3, 4, 5
PR_RR, PR_BR, PR_RT, PR_BT = 6, 7, 8, 9          # pseudoranges [m]: rover/base x ref/tgt
CP_RR, CP_BR, CP_RT, CP_BT = 10, 11, 12, 13      # carriers [m]
SAT_RR, SAT_RT, SAT_BR, SAT_BT = (slice(14, 17), slice(17, 20),   # sat ECEF at rover
                                  slice(20, 23), slice(23, 26))   # / base signal time
SATNO_REF, SATNO_TGT = 26, 27
N_DD_COLS = 28

D_EPOCH, D_SAT, D_HZ, D_LAM, D_EL = 0, 1, 2, 3, 4
D_SATPOS, D_SATVEL = slice(5, 8), slice(8, 11)
D_SNR = 11
N_DOP_COLS = 12

SYSTEMS = (uGNSS.GPS, uGNSS.GAL, uGNSS.QZS, uGNSS.BDS)


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
        has_s = hasattr(obs, "S")
        for f in range(obs.D.shape[1]):
            if obs.D[i, f] != 0.0:
                lam = obs.sig[sys_][uTYP.L][f].wavelength()
                snr = float(obs.S[i, f]) if has_s else 0.0
                out.append((k, s, obs.D[i, f], lam, el, *rs[i, :3], *vs[i, :3], snr))
                break


def prepare(rover_obs, base_obs, nav_file, base_ecef=None, max_epochs=None,
            elmin_deg=15.0, gf_jump=_TC_DEFAULTS.thres_slip,
            sigs_rover=None, sigs_base=None, log=print):
    """Decode the RINEX files and build the batch measurement arrays."""
    nav_nf = 3
    dec, decb = rn.rnxdec(), rn.rnxdec()
    nav = gn.Nav(nf=nav_nf)
    dec.decode_nav(nav_file, nav)
    dec.decode_obsh(rover_obs)
    decb.decode_obsh(base_obs)
    if sigs_rover is not None and sigs_base is not None:
        dec.setSignals([rSigRnx(s) for s in sigs_rover])
        decb.setSignals([rSigRnx(s) for s in sigs_base])
    else:
        sigs, sigsb = auto_detect_signals(
            dec.sig_map, decb.sig_map, max_freq=nav_nf,
            required=(uTYP.C, uTYP.L, uTYP.S),
            systems=list(SYSTEMS), strict_freq=False)
        rov_picks_by_sys = {}
        for s in sigs:
            if s.typ != uTYP.L:
                continue
            band = int(s.sig) // 100
            bands = rov_picks_by_sys.setdefault(s.sys, [])
            if band not in bands:
                bands.append(band)
        for sys_id, bands in rov_picks_by_sys.items():
            rov_typ_d = {int(s.sig) // 100: s
                         for s in dec.sig_map.get(sys_id, {}).values()
                         if s.typ == uTYP.D}
            for band in bands:
                if band in rov_typ_d:
                    sigs.append(rov_typ_d[band])
        dec.setSignals(sigs)
        decb.setSignals(sigsb)
    rb = np.asarray(base_ecef if base_ecef is not None else decb.pos, float)
    nav.rb = list(rb)
    nav.elmin = np.deg2rad(elmin_deg)
    rtk = rtkpos(nav, dec.pos)
    pos0 = np.array(dec.pos, float)
    nav.x[0:3] = pos0
    up = _up(pos0)

    rows, dop, tow = [], [], []
    arc_of, last_used, last_gf, last_dop = {}, {}, {}, {}
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

        slip_gf = set()
        slip_dop = set()
        tow_k = tow[-1]
        for kk, s in enumerate(dd.sat):
            s, iu = int(s), dd.iu[kk]
            sys_ = sat2prn(s)[0]
            if sys_ not in SYSTEMS:
                continue
            lams_s = [sig.wavelength() for sig in obs.sig[sys_][uTYP.L]]
            if detect_gf_slip(obs, iu, s, lams_s, nav.nf, last_gf, gf_jump, epoch_idx=k):
                slip_gf.add(s)
            for f in detect_dop_slip(obs, iu, s, nav.nf, tow_k, last_dop):
                slip_dop.add((s, f))

        def arc(s, f, iu, ir):
            nonlocal n_arc
            key = (s, f)
            slip = (last_used.get(key) != k - 1 or (obs.lli[iu, f] & 1)
                    or (fresh_base and (obsb.lli[ir, f] & 1))
                    or s in slip_gf or key in slip_dop)
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
                pool = [kk for kk in ok if not (sys_ == uGNSS.BDS and _is_bds_geo(int(dd.sat[kk])))] or ok
                ridx = max(pool, key=lambda kk: dd.el[kk])
                sr, iur, irr = int(dd.sat[ridx]), dd.iu[ridx], dd.ir[ridx]
                a_ref = arc(sr, f, iur, irr)
                used_now.append((sr, f))
                for kk in ok:
                    if kk == ridx:
                        continue
                    st, iut, irt = int(dd.sat[kk]), dd.iu[kk], dd.ir[kk]
                    a_tgt = arc(st, f, iut, irt)
                    used_now.append((st, f))
                    has_sr, has_sb = hasattr(obs, "S"), hasattr(obsb, "S")
                    snrs = (
                        obs.S[iur, f] if has_sr else 0.0,
                        obsb.S[irr, f] if has_sb else 0.0,
                        obs.S[iut, f] if has_sr else 0.0,
                        obsb.S[irt, f] if has_sb else 0.0,
                    )
                    w = dd_snr_weight(snrs)
                    rows.append((
                        k, f, lam, w, a_ref, a_tgt,
                        obs.P[iur, f], obsb.P[irr, f], obs.P[iut, f], obsb.P[irt, f],
                        obs.L[iur, f] * lam, obsb.L[irr, f] * lam,
                        obs.L[iut, f] * lam, obsb.L[irt, f] * lam,
                        *dd.rs[iur, :3], *dd.rs[iut, :3],
                        *dd.rsb[irr, :3], *dd.rsb[irt, :3], sr, st))
        for key in used_now:
            last_used[key] = k
    out = dict(rows=np.array(rows, dtype=float).reshape(-1, N_DD_COLS),
               dop=np.array(dop, dtype=float).reshape(-1, N_DOP_COLS),
               tow=np.array(tow), rb=rb, pos0=pos0)
    log(f"front end: {len(tow)} epochs, {len(rows)} DD rows, {len(dop)} Doppler rows, "
        f"{n_arc} arcs ({time.perf_counter() - t0:.1f}s)")
    return out


def save(path, data):
    np.savez_compressed(path, **data)


def load(path):
    z = np.load(path)
    return {key: z[key] for key in ("rows", "dop", "tow", "rb", "pos0")}


def _trop_saast(llh, sin_el, humi=0.7):
    lat, hgt = llh[0], np.maximum(llh[2], 0.0)
    pres = 1013.25 * (1.0 - 2.2557e-5 * hgt) ** 5.2568
    temp = 15.0 - 6.5e-3 * hgt + 273.16
    e = 6.108 * humi * np.exp((17.15 * temp - 4684.0) / (temp - 38.45))
    zhd = 0.0022768 * pres / (1.0 - 0.00266 * np.cos(2.0 * lat) - 0.00028 * hgt / 1e3)
    zwd = 0.002277 * (1255.0 / temp + 0.05) * e
    return (zhd + zwd) / np.maximum(sin_el, 0.1)


def tropo_correct(rows, rover_ecef, rb):
    """Copy of the DD rows with modelled tropo removed from all eight PR/CP."""
    out = rows.copy()
    xr = rover_ecef[rows[:, EPOCH].astype(int)].copy()
    xr[~np.all(np.isfinite(xr), axis=1)] = rb
    llh_r = np.array([ecef2pos(p) for p in xr]).T
    llh_b = np.array(ecef2pos(rb))

    def delay(sat, x, llh):
        lat, lon = llh[0], llh[1]
        up = np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)
        los = sat - x
        return _trop_saast(llh, np.sum(los * up, axis=-1) / np.linalg.norm(los, axis=-1))

    t = (delay(rows[:, SAT_RR], xr, llh_r), delay(rows[:, SAT_BR], rb, llh_b),
         delay(rows[:, SAT_RT], xr, llh_r), delay(rows[:, SAT_BT], rb, llh_b))
    for cols in ((PR_RR, PR_BR, PR_RT, PR_BT), (CP_RR, CP_BR, CP_RT, CP_BT)):
        for c, d in zip(cols, t):
            out[:, c] -= d
    return out
