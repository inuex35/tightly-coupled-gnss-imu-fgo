"""Satellite arc bookkeeping and cycle-slip detection."""

from dataclasses import dataclass, field
from typing import Optional

from cssrlib.gnss import time2gpst

from .utils import get_wavelengths

_DOP_SLIP_THRES_CYC = 1.0  # Doppler-phase cycle-slip threshold [cycles]
_DOP_SLIP_MAX_DT_S = 1.0   # max epoch gap [s] for Doppler-integrated phase


@dataclass
class SatState:
    """Bookkeeping for a single (sat, freq) continuous carrier-phase arc."""

    outage_count: int = 0            # epochs since last seen
    amb_key: Optional[int] = None    # persistent GTSAM symbol for this arc's N
    amb_gen: int = 0                 # monotonic arc generation (++ on slip / reset)


@dataclass
class SatStateMap:
    """Per-(sat, freq) ``SatState`` map, slip detectors' history, and DD ref sats."""

    track: dict = field(default_factory=dict)            # {(sat,f): SatState}
    prev_gf_m: dict = field(default_factory=dict)        # {(sat,fa,fb): gf [m]}
    prev_dop_phi: dict = field(default_factory=dict)     # {(sat,f): (L, D, tow)}
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


def _detect_gf_slip(tc, obs, row, s, lams, nf, reset_keys):
    """Multi-frequency geometry-free (La-Lb) jump detector."""
    n_f = min(nf, len(lams), obs.L.shape[1])
    prev_gf = tc._sat_states.prev_gf_m
    for fa in range(n_f):
        La = float(obs.L[row, fa])
        if La == 0.0 or lams[fa] <= 0.0:
            continue
        for fb in range(fa + 1, n_f):
            Lb = float(obs.L[row, fb])
            if Lb == 0.0 or lams[fb] <= 0.0:
                continue
            gf = La * lams[fa] - Lb * lams[fb]
            key = (int(s), fa, fb)
            if key in prev_gf and prev_gf[key] != 0.0:
                if abs(gf - prev_gf[key]) > tc.cfg.thres_slip:
                    for f in range(nf):
                        reset_keys.add((s, f))
            prev_gf[key] = gf


def _detect_dop_slip(tc, obs, row, s, nf, tow, reset_keys):
    """Per-frequency Doppler-aided cycle-slip detector: |dPhi + D_mean*dt|."""
    if tow is None or not hasattr(obs, "D"):
        return
    n_f = min(nf, obs.L.shape[1], obs.D.shape[1])
    prev_dop = tc._sat_states.prev_dop_phi
    for f in range(n_f):
        L_val = float(obs.L[row, f])
        D_val = float(obs.D[row, f])
        if L_val == 0.0 or D_val == 0.0:
            continue
        key = (int(s), f)
        if key in prev_dop:
            pL, pD, ptow = prev_dop[key]
            dt = tow - ptow
            if 0.0 < dt <= _DOP_SLIP_MAX_DT_S:
                if abs((L_val - pL) + 0.5 * (D_val + pD) * dt) > _DOP_SLIP_THRES_CYC:
                    reset_keys.add((s, f))
        prev_dop[key] = (L_val, D_val, tow)


def detect_slips_and_reset_ambiguities(tc, obs, obs_sd, sat, iu):
    """LLI + multi-freq GF + Doppler-phase slip detection plus outage expiry."""
    nf = tc.nav.nf
    ns = len(sat)
    reset_keys = set()
    current_sats = set()
    tow = time2gpst(obs.t)[1] if getattr(obs, "t", None) is not None else None

    for i in range(ns):
        s = sat[i]
        lams = get_wavelengths(obs_sd, s, glo_ch=tc.nav.glo_ch)
        for f in range(nf):
            current_sats.add((s, f))
            sat_state = tc._sat_states.get(s, f)
            sat_state.outage_count = 0
            if hasattr(obs, 'lli') and (int(obs.lli[iu[i], f]) & 1) != 0:
                reset_keys.add((s, f))
        if nf >= 2 and len(lams) >= 2:
            _detect_gf_slip(tc, obs, iu[i], s, lams, nf, reset_keys)
        _detect_dop_slip(tc, obs, iu[i], s, nf, tow, reset_keys)

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
