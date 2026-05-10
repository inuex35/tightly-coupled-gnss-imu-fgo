"""Ambiguity / DD-signal helper utilities (pure functions)."""

from __future__ import annotations

from typing import List

from cssrlib.gnss import sat2prn, uGNSS, uTYP


def get_wavelengths(obs_sd, sat: int, glo_ch: dict | None = None) -> List[float]:
    """Carrier-phase wavelengths [m] for one satellite, indexed by frequency.

    Returns empty list if the satellite's system has no L-signal
    (e.g. a PR-only setup) or isn't tracked in `obs_sd.sig`.

    GLONASS wavelengths depend on the FDMA channel number, looked up in
    `glo_ch` (nav.glo_ch equivalent). Pass the channel map explicitly so
    this function stays pure — default {} means channel 0 for every sat.
    """
    sys_i, _ = sat2prn(sat)
    if sys_i not in obs_sd.sig:
        return []
    if uTYP.L not in obs_sd.sig[sys_i]:
        return []
    sigs = obs_sd.sig[sys_i][uTYP.L]
    if sys_i == uGNSS.GLO:
        ch_map = glo_ch or {}
        return [s.wavelength(ch_map.get(sat, 0)) for s in sigs]
    return [s.wavelength() for s in sigs]
