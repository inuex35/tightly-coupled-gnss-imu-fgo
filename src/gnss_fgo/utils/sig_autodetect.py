"""Auto-detect compatible RINEX signal pairs from rover/base obs headers.

Mimics RTKLIB's behaviour of using whatever signals the obs file declares,
without requiring the caller to hand-craft per-system signal lists.

Typical usage:

    dec = rnxdec(); dec.decode_obsh(rover_obs)
    decb = rnxdec(); decb.decode_obsh(base_obs)
    sigs, sigsb = auto_detect_signals(dec.sig_map, decb.sig_map, max_freq=2)
    dec.setSignals(sigs); decb.setSignals(sigsb)
"""
from cssrlib.gnss import uTYP

# Band priority — pick lowest-numbered bands first when more than max_freq
# are common (L1 always preferred, then L2, then L5/L6/L7/L8).
_BAND_PRIORITY = (1, 2, 5, 7, 6, 8, 3, 4, 9)


def _group_by_band(sigs, typ):
    """Group rSigRnx values from a sig_map system by their frequency band.
    Returns {band: rSigRnx} for the requested observation type."""
    out = {}
    for s in sigs.values():
        if s.typ != typ:
            continue
        band = int(s.sig) // 100
        # First-seen wins per (typ, band) — sig_map preserves header order.
        out.setdefault(band, s)
    return out


def auto_detect_signals(sig_map_rov, sig_map_base, max_freq=2,
              required=(uTYP.C, uTYP.L, uTYP.S),
              systems=None, strict_freq=True):
    """Build rover/base signal lists with matching (sys, typ, band) coverage.

    Args:
      sig_map_rov, sig_map_base: rnxdec.sig_map populated by decode_obsh.
      max_freq: number of frequency bands to keep per system (RTKLIB nf).
      required: observation types each frequency band must provide on both
        rover and base (default C+L+S — pseudorange, phase, SNR).
      systems: iterable of uGNSS values to consider; default = all systems
        present on both rover and base.
      strict_freq: drop systems that can't supply max_freq common bands.
        cssrlib's RTK indexes sigsCN[f] up to nav.nf-1 unconditionally, so
        a system with fewer than max_freq bands triggers IndexError. Set
        False if the downstream consumer is happy with partial coverage.

    Returns:
      (sigs, sigsb) — each a list of rSigRnx ready for setSignals().
    """
    rov_systems = set(sig_map_rov.keys())
    base_systems = set(sig_map_base.keys())
    if systems is None:
        systems = rov_systems & base_systems
    else:
        systems = set(systems) & rov_systems & base_systems

    sigs, sigsb = [], []
    for sys in systems:
        rov_by_typ = {t: _group_by_band(sig_map_rov[sys], t) for t in required}
        base_by_typ = {t: _group_by_band(sig_map_base[sys], t) for t in required}
        # Bands with full coverage on both sides for every required type.
        common_bands = set(rov_by_typ[required[0]].keys())
        for t in required[1:]:
            common_bands &= rov_by_typ[t].keys()
        for t in required:
            common_bands &= base_by_typ[t].keys()
        if not common_bands:
            continue
        if strict_freq and len(common_bands) < max_freq:
            continue
        # Prefer canonical band order (L1 then L2 then L5 ...).
        ordered = [b for b in _BAND_PRIORITY if b in common_bands]
        ordered += sorted(b for b in common_bands if b not in ordered)
        for band in ordered[:max_freq]:
            for t in required:
                sigs.append(rov_by_typ[t][band])
                sigsb.append(base_by_typ[t][band])
    return sigs, sigsb
