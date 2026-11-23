"""Structured runtime state for the TC pipeline."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SatState:
    """Bookkeeping for a single (sat, freq) continuous carrier-phase arc."""

    outc: int = 0                    # epochs since last seen
    amb_key: Optional[int] = None    # persistent GTSAM symbol for this arc's N
    amb_gen: int = 0                 # monotonic arc generation (++ on slip / reset)


_EMPTY_SAT_STATE = SatState()


@dataclass
class SatStateMap:
    """Per-(sat, freq) ``SatState`` map + per-sat GF slip state."""

    track: dict = field(default_factory=dict)            # {(sat,f): SatState}
    gf: dict = field(default_factory=dict)               # {sat: gf [m]}
    maxout: int = 5

    def get(self, sat: int, freq: int) -> SatState:
        key = (int(sat), int(freq))
        st = self.track.get(key)
        if st is None:
            st = SatState()
            self.track[key] = st
        return st

    def at(self, sat: int, freq: int) -> SatState:
        return self.track.get((int(sat), int(freq)), _EMPTY_SAT_STATE)

    def items(self):
        return self.track.items()

    def keys(self):
        return self.track.keys()

    def values(self):
        return self.track.values()

    def amb_items(self):
        for k, st in self.track.items():
            if st.amb_key is not None:
                yield k, st.amb_key

    def amb_keys_dict(self) -> dict:
        return {k: st.amb_key for k, st in self.track.items()
                if st.amb_key is not None}

    def amb_count(self) -> int:
        return sum(1 for st in self.track.values() if st.amb_key is not None)

    def amb_key_values(self) -> list:
        return [st.amb_key for st in self.track.values()
                if st.amb_key is not None]


@dataclass
class CurrentEpochState:
    """State that lives exactly one epoch."""

    ref_sats: dict = field(default_factory=dict)
    seeded_amb_keys: set = field(default_factory=set)


@dataclass
class ArDiagnostics:
    """Per-AR-call diagnostic counters."""

    outcome: str = 'not_called'
    exception: Optional[str] = None
    amb_not_in_obs: Optional[int] = None
    sat_in_obs_size: Optional[int] = None
    resamb_raw_nb: Optional[int] = None
    partial_dropped: Optional[int] = None
