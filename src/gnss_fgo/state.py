"""Structured runtime state for the TC pipeline."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SatState:
    """Bookkeeping for a single (sat, freq) continuous carrier-phase arc."""

    outc: int = 0                    # epochs since last seen
    amb_key: Optional[int] = None    # persistent GTSAM symbol for this arc's N
    amb_gen: int = 0                 # monotonic arc generation (++ on slip / reset)


@dataclass
class SatStateMap:
    """Per-(sat, freq) ``SatState`` map, per-sat GF slip state, and DD ref sats."""

    track: dict = field(default_factory=dict)            # {(sat,f): SatState}
    gf: dict = field(default_factory=dict)               # {sat: gf [m]}
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

    def values(self):
        return self.track.values()

    def amb_keys_dict(self) -> dict:
        return {k: st.amb_key for k, st in self.track.items()
                if st.amb_key is not None}

    def amb_count(self) -> int:
        return sum(1 for st in self.track.values() if st.amb_key is not None)

    def amb_key_values(self) -> list:
        return [st.amb_key for st in self.track.values()
                if st.amb_key is not None]
