"""Structured runtime state for the TC pipeline."""

from collections.abc import MutableMapping
from dataclasses import dataclass, field
from typing import Optional


class SatFieldView(MutableMapping):
    """``MutableMapping`` view of one ``SatState`` field across all entries."""

    __slots__ = ('_map', '_field', '_absent')

    def __init__(self, sat_map, field_name, absent=None):
        self._map = sat_map
        self._field = field_name
        self._absent = absent

    def _is_set(self, v):
        return v != self._absent

    def _reset(self, st):
        setattr(st, self._field, self._absent)

    def __getitem__(self, key):
        st = self._map.track.get(key)
        if st is None:
            raise KeyError(key)
        v = getattr(st, self._field)
        if not self._is_set(v):
            raise KeyError(key)
        return v

    def __setitem__(self, key, value):
        s, f = key
        setattr(self._map.get(s, f), self._field, value)

    def __delitem__(self, key):
        raise TypeError('SatFieldView entries are reset via clear()')

    def __iter__(self):
        for k, st in self._map.track.items():
            if self._is_set(getattr(st, self._field)):
                yield k

    def __len__(self):
        return sum(1 for _ in iter(self))

    def __contains__(self, key):
        st = self._map.track.get(key)
        return st is not None and self._is_set(getattr(st, self._field))

    def get(self, key, default=None):
        st = self._map.track.get(key)
        if st is None:
            return default
        v = getattr(st, self._field)
        return v if self._is_set(v) else default

    def values(self):
        return [getattr(st, self._field) for st in self._map.track.values()
                if self._is_set(getattr(st, self._field))]

    def items(self):
        return [(k, getattr(st, self._field)) for k, st in self._map.track.items()
                if self._is_set(getattr(st, self._field))]

    def keys(self):
        return list(iter(self))

    def clear(self):
        for st in self._map.track.values():
            self._reset(st)


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

    ar_joint_cache: Optional[dict] = None
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
