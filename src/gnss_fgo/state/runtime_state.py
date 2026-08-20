"""Structured runtime state for the TC pipeline."""

from enum import Enum


class TcState(Enum):
    """Five-state FSM landing point. Not yet wired into the runner —
    transitions still happen via the flag fields below."""
    NORMAL    = 'normal'      # Fix flowing, residuals clean.
    DEGRADED  = 'degraded'    # Some sat outliers / CP-hold engaged.
    DR        = 'dr'          # Dead reckoning — GDOP skip, IMU-only update.
    INS_ONLY  = 'ins_only'    # Long outage, FLS frozen.
    RECOVERY  = 'recovery'    # Just warm-reset, gating new measurements.


from collections.abc import MutableMapping
from dataclasses import dataclass, field
from typing import Optional


_SENTINEL = object()


class SatFieldView(MutableMapping):
    """``MutableMapping`` view of one ``SatState`` field across all entries."""

    __slots__ = ('_map', '_field', '_absent', '_list_default')

    def __init__(self, sat_map, field_name, absent=None):
        self._map = sat_map
        self._field = field_name
        self._absent = absent
        # mutable defaults must be created fresh, not shared
        self._list_default = isinstance(absent, list)

    def _is_set(self, v):
        if self._list_default:
            return bool(v)
        return v != self._absent

    def _reset(self, st):
        setattr(st, self._field, [] if self._list_default else self._absent)

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
        st = self._map.track.get(key)
        if st is None or not self._is_set(getattr(st, self._field)):
            raise KeyError(key)
        self._reset(st)

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

    def pop(self, key, default=_SENTINEL):
        st = self._map.track.get(key)
        if st is None or not self._is_set(getattr(st, self._field)):
            if default is _SENTINEL:
                raise KeyError(key)
            return default
        v = getattr(st, self._field)
        self._reset(st)
        return v

    def setdefault(self, key, default):
        st = self._map.track.get(key)
        if st is not None and self._is_set(getattr(st, self._field)):
            return getattr(st, self._field)
        s, f = key
        setattr(self._map.get(s, f), self._field, default)
        return default

    def clear(self):
        for st in self._map.track.values():
            self._reset(st)


@dataclass
class SatState:
    """All bookkeeping for a single (sat, freq) pair."""

    # Slip-detector memory
    cmc: Optional[float] = None              # CMC observation [m]
    prev_phase: Optional[tuple] = None       # (cycles, tow_s) for Doppler-phase slip
    outc: int = 0                            # epochs since last seen
    # Ambiguity bookkeeping
    amb_key: Optional[int] = None            # GTSAM symbol for N
    amb_gen: int = 0                         # generation counter (++ on slip / reset)
    amb_lam: float = 0.0                     # wavelength [m]
    amb_init_epoch: Optional[int] = None     # epoch when N was last initialised
    held_value: Optional[float] = None       # conditioned-out held integer [cyc]
    last_held_value: Optional[float] = None  # last held integer for float re-seed [cyc]
    release_seed_pending: bool = False       # one-shot unary prior on first float epoch
    # Per-(sat, freq) quality counters
    fix_streak: int = 0                      # consecutive Fix epochs

    def activate_hold(self, value: float) -> None:
        """Move this ambiguity out of the graph and pin its integer externally."""
        held = float(value)
        self.held_value = held
        self.last_held_value = held
        self.release_seed_pending = False

    def release_hold(self, seed: bool = True) -> None:
        """Drop active hold, optionally queue a one-shot float re-seed prior."""
        if self.held_value is not None:
            self.last_held_value = float(self.held_value)
        self.held_value = None
        self.release_seed_pending = bool(seed and self.last_held_value is not None)

    def clear_hold(self) -> None:
        """Forget all held-integer state (slip / reset / outage path)."""
        self.held_value = None
        self.last_held_value = None
        self.release_seed_pending = False


_EMPTY_SAT_STATE = SatState()


@dataclass
class SatStateMap:
    """Per-(sat, freq) ``SatState`` map + per-sat GF / MW state."""

    track: dict = field(default_factory=dict)            # {(sat,f): SatState}
    gf: dict = field(default_factory=dict)               # {sat: gf [m]}
    maxout: int = 5

    def get(self, sat: int, freq: int) -> SatState:
        """Get-or-create the ``SatState`` for ``(sat, freq)``."""
        key = (int(sat), int(freq))
        st = self.track.get(key)
        if st is None:
            st = SatState()
            self.track[key] = st
        return st

    def at(self, sat: int, freq: int) -> SatState:
        """Return ``SatState`` for ``(sat, freq)`` *without* creating."""
        return self.track.get((int(sat), int(freq)), _EMPTY_SAT_STATE)

    def items(self):
        return self.track.items()

    def keys(self):
        return self.track.keys()

    def values(self):
        return self.track.values()

    # --- Ambiguity convenience views (for hot-loop iteration) ---

    def amb_items(self):
        """Iterate ``((sat, f), amb_key)`` for entries with non-None amb_key."""
        for k, st in self.track.items():
            if st.amb_key is not None:
                yield k, st.amb_key

    def amb_keys_dict(self) -> dict:
        """Snapshot ``{(sat, f): amb_key}`` for entries with non-None amb_key."""
        return {k: st.amb_key for k, st in self.track.items()
                if st.amb_key is not None}

    def amb_count(self) -> int:
        """Number of entries with non-None amb_key."""
        return sum(1 for st in self.track.values() if st.amb_key is not None)

    def held_items(self):
        """Iterate ``((sat, f), held_value)`` for active conditioned-out holds."""
        for k, st in self.track.items():
            if st.held_value is not None:
                yield k, st.held_value

    def amb_key_values(self) -> list:
        """List of all non-None amb_key values (for FLS keep_keys)."""
        return [st.amb_key for st in self.track.values()
                if st.amb_key is not None]

@dataclass
class RecoveryState:
    """Cross-cutting recovery / degradation flags."""

    skip_count: int = 0
    recov_cp_hold: int = 0
    cp_hold_retrigger_streak: int = 0   # consecutive re-arms while active (loop audit)
    pim_discontinuity: bool = False
    ddpr_bad_count: int = 0
    zupt_anchor_pose: object = None
    zupt_anchor_start_ep: int | None = None

    def start_cp_hold(self, hold_n, reason, info, value=None,
                      skip_if_active=False):
        """Engage global CP-hold for ``hold_n`` epochs (idempotent max).

        skip_if_active prevents re-trigger during an active hold —
        fde/innovation would otherwise fire every epoch of the recovery
        and loop forever.
        """
        if self.recov_cp_hold > 0:
            self.cp_hold_retrigger_streak += 1
            info['cp_hold_retrigger_streak'] = self.cp_hold_retrigger_streak
        else:
            self.cp_hold_retrigger_streak = 0
        if skip_if_active and self.recov_cp_hold > 0:
            return False
        self.recov_cp_hold = max(self.recov_cp_hold, hold_n)
        info[f'cp_hold_{reason}'] = value if value is not None else True
        return True

    def tick_cp_hold(self, info):
        """One CP-hold countdown step (call only while the hold is active)."""
        self.recov_cp_hold -= 1
        info['recov_cp_hold'] = self.recov_cp_hold + 1

    def reset(self):
        self.skip_count = 0
        self.recov_cp_hold = 0
        self.cp_hold_retrigger_streak = 0
        self.pim_discontinuity = False
        self.ddpr_bad_count = 0
        self.zupt_anchor_pose = None
        self.zupt_anchor_start_ep = None


@dataclass
class MresSignalsState:
    """Snapshot of the previous-epoch main DDPR residuals."""

    last_res: float = 0.0
    per_sat: dict = field(default_factory=dict)
    epoch: int = -10**9

    def update(self, last_res: float, per_sat: dict, epoch: int) -> None:
        self.last_res = last_res
        self.per_sat = per_sat
        self.epoch = epoch

    def reset(self) -> None:
        self.last_res = 0.0
        self.per_sat = {}
        self.epoch = -10**9
