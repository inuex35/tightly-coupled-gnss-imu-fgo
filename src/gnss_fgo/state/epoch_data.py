"""Structured context passed through the Phase 2 pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class EpochData:
    """Per-epoch mutable context shared by the five Phase 2 stages."""

    # ── Raw epoch inputs ──────────────────────────────────────────
    obs: Any
    obsb: Any
    rs: np.ndarray
    vs: np.ndarray
    rsb: np.ndarray
    sat: np.ndarray
    el: np.ndarray
    iu: np.ndarray
    obs_sd: Any
    ir_map: dict
    info: dict
    ns: int
    init_ecef: np.ndarray
    R_enu2ecef: np.ndarray

    # ── Filled by `imu_prediction` ────────────────────────────────
    key_idx: int | None = None
    tow: float | None = None
    imu_idx_prev: int | None = None
    pim: Any = None
    n_imu: int = 0
    gyro_mean: np.ndarray | None = None
    graph: Any = None
    values: Any = None
    estimate: Any = None
    pose_p: Any = None
    vel_prev: np.ndarray | None = None
    bias_prev: Any = None
    pred_nav: Any = None

    # ── Filled by `quality_gate` ──────────────────────────────────
    pred_ecef: np.ndarray | None = None

    # ── Filled by `solve` ─────────────────────────────────────────
    nv: int = 0
    pose_tc: Any = None
    ecef_tc: np.ndarray | None = None
    nb: int = 0
    xa: np.ndarray | None = None

    # ── Filled by `validate_fix` / `report` ───────────────────────
    sol: np.ndarray | None = None
    tag: str | None = None
