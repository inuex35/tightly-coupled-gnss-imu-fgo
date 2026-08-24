"""Structured context passed through the Phase 2 pipeline stages.

Write convention (enforced by review, checked by stage_contract):
EpochData fields are written ONLY at stage top-level functions —
``run(tc, epoch)`` of Stages A/B/D/E and the C1–C4 sub-stage functions.
Helpers below them are pure: they return values, the stage applies
them. Stage A is the initializer and populates most fields; later
stages own the fields their STAGE_WRITES tuple declares.

The five stages run in sequence:

    preprocess  →  gate  →  optimize  →  postprocess  →  output

Each stage reads a subset of the fields populated by previous stages and
writes its own outputs onto the shared ``EpochData``. The formal
data-flow declaration is each stage module's ``STAGE_READS`` /
``STAGE_WRITES`` tuples, checked by ``stage_contract.validate_pipeline()``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class EpochData:
    """Per-epoch mutable context shared by the five Phase 2 stages."""

    # ── Raw epoch inputs (provided by the caller of run_tc_epoch) ─
    obs: Any                       # cssrlib rover Obs
    obsb: Any                      # cssrlib base Obs
    rs:  np.ndarray                # rover sat ECEF (n_sat, 3+)
    vs:  np.ndarray                # rover sat velocity ECEF (n_sat, 3)
    rsb: np.ndarray                # base sat ECEF (n_sat, 3+)
    sat: np.ndarray                # sat-id list (n_sat,)
    el:  np.ndarray                # elevation rad (n_sat,)
    iu:  np.ndarray                # rover obs index (n_sat,)
    obs_sd: Any                    # double-difference partner obs
    ir_map: dict                   # base obs index map
    info: dict                     # per-epoch diagnostics dict
    ns:  int                       # number of usable sats
    init_ecef:  np.ndarray                # initial-pose ECEF (3,)
    R_enu2ecef: np.ndarray         # ENU→ECEF rotation (3,3)

    # ── Filled by `preprocess` ─────────────────────────────────────
    key_idx: int | None = None              # current Phase-2 epoch index
    tow: float | None = None           # GPS time-of-week
    imu_idx_prev: int | None = None    # IMU cursor before PIM build
    pim: Any = None                    # PreintegratedCombinedMeasurements
    n_imu: int = 0                     # samples integrated this epoch
    gyro_mean: np.ndarray | None = None  # body-frame mean gyro (3,)
    graph: Any = None                     # gtsam.NonlinearFactorGraph
    values: Any = None                     # gtsam.Values (initial vals)
    estimate: Any = None                   # gtsam.Values (smoother est)
    pose_p:  Any = None                # gtsam.Pose3 — Xpose(key_idx-1)
    vel_prev:   np.ndarray | None = None  # vel(key_idx-1)
    bias_prev:  Any = None                # imuBias.ConstantBias(key_idx-1)
    pred_nav: Any = None               # gtsam.NavState (predicted)

    # ── Filled by `gate` ──────────────────────────────────────────
    skip_cp_now: bool = False
    pred_ecef: np.ndarray | None = None  # IMU-pred antenna ECEF (3,)
    prev_amb_values: dict = field(default_factory=dict)

    # ── Filled by `optimize` ──────────────────────────────────────
    nv: int = 0                        # # DD factors built
    pose_tc:  Any = None               # smoother Pose3 (key_idx)
    ecef_tc:  np.ndarray | None = None # antenna ECEF (3,) at key_idx
    nb: int = 0                        # # accepted DDs after FDE
    xa: np.ndarray | None = None       # LAMBDA-fixed full state vec

    # ── Filled by `postprocess` / `output` ────────────────────────
    sol: np.ndarray | None = None      # output ECEF (3,)
    tag: str | None = None             # 'FIX' / 'FLT'
