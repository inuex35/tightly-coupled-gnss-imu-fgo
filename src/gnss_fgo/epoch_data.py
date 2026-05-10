"""Structured context passed through the Phase 2 pipeline stages.

The five stages run in sequence:

    preprocess  →  gate  →  optimize  →  postprocess  →  output

Each stage reads a subset of the fields populated by previous stages and
writes its own outputs onto the shared ``EpochData``.  Field types
were tightened from ``Any`` to the concrete GTSAM / numpy types during
the Phase 3 contract refactor — see each stage module's
``STAGE_READS`` / ``STAGE_WRITES`` tuples for the formal data-flow
declaration that's checked by ``stage_contract.validate_pipeline()``.
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
    dts: np.ndarray                # rover sat clock bias (n_sat,)
    rsb: np.ndarray                # base sat ECEF (n_sat, 3+)
    sat: np.ndarray                # sat-id list (n_sat,)
    el:  np.ndarray                # elevation rad (n_sat,)
    iu:  np.ndarray                # rover obs index (n_sat,)
    obs_sd: Any                    # double-difference partner obs
    ir_map: dict                   # base obs index map
    ref_vel:  np.ndarray           # truth velocity ENU (3,) — diag only
    ref_ecef: np.ndarray           # truth position ECEF (3,) — diag only
    info: dict                     # per-epoch diagnostics dict
    ns:  int                       # number of usable sats
    init_ecef:  np.ndarray                # initial-pose ECEF (3,)
    R:   np.ndarray                # ECEF→ENU rotation (3,3)

    # ── Filled by `preprocess` ─────────────────────────────────────
    kk: int | None = None              # current Phase-2 epoch index
    is_recovery: bool = False          # epoch follows skip_count > 0
    tow: float | None = None           # GPS time-of-week
    imu_idx_prev: int | None = None    # IMU cursor before PIM build
    pim: Any = None                    # PreintegratedCombinedMeasurements
    n_imu: int = 0                     # samples integrated this epoch
    gyro_mean: np.ndarray | None = None  # body-frame mean gyro (3,)
    g3: Any = None                     # gtsam.NonlinearFactorGraph
    v3: Any = None                     # gtsam.Values (initial vals)
    est2: Any = None                   # gtsam.Values (smoother est)
    pose_p:  Any = None                # gtsam.Pose3 — Xpose(kk-1)
    vel_p:   np.ndarray | None = None  # vel(kk-1)
    bias_p:  Any = None                # imuBias.ConstantBias(kk-1)
    pred:    Any = None                # gtsam.NavState (predicted)

    # ── Filled by `gate` ──────────────────────────────────────────
    remove_indices: list = field(default_factory=list)
    slip_keys: set = field(default_factory=set)
    skip_cp_now: bool = False
    pred_enu:  np.ndarray | None = None  # IMU-pred antenna ENU (3,)
    pred_ecef: np.ndarray | None = None  # IMU-pred antenna ECEF (3,)
    prev_amb_tc: dict = field(default_factory=dict)

    # ── Filled by `optimize` ──────────────────────────────────────
    nv: int = 0                        # # DD factors built
    pose_tc:  Any = None               # smoother Pose3 (kk)
    ecef_tc:  np.ndarray | None = None # antenna ECEF (3,) at kk
    nb: int = 0                        # # accepted DDs after FDE
    xa: np.ndarray | None = None       # LAMBDA-fixed full state vec

    # ── Filled by `postprocess` / `output` ────────────────────────
    sol: np.ndarray | None = None      # output ECEF (3,)
    tag: str | None = None             # 'FIX' / 'FLT'
