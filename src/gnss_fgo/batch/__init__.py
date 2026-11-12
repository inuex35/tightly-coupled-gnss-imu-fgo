"""Batch (whole-run, non-causal) tightly-coupled GNSS RTK + IMU.

Same measurement model as the sequential smoother, solved as one factor
graph over the entire drive:

  frontend.prepare     RINEX -> DD rows (+ SD arc ids) and rover Doppler rows
  gnss_graph           GNSS-only batch, the initial trajectory
  tc_graph             IMU + DD + SD Doppler + NHC + ZUPT graph, FDE
  ar                   per-epoch LAMBDA on selected-inversion covariances,
                       integers merged per SD arc (no fix-and-hold)
  pipeline.run_batch   float -> FDE -> AR -> fixed solve -> FIX validation

The LM solves use gtsam.cuda (cuDSS) when the gtsam build has CUDA, else
CPU LM. LAMBDA uses lambda_c (RTKLIB lambda.c) once built, else cssrlib.
"""
from .config import BatchConfig
from .pipeline import load_imu_csv, run_batch

__all__ = ["BatchConfig", "run_batch", "load_imu_csv"]
