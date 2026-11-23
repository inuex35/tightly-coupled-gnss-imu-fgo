# gnss_fgo.batch — whole-run (batch) tightly-coupled RTK + IMU

Non-causal post-processing smoother solving the entire drive as one GTSAM factor graph (`frontend` → `gnss_graph` → `tc_graph` → `selinv` + `ar` → fixed solve).

## Usage

```bash
BASE_LLH=35.66633426,139.79220181,59.82 LEVER_ARM=0.31,0,0.55 \
python examples/run_batch_tc.py rover.obs base.obs base.nav imu.csv reference.csv
```

Uses the same custom GTSAM wheel (DD, SD Doppler, NHC) and `cssrlib` fork as the sequential pipeline (`CUDA` / cuDSS optional). All tunables are env vars in `config.py`.
