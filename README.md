# tightly-coupled-gnss-imu-fgo

**Open-source tightly-coupled GNSS RTK + IMU factor-graph optimization**

*carrier-phase RTK · LAMBDA ambiguity resolution · GTSAM · urban driving · reproducible*

The full RTK chain lives inside the graph: double-differenced
pseudorange **and carrier phase** with Huber M-estimator noise,
continuous integer ambiguity resolution (LAMBDA + partial AR), and
100 Hz IMU preintegration — targeting centimeter FIX in deep urban
canyons, with IMU + single-differenced Doppler bridging NLOS storms
and tunnels.

## What's inside

- **RTK in the graph** — DD pseudorange + DD carrier-phase factors with
  Huber robust noise on a QR-factorized GTSAM
  `IncrementalFixedLagSmoother`, float ambiguities chained via per-epoch
  random walk, clock-free seeding (SD phase − SD code)
- **Integer ambiguity resolution** — continuous LAMBDA + partial AR
  with a dimension-adaptive fixed-failure-rate ratio test (FFRT) and
  full-AR DD integer fix-and-hold (`armode=2`)
- **Tight IMU coupling** — 100 Hz `CombinedImuFactor` preintegration;
  NHC vehicle constraint (C++ factor with exact Jacobians)
- **Velocity through outages** — between-satellite single-differenced
  Doppler (clock-free), injected on every epoch
- **Integrity & recovery** — LLI + geometry-free cycle-slip detection
  and ambiguity reset/re-seeding
- **GTSAM integration** — DD and Doppler factors are upstream in
  [GTSAM](https://github.com/borglab/gtsam) (see the
  [gtsam.org post](https://gtsam.org/2026/06/10/rtk-gnss-double-difference.html));
  NHC and SD Doppler are contributed / pending upstream

## Results

Everything is measured on open data with all-default settings, and is
reproducible end to end — three full urban-Tokyo drives and three
urban-Nagoya drives
([PPC-Dataset](https://github.com/taroz/PPC-Dataset)):

| run         | length    | AllRMS  | median  | FixRMS  | fix %  | <50 cm |
|-------------|-----------|---------|---------|---------|--------|--------|
| tokyo run1  | 11951 ep  |  0.91 m | 0.040 m | 0.11 m  | 79.1 % | 86.1 % |
| tokyo run2  |  9151 ep  |  1.37 m | 0.032 m | 0.06 m  | 84.4 % | 89.6 % |
| tokyo run3  | 15301 ep  |  1.58 m | 0.030 m | 0.17 m  | 88.0 % | 92.3 % |
| nagoya run1 |  7651 ep  |  2.63 m | 0.126 m | 0.12 m  | 80.9 % | 84.7 % |
| nagoya run2 |  9451 ep  |  2.87 m | 0.231 m | 0.37 m  | 63.8 % | 57.4 % |
| nagoya run3 |  5201 ep  |  0.98 m | 0.130 m | 0.13 m  | 64.7 % | 82.5 % |


![tokyo_defaults](docs/tokyo_defaults.png)

![nagoya_defaults](docs/nagoya_defaults.png)

## Quick start

Linux x86_64, Python 3.12 (CI-tested; 3.11 should work). **Stock PyPI `gtsam`/`cssrlib` will
not work** — both come from forks:

```bash
python3.12 -m venv venv && . venv/bin/activate
pip install numpy matplotlib

# GTSAM wheel with the custom factors (DD, Doppler, SD Doppler, NHC),
# rebuilt weekly against upstream develop:
gh release download custom-wheels-latest -R inuex35/gtsam -p '*cp312*manylinux*' -D wheels
pip install "$(ls wheels/*.whl | sort | tail -1)"   # newest, in case the rolling release carries a stale one

# cssrlib DD-only RTK core (pinned; carries nav.sat_band_plan,
# satposs signal-flight-time fix, indexed findeph bisect, and sync_obs_hold dt_fill):
pip install "cssrlib @ git+https://github.com/inuex35/cssrlib.git@6f7a56caf3e4eb5e8cd6159e8456a785110d8d09"
```

Run (datasets not included; lay out PPC-Dataset under
`data/PPC-Dataset/tokyo/run{1,2,3}/`):

```bash
LEVER_ARM=0.31,0,0.55 \
python examples/run_imu_gnss_tc.py \
  rover.obs base.obs base.nav imu.csv reference.csv
```

## Architecture

Flat module layout under `src/gnss_fgo/`:

| Module | Role |
|---|---|
| `runner.py` | `ImuGnssTc.process(obs, obsb)` entry point and epoch flow (`bootstrap` / IMU prediction → slip check → factors → iSAM2 → `run_ar`) |
| `factors.py` | measurement → GTSAM factor builders (DD PR/CP, SD Doppler, NHC, ambiguity seeding), `SatStateMap`, and cycle-slip detection |
| `ar.py` | LAMBDA + partial AR with dimension-adaptive FFRT and full-AR fix-and-hold |
| `config.py` | `TcConfig` and `BatchConfig` |
| `utils.py` | geometry, IMU params/preintegration, signal auto-detection, Huber noise |
| `batch.py`, `batch_ar.py`, `batch_graph.py` | offline whole-run batch TC pipeline and global AR |

## Configuration

Every `TcConfig` field in `config.py` is an env var (`ERR_A`, `ERR_ERATIO_PR`,
`ELMIN_DEG`, `AMB_RW_SIGMA`, `ARMODE`, `FLS_LAG`, …) — see
`config.py` for all 11 parameters. The example script adds its own
env switches (`LEVER_ARM`, `MAX_EP`, `SAVE_NPZ`, …).

## License

BSD 3-Clause. Built on [GTSAM](https://github.com/borglab/gtsam) and
[cssrlib](https://github.com/hirokawa/cssrlib); evaluated on
[PPC-Dataset](https://github.com/taroz/PPC-Dataset).
