# tightly-coupled-gnss-imu-fgo

**Open-source tightly-coupled GNSS RTK + IMU factor-graph optimization**

*carrier-phase RTK · LAMBDA ambiguity resolution · GTSAM · urban driving · reproducible*

The full RTK chain lives inside the graph: double-differenced
pseudorange **and carrier phase**, integer ambiguity resolution
(LAMBDA, exclusion retry, partial AR) with fix-and-hold, FDE, and
100 Hz IMU preintegration — targeting centimeter FIX in deep urban
canyons, with IMU + single-differenced Doppler bridging NLOS storms
and tunnels.

## What's inside

- **RTK in the graph** — DD pseudorange + DD carrier-phase factors on a
  GTSAM `IncrementalFixedLagSmoother`, ambiguities as float states,
  clock-free seeding (SD phase − SD code)
- **Integer ambiguity resolution** — LAMBDA with a dimension-adaptive
  ratio test, ranked subset retry, fix-and-hold, and
  geometry/residual acceptance gates
- **Tight IMU coupling** — 100 Hz `CombinedImuFactor` preintegration;
  NHC and ZUPT vehicle constraints (C++ factors with exact Jacobians)
- **Velocity through outages** — between-satellite single-differenced
  Doppler (clock-free), injected on every epoch
- **Integrity & recovery** — LLI + geometry-free slip detection,
  post-fit FDE, and an escalation ladder from CP-hold to warm reset
- **Contributed upstream** — the DD, Doppler and undifferenced GNSS
  factors are merged into
  [GTSAM](https://github.com/borglab/gtsam) itself; see the
  [gtsam.org post](https://gtsam.org/2026/06/10/rtk-gnss-double-difference.html)
  for the technique

## Results

Everything is measured on open data with all-default settings, and is
reproducible end to end — three full urban-Tokyo drives and three
urban-Nagoya drives
([PPC-Dataset](https://github.com/taroz/PPC-Dataset)):

| run         | length    | AllRMS  | median  | FixRMS  | fix %  | <50 cm |
|-------------|-----------|---------|---------|---------|--------|--------|
| tokyo run1  | 11928 ep  | 14.10 m | 0.099 m | 0.29 m  | 63.3 % | 70.8 % |
| tokyo run2  |  9151 ep  |  6.28 m | 0.041 m | 0.99 m  | 77.2 % | 81.0 % |
| tokyo run3  | 15301 ep  |  8.83 m | 0.050 m | 0.20 m  | 74.6 % | 78.7 % |
| nagoya run1 |  7602 ep  | 14.65 m | 0.138 m | 0.15 m  | 57.9 % | 72.5 % |
| nagoya run2 |  9451 ep  | 27.07 m | 0.245 m | 0.36 m  | 49.7 % | 54.0 % |
| nagoya run3 |  5201 ep  | 19.43 m | 0.685 m | 1.81 m  | 46.2 % | 48.1 % |


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

# cssrlib DD-only RTK core (pinned; this revision carries the
# nav.sat_band_plan admission policy the defaults use and the satposs
# signal-flight-time fix the results below depend on):
pip install "cssrlib @ git+https://github.com/inuex35/cssrlib.git@5b3711a73f6d8eb3a4b5429d7cf31783cb41927d"
```

Run (datasets not included; lay out PPC-Dataset under
`data/PPC-Dataset/tokyo/run{1,2,3}/`):

```bash
LEVER_ARM=0.31,0,0.55 \
python examples/run_imu_gnss_tc.py \
  rover.obs base.obs base.nav imu.csv reference.csv
```

## Architecture

Packages are the roles:

| Package | Role |
|---|---|
| `pipeline/` | the epoch flow: `imu_prediction` → `quality_gate` → `solve` (`measurement_factors` / `update_smoother` / `fix_ambiguities` / `check_postfit`) → `validate_fix` → `report` |
| `factors/` | measurement → GTSAM factor builders, one file per family |
| `ar/` | LAMBDA core, ranked subset retry, fix-and-hold |
| `integrity/` | slip detection, sanity ladder, outage recovery |
| `state/` | records (`SatState`, `EpochData`) and the stage I/O contract |

Phase 1 bootstrap (stationary GNSS-only) is `initialization.py`;
`runner.py` owns state and the cssrlib boundary.

## Configuration

Every `config.py` field is an env var (`DOPPLER_SD_SIGMA`,
`SIG_PR`, `ZUPT_MAX_SPEED`, …) — see `config.py` for the complete,
commented list. The example script adds its own env switches
(`LEVER_ARM`, `MAX_EP`, `SAVE_NPZ`, …).

## License

BSD 3-Clause. Built on [GTSAM](https://github.com/borglab/gtsam) and
[cssrlib](https://github.com/hirokawa/cssrlib); evaluated on
[PPC-Dataset](https://github.com/taroz/PPC-Dataset).
