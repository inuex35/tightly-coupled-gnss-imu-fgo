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

![tokyo_defaults](docs/tokyo_defaults.png)

Everything is measured on open data with all-default settings, and is
reproducible end to end — three full urban-Tokyo drives and three
urban-Nagoya drives
([PPC-Dataset](https://github.com/taroz/PPC-Dataset)):

| run         | length    | AllRMS  | median  | FixRMS  | fix %  | <50 cm |
|-------------|-----------|---------|---------|---------|--------|--------|
| tokyo run1  | 11928 ep  | 14.10 m | 0.087 m | 0.29 m  | 62.5 % | 71.1 % |
| tokyo run2  |  9151 ep  |  7.12 m | 0.041 m | 0.32 m  | 75.0 % | 79.8 % |
| tokyo run3  | 15301 ep  |  8.00 m | 0.049 m | 0.21 m  | 72.7 % | 76.7 % |
| nagoya run1 |  7602 ep  | 14.95 m | 0.141 m | 0.13 m  | 56.9 % | 68.5 % |
| nagoya run2 |  9451 ep  | 27.07 m | 0.243 m | 0.36 m  | 49.7 % | 54.0 % |
| nagoya run3 |  5201 ep  | 19.15 m | 0.799 m | 0.88 m  | 43.1 % | 46.7 % |

tokyo run1 is the hardest route — a deep canyon plus a full tunnel
blackout, bridged by IMU + SD Doppler dead reckoning.

![nagoya_defaults](docs/nagoya_defaults.png)

The defaults admit each satellite on the bands it actually transmits
(`SAT_BAND_PLAN=1` — a pre-IIF GPS without L5 or a B1I-only BDS-2 is
judged on what it broadcasts instead of being discarded wholesale),
run LAMBDA's ratio test against the demo5/FFRT dimension-adaptive
threshold (`AR_THRESAR_MIN=1.5`, `AR_THRESAR_MAX=3.0`) instead of a
fixed 2.0, and recover declined fixes with the ranked subset retry
alone — the demo5 single-satellite exclusion retry was measured
per-mechanism across every dataset and deleted (its recovered fixes
degraded FixRMS everywhere it fired). Each choice was adopted from
per-dataset A/B measurement on this exact revision pair. Revert with
`SAT_BAND_PLAN=0 AR_THRESAR_MIN=0 AR_THRESAR_MAX=0
SUBSET_AR_ENABLE=0`.

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
The measured defaults and their reverts: `SAT_BAND_PLAN=0` (strict
all-band admission), `AR_THRESAR_MIN=0 AR_THRESAR_MAX=0` (fixed AR
ratio threshold), `P1_FDE_ENABLE=0` (no Phase-1 FDE screen).
A few recovery-path priors (warm-reset and outage anchors,
Phase-2 seed sigmas) are hardcoded constants, not knobs.

## License

BSD 3-Clause. Built on [GTSAM](https://github.com/borglab/gtsam) and
[cssrlib](https://github.com/hirokawa/cssrlib); evaluated on
[PPC-Dataset](https://github.com/taroz/PPC-Dataset).
