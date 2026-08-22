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
- **Integer ambiguity resolution** — LAMBDA with ratio test, exclusion
  retry, partial AR, fix-and-hold, and geometry/residual acceptance
  gates
- **Tight IMU coupling** — 100 Hz `CombinedImuFactor` preintegration;
  NHC and ZUPT vehicle constraints (C++ factors with exact Jacobians)
- **Velocity through outages** — between-satellite single-differenced
  Doppler (clock-free), injected even on GDOP-skipped epochs
- **Integrity & recovery** — three slip/multipath detectors, post-fit
  FDE, and an escalation ladder from CP-hold to warm reset
- **Default-on, not opt-in** — the results below use the defaults
  (DD + carrier + SD Doppler + NHC/ZUPT + LAMBDA AR); nothing is
  hand-tuned per run.
- **Contributed upstream** — the DD, Doppler and undifferenced GNSS
  factors are merged into
  [GTSAM](https://github.com/borglab/gtsam) itself; see the
  [gtsam.org post](https://gtsam.org/2026/06/10/rtk-gnss-double-difference.html)
  for the technique

## Results

![tokyo_defaults](docs/tokyo_defaults.png)

Everything is measured on open data
([PPC-Dataset](https://github.com/taroz/PPC-Dataset), three full
urban-Tokyo drives) with all-default settings, and is reproducible
end to end:

| run  | length    | AllRMS  | median  | FixRMS  | fix %  | <50 cm |
|------|-----------|---------|---------|---------|--------|--------|
| run1 | 11928 ep  | 23.39 m | 0.26 m  | 0.54 m  | 48.9 % | 55.2 % |
| run2 |  9151 ep  |  6.23 m | 0.053 m | 0.38 m  | 67.2 % | 77.3 % |
| run3 | 15301 ep  | 16.85 m | 0.047 m | 0.48 m  | 68.5 % | 75.7 % |

run1 is the hardest route — a deep canyon plus a full tunnel blackout,
bridged by IMU + SD Doppler dead reckoning.

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

# cssrlib DD-only RTK core (pinned):
pip install "cssrlib @ git+https://github.com/inuex35/cssrlib.git@24ec6450e9a9d1fc69b451006a6c4eac5fdf4fd8"
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
| `ar/` | LAMBDA core, exclusion retry, subset search, fix-and-hold |
| `integrity/` | slip detection, sanity ladder, outage recovery |
| `state/` | records (`SatState`, `EpochData`) and the stage I/O contract |

Phase 1 bootstrap (stationary GNSS-only) is `initialization.py`;
`runner.py` owns state and the cssrlib boundary.

## Configuration

Every `config.py` field is an env var (`DOPPLER_SD_SIGMA`,
`SIG_PR`, `ZUPT_MAX_SPEED`, …) — see `config.py` for the complete,
commented list. The example script adds its own env switches
(`LEVER_ARM`, `MAX_EP`, `SAVE_NPZ`, …).
A few recovery-path priors (warm-reset and outage anchors,
Phase-2 seed sigmas) are hardcoded constants, not knobs.

## License

BSD 3-Clause. Built on [GTSAM](https://github.com/borglab/gtsam) and
[cssrlib](https://github.com/hirokawa/cssrlib); evaluated on
[PPC-Dataset](https://github.com/taroz/PPC-Dataset).
