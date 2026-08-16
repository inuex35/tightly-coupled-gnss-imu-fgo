# tightly coupled gnss imu fgo

Tightly-coupled IMU + GNSS RTK on **GTSAM** factor graphs and the
**cssrlib** observation model. Targets urban-canyon RTK conditions.

---

## Install

```bash
python3.12 -m venv venv && . venv/bin/activate
pip install numpy matplotlib

# 1) GTSAM — use the prebuilt custom wheel (recommended).
#    Branch custom/develop = upstream borglab develop + NhcFactor +
#    SingleDifferenceDopplerFactor(/Arm) + the DD factors this pipeline
#    needs. CI re-merges upstream weekly and refreshes the release.
pip install https://github.com/inuex35/gtsam/releases/download/custom-wheels-latest/gtsam_develop-4.3a2.dev202608161310-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
#    (cp311 wheel is on the same release; check the release page for the
#    current filenames: https://github.com/inuex35/gtsam/releases/tag/custom-wheels-latest)

# 2) cssrlib — the inuex35 fork's DD-only RTK core (upstream PyPI cssrlib
#    lacks prepare_double_difference_measurements / the layered frontend).
pip install -e "git+https://github.com/inuex35/cssrlib-numba.git@55e0c29#egg=cssrlib"

# 3) this repo
pip install -e .   # or just export PYTHONPATH=src
```

Building GTSAM from source instead: clone `inuex35/gtsam`, checkout
`custom/develop`, then
`cmake -B build -DGTSAM_BUILD_PYTHON=1 -DPYTHON_EXECUTABLE=$(which python) && make -C build -j4 python-install`
(~20 min, needs ~11 GB RAM). The stock PyPI `gtsam` wheel will NOT work —
it lacks the DD / Doppler / NHC factors.

---

## Run

```bash
LEVER_ARM=0.31,0,0.55 \
SAVE_NPZ=results/tc.npz \
python examples/run_imu_gnss_tc.py \
  rover.obs base.obs base.nav imu.csv reference.csv
```

`imu.csv` is 100 Hz IMU samples (FLU body frame).
`reference.csv` is the truth trajectory used for error reporting only.

### Defaults — tokyo PPC, full-length (NF=3, GPS L1/L2/L5 + Galileo E1/E5a/E5b + QZSS L1/L2/L5 + BDS B1C/B1I/B2a)

| run  | length    | AllRMS  | median  | p90     | FixRMS  | fix %  | <50 cm |
|------|-----------|---------|---------|---------|---------|--------|--------|
| run1 | 11928 ep  | 27.10 m | 0.47 m  | 42.5 m  | 2.316 m | 40.8 % | 50.7 % |
| run2 |  9151 ep  |  5.36 m | 0.056 m |  3.1 m  | 0.267 m | 62.2 % | 74.5 % |
| run3 | 15301 ep  | 19.69 m | 0.050 m |  5.9 m  | 0.463 m | 68.9 % | 75.2 % |

Sequential runs, all defaults (`doppler_sd_sigma=0.5`, `doppler_huber=1.0`,
`doppler_skip_aid=1`, C++ `gtsam.NhcFactor`, cp−pr ambiguity seeding).
run1's residual tail is one deep canyon + a full tunnel blackout; its
FixRMS is dominated by wrong fixes in the NLOS zone (open issue).

![tokyo_defaults](docs/tokyo_defaults.png)

---

## Architecture

Phase 1 (`phase1_rtk.py`) is stationary GNSS-only RTK; Phase 2 is the
moving IMU+DD pipeline. Each Phase-2 epoch flows through five stages:

```
A  preprocess/stage.py   IMU preintegration, prediction, IMU chain
B  preprocess/gate.py    slip/CMC detection, holds, prev-N carry, GDOP gate
C  optimize/stage.py     C1 build → C2 smoother → C3 AR → C4 post-fit diag
D  validation/postprocess.py  FIX/FLT verdict, innovation policy
E  validation/output.py  result tuple + bookkeeping
```

| Where | What |
|---|---|
| `runner.py` | `ImuGnssTc` — owns state, the cssrlib engine boundary (10-method delegation block) |
| `tightly_coupled.py` | Phase-2 epoch entry (`run_tc_epoch`) and Phase-1→2 transition |
| `recovery.py` | cross-cutting outage paths: GDOP skip, IMU-only, warm reset, solve-failure handling |
| `buildfactor/` | one module per factor family: DD PR/CP (`factors.py`), ambiguity seeding (`amb_seed.py`), SD Doppler (`doppler_sd.py`), raw Doppler, NHC, ZUPT, TDCP, PIM |
| `optimize/` | `build.py` (C1), `isam.py` (C2, ISAM2/FLS glue), `ar_stage.py` (C3), `postfit_diag.py` (C4) |
| `validation/` | `residuals.py` (residual tests, FDE, DDPR sanity), `postprocess.py` (D), `output.py` (E) |
| `ar/` | the native ambiguity-resolution core (LAMBDA, retry, subset, hold) + cssrlib nav bridge |
| `epoch_data.py`, `stage_contract.py` | the `EpochData` carrier and the stage reads/writes contract checker (`ENABLE_STAGE_CONTRACT_CHECK=1`) |

---

## Configuration

All knobs are env vars; see `config.py` for the full list. Most-used:

| Knob | Default | Effect |
|---|---|---|
| `LEVER_ARM` | `0,0,0` | IMU → antenna lever in body FLU [m] |
| `MAX_EP` | all | epoch cap |
| `SAVE_NPZ` | none | write per-epoch diagnostics |
| `DOPPLER_SD_SIGMA` | `0.5` | SD Doppler σ [m/s], 0 = off |
| `DOPPLER_HUBER` | `1.0` | robust width [m/s] on SD Doppler |
| `AR_NATIVE_RESOLVER` | `0` | 1 = native `ar/` LAMBDA path |

`TC_PRESET=<name>` loads a knob bundle first; explicit env vars
still override.

---

## Reproducibility

Benchmark runs must be sequential (one pipeline per machine): parallel
runs perturb each other's floating-point summation order through the
threaded solver and flip near-threshold AR decisions. Clear
`__pycache__` before A/B comparisons.
