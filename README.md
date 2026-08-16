# tightly-coupled-gnss-imu-fgo

Tightly-coupled RTK: GNSS double differences and a 100 Hz IMU fused on a
**GTSAM** factor graph, with **cssrlib** as the observation front end.
Built for urban canyons — NLOS storms, GDOP collapses, full tunnel
blackouts — where a loosely-coupled filter falls apart.

---

## Install

Python 3.11/3.12 on Linux x86_64.

```bash
python3.12 -m venv venv && . venv/bin/activate
pip install numpy matplotlib

# 1) GTSAM — prebuilt wheel with this project's factors
#    (DD pseudorange/carrier, Doppler, SD Doppler, NHC).
#    The release is rebuilt weekly against upstream gtsam develop;
#    download resolves the current filename:
gh release download custom-wheels-latest -R inuex35/gtsam -p '*cp312*'
pip install gtsam_develop-*.whl

# 2) cssrlib — the inuex35 fork's DD-only RTK core
pip install -e "git+https://github.com/inuex35/cssrlib-numba.git@55e0c29#egg=cssrlib"

# 3) this repo (no packaging yet — run from the source tree)
git clone https://github.com/inuex35/tightly-coupled-gnss-imu-fgo
cd tightly-coupled-gnss-imu-fgo
```

The stock PyPI `gtsam` wheel will **not** work — it lacks every factor
this pipeline is built on. If you prefer building from source: clone
`inuex35/gtsam`, branch `custom/develop`, then
`cmake -B build -DGTSAM_BUILD_PYTHON=1 -DPYTHON_EXECUTABLE=$(which python)`
and `make -C build -j4 python-install` (~20 min, ~11 GB RAM).
Upstream cssrlib from PyPI is equally insufficient — the fork carries
the layered DD front end (`prepare_double_difference_measurements` and
friends) that this pipeline calls.

---

## Run

```bash
LEVER_ARM=0.31,0,0.55 \
python examples/run_imu_gnss_tc.py \
  rover.obs base.obs base.nav imu.csv reference.csv
```

Inputs: rover/base RINEX observations, broadcast nav, 100 Hz IMU CSV
(FLU body frame), and a truth trajectory CSV — used only for the error
columns in the log. Signals are auto-detected from the RINEX headers
(up to 3 bands per system; GLONASS is excluded — FDMA biases do not
cancel in the double difference). Add `SAVE_NPZ=out.npz` for per-epoch
diagnostics.

### Results — tokyo PPC, full length, all defaults

NF=3: GPS L1/L2/L5, Galileo E1/E5a/E5b, QZSS L1/L2/L5, BDS B1C/B1I/B2a.
5 Hz epochs, vehicle-mounted consumer-grade IMU, deep urban Tokyo.

| run  | length    | AllRMS  | median  | p90    | FixRMS  | fix %  | <50 cm |
|------|-----------|---------|---------|--------|---------|--------|--------|
| run1 | 11928 ep  | 27.10 m | 0.47 m  | 42.5 m | 2.32 m  | 40.8 % | 50.7 % |
| run2 |  9151 ep  |  5.36 m | 0.056 m |  3.1 m | 0.27 m  | 62.2 % | 74.5 % |
| run3 | 15301 ep  | 19.69 m | 0.050 m |  5.9 m | 0.46 m  | 68.9 % | 75.2 % |

run1 is the hardest route: its tail is one deep canyon plus a full
tunnel blackout, and its FixRMS is dominated by wrong fixes in the NLOS
zone (open issue). Runs are sequential — see Reproducibility.

![tokyo_defaults](docs/tokyo_defaults.png)

---

## Architecture

Phase 1 (`phase1_rtk.py`) is stationary GNSS-only RTK used to bootstrap
attitude and biases; Phase 2 is the moving IMU+DD pipeline. Every
Phase-2 epoch flows through five stages:

```
A  preprocess/stage.py        IMU preintegration, prediction, IMU chain
B  preprocess/gate.py         slip/CMC detection, holds, prev-N carry, GDOP gate
C  optimize/stage.py          C1 build → C2 smoother → C3 AR → C4 post-fit diag
D  validation/postprocess.py  FIX/FLT verdict, innovation policy
E  validation/output.py       result tuple + bookkeeping
```

| Where | What |
|---|---|
| `runner.py` | `ImuGnssTc`: state owner; the cssrlib engine boundary is one explicit ten-method delegation block |
| `tightly_coupled.py` | Phase-2 epoch entry (`run_tc_epoch`) and the Phase-1→2 transition |
| `recovery.py` | cross-cutting escape paths: GDOP skip, IMU-only epochs, warm reset, solve-failure handling |
| `buildfactor/` | one module per factor family: DD PR/CP, ambiguity seeding (`amb_seed.py`), SD Doppler, raw Doppler, NHC, ZUPT, TDCP, IMU preintegration |
| `optimize/` | `build.py` (C1), `isam.py` (C2: ISAM2/fixed-lag glue), `ar_stage.py` (C3), `postfit_diag.py` (C4) |
| `validation/` | `residuals.py` (residual tests, FDE, DDPR sanity), `postprocess.py` (D), `output.py` (E) |
| `ar/` | native ambiguity resolution: LAMBDA core, demo5-style retry, subset search, fix-and-hold, cssrlib nav bridge |
| `epoch_data.py`, `stage_contract.py` | the `EpochData` carrier and its per-stage reads/writes contract (`ENABLE_STAGE_CONTRACT_CHECK=1`) |

---

## Configuration

Every knob is an env var mirroring a `TcConfig` field — see `config.py`
for the complete, commented list. The ones you will actually touch:

| Knob | Default | Effect |
|---|---|---|
| `LEVER_ARM` | `0,0,0` | IMU → antenna lever arm, body FLU [m] |
| `MAX_EP` | all | stop after N epochs |
| `SAVE_NPZ` | off | write per-epoch diagnostics |
| `DOPPLER_SD_SIGMA` | `0.5` | SD Doppler σ [m/s]; `0` disables |
| `DOPPLER_HUBER` | `1.0` | robust width on SD Doppler [m/s] |
| `AR_NATIVE_RESOLVER` | `0` | `1` = the native `ar/` LAMBDA path instead of cssrlib's |

`TC_PRESET=<name>` applies a named bundle first; explicit env vars
still win.

---

## Reproducibility

One pipeline per machine: concurrent runs perturb each other's
floating-point summation order through the threaded solver, and
near-threshold AR ratio tests flip. Sequential reruns are
line-identical. Clear `__pycache__` before any A/B claim.
