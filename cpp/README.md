# C++ port

A C++17 port of the whole pipeline (`src/gnss_fgo` + `examples/run_imu_gnss_tc.py`)
with no Python dependency. The estimator is the same GTSAM factor graph,
module for module; the GNSS front-end follows
[RTKLIB](https://github.com/rtklibexplorer/RTKLIB) (rtklibexplorer, demo5) instead of
cssrlib: RINEX decoding, signal selection, broadcast ephemerides,
satellite orbits and observation editing.

| Python | C++ |
|---|---|
| `factors/` | `factors.cpp` (DD PR/CP, held-integer DDCP, ambiguity seeding, SD Doppler, NHC, ZUPT, IMU chain) |
| `ar/` | `ar.cpp` (problem, LAMBDA resolver, subset retry, gates, fix-and-hold), `mlambda.cpp` |
| `integrity/` | `integrity.cpp` (slip detection, sanity ladder, recovery) |
| `pipeline/` | `pipeline.cpp` (stages A–E, residuals, FDE, smoother glue) |
| `initialization.py` | `initialization.cpp` |
| `runner.py`, `state/` | `runner.{h,cpp}`, `state.h` |
| `config.py` | `config.{h,cpp}` — every field is still an env var of the same name |
| cssrlib (`rnxdec`, `satposs`, `qcedit`, `ddidx`, …) | `rinex.cpp`, `frontend.cpp`, `nav_state.h` on RTKLIB |
| `examples/run_imu_gnss_tc.py` | `apps/run_imu_gnss_tc.cpp` |

## Build

Dependencies: Eigen, Boost and the GTSAM fork with the custom GNSS factors
(`inuex35/gtsam`, branch `custom/develop`). RTKLIB is fetched by CMake
(pinned commit).

```bash
# GTSAM fork (C++ only)
git clone -b custom/develop https://github.com/inuex35/gtsam.git
cmake -S gtsam -B gtsam/build -DCMAKE_BUILD_TYPE=Release -DGTSAM_BUILD_TESTS=OFF \
  -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF -DGTSAM_BUILD_UNSTABLE=OFF -DGTSAM_WITH_TBB=OFF \
  -DGTSAM_USE_SYSTEM_EIGEN=ON -DCMAKE_INSTALL_PREFIX=$HOME/opt/gtsam
cmake --build gtsam/build -j4 --target install

# this port
cmake -S cpp -B cpp/build -DCMAKE_PREFIX_PATH=$HOME/opt/gtsam
cmake --build cpp/build -j4
ctest --test-dir cpp/build
```

`-DRTKLIB_SOURCE_DIR=<checkout>` builds against a local RTKLIB instead of
fetching it.

## Run

Same arguments and environment variables as the Python example:

```bash
LEVER_ARM=0.31,0,0.55 SAVE_CSV=run1.csv \
cpp/build/run_imu_gnss_tc rover.obs base.obs base.nav imu.csv reference.csv
```

`SAVE_CSV` writes one row per epoch (solution, truth, ENU error and every
scalar diagnostic); it replaces the Python `SAVE_NPZ`.

## Front-end (RTKLIB)

- **Signals.** RTKLIB's frequency slots and code priorities: slot f is
  RTKLIB frequency index f — GPS/QZSS L1/L2/L5, Galileo E1/E5b/E5a,
  BeiDou B1I/B2I(B2b)/B2a. GLONASS stays excluded (FDMA biases do not
  cancel in the DD).
- **Satellites.** `satposs` with broadcast ephemerides (RTKLIB's record
  selection) and `satexclude` (health, URA). The one RTKLIB change is a
  build-time table size: BeiDou PRNs up to 63 instead of demo5's 46, so the
  BeiDou-3 GEOs C59–C63 are decoded.
- **Editing.** Elevation mask (`ELMIN_DEG`) on both receivers and an SNR
  mask (`CNR_MIN_DBHZ`, every slot, both receivers), per slot as RTKLIB
  does. On top of that, `STRICT_SAT_GATE=1` (default) admits a satellite
  only while every slot it has produced is usable — the policy the Python
  pipeline (cssrlib's `sat_band_plan`) ran with and the estimator is tuned
  for; `STRICT_SAT_GATE=0` is RTKLIB's per-slot editing alone.
- **Numbering, time, geometry.** RTKLIB satellite numbers (`satno`, system
  order GPS, GLONASS, Galileo, QZSS, BeiDou, NavIC, SBAS) and RTKLIB's
  `geodist`/`satazel`/`ecef2pos`/`dops`/time functions throughout, the
  estimator included (`gnss.{h,cpp}` wrap them). Switching from cssrlib's
  numbering changed no result.

## Results and verification

The estimator was first verified against the Python pipeline with a
front-end that reproduced cssrlib exactly: on tokyo run1 all 11 928 epochs
gave the same FIX/FLT tag and 3D errors within 1 cm of the Python run.
With the RTKLIB front-end the inputs differ (signals, ephemeris records,
admitted satellites), so the numbers move:

| run | Python AllRMS / median / FixRMS / fix % | C++ (RTKLIB, gate on) | C++ (RTKLIB, gate off) |
|---|---|---|---|
| tokyo run1  | 10.27 / 0.102 / 0.48 / 63.3 | 11.07 / 0.117 / 0.51 / 58.8 | 46.50 / 0.328 / 0.68 / 30.3 |
| tokyo run2  |  5.14 / 0.041 / 0.12 / 76.8 |  2.60 / 0.045 / 0.12 / 72.3 | 14.47 / 0.202 / 0.30 / 54.1 |
| tokyo run3  |  6.11 / 0.050 / 0.18 / 73.0 |  5.24 / 0.043 / 0.20 / 73.8 | 13.25 / 0.133 / 0.26 / 60.6 |
| nagoya run1 | 14.62 / 0.135 / 0.14 / 63.1 | 24.13 / 0.132 / 0.15 / 66.5 | 73.38 / 0.265 / 0.46 / 50.9 |
| nagoya run2 | 23.41 / 0.234 / 0.29 / 50.2 | 16.09 / 1.145 / 1.88 / 53.7 | 76.91 / 10.19 / 0.38 / 32.6 |
| nagoya run3 | 19.54 / 0.507 / 0.60 / 46.0 | 15.98 / 1.760 / 2.53 / 42.4 | 47.09 / 18.78 / 1.13 / 22.8 |

(metres / metres / metres / %, all-default settings, `LEVER_ARM=0.31,0,0.55`.)
RTKLIB's per-slot editing keeps many satellites the strict gate drops
(low-SNR or intermittently tracked, i.e. likely NLOS), and the estimator's
tuning does not absorb them. With the gate, tokyo stays close to the
Python runs (run2/run3 better AllRMS, run1 a few points of fix rate
lower); nagoya run2/run3 lose median accuracy and still need a look.

The C++ binary runs about 3x faster than the Python pipeline (tokyo run1:
148 s vs 438 s, both measured with three runs in parallel on 4 cores).
