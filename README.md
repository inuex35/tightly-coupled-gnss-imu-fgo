# tightly-coupled-gnss-imu-fgo

Tightly-coupled RTK: GNSS double differences and a 100 Hz IMU fused on a
GTSAM factor graph (cssrlib front end), targeting centimeter FIX in deep
urban canyons. Ambiguities are float states in the graph, fixed by LAMBDA
with fix-and-hold; IMU + SD Doppler bridge NLOS storms and tunnels.

## Install

Linux x86_64, Python 3.11/3.12. **Stock PyPI `gtsam`/`cssrlib` will not
work** — both come from forks:

```bash
python3.12 -m venv venv && . venv/bin/activate
pip install numpy matplotlib

# GTSAM wheel with the custom factors (DD, Doppler, SD Doppler, NHC),
# rebuilt weekly against upstream develop:
gh release download custom-wheels-latest -R inuex35/gtsam -p '*cp312*'
pip install gtsam_develop-*.whl

# cssrlib DD-only RTK core (pinned):
pip install -e "git+https://github.com/inuex35/cssrlib-numba.git@55e0c29#egg=cssrlib"
```

No packaging yet — run from the source tree. Datasets are not included
(results use [PPC-Dataset](https://github.com/taroz/PPC-Dataset) tokyo,
laid out as `data/PPC-Dataset/tokyo/run{1,2,3}/`).

## Run

```bash
LEVER_ARM=0.31,0,0.55 \
python examples/run_imu_gnss_tc.py \
  rover.obs base.obs base.nav imu.csv reference.csv
```

### Results — tokyo PPC, full length, all defaults

| run  | length    | AllRMS  | median  | FixRMS  | fix %  | <50 cm |
|------|-----------|---------|---------|---------|--------|--------|
| run1 | 11928 ep  | 21.35 m | 0.23 m  | 0.65 m  | 49.6 % | 55.4 % |
| run2 |  9151 ep  |  6.34 m | 0.053 m | 0.35 m  | 67.6 % | 75.4 % |
| run3 | 15301 ep  | 18.61 m | 0.051 m | 0.47 m  | 67.8 % | 71.5 % |

![tokyo_defaults](docs/tokyo_defaults.png)

## Architecture

Phase 1 (`phase1_rtk.py`): stationary GNSS-only bootstrap. Phase 2:
each epoch runs Stages A–E — IMU preintegration (`preprocess/stage.py`),
quality gate & holds (`preprocess/gate.py`), build→solve→AR→post-fit
(`optimize/`), FIX/FLT verdict (`validation/`) — with cross-cutting
outage/reset paths in `recovery.py`. Factor builders live one-per-family
in `buildfactor/`, the native LAMBDA path in `ar/`.

## Configuration

All knobs are env vars mirroring `config.py` fields (`LEVER_ARM`,
`MAX_EP`, `SAVE_NPZ`, `DOPPLER_SD_SIGMA`, …); see `config.py` for the
commented list.
