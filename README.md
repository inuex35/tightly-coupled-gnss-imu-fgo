# tightly coupled gnss imu fgo

Tightly-coupled IMU + GNSS RTK on **GTSAM** factor graphs and the
**cssrlib** observation model. Targets urban-canyon RTK conditions.

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

| run  | length    | AllRMS  | FixRMS  | fix %  | <50 cm |
|------|-----------|---------|---------|--------|--------|
| run1 | 11928 ep  | 47.40 m | 0.815 m | 49.5 % | 56.7 % |
| run2 |  9151 ep  | 32.08 m | 0.277 m | 60.8 % | 69.9 % |
| run3 | 15301 ep  | 34.52 m | 0.211 m | 59.4 % | 67.9 % |

![tokyo_defaults](docs/tokyo_defaults.png)

---

## Architecture

Each epoch flows through six layers:

```
dataloader → preprocess → buildfactor → optimize → AR → validation
```

| Layer | Role | Package |
|---|---|---|
| dataloader   | RINEX / IMU CSV / reference loading | `examples/run_imu_gnss_tc.py`, `utils/geometry.py` |
| preprocess   | slip detection, sat selection, ref-sat, hold/release | `preprocess/` |
| buildfactor  | DD pseudorange / carrier-phase, IMU PIM, NHC, Doppler, ZUPT | `buildfactor/` |
| optimize     | ISAM2 / FixedLagSmoother update, LAMBDA AR | `optimize/` |
| validation   | post-fit FDE, cp-hold FSM, sanity recovery | `validation/` |
| utils        | LS solvers, geometry, IMU, robust kernels, RTK shim | `utils/` |

`runner.py` owns the `ImuGnssTc` class and the per-epoch `process()`
entry point. `phase.py` routes each epoch through the
phase-appropriate stages (Phase 1: GNSS-only RTK; Phase 2: IMU + DD).
`epoch.py` defines the `EpochData` carrier and the stage-contract
validator (set `ENABLE_STAGE_CONTRACT_CHECK=1` to verify at startup).

---

## Configuration

All knobs are env vars; see `config.py` for full list. Most-used:

| Knob | Default | Effect |
|---|---|---|
| `LEVER_ARM` | `0,0,0` | IMU → antenna lever in body FLU [m] |
| `MAX_EP` | all | epoch cap |
| `SAVE_NPZ` | none | write per-epoch diagnostics |

`TC_PRESET=<name>` loads a knob bundle first; explicit env vars
still override.

---

## Requirements

- GTSAM (built with the project's DD factors)
- cssrlib — install from <https://github.com/inuex35/cssrlib-numba>
- numpy
