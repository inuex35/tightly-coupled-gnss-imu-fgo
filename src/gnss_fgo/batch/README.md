# gnss_fgo.batch — whole-run (batch) tightly-coupled RTK + IMU

The same measurement model as the sequential fixed-lag pipeline, solved as
**one factor graph over the entire drive** (post-processing, non-causal):

```
frontend.prepare     RINEX -> DD rows (+ SD ambiguity arc ids) and rover Doppler rows
gnss_graph           GNSS-only batch (float -> AR -> fix): initial trajectory
tc_graph             X/V/B per epoch: CombinedImuFactor, DD PR/CP Arm, SD Doppler Arm,
                     NHC, ZUPT; post-fit FDE
ar                   per-epoch LAMBDA on selected-inversion covariances,
                     integers merged per SD arc (union-find), no fix-and-hold
pipeline.run_batch   float -> FDE -> AR -> fixed solve -> per-epoch FIX validation
```

## How it works

1. **Float.** All epochs, IMU preintegration, DD pseudorange / carrier, SD
   Doppler, NHC and ZUPT go into one graph; ambiguities are one SD float
   per satellite/frequency *arc* (a continuous carrier track; a new arc
   starts on LLI, a geometry-free jump or a gap). Levenberg-Marquardt runs
   on `gtsam.cuda` (cuDSS) when the gtsam build has CUDA, else on the CPU.
2. **FDE.** DD / Doppler factors whose float residuals exceed thresholds are
   removed in place (`graph.remove`) and the graph is re-solved.
3. **Covariances.** The joint covariance of the arcs in use at every epoch
   comes from one forward/backward sweep of a time-chain selected inversion
   (`selinv.py`, Takahashi / RTS form) — ~1-3 s for a whole run instead of
   one Bayes-tree query per epoch.
4. **Integers.** LAMBDA + ratio test per epoch (RTKLIB `lambda.c`, all
   epochs in one batched C call). Each accepted DD integer is a relation
   `N_tgt - N_ref = n` between arcs; a union-find with offsets merges them,
   strongest ratio first, rejecting contradictions. A relation then holds
   for the arcs' whole life — forward *and* backward in time, and
   transitively across satellites.
5. **Fixed solve.** The relations become `BetweenFactorDouble` constraints;
   the graph is re-solved.
6. **FIX validation.** An epoch is FIX when at least `FIX_MIN_SATS` distinct
   target satellites (L1 and L2 of one satellite count once) and
   `FIX_MIN_PAIRS` DD pairs have a resolved integer, and every resolved
   carrier residual is within `FIX_RES`.

## Usage

```bash
# once: C LAMBDA (optional; cssrlib mlambda is used otherwise)
python -c "from gnss_fgo.batch import lambda_c; lambda_c.build()"

BASE_LLH=35.66633426,139.79220181,59.82 LEVER_ARM=0.31,0,0.55 \
python examples/run_batch_tc.py rover.obs base.obs base.nav imu.csv reference.csv
```

Needs the gtsam build with the custom factors (DD/SD-Doppler Arm, NHC);
CUDA is optional. All tunables are env vars, see `config.py`.

## Accuracy (PPC-Dataset)

All epochs, antenna position vs `reference.csv` (POS/LV at the antenna phase
centre), defaults of `config.py`, base coordinates from the dataset README.
FIX = validated FIX (step 6). Time = whole pipeline after the front end on
an RTX 4060 (cuDSS) + 28-thread CPU.

| run         | epochs | FIX %  | FixRMS  | AllRMS  | median  | <0.5 m | <3 m   | max     | time  |
|-------------|-------:|-------:|--------:|--------:|--------:|-------:|-------:|--------:|------:|
| tokyo run1  | 11928  | 66.2 % | 0.139 m | 0.714 m | 0.059 m | 79.6 % | 99.5 % | 3.04 m  | 29 s  |
| tokyo run2  |  9151  | 70.2 % | 0.058 m | 0.331 m | 0.038 m | 91.0 % | 100 %  | 2.66 m  | 20 s  |
| tokyo run3  | 15301  | 71.2 % | 0.045 m | 0.725 m | 0.032 m | 83.6 % | 96.6 % | 3.11 m  | 60 s  |
| nagoya run1 |  7602  | 75.0 % | 0.108 m | 1.857 m | 0.109 m | 93.5 % | 98.9 % | 24.57 m | 15 s  |
| nagoya run2 |  9451  | 47.0 % | 0.217 m | 3.883 m | 0.253 m | 59.8 % | 88.2 % | 13.56 m | 20 s  |
| nagoya run3 |  5201  | 12.1 % | 0.139 m | 1.167 m | 0.620 m | 42.4 % | 96.7 % | 3.06 m  | 10 s  |

For reference, the sequential pipeline (top-level README): AllRMS 14.10 /
6.28 / 8.83 m (tokyo run1-3), 14.65 / 27.07 / 19.43 m (nagoya run1-3).
**The batch is a smoother**: it uses future measurements and carries
integers backward in time, so it is not comparable to a real-time result,
and its FIX definition (step 6) differs from the sequential one.

## Notes from tuning

- **SD Doppler sigma** (`DOP_SIGMA0`, default 0.1 m/s at zenith). Residuals
  at the reference trajectory are 0.06-0.3 m/s; with 0.5 m/s the
  20-30 s base outages (no DD at all) drifted by up to 35 m.
- **ZUPT needs a speed gate.** The IMU stillness test alone also fires at
  constant speed (46 % of epochs, ~1000 of them above 1 m/s); the gate uses
  a per-epoch Doppler least-squares speed (`ZUPT_SPEED_SRC=doppler`).
- **FIX counts satellites, not pairs.** With pair counting, L1+L2 of three
  satellites passed as FIX at 1 m error.
- **Not adopted** (measured worse or no gain): partial AR in the batch
  (adds relations on doubtful arcs), requiring several epochs to agree on a
  relation, MW wide-lane integers (8-16 % wrong at usable coverage in urban
  code multipath), the Saastamoinen DD tropo correction (`TROPO=1`, no gain
  on the 9 km Nagoya baseline), tighter yaw gyro noise.
- **Nagoya** carrier residuals against the reference are ~3 cm,
  non-dispersive and elevation-independent (tokyo ~1 cm), and nagoya run3
  has few satellites and a poor float (median 0.76 m), hence its low FIX
  rate.
- The batch covariance is ~11-15x optimistic (time-correlated code errors
  are modelled as white); the ratio test is scale-invariant, so AR is not
  affected directly.
- Copy `optimize()` results (`gtsam.Values(opt.optimize())`): the returned
  reference keeps the CUDA optimizer and its GPU buffers alive and made
  later solves ~3x slower.
