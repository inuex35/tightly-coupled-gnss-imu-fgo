# Refactor TODO

State as of 2026-08-19 (post #12/#14): package graph is a DAG, pyflakes
clean, 12 unit tests (`tests/`, generic Jacobian checker in
`tests/factor_check.py`), results 21.35 / 6.34 / 18.61 m AllRMS.

Verification discipline for every step below: sequential full runs only
(one pipeline per machine — TBB), clear `__pycache__`, line-diff against
the previous full-run logs on all three tokyo runs before merging.

## tightly (priority order)

1. **Explicit stage I/O** — the big one. Stages communicate by mutating
   `tc` (~50 mutable fields); make each stage's reads/writes appear in
   its signature and centralize `tc` writes. `stage_contract.py` lists
   today's implicit contract — use it as the checklist. Days of work.
2. **Make the implicit FSMs classes** — the CP-hold machinery
   (`_recov_cp_hold` / release streaks / forced holds) and the sanity
   ladder (`sanity.py`, moved but still a function chain). The state
   transitions should be readable from one place.
3. **Move the 21 direct `os.environ` reads into TcConfig** — e.g.
   `SKIP_BIAS_PRIOR_SIGMA`, `BOOT_DDPR_SIGMA` (recovery/build),
   phase1_rtk (7 sites), utils/geometry (5). They bypass `TC_PRESET`
   and hurt reproducibility. Half a day, behavior-neutral with same
   defaults.
4. Small: drop the `REF_SWITCH_RESET_ALWAYS` env hook in
   `buildfactor/factors.py`; clarify the `initialization.py` /
   `tightly_coupled.py` boundary (both touch the Phase-1→2 transition).

## cssrlib fork (`gtsam-gnss-frontend`, surveyed not started)

5. **Split `estimation/residuals.py` (839 lines)** — `zdres` (~280
   lines) should get the plan/core separation `sdres` already has
   (`_sdres_build_plan` / `_sdres_core`).
6. `estimation/ekf.py` (439) and the `domain/structs.py` Nav facade
   (455) — review after 5.
7. Document the `nav.slip` producer/consumer contract (qcedit sets,
   `update_ambiguities` consumes+clears). fgo does not use either path
   today (verified), so priority is low.

## Known non-goals / already decided

- cssrlib-dialect method names on the engine boundary (`zdres`,
  `valpos`, `IB`, `_last_s0/_last_s1`) stay — renaming breaks the
  correspondence with upstream; the delegation block in `runner.py`
  documents them.
- `ar_native_resolver` stays default-off until a full-length run
  matches (the tail diverges; equivalence proven only to ep3000).
