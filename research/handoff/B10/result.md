# B10 result — sim who-bowls usage alignment (debutants + non-bowlers)

**NOTE:** the executor session died between producing
`raw/gate_output.txt` (2026-07-31 00:05 local) and writing this file. This
result.md was transcribed by the ORCHESTRATOR directly from the raw
artifacts in `research/handoff/B10/raw/` (gate_output.txt, run_blind.log,
run_b10.log, unit_check_pre_armb.txt). All numbers below are copied
verbatim from those files.

## Commits

- claim `91be8d7`, plan `82c00de`, implement `ad144ea` (22:27:01 -0400,
  BEFORE the first eval started 22:28:41 — gate script pre-committed).
- `git show --stat ad144ea`: `scripts/auto/b10_build_usage_sidecar.py`
  (+97), `scripts/auto/b10_gate_analysis.py` (+359),
  `scripts/auto/b10_unit_check.py` (+394), `scripts/sim_v1_2.py`
  (+216/−8) — 4 files, 1058 insertions, 8 deletions vs claim.

## Eval runs (both to completion, sequential, never concurrent)

- Arm A (blind twin): 2026-07-30 22:28:41 → 23:16:18 (**2857 s**),
  261/261, detail `models/auto/b10/detail_blind_s43_n261.json`. Startup:
  venue encoder ACTIVE (467 venues) / Bowler selector: empirical /
  Run-out dismissal channel ACTIVE (p_runout=0.0751,
  nonstriker_share=0.4685) / vector calibrator; NO B10 line.
- Arm B (b10): 2026-07-30 23:16:46 → 2026-07-31 00:04:07 (**2841 s**),
  261/261, detail `models/auto/b10/detail_b10_s43_n261.json`. Startup:
  same four lines PLUS `B10 usage-aligned bowler selector ACTIVE
  (k_u=5.0)`.
- `b10_unit_check.py` re-run immediately before Arm B
  (`raw/unit_check_pre_armb.txt`): **B10 UNIT CHECK PASSED** (d15 30/30,
  legacy parity float-exact, exp_balls parity vs B9 AsOfUsage exact,
  production `models/bowler_phase_usage.json` md5 unchanged
  ea0c73d3ddb48f499b6273f9a397b0e3).

## GATE 1 (primary) — NOT MET

- `top_bowler` paired dBrier (b10 − blind), n=5831, drop 4:
  blind 0.0772 → b10 0.0770, delta **−0.0002 [−0.0008, +0.0005]** →
  straddles 0, NOT CI-clean. Positional cross-check identical.
- G5 coverage: blind 3095/3108 = 0.9958; b10 **3105/3108 = 0.9990** →
  ≥0.90 MET. (The AND fails on the Brier arm.)

## GATE 2 (guards, no CI-clean regression) — MET

- `bowler_wkts_1plus` n=3107: 0.2584 → 0.2535, **−0.0049
  [−0.0075, −0.0023]** DOWN(better) — CI-clean IMPROVEMENT.
- `bowler_wkts_2plus`: −0.0010 [−0.0026, +0.0005] ~noise (positional
  cross-check −0.0026 [−0.0046, −0.0004] better).
- `batter_runs_mae`: +0.0145 [−0.0400, +0.0701] ~noise.
- `team_first_over_mae`: +0.0120 [−0.0008, +0.0242] ~noise.

Pre-committed verdict MAPPING printed by the gate script: **TABLED**
(GATE 1 not met, GATE 2 met — orchestrator decides).

## Context (verbatim from gate_output.txt)

- Full family scan CI-excludes-0 (either direction), out of 32 families:
  `bowler_wkts_1plus` −0.0049 [−0.0075, −0.0023] better;
  `first_wicket_runs_ou_30_5` −0.0024 [−0.0046, −0.0000] better;
  `highest_over_runs_ou_24_5` −0.0014 [−0.0030, −0.0000] better.
  ZERO families CI-clean worse.
- B9 sim−usage top_bowler margin recomputation: blind +0.0028
  [+0.0017, +0.0040] (head-only +0.0035); b10 +0.0026 [+0.0014, +0.0038]
  (head-only +0.0031). B9's headline on the stale D15 detail was +0.0038
  (head-only +0.0049). The gap to the usage baseline barely closes.
- Relaxation triggers in the b10 run log: **8** (plan expected ~0);
  exactly 1 B10-ACTIVE startup line.
- Drift check (descriptive), blind arm vs pre-I5-refactor D15 detail,
  SAME seed 43: `top_bowler` **−0.0010 [−0.0017, −0.0003]** CI-clean
  (the intended-inert I5/I9 refactors did move the legacy path's draws);
  bowler_wkts_1plus +0.0015, 2plus −0.0007, batter_runs_mae −0.0050,
  team_first_over_mae −0.0037 — all noise. The twin-fresh-run decision
  (plan adaptation 1) was vindicated; the D15 detail is no longer a valid
  paired baseline.

## Unit-check mechanism finding (raw/unit_check_pre_armb.txt, verbatim)

Weight-mechanism table over the 261 test lineups: true debutants
1.153% → 8.733% full-XI share (as planned, ≈9%); known bowlers
12.018% → 11.385% (renormalization only); unknown normalized share ==
as-of expected-balls share (max |delta| 0.000e+00). BUT veteran
never-bowlers (≥20 appearances, 0 career balls): **0.270% → 0.496%
(ratio 1.83) — the WRONG direction.** B9's exp_balls formula shrinks a
0-ball veteran to k_u·prior/(k_u+n) ≈ 1–2 balls at k_u=5, above the
legacy α share; the B9 usage baseline itself prices that group 1.30% vs
sim 1.29% (never fixed defect (b) either). B10 as specified attacks
defect (a) and MOVES (b) THE WRONG WAY — consistent with top_bowler not
improving while the debutant-driven wicket families do.

## Crashes / anomalies

- No crashes in either eval; both ~47.5 min (within budget).
- The executor session itself was cut after the gate run, before
  result.md — recovered by the orchestrator (this file).
