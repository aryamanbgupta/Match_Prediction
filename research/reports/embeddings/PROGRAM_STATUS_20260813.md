# CricML process-metric and prediction program status

> Role: closure summary of the priority list. The verdict deltas live in
> `CLAIM_STATUS_AUDIT_20260811.md`; the standing prompt in
> `PROGRAM_GOAL_PRIORITY_18.md`; reading order in `README.md`. Numbers here
> restate the underlying experiment reports, which stay authoritative.

2026-08-13 · no golden or forward holdout read.

## Outcome

The original six priorities are closed. The simulator now passes its registered
structural PPC, permanent causal/symmetry contracts are in CI, and the first
future-player xR decision gate has been run. That future gate failed, so the
program correctly stops short of promoting xR/Delta or the auction payload.

| item | status | authoritative result |
|---|---|---|
| 1 claim audit | complete | invalid sequence, exposure-bias, T4, and xR claims downgraded |
| 2 fair T1 ablation | complete, broad gate failed | full T1 minus MLP test LL -0.000387 [-0.001215,+0.000489] |
| 3 simulator parity/PPC | complete, pass | exact features; B18 extras pass; roster bowling pass; 480/480 symmetry |
| 4 calibration guardrail | complete | no post-hoc calibration used; future tests must retain raw results |
| 5 same-cohort xR/xW | complete as measurement | all realized-information stages add information in runs/wicket units |
| 6 reproducibility | complete | configs, seeds, checksums, builders, raw evidence, reports, tests, CI |
| 7 permanent counterfactual tests | complete | first-over, storage-swap, order behavior, causal priors and state parity in CI |
| 8 decision payload | locked, not promoted | internal luck-adjusted auction shortlist; future gate failed |
| 14 learned bowling policy | complete, pass | 6.010/5.949 bowlers vs 6.125/6.000 observed |
| 18 future xR validation | v1 complete, fail | Delta does not beat EB at N=50/100/250; N=500 weak, N=1000 unavailable |

## Clean simulator closure

- Feature parity: 59,291 regulation rows, zero above `1e-6`, maximum absolute
  delta `2.384e-7`.
- Corrected B18 comparison: extras 5.127 versus 4.583 observed; first-innings
  bias -0.288 versus -6.296 flat; MAE 30.153 versus 31.205 flat; all three
  gates pass.
- Roster-aware bowling: unique bowlers 6.010/5.949 versus 6.125/6.000;
  first-innings MAE guard passes; winner LL has no CI-clean regression;
  batting-order behavior is reference-compatible; all five gates pass.
- Remaining monitoring diagnostics, not registered blockers: innings-two
  wickets/duration and death-over chase totals.

## Future player result

The strictly later-match target uses 100 future deliveries. Validation chooses
EB/RAA/Delta shrinkage constants; test uses them frozen.

| N | players | EB MAE | RAA MAE | Delta MAE | Delta−EB [95% CI] |
|---:|---:|---:|---:|---:|---:|
| 50 | 144 | 0.18331 | **0.17798** | 0.20564 | +0.02233 [+0.00350,+0.04193] |
| 100 | 114 | 0.19207 | **0.18761** | 0.20269 | +0.01062 [-0.00816,+0.02978] |
| 250 | 48 | 0.20977 | **0.20909** | 0.22787 | +0.01809 [-0.00369,+0.04091] |
| 500 | 11 | 0.18780 | 0.17865 | **0.17321** | -0.01458 [-0.04748,+0.02074] |
| 1000 | 0 | — | — | — | unavailable |

RAA has favorable point estimates at the useful smaller N values but no clean
advantage over EB. Delta needs extreme validation-selected shrinkage (`k=1600`)
and still underperforms. This is evidence against using the current shot-xR
decomposition as a player-quality or auction signal.

## Work deliberately not started

Items 9-13 and 15-17 remain research hypotheses, not mechanical cleanup. The
failed future gate removes the rationale for optimizing downstream models or
decisions around the present metric. Starting them on the already inspected
test cohort would create post-result selection risk.

The next defensible evidence must come from one of:

1. a later, untouched temporal cohort when available;
2. a broader releasable line/length/shot/contact annotation source; or
3. a newly registered RAA-style metric tested on genuinely fresh data.

Until then, keep the simulator repairs, the honest T1 negative result, and the
internal descriptive xR decomposition; do not promote the auction payload.
