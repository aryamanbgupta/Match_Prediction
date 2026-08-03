# A13 orchestrator plan — 2026-08-03

## Claim rationale (queue state at claim time)

PENDING pool: A10 (P3), A13 (P3), B11 (P3), D5 (P2). D5 is nominally
highest-priority but is NOT claimable this iteration: B10/B12 records mark its
status a supervisor call, and B13 (2026-08-03, the immediately preceding
iteration) skipped it at claim and resolved its mechanism negatively —
"non-bowler top_bowler tail is a per-ball-rate problem, not who-bowls; selector
share alignment now fully explored post B10/B12/B13; recommend retiring D5
SUPERSEDED." Claiming D5 would re-open a lever the loop just closed, against
an open supervisor-call flag. A13 carries the freshest supervisor directive in
the queue (RE-POINT 2026-08-02, post ball-stack promotion) → A13 is the claim.

## Idea

**A13** — sim dispersion calibration on sampled score totals. Premise
(established on the LEGACY balanced-weights stack): the sim under-disperses
score totals (idea text: `team_first_over` P10–P90 coverage 53% vs ideal 80%,
`batter_runs` 72%; B5/B15 quote layer: cp15 coverage 0.664), inflating
over/under tail Brier. Method: fit per-family dispersion scalars on val,
widen sampled-total spread to ~80% nominal coverage.

## Supervisor RE-POINT (2026-08-02) — governs this iteration

Run on the PROMOTED stack (`models/xgb_i7_noweights_production`, i7 stats,
NO calibrator; baseline detail
`models/auto/d16/detail_noweights_raw_s46_n261.json`, seed 46).
**STEP 0 before fitting anything:** re-measure P10–P90 coverage on the
promoted stack — the under-dispersion premise predates the promotion, and
uniform-weight training plausibly changes the spread of sampled totals.
**"If coverage is already ~in band on the promoted stack, log A13 as
SUPERSEDED-by-D16 without burning the second eval."**

## Pre-committed STEP 0 decision rule

- Coverage band = **[0.70, 0.90]** — the only quantitative P10–P90 acceptance
  band the program has ever used (B5 GATE 2, reused by B14/B15). Ideal 80%.
- Read per-family P10–P90 coverage for all six continuous families from the
  promoted-stack baseline report
  (`models/auto/d16/report_noweights_raw_s46_n261.md`, produced by the frozen
  eval framework during the D16 eval), and VERIFY by recomputing each family's
  coverage from the per-row `sim_p10 <= actual <= sim_p90` fields in
  `detail_noweights_raw_s46_n261.json` (same rule as `prop_backtest.py:820`).
- **If every under-dispersed family is inside [0.70, 0.90]** → the premise is
  resolved; verdict **SUPERSEDED-by-D16** per the supervisor rule; no
  implementation, no sim runs, nothing to revert. A family already ABOVE 0.90
  (over-dispersion) cannot justify proceeding: the A13 method only widens
  spread and would push it further out of band.
- **If any under-dispersed family is below 0.70** on the recompute → proceed
  to the full plan: fit per-family dispersion scalars on a val-split sim run
  of the promoted stack (val detail does not exist → one ~545-match run),
  apply post-hoc to the frozen s46 test detail, gate = coverage moves toward
  80% AND pooled tail Brier (pp_total + first_wicket + highest_over) improves
  paired CI-clean AND batter_runs_mae does not regress (launch executor for
  this branch; recipe-B budget applies).

## STEP 0 evidence already on file (orchestrator read, pre-claim orientation)

Promoted stack (d16 noweights raw, s46, n=261) report coverage:
batter_runs **82.51%**, team_total_fours **76.05%**, team_total_sixes
**76.44%**, team_first_over **77.20%**, highest_individual **73.56%**,
batter_fours **91.37%** (over-dispersed). Controls: d16 control twin
(i7 + balanced weights + vector calibrator, same seed) 68.97–77.23 on the
under-dispersion families; legacy canonical b12 70.11–78.72; premise-era a8
vec 64.56–73.77. The no-weights retrain moved every under-dispersed family
toward nominal; none is below 0.72 on the promoted stack.

## Executor

None launched for STEP 0 — the supervisor rule terminates the idea before
implementation/evaluation, and the remaining work is a seconds-scale
verification recompute over committed detail JSONs (orchestrator-permitted:
reading files + report writing; no training/eval/sim/backtest is run).
The executor branch activates only if the recompute contradicts the report
tables (see contingency above).

## Anything easy to get wrong

- Do NOT quote the idea text's 53%/72% as the current baseline — those are
  legacy-stack numbers; the current baseline is the d16 noweights report.
- The b12/a8 rows are context only (legacy path); the verdict binds to the
  PROMOTED stack per the re-point.
- `batter_fours` 91.37% is out of band HIGH — record it, but it argues
  against the A13 widening method, not for it.
