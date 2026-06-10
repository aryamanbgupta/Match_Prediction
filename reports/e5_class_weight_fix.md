# E5 — Root cause of the sim's tail-event overshoot + fix ✅ LANDED (vector-scaling ball calibrator)

**Date**: 2026-06-09 · **Branch**: `improvement-experiments`
**Scripts**: `scripts/e5_teacher_forced_bias.py` (diagnostic),
`scripts/calibration.py::{PriorCorrectionCalibrator,VectorScalingCalibrator}` (fix),
`scripts/sim_eval/prop_backtest.py --ball-calibrator vector` (wiring)
**Artifact**: `models/xgb_v3/vector_scaling_calibrator_v1.pkl` (6 params, val-fit)

## What started as "add intent features" became a root-cause hunt

Original hypothesis: the sim over-states PP totals / bowler wickets because the
ball model lacks intent conditioning (RRR-vs-par, settling-in ramp). **Falsified
immediately**: v7's feature list already contains the chase/pressure/momentum
groups. The teacher-forced audit then found the real mechanism.

## Root cause (mechanism, with numbers)

`xgboost_v2.py` trains the ball model with sklearn **`balanced` class weights**
(line ~235). The booster therefore approximates the *weighted* posterior — rare
classes (wicket 5.4%, six 4.5%, two 7.6%) get probability inflated by ~1/freq.
The sim's `XGBoostModelV2` **samples these tilted probabilities raw**
(`ball_calibrator=None` everywhere since v4).

Teacher-forced on 186,667 real test deliveries (real features, real outcomes):

| | pred raw | actual | note |
|---|---:|---:|---|
| P(wicket) per ball | 0.121 | 0.057 | 2.1× over |
| P(four) | +0.052 abs | — | over |
| P(six) | +0.041 abs | — | over |
| P(dot) / P(one) | −0.058 / −0.128 abs | — | under |
| runs/ball | 1.667 | 1.284 | +0.38 |

In rollouts the boundary and wicket inflations partially cancel for innings
totals (innings end early but score fast), which is why winner-market evals
(v4→v7) never caught it — while tail-count props (`bowler_wkts_kplus`,
`pp_total_ou`, `batter_fours_3plus`) amplify it. This is exactly the E2 /
prop-framework bias list.

Secondary finding: the XGB sim wrapper never sets `venue_encoded` (missing
keys default to 0), so every simulated ball scores as venue code 0. Second-
order relative to the tilt (real-vs-0 teacher-forced deltas are small) but an
out-of-distribution input on every ball — filed in TODO with the
`innings_id` hash bug.

## Fix

Theoretical prior correction (divide by class weights) **over-corrects** —
the early-stopped booster doesn't reach the full tilt (corrected runs/ball
1.04 vs 1.28). Landed instead: **`VectorScalingCalibrator`** — 6
multiplicative parameters fit on the **validation split only** by iterative
marginal matching, fit under the sim's input distribution (venue_encoded=0).

Teacher-forced on held-out test: runs/ball Δ **+0.383 → +0.024**, P(wicket)
Δ **+0.065 → −0.002**, per-phase deltas ≈ 0, test multiclass LL
**1.608 → 1.520**.

## Sim-level A/B (n=60 × 100 sims, empirical selector, seed 42)

Fair-baseline audit (`prop_fair_baselines.py`), calibrated vs the raw sim's
n=261 verdicts:

| family | raw Δ(sim−base) n261 | calibrated Δ n60 | movement |
|---|---:|---:|---|
| `pp_total_ou_45_5/50_5/55_5` | +0.011/+0.026/+0.031 | **−0.032/−0.011/−0.006** | overshoot GONE |
| `bowler_wkts_1/2/3plus` | +0.032/+0.036/+0.014 | +0.023/+0.014/+0.001 | halved (residual = attribution, the G2 wicket-model problem) |
| `top_bowler` | −0.001 (parity) | **−0.0037 [−0.0064, −0.0007]** | **first binary family to beat a fair baseline** |
| `innings_runs_ou_*` | −0.012…−0.008 (parity) | −0.040…−0.025 (parity, favorable) | improved |
| `match_total_sixes_*` | −0.007/+0.001 | −0.040/−0.031 | improved |
| `batter_runs_mae` | −0.71 (sim better) | +0.31 (parity); paired vs raw +0.89 [+0.56, +1.20] | **the one regression** |

Paired A/B (calibrated vs raw, same matches): all binary families tied;
`batter_runs_mae` significantly worse — the raw sim's boundary-inflation and
wicket-inflation biases happened to cancel for per-batter run totals, and
removing both breaks that accidental cancellation. Per-batter expected runs
should be read from the raw sim (or the E2 career baseline) until the wicket
*attribution* model (G2) improves.

**Full n=261 calibrated rerun launched for confirmation** —
`reports/prop_calibration_{detail,report}_vec_n261.*` when complete.

## Verdict

- **Landed for all prop/score/scenario uses**: `--ball-calibrator vector`.
  It removes the documented marginal-rate biases at the source instead of
  betting against them ("inverse plays" are dead — E2 showed the fair
  baseline beats both the raw sim and its inverse).
- Winner-market use is unaffected (the direct model owns that market; the
  sim envelope is consumed at w=0 in the blend pipeline).
- Known cost: per-batter runs MAE. Known residual: bowler wicket attribution
  (G2) — now cleanly isolated as the next ball-level target, no longer
  confounded with the marginal tilt.

## Why this went undetected for 14 months

The class weighting predates v4. Every eval that aggregated to match level
(winner LL, ROI) blended the offsetting biases; per-ball accuracy metrics
were never compared against per-ball *base rates*; and the prop framework
measured against base-rate Brier, which the tilt still beats on ranking
families. The teacher-forced marginal audit (10 lines of numpy) is the
check that catches it — added to the E-series toolkit as
`e5_teacher_forced_bias.py`, rerunnable after any ball-model retrain.
