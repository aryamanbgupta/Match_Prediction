# v7 sim — prop calibration backtest (Phase 1)

Matches: 60 | Sims/match: 100 | Test set: `data/polymarket_test` | Model: `models/xgb_v3/xgboost_model_v3.pkl`

## Binary props

| family | n | base rate | sim Brier | base Brier | sim log loss |
|---|---:|---:|---:|---:|---:|
| top_batter | 1321 | 0.090 | 0.0783 | 0.0820 | 0.2807 |
| top_bowler | 1321 | 0.089 | 0.0802 | 0.0813 | 0.2937 |
| batter_50plus | 985 | 0.088 | 0.0795 | 0.0805 | 0.2918 |
| batter_6plus_six | 985 | 0.414 | 0.2289 | 0.2426 | 0.6484 |

Notes:
- Sim Brier < base Brier ⇒ sim has signal beyond the base rate (prop-level edge over a flat predictor).
- Base Brier is `var(y)` -- the score from always predicting the marginal hit rate.

### Reliability — top_batter

| bin | n | mean p | actual hit rate |
|---|---:|---:|---:|
| (0.0, 0.1] | 842 | 0.033 | 0.052 |
| (0.1, 0.2] | 302 | 0.148 | 0.116 |
| (0.2, 0.3] | 143 | 0.248 | 0.210 |
| (0.3, 0.4] | 29 | 0.345 | 0.345 |
| (0.4, 0.5] | 5 | 0.436 | 0.000 |

### Reliability — top_bowler

| bin | n | mean p | actual hit rate |
|---|---:|---:|---:|
| (0.0, 0.1] | 870 | 0.063 | 0.070 |
| (0.1, 0.2] | 427 | 0.140 | 0.124 |
| (0.2, 0.3] | 24 | 0.229 | 0.167 |

### Reliability — batter_50plus

| bin | n | mean p | actual hit rate |
|---|---:|---:|---:|
| (0.0, 0.1] | 605 | 0.034 | 0.055 |
| (0.1, 0.2] | 252 | 0.146 | 0.135 |
| (0.2, 0.3] | 96 | 0.239 | 0.135 |
| (0.3, 0.4] | 28 | 0.343 | 0.250 |
| (0.4, 0.5] | 4 | 0.438 | 0.000 |

### Reliability — batter_6plus_six

| bin | n | mean p | actual hit rate |
|---|---:|---:|---:|
| (0.0, 0.1] | 17 | 0.061 | 0.059 |
| (0.1, 0.2] | 68 | 0.162 | 0.206 |
| (0.2, 0.3] | 136 | 0.265 | 0.346 |
| (0.3, 0.4] | 235 | 0.352 | 0.353 |
| (0.4, 0.5] | 213 | 0.453 | 0.390 |
| (0.5, 0.6] | 197 | 0.551 | 0.558 |
| (0.6, 0.7] | 92 | 0.649 | 0.576 |
| (0.7, 0.8] | 24 | 0.738 | 0.625 |
| (0.8, 0.9] | 3 | 0.820 | 0.667 |

## Continuous props

| family | n | MAE | mean bias (sim − actual) | P10–P90 coverage |
|---|---:|---:|---:|---:|
| batter_runs_mae | 985 | 15.10 | +0.53 | 77.16% |
| team_total_fours_mae | 120 | 4.87 | +3.03 | 66.67% |
| team_total_sixes_mae | 120 | 3.15 | +0.17 | 75.83% |
| team_first_over_mae | 120 | 3.49 | -0.97 | 69.17% |
| highest_individual_mae | 60 | 19.83 | -1.56 | 70.00% |

Note: P10–P90 ideal coverage is 80%. Lower ⇒ sim under-disperses (over-confident); higher ⇒ over-disperses.
