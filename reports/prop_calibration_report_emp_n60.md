# v7 sim — prop calibration backtest (Phase 1)

Matches: 60 | Sims/match: 100 | Test set: `data/polymarket_test` | Model: `models/xgb_v3/xgboost_model_v3.pkl`

## Binary props

| family | n | base rate | sim Brier | base Brier | sim log loss |
|---|---:|---:|---:|---:|---:|
| top_batter | 1321 | 0.090 | 0.0777 | 0.0820 | 0.2723 |
| top_bowler | 1321 | 0.089 | 0.0788 | 0.0813 | 0.2951 |
| batter_50plus | 985 | 0.088 | 0.0786 | 0.0805 | 0.3251 |
| batter_6plus_six | 985 | 0.414 | 0.2314 | 0.2426 | 0.6551 |

Notes:
- Sim Brier < base Brier ⇒ sim has signal beyond the base rate (prop-level edge over a flat predictor).
- Base Brier is `var(y)` -- the score from always predicting the marginal hit rate.

### Reliability — top_batter

| bin | n | mean p | actual hit rate |
|---|---:|---:|---:|
| (0.0, 0.1] | 862 | 0.035 | 0.051 |
| (0.1, 0.2] | 290 | 0.150 | 0.138 |
| (0.2, 0.3] | 129 | 0.251 | 0.171 |
| (0.3, 0.4] | 34 | 0.346 | 0.324 |
| (0.4, 0.5] | 6 | 0.437 | 0.333 |

### Reliability — top_bowler

| bin | n | mean p | actual hit rate |
|---|---:|---:|---:|
| (0.0, 0.1] | 824 | 0.023 | 0.038 |
| (0.1, 0.2] | 299 | 0.156 | 0.177 |
| (0.2, 0.3] | 138 | 0.243 | 0.174 |
| (0.3, 0.4] | 54 | 0.343 | 0.167 |
| (0.4, 0.5] | 6 | 0.450 | 0.167 |

### Reliability — batter_50plus

| bin | n | mean p | actual hit rate |
|---|---:|---:|---:|
| (0.0, 0.1] | 708 | 0.031 | 0.062 |
| (0.1, 0.2] | 201 | 0.139 | 0.149 |
| (0.2, 0.3] | 64 | 0.237 | 0.141 |
| (0.3, 0.4] | 10 | 0.346 | 0.400 |
| (0.4, 0.5] | 2 | 0.422 | 0.000 |

### Reliability — batter_6plus_six

| bin | n | mean p | actual hit rate |
|---|---:|---:|---:|
| (0.0, 0.1] | 23 | 0.066 | 0.130 |
| (0.1, 0.2] | 84 | 0.161 | 0.214 |
| (0.2, 0.3] | 181 | 0.256 | 0.337 |
| (0.3, 0.4] | 201 | 0.354 | 0.353 |
| (0.4, 0.5] | 230 | 0.450 | 0.452 |
| (0.5, 0.6] | 144 | 0.551 | 0.590 |
| (0.6, 0.7] | 92 | 0.643 | 0.511 |
| (0.7, 0.8] | 29 | 0.744 | 0.621 |
| (0.8, 0.9] | 1 | 0.860 | 1.000 |

## Continuous props

| family | n | MAE | mean bias (sim − actual) | P10–P90 coverage |
|---|---:|---:|---:|---:|
| batter_runs_mae | 985 | 14.53 | -1.72 | 77.56% |
| team_total_fours_mae | 120 | 4.23 | +1.52 | 71.67% |
| team_total_sixes_mae | 120 | 3.04 | -0.20 | 78.33% |
| team_first_over_mae | 120 | 3.48 | -1.00 | 69.17% |
| highest_individual_mae | 60 | 20.13 | -8.55 | 75.00% |

Note: P10–P90 ideal coverage is 80%. Lower ⇒ sim under-disperses (over-confident); higher ⇒ over-disperses.
