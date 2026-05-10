# v7 sim — prop calibration backtest (Phase 1)

Matches: 30 | Sims/match: 100 | Test set: `data/polymarket_test` | Model: `models/xgb_v3/xgboost_model_v3.pkl`

## Binary props

| family | n | base rate | sim Brier | base Brier | sim log loss |
|---|---:|---:|---:|---:|---:|
| top_batter | 660 | 0.091 | 0.0792 | 0.0826 | 0.2678 |
| top_bowler | 660 | 0.091 | 0.0823 | 0.0826 | 0.3018 |
| batter_50plus | 486 | 0.109 | 0.0989 | 0.0972 | 0.3610 |
| batter_6plus_six | 486 | 0.444 | 0.2311 | 0.2469 | 0.6516 |

Notes:
- Sim Brier < base Brier ⇒ sim has signal beyond the base rate (prop-level edge over a flat predictor).
- Base Brier is `var(y)` -- the score from always predicting the marginal hit rate.

### Reliability — top_batter

| bin | n | mean p | actual hit rate |
|---|---:|---:|---:|
| (0.0, 0.1] | 412 | 0.032 | 0.044 |
| (0.1, 0.2] | 165 | 0.150 | 0.164 |
| (0.2, 0.3] | 67 | 0.248 | 0.179 |
| (0.3, 0.4] | 14 | 0.342 | 0.214 |
| (0.4, 0.5] | 2 | 0.430 | 0.000 |

### Reliability — top_bowler

| bin | n | mean p | actual hit rate |
|---|---:|---:|---:|
| (0.0, 0.1] | 430 | 0.063 | 0.074 |
| (0.1, 0.2] | 219 | 0.139 | 0.123 |
| (0.2, 0.3] | 11 | 0.230 | 0.091 |

### Reliability — batter_50plus

| bin | n | mean p | actual hit rate |
|---|---:|---:|---:|
| (0.0, 0.1] | 274 | 0.035 | 0.080 |
| (0.1, 0.2] | 145 | 0.144 | 0.152 |
| (0.2, 0.3] | 48 | 0.241 | 0.146 |
| (0.3, 0.4] | 17 | 0.339 | 0.118 |
| (0.4, 0.5] | 2 | 0.425 | 0.000 |

### Reliability — batter_6plus_six

| bin | n | mean p | actual hit rate |
|---|---:|---:|---:|
| (0.0, 0.1] | 9 | 0.054 | 0.000 |
| (0.1, 0.2] | 32 | 0.161 | 0.219 |
| (0.2, 0.3] | 49 | 0.264 | 0.265 |
| (0.3, 0.4] | 96 | 0.356 | 0.417 |
| (0.4, 0.5] | 104 | 0.455 | 0.452 |
| (0.5, 0.6] | 121 | 0.554 | 0.537 |
| (0.6, 0.7] | 60 | 0.646 | 0.600 |
| (0.7, 0.8] | 12 | 0.739 | 0.500 |
| (0.8, 0.9] | 3 | 0.820 | 0.667 |

## Continuous props

| family | n | MAE | mean bias (sim − actual) | P10–P90 coverage |
|---|---:|---:|---:|---:|
| batter_runs_mae | 486 | 16.03 | -0.10 | 76.75% |
| team_total_fours_mae | 60 | 4.73 | +2.88 | 70.00% |
| team_total_sixes_mae | 60 | 3.68 | -0.00 | 71.67% |
| team_first_over_mae | 60 | 3.83 | -1.41 | 61.67% |
| highest_individual_mae | 30 | 16.39 | -2.93 | 80.00% |

Note: P10–P90 ideal coverage is 80%. Lower ⇒ sim under-disperses (over-confident); higher ⇒ over-disperses.
