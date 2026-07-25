# v7 sim — prop calibration backtest (Phase 1)

Matches: 30 | Sims/match: 100 | Test set: `data/polymarket_test` | Model: `models/xgb_i5/xgboost_model_i5.pkl`

## Binary props

| family | n | base rate | sim Brier [95% CI] | base Brier | sim log loss | skill |
|---|---:|---:|---:|---:|---:|---:|
| top_batter | 660 | 0.091 | 0.0760 [0.0608, 0.0914] | 0.0826 | 0.2866 | +0.080 |
| top_bowler | 660 | 0.091 | 0.0751 [0.0608, 0.0909] | 0.0826 | 0.2507 | +0.091 |
| batter_50plus | 486 | 0.109 | 0.0931 [0.0747, 0.1137] | 0.0972 | 0.3306 | +0.042 |
| batter_6plus_six | 486 | 0.444 | 0.2187 [0.2057, 0.2309] | 0.2469 | 0.6199 | +0.114 |
| innings_runs_ou_160_5 | 60 | 0.600 | 0.2756 [0.2101, 0.3503] | 0.2400 | 0.8153 | -0.148 |
| innings_runs_ou_170_5 | 60 | 0.500 | 0.2952 [0.2311, 0.3641] | 0.2500 | 0.8132 | -0.181 |
| innings_runs_ou_180_5 | 60 | 0.417 | 0.2679 [0.2150, 0.3251] | 0.2431 | 0.7437 | -0.102 |
| batter_fours_1plus | 486 | 0.623 | 0.2154 [0.1978, 0.2358] | 0.2348 | 0.6282 | +0.082 |
| batter_fours_2plus | 486 | 0.389 | 0.2176 [0.1997, 0.2353] | 0.2377 | 0.6178 | +0.084 |
| batter_fours_3plus | 486 | 0.249 | 0.1806 [0.1600, 0.1995] | 0.1870 | 0.5339 | +0.034 |
| bowler_wkts_1plus | 353 | 0.626 | 0.2174 [0.2008, 0.2332] | 0.2341 | 0.6263 | +0.071 |
| bowler_wkts_2plus | 353 | 0.292 | 0.2040 [0.1784, 0.2295] | 0.2066 | 0.5979 | +0.013 |
| bowler_wkts_3plus | 353 | 0.096 | 0.0869 [0.0621, 0.1120] | 0.0870 | 0.3429 | +0.001 |
| team_highest_individual_ou_29_5 | 60 | 0.967 | 0.0342 [0.0016, 0.0829] | 0.0322 | 0.1725 | -0.063 |
| team_highest_individual_ou_34_5 | 60 | 0.933 | 0.0672 [0.0210, 0.1297] | 0.0622 | 0.2759 | -0.080 |
| team_highest_individual_ou_39_5 | 60 | 0.900 | 0.1065 [0.0487, 0.1724] | 0.0900 | 0.4093 | -0.184 |
| pp_total_ou_45_5 | 60 | 0.767 | 0.1962 [0.1331, 0.2625] | 0.1789 | 0.5997 | -0.097 |
| pp_total_ou_50_5 | 60 | 0.483 | 0.2685 [0.2189, 0.3188] | 0.2497 | 0.7377 | -0.075 |
| pp_total_ou_55_5 | 60 | 0.383 | 0.2283 [0.1904, 0.2643] | 0.2364 | 0.6450 | +0.034 |
| match_total_sixes_ou_15_5 | 30 | 0.367 | 0.2438 [0.1588, 0.3429] | 0.2322 | 0.7173 | -0.050 |
| match_total_sixes_ou_20_5 | 30 | 0.267 | 0.2000 [0.1169, 0.2920] | 0.1956 | 0.5952 | -0.023 |
| first_wicket_runs_ou_30_5 | 60 | 0.383 | 0.2220 [0.1959, 0.2507] | 0.2364 | 0.6379 | +0.061 |
| bowler_economy_ou_8_5 | 353 | 0.575 | 0.2653 [0.2482, 0.2824] | 0.2444 | 0.7336 | -0.086 |
| bowler_economy_ou_10_5 | 353 | 0.331 | 0.2352 [0.2134, 0.2560] | 0.2216 | 0.7353 | -0.062 |
| p_tie | 30 | 0.000 | 0.0003 [0.0002, 0.0005] | 0.0000 | 0.0135 | – |
| highest_over_runs_ou_18_5 | 30 | 0.600 | 0.2432 [0.1604, 0.3325] | 0.2400 | 0.6871 | -0.013 |
| highest_over_runs_ou_24_5 | 30 | 0.200 | 0.1473 [0.0537, 0.2485] | 0.1600 | 0.4407 | +0.079 |

Notes:
- Sim Brier < base Brier ⇒ sim has signal beyond the base rate (prop-level edge over a flat predictor).
- Base Brier is `var(y)` -- the score from always predicting the marginal hit rate.
- Skill = `1 − Brier/base_Brier`. Positive ⇒ sim beats base rate.
- Bootstrap CIs: 1000 resamples at the row level (n.b. not paired by match — match-level pairing would tighten CIs further).

### Reliability — top_batter

| bin | n | mean p | actual hit rate |
|---|---:|---:|---:|
| (0.0, 0.1] | 412 | 0.025 | 0.039 |
| (0.1, 0.2] | 150 | 0.153 | 0.133 |
| (0.2, 0.3] | 73 | 0.246 | 0.205 |
| (0.3, 0.4] | 23 | 0.347 | 0.348 |
| (0.4, 0.5] | 2 | 0.415 | 0.500 |

### Reliability — top_bowler

| bin | n | mean p | actual hit rate |
|---|---:|---:|---:|
| (0.0, 0.1] | 388 | 0.030 | 0.026 |
| (0.1, 0.2] | 198 | 0.152 | 0.157 |
| (0.2, 0.3] | 67 | 0.237 | 0.239 |
| (0.3, 0.4] | 7 | 0.320 | 0.429 |

### Reliability — batter_50plus

| bin | n | mean p | actual hit rate |
|---|---:|---:|---:|
| (0.0, 0.1] | 249 | 0.028 | 0.044 |
| (0.1, 0.2] | 130 | 0.148 | 0.146 |
| (0.2, 0.3] | 73 | 0.246 | 0.178 |
| (0.3, 0.4] | 27 | 0.347 | 0.333 |
| (0.4, 0.5] | 7 | 0.431 | 0.143 |

### Reliability — batter_6plus_six

| bin | n | mean p | actual hit rate |
|---|---:|---:|---:|
| (0.0, 0.1] | 30 | 0.046 | 0.000 |
| (0.1, 0.2] | 17 | 0.155 | 0.059 |
| (0.2, 0.3] | 36 | 0.250 | 0.306 |
| (0.3, 0.4] | 65 | 0.354 | 0.354 |
| (0.4, 0.5] | 96 | 0.459 | 0.427 |
| (0.5, 0.6] | 109 | 0.557 | 0.523 |
| (0.6, 0.7] | 88 | 0.651 | 0.614 |
| (0.7, 0.8] | 40 | 0.746 | 0.625 |
| (0.8, 0.9] | 5 | 0.835 | 0.800 |

## Continuous props

| family | n | MAE [95% CI] | mean bias (sim − actual) | P10–P90 coverage |
|---|---:|---:|---:|---:|
| batter_runs_mae | 486 | 15.56 [14.16, 16.92] | +0.84 | 76.75% |
| team_total_fours_mae | 60 | 3.87 [3.21, 4.58] | +0.36 | 68.33% |
| team_total_sixes_mae | 60 | 3.33 [2.71, 4.11] | +0.52 | 73.33% |
| team_first_over_mae | 60 | 3.90 [3.10, 4.69] | -1.80 | 55.00% |
| highest_individual_mae | 30 | 17.25 [13.61, 21.33] | -0.83 | 73.33% |
| batter_fours_mae | 486 | 1.51 [1.38, 1.65] | +0.03 | 88.68% |

Note: P10–P90 ideal coverage is 80%. Lower ⇒ sim under-disperses (over-confident); higher ⇒ over-disperses.
