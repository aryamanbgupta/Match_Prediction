# v7 sim — prop calibration backtest (Phase 1)

Matches: 33 | Sims/match: 100 | Test set: `data/mlc_2025` | Model: `models/xgb_v3/xgboost_model_v3.pkl`

## Binary props

| family | n | base rate | sim Brier [95% CI] | base Brier | sim log loss | skill |
|---|---:|---:|---:|---:|---:|---:|
| top_batter | 726 | 0.091 | 0.0770 [0.0620, 0.0928] | 0.0826 | 0.2796 | +0.069 |
| top_bowler | 726 | 0.091 | 0.0818 [0.0665, 0.0970] | 0.0826 | 0.3125 | +0.010 |
| batter_50plus | 518 | 0.114 | 0.0982 [0.0774, 0.1204] | 0.1009 | 0.3909 | +0.027 |
| batter_6plus_six | 518 | 0.448 | 0.2560 [0.2420, 0.2694] | 0.2473 | 0.7055 | -0.035 |
| innings_runs_ou_160_5 | 66 | 0.621 | 0.2259 [0.1810, 0.2713] | 0.2353 | 0.6479 | +0.040 |
| innings_runs_ou_170_5 | 66 | 0.561 | 0.2556 [0.2070, 0.3064] | 0.2463 | 0.7282 | -0.038 |
| innings_runs_ou_180_5 | 66 | 0.470 | 0.2740 [0.2144, 0.3355] | 0.2491 | 0.9352 | -0.100 |
| batter_fours_1plus | 518 | 0.608 | 0.2357 [0.2165, 0.2545] | 0.2383 | 0.6768 | +0.011 |
| batter_fours_2plus | 518 | 0.384 | 0.2381 [0.2233, 0.2528] | 0.2366 | 0.6680 | -0.007 |
| batter_fours_3plus | 518 | 0.230 | 0.1826 [0.1658, 0.1997] | 0.1770 | 0.5462 | -0.032 |
| bowler_wkts_1plus | 388 | 0.598 | 0.2651 [0.2405, 0.2890] | 0.2404 | 0.7468 | -0.103 |
| bowler_wkts_2plus | 388 | 0.271 | 0.2267 [0.2047, 0.2484] | 0.1974 | 0.6925 | -0.149 |
| bowler_wkts_3plus | 388 | 0.103 | 0.1026 [0.0805, 0.1246] | 0.0925 | 0.4713 | -0.110 |
| team_highest_individual_ou_29_5 | 66 | 0.955 | 0.0516 [0.0153, 0.0989] | 0.0434 | 0.2161 | -0.190 |
| team_highest_individual_ou_34_5 | 66 | 0.864 | 0.1159 [0.0698, 0.1638] | 0.1178 | 0.3831 | +0.016 |
| team_highest_individual_ou_39_5 | 66 | 0.788 | 0.1509 [0.1133, 0.1913] | 0.1671 | 0.4683 | +0.097 |
| pp_total_ou_45_5 | 66 | 0.545 | 0.3623 [0.2719, 0.4489] | 0.2479 | 1.1197 | -0.461 |
| pp_total_ou_50_5 | 66 | 0.485 | 0.3323 [0.2601, 0.4017] | 0.2498 | 0.9483 | -0.330 |
| pp_total_ou_55_5 | 66 | 0.394 | 0.3193 [0.2650, 0.3763] | 0.2388 | 0.8676 | -0.337 |
| match_total_sixes_ou_15_5 | 33 | 0.606 | 0.3761 [0.2774, 0.4734] | 0.2388 | 1.0829 | -0.575 |
| match_total_sixes_ou_20_5 | 33 | 0.394 | 0.2839 [0.1934, 0.3818] | 0.2388 | 0.8430 | -0.189 |
| first_wicket_runs_ou_30_5 | 66 | 0.333 | 0.2475 [0.1912, 0.3061] | 0.2222 | 0.7118 | -0.114 |
| bowler_economy_ou_8_5 | 388 | 0.552 | 0.2666 [0.2498, 0.2839] | 0.2473 | 0.7382 | -0.078 |
| bowler_economy_ou_10_5 | 388 | 0.332 | 0.2414 [0.2241, 0.2608] | 0.2219 | 0.6804 | -0.088 |
| p_tie | 33 | 0.000 | 0.0001 [0.0000, 0.0001] | 0.0000 | 0.0064 | – |
| highest_over_runs_ou_18_5 | 33 | 0.758 | 0.1860 [0.1183, 0.2582] | 0.1837 | 0.5582 | -0.013 |
| highest_over_runs_ou_24_5 | 33 | 0.303 | 0.2674 [0.1447, 0.4030] | 0.2112 | 0.9495 | -0.266 |

Notes:
- Sim Brier < base Brier ⇒ sim has signal beyond the base rate (prop-level edge over a flat predictor).
- Base Brier is `var(y)` -- the score from always predicting the marginal hit rate.
- Skill = `1 − Brier/base_Brier`. Positive ⇒ sim beats base rate.
- Bootstrap CIs: 1000 resamples at the row level (n.b. not paired by match — match-level pairing would tighten CIs further).

### Reliability — top_batter

| bin | n | mean p | actual hit rate |
|---|---:|---:|---:|
| (0.0, 0.1] | 461 | 0.034 | 0.033 |
| (0.1, 0.2] | 174 | 0.148 | 0.178 |
| (0.2, 0.3] | 71 | 0.243 | 0.211 |
| (0.3, 0.4] | 14 | 0.346 | 0.286 |
| (0.4, 0.5] | 6 | 0.443 | 0.167 |

### Reliability — top_bowler

| bin | n | mean p | actual hit rate |
|---|---:|---:|---:|
| (0.0, 0.1] | 471 | 0.027 | 0.055 |
| (0.1, 0.2] | 140 | 0.155 | 0.129 |
| (0.2, 0.3] | 82 | 0.242 | 0.207 |
| (0.3, 0.4] | 29 | 0.339 | 0.172 |
| (0.4, 0.5] | 4 | 0.422 | 0.000 |

### Reliability — batter_50plus

| bin | n | mean p | actual hit rate |
|---|---:|---:|---:|
| (0.0, 0.1] | 338 | 0.036 | 0.089 |
| (0.1, 0.2] | 128 | 0.144 | 0.117 |
| (0.2, 0.3] | 39 | 0.235 | 0.282 |
| (0.3, 0.4] | 10 | 0.353 | 0.200 |
| (0.4, 0.5] | 3 | 0.423 | 0.333 |

### Reliability — batter_6plus_six

| bin | n | mean p | actual hit rate |
|---|---:|---:|---:|
| (0.0, 0.1] | 13 | 0.076 | 0.000 |
| (0.1, 0.2] | 23 | 0.165 | 0.261 |
| (0.2, 0.3] | 64 | 0.257 | 0.422 |
| (0.3, 0.4] | 90 | 0.358 | 0.467 |
| (0.4, 0.5] | 120 | 0.456 | 0.483 |
| (0.5, 0.6] | 99 | 0.556 | 0.475 |
| (0.6, 0.7] | 73 | 0.645 | 0.507 |
| (0.7, 0.8] | 29 | 0.739 | 0.379 |
| (0.8, 0.9] | 7 | 0.826 | 0.571 |

## Continuous props

| family | n | MAE [95% CI] | mean bias (sim − actual) | P10–P90 coverage |
|---|---:|---:|---:|---:|
| batter_runs_mae | 518 | 16.86 [15.60, 18.33] | -1.60 | 74.32% |
| team_total_fours_mae | 66 | 5.08 [4.29, 5.92] | +3.64 | 63.64% |
| team_total_sixes_mae | 66 | 4.54 [3.77, 5.34] | -0.70 | 62.12% |
| team_first_over_mae | 66 | 3.36 [2.80, 4.00] | -0.54 | 74.24% |
| highest_individual_mae | 33 | 23.50 [18.74, 29.27] | -10.56 | 63.64% |
| batter_fours_mae | 518 | 1.59 [1.49, 1.71] | +0.33 | 90.73% |

Note: P10–P90 ideal coverage is 80%. Lower ⇒ sim under-disperses (over-confident); higher ⇒ over-disperses.
