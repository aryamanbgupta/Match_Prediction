# xgboost_model_i7 — prop calibration backtest

Matches: 261 | Sims/match: 100 | Test set: `data/polymarket_test` | Model: `models/xgb_i7_noweights_production/xgboost_model_i7.pkl`

## Binary props

| family | n | base rate | sim Brier [95% CI] | base Brier | sim log loss | skill |
|---|---:|---:|---:|---:|---:|---:|
| top_batter | 5835 | 0.089 | 0.0756 [0.0707, 0.0811] | 0.0810 | 0.2577 | +0.067 |
| top_bowler | 5835 | 0.088 | 0.0777 [0.0723, 0.0831] | 0.0806 | 0.2801 | +0.036 |
| batter_50plus | 4254 | 0.090 | 0.0784 [0.0717, 0.0853] | 0.0821 | 0.2799 | +0.045 |
| batter_6plus_six | 4254 | 0.398 | 0.2182 [0.2145, 0.2221] | 0.2397 | 0.6228 | +0.089 |
| innings_runs_ou_160_5 | 522 | 0.481 | 0.2378 [0.2231, 0.2531] | 0.2496 | 0.6698 | +0.047 |
| innings_runs_ou_170_5 | 522 | 0.375 | 0.2312 [0.2112, 0.2523] | 0.2345 | 0.6700 | +0.014 |
| innings_runs_ou_180_5 | 522 | 0.278 | 0.1993 [0.1759, 0.2223] | 0.2006 | 0.6261 | +0.007 |
| batter_fours_1plus | 4254 | 0.582 | 0.2190 [0.2147, 0.2237] | 0.2433 | 0.6298 | +0.100 |
| batter_fours_2plus | 4254 | 0.361 | 0.2009 [0.1958, 0.2055] | 0.2306 | 0.5832 | +0.129 |
| batter_fours_3plus | 4254 | 0.225 | 0.1554 [0.1492, 0.1618] | 0.1744 | 0.4728 | +0.109 |
| bowler_wkts_1plus | 3107 | 0.620 | 0.2431 [0.2377, 0.2483] | 0.2357 | 0.6839 | -0.031 |
| bowler_wkts_2plus | 3107 | 0.285 | 0.2094 [0.2011, 0.2183] | 0.2037 | 0.6535 | -0.028 |
| bowler_wkts_3plus | 3107 | 0.103 | 0.0944 [0.0852, 0.1031] | 0.0921 | 0.4289 | -0.025 |
| team_highest_individual_ou_29_5 | 522 | 0.910 | 0.0757 [0.0576, 0.0959] | 0.0819 | 0.2853 | +0.075 |
| team_highest_individual_ou_34_5 | 522 | 0.833 | 0.1308 [0.1121, 0.1522] | 0.1389 | 0.4325 | +0.058 |
| team_highest_individual_ou_39_5 | 522 | 0.759 | 0.1756 [0.1578, 0.1940] | 0.1831 | 0.5375 | +0.041 |
| pp_total_ou_45_5 | 522 | 0.603 | 0.2374 [0.2265, 0.2483] | 0.2393 | 0.6679 | +0.008 |
| pp_total_ou_50_5 | 522 | 0.433 | 0.2351 [0.2198, 0.2502] | 0.2455 | 0.6622 | +0.042 |
| pp_total_ou_55_5 | 522 | 0.295 | 0.2016 [0.1814, 0.2252] | 0.2080 | 0.5969 | +0.031 |
| match_total_sixes_ou_15_5 | 261 | 0.314 | 0.1984 [0.1725, 0.2286] | 0.2155 | 0.5826 | +0.079 |
| match_total_sixes_ou_20_5 | 261 | 0.115 | 0.1006 [0.0694, 0.1341] | 0.1017 | 0.3500 | +0.011 |
| first_wicket_runs_ou_30_5 | 522 | 0.370 | 0.2392 [0.2205, 0.2574] | 0.2330 | 0.6741 | -0.026 |
| bowler_economy_ou_8_5 | 3107 | 0.461 | 0.2492 [0.2437, 0.2548] | 0.2485 | 0.7004 | -0.003 |
| bowler_economy_ou_10_5 | 3107 | 0.252 | 0.1925 [0.1819, 0.2020] | 0.1887 | 0.5899 | -0.020 |
| p_tie | 261 | 0.000 | 0.0003 [0.0002, 0.0003] | 0.0000 | 0.0125 | – |
| highest_over_runs_ou_18_5 | 261 | 0.621 | 0.2433 [0.2287, 0.2586] | 0.2354 | 0.6812 | -0.034 |
| highest_over_runs_ou_24_5 | 261 | 0.107 | 0.0969 [0.0675, 0.1274] | 0.0958 | 0.3504 | -0.012 |

Notes:
- Sim Brier < base Brier ⇒ sim has signal beyond the base rate (prop-level edge over a flat predictor).
- Base Brier is `var(y)` -- the score from always predicting the marginal hit rate.
- Skill = `1 − Brier/base_Brier`. Positive ⇒ sim beats base rate.
- Bootstrap CIs: 1000 resamples at the row level (n.b. not paired by match — match-level pairing would tighten CIs further).

### Reliability — top_batter

| bin | n | mean p | actual hit rate |
|---|---:|---:|---:|
| (0.0, 0.1] | 3531 | 0.031 | 0.033 |
| (0.1, 0.2] | 1672 | 0.155 | 0.170 |
| (0.2, 0.3] | 614 | 0.237 | 0.192 |
| (0.3, 0.4] | 18 | 0.325 | 0.111 |

### Reliability — top_bowler

| bin | n | mean p | actual hit rate |
|---|---:|---:|---:|
| (0.0, 0.1] | 3384 | 0.034 | 0.043 |
| (0.1, 0.2] | 2033 | 0.154 | 0.151 |
| (0.2, 0.3] | 415 | 0.228 | 0.154 |
| (0.3, 0.4] | 3 | 0.310 | 0.333 |

### Reliability — batter_50plus

| bin | n | mean p | actual hit rate |
|---|---:|---:|---:|
| (0.0, 0.1] | 2645 | 0.034 | 0.052 |
| (0.1, 0.2] | 1422 | 0.145 | 0.148 |
| (0.2, 0.3] | 186 | 0.227 | 0.199 |
| (0.3, 0.4] | 1 | 0.310 | 0.000 |

### Reliability — batter_6plus_six

| bin | n | mean p | actual hit rate |
|---|---:|---:|---:|
| (0.0, 0.1] | 113 | 0.058 | 0.035 |
| (0.1, 0.2] | 267 | 0.152 | 0.112 |
| (0.2, 0.3] | 479 | 0.257 | 0.217 |
| (0.3, 0.4] | 1022 | 0.359 | 0.353 |
| (0.4, 0.5] | 1490 | 0.452 | 0.463 |
| (0.5, 0.6] | 803 | 0.543 | 0.574 |
| (0.6, 0.7] | 80 | 0.629 | 0.562 |

## Continuous props

| family | n | MAE [95% CI] | mean bias (sim − actual) | P10–P90 coverage |
|---|---:|---:|---:|---:|
| batter_runs_mae | 4254 | 13.86 [13.45, 14.27] | -0.29 | 82.39% |
| team_total_fours_mae | 522 | 3.49 [3.26, 3.73] | -0.41 | 76.05% |
| team_total_sixes_mae | 522 | 2.65 [2.45, 2.85] | -0.40 | 76.44% |
| team_first_over_mae | 522 | 3.39 [3.17, 3.65] | -0.51 | 76.82% |
| highest_individual_mae | 261 | 16.30 [14.78, 17.88] | -2.77 | 72.80% |
| batter_fours_mae | 4254 | 1.36 [1.32, 1.40] | -0.08 | 91.26% |

Note: P10–P90 ideal coverage is 80%. Lower ⇒ sim under-disperses (over-confident); higher ⇒ over-disperses.
