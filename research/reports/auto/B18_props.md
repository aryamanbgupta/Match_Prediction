# xgboost_model_i7 — prop calibration backtest

Matches: 261 | Sims/match: 100 | Test set: `data/polymarket_test` | Model: `models/auto/b18/xgboost_model_i7.pkl`

## Binary props

| family | n | base rate | sim Brier [95% CI] | base Brier | sim log loss | skill |
|---|---:|---:|---:|---:|---:|---:|
| top_batter | 5835 | 0.089 | 0.0751 [0.0700, 0.0805] | 0.0810 | 0.2555 | +0.074 |
| top_bowler | 5835 | 0.088 | 0.0773 [0.0720, 0.0827] | 0.0806 | 0.2912 | +0.041 |
| batter_50plus | 4254 | 0.090 | 0.0779 [0.0709, 0.0842] | 0.0821 | 0.2800 | +0.052 |
| batter_6plus_six | 4254 | 0.398 | 0.2188 [0.2149, 0.2226] | 0.2397 | 0.6248 | +0.087 |
| innings_runs_ou_160_5 | 522 | 0.481 | 0.2227 [0.2103, 0.2351] | 0.2496 | 0.6323 | +0.108 |
| innings_runs_ou_170_5 | 522 | 0.375 | 0.2183 [0.2031, 0.2342] | 0.2345 | 0.6246 | +0.069 |
| innings_runs_ou_180_5 | 522 | 0.278 | 0.1831 [0.1655, 0.2015] | 0.2006 | 0.5454 | +0.087 |
| batter_fours_1plus | 4254 | 0.582 | 0.2180 [0.2135, 0.2230] | 0.2433 | 0.6276 | +0.104 |
| batter_fours_2plus | 4254 | 0.361 | 0.2004 [0.1957, 0.2050] | 0.2306 | 0.5797 | +0.131 |
| batter_fours_3plus | 4254 | 0.225 | 0.1553 [0.1495, 0.1610] | 0.1744 | 0.4741 | +0.109 |
| bowler_wkts_1plus | 3107 | 0.620 | 0.2452 [0.2399, 0.2501] | 0.2357 | 0.6886 | -0.040 |
| bowler_wkts_2plus | 3107 | 0.285 | 0.2101 [0.2012, 0.2194] | 0.2037 | 0.6652 | -0.031 |
| bowler_wkts_3plus | 3107 | 0.103 | 0.0941 [0.0853, 0.1031] | 0.0921 | 0.4289 | -0.021 |
| team_highest_individual_ou_29_5 | 522 | 0.910 | 0.0757 [0.0577, 0.0961] | 0.0819 | 0.2871 | +0.076 |
| team_highest_individual_ou_34_5 | 522 | 0.833 | 0.1280 [0.1087, 0.1493] | 0.1389 | 0.4204 | +0.078 |
| team_highest_individual_ou_39_5 | 522 | 0.759 | 0.1707 [0.1531, 0.1896] | 0.1831 | 0.5232 | +0.068 |
| pp_total_ou_45_5 | 522 | 0.603 | 0.2241 [0.2126, 0.2360] | 0.2393 | 0.6387 | +0.063 |
| pp_total_ou_50_5 | 522 | 0.433 | 0.2249 [0.2150, 0.2348] | 0.2455 | 0.6394 | +0.084 |
| pp_total_ou_55_5 | 522 | 0.295 | 0.1958 [0.1810, 0.2135] | 0.2080 | 0.5780 | +0.059 |
| match_total_sixes_ou_15_5 | 261 | 0.314 | 0.1976 [0.1727, 0.2269] | 0.2155 | 0.5826 | +0.083 |
| match_total_sixes_ou_20_5 | 261 | 0.115 | 0.0996 [0.0698, 0.1317] | 0.1017 | 0.3476 | +0.021 |
| first_wicket_runs_ou_30_5 | 522 | 0.370 | 0.2322 [0.2216, 0.2435] | 0.2330 | 0.6569 | +0.004 |
| bowler_economy_ou_8_5 | 3107 | 0.461 | 0.2467 [0.2415, 0.2517] | 0.2485 | 0.6873 | +0.007 |
| bowler_economy_ou_10_5 | 3107 | 0.252 | 0.1911 [0.1821, 0.2010] | 0.1887 | 0.5765 | -0.013 |
| p_tie | 261 | 0.000 | 0.0003 [0.0003, 0.0004] | 0.0000 | 0.0138 | – |
| highest_over_runs_ou_18_5 | 261 | 0.621 | 0.2282 [0.2127, 0.2448] | 0.2354 | 0.6466 | +0.031 |
| highest_over_runs_ou_24_5 | 261 | 0.107 | 0.0963 [0.0668, 0.1266] | 0.0958 | 0.3482 | -0.005 |

Notes:
- Sim Brier < base Brier ⇒ sim has signal beyond the base rate (prop-level edge over a flat predictor).
- Base Brier is `var(y)` -- the score from always predicting the marginal hit rate.
- Skill = `1 − Brier/base_Brier`. Positive ⇒ sim beats base rate.
- Bootstrap CIs: 1000 resamples at the row level (n.b. not paired by match — match-level pairing would tighten CIs further).

### Reliability — top_batter

| bin | n | mean p | actual hit rate |
|---|---:|---:|---:|
| (0.0, 0.1] | 3480 | 0.029 | 0.030 |
| (0.1, 0.2] | 1686 | 0.154 | 0.161 |
| (0.2, 0.3] | 637 | 0.239 | 0.210 |
| (0.3, 0.4] | 32 | 0.327 | 0.312 |

### Reliability — top_bowler

| bin | n | mean p | actual hit rate |
|---|---:|---:|---:|
| (0.0, 0.1] | 3442 | 0.038 | 0.040 |
| (0.1, 0.2] | 2046 | 0.153 | 0.157 |
| (0.2, 0.3] | 345 | 0.229 | 0.162 |
| (0.3, 0.4] | 2 | 0.320 | 0.000 |

### Reliability — batter_50plus

| bin | n | mean p | actual hit rate |
|---|---:|---:|---:|
| (0.0, 0.1] | 2568 | 0.032 | 0.039 |
| (0.1, 0.2] | 1409 | 0.149 | 0.167 |
| (0.2, 0.3] | 275 | 0.230 | 0.171 |
| (0.3, 0.4] | 2 | 0.315 | 0.500 |

### Reliability — batter_6plus_six

| bin | n | mean p | actual hit rate |
|---|---:|---:|---:|
| (0.0, 0.1] | 113 | 0.060 | 0.071 |
| (0.1, 0.2] | 273 | 0.155 | 0.117 |
| (0.2, 0.3] | 494 | 0.254 | 0.194 |
| (0.3, 0.4] | 906 | 0.356 | 0.357 |
| (0.4, 0.5] | 1327 | 0.454 | 0.460 |
| (0.5, 0.6] | 968 | 0.544 | 0.548 |
| (0.6, 0.7] | 168 | 0.629 | 0.560 |
| (0.7, 0.8] | 5 | 0.709 | 0.200 |

## Continuous props

| family | n | MAE [95% CI] | mean bias (sim − actual) | P10–P90 coverage |
|---|---:|---:|---:|---:|
| batter_runs_mae | 4254 | 13.96 [13.55, 14.35] | +0.20 | 81.90% |
| team_total_fours_mae | 522 | 3.50 [3.28, 3.74] | -0.07 | 76.25% |
| team_total_sixes_mae | 522 | 2.64 [2.46, 2.84] | -0.27 | 77.59% |
| team_first_over_mae | 522 | 3.38 [3.17, 3.63] | -0.14 | 78.54% |
| highest_individual_mae | 261 | 16.16 [14.70, 17.69] | -0.80 | 73.95% |
| batter_fours_mae | 4254 | 1.37 [1.33, 1.41] | -0.03 | 91.63% |

Note: P10–P90 ideal coverage is 80%. Lower ⇒ sim under-disperses (over-confident); higher ⇒ over-disperses.
