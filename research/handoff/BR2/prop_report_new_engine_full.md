# xgboost_model_i7 — prop calibration backtest

Matches: 250 | Sims/match: 100 | Test set: `data/polymarket_test_v2` | Model: `models/xgb_i7_noweights_production/xgboost_model_i7.pkl`

**Voided (not settled): 5** — D/L-shortened or no-result matches settle truncated actuals against a full-20-over sim where a real book voids or re-lines.
- `1493270` (outcome.method=D/L)
- `1493279` (outcome.method=D/L)
- `1494267` (outcome.method=D/L)
- `1507719` (outcome.method=D/L)
- `1507720` (outcome.method=D/L)

## Binary props

| family | n | base rate | sim Brier [95% CI] | base Brier | sim log loss | skill |
|---|---:|---:|---:|---:|---:|---:|
| top_batter | 5593 | 0.089 | 0.0758 [0.0704, 0.0812] | 0.0814 | 0.2602 | +0.069 |
| top_bowler | 5593 | 0.089 | 0.0780 [0.0727, 0.0835] | 0.0814 | 0.2769 | +0.042 |
| batter_50plus | 4107 | 0.092 | 0.0797 [0.0730, 0.0869] | 0.0836 | 0.2913 | +0.046 |
| batter_6plus_six | 4107 | 0.399 | 0.2196 [0.2157, 0.2236] | 0.2397 | 0.6262 | +0.084 |
| innings_runs_ou_160_5 | 500 | 0.490 | 0.2310 [0.2153, 0.2457] | 0.2499 | 0.6502 | +0.076 |
| innings_runs_ou_170_5 | 500 | 0.382 | 0.2298 [0.2090, 0.2511] | 0.2361 | 0.6576 | +0.027 |
| innings_runs_ou_180_5 | 500 | 0.280 | 0.1936 [0.1685, 0.2186] | 0.2016 | 0.5853 | +0.039 |
| batter_fours_1plus | 4107 | 0.584 | 0.2173 [0.2126, 0.2220] | 0.2430 | 0.6256 | +0.106 |
| batter_fours_2plus | 4107 | 0.360 | 0.2014 [0.1964, 0.2062] | 0.2305 | 0.5813 | +0.126 |
| batter_fours_3plus | 4107 | 0.225 | 0.1560 [0.1494, 0.1625] | 0.1744 | 0.4749 | +0.105 |
| bowler_wkts_1plus | 3005 | 0.616 | 0.2448 [0.2398, 0.2501] | 0.2366 | 0.6840 | -0.035 |
| bowler_wkts_2plus | 3005 | 0.282 | 0.2081 [0.1999, 0.2165] | 0.2026 | 0.6333 | -0.027 |
| bowler_wkts_3plus | 3005 | 0.101 | 0.0929 [0.0834, 0.1021] | 0.0909 | 0.4073 | -0.022 |
| team_highest_individual_ou_29_5 | 500 | 0.918 | 0.0682 [0.0487, 0.0874] | 0.0753 | 0.2633 | +0.095 |
| team_highest_individual_ou_34_5 | 500 | 0.844 | 0.1202 [0.1000, 0.1423] | 0.1317 | 0.4041 | +0.087 |
| team_highest_individual_ou_39_5 | 500 | 0.772 | 0.1636 [0.1461, 0.1827] | 0.1760 | 0.5068 | +0.071 |
| pp_total_ou_45_5 | 500 | 0.616 | 0.2358 [0.2260, 0.2468] | 0.2365 | 0.6629 | +0.003 |
| pp_total_ou_50_5 | 500 | 0.440 | 0.2345 [0.2203, 0.2491] | 0.2464 | 0.6589 | +0.048 |
| pp_total_ou_55_5 | 500 | 0.304 | 0.2036 [0.1832, 0.2256] | 0.2116 | 0.5971 | +0.038 |
| match_total_sixes_ou_15_5 | 250 | 0.316 | 0.1924 [0.1663, 0.2195] | 0.2161 | 0.5650 | +0.110 |
| match_total_sixes_ou_20_5 | 250 | 0.120 | 0.1041 [0.0762, 0.1394] | 0.1056 | 0.4035 | +0.014 |
| first_wicket_runs_ou_30_5 | 500 | 0.362 | 0.2358 [0.2185, 0.2544] | 0.2310 | 0.6662 | -0.021 |
| bowler_economy_ou_8_5 | 3005 | 0.461 | 0.2503 [0.2445, 0.2563] | 0.2485 | 0.6944 | -0.007 |
| bowler_economy_ou_10_5 | 3005 | 0.254 | 0.1935 [0.1842, 0.2030] | 0.1893 | 0.5827 | -0.023 |
| p_tie | 250 | 0.012 | 0.0119 [0.0003, 0.0275] | 0.0119 | 0.0641 | -0.007 |
| highest_over_runs_ou_18_5 | 250 | 0.628 | 0.2407 [0.2248, 0.2556] | 0.2336 | 0.6746 | -0.030 |
| highest_over_runs_ou_24_5 | 250 | 0.112 | 0.1022 [0.0701, 0.1400] | 0.0995 | 0.4163 | -0.027 |

Notes:
- Sim Brier < base Brier ⇒ sim has signal beyond the base rate (prop-level edge over a flat predictor).
- Base Brier is `var(y)` -- the score from always predicting the marginal hit rate.
- Skill = `1 − Brier/base_Brier`. Positive ⇒ sim beats base rate.
- Bootstrap CIs: 1000 resamples at the row level (n.b. not paired by match — match-level pairing would tighten CIs further).

### Reliability — top_batter

| bin | n | mean p | actual hit rate |
|---|---:|---:|---:|
| (0.0, 0.1] | 3335 | 0.030 | 0.032 |
| (0.1, 0.2] | 1681 | 0.156 | 0.166 |
| (0.2, 0.3] | 556 | 0.236 | 0.200 |
| (0.3, 0.4] | 21 | 0.322 | 0.143 |

### Reliability — top_bowler

| bin | n | mean p | actual hit rate |
|---|---:|---:|---:|
| (0.0, 0.1] | 3207 | 0.033 | 0.043 |
| (0.1, 0.2] | 2016 | 0.153 | 0.150 |
| (0.2, 0.3] | 369 | 0.229 | 0.163 |
| (0.3, 0.4] | 1 | 0.320 | 0.000 |

### Reliability — batter_50plus

| bin | n | mean p | actual hit rate |
|---|---:|---:|---:|
| (0.0, 0.1] | 2548 | 0.034 | 0.052 |
| (0.1, 0.2] | 1362 | 0.145 | 0.154 |
| (0.2, 0.3] | 196 | 0.227 | 0.184 |
| (0.3, 0.4] | 1 | 0.340 | 0.000 |

### Reliability — batter_6plus_six

| bin | n | mean p | actual hit rate |
|---|---:|---:|---:|
| (0.0, 0.1] | 100 | 0.066 | 0.040 |
| (0.1, 0.2] | 277 | 0.159 | 0.083 |
| (0.2, 0.3] | 438 | 0.260 | 0.237 |
| (0.3, 0.4] | 923 | 0.359 | 0.352 |
| (0.4, 0.5] | 1498 | 0.453 | 0.465 |
| (0.5, 0.6] | 800 | 0.542 | 0.551 |
| (0.6, 0.7] | 71 | 0.625 | 0.606 |

## Continuous props

| family | n | MAE [95% CI] | mean bias (sim − actual) | P10–P90 coverage |
|---|---:|---:|---:|---:|
| batter_runs_mae | 4107 | 14.01 [13.60, 14.40] | -0.25 | 82.30% |
| team_total_fours_mae | 500 | 3.38 [3.15, 3.60] | -0.49 | 77.80% |
| team_total_sixes_mae | 500 | 2.62 [2.43, 2.83] | -0.33 | 78.00% |
| team_first_over_mae | 500 | 3.30 [3.05, 3.56] | -0.77 | 79.80% |
| highest_individual_mae | 250 | 15.99 [14.54, 17.65] | -3.29 | 76.00% |
| batter_fours_mae | 4107 | 1.37 [1.32, 1.41] | -0.08 | 91.40% |

Note: P10–P90 ideal coverage is 80%. Lower ⇒ sim under-disperses (over-confident); higher ⇒ over-disperses.
