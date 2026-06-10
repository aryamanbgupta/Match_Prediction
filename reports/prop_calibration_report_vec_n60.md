# v7 sim — prop calibration backtest (Phase 1)

Matches: 60 | Sims/match: 100 | Test set: `data/polymarket_test` | Model: `models/xgb_v3/xgboost_model_v3.pkl`

## Binary props

| family | n | base rate | sim Brier [95% CI] | base Brier | sim log loss | skill |
|---|---:|---:|---:|---:|---:|---:|
| top_batter | 1321 | 0.090 | 0.0787 [0.0687, 0.0897] | 0.0820 | 0.3075 | +0.040 |
| top_bowler | 1321 | 0.089 | 0.0773 [0.0660, 0.0887] | 0.0813 | 0.2756 | +0.049 |
| batter_50plus | 985 | 0.088 | 0.0824 [0.0696, 0.0942] | 0.0805 | 0.3138 | -0.024 |
| batter_6plus_six | 985 | 0.414 | 0.2303 [0.2205, 0.2411] | 0.2426 | 0.6482 | +0.051 |
| innings_runs_ou_160_5 | 120 | 0.550 | 0.2356 [0.1949, 0.2788] | 0.2475 | 0.6849 | +0.048 |
| innings_runs_ou_170_5 | 120 | 0.425 | 0.2527 [0.2135, 0.2928] | 0.2444 | 0.7214 | -0.034 |
| innings_runs_ou_180_5 | 120 | 0.317 | 0.2188 [0.1756, 0.2691] | 0.2164 | 0.6560 | -0.011 |
| batter_fours_1plus | 985 | 0.589 | 0.2283 [0.2141, 0.2422] | 0.2421 | 0.7126 | +0.057 |
| batter_fours_2plus | 985 | 0.375 | 0.2148 [0.2025, 0.2266] | 0.2343 | 0.6227 | +0.083 |
| batter_fours_3plus | 985 | 0.217 | 0.1677 [0.1547, 0.1806] | 0.1701 | 0.5450 | +0.014 |
| bowler_wkts_1plus | 716 | 0.616 | 0.2521 [0.2377, 0.2672] | 0.2366 | 0.7673 | -0.065 |
| bowler_wkts_2plus | 716 | 0.296 | 0.2172 [0.1968, 0.2381] | 0.2084 | 0.7781 | -0.042 |
| bowler_wkts_3plus | 716 | 0.101 | 0.0904 [0.0720, 0.1108] | 0.0904 | 0.4246 | +0.001 |
| team_highest_individual_ou_29_5 | 120 | 0.942 | 0.0553 [0.0175, 0.0974] | 0.0549 | 0.2311 | -0.007 |
| team_highest_individual_ou_34_5 | 120 | 0.842 | 0.1341 [0.0821, 0.1909] | 0.1333 | 0.4653 | -0.006 |
| team_highest_individual_ou_39_5 | 120 | 0.783 | 0.1730 [0.1211, 0.2315] | 0.1697 | 0.5565 | -0.020 |
| pp_total_ou_45_5 | 120 | 0.692 | 0.2350 [0.2027, 0.2643] | 0.2133 | 0.6656 | -0.102 |
| pp_total_ou_50_5 | 120 | 0.450 | 0.2626 [0.2248, 0.2975] | 0.2475 | 0.7304 | -0.061 |
| pp_total_ou_55_5 | 120 | 0.317 | 0.2248 [0.1760, 0.2721] | 0.2164 | 0.6705 | -0.039 |
| match_total_sixes_ou_15_5 | 60 | 0.333 | 0.2124 [0.1596, 0.2677] | 0.2222 | 0.6284 | +0.044 |
| match_total_sixes_ou_20_5 | 60 | 0.183 | 0.1334 [0.0718, 0.2088] | 0.1497 | 0.4565 | +0.109 |
| first_wicket_runs_ou_30_5 | 120 | 0.400 | 0.2444 [0.2179, 0.2720] | 0.2400 | 0.6852 | -0.018 |
| bowler_economy_ou_8_5 | 716 | 0.503 | 0.2512 [0.2412, 0.2615] | 0.2500 | 0.7131 | -0.005 |
| bowler_economy_ou_10_5 | 716 | 0.278 | 0.2082 [0.1900, 0.2277] | 0.2007 | 0.6197 | -0.037 |
| p_tie | 60 | 0.000 | 0.0004 [0.0003, 0.0007] | 0.0000 | 0.0154 | – |
| highest_over_runs_ou_18_5 | 60 | 0.567 | 0.2471 [0.2125, 0.2867] | 0.2456 | 0.6910 | -0.006 |
| highest_over_runs_ou_24_5 | 60 | 0.150 | 0.1412 [0.0640, 0.2339] | 0.1275 | 0.7167 | -0.108 |

Notes:
- Sim Brier < base Brier ⇒ sim has signal beyond the base rate (prop-level edge over a flat predictor).
- Base Brier is `var(y)` -- the score from always predicting the marginal hit rate.
- Skill = `1 − Brier/base_Brier`. Positive ⇒ sim beats base rate.
- Bootstrap CIs: 1000 resamples at the row level (n.b. not paired by match — match-level pairing would tighten CIs further).

### Reliability — top_batter

| bin | n | mean p | actual hit rate |
|---|---:|---:|---:|
| (0.0, 0.1] | 879 | 0.024 | 0.048 |
| (0.1, 0.2] | 221 | 0.153 | 0.149 |
| (0.2, 0.3] | 141 | 0.252 | 0.170 |
| (0.3, 0.4] | 66 | 0.346 | 0.258 |
| (0.4, 0.5] | 11 | 0.447 | 0.182 |
| (0.5, 0.6] | 3 | 0.530 | 0.333 |

### Reliability — top_bowler

| bin | n | mean p | actual hit rate |
|---|---:|---:|---:|
| (0.0, 0.1] | 806 | 0.031 | 0.036 |
| (0.1, 0.2] | 350 | 0.152 | 0.169 |
| (0.2, 0.3] | 146 | 0.244 | 0.185 |
| (0.3, 0.4] | 19 | 0.337 | 0.158 |

### Reliability — batter_50plus

| bin | n | mean p | actual hit rate |
|---|---:|---:|---:|
| (0.0, 0.1] | 580 | 0.026 | 0.041 |
| (0.1, 0.2] | 177 | 0.149 | 0.124 |
| (0.2, 0.3] | 123 | 0.250 | 0.195 |
| (0.3, 0.4] | 79 | 0.350 | 0.152 |
| (0.4, 0.5] | 19 | 0.447 | 0.158 |
| (0.5, 0.6] | 7 | 0.532 | 0.286 |

### Reliability — batter_6plus_six

| bin | n | mean p | actual hit rate |
|---|---:|---:|---:|
| (0.0, 0.1] | 50 | 0.038 | 0.020 |
| (0.1, 0.2] | 80 | 0.161 | 0.250 |
| (0.2, 0.3] | 114 | 0.258 | 0.316 |
| (0.3, 0.4] | 159 | 0.355 | 0.346 |
| (0.4, 0.5] | 198 | 0.454 | 0.429 |
| (0.5, 0.6] | 169 | 0.553 | 0.586 |
| (0.6, 0.7] | 160 | 0.651 | 0.500 |
| (0.7, 0.8] | 45 | 0.746 | 0.578 |
| (0.8, 0.9] | 10 | 0.833 | 0.600 |

## Continuous props

| family | n | MAE [95% CI] | mean bias (sim − actual) | P10–P90 coverage |
|---|---:|---:|---:|---:|
| batter_runs_mae | 985 | 15.42 [14.57, 16.28] | +2.00 | 73.20% |
| team_total_fours_mae | 120 | 3.98 [3.46, 4.51] | +0.22 | 60.83% |
| team_total_sixes_mae | 120 | 2.96 [2.52, 3.44] | -0.10 | 75.83% |
| team_first_over_mae | 120 | 3.71 [3.17, 4.26] | -2.34 | 59.17% |
| highest_individual_mae | 60 | 19.65 [16.46, 23.29] | +3.33 | 63.33% |
| batter_fours_mae | 985 | 1.47 [1.39, 1.56] | +0.08 | 88.12% |

Note: P10–P90 ideal coverage is 80%. Lower ⇒ sim under-disperses (over-confident); higher ⇒ over-disperses.
