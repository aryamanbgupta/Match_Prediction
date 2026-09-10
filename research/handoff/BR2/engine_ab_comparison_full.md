# Prop selector comparison — new_engine vs recorded

- Left  (`new_engine`):  `eval_out/br2_gates/prop_detail_new_engine_full.json`
- Right (`recorded`): `reports/prop_calibration_detail_emp_n261.json`
- Paired bootstrap by match, 1000 resamples, seed=42

## Binary props (Brier)

| family | n | base rate | new_engine Brier | recorded Brier | Δ Brier (new_engine−recorded) | 95% CI | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| top_batter | 5593 | 0.089 | 0.0758 | 0.0778 | -0.0020 | [-0.0032, -0.0008] | ✅ new_engine better |
| top_bowler | 5593 | 0.089 | 0.0780 | 0.0798 | -0.0018 | [-0.0033, -0.0004] | ✅ new_engine better |
| batter_50plus | 4107 | 0.092 | 0.0797 | 0.0827 | -0.0030 | [-0.0044, -0.0016] | ✅ new_engine better |
| batter_6plus_six | 4107 | 0.399 | 0.2196 | 0.2290 | -0.0095 | [-0.0127, -0.0060] | ✅ new_engine better |
| innings_runs_ou_160_5 | 500 | 0.490 | 0.2310 | 0.2431 | -0.0122 | [-0.0275, +0.0034] | ≈ tied |
| innings_runs_ou_170_5 | 500 | 0.382 | 0.2298 | 0.2336 | -0.0038 | [-0.0200, +0.0109] | ≈ tied |
| innings_runs_ou_180_5 | 500 | 0.280 | 0.1936 | 0.2030 | -0.0093 | [-0.0264, +0.0068] | ≈ tied |
| batter_fours_1plus | 4107 | 0.584 | 0.2173 | 0.2247 | -0.0074 | [-0.0104, -0.0043] | ✅ new_engine better |
| batter_fours_2plus | 4107 | 0.360 | 0.2014 | 0.2093 | -0.0080 | [-0.0111, -0.0049] | ✅ new_engine better |
| batter_fours_3plus | 4107 | 0.225 | 0.1560 | 0.1642 | -0.0082 | [-0.0110, -0.0056] | ✅ new_engine better |
| bowler_wkts_1plus | 3005 | 0.616 | 0.2448 | 0.2627 | -0.0180 | [-0.0234, -0.0125] | ✅ new_engine better |
| bowler_wkts_2plus | 3005 | 0.282 | 0.2081 | 0.2370 | -0.0289 | [-0.0354, -0.0223] | ✅ new_engine better |
| bowler_wkts_3plus | 3005 | 0.101 | 0.0929 | 0.1052 | -0.0123 | [-0.0158, -0.0088] | ✅ new_engine better |
| team_highest_individual_ou_29_5 | 500 | 0.918 | 0.0682 | 0.0835 | -0.0154 | [-0.0216, -0.0091] | ✅ new_engine better |
| team_highest_individual_ou_34_5 | 500 | 0.844 | 0.1202 | 0.1340 | -0.0138 | [-0.0262, -0.0013] | ✅ new_engine better |
| team_highest_individual_ou_39_5 | 500 | 0.772 | 0.1636 | 0.1895 | -0.0259 | [-0.0404, -0.0106] | ✅ new_engine better |
| pp_total_ou_45_5 | 500 | 0.616 | 0.2358 | 0.2623 | -0.0265 | [-0.0499, -0.0017] | ✅ new_engine better |
| pp_total_ou_50_5 | 500 | 0.440 | 0.2345 | 0.2827 | -0.0483 | [-0.0773, -0.0200] | ✅ new_engine better |
| pp_total_ou_55_5 | 500 | 0.304 | 0.2036 | 0.2449 | -0.0413 | [-0.0677, -0.0174] | ✅ new_engine better |
| match_total_sixes_ou_15_5 | 250 | 0.316 | 0.1924 | 0.2217 | -0.0293 | [-0.0516, -0.0076] | ✅ new_engine better |
| match_total_sixes_ou_20_5 | 250 | 0.120 | 0.1041 | 0.1097 | -0.0056 | [-0.0182, +0.0088] | ≈ tied |
| first_wicket_runs_ou_30_5 | 500 | 0.362 | 0.2358 | 0.2527 | -0.0169 | [-0.0284, -0.0057] | ✅ new_engine better |
| bowler_economy_ou_8_5 | 3005 | 0.461 | 0.2503 | 0.2562 | -0.0059 | [-0.0150, +0.0031] | ≈ tied |
| bowler_economy_ou_10_5 | 3005 | 0.254 | 0.1935 | 0.1966 | -0.0030 | [-0.0093, +0.0033] | ≈ tied |
| p_tie | 250 | 0.012 | 0.0119 | 0.0001 | +0.0118 | [+0.0002, +0.0237] | ❌ recorded better |
| highest_over_runs_ou_18_5 | 250 | 0.628 | 0.2407 | 0.2456 | -0.0050 | [-0.0194, +0.0083] | ≈ tied |
| highest_over_runs_ou_24_5 | 250 | 0.112 | 0.1022 | 0.1055 | -0.0033 | [-0.0077, +0.0004] | ≈ tied |

## Continuous props (MAE)

| family | n | new_engine MAE | recorded MAE | Δ MAE | 95% CI | Verdict |
|---|---:|---:|---:|---:|---:|---|
| batter_runs_mae | 4107 | 14.01 | 14.09 | -0.08 | [-0.24, +0.07] | ≈ tied |
| team_total_fours_mae | 500 | 3.38 | 3.89 | -0.51 | [-0.76, -0.28] | ✅ new_engine better |
| team_total_sixes_mae | 500 | 2.62 | 2.90 | -0.28 | [-0.41, -0.14] | ✅ new_engine better |
| team_first_over_mae | 500 | 3.30 | 3.34 | -0.04 | [-0.13, +0.06] | ≈ tied |
| highest_individual_mae | 250 | 15.99 | 16.65 | -0.66 | [-1.68, +0.31] | ≈ tied |
| batter_fours_mae | 4107 | 1.37 | 1.45 | -0.08 | [-0.10, -0.06] | ✅ new_engine better |

## Validation gates

**Gate G2 — top_bowler skill improvement**
- new_engine Brier 0.0780 vs recorded Brier 0.0793 (baseline 0.0806); gap closed = 1.7%. ❌ FAIL (target ≥40%).

**Gate G3 — top_batter no-regression**
- Δ Brier (new_engine − recorded) = -0.0017. ✅ PASS (target ≤ +0.003).
