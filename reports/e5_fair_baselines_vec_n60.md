# E2 — Prop families vs FAIR baselines (not base rates)

Detail: `prop_calibration_detail_vec_n60.json` (n=60 matches). Baselines built strictly as-of each match date from `data/t20s_json` (male T20s, innings 1–2). Δ = sim − baseline; **negative Δ ⇒ sim beats the fair baseline**. 95% CIs from cluster bootstrap by match (2,000 resamples).

## Binary families (Brier)

| family | n | Brier sim | Brier fair-base | ΔBrier | Δ 95% CI | verdict |
|---|---:|---:|---:|---:|---|---|
| `batter_50plus` | 985 | 0.0824 | 0.0766 | +0.0058 | [+0.0020, +0.0095] | ❌ baseline wins |
| `batter_6plus_six` | 985 | 0.2303 | 0.2240 | +0.0062 | [-0.0027, +0.0153] | ≈ parity |
| `batter_fours_1plus` | 985 | 0.2283 | 0.2256 | +0.0026 | [-0.0062, +0.0111] | ≈ parity |
| `batter_fours_2plus` | 985 | 0.2148 | 0.2133 | +0.0015 | [-0.0072, +0.0103] | ≈ parity |
| `batter_fours_3plus` | 985 | 0.1677 | 0.1592 | +0.0085 | [+0.0012, +0.0163] | ❌ baseline wins |
| `bowler_wkts_1plus` | 716 | 0.2521 | 0.2290 | +0.0231 | [+0.0101, +0.0362] | ❌ baseline wins |
| `bowler_wkts_2plus` | 716 | 0.2172 | 0.2029 | +0.0143 | [+0.0042, +0.0241] | ❌ baseline wins |
| `bowler_wkts_3plus` | 716 | 0.0904 | 0.0893 | +0.0011 | [-0.0023, +0.0043] | ≈ parity |
| `first_wicket_runs_ou_30_5` | 120 | 0.2444 | 0.2357 | +0.0087 | [-0.0140, +0.0325] | ≈ parity |
| `highest_over_runs_ou_18_5` | 60 | 0.2471 | 0.2496 | -0.0025 | [-0.0399, +0.0365] | ≈ parity |
| `highest_over_runs_ou_24_5` | 60 | 0.1412 | 0.1335 | +0.0077 | [-0.0031, +0.0203] | ≈ parity |
| `innings_runs_ou_160_5` | 120 | 0.2356 | 0.2760 | -0.0404 | [-0.0986, +0.0132] | ≈ parity |
| `innings_runs_ou_170_5` | 120 | 0.2527 | 0.2779 | -0.0253 | [-0.0885, +0.0375] | ≈ parity |
| `innings_runs_ou_180_5` | 120 | 0.2188 | 0.2448 | -0.0260 | [-0.0803, +0.0255] | ≈ parity |
| `match_total_sixes_ou_15_5` | 60 | 0.2124 | 0.2524 | -0.0400 | [-0.1186, +0.0343] | ≈ parity |
| `match_total_sixes_ou_20_5` | 60 | 0.1334 | 0.1644 | -0.0310 | [-0.0803, +0.0064] | ≈ parity |
| `pp_total_ou_45_5` | 120 | 0.2350 | 0.2669 | -0.0319 | [-0.0661, +0.0019] | ≈ parity |
| `pp_total_ou_50_5` | 120 | 0.2626 | 0.2734 | -0.0108 | [-0.0421, +0.0228] | ≈ parity |
| `pp_total_ou_55_5` | 120 | 0.2248 | 0.2308 | -0.0060 | [-0.0290, +0.0186] | ≈ parity |
| `team_highest_individual_ou_29_5` | 120 | 0.0553 | 0.0564 | -0.0010 | [-0.0063, +0.0051] | ≈ parity |
| `team_highest_individual_ou_34_5` | 120 | 0.1341 | 0.1385 | -0.0044 | [-0.0141, +0.0057] | ≈ parity |
| `team_highest_individual_ou_39_5` | 120 | 0.1730 | 0.1784 | -0.0053 | [-0.0230, +0.0128] | ≈ parity |
| `top_batter` | 1321 | 0.0787 | 0.0756 | +0.0030 | [+0.0003, +0.0057] | ❌ baseline wins |
| `top_bowler` | 1321 | 0.0773 | 0.0810 | -0.0037 | [-0.0064, -0.0007] | ✅ sim adds skill |

## Continuous families (MAE)

| family | n | MAE sim | MAE fair-base | ΔMAE | Δ 95% CI | verdict |
|---|---:|---:|---:|---:|---|---|
| `batter_runs_mae` | 985 | 15.42 | 15.11 | +0.31 | [-0.10, +0.72] | ≈ parity |
| `highest_individual_mae` | 60 | 19.65 | 23.52 | -3.87 | [-8.29, +0.27] | ≈ parity |
| `team_first_over_mae` | 120 | 3.71 | 3.39 | +0.33 | [+0.10, +0.56] | ❌ baseline wins |
| `team_total_fours_mae` | 120 | 3.98 | 4.12 | -0.15 | [-0.59, +0.25] | ≈ parity |
| `team_total_sixes_mae` | 120 | 2.96 | 3.25 | -0.30 | [-0.78, +0.16] | ≈ parity |

## Skipped families

- `bowler_economy_ou_*`: fair career baseline ill-defined without modelling overs bowled per spell.
- `p_tie`: degenerate (ties ~0.4% of matches).
- `team_total_fours_mae`: team fours not tracked in the corpus venue log (add on next corpus-cache rebuild).
