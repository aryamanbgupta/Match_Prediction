# E2 — Prop families vs FAIR baselines (not base rates)

Detail: `prop_calibration_detail_vec_n261.json` (n=261 matches). Baselines built strictly as-of each match date from `data/t20s_json` (male T20s, innings 1–2). Δ = sim − baseline; **negative Δ ⇒ sim beats the fair baseline**. 95% CIs from cluster bootstrap by match (2,000 resamples).

## Binary families (Brier)

| family | n | Brier sim | Brier fair-base | ΔBrier | Δ 95% CI | verdict |
|---|---:|---:|---:|---:|---|---|
| `batter_50plus` | 4247 | 0.0834 | 0.0789 | +0.0044 | [+0.0024, +0.0063] | ❌ baseline wins |
| `batter_6plus_six` | 4247 | 0.2283 | 0.2221 | +0.0062 | [+0.0020, +0.0103] | ❌ baseline wins |
| `batter_fours_1plus` | 4247 | 0.2251 | 0.2241 | +0.0010 | [-0.0038, +0.0057] | ≈ parity |
| `batter_fours_2plus` | 4247 | 0.2095 | 0.2077 | +0.0018 | [-0.0024, +0.0062] | ≈ parity |
| `batter_fours_3plus` | 4247 | 0.1647 | 0.1596 | +0.0050 | [+0.0016, +0.0084] | ❌ baseline wins |
| `bowler_wkts_1plus` | 3107 | 0.2614 | 0.2309 | +0.0305 | [+0.0234, +0.0373] | ❌ baseline wins |
| `bowler_wkts_2plus` | 3107 | 0.2162 | 0.2003 | +0.0159 | [+0.0112, +0.0208] | ❌ baseline wins |
| `bowler_wkts_3plus` | 3107 | 0.0943 | 0.0914 | +0.0029 | [+0.0013, +0.0044] | ❌ baseline wins |
| `first_wicket_runs_ou_30_5` | 522 | 0.2401 | 0.2356 | +0.0046 | [-0.0055, +0.0149] | ≈ parity |
| `highest_over_runs_ou_18_5` | 261 | 0.2900 | 0.2400 | +0.0499 | [+0.0217, +0.0781] | ❌ baseline wins |
| `highest_over_runs_ou_24_5` | 261 | 0.1039 | 0.0956 | +0.0083 | [+0.0017, +0.0152] | ❌ baseline wins |
| `innings_runs_ou_160_5` | 522 | 0.2398 | 0.2537 | -0.0138 | [-0.0365, +0.0096] | ≈ parity |
| `innings_runs_ou_170_5` | 522 | 0.2389 | 0.2403 | -0.0014 | [-0.0232, +0.0219] | ≈ parity |
| `innings_runs_ou_180_5` | 522 | 0.2021 | 0.2084 | -0.0063 | [-0.0249, +0.0129] | ≈ parity |
| `match_total_sixes_ou_15_5` | 261 | 0.2263 | 0.2298 | -0.0035 | [-0.0337, +0.0249] | ≈ parity |
| `match_total_sixes_ou_20_5` | 261 | 0.1028 | 0.1042 | -0.0014 | [-0.0141, +0.0100] | ≈ parity |
| `pp_total_ou_45_5` | 522 | 0.2557 | 0.2516 | +0.0041 | [-0.0113, +0.0193] | ≈ parity |
| `pp_total_ou_50_5` | 522 | 0.2675 | 0.2551 | +0.0124 | [-0.0026, +0.0275] | ≈ parity |
| `pp_total_ou_55_5` | 522 | 0.2298 | 0.2147 | +0.0151 | [+0.0030, +0.0267] | ❌ baseline wins |
| `team_highest_individual_ou_29_5` | 522 | 0.0811 | 0.0799 | +0.0012 | [-0.0018, +0.0048] | ≈ parity |
| `team_highest_individual_ou_34_5` | 522 | 0.1339 | 0.1380 | -0.0041 | [-0.0097, +0.0012] | ≈ parity |
| `team_highest_individual_ou_39_5` | 522 | 0.1774 | 0.1822 | -0.0048 | [-0.0134, +0.0037] | ≈ parity |
| `top_batter` | 5835 | 0.0784 | 0.0750 | +0.0034 | [+0.0021, +0.0047] | ❌ baseline wins |
| `top_bowler` | 5835 | 0.0778 | 0.0801 | -0.0023 | [-0.0038, -0.0008] | ✅ sim adds skill |

## Continuous families (MAE)

| family | n | MAE sim | MAE fair-base | ΔMAE | Δ 95% CI | verdict |
|---|---:|---:|---:|---:|---|---|
| `batter_runs_mae` | 4247 | 14.64 | 14.74 | -0.10 | [-0.30, +0.11] | ≈ parity |
| `highest_individual_mae` | 261 | 16.41 | 18.45 | -2.05 | [-3.82, -0.39] | ✅ sim adds skill |
| `team_first_over_mae` | 522 | 3.53 | 3.38 | +0.15 | [+0.04, +0.26] | ❌ baseline wins |
| `team_total_fours_mae` | 522 | 3.68 | 3.65 | +0.02 | [-0.16, +0.20] | ≈ parity |
| `team_total_sixes_mae` | 522 | 2.87 | 2.97 | -0.10 | [-0.28, +0.08] | ≈ parity |

## Skipped families

- `bowler_economy_ou_*`: fair career baseline ill-defined without modelling overs bowled per spell.
- `p_tie`: degenerate (ties ~0.4% of matches).
- `team_total_fours_mae`: team fours not tracked in the corpus venue log (add on next corpus-cache rebuild).
