# E2 v2 — Prop families vs FAIR baselines (not base rates)

Detail: `prop_detail_new_engine_full.json` (n=250 matches). Baselines built strictly as-of each match date from `data/t20s_json` (male T20s, innings 1–2). Δ = sim − baseline; **negative Δ ⇒ sim beats the fair baseline**. 95% CIs from cluster bootstrap by match (2,000 resamples).

**Baseline version:** `e2-v2-usage-top-bowler`. `top_bowler` uses EB-shrunk expected deliveries (K=5 XI appearances) × wickets/delivery (K=120 deliveries), normalized within the team. XI histories include zero-ball appearances. `bowler_wkts_{1,2,3}plus` retains the stronger EB-shrunk as-of threshold-rate baseline (K=20 bowling appearances).

## Binary families (Brier)

| family | n | Brier sim | Brier fair-base | ΔBrier | Δ 95% CI | verdict |
|---|---:|---:|---:|---:|---|---|
| `batter_50plus` | 4107 | 0.0797 | 0.0800 | -0.0003 | [-0.0013, +0.0007] | ≈ parity |
| `batter_6plus_six` | 4107 | 0.2196 | 0.2222 | -0.0027 | [-0.0046, -0.0006] | ✅ sim adds skill |
| `batter_fours_1plus` | 4107 | 0.2173 | 0.2242 | -0.0069 | [-0.0100, -0.0038] | ✅ sim adds skill |
| `batter_fours_2plus` | 4107 | 0.2014 | 0.2076 | -0.0062 | [-0.0090, -0.0036] | ✅ sim adds skill |
| `batter_fours_3plus` | 4107 | 0.1560 | 0.1594 | -0.0034 | [-0.0055, -0.0013] | ✅ sim adds skill |
| `bowler_wkts_1plus` | 3005 | 0.2448 | 0.2316 | +0.0131 | [+0.0083, +0.0177] | ❌ baseline wins |
| `bowler_wkts_2plus` | 3005 | 0.2081 | 0.1993 | +0.0088 | [+0.0054, +0.0121] | ❌ baseline wins |
| `bowler_wkts_3plus` | 3005 | 0.0929 | 0.0902 | +0.0027 | [+0.0016, +0.0039] | ❌ baseline wins |
| `first_wicket_runs_ou_30_5` | 500 | 0.2358 | 0.2329 | +0.0029 | [-0.0044, +0.0098] | ≈ parity |
| `highest_over_runs_ou_18_5` | 250 | 0.2407 | 0.2318 | +0.0089 | [-0.0043, +0.0232] | ≈ parity |
| `highest_over_runs_ou_24_5` | 250 | 0.1022 | 0.0996 | +0.0026 | [-0.0017, +0.0074] | ≈ parity |
| `innings_runs_ou_160_5` | 500 | 0.2310 | 0.2519 | -0.0209 | [-0.0337, -0.0071] | ✅ sim adds skill |
| `innings_runs_ou_170_5` | 500 | 0.2298 | 0.2399 | -0.0101 | [-0.0217, +0.0022] | ≈ parity |
| `innings_runs_ou_180_5` | 500 | 0.1936 | 0.2079 | -0.0142 | [-0.0231, -0.0052] | ✅ sim adds skill |
| `match_total_sixes_ou_15_5` | 250 | 0.1924 | 0.2120 | -0.0196 | [-0.0362, -0.0031] | ✅ sim adds skill |
| `match_total_sixes_ou_20_5` | 250 | 0.1041 | 0.1066 | -0.0024 | [-0.0087, +0.0031] | ≈ parity |
| `pp_total_ou_45_5` | 500 | 0.2358 | 0.2485 | -0.0127 | [-0.0228, -0.0029] | ✅ sim adds skill |
| `pp_total_ou_50_5` | 500 | 0.2345 | 0.2519 | -0.0174 | [-0.0274, -0.0073] | ✅ sim adds skill |
| `pp_total_ou_55_5` | 500 | 0.2036 | 0.2145 | -0.0109 | [-0.0199, -0.0017] | ✅ sim adds skill |
| `team_highest_individual_ou_29_5` | 500 | 0.0682 | 0.0724 | -0.0043 | [-0.0086, -0.0005] | ✅ sim adds skill |
| `team_highest_individual_ou_34_5` | 500 | 0.1202 | 0.1282 | -0.0080 | [-0.0147, -0.0021] | ✅ sim adds skill |
| `team_highest_individual_ou_39_5` | 500 | 0.1636 | 0.1724 | -0.0088 | [-0.0164, -0.0016] | ✅ sim adds skill |
| `top_batter` | 5593 | 0.0758 | 0.0752 | +0.0006 | [-0.0002, +0.0013] | ≈ parity |
| `top_bowler` | 5593 | 0.0780 | 0.0754 | +0.0026 | [+0.0017, +0.0035] | ❌ baseline wins |

## I13 count-baseline candidate decision

The analogous expected-balls × wicket-rate Poisson tail was evaluated but not promoted. Positive Δ below means that candidate has worse Brier score than the retained as-of threshold-rate baseline.

| family | retained Brier | usage-count Brier | Δ candidate − retained | Δ 95% CI | decision |
|---|---:|---:|---:|---|---|
| `bowler_wkts_1plus` | 0.2316 | 0.2379 | +0.0063 | [+0.0024, +0.0098] | retain threshold-rate |
| `bowler_wkts_2plus` | 0.1993 | 0.2035 | +0.0041 | [+0.0021, +0.0063] | retain threshold-rate |
| `bowler_wkts_3plus` | 0.0902 | 0.0912 | +0.0010 | [+0.0002, +0.0018] | retain threshold-rate |

## Continuous families (MAE)

| family | n | MAE sim | MAE fair-base | ΔMAE | Δ 95% CI | verdict |
|---|---:|---:|---:|---:|---|---|
| `batter_runs_mae` | 4107 | 14.01 | 14.80 | -0.79 | [-0.93, -0.66] | ✅ sim adds skill |
| `highest_individual_mae` | 250 | 15.99 | 16.39 | -0.40 | [-1.02, +0.17] | ≈ parity |
| `team_first_over_mae` | 500 | 3.30 | 3.32 | -0.02 | [-0.08, +0.04] | ≈ parity |
| `team_total_fours_mae` | 500 | 3.38 | 3.55 | -0.17 | [-0.27, -0.07] | ✅ sim adds skill |
| `team_total_sixes_mae` | 500 | 2.62 | 2.87 | -0.25 | [-0.37, -0.14] | ✅ sim adds skill |

## Skipped families

- `bowler_economy_ou_*`: fair career baseline ill-defined without modelling overs bowled per spell.
- `p_tie`: degenerate (ties ~0.4% of matches).
