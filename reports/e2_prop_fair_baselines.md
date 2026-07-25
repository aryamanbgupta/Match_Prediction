# E2 v2 — Prop families vs FAIR baselines (not base rates)

Detail: `detail_d15_s43_n261.json` (n=261 matches). Baselines built strictly as-of each match date from `data/t20s_json` (male T20s, innings 1–2). Δ = sim − baseline; **negative Δ ⇒ sim beats the fair baseline**. 95% CIs from cluster bootstrap by match (2,000 resamples).

**Baseline version:** `e2-v2-usage-top-bowler`. `top_bowler` uses EB-shrunk expected deliveries (K=5 XI appearances) × wickets/delivery (K=120 deliveries), normalized within the team. XI histories include zero-ball appearances. `bowler_wkts_{1,2,3}plus` retains the stronger EB-shrunk as-of threshold-rate baseline (K=20 bowling appearances).

## Binary families (Brier)

| family | n | Brier sim | Brier fair-base | ΔBrier | Δ 95% CI | verdict |
|---|---:|---:|---:|---:|---|---|
| `batter_50plus` | 4252 | 0.0803 | 0.0789 | +0.0014 | [+0.0000, +0.0029] | ❌ baseline wins |
| `batter_6plus_six` | 4252 | 0.2241 | 0.2221 | +0.0020 | [-0.0014, +0.0053] | ≈ parity |
| `batter_fours_1plus` | 4252 | 0.2223 | 0.2241 | -0.0019 | [-0.0060, +0.0024] | ≈ parity |
| `batter_fours_2plus` | 4252 | 0.2057 | 0.2075 | -0.0018 | [-0.0054, +0.0019] | ≈ parity |
| `batter_fours_3plus` | 4252 | 0.1613 | 0.1595 | +0.0018 | [-0.0011, +0.0045] | ≈ parity |
| `bowler_wkts_1plus` | 3107 | 0.2569 | 0.2309 | +0.0260 | [+0.0196, +0.0322] | ❌ baseline wins |
| `bowler_wkts_2plus` | 3107 | 0.2162 | 0.2003 | +0.0159 | [+0.0112, +0.0207] | ❌ baseline wins |
| `bowler_wkts_3plus` | 3107 | 0.0945 | 0.0914 | +0.0031 | [+0.0016, +0.0046] | ❌ baseline wins |
| `first_wicket_runs_ou_30_5` | 522 | 0.2390 | 0.2356 | +0.0034 | [-0.0055, +0.0125] | ≈ parity |
| `highest_over_runs_ou_18_5` | 261 | 0.2417 | 0.2400 | +0.0017 | [-0.0193, +0.0216] | ≈ parity |
| `highest_over_runs_ou_24_5` | 261 | 0.1008 | 0.0956 | +0.0052 | [-0.0004, +0.0109] | ≈ parity |
| `innings_runs_ou_160_5` | 522 | 0.2476 | 0.2537 | -0.0060 | [-0.0304, +0.0187] | ≈ parity |
| `innings_runs_ou_170_5` | 522 | 0.2349 | 0.2403 | -0.0054 | [-0.0293, +0.0189] | ≈ parity |
| `innings_runs_ou_180_5` | 522 | 0.1953 | 0.2084 | -0.0131 | [-0.0354, +0.0092] | ≈ parity |
| `match_total_sixes_ou_15_5` | 261 | 0.2079 | 0.2298 | -0.0219 | [-0.0585, +0.0129] | ≈ parity |
| `match_total_sixes_ou_20_5` | 261 | 0.1093 | 0.1042 | +0.0051 | [-0.0131, +0.0225] | ≈ parity |
| `pp_total_ou_45_5` | 522 | 0.2500 | 0.2516 | -0.0015 | [-0.0217, +0.0200] | ≈ parity |
| `pp_total_ou_50_5` | 522 | 0.2540 | 0.2551 | -0.0011 | [-0.0220, +0.0197] | ≈ parity |
| `pp_total_ou_55_5` | 522 | 0.2184 | 0.2147 | +0.0038 | [-0.0136, +0.0208] | ≈ parity |
| `team_highest_individual_ou_29_5` | 522 | 0.0806 | 0.0799 | +0.0007 | [-0.0024, +0.0040] | ≈ parity |
| `team_highest_individual_ou_34_5` | 522 | 0.1335 | 0.1380 | -0.0045 | [-0.0102, +0.0009] | ≈ parity |
| `team_highest_individual_ou_39_5` | 522 | 0.1750 | 0.1822 | -0.0072 | [-0.0151, +0.0005] | ≈ parity |
| `top_batter` | 5835 | 0.0763 | 0.0750 | +0.0013 | [+0.0002, +0.0024] | ❌ baseline wins |
| `top_bowler` | 5835 | 0.0785 | 0.0747 | +0.0038 | [+0.0026, +0.0051] | ❌ baseline wins |

## I13 count-baseline candidate decision

The analogous expected-balls × wicket-rate Poisson tail was evaluated but not promoted. Positive Δ below means that candidate has worse Brier score than the retained as-of threshold-rate baseline.

| family | retained Brier | usage-count Brier | Δ candidate − retained | Δ 95% CI | decision |
|---|---:|---:|---:|---|---|
| `bowler_wkts_1plus` | 0.2309 | 0.2377 | +0.0067 | [+0.0031, +0.0106] | retain threshold-rate |
| `bowler_wkts_2plus` | 0.2003 | 0.2045 | +0.0043 | [+0.0022, +0.0064] | retain threshold-rate |
| `bowler_wkts_3plus` | 0.0914 | 0.0926 | +0.0011 | [+0.0004, +0.0020] | retain threshold-rate |

## Continuous families (MAE)

| family | n | MAE sim | MAE fair-base | ΔMAE | Δ 95% CI | verdict |
|---|---:|---:|---:|---:|---|---|
| `batter_runs_mae` | 4252 | 14.41 | 14.73 | -0.33 | [-0.49, -0.16] | ✅ sim adds skill |
| `highest_individual_mae` | 261 | 16.49 | 18.45 | -1.96 | [-3.52, -0.49] | ✅ sim adds skill |
| `team_first_over_mae` | 522 | 3.41 | 3.38 | +0.03 | [-0.05, +0.11] | ≈ parity |
| `team_total_fours_mae` | 522 | 3.76 | 3.65 | +0.11 | [-0.06, +0.28] | ≈ parity |
| `team_total_sixes_mae` | 522 | 2.86 | 2.97 | -0.11 | [-0.31, +0.10] | ≈ parity |

## Skipped families

- `bowler_economy_ou_*`: fair career baseline ill-defined without modelling overs bowled per spell.
- `p_tie`: degenerate (ties ~0.4% of matches).
