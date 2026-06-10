# E2 — Prop families vs FAIR baselines (not base rates)

Detail: `prop_calibration_detail_emp_n60.json` (n=60 matches). Baselines built strictly as-of each match date from `data/t20s_json` (male T20s, innings 1–2). Δ = sim − baseline; **negative Δ ⇒ sim beats the fair baseline**. 95% CIs from cluster bootstrap by match (2,000 resamples).

## Binary families (Brier)

| family | n | Brier sim | Brier fair-base | ΔBrier | Δ 95% CI | verdict |
|---|---:|---:|---:|---:|---|---|
| `batter_50plus` | 985 | 0.0786 | 0.0766 | +0.0020 | [-0.0011, +0.0051] | ≈ parity |
| `batter_6plus_six` | 985 | 0.2314 | 0.2240 | +0.0074 | [+0.0006, +0.0145] | ❌ baseline wins |
| `top_batter` | 1321 | 0.0777 | 0.0756 | +0.0021 | [-0.0007, +0.0050] | ≈ parity |
| `top_bowler` | 1321 | 0.0788 | 0.0812 | -0.0024 | [-0.0048, +0.0000] | ≈ parity |

## Continuous families (MAE)

| family | n | MAE sim | MAE fair-base | ΔMAE | Δ 95% CI | verdict |
|---|---:|---:|---:|---:|---|---|
| `batter_runs_mae` | 985 | 14.53 | 15.11 | -0.58 | [-0.93, -0.24] | ✅ sim adds skill |
| `highest_individual_mae` | 60 | 20.13 | 23.52 | -3.39 | [-5.89, -1.12] | ✅ sim adds skill |
| `team_first_over_mae` | 120 | 3.48 | 3.39 | +0.10 | [-0.10, +0.29] | ≈ parity |
| `team_total_fours_mae` | 120 | 4.23 | 4.12 | +0.11 | [-0.48, +0.67] | ≈ parity |
| `team_total_sixes_mae` | 120 | 3.04 | 3.25 | -0.21 | [-0.70, +0.25] | ≈ parity |

## Skipped families

- `bowler_economy_ou_*`: fair career baseline ill-defined without modelling overs bowled per spell.
- `p_tie`: degenerate (ties ~0.4% of matches).
- `team_total_fours_mae`: team fours not tracked in the corpus venue log (add on next corpus-cache rebuild).
