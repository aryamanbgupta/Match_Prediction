# Prop selector comparison — vector-cal vs raw

- Left  (`vector-cal`):  `reports/prop_calibration_detail_vec_n60.json`
- Right (`raw`): `reports/prop_calibration_detail_emp_n60.json`
- Paired bootstrap by match, 1000 resamples, seed=42

## Binary props (Brier)

| family | n | base rate | vector-cal Brier | raw Brier | Δ Brier (vector-cal−raw) | 95% CI | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| top_batter | 1321 | 0.090 | 0.0787 | 0.0777 | +0.0009 | [-0.0010, +0.0027] | ≈ tied |
| top_bowler | 1321 | 0.089 | 0.0773 | 0.0788 | -0.0015 | [-0.0035, +0.0007] | ≈ tied |
| batter_50plus | 985 | 0.088 | 0.0824 | 0.0786 | +0.0038 | [-0.0001, +0.0076] | ≈ tied |
| batter_6plus_six | 985 | 0.414 | 0.2303 | 0.2314 | -0.0011 | [-0.0079, +0.0055] | ≈ tied |

## Continuous props (MAE)

| family | n | vector-cal MAE | raw MAE | Δ MAE | 95% CI | Verdict |
|---|---:|---:|---:|---:|---:|---|
| batter_runs_mae | 985 | 15.42 | 14.53 | +0.89 | [+0.56, +1.20] | ❌ raw better |
| team_total_fours_mae | 120 | 3.98 | 4.23 | -0.26 | [-0.55, +0.05] | ≈ tied |
| team_total_sixes_mae | 120 | 2.96 | 3.04 | -0.09 | [-0.22, +0.04] | ≈ tied |
| team_first_over_mae | 120 | 3.71 | 3.48 | +0.23 | [-0.01, +0.49] | ≈ tied |
| highest_individual_mae | 60 | 19.65 | 20.13 | -0.48 | [-3.31, +2.44] | ≈ tied |

## Validation gates

**Gate G2 — top_bowler skill improvement**
- vector-cal Brier 0.0773 vs raw Brier 0.0788 (baseline 0.0813); gap closed = 1.8%. ❌ FAIL (target ≥40%).

**Gate G3 — top_batter no-regression**
- Δ Brier (vector-cal − raw) = +0.0009. ✅ PASS (target ≤ +0.003).
