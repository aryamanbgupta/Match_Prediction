# E2 — Prop families vs FAIR baselines (not base rates)

Detail: `prop_calibration_detail_emp_n261.json` (n=261 matches). Baselines built strictly as-of each match date from `data/t20s_json` (male T20s, innings 1–2). Δ = sim − baseline; **negative Δ ⇒ sim beats the fair baseline**. 95% CIs from cluster bootstrap by match (2,000 resamples).

## Binary families (Brier)

| family | n | Brier sim | Brier fair-base | ΔBrier | Δ 95% CI | verdict |
|---|---:|---:|---:|---:|---|---|
| `batter_50plus` | 4254 | 0.0813 | 0.0788 | +0.0025 | [+0.0010, +0.0040] | ❌ baseline wins |
| `batter_6plus_six` | 4254 | 0.2288 | 0.2220 | +0.0067 | [+0.0032, +0.0103] | ❌ baseline wins |
| `batter_fours_1plus` | 4254 | 0.2250 | 0.2241 | +0.0009 | [-0.0033, +0.0050] | ≈ parity |
| `batter_fours_2plus` | 4254 | 0.2095 | 0.2074 | +0.0021 | [-0.0018, +0.0060] | ≈ parity |
| `batter_fours_3plus` | 4254 | 0.1642 | 0.1594 | +0.0048 | [+0.0017, +0.0078] | ❌ baseline wins |
| `bowler_wkts_1plus` | 3106 | 0.2626 | 0.2309 | +0.0317 | [+0.0249, +0.0385] | ❌ baseline wins |
| `bowler_wkts_2plus` | 3106 | 0.2365 | 0.2003 | +0.0362 | [+0.0296, +0.0432] | ❌ baseline wins |
| `bowler_wkts_3plus` | 3106 | 0.1056 | 0.0914 | +0.0142 | [+0.0111, +0.0176] | ❌ baseline wins |
| `first_wicket_runs_ou_30_5` | 522 | 0.2526 | 0.2356 | +0.0171 | [+0.0052, +0.0300] | ❌ baseline wins |
| `highest_over_runs_ou_18_5` | 261 | 0.2470 | 0.2400 | +0.0070 | [-0.0139, +0.0274] | ≈ parity |
| `highest_over_runs_ou_24_5` | 261 | 0.1011 | 0.0956 | +0.0055 | [+0.0000, +0.0116] | ❌ baseline wins |
| `innings_runs_ou_160_5` | 522 | 0.2416 | 0.2537 | -0.0120 | [-0.0338, +0.0106] | ≈ parity |
| `innings_runs_ou_170_5` | 522 | 0.2319 | 0.2403 | -0.0083 | [-0.0289, +0.0128] | ≈ parity |
| `innings_runs_ou_180_5` | 522 | 0.1989 | 0.2084 | -0.0095 | [-0.0294, +0.0107] | ≈ parity |
| `match_total_sixes_ou_15_5` | 261 | 0.2231 | 0.2298 | -0.0068 | [-0.0360, +0.0225] | ≈ parity |
| `match_total_sixes_ou_20_5` | 261 | 0.1056 | 0.1042 | +0.0014 | [-0.0132, +0.0151] | ≈ parity |
| `pp_total_ou_45_5` | 522 | 0.2623 | 0.2516 | +0.0108 | [-0.0143, +0.0375] | ≈ parity |
| `pp_total_ou_50_5` | 522 | 0.2812 | 0.2551 | +0.0261 | [-0.0027, +0.0544] | ≈ parity |
| `pp_total_ou_55_5` | 522 | 0.2460 | 0.2147 | +0.0314 | [+0.0062, +0.0570] | ❌ baseline wins |
| `team_highest_individual_ou_29_5` | 522 | 0.0897 | 0.0799 | +0.0099 | [+0.0021, +0.0167] | ❌ baseline wins |
| `team_highest_individual_ou_34_5` | 522 | 0.1405 | 0.1380 | +0.0025 | [-0.0136, +0.0178] | ≈ parity |
| `team_highest_individual_ou_39_5` | 522 | 0.1939 | 0.1822 | +0.0117 | [-0.0062, +0.0297] | ≈ parity |
| `top_batter` | 5835 | 0.0775 | 0.0750 | +0.0025 | [+0.0013, +0.0038] | ❌ baseline wins |
| `top_bowler` | 5835 | 0.0793 | 0.0801 | -0.0008 | [-0.0020, +0.0006] | ≈ parity |

## Continuous families (MAE)

| family | n | MAE sim | MAE fair-base | ΔMAE | Δ 95% CI | verdict |
|---|---:|---:|---:|---:|---|---|
| `batter_runs_mae` | 4254 | 14.02 | 14.73 | -0.71 | [-0.89, -0.52] | ✅ sim adds skill |
| `highest_individual_mae` | 261 | 16.56 | 18.45 | -1.89 | [-2.97, -0.90] | ✅ sim adds skill |
| `team_first_over_mae` | 522 | 3.40 | 3.38 | +0.02 | [-0.08, +0.12] | ≈ parity |
| `team_total_sixes_mae` | 522 | 2.91 | 2.97 | -0.06 | [-0.24, +0.12] | ≈ parity |

## Skipped families

- `bowler_economy_ou_*`: fair career baseline ill-defined without modelling overs bowled per spell.
- `p_tie`: degenerate (ties ~0.4% of matches).
- `team_total_fours_mae`: team fours not tracked in the corpus venue log (add on next corpus-cache rebuild).

## Caveat — MC noise floor in the sim's Brier

Sim probabilities are MC estimates from **100 sims**, so sampling noise
alone inflates the sim's Brier by Var(p̂) ≈ p(1−p)/100 ≤ **0.0025**
(worst case at p=0.5; ~0.001 at the top_batter p≈0.1 range). The fair
baselines are deterministic rates and carry no such penalty.
Consequently:

- "❌ baseline wins" rows with Δ ≲ 0.003 (`top_batter` +0.0025,
  `batter_50plus` +0.0025) should be read as **parity within MC noise**,
  not as the sim being genuinely worse.
- Rows with Δ ≥ +0.01 (`bowler_wkts_*` +0.014…+0.036, `pp_total_ou_55_5`
  +0.031, `first_wicket_runs_ou_30_5` +0.017, `team_highest_individual_
  ou_29_5` +0.010) are far above the noise floor — **robustly real**.
- No "✅" verdict among binary families is affected (there are none).

## Verdict (rewrites the 2026-05-12 prop framework conclusions)

1. **No binary prop family demonstrates sim skill over a fair baseline.**
   The `prop_framework_summary.md` "ship as-is" list (batter fours,
   top-batter/bowler ranking, innings totals) was an artifact of
   measuring skill against *base rates*. Against as-of EB-shrunk
   career/venue/positional baselines, the best binary families are
   parity; several lose.
2. **The sim's real value-add is continuous score distributions**:
   per-batter expected runs (MAE 14.02 vs career baseline 14.73,
   Δ CI [−0.89, −0.52]) and match highest-individual score (16.56 vs
   18.45, Δ CI [−2.97, −0.90]). This is consistent with the sim's
   scenario-generator role: it composes venue + lineup + matchup
   interactions that a single career rate can't.
3. **The "inverse play" thesis needs reframing.** Fading the sim on
   `bowler_wkts_*` / `pp_total_*` is only profitable against a
   counterparty pricing *like the sim*. A market pricing at the fair
   baseline (the natural anchor for a competent book) leaves no room:
   the baseline already beats both the sim and its inverse. Without
   captured prop lines there is no evidence of a deployable edge.
4. **Betting candidates from the sim, if any, live in the continuous
   families** (player runs O/U at bookmaker lines, highest-score
   markets) where the sim genuinely beats career-rate pricing. These
   need real prop lines (scraper restoration) before any ROI claim.

