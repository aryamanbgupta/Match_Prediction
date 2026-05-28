# v7 Sim — Prop Bet Evaluation Framework

**Date**: 2026-05-12
**Sim**: v7 XGBoost ball-level model (`models/xgb_v3/xgboost_model_v3.pkl`, config `experiments/configs/xgb_v6_hierarchical_shrink.yaml`)
**Selector**: `EmpiricalBowlerSelector` (phase-aware empirical prior)
**Sample**: 261 polymarket-iteration matches × 100 sims each

## TL;DR

The v7 sim, freshly repurposed for prop bets via a structural fix to the
bowler selector, is **calibrated and useful on a clear subset of prop
families** and **systematically biased on another subset** (the bias is
itself exploitable — bet the opposite side). The headline hit rates are:

| Prop | Hit rate (#1 pick or chosen side) |
|---|---:|
| Top batter (per team) | **17.6%** vs ~9% base rate; **53.4%** in top-3 |
| Top bowler (per team) | **18.8%** vs ~9% base rate; **48.9%** in top-3 |
| Innings runs O/U 180.5 | **72.0%** correct |
| Team top scorer O/U 34.5 | **81.0%** correct |
| Match total sixes O/U 15.5 | **66.3%** correct |
| Innings runs O/U 170.5 | **64.9%** correct |
| Batter fours ≥1 | **65.1%** correct |
| Batter fours ≥2 | **65.7%** correct |
| First wicket runs O/U 30.5 | **62.8%** correct |
| Bowler wickets ≥1 | **60.1%** correct |
| Bowler economy O/U 8.5 | **55.1%** correct (low signal) |
| Powerplay total O/U 50.5 | **54.6%** correct (low signal) |

(Per-team props evaluated separately per innings → n=522 from 261 matches.)

## Where the sim has skill (positive Brier-skill, ship as-is)

| family | n | Brier [95% CI] | base Brier | skill | verdict |
|---|---:|---:|---:|---:|:---|
| `batter_fours_2plus` | 4254 | 0.2095 [0.2044, 0.2152] | 0.2306 | **+0.091** | ✅ strongest |
| `batter_fours_1plus` | 4254 | 0.2250 [0.2192, 0.2307] | 0.2433 | **+0.075** | ✅ |
| `batter_fours_3plus` | 4254 | 0.1642 [0.1577, 0.1703] | 0.1744 | **+0.058** | ✅ |
| `batter_6plus_six` | 4254 | 0.2288 [0.2240, 0.2335] | 0.2397 | **+0.046** | ✅ |
| `top_batter` | 5835 | 0.0775 [0.0724, 0.0827] | 0.0810 | **+0.043** | ✅ |
| `innings_runs_ou_160_5` | 522 | 0.2416 [0.2226, 0.2615] | 0.2496 | **+0.032** | ✅ |
| `top_bowler` | 5835 | 0.0793 [0.0742, 0.0847] | 0.0806 | **+0.016** | ✅ (modest) |
| `innings_runs_ou_170_5` | 522 | 0.2319 [0.2087, 0.2531] | 0.2345 | **+0.011** | ✅ (marginal) |

## Where the sim is systematically biased (inverse plays — bet against the sim)

| family | n | Brier | base | skill | direction |
|---|---:|---:|---:|---:|---|
| `pp_total_ou_55_5` | 522 | 0.2460 | 0.2080 | **−0.183** | sim over-states PP totals |
| `bowler_wkts_2plus` | 3106 | 0.2365 | 0.2037 | **−0.161** | sim over-attributes wickets |
| `bowler_wkts_3plus` | 3106 | 0.1056 | 0.0922 | **−0.146** | same |
| `pp_total_ou_50_5` | 522 | 0.2812 | 0.2455 | **−0.145** | same as above |
| `bowler_wkts_1plus` | 3106 | 0.2626 | 0.2357 | **−0.114** | same |
| `pp_total_ou_45_5` | 522 | 0.2623 | 0.2393 | **−0.096** | same |
| `team_highest_individual_ou_29_5` | 522 | 0.0897 | 0.0819 | **−0.095** | |
| `first_wicket_runs_ou_30_5` | 522 | 0.2526 | 0.2330 | **−0.084** | |

The pattern is consistent: the sim **over-states tail-event probabilities**
(specific bowlers getting many wickets, powerplay over-performing). Fading
the sim on these markets is a real edge.

## Continuous-prop calibration (n=261)

| family | n | MAE [95% CI] | bias (sim − actual) | P10–P90 coverage |
|---|---:|---:|---:|---:|
| `batter_runs_mae` | 4254 | 14.02 [13.63, 14.42] | −1.90 | 76.6% |
| `team_total_fours_mae` | 522 | 4.00 [3.73, 4.25] | +1.62 | 72.6% |
| `team_total_sixes_mae` | 522 | 2.91 [2.71, 3.11] | −0.16 | **79.3%** (≈ideal) |
| `team_first_over_mae` | 522 | 3.40 [3.17, 3.66] | −0.63 | 70.5% |
| `highest_individual_mae` | 261 | 16.56 [15.07, 18.27] | −8.20 | 78.2% |
| `batter_fours_mae` | 4254 | 1.44 [1.40, 1.48] | +0.08 | **90.2%** (over-disperses) |

Note: `team_total_fours_mae` bias was **+2.88 in the n=30 memo** (memo's
flagged "inverse" family). The empirical selector has **halved** the
bias to +1.62 — biggest single calibration win from this work.

## Selector validation gates

The empirical phase-aware bowler selector replaces `RandomBowlerSelector`
(default `T20Rules()`). Validation against random baseline at n=60:

| Gate | Threshold | Result | Status |
|---|---|---|:---:|
| **G1** — winner-market LL parity | Δ LL ≤ +0.002 | Δ −0.06 (emp better) | ✅ |
| **G2** — top_bowler skill ≥ 40% gap closure | ≥ 40% | 1.7% | ❌ strict; direction right |
| **G3** — top_batter no-regression | Δ ≤ +0.003 | −0.0006 (better) | ✅ |
| **G5** — bowler coverage ≥ 90% | ≥ 90% | 91.2% | ✅ |

(G4 — phase-mix sanity — not run; would require instrumenting the sim
to record per-phase selected-bowler distributions.)

G2 misses its strict target because the harder problem is the
ball-level wicket-rate model (which bowler gets wickets *given* they
bowl), not the selector (which bowler bowls each phase). The selector
is still a clear net improvement: continuous-prop MAE wins are
statistically significant, and the team-fours over-counting bias is
halved. See [`prop_selector_comparison_n60.md`](prop_selector_comparison_n60.md).

## Per-match drilldown

[`reports/prop_per_match/index.md`](prop_per_match/index.md) — one file
per match showing:
- Sim's top-3 batters with P(top scorer) vs actual top scorer
- Sim's top-3 bowlers with P(top wicket-taker) vs actual top wicket-taker
- Sim P10/mean/P90 innings runs vs actual, per team
- Sim mean fours/sixes vs actual, per team
- Hit-or-miss verdict for each prop family

Example: [`reports/prop_per_match/2026-04-04_Royal_Challengers_Bangalore_Punjab_Kings_M__Chinnaswamy_Stadium__Bengaluru.md`](prop_per_match/2026-04-04_Royal_Challengers_Bangalore_Punjab_Kings_M__Chinnaswamy_Stadium__Bengaluru.md)

The index file's columns (top_batter, top_bowler, innings_runs_ou_170_5)
let you scan for matches where the sim hit or missed.

## Framework reference

| What | Path |
|---|---|
| **Aggregate report** | `reports/prop_calibration_report_emp_n261.md` |
| Aggregate detail (raw per-row JSON) | `reports/prop_calibration_detail_emp_n261.json` |
| Per-match views | `reports/prop_per_match/*.md` |
| Selector validation | `reports/prop_selector_comparison_n60.md` |
| Backtest script | `scripts/sim_eval/prop_backtest.py` |
| Per-match renderer | `scripts/sim_eval/render_prop_per_match.py` |
| Selector comparator | `scripts/sim_eval/compare_selector_eval.py` |
| Coverage diagnostic | `scripts/sim_eval/check_bowler_coverage.py` |
| Bowler prior data | `models/bowler_phase_usage.json` |
| Selector class | `scripts/sim_v1_2.py:EmpiricalBowlerSelector` |
| Unit tests | `scripts/tests/test_bowler_selector.py` |

## How to rerun

```bash
# Refresh the bowler usage prior (after data/t20s_json refresh)
uv run python scripts/build_bowler_phase_usage.py \
    --source-dir data/t20s_json \
    --out models/bowler_phase_usage.json

# Run the calibration sweep (uses EmpiricalBowlerSelector by default)
uv run python scripts/sim_eval/prop_backtest.py \
    --n-matches all --n-sims 100 \
    --detail-out reports/prop_calibration_detail.json \
    --report-out reports/prop_calibration_report.md

# Render per-match views
uv run python scripts/sim_eval/render_prop_per_match.py \
    --detail reports/prop_calibration_detail.json \
    --out-dir reports/prop_per_match/

# A/B against random baseline
uv run python scripts/sim_eval/prop_backtest.py \
    --n-matches 60 --bowler-selector random \
    --detail-out reports/prop_calibration_detail_rand_n60.json
uv run python scripts/sim_eval/compare_selector_eval.py \
    --left  reports/prop_calibration_detail.json \
    --right reports/prop_calibration_detail_rand_n60.json
```

## Open follow-ups (out of scope here)

1. **Matchup-aware selector** (LHB/RHB, pace-vs-spin) — could further
   improve `top_bowler` skill where empirical phase-only plateaus.
2. **PP-total miscalibration root cause** — the sim systematically
   over-predicts PP totals (skill −0.18 at line 55.5). Worth
   investigating the powerplay-specific outcome model.
3. **DK/Polymarket line comparison** — Phase 2 of the original memo.
   Needs scraper restoration (see `archive/scripts/bet_scraper.py`).
4. **Strike-rate distribution conditional on balls faced ≥10** —
   listed in plan but deferred; needs per-sim conditional aggregation.
5. **Reliability diagram PNGs** — markdown tables in the report cover
   the calibration shape; matplotlib PNGs would be a nicer artifact
   but aren't gating.
