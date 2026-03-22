# CricML Improvements & Research Findings

**Last Updated**: March 2025

---

## Honest Assessment

### What's Working
- **Ball-level simulation approach**: Validated by Kuo 2021 (beat Bet365 using similar methods). 4M+ training examples vs 15K matches.
- **Temporal integrity**: Stats snapshots prevent data leakage. Correct approach.
- **Chunked caching**: 14x memory reduction (7.6GB → 550MB), lazy loading works well.
- **Feature engineering**: 46+ features covering player stats, matchups, metadata, momentum.

### What's Broken
- **Critical bug (FIXED)**: XGBoost `class_to_outcome` mapping was 8-class instead of 6-class. Class 4 (six) mapped to 'four', class 5 (wicket) mapped to 'six'. **Wickets never occurred in XGBoost simulation.** All previous evaluation results (29.4% edge, 42/44 profitable bets) were artifacts. Fixed in `sim_v1_2.py` for both `XGBoostModelV2` and `XGBoostModel`.
- **Evaluation JSON was incomplete (FIXED)**: `run_sim_eval.py` only saved log loss, brier score, and a misleading `profitable_bets` count (always = n_matches by construction). Real betting metrics (P&L, ROI, win rate, Kelly) were computed but never persisted. Now saves all metrics with timestamped filenames.
- **No team-level signal**: Model can't distinguish India from Scotland. Only sees current batter/bowler, not aggregate team strength.
- **Miscalibrated evaluation**: 44-match test set is too small for meaningful confidence intervals.

### Post-Bug-Fix Evaluation Results (March 2025)
All 4 models evaluated on 44 T20 World Cup 2024 matches with 100 simulations each:

| Model | Avg Log Loss | Flat ROI | Win Rate | Frac Kelly ROI |
|-------|-------------|----------|----------|----------------|
| XGBoost v3 | 0.8754 | -43.9% | 26.8% | -5.7% |
| LSTM v1 | 0.7229 | -63.4% | 12.2% | -6.3% |
| Transformer v1 | 0.6880 | -63.4% | 12.2% | -5.7% |
| MLP v1 | 0.6997 | -63.8% | 12.2% | -6.1% |

**Conclusion**: All models lose money. The #1 blocker is missing team-level features — models predict ~50/50 for every match regardless of team strength, then always bet on the underdog (perceived "edge" is really just disagreement with market).

### Post-Team-Strength Evaluation Results (March 2026)
XGBoost v4 with player-level ELO + aggregated team strength (9 new features, 72 total):

| Model | Avg Log Loss | Avg Brier | Flat ROI | Win Rate | Frac Kelly ROI |
|-------|-------------|-----------|----------|----------|----------------|
| XGBoost v3 (baseline) | 0.8754 | 0.3168 | -43.9% | 26.8% | -5.7% |
| **XGBoost v4 (team strength)** | **0.7100** | **0.2554** | -44.8% | 24.4% | -5.5% |
| Change | **-18.9%** | **-19.4%** | ~same | ~same | ~same |

**Key findings**:
- **Log Loss improved 19%** — model now produces more differentiated win probabilities
- **Brier Score improved 19%** — probability calibration significantly better
- ROI unchanged — better probabilities haven't translated to profitable bets yet (needs calibration layer)
- 3 team features in XGBoost top 15: `striker_elo` (#9), `team_batting_sr` (#10), `batting_team_elo` (#14)

**ELO fixes applied** (vs initial broken implementation that had zero impact):
1. Fixed outcome scaling: dot ball was mapping to 0.0 (same as wicket), now maps to 0.4
2. Gender filter: women's cricket excluded from ELO (different competition pool)
3. Context-aware K-factor: premium matches (IPL/WC/full-member T20I) K=4.0, standard leagues K=2.0, associate bilaterals K=1.0

**Experiment**: `xgb_v4_team_strength_20260321_110247_40befa4`

---

## Research Findings

### Calibration > Accuracy
- Walsh & Joshi 2024: Calibration-optimized models generate **69.86% higher betting returns** than accuracy-optimized ones.
- Target ECE (Expected Calibration Error) < 0.015.
- Methods: Isotonic regression, temperature scaling, Platt scaling.

### Realistic Edge Sizes
- Market-beating edge is typically **1-3% ROI** (not 29%).
- Kuo 2021 achieved ~2-3% edge over Bet365 using ball-by-ball simulation.
- Any model showing >5% edge on a small sample is likely overfitting or has a bug.

### Decorrelation from Market
- Hubacek 2019: Profitability comes from **decorrelation** with market predictions, not just accuracy.
- Models that agree with the market on most games but disagree on a few high-confidence spots are most profitable.

### GBMs vs Deep Learning on Tabular Data
- GBMs (XGBoost, LightGBM) still outperform deep learning on structured/tabular data.
- Neural models (LSTM ~44%, Transformer ~44%) significantly underperform XGBoost (~55-60%) on ball prediction.
- Deep learning adds value primarily through sequence modeling and ensemble diversity.

### Ensemble Stacking
- Best approach: 3-7 diverse base models + logistic regression meta-learner.
- Diversity matters more than individual model accuracy.
- Candidate models: XGBoost, LightGBM, LSTM, Transformer (different feature subsets).

### Evaluation Best Practices
- **CLV (Closing Line Value)** is the gold standard metric for betting models.
- Minimum 200+ matches for statistically significant evaluation.
- Bootstrap confidence intervals on all metrics.
- Multiple baselines: 50-50, market odds, home team advantage.
- Minimum edge threshold of 3-5% before placing a bet (Kelly sizing).

---

## Feature Improvements Roadmap

### Team-Level Features (IMPLEMENTED — Phase 1)
Player-level ELO + aggregated team stats implemented in March 2026:
- `striker_elo` / `bowler_elo_rating` — individual player ELO ratings (ball-by-ball updates)
- `batting_team_elo` / `bowling_team_elo` — sum of 11 players' ELOs
- `elo_diff` — batting team ELO minus bowling team ELO
- `team_batting_avg` / `team_batting_sr` — aggregate of lineup batting stats
- `team_bowling_avg` / `team_bowling_econ` — aggregate of lineup bowling stats

See `docs/research/team_level_features.md` for full options analysis.

### ELO Rating System — Implemented + Future Improvements
**Current implementation**:
- Ball-by-ball ELO with linear outcome scaling: wicket=0.0, dot=0.4, 1=0.5, 2=0.6, 4=0.8, 6=1.0
- Context-aware K-factor: premium matches (IPL/WC/full-member T20I) K=4.0, standard leagues K=2.0, associate bilaterals K=1.0
- Gender-separated: women's cricket excluded from ELO pool (no cross-gender bridge players)
- Follows players across leagues (implicit cross-league calibration)
- Team strength = sum of 11 player ELOs
- ELO range after fixes: batting ±70 points, bowling ±45 points per player

**Bugs fixed**:
- Outcome scaling: dot balls were mapping to 0.0 (same as wickets), causing systematic batting ELO deflation. Fixed to 0.4 (mild negative, not catastrophic).
- Flat K=1.0: all matches weighted equally regardless of quality. Fixed with 3-tier K-factor.
- Gender contamination: women's associate cricket bowlers dominated top bowling ELOs. Fixed by filtering women's matches from ELO computation.
- Feature list: team_strength features were in parquet but not in xgboost_v2.py's hardcoded feature list. Fixed.

**Potential improvements** (ordered by expected impact):
- **Phase-specific outcome scaling**: A dot in death overs is worse than a dot in the powerplay. Use game-phase-aware baseline (e.g., expected runs by over number) instead of flat 0.4 for dots. This would make ELO updates more contextually accurate.
- **Phase-specific K-factors**: Higher K in death overs (K=6-8) where outcomes are more decisive, lower K in middle overs (K=2-3) where outcomes are more routine. Stacks with match-importance K.
- **ELO decay for inactive players**: Rating deviation grows when a player hasn't played recently (Glicko-2 style). Inactive players regress toward global mean. Prevents stale ratings for retired/rested players.
- **Separate pace/spin bowling ELO**: A bowler may be elite with pace but mediocre with spin variations. Track separate ELOs by bowling type. Would need changes to feature_registry and sim_v1_2.py.
- **Glicko-2 adaptation**: Track uncertainty (sigma) alongside rating (mu). Higher uncertainty → larger updates. Naturally handles new players and returning players.
- **Wicket type weighting**: Bowled/LBW is more "bowler's doing" than a run-out or stumping. Could weight wicket types differently in the ELO update.
- **Bridge-player league normalization** (for aggregated stats): Compute explicit league difficulty factors from players who compete in multiple leagues. Normalize raw stats (batting avg, SR) before aggregating into team strength features. Would improve `team_batting_avg` / `team_bowling_econ` features but not ELO itself.
- **Time decay weighting**: Recent balls weighted more heavily in ELO calculation, with exponential decay. More recent form is more predictive.

### Phase-Aware Bowler Selection
- Current simulation selects bowlers randomly.
- Real teams use pace in powerplay/death, spin in middle overs.
- Implement phase-based bowling probability distributions.

### Second-Innings Aggression Adjustment
- Chase targets should affect batting aggression.
- Required run rate should influence outcome probabilities.
- Currently chase features exist but may not be sufficient.

---

## Calibration System (IMPLEMENTED & TESTED — March 2026)

### What Was Built
Full calibration pipeline with two independent techniques:
- **Ball-level calibration**: Per-class isotonic regression on XGBoost's 6-class outputs (dot, 1, 2, 4, 6, wicket). Fitted on 226K validation balls. Applied inside `predict_next_ball()` before simulation sampling.
- **Match-level calibration**: Platt scaling (2-param logistic) with LOOCV on match win probabilities after Monte Carlo simulation. Applied before [5%, 95%] probability clipping.

### Implementation
| File | What |
|------|------|
| `scripts/calibration.py` | PlattCalibrator, IsotonicCalibrator, BallLevelCalibrator, BallLevelCalibrationDiagnostics, ECE/reliability functions |
| `scripts/sim_v1_2.py` | `XGBoostModelV2` accepts optional `ball_calibrator` param |
| `scripts/sim_eval/match_evaluator.py` | `evaluate_all_with_calibration()` method with LOOCV two-pass approach |
| `scripts/sim_eval/run_sim_eval.py` | CLI flags: `--calibrate`, `--calibration-method`, `--ball-calibrate`, `--ball-diagnostics` |
| `scripts/run_experiment.py` | Calibration flags passthrough from YAML configs |
| `experiments/configs/xgb_v4_*_calibration.yaml` | 4 experiment configs for A/B testing |

All calibration is **off by default**. Enable via CLI flags only.

### Ball-Level Diagnostics (Pre-Calibration)
On 226,326 validation balls:
| Class | ECE | Quality |
|-------|-----|---------|
| dot | 0.089 | poor |
| one | 0.133 | poor |
| two | 0.054 | fair |
| four | 0.062 | poor |
| six | 0.022 | good |
| wicket | 0.063 | poor |
| **Overall (weighted)** | **0.097** | **poor** |

### Experiment Results (4-way A/B test, 100 sims, 44 matches)
| Experiment | Log Loss | Brier | Flat ROI | Frac Kelly ROI | Win Rate |
|------------|----------|-------|----------|----------------|----------|
| A: No calibration (baseline) | 0.710 | 0.255 | -44.8% | -5.5% | 24.4% |
| B: Ball-level only | 0.791 | 0.294 | -47.4% | -6.9% | 24.4% |
| C: Match-level Platt | **0.667** | **0.238** | -64.9% | -5.4% | 12.2% |
| D: Both | 0.676 | 0.242 | -60.3% | -5.9% | 14.6% |

Calibration comparison (C & D only):
| | ECE before | ECE after | Log Loss before→after | Brier before→after |
|--|-----------|-----------|----------------------|-------------------|
| C: Match-only | 0.186 | 0.182 | 0.710 → 0.718 | 0.255 → 0.262 |
| D: Both | 0.228 | 0.246 | 0.791 → 0.732 | 0.294 → 0.268 |

### Conclusions
1. **Ball-level calibration hurts** — isotonic correction at ball level distorted probability distributions, worsening all metrics (log loss +11%, Brier +15%).
2. **Match-level Platt improves calibration metrics** (log loss -6%, Brier -7%) but **destroys betting performance** (flat ROI -44.8% → -64.9%, win rate halved). Platt compresses predictions toward 50%, reducing edge on every bet.
3. **Calibration cannot create signal** — it can only fix the mapping from predicted→actual probabilities. When the model predicts Ireland > India (because ELO ratings are wrong), calibration can't fix that — it just makes the wrong prediction closer to 50%.
4. **Root cause confirmed**: The bottleneck is discriminative power (resolution), not calibration. ELO features need refinement before calibration adds value.

### Decision: Calibration Disabled by Default
All calibration code is kept but disabled. Enable later when the model has better resolution:
```bash
# Ball diagnostics (read-only)
uv run python scripts/sim_eval/run_sim_eval.py --ball-diagnostics ...
# Ball-level calibration
uv run python scripts/sim_eval/run_sim_eval.py --ball-calibrate ...
# Match-level Platt calibration
uv run python scripts/sim_eval/run_sim_eval.py --calibrate --calibration-method platt ...
# Both
uv run python scripts/sim_eval/run_sim_eval.py --ball-calibrate --calibrate ...
```

Experiments: `xgb_v4_no_calibration_20260321_215322`, `xgb_v4_ball_calibration_20260322_004600`, `xgb_v4_match_calibration_20260322_034532`, `xgb_v4_both_calibration_20260322_063841`

---

## Evaluation Improvements

1. **Expand test set**: 200+ matches minimum (currently 44).
2. **Add baselines**: 50-50 random, market odds passthrough, home team always wins.
3. **Bootstrap confidence intervals**: 1000 resamples on all metrics.
4. **CLV tracking**: Compare model predictions to closing line (not opening odds).
5. **Minimum edge threshold**: Only "bet" when model edge > 3-5%.

---

## What NOT To Do

- Don't chase ball-level accuracy beyond ~60% — individual balls are inherently noisy.
- Don't optimize for the 44-match test set — too small, leads to overfitting.
- Don't increase simulations beyond 1000 — diminishing returns, 1000 is sufficient for stable estimates.
- Don't add features without validating that existing features are working correctly (fix bugs first).
- Don't trust any evaluation results that show >5% ROI on a small sample without extensive validation.
