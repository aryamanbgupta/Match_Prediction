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

### Team-Level Features (HIGH PRIORITY)
- `team_batting_avg` — aggregate of all 11 batters' averages
- `team_batting_sr` — aggregate team strike rate
- `opp_bowling_avg` — opposition bowlers' quality
- `opp_bowling_econ` — opposition economy rate
- `relative_strength` — `team_bat - opp_bowl` (normalized)
- `batting_depth_index` — quality of middle/lower order batters

### ELO Rating System
- Team-level ELO updated after each match.
- Can also do player-level ELO (batter rating, bowler rating).
- Provides a single "team strength" signal that accumulates over time.

### Phase-Aware Bowler Selection
- Current simulation selects bowlers randomly.
- Real teams use pace in powerplay/death, spin in middle overs.
- Implement phase-based bowling probability distributions.

### Second-Innings Aggression Adjustment
- Chase targets should affect batting aggression.
- Required run rate should influence outcome probabilities.
- Currently chase features exist but may not be sufficient.

---

## Calibration Plan

1. **Post-hoc calibration**: Apply isotonic regression to model output probabilities.
2. **Match-level calibration layer**: Train a small model on simulated match outcomes vs actual outcomes.
3. **Target**: ECE < 0.015 on held-out matches.
4. **Validation**: Reliability diagrams (predicted prob vs observed frequency).

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
