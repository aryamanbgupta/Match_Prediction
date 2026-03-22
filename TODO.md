# TODO

## Completed
- ✅ Make the outcome categories: squish/ round down to 0,1,2,4,6,W
- ✅ LSTM model architecture (scripts/lstm_v1.py, LSTMModelV1 in sim_v1_2.py)
- ✅ LSTM hyperparameter improvements (FocalLoss, WarmupCosine scheduler, LayerNorm)
- ✅ LSTM feature extraction alignment (59 continuous features)
- ✅ Root cause analysis: Both XGBoost and LSTM have weak match predictions
- ✅ Fix XGBoost class_to_outcome bug (sim_v1_2.py) — was 8-class, should be 6-class
- ✅ Feature registry (scripts/feature_registry.py) — central feature definitions
- ✅ Experiment tracking infrastructure (scripts/experiment_tracker.py)
- ✅ Experiment config schema (experiments/configs/*.yaml)
- ✅ Pipeline runner (scripts/run_experiment.py)
- ✅ Experiment comparison tool (scripts/compare_experiments.py)

## Root Cause Analysis (Dec 2024)
**Why both models predict ~50% for all matches:**
1. **Weak correlation**: Player stats have r=0.06-0.11 with ball outcomes (explains <1.5% of variance)
2. **Missing team-level features**: Model only sees current batter/bowler, not aggregate team strength
3. **Ball-level noise**: Even Kohli hits dots 30% of balls; individual outcomes are random
4. **Signal dilution**: Small differences per ball don't compound meaningfully to match outcomes

## Critical (P0)
- [x] **Re-evaluate all models after bug fix** — Done (March 2025). All 4 models lose money:
  - XGBoost v3: -43.9% flat ROI, 26.8% win rate (best of 4)
  - LSTM v1: -63.4% flat ROI, 12.2% win rate
  - Transformer v1: -63.4% flat ROI, 12.2% win rate
  - MLP v1: -63.8% flat ROI, 12.2% win rate
  - Root cause: no team-level signal — models can't distinguish strong vs weak teams
- [x] **Fix evaluation JSON output** — now saves full betting metrics (P&L, ROI, win rate, Kelly, actual_winner per match) with timestamped filenames
- [ ] **Expand test set to 200+ matches** — 44 matches is too small for statistical significance

## High Priority (P1)
- [ ] **Empirical outcome distributions as features** — Replace lossy summary stats (avg/SR) with full 6-class outcome distributions P(0,1,2,4,6,W) for each context. This is multi-class target encoding — directly aligned with the prediction target. Current summary stats have r=0.06-0.11 with ball outcomes; direct historical rates should correlate much more strongly. Implementation: expand `PlayerStatsTracker` to track outcome counts per class instead of just `{runs, balls, dismissals}`. Hierarchy of distributions (each adds 6 features):
  - **Batter overall**: P(0,1,2,4,6,W | this batter) — captures player shape (power hitter vs accumulator), not just mean
  - **Bowler overall**: P(0,1,2,4,6,W | this bowler) — same idea, from bowler's perspective
  - **Batter vs pace/spin**: P(0,1,2,4,6,W | batter, bowler type) — type-specific distributions (min 30 balls, fallback to overall)
  - **Bowler vs LHB/RHB**: P(0,1,2,4,6,W | bowler, batter hand) — handedness-specific (min 30 balls, fallback to overall)
  - **Venue**: P(0,1,2,4,6,W | venue) — venue-specific outcome rates
  - **Phase (global)**: P(0,1,2,4,6,W | powerplay/middle/death) — phase-specific priors with massive sample sizes
  - ~36-48 new features total. May allow dropping `batter_encoded`/`bowler_encoded` (XGBoost can't learn from label-encoded IDs anyway).
  - Temporal integrity already handled by stats cache. Not target leakage — uses only pre-match historical data.
- [ ] **Expand test set to 200+ matches** — 44 matches is too small for statistical significance
- [ ] **Add evaluation baselines** — 50-50 random, market odds passthrough, home team always wins
- [x] **Add team-level features** — ELO + aggregated team stats (9 features). 19% log loss improvement.
- [x] **Add calibration layer** — Implemented ball-level (isotonic) + match-level (Platt LOOCV). Tested scientifically: calibration improves probability metrics but hurts betting ROI. Disabled by default; enable via `--calibrate` / `--ball-calibrate` flags when model resolution improves.
- [x] **ELO rating system** — Player-level ball-by-ball ELO with context-aware K-factor. See IMPROVEMENTS.md.
- [x] **Feature importance analysis** — `scripts/analyze_features.py` (gain/weight/cover, correlation, redundancy, group-level)
- [x] **Tier 1 features: venue profile + match context** — 11 new features across 2 groups:
  - `venue_profile`: boundary_pct, dot_pct, wicket_rate, powerplay_avg, death_avg, first_innings_avg, chase_win_pct
  - `match_context`: chose_to_bat, match_importance, is_international, competition_tier

## Medium Priority (P2) — Tier 2 Features
- [ ] **Batting order position context**:
  - `batting_position` (1-11), `remaining_batting_quality` (avg ELO of batters to come), `top_order_wickets`
- [ ] **Bowler workload / spell context**:
  - `bowler_spell_balls`, `bowler_overs_left` (max 4 per T20), `new_bowler` (first ball of spell)
- [ ] **Enhanced momentum features**:
  - `scoring_acceleration` (last 10 vs last 30 run rate), `wicket_cluster` (wickets in last 10 balls), `boundaries_in_last_over`
- [ ] **Home advantage (explicit)**:
  - `is_home_team` (3+ matches at venue in last 2 years), `team_venue_win_pct`
- [ ] Phase-aware bowler selection (pace in powerplay/death, spin in middle overs)
- [ ] Second-innings aggression adjustment based on required run rate
- [ ] Unknown player encoding (bottom 5-10%) for new/unseen players
- [ ] Bootstrap confidence intervals on evaluation metrics
- [ ] CLV (Closing Line Value) tracking
- [ ] Minimum edge threshold (3-5%) before placing a bet

## Low Priority (P3) — Tier 3 Features (External Data)
- [ ] **Weather / conditions data** (temperature, humidity, dew, wind via weather API)
- [ ] **Ground dimensions** (boundary distances — affects 4/6 rates)
- [ ] Ensemble stacking: 3-7 diverse models + logistic regression meta-learner
- [ ] Add time decay to features
- [ ] Consider regression model (predict E[runs]) instead of classification
- [ ] Match-level model: predict P(team1 wins) directly from lineups

## Research Notes
See [IMPROVEMENTS.md](IMPROVEMENTS.md) for detailed research findings:
- Calibration-optimized models generate 69.86% higher returns than accuracy-optimized (Walsh & Joshi 2024)
- Realistic market-beating edge is 1-3% ROI
- GBMs still outperform deep learning on tabular data
- Don't chase ball accuracy >60%, don't optimize for small test set
