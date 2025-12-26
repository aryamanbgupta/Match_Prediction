# TODO

## Completed
- ✅ Make the outcome categories: squish/ round down to 0,1,2,4,6,W
- ✅ LSTM model architecture (scripts/lstm_v1.py, LSTMModelV1 in sim_v1_2.py)
- ✅ LSTM hyperparameter improvements (FocalLoss, WarmupCosine scheduler, LayerNorm)
- ✅ LSTM feature extraction alignment (59 continuous features)
- ✅ Root cause analysis: Both XGBoost and LSTM have weak match predictions

## Root Cause Analysis (Dec 2024)
**Why both models predict ~50% for all matches:**
1. **Weak correlation**: Player stats have r=0.06-0.11 with ball outcomes (explains <1.5% of variance)
2. **Missing team-level features**: Model only sees current batter/bowler, not aggregate team strength
3. **Ball-level noise**: Even Kohli hits dots 30% of balls; individual outcomes are random
4. **Signal dilution**: Small differences per ball don't compound meaningfully to match outcomes

## In Progress
- **Add team-level features** (HIGH PRIORITY):
  - `team_batting_avg` - aggregate of all 11 batters
  - `team_batting_sr` - aggregate strike rate
  - `opp_bowling_avg` - opposition bowlers' quality
  - `opp_bowling_econ` - opposition economy
  - `relative_strength` - team_bat - opp_bowl (normalized)
  - `batting_depth` - quality of middle/lower order

## Pending (Prioritized)
1. **Add team-level features to training data** - Requires re-running parsing_v2.py
2. **Re-train XGBoost with team features** - Should significantly improve match predictions
3. Consider regression model (predict E[runs]) instead of classification
4. Make unknown player encoding (bottom 5-10%) for new players
5. Betting sim: fix odds json generation (currently ignores 30% of matches)
6. Try Transformer architecture
7. Add time decay to features
8. Add more weight to top team matches

## Alternative Approaches to Consider
- **Match-level model**: Predict P(team1 wins) directly from lineups (skip ball simulation)
- **Two-stage model**: Ball prediction + match-level calibration
- **ELO/rating system**: Team strength ratings updated after each match
