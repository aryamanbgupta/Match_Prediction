# Model Improvement Roadmap

**Generated**: January 2026
**Based on**: T20 World Cup 2024 Evaluation (44 matches)
**Current Performance**: Log Loss 1.64, ROI -48.1% (Flat), -5.5% (Fractional Kelly)

---

## Executive Summary

The v3 model shows improved directional predictions after the lineup extraction bug fix, but suffers from:
1. **Probability calibration issues** - Overconfident at extremes
2. **Missing opposition quality context** - Associate players appear stronger than internationals
3. **Technical bugs** - Tie probabilities not normalized

Priority should be given to quick fixes (#1, #2) followed by fundamental improvements (#3, #4).

---

## Issue 1: Tie Probability Not Normalized

### Severity: High | Effort: Low (5 min)

### Problem
Simulated probabilities don't sum to 1.0 because ties are excluded but not normalized:

```
Scotland vs England:  85% + 10% = 95%  (5% missing = ties)
Australia vs Oman:    55% + 35% = 90%  (10% missing = ties)
Sri Lanka vs Bangladesh: 25% + 70% = 95%
```

### Root Cause
In `scripts/sim_eval/match_evaluator.py`, line 190-192:
```python
simulated_win_prob = {
    team1: aggregated['win_probability'][team1],
    team2: aggregated['win_probability'][team2]
}
```
This extracts raw probabilities without normalizing for ties.

### Fix
```python
# Normalize probabilities to exclude ties
team1_raw = aggregated['win_probability'][team1]
team2_raw = aggregated['win_probability'][team2]
total = team1_raw + team2_raw

if total > 0:
    simulated_win_prob = {
        team1: team1_raw / total,
        team2: team2_raw / total
    }
else:
    # Fallback for edge case
    simulated_win_prob = {team1: 0.5, team2: 0.5}
```

### Expected Impact
- Probabilities will sum to 1.0
- Metrics (log loss, Brier score) will be more accurate
- Minor improvement in calibration

---

## Issue 2: Extreme Predictions (0%/100%)

### Severity: High | Effort: Low (5 min)

### Problem
Model produces 0% or 100% probabilities when simulations are unanimous:

| Match | Prediction | Outcome | Log Loss |
|-------|------------|---------|----------|
| Nepal vs Netherlands | 100% Nepal | Netherlands won | **34.5** |
| Canada vs USA | 100% USA | USA won | 0.0 |
| Pakistan vs USA | 0% Pakistan | USA won (upset) | 0.0 |

A single wrong 100% prediction causes catastrophic log loss.

### Root Cause
With 100-1000 simulations, unanimous results are common for mismatched teams. No dampening is applied.

### Fix
Add probability clipping in `scripts/sim_v1_2.py` `ResultAggregator.aggregate()`:

```python
EPSILON = 0.02  # Never more confident than 98%

# After calculating raw probabilities
win_probability = {
    team1: max(EPSILON, min(1 - EPSILON, team1_wins / n_sims)),
    team2: max(EPSILON, min(1 - EPSILON, team2_wins / n_sims)),
    'tie': ties / n_sims
}
```

### Expected Impact
- Maximum log loss capped at ~3.9 instead of 34.5
- More realistic probability estimates
- Reduced variance in betting returns

---

## Issue 3: No Opposition Quality Adjustment

### Severity: Critical | Effort: Medium-High (4-8 hours)

### Problem
Player statistics don't account for quality of opposition. Associate nation players have inflated stats:

```
Uganda's Riazat Ali Shah:  avg=37.0, sr=127.7  (vs Nepal, PNG, Kenya)
WI's Nicholas Pooran:      avg=30.3, sr=150.1  (vs Australia, England, India)
```

The model sees Riazat as the "better" player, leading to predictions like:
- **West Indies 45% vs Uganda 55%** (actual: WI won easily)

### Root Cause
Training data treats all T20 cricket equally. A boundary against Kenya counts the same as one against Australia.

### Proposed Solutions

#### Option A: Team ELO/Rating Feature (Recommended)
Add ICC rankings or calculated ELO as model features:

```python
# In parsing_v2.py feature extraction
'batting_team_elo': get_team_elo(batting_team, match_date),
'bowling_team_elo': get_team_elo(bowling_team, match_date),
'elo_difference': batting_elo - bowling_elo
```

**Pros**: Simple to implement, captures team strength
**Cons**: Requires ELO calculation or external data source

#### Option B: Strength of Schedule Adjustment
Weight player stats by opponent quality:

```python
# Adjusted average = sum(runs * opponent_weight) / sum(balls * opponent_weight)
# Where opponent_weight = opponent_elo / average_elo
```

**Pros**: Directly addresses the issue
**Cons**: Requires reprocessing all historical stats, complex implementation

#### Option C: Competition Level Feature
Tag matches by competition tier:

```python
'competition_tier': 1 if team in TOP_12_TEAMS else 2 if team in ASSOCIATES else 3
'opponent_tier': ...
```

**Pros**: Simple heuristic
**Cons**: Doesn't capture within-tier differences

#### Option D: Filter Training Data
Only train on matches involving top 12 teams.

**Pros**: Simplest approach
**Cons**: Loses data, doesn't help when evaluating associate matches

### Recommended Approach
Start with **Option A** (Team ELO) as a new feature, then consider **Option B** for v4.

### Data Sources for ELO
- ICC T20I Rankings (official but lagging)
- Calculate from match results in training data
- Use existing ratings from ESPNcricinfo

---

## Issue 4: Probability Calibration (Overconfidence)

### Severity: High | Effort: Medium (1-2 hours)

### Problem
Model is systematically overconfident at high probabilities and underconfident at low probabilities:

| Predicted | Actual | Difference | Sample Size |
|-----------|--------|------------|-------------|
| 3.3% | 22.2% | +18.9% (underconfident) | n=9 |
| 12.5% | 62.5% | +50.0% (underconfident) | n=8 |
| 54.2% | 53.8% | -0.4% (well calibrated) | n=13 |
| 82.3% | 63.6% | -18.6% (overconfident) | n=11 |
| 93.0% | 60.0% | -33.0% (overconfident) | n=5 |

### Root Cause
Raw simulation win percentages don't account for:
1. Model uncertainty
2. Sample size of simulations
3. Systematic biases in ball-level predictions

### Fix: Post-hoc Calibration Layer

#### Step 1: Collect Calibration Data
Run simulations on validation set (held-out matches from training period):

```python
# Generate calibration dataset
calibration_data = []
for match in validation_matches:
    raw_prob = simulate_match(match)
    actual_winner = get_actual_winner(match)
    calibration_data.append((raw_prob, 1 if team1_won else 0))
```

#### Step 2: Train Calibration Model
```python
from sklearn.isotonic import IsotonicRegression

# Isotonic regression (non-parametric, preserves ordering)
calibrator = IsotonicRegression(out_of_bounds='clip')
calibrator.fit(raw_probs, actual_outcomes)

# Save calibrator
import joblib
joblib.dump(calibrator, 'models/xgb_v3/probability_calibrator.pkl')
```

#### Step 3: Apply During Evaluation
```python
# In match_evaluator.py
calibrator = joblib.load('models/xgb_v3/probability_calibrator.pkl')
calibrated_prob = calibrator.predict([raw_sim_prob])[0]
```

### Alternative: Platt Scaling
```python
from sklearn.linear_model import LogisticRegression

# Platt scaling (parametric, assumes sigmoid relationship)
platt_scaler = LogisticRegression()
platt_scaler.fit(raw_probs.reshape(-1, 1), actual_outcomes)
```

### Expected Impact
- Better alignment between predicted and actual probabilities
- Reduced log loss (target: < 0.7)
- More profitable betting signals

---

## Issue 5: Simulation Speed

### Severity: Low | Effort: Low (30 min)

### Problem
Average simulation time: **50.7 seconds per match** (1000 simulations)

This is slow for a production system and limits experimentation.

### Likely Causes
1. Python GIL limiting parallelism
2. Repeated model inference overhead
3. Stats provider lookups per ball

### Potential Fixes
1. **Batch model inference**: Predict multiple balls at once
2. **Vectorized simulation**: Use NumPy for ball outcomes
3. **Caching**: Pre-compute common feature combinations
4. **Reduce simulations**: Use 500 instead of 1000 for similar accuracy

### Expected Impact
- 5-10x speedup possible
- Enable more extensive hyperparameter tuning
- Faster iteration on improvements

---

## Issue 6: Unknown Player Handling

### Severity: Medium | Effort: Medium (2-3 hours)

### Problem
Players not in training data get default/unknown encodings, which may produce unreliable predictions.

### Current Behavior
```python
# In sim_v1_2.py
'batter_hand': 2,  # unknown
'bowler_arm': 2,   # unknown
'is_pace': 2,      # unknown
```

### Proposed Improvements
1. **Position-based defaults**: Use batting position to infer likely stats
2. **Team average**: Use team's average player stats for unknowns
3. **Recent debutant stats**: Use average stats of recent debutants
4. **Explicit unknown flag**: Add `is_unknown_player` feature

---

## Implementation Priority

### Phase 1: Quick Wins (Day 1)
- [ ] Fix tie probability normalization (Issue #1)
- [ ] Add probability clipping (Issue #2)
- [ ] Re-run evaluation to measure impact

### Phase 2: Calibration (Day 2-3)
- [ ] Generate calibration dataset from validation matches
- [ ] Train and save calibration model
- [ ] Integrate into evaluation pipeline
- [ ] Measure calibration improvement

### Phase 3: Team Quality (Week 1-2)
- [ ] Calculate/obtain team ELO ratings
- [ ] Add team ELO features to parsing_v2.py
- [ ] Retrain model with new features
- [ ] Evaluate improvement

### Phase 4: Optimization (Week 2-3)
- [ ] Profile simulation bottlenecks
- [ ] Implement batch inference
- [ ] Optimize stats provider lookups
- [ ] Benchmark improvements

---

## Success Metrics

| Metric | Current | Target (Phase 1) | Target (Phase 3) |
|--------|---------|------------------|------------------|
| Log Loss | 1.64 | < 1.2 | < 0.7 |
| Brier Score | 0.29 | < 0.25 | < 0.20 |
| Flat Staking ROI | -48.1% | > -20% | > 0% |
| Fractional Kelly ROI | -5.5% | > 0% | > 5% |
| Calibration Error | ~25% | < 15% | < 10% |

---

## Appendix: Evaluation Results Summary

### Betting Performance by Strategy
| Strategy | ROI | Win Rate | Sharpe |
|----------|-----|----------|--------|
| Flat Staking | -48.1% | 28.9% | -3.14 |
| Full Kelly | -21.9% | 24.2% | -2.44 |
| Fractional Kelly (25%) | -5.5% | 24.2% | -2.44 |

### Performance by Match Type
| Type | Win Rate | Flat ROI | Avg Edge |
|------|----------|----------|----------|
| Favorites (odds < 2.0) | 64.3% | +0.3% | 18.7% |
| Underdogs (odds >= 2.0) | 7.4% | -73.3% | 42.1% |

### Notable Correct Predictions
1. Papua New Guinea vs Uganda: Model 60% Uganda, Market 22.5% → Uganda won (+37.5% edge)
2. Sri Lanka vs Bangladesh: Model 70% Bangladesh, Market 37.8% → Bangladesh won (+32.2% edge)
3. India vs Pakistan: Model 95% India, Market 68.3% → India won (+26.7% edge)

### Notable Failures
1. Nepal vs Netherlands: Model 100% Nepal → Netherlands won (Log Loss: 34.5)
2. West Indies vs Uganda: Model 45% WI, 55% Uganda → WI won (inverted prediction)
3. Australia vs England: Model 15% Australia → Australia won

---

*Last updated: January 2026*
