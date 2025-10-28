# Design Decisions

Key architectural decisions and their rationale for the CricML Match Prediction system.

---

## 1. Temporal Data Integrity (CRITICAL)

### Problem
Player statistics in training vs simulation must match temporal reality. If we use future data during simulation, we create data leakage and overestimate model performance.

### Solution
Date-indexed player stats cache with snapshots taken BEFORE each match is processed.

### Implementation

**During Training (parsing_v2.py)**:
```python
for match in sorted_matches:
    match_date = extract_date(match)

    # CRITICAL: Take snapshot BEFORE processing match
    if match_date not in stats_snapshots:
        stats_snapshots[match_date] = deep_copy(player_stats_tracker)

    # Now process match and update tracker
    for ball in match:
        extract_features(ball, player_stats_tracker)  # Use BEFORE state
        player_stats_tracker.update(ball)             # Update AFTER
```

**During Simulation (stats_provider.py)**:
```python
def get_batting_stats(player_id, target_date):
    # Binary search for most recent snapshot ≤ target_date
    snapshot_date = bisect_right(self.dates, target_date) - 1
    snapshot = load_snapshot(snapshot_date)

    # Returns player stats as of snapshot_date
    # Never includes future performance
    return calculate_stats(snapshot[player_id])
```

### Why This Matters

**BAD (Data Leakage)**:
```
Simulating match on 2024-06-15
Uses player's complete 2024 stats (includes games after June 15)
→ Model sees future performance
→ Unrealistic accuracy
```

**GOOD (Temporal Integrity)**:
```
Simulating match on 2024-06-15
Uses player's stats as of 2024-06-14 (only historical data)
→ Model sees only past performance
→ Realistic accuracy
```

### Validation
- ✅ Player counts increase monotonically over time (105 early → 7,223 late)
- ✅ Stats match training data exactly when dates align
- ✅ No future data accessible during simulation

**Code**: `parsing_v2.py:481-486`, `stats_provider.py:57-78`

---

## 2. Feature Engineering Order

### Problem
Features must reflect state BEFORE ball is bowled, not after.

### Decision
Extract features first, then update stats.

### Rationale
Model should only know what's observable before making a prediction. Including outcome information in features creates data leakage.

### Implementation

```python
# 1. Extract features (uses tracker state BEFORE ball)
batting_avg = tracker.get_batting_avg(batter_id)
bowling_econ = tracker.get_bowling_econ(bowler_id)
h2h_sr = tracker.get_h2h_sr(batter_id, bowler_id)

features = {
    'batsman_avg': batting_avg,
    'bowler_econ': bowling_econ,
    'h2h_sr': h2h_sr,
    # ... 26 more features
}

# 2. Record feature vector + outcome
ball_record = {**features, 'ball_outcome': runs}

# 3. Update tracker (now state reflects AFTER ball)
tracker.update_batting(batter_id, runs, is_out)
tracker.update_bowling(bowler_id, runs, is_wicket)
tracker.update_h2h(batter_id, bowler_id, runs, is_out)
```

### Example

**Ball at over 5.2**:
- Feature `last_10_balls_runs`: Sum of balls 5.1, 5.0, 4.5, 4.4, ..., 4.2 (last 10)
- Does NOT include current ball (5.2) - we haven't observed outcome yet
- After prediction, update history to include 5.2 for next ball

**Code**: `parsing_v2.py:292-333`

---

## 3. Chunked Stats Cache with Lazy Loading

### Problem
Full player stats cache is 7.6GB. Loading into memory at startup is slow and wasteful.

### Solution
Split cache into 69 chunks (~110MB each), load on-demand with LRU eviction.

### Architecture

```
Metadata (~10KB):
  - List of 69 chunk files
  - Date index for each chunk
  - Loaded at startup

Chunks (69 × ~110MB):
  - Loaded on-demand
  - LRU cache (keep 5 most recent)
  - ~550MB max memory usage
```

### Implementation

```python
class StatsProvider:
    def __init__(self):
        # Load only metadata
        self.metadata = pickle.load('metadata.pkl')  # ~10KB

        # Build date → chunk_idx mapping
        self.date_to_chunk_idx = {}
        for chunk_idx, chunk_info in enumerate(self.metadata['chunks']):
            for date in chunk_info['dates']:
                self.date_to_chunk_idx[date] = chunk_idx

        # LRU cache for chunks
        self.chunk_cache = OrderedDict()  # Max 5 chunks

    def get_stats(self, player_id, date):
        # Binary search for snapshot date
        snapshot_date = self._find_snapshot_date(date)

        # Find which chunk contains this date
        chunk_idx = self.date_to_chunk_idx[snapshot_date]

        # Load chunk (from cache or disk)
        if chunk_idx not in self.chunk_cache:
            self._load_chunk(chunk_idx)  # Lazy load

        # Return stats
        return self.chunk_cache[chunk_idx][snapshot_date][player_id]
```

### Performance

| Metric | Monolithic | Chunked |
|--------|------------|---------|
| Startup time | 30-45 sec | 1-2 sec |
| Memory usage | 7.6 GB | 300-550 MB |
| Query speed | <0.01ms | <0.01ms (after chunk load) |
| Cache hit rate | N/A | ~95%+ |

### Trade-offs

**Pros**:
- 14x memory reduction
- Fast startup (~2 sec vs ~40 sec)
- Scales to larger datasets

**Cons**:
- Slight I/O overhead on chunk miss (~0.5 sec)
- More complex implementation
- Chunk boundaries must be managed

### Why 69 Chunks?

```python
total_dates = 3442
save_interval = 50  # Snapshots per chunk
num_chunks = ceil(3442 / 50) = 69
```

Balancing chunk size:
- Too small: Excessive I/O, many files
- Too large: Long load times, less granular caching
- 50 dates (~110MB) provides good balance

**Code**: `parsing_v2.py:273-329`, `stats_provider.py:20-95`

---

## 4. Class Remapping for Rare Outcomes

### Problem
Some ball outcomes are extremely rare in the data:
- 3 runs: ~0.5% of balls
- 5 runs: ~0.1% of balls
- 7+ runs: ~0.01% of balls

Training on imbalanced classes leads to poor predictions for rare events.

### Solution
Normalize rare outcomes to nearest common class.

### Mapping

```python
# Raw data
0 runs → 0 (dot ball)
1 run → 1
2 runs → 2
3 runs → 2  # NORMALIZED (rare, similar to 2)
4 runs → 4
5 runs → 4  # NORMALIZED (rare, similar to 4)
6 runs → 6
7+ runs → 6 # NORMALIZED (very rare, all-run boundaries)
wicket → 7

# Model classes (after remapping)
0: dot (0 runs)
1: single (1 run)
2: two (2 runs)
3: four (4 runs boundary)
4: six (6 runs boundary)
5: wicket
```

### Rationale

**3-run balls**:
- Usually overthrows or misfields
- Semantically similar to 2-run balls (running between wickets)
- Combining improves model learning for "running" outcomes

**5-run balls**:
- Usually overthrow boundaries
- Semantically similar to 4-run balls (boundary scored)
- Combining improves model learning for "boundary" outcomes

**7+ run balls**:
- Extremely rare (no-ball + 6, or multiple overthrows)
- Similar to 6-run balls (maximum typical outcome)
- Prevents overfitting to ultra-rare cases

### Impact

| Class | Before Normalization | After Normalization |
|-------|---------------------|---------------------|
| Dot | 32% | 32% |
| 1 | 40% | 40% |
| 2 | 8% + 0.5% (3s) = 8.5% | 8.5% |
| 4 | 10% + 0.1% (5s) = 10.1% | 10.1% |
| 6 | 4% + 0.01% (7+) = 4.01% | 4.01% |
| Wicket | 5% | 5% |

Reduces 8 classes → 6 classes while preserving semantic meaning.

**Code**: `parsing_v2.py:147-163`, `xgboost_v2.py:150-165`

---

## 5. Ball-Level Modeling vs Direct Match Prediction

### Problem
Predicting match outcomes directly has limited training data (thousands of matches).

### Solution
Predict individual balls (millions of examples), then simulate matches via Monte Carlo.

### Comparison

| Approach | Training Examples | Uncertainty | Granularity |
|----------|------------------|-------------|-------------|
| Direct match prediction | ~15,000 matches | Limited | Coarse |
| Ball-level + simulation | ~4,000,000 balls | Rich (via simulation variance) | Fine-grained |

### Why Ball-Level is Better

1. **More Data**: 4M balls vs 15K matches = 266x more training examples
2. **Better Features**: Momentum, pressure, phase-specific patterns
3. **Natural Uncertainty**: Simulation variance captures game uncertainty
4. **Interpretability**: Can analyze ball-by-ball predictions
5. **Flexibility**: Same model works for any match format (T20, ODI, Test)

### Monte Carlo Simulation

```python
# Run 1000 simulations
results = []
for i in range(1000):
    # Predict each ball probabilistically
    for ball in range(120):
        outcome_probs = model.predict(state)
        outcome = sample(outcome_probs)
        state.update(outcome)

    results.append(state.winner)

# Aggregate
win_prob = count(results == "India") / 1000
```

### Trade-offs

**Pros**:
- 266x more training data
- Rich uncertainty quantification
- Granular feature engineering

**Cons**:
- Slower inference (1000 simulations vs 1 prediction)
- More complex pipeline
- Simulation error can compound

**Decision**: Pros vastly outweigh cons for this application.

---

## 6. Parallel Simulation with Multiprocessing

### Problem
Simulating 1000 matches sequentially takes 10-30 seconds per match.

### Solution
Parallelize simulations across CPU cores using multiprocessing.

### Implementation

```python
from multiprocessing import Pool

def simulate_single(args):
    state, seed = args
    # Set random seed for reproducibility
    random.seed(seed)
    # Run simulation
    return engine.simulate_match(state)

# Parallel execution
with Pool(processes=4) as pool:
    args = [(state.copy(), seed + i) for i in range(1000)]
    results = pool.map(simulate_single, args)
```

### Performance

| Configuration | Time per 1000 sims | Speedup |
|---------------|-------------------|---------|
| Serial (1 core) | 30-40 sec | 1x |
| Parallel (4 cores) | 8-12 sec | 3-4x |
| Parallel (8 cores) | 5-8 sec | 5-6x |

### Requirements for Parallelization

1. **Independent simulations**: No shared state
2. **Pickle-serializable**: Model and state must serialize
3. **Deterministic**: Use unique seeds for reproducibility

### Trade-offs

**Pros**:
- 3-6x speedup on multi-core machines
- Scales with CPU cores
- No code changes to simulation logic

**Cons**:
- Pickling overhead (~100ms startup)
- Memory duplication (each worker has model copy)
- Not available on all systems

**Code**: `sim_v1_2.py:986-1010`

---

## 7. Betting Edge: Margin-Free Market Probabilities

### Problem
Bookmaker odds include 5-10% margin (overround). Direct comparison to model probabilities is unfair.

### Solution
Remove bookmaker margin by normalizing implied probabilities to sum to 1.0.

### Example

**Raw Odds**:
```python
odds = {'India': 2.10, 'Australia': 1.75}
```

**Implied Probabilities (with margin)**:
```python
india_prob = 1 / 2.10 = 0.476 (47.6%)
australia_prob = 1 / 1.75 = 0.571 (57.1%)
total = 0.476 + 0.571 = 1.047  # 4.7% overround (bookmaker margin)
```

**Normalized (margin-free)**:
```python
fair_india = 0.476 / 1.047 = 0.455 (45.5%)
fair_australia = 0.571 / 1.047 = 0.545 (54.5%)
total = 0.455 + 0.545 = 1.000  # No margin
```

### Edge Calculation

```python
# Model prediction
model_prob = {'India': 0.65, 'Australia': 0.35}

# Edge (vs margin-free market)
edge = {
    'India': 0.65 - 0.455 = +0.195 (+19.5%)
    'Australia': 0.35 - 0.545 = -0.195 (-19.5%)
}
```

### Interpretation

- **Positive edge**: Model favors team more than market
- **Negative edge**: Model favors team less than market
- **Magnitude**: Strength of disagreement

### Why This Matters

Without margin removal:
```python
# Model vs raw market
edge = 0.65 - 0.476 = +0.174  # Underestimates disagreement
```

With margin removal:
```python
# Model vs fair market
edge = 0.65 - 0.455 = +0.195  # True disagreement
```

**Code**: `loaders.py:181-216`

---

## 8. Signed Edge for Prediction Quality

### Problem
Absolute edge shows disagreement magnitude but not prediction quality.

### Solution
Track "signed edge" that's positive for correct predictions, negative for incorrect.

### Calculation

```python
# Compute edge
edge = model_prob - market_prob

# Sign based on correctness
if predicted_winner == actual_winner:
    signed_edge = +abs(edge)  # Correct + confident
else:
    signed_edge = -abs(edge)  # Wrong + confident
```

### Example

**Match 1**:
```
Model: India 70%, Market: India 55%
Edge: +15%
Actual: India wins
Signed edge: +15% (correct + high confidence)
```

**Match 2**:
```
Model: India 70%, Market: India 55%
Edge: +15%
Actual: Australia wins
Signed edge: -15% (wrong + high confidence)
```

### Interpretation

- **High positive signed edge**: Model correctly identified value
- **High negative signed edge**: Model overconfident and wrong
- **Near zero**: Model aligns with market or low confidence

### Use Cases

1. **Model evaluation**: Average signed edge shows prediction quality
2. **Feature importance**: Which features drive high-quality predictions?
3. **Betting strategy**: Only bet on high positive signed edge

**Code**: `match_evaluator.py:463-502`

---

## 9. Model Output: Adding Extras Probability

### Problem
XGBoost trained on 6 classes (dot, 1, 2, 4, 6, wicket). Doesn't predict extras (wides, no-balls).

### Solution
Add fixed 1% probability for wides and no-balls, then normalize.

### Implementation

```python
# Model outputs 6 classes
model_probs = {
    'dot': 0.32,
    'one': 0.39,
    'two': 0.08,
    'four': 0.10,
    'six': 0.04,
    'wicket': 0.05
}  # Sum = 0.98

# Add extras (not trained)
model_probs['wide'] = 0.01
model_probs['no_ball'] = 0.01

# Normalize to sum = 1.0
total = sum(model_probs.values())  # 1.00
outcome_probs = {k: v/total for k, v in model_probs.items()}
```

### Rationale

**Why not train on extras?**
- Extras are rare (~5% of balls)
- High variance (some bowlers bowl many, others few)
- Different types (wide, no-ball, bye, leg-bye) with different causes
- Model struggles to predict reliably

**Why add fixed probabilities?**
- Simulations need valid extras (T20 matches have ~5-10 extras)
- Fixed 2% (1% wide + 1% no-ball) approximates real distribution
- Normalized to ensure probabilities sum to 1.0

### Impact

Average T20 match:
- 240 balls (2 innings × 120 balls)
- 240 × 0.02 = 4.8 expected extras
- Close to real average of 5-10 extras

**Code**: `sim_v1_2.py:636-644`

---

## 10. Strike Rotation: No Rotation on Wickets

### Problem
Cricket rule: When batsman gets out, incoming batsman takes striker position (no rotation).

### Implementation

```python
# Handle wicket
if outcome == Outcome.WICKET:
    self.wickets[team_idx] += 1
    self.batsmen_out[team_idx].append(self.striker_idx)
    self.striker_idx = self.get_next_batsman_idx()
    # NO strike rotation here

# Rotate strike for odd runs only
if runs % 2 == 1:
    self.striker_idx, self.non_striker_idx = \
        self.non_striker_idx, self.striker_idx

# Rotate strike at end of over
if balls % 6 == 0:
    self.striker_idx, self.non_striker_idx = \
        self.non_striker_idx, self.striker_idx
```

### Why This Matters

**Wrong Implementation**:
```
Ball 1: Batsman A gets out
→ Rotate strike (wrong!)
Ball 2: Batsman C faces (should be Batsman B)
```

**Correct Implementation**:
```
Ball 1: Batsman A gets out
→ Batsman B comes in at striker position
→ No rotation
Ball 2: Batsman B faces (correct!)
```

**Code**: `sim_v1_2.py:250-259`

---

## Summary of Key Decisions

| Decision | Rationale | Impact |
|----------|-----------|--------|
| Temporal integrity | Prevent data leakage | Realistic evaluation |
| Feature order | Only use pre-ball state | No information leakage |
| Chunked cache | Memory efficiency | 14x reduction (7.6GB → 550MB) |
| Class remapping | Handle rare outcomes | Better learning |
| Ball-level modeling | More training data | 266x more examples |
| Parallel simulation | Faster inference | 3-6x speedup |
| Margin-free odds | Fair comparison | Accurate edge calculation |
| Signed edge | Prediction quality | Better evaluation |
| Fixed extras | Practical simulation | Realistic match flow |
| Strike rotation | Cricket rules | Correct simulation |

---

**For implementation details, see [OPERATIONS.md](./OPERATIONS.md).**

**For data specifications, see [DATA_FORMATS.md](./DATA_FORMATS.md).**
