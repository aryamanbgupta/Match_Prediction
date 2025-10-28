# Operations Guide

Complete guide for running all pipelines and common operations in the CricML Match Prediction system.

---

## Quick Start

### Training Pipeline (Full)
```bash
# Step 1: Feature engineering (~10-15 min)
python scripts/parsing_v2.py

# Step 2: Model training (~30-60 min)
python scripts/xgboost_v2.py

# Step 3: Evaluation (~5-10 min for 45 matches)
python scripts/sim_eval/run_sim_eval.py \
    --test-dir data/betting_test \
    --odds betting_odds_v3.json \
    --n-sims 1000
```

### Quick Simulation (Standalone)
```python
from scripts.sim_v1_2 import *
from scripts.stats_provider import StatsProvider

# Load model
stats_provider = StatsProvider('models')
model = XGBoostModelV2(
    'models/xgb/xgboost_model_v2.pkl',
    'models/xgb/batter_encoder_v2.pkl',
    'models/xgb/bowler_encoder_v2.pkl',
    'models/xgb/feature_columns_v2.txt',
    stats_provider=stats_provider
)
engine = SimulationEngine(model, T20Rules())

# Run simulation (see examples below for full code)
```

---

## Training Pipeline

### 1. Feature Engineering (`parsing_v2.py`)

**Purpose**: Transform raw JSON into ML-ready features with temporal integrity.

**Input**:
- `data/t20s_json/*.json` - 15,000+ raw match files

**Output**:
- `data/xgb_data/cricket_data_v2_train.parquet`
- `data/xgb_data/cricket_data_v2_validation.parquet`
- `data/xgb_data/cricket_data_v2_test.parquet`
- `models/cache_chunks/cache_chunk_*.pkl` (69 files, ~7.6GB)
- `models/player_stats_cache_metadata.pkl`

**Command**:
```bash
python scripts/parsing_v2.py
```

**Performance**:
- Duration: ~10-15 minutes
- Memory: ~4-8 GB
- Processes: 8,341 matches → 4+ million ball records
- Cache: 3,442 date snapshots across 69 chunks

**What It Does**:
1. Loads matches chronologically
2. For each match:
   - Takes snapshot of current player stats (BEFORE match)
   - Processes each ball and extracts 29 features
   - Updates player stats (AFTER each ball)
3. Saves chunked cache (every 50 date snapshots)
4. Splits data into train/val/test by date
5. Outputs parquet files for training

**Temporal Splits**:
- Train: Matches before 2023
- Validation: Matches 2022-2024
- Test: Matches 2024
- Golden test: Matches Oct 2024+

---

### 2. Model Training (`xgboost_v2.py`)

**Purpose**: Train XGBoost classifier with Optuna hyperparameter tuning.

**Input**:
- `data/xgb_data/cricket_data_v2_*.parquet`

**Output**:
- `models/xgb/xgboost_model_v2.pkl`
- `models/xgb/xgboost_model_v2_optimized.pkl`
- `models/xgb/batter_encoder_v2.pkl`
- `models/xgb/bowler_encoder_v2.pkl`
- `models/xgb/feature_columns_v2.txt`
- `models/xgb/optuna_study_v2.pkl`

**Command**:
```bash
python scripts/xgboost_v2.py
```

**Performance**:
- Duration: ~30-60 minutes (with Optuna 50 trials)
- Memory: ~8-16 GB
- Results: ~55-60% ball-level accuracy

**What It Does**:
1. Loads parquet files
2. Encodes player IDs (fit LabelEncoders)
3. Filters and remaps outcome classes
4. Calculates balanced class weights
5. Runs Optuna hyperparameter search (50 trials)
6. Trains final model with best parameters
7. Evaluates on test set
8. Saves all artifacts

**Model Configuration**:
- 6-class classifier (dot, 1, 2, 4, 6, wicket)
- 29 input features
- Balanced class weights
- Early stopping (100 rounds)

---

## Simulation

### Evaluation Pipeline (`run_sim_eval.py`)

**Purpose**: Evaluate model predictions against betting market odds.

**Command**:
```bash
python scripts/sim_eval/run_sim_eval.py \
    --test-dir data/betting_test \
    --odds betting_odds_v3.json \
    --n-sims 1000
```

**Options**:
- `--test-dir`: Directory with test match JSONs
- `--odds`: Path to betting odds JSON file
- `--n-sims`: Number of simulations per match (default: 1000)
- `--parallel`: Enable parallel simulation (default: True)
- `--n-workers`: Number of CPU workers (default: auto)

**Performance**:
- ~10-30 seconds per match (1000 simulations)
- Parallelizes across CPU cores
- Memory: ~2-3 GB (loads stats cache once)

**Output**:
```
MATCH LEVEL EVALUATION SUMMARY
============================================================

Matches evaluated: 45
Total simulation time: 450.2s
Average time per match: 10.0s

--- Performance Metrics ---
Average Log Loss: 0.6234
Average Brier Score: 0.1845
Average Edge (magnitude): 8.3%
Average Signed Edge: +2.1% (underconfident)
Profitable opportunities (edge > 0.0%): 28

--- Actual Betting Performance ---
Total P&L: +12.45 units
ROI: +44.5%
Win Rate: 57.1%
Bets Placed: 28
```

---

### Standalone Simulation (Python API)

**Full Example**:
```python
from scripts.sim_v1_2 import *
from scripts.stats_provider import StatsProvider
from datetime import datetime

# Step 1: Load model components
stats_provider = StatsProvider('models')
model = XGBoostModelV2(
    'models/xgb/xgboost_model_v2.pkl',
    'models/xgb/batter_encoder_v2.pkl',
    'models/xgb/bowler_encoder_v2.pkl',
    'models/xgb/feature_columns_v2.txt',
    stats_provider=stats_provider
)
engine = SimulationEngine(model, T20Rules())

# Step 2: Define team lineups (11 players in batting order)
india_players = [
    Player("253802", "Rohit Sharma", "India", "batsman"),
    Player("277906", "Virat Kohli", "India", "batsman"),
    Player("326016", "Suryakumar Yadav", "India", "batsman"),
    # ... add 8 more players (11 total)
]

australia_players = [
    Player("219889", "David Warner", "Australia", "batsman"),
    Player("267192", "Travis Head", "Australia", "batsman"),
    # ... add 9 more players (11 total)
]

india = TeamLineup("India", india_players)
australia = TeamLineup("Australia", australia_players)

# Step 3: Create match state
state = MatchState(
    team1_lineup=india,
    team2_lineup=australia,
    batting_first="India",
    venue="MCG",
    match_date=datetime(2024, 6, 15)  # Must be ≤ latest cache date
)

# Step 4: Run simulations
config = SimulationConfig(
    n_simulations=1000,
    parallel=True,
    n_workers=4,
    random_seed=42,
    verbose=True
)
results = engine.simulate_multiple(state, config)

# Step 5: Analyze results
summary = ResultAggregator.aggregate(results)

print(f"\n{'='*60}")
print(f"Win Probabilities:")
print(f"  {india.team_name}: {summary['win_probability'][india.team_name]:.1%}")
print(f"  {australia.team_name}: {summary['win_probability'][australia.team_name]:.1%}")

print(f"\nExpected Scores:")
india_stats = summary['score_stats'][india.team_name]
australia_stats = summary['score_stats'][australia.team_name]
print(f"  {india.team_name}: {india_stats['mean']:.0f} ± {india_stats['std']:.0f}")
print(f"  {australia.team_name}: {australia_stats['mean']:.0f} ± {australia_stats['std']:.0f}")

print(f"\nScore Distributions:")
print(f"  {india.team_name}: 25th={india_stats['percentiles'][25]:.0f}, "
      f"50th={india_stats['percentiles'][50]:.0f}, "
      f"75th={india_stats['percentiles'][75]:.0f}")
print(f"  {australia.team_name}: 25th={australia_stats['percentiles'][25]:.0f}, "
      f"50th={australia_stats['percentiles'][50]:.0f}, "
      f"75th={australia_stats['percentiles'][75]:.0f}")
print(f"{'='*60}\n")
```

---

## Common Operations

### Operation 1: Retrain Model with New Data

```bash
# 1. Add new matches to data folder
cp new_matches/*.json data/t20s_json/

# 2. Re-run feature engineering (updates cache and parquet files)
python scripts/parsing_v2.py

# 3. Re-train model
python scripts/xgboost_v2.py

# 4. Verify performance on test set
python scripts/sim_eval/run_sim_eval.py \
    --test-dir data/betting_test \
    --odds betting_odds_v3.json \
    --n-sims 1000
```

**Notes**:
- Feature engineering rebuilds entire cache (ensures temporal integrity)
- Takes ~10-15 minutes for full dataset
- Model training takes ~30-60 minutes with Optuna

---

### Operation 2: Simulate a Specific Match

See "Standalone Simulation" example above. Key steps:
1. Load model + stats provider
2. Create team lineups (player IDs must match training data)
3. Create MatchState
4. Run simulation with SimulationConfig
5. Aggregate and analyze results

**Important**: Player IDs must match those in training data. To find player IDs:

```python
from scripts.stats_provider import StatsProvider

provider = StatsProvider('models')
print(f"Total batting players: {provider.metadata['num_players_batting']}")
print(f"Date range: {provider.dates[0]} to {provider.dates[-1]}")

# Check if player exists
player_id = "253802"  # Example: Rohit Sharma
batting_stats = provider.get_batting_stats(player_id, '2024-06-01')
if batting_stats['avg'] > 0:
    print(f"Player {player_id} found: Avg={batting_stats['avg']:.1f}, SR={batting_stats['sr']:.1f}")
```

---

### Operation 3: Evaluate Single Match

```python
from scripts.sim_eval.match_evaluator import MatchLevelEvaluator
from scripts.sim_eval.loaders import TestMatchLoader, BettingOddsLoader

# Load model + engine (as in standalone simulation)
# ...

# Load specific match
loader = TestMatchLoader()
matches = loader.load_matches('data/betting_test/')
match_id, match_state = matches[0]  # First match

# Load betting odds
odds_lookup = BettingOddsLoader.load_odds('betting_odds_v3.json')

# Evaluate
evaluator = MatchLevelEvaluator(model, engine, n_simulations=1000)
result = evaluator._evaluate_single_match(match_id, match_state, odds_lookup[match_id])

# Print results
print(f"\nMatch: {match_id}")
print(f"  Teams: {result.team1} vs {result.team2}")
print(f"\nSimulated Win Probabilities:")
print(f"  {result.team1}: {result.simulated_win_prob[result.team1]:.1%}")
print(f"  {result.team2}: {result.simulated_win_prob[result.team2]:.1%}")
print(f"\nMarket Win Probabilities:")
print(f"  {result.team1}: {result.market_win_prob[result.team1]:.1%}")
print(f"  {result.team2}: {result.market_win_prob[result.team2]:.1%}")
print(f"\nEdge:")
print(f"  {result.team1}: {result.edge[result.team1]:+.1%}")
print(f"  {result.team2}: {result.edge[result.team2]:+.1%}")
print(f"\nMetrics:")
print(f"  Log Loss: {result.log_loss:.3f}")
print(f"  Brier Score: {result.brier_score:.3f}")
print(f"  Actual Winner: {result.actual_winner}")
```

---

### Operation 4: Inspect Player Stats Cache

```python
from scripts.stats_provider import StatsProvider

provider = StatsProvider('models')

# Cache metadata
print(f"Cache Info:")
print(f"  Chunks: {provider.metadata['num_chunks']}")
print(f"  Dates: {provider.metadata['num_dates']}")
print(f"  Date range: {provider.dates[0]} to {provider.dates[-1]}")
print(f"  Batting players: {provider.metadata['num_players_batting']:,}")
print(f"  Bowling players: {provider.metadata['num_players_bowling']:,}")

# Query specific player
player_id = "253802"  # Example
date = "2024-06-15"

batting = provider.get_batting_stats(player_id, date)
bowling = provider.get_bowling_stats(player_id, date)

print(f"\nPlayer {player_id} stats as of {date}:")
print(f"  Batting: Avg={batting['avg']:.1f}, SR={batting['sr']:.1f}")
print(f"  Bowling: Avg={bowling['avg']:.1f}, Econ={bowling['econ']:.1f}")

# H2H matchup
bowler_id = "290630"  # Example
h2h = provider.get_h2h_stats(player_id, bowler_id, date)
print(f"  H2H vs {bowler_id}: Avg={h2h['avg']:.1f}, SR={h2h['sr']:.1f}")
```

---

### Operation 5: Debug Simulation Issues

**Enable Verbose Logging**:
```python
config = SimulationConfig(
    n_simulations=10,      # Fewer for debugging
    parallel=False,        # Sequential for traceback
    verbose=True,          # Print progress
    random_seed=42         # Reproducible
)

try:
    result = engine.simulate_match(state, "debug_match")

    # Inspect ball-by-ball
    innings1 = result.innings[0]
    print(f"\nInnings 1 - {innings1.batting_team}:")
    print(f"Total: {innings1.total_runs}/{innings1.total_wickets} in {innings1.total_balls} balls")

    print(f"\nFirst 20 balls:")
    for i, ball in enumerate(innings1.balls[:20]):
        print(f"  {ball.over}.{ball.ball}: {ball.outcome.name} → {ball.runs} runs "
              f"(Score: {ball.team_runs}/{ball.team_wickets})")

except Exception as e:
    import traceback
    traceback.print_exc()

    print(f"\nState at failure:")
    print(f"  Innings: {state.innings}")
    print(f"  Balls: {state.balls}")
    print(f"  Overs: {state.overs_completed:.1f}")
    print(f"  Score: {state.runs}")
    print(f"  Wickets: {state.wickets}")
    print(f"  Striker: {state.striker_idx}, Non-striker: {state.non_striker_idx}")
    print(f"  Bowler: {state.bowler_idx}")
```

---

## Performance Benchmarks

| Operation | Duration | Memory | Notes |
|-----------|----------|--------|-------|
| Feature Engineering | 10-15 min | 4-8 GB | Full dataset (15K matches) |
| Model Training | 30-60 min | 8-16 GB | With Optuna (50 trials) |
| Stats Cache Load | 1-2 sec | 300-550 MB | Lazy loading (5 chunks max) |
| Single Match Simulation | 0.01-0.1 sec | ~2 GB | 1000 simulations, parallel |
| Full Evaluation (45 matches) | 5-10 min | ~2-3 GB | 1000 sims/match, parallel |

**Optimization Tips**:
- Use `parallel=True` for simulations (4x speedup on 4 cores)
- Reduce `n_simulations` for faster testing (100-200 is often sufficient)
- Stats cache uses LRU eviction (~95% hit rate for sequential dates)
- Parquet files are columnar and fast to load

---

## Troubleshooting

### Issue: "Player not found in training data"

**Cause**: Player ID not in encoder vocabulary.

**Solution**:
```python
# Check if player exists in training data
import joblib
encoder = joblib.load('models/xgb/batter_encoder_v2.pkl')
print(f"Known players: {len(encoder.classes_)}")
print(f"Sample IDs: {encoder.classes_[:10]}")

# If player is new, retrain with updated data
# OR use a known player with similar stats
```

---

### Issue: "KeyError: date not in cache"

**Cause**: Match date is before earliest cache snapshot or after latest.

**Solution**:
```python
provider = StatsProvider('models')
print(f"Cache date range: {provider.dates[0]} to {provider.dates[-1]}")

# Use date within range
match_date = datetime.strptime(provider.dates[-1], '%Y-%m-%d')
state = MatchState(..., match_date=match_date)
```

---

### Issue: Simulation very slow

**Causes**:
1. `parallel=False` (sequential processing)
2. Too many simulations (`n_simulations > 1000`)
3. Stats cache not loaded

**Solutions**:
```python
# Enable parallelization
config = SimulationConfig(
    n_simulations=1000,
    parallel=True,
    n_workers=4  # Or os.cpu_count()
)

# Reduce simulations for testing
config = SimulationConfig(n_simulations=100)

# Preload stats provider before simulation
stats_provider = StatsProvider('models')  # Loads in ~2 sec
model = XGBoostModelV2(..., stats_provider=stats_provider)
```

---

### Issue: Out of memory during training

**Cause**: Loading full parquet files into memory.

**Solutions**:
```python
# In xgboost_v2.py, use chunked loading:
import pandas as pd

# Load in chunks
chunksize = 500000
chunks = []
for chunk in pd.read_parquet('data/xgb_data/cricket_data_v2_train.parquet', chunksize=chunksize):
    # Process chunk
    chunks.append(chunk)

# Or reduce data size for testing
df_train = pd.read_parquet('...').sample(frac=0.1)  # Use 10% of data
```

---

## Environment Setup

### Dependencies
See `requirements.txt`:
```
xgboost>=2.0.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
optuna>=3.0.0
joblib>=1.3.0
```

### Installation
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Or use uv (faster)
uv pip install -r requirements.txt
```

### System Requirements
- Python 3.11+
- 16GB RAM (minimum 8GB)
- 10GB disk space for cache
- 4+ CPU cores (recommended for parallel simulation)

---

**For detailed implementation information, see [CLAUDE_REFERENCE.md](../CLAUDE_REFERENCE.md).**
