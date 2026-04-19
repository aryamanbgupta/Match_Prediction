# CricML Match Prediction System - Agent Reference Guide

**Last Updated**: October 2024 (Stats Cache System Implemented)
**Branch**: feature/player-stats-cache
**Purpose**: Portfolio project with potential as production analytics tool

---

## Executive Summary

A comprehensive ball-by-ball T20 cricket prediction system that uses machine learning to predict individual ball outcomes, simulates complete matches via Monte Carlo methods, and evaluates predictions against betting market odds.

**Core Pipeline**: Raw JSON → Feature Engineering → XGBoost Classifier → Monte Carlo Simulation → Betting Evaluation

**Key Innovation**: Temporal player stats cache with lazy loading prevents data leakage in simulations by using only historically available statistics. The cache stores 3,442 date snapshots from 8,341 matches across 69 chunks (~7.6GB on disk, ~300-550MB in memory).

---

## Table of Contents

1. [High-Level Architecture](#high-level-architecture)
2. [Directory Structure](#directory-structure)
3. [Core Modules](#core-modules)
4. [Data Flows](#data-flows)
5. [Key Classes & Interfaces](#key-classes--interfaces)
6. [Entry Points](#entry-points)
7. [Design Decisions](#design-decisions)
8. [Data Formats](#data-formats)
9. [Common Operations](#common-operations)

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         TRAINING PIPELINE                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Raw JSON Files (data/t20s_json/)                                   │
│         │                                                            │
│         ▼                                                            │
│  ┌──────────────────────────────────────┐                          │
│  │   parsing_v2.py                      │                          │
│  │   - PlayerStatsTracker               │                          │
│  │   - InningsFeatureCalculator         │                          │
│  │   - Temporal feature engineering     │                          │
│  └──────────────┬───────────────────────┘                          │
│                 │                                                    │
│                 ├──► Parquet Files (data/xgb_data/)                │
│                 │    - train, val, test splits                      │
│                 │                                                    │
│                 └──► Player Stats Cache (models/)                   │
│                      - Date-indexed snapshots                       │
│                                                                       │
│  ┌──────────────────────────────────────┐                          │
│  │   xgboost_v2.py                      │                          │
│  │   - XGBoost classifier               │                          │
│  │   - Optuna hyperparameter tuning     │                          │
│  │   - Class balancing                  │                          │
│  └──────────────┬───────────────────────┘                          │
│                 │                                                    │
│                 ▼                                                    │
│  Trained Model Artifacts (models/xgb/)                             │
│  - xgboost_model_v2.pkl                                            │
│  - batter_encoder_v2.pkl                                           │
│  - bowler_encoder_v2.pkl                                           │
│  - feature_columns_v2.txt                                          │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                      SIMULATION PIPELINE                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Test Match JSON (data/betting_test/)                              │
│         │                                                            │
│         ▼                                                            │
│  ┌──────────────────────────────────────┐                          │
│  │   loaders.py                         │                          │
│  │   - TestMatchLoader                  │                          │
│  │   - Creates initial MatchState       │                          │
│  └──────────────┬───────────────────────┘                          │
│                 │                                                    │
│                 ▼                                                    │
│  ┌──────────────────────────────────────┐                          │
│  │   sim_v1_2.py                        │                          │
│  │   ┌────────────────────────────────┐ │                          │
│  │   │ SimulationEngine               │ │                          │
│  │   │   ├─► XGBoostModelV2           │ │                          │
│  │   │   │    └─► StatsProvider       │ │◄─ cache_chunks/*.pkl     │
│  │   │   │         (temporal stats)    │ │                          │
│  │   │   ├─► T20Rules                 │ │                          │
│  │   │   └─► Monte Carlo loop         │ │                          │
│  │   └────────────────────────────────┘ │                          │
│  └──────────────┬───────────────────────┘                          │
│                 │                                                    │
│                 ▼                                                    │
│  Simulation Results (1000+ per match)                              │
│  - Win probabilities                                                │
│  - Score distributions                                              │
│  - Ball-by-ball records                                             │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                      EVALUATION PIPELINE                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Betting Odds JSON (betting_odds_v3.json)                          │
│         │                                                            │
│         ▼                                                            │
│  ┌──────────────────────────────────────┐                          │
│  │   match_evaluator.py                 │                          │
│  │   - MatchLevelEvaluator              │                          │
│  │   - Metrics calculation:             │                          │
│  │     • Log Loss                       │                          │
│  │     • Brier Score                    │                          │
│  │     • Edge (model vs market)         │                          │
│  │     • Calibration                    │                          │
│  │     • ROI & Win Rate                 │                          │
│  └──────────────┬───────────────────────┘                          │
│                 │                                                    │
│                 ▼                                                    │
│  Evaluation Results                                                 │
│  - match_evaluation_results.json                                    │
│  - Printed summary statistics                                       │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
Match_Prediction/
│
├── data/                          # All data files
│   ├── t20s_json/                # Raw match JSON (15000+ files)
│   ├── train/                    # JSON matches pre-2023
│   ├── validation/               # JSON matches 2022-2024
│   ├── test/                     # JSON matches 2024
│   ├── golden_test/              # JSON matches Oct 2024+
│   ├── betting_test/             # T20 World Cup 2024 matches
│   ├── betting_info/             # Scraping metadata
│   └── xgb_data/                 # Processed parquet files
│       ├── cricket_data_v2_train.parquet
│       ├── cricket_data_v2_validation.parquet
│       ├── cricket_data_v2_test.parquet
│       └── cricket_data_v2_betting_test.parquet
│
├── models/                        # Trained models & artifacts
│   ├── xgb/                      # XGBoost model directory
│   │   ├── xgboost_model_v2.pkl
│   │   ├── xgboost_model_v2_optimized.pkl
│   │   ├── batter_encoder_v2.pkl
│   │   ├── bowler_encoder_v2.pkl
│   │   ├── feature_columns_v2.txt
│   │   └── optuna_study_v2.pkl
│   ├── cache_chunks/             # Player stats cache chunks (69 files)
│   │   ├── cache_chunk_0.pkl    # ~110MB each
│   │   ├── cache_chunk_1.pkl
│   │   └── ... (67 more chunks)
│   ├── player_stats_cache_metadata.pkl  # Cache index & metadata
│   ├── gradient_boosting_model.pkl  # Legacy v1 model
│   ├── batter_encoder.pkl           # Legacy v1 encoder
│   └── bowler_encoder.pkl           # Legacy v1 encoder
│
├── scripts/                       # All Python scripts
│   ├── parsing_v2.py             # Feature engineering pipeline
│   ├── xgboost_v2.py             # Model training script
│   ├── sim_v1_2.py               # Simulation engine (MAIN)
│   ├── stats_provider.py         # Temporal stats access
│   │
│   ├── sim_eval/                 # Evaluation module
│   │   ├── run_sim_eval.py      # Main evaluation script
│   │   ├── match_evaluator.py   # Evaluation logic
│   │   └── loaders.py           # Data loaders
│   │
│   ├── data_parsing.py           # Legacy v1 parser
│   ├── gbm_v1.py                 # Legacy GBM training
│   ├── sim_v1.py                 # Legacy simulation
│   ├── sim_v1_1.py               # Legacy simulation
│   ├── parse_betting_odds.py    # Odds scraper v1
│   └── parse_betting_odds_v2.py # Odds scraper v2
│
├── features/                      # Feature definitions
│   └── v1/                       # Feature set v1
│
├── notebooks/                     # Jupyter notebooks
│
├── cricket_data_v2_with_features.parquet  # Full processed data
├── betting_odds_v3.json          # Latest betting odds
├── match_evaluation_results.json # Latest evaluation results
│
├── cricket_prediction_workflow.md # High-level design doc
├── cricket_simulation_docs.md     # Simulation system docs
├── TODO.md                        # Task list
├── README.md                      # (Currently empty)
├── CLAUDE.md                      # This file
│
├── requirements.txt               # Python dependencies
├── pyproject.toml                # Project config
└── uv.lock                       # Dependency lock file
```

---

## Core Modules

### 1. `scripts/parsing_v2.py` - Feature Engineering Pipeline

**Purpose**: Transform raw cricket JSON into ML-ready features with temporal integrity.

**Key Classes**:
- `PlayerStatsTracker`: Accumulates player statistics across matches
  - Batting stats (runs, balls, dismissals)
  - Bowling stats (runs_given, balls_bowled, wickets)
  - Head-to-head matchup records
  - Recent form tracking (last 5 matches)

- `InningsFeatureCalculator`: Calculates momentum features per innings
  - Rolling windows (last 5, 10, 30 balls)
  - Balls since boundary
  - Dot ball pressure

**Key Functions**:
- `parse_match_data_v2(json_data, tracker)`: Parses single match
- `process_folder_v2_with_splits(folder_path)`: Processes entire dataset
  - Creates temporal splits (train/val/test)
  - Generates player stats cache
  - Returns separate parquet files

**Features Generated** (29 total):
```python
Basic Features (12):
  - inning_idx, score, wickets, balls_bowled
  - run_rate, wickets_ratio, balls_ratio, wickets_in_hand
  - is_powerplay, is_middle_overs, is_death_overs, balls_in_over

Player Features (6):
  - batter_encoded, bowler_encoded
  - batsman_avg, batsman_sr
  - bowler_avg, bowler_econ

H2H Features (2):
  - h2h_avg, h2h_sr

Momentum Features (5):
  - last_5_balls_runs, last_10_balls_runs, last_30_balls_runs
  - balls_since_boundary, last_10_dots

Pressure Features (2):
  - dot_percentage_recent
  - boundary_percentage_recent
```

**Critical Design Decision**: Features reflect state BEFORE ball is bowled, preventing data leakage.

**Output Files**:
- `data/xgb_data/cricket_data_v2_train.parquet`
- `data/xgb_data/cricket_data_v2_validation.parquet`
- `data/xgb_data/cricket_data_v2_test.parquet`
- `models/cache_chunks/cache_chunk_*.pkl` (69 files)
- `models/player_stats_cache_metadata.pkl`

**Stats Cache Building** (integrated into parsing):

The parser now builds a temporal stats cache simultaneously with feature extraction:

```python
# Initialization
stats_snapshots = {}  # Accumulator for current chunk
cache_chunks = []     # List of saved chunk files
save_interval = 50    # Save every 50 snapshots to avoid memory issues

# For each match (processed chronologically):
for match_file in sorted(json_files):
    match_date_str = match_date.strftime('%Y-%m-%d')

    # CRITICAL: Take snapshot BEFORE processing match
    if match_date_str not in stats_snapshots:
        stats_snapshots[match_date_str] = deep_copy_stats(player_stats_tracker)

        # Save chunk every 50 snapshots
        if len(stats_snapshots) >= save_interval:
            chunk_file = Path(f'models/cache_chunks/cache_chunk_{len(cache_chunks)}.pkl')
            with open(chunk_file, 'wb') as f:
                pickle.dump(stats_snapshots, f, protocol=pickle.HIGHEST_PROTOCOL)

            cache_chunks.append(chunk_file)
            print(f"💾 Saved snapshot chunk {len(cache_chunks)} ({len(stats_snapshots)} dates)")
            stats_snapshots = {}  # Clear for next chunk

    # Process match and update tracker
    process_match(match_data, player_stats_tracker)

# After all matches: Build metadata with date indices
chunks_with_dates = []
for i, chunk_file in enumerate(cache_chunks):
    # Load chunk to extract date list
    with open(chunk_file, 'rb') as f:
        chunk_data = pickle.load(f)

    dates = sorted(chunk_data.keys())
    chunks_with_dates.append({
        'file': str(chunk_file.relative_to('models')),
        'dates': dates,
        'num_dates': len(dates)
    })
    del chunk_data  # Free memory

# Save metadata
metadata = {
    'num_chunks': len(cache_chunks),
    'num_matches': processed_files,
    'num_dates': total_dates,
    'num_players_batting': len(player_stats_tracker.batting_stats),
    'num_players_bowling': len(player_stats_tracker.bowling_stats),
    'num_h2h_matchups': len(player_stats_tracker.h2h_stats),
    'build_timestamp': datetime.now().isoformat(),
    'chunk_files': [str(f.relative_to('models')) for f in cache_chunks],
    'chunks': chunks_with_dates  # Date indices for lazy loading
}

with open('models/player_stats_cache_metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)
```

**Why Chunking?**
- **Memory efficiency**: Avoids loading 7.6GB into memory during parsing
- **Incremental saves**: Prevents data loss if parsing crashes
- **Lazy loading**: Enables fast simulation startup (load metadata only)

**Cache Statistics** (from actual run on 8,341 matches):
```
Total Chunks:        69
Total Snapshots:     3,442
Date Range:          2005-02-17 to 2025-06-15
Players (Batting):   7,240
Players (Bowling):   5,319
H2H Matchups:        ~50,000
Total Size on Disk:  7.6GB
Avg Chunk Size:      ~110MB
Build Time:          ~10-15 minutes
```

**Usage**:
```bash
python scripts/parsing_v2.py
# Processes data/t20s_json/ folder
# Outputs:
#   - Parquet files for training
#   - 69 cache chunk files
#   - Metadata file with date indices
# Takes ~10-15 minutes for full dataset
```

---

### 2. `scripts/xgboost_v2.py` - Model Training

**Purpose**: Train XGBoost multi-class classifier with Optuna hyperparameter tuning.

**Model Architecture**:
- **Input**: 29 features (see parsing_v2.py)
- **Output**: 6 classes (mapped from 8 internal classes)
  - Class 0: Dot balls (0 runs)
  - Class 1: Singles (1 run)
  - Class 2: Twos (2 runs)
  - Class 3: Fours (4 runs)
  - Class 4: Sixes (6 runs)
  - Class 5: Wickets (-1 → 7 → 5 via remapping)
- **Loss**: Multi-class log loss
- **Optimization**: Optuna TPE sampler (50 trials)

**Class Mapping Logic**:
```python
# Raw data: 0,1,2,4,6,7 (3,5 normalized away)
# Internal training: 0,1,2,3,4,5
class_mapping = {0: 0, 1: 1, 2: 2, 4: 3, 6: 4, 7: 5}

# Model predictions map to outcomes:
class_to_outcome = {
    0: 'dot', 1: 'one', 2: 'two', 3: 'four',
    4: 'six', 5: 'wicket'
}
```

**Key Features**:
- Class balancing via `compute_class_weight('balanced')`
- Sample weights during training
- Early stopping (100 rounds)
- Validation monitoring
- Model versioning (saves both regular and optimized versions)

**Hyperparameters Tuned**:
- n_estimators: 50-500
- max_depth: 3-10
- learning_rate: 0.01-0.3 (log scale)
- subsample: 0.6-1.0
- colsample_bytree: 0.6-1.0
- reg_alpha: 0.01-10.0 (log scale)
- reg_lambda: 0.01-10.0 (log scale)

**Output Files**:
- `models/xgb/xgboost_model_v2.pkl`
- `models/xgb/xgboost_model_v2_optimized.pkl`
- `models/xgb/batter_encoder_v2.pkl`
- `models/xgb/bowler_encoder_v2.pkl`
- `models/xgb/feature_columns_v2.txt`
- `models/xgb/optuna_study_v2.pkl`

**Usage**:
```bash
python scripts/xgboost_v2.py
# Requires parquet files from parsing_v2.py
# Takes ~30-60 minutes with Optuna
```

**Performance Metrics**:
- Validation Accuracy: ~55-60%
- Validation Log Loss: ~1.1-1.3
- Test Accuracy: ~55-60%
- Test Log Loss: ~1.1-1.3

---

### 3. `scripts/stats_provider.py` - Temporal Stats Access with Lazy Loading

**Purpose**: Provide temporal player statistics for simulations without data leakage, using memory-efficient lazy loading.

**Key Class**: `StatsProvider`

**Design Philosophy**:
- **Lazy loading**: Load chunks on-demand instead of entire cache
- **LRU cache**: Keep only 5 most recently used chunks in memory (~550MB max)
- **Binary search**: O(log n) temporal lookups across 3,442 date snapshots
- **Returns same format as training features**: Ensures simulation-training consistency

**Initialization**:
```python
provider = StatsProvider('models')  # Pass directory, not file

# Output:
# Loading player stats cache from models...
#   Found 69 cache chunks
#   ✓ Initialized lazy loading for 3,442 date snapshots
#   Date range: 2005-02-17 to 2025-06-15
#   Players: 7,240 batters, 5,319 bowlers
#   Cache size: 5 chunks (~550MB max)
```

**Methods**:
```python
# Get stats as of specific date
batting = provider.get_batting_stats('player_id', '2024-06-01')
# Returns: {'avg': 31.4, 'sr': 140.2}

bowling = provider.get_bowling_stats('player_id', '2024-06-01')
# Returns: {'avg': 28.5, 'econ': 8.2}

h2h = provider.get_h2h_stats('batter_id', 'bowler_id', '2024-06-01')
# Returns: {'avg': 25.0, 'sr': 130.0}

# Convenience method
all_stats = provider.get_all_stats('batter_id', 'bowler_id', '2024-06-01')
# Returns all 6 features at once
```

**Per-match memo wrapper**: `StatsProviderCache` (same module) wraps a `StatsProvider` and caches `get_team_batting_elo`, `get_team_bowling_elo`, `get_team_batting_strength`, `get_team_bowling_strength`, and `get_venue_profile` keyed on `(tuple(lineup_ids), date)` or `(venue, date)`. These are constant for all sims of a single match, so memoizing avoids re-running an 11-player loop per ball. `wrap_with_cache(provider)` is idempotent and is applied automatically inside every model class's `__init__` — callers pass a plain `StatsProvider` and don't need to opt in. The wrapper is pickle-safe for `multiprocessing.Pool.starmap`.

**Cache Architecture** (Chunked Format):

```
models/
├── cache_chunks/
│   ├── cache_chunk_0.pkl     # Dates: 2005-02-17 to 2007-04-20 (50 snapshots)
│   ├── cache_chunk_1.pkl     # Dates: 2007-04-22 to 2009-06-15 (50 snapshots)
│   ├── ...
│   └── cache_chunk_68.pkl    # Dates: 2024-11-01 to 2025-06-15 (42 snapshots)
│
└── player_stats_cache_metadata.pkl  # Index & metadata
```

**Chunk Structure** (each .pkl file):
```python
{
    '2020-01-15': {
        'batting': {
            'player_id_1': {'runs': 450, 'balls': 320, 'dismissals': 12},
            'player_id_2': {'runs': 890, 'balls': 650, 'dismissals': 22},
            ...  # ~3,000 players
        },
        'bowling': {
            'player_id_1': {'runs_given': 1250, 'balls_bowled': 540, 'wickets': 18},
            ...  # ~2,500 players
        },
        'h2h': {
            ('batter_1', 'bowler_1'): {'runs': 45, 'balls': 32, 'dismissals': 1},
            ...  # ~50,000 matchups
        }
    },
    '2020-01-16': {...},
    ...  # ~50 dates per chunk
}
```

**Metadata Structure**:
```python
{
    'num_chunks': 69,
    'num_matches': 8341,
    'num_dates': 3442,
    'num_players_batting': 7240,
    'num_players_bowling': 5319,
    'num_h2h_matchups': ~50000,
    'build_timestamp': '2024-10-14T18:56:00',

    'chunks': [
        {
            'file': 'cache_chunks/cache_chunk_0.pkl',
            'dates': ['2005-02-17', '2005-02-19', ..., '2007-04-20'],
            'num_dates': 50
        },
        ...  # 69 chunk entries
    ]
}
```

**Lazy Loading Implementation**:

```python
class StatsProvider:
    def __init__(self, cache_dir: str = 'models', max_cached_chunks: int = 5):
        """
        Lazy loading with LRU cache

        - Loads metadata immediately (~10KB)
        - Builds date-to-chunk index for O(1) chunk lookup
        - Keeps max 5 chunks in memory (OrderedDict LRU)
        """
        # Load metadata
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)

        # Build date-to-chunk-index mapping
        self.date_to_chunk_idx = {}
        all_dates = []
        for chunk_idx, chunk_info in enumerate(self.metadata['chunks']):
            for date in chunk_info['dates']:
                self.date_to_chunk_idx[date] = chunk_idx
                all_dates.append(date)

        self.dates = sorted(all_dates)  # For binary search
        self.chunk_cache = OrderedDict()  # LRU cache

    def _load_chunk(self, chunk_idx: int) -> Dict:
        """Load chunk from disk and add to LRU cache"""
        # Check if already cached
        if chunk_idx in self.chunk_cache:
            self.chunk_cache.move_to_end(chunk_idx)  # Mark as recently used
            return self.chunk_cache[chunk_idx]

        # Load from disk
        chunk_path = self.cache_dir / self.metadata['chunks'][chunk_idx]['file']
        with open(chunk_path, 'rb') as f:
            chunk_data = pickle.load(f)

        # Add to cache
        self.chunk_cache[chunk_idx] = chunk_data

        # Evict oldest if cache full (LRU eviction)
        if len(self.chunk_cache) > self.max_cached_chunks:
            self.chunk_cache.popitem(last=False)  # Remove first item

        return chunk_data

    def _get_snapshot_for_date(self, target_date: str) -> Optional[Dict]:
        """Find most recent snapshot before or on target_date"""
        # Binary search for rightmost date ≤ target_date
        idx = bisect.bisect_right(self.dates, target_date)
        if idx == 0:
            return None  # No history exists

        snapshot_date = self.dates[idx - 1]

        # Find which chunk contains this date
        chunk_idx = self.date_to_chunk_idx[snapshot_date]

        # Load chunk (from cache or disk)
        chunk_data = self._load_chunk(chunk_idx)

        return chunk_data[snapshot_date]
```

**Temporal Lookup Logic**:
1. **Binary search**: Find rightmost date ≤ target_date in sorted date list (O(log n))
2. **Chunk lookup**: Use date-to-chunk index to find which chunk file (O(1))
3. **Lazy load**: Load chunk if not in LRU cache, evict oldest if needed
4. **Return snapshot**: Extract snapshot for that date from chunk
5. **If no prior date exists**: Return zeros (unknown player or before their career)

**Performance Characteristics**:
```
Cache Load Time:  ~1-2 seconds (metadata only)
Memory Usage:     300-550MB (5 chunks max, vs 7.6GB for all chunks)
Query Speed:      <0.01ms after chunk is cached
Disk Size:        7.6GB total (69 chunks × ~110MB)
Cache Hit Rate:   ~95%+ for sequential date queries
LRU Evictions:    Rare during simulation (dates are usually sequential)
```

**Integration**: Used by `XGBoostModelV2` during simulations to provide real player statistics.

**Validation**:
- ✅ All stats match training data exactly (verified via `validate_training_cache_match.py`)
- ✅ Temporal validity confirmed (player counts increase over time: 105 early → 7,223 late)
- ✅ No data leakage (snapshots taken BEFORE match processing)

---

### 4. `scripts/sim_v1_2.py` - Monte Carlo Simulation Engine

**Purpose**: Simulate complete T20 matches ball-by-ball using trained models.

**Core Architecture**:

```
┌─────────────────────────────────────────────────────────────┐
│                   SimulationEngine                          │
│                                                               │
│   ┌─────────────┐      ┌──────────────┐                    │
│   │ Prediction  │      │   T20Rules   │                    │
│   │   Model     │◄─────┤   Engine     │                    │
│   │ (XGBoost)   │      │              │                    │
│   └─────────────┘      └──────────────┘                    │
│         │                      │                             │
│         ▼                      ▼                             │
│    Extract Features ──► Predict Ball ──► Update State       │
│         │                                      │             │
│         └──────────────────────────────────────┘             │
│                      (Loop until match ends)                 │
└─────────────────────────────────────────────────────────────┘
```

**Key Classes**:

1. **Data Classes** (using @dataclass):
   - `Outcome(Enum)`: Ball outcomes (DOT, ONE, TWO, FOUR, SIX, WICKET, WIDE, NO_BALL)
   - `Player`: Player representation (player_id, name, team, role)
   - `TeamLineup`: Team with 11 players in batting order
   - `MatchState`: Complete match state (teams, innings, scores, wickets, players, history)

2. **Rules Engine**:
   - `BowlerSelector(ABC)`: Interface for bowler selection strategies
   - `RandomBowlerSelector`: Simple random selection (current implementation)
   - `T20Rules`: Enforces cricket rules and game flow
     - `select_next_bowler()`: Choose bowler for next over
     - `is_legal_outcome()`: Validate outcomes
     - `process_ball()`: Update state after ball
     - `simulate_ball()`: Full ball simulation cycle

3. **Prediction Models** (inherit from `PredictionModel(ABC)`):
   - `XGBoostModel`: V1 model (9 basic features)
   - `XGBoostModelV2`: V2 model (29 comprehensive features, with StatsProvider)
   - `DummyModel`: Testing model (fixed probabilities)

4. **Simulation**:
   - `SimulationConfig`: Configuration (n_simulations, parallel, seed, verbose)
   - `BallResult`: Single ball outcome record
   - `InningsResult`: Innings summary with batting/bowling cards
   - `MatchResult`: Complete match result
   - `SimulationEngine`: Main orchestrator
   - `ResultAggregator`: Aggregate multiple simulations

**MatchState Management**:

The `MatchState` class is the heart of the simulation:

```python
@dataclass
class MatchState:
    # Immutable setup
    team1_lineup: TeamLineup
    team2_lineup: TeamLineup
    batting_first: str
    venue: str
    match_date: datetime

    # Dynamic state
    innings: int = 1                # 1 or 2
    balls: int = 0                  # 0-119 for T20
    runs: np.ndarray                # [team1_runs, team2_runs]
    wickets: np.ndarray             # [team1_wickets, team2_wickets]

    # Current players
    striker_idx: int = 0            # 0-10 (batting order)
    non_striker_idx: int = 1
    bowler_idx: int = 0
    last_bowler_idx: int = -1

    # Tracking
    batsmen_out: Dict[int, List[int]]  # team_idx -> [out_indices]
    history: np.ndarray                # Ball-by-ball records (300x9)
    bowler_balls: Dict                 # Bowler overs tracking
    batsman_stats: Dict                # In-match batting stats

    # Properties (computed)
    @property
    def overs_completed(self) -> float
    @property
    def target(self) -> Optional[int]
    @property
    def required_run_rate(self) -> Optional[float]
    # ... many more
```

**Ball Simulation Flow**:

```python
# 1. Extract features from current state
features = model.extract_features(state)

# 2. Predict outcome probabilities
probs = model.predict_next_ball(features)
# {'dot': 0.32, 'one': 0.39, 'two': 0.08, ...}

# 3. Sample outcome
outcome = random.choices(outcomes, weights=probs)[0]

# 4. Validate and process
if not rules.is_legal_outcome(state, outcome):
    outcome = Outcome.DOT
runs = rules.process_ball(state, outcome)

# 5. Update state
state.update(outcome, runs)

# 6. Check for end conditions
if state.is_innings_over():
    if state.innings == 1:
        state.start_new_innings()
    else:
        break  # Match over

# 7. Select new bowler if over ended
if balls % 6 == 0:
    state.bowler_idx = rules.select_next_bowler(state)
```

**XGBoostModelV2 Integration**:

```python
class XGBoostModelV2(PredictionModel):
    def __init__(self, model_path, batter_encoder_path,
                 bowler_encoder_path, feature_columns_path,
                 stats_provider=None):
        self.model = joblib.load(model_path)
        self.batter_encoder = joblib.load(batter_encoder_path)
        self.bowler_encoder = joblib.load(bowler_encoder_path)
        self.stats_provider = stats_provider  # KEY: Temporal stats

        with open(feature_columns_path) as f:
            self.feature_columns = [line.strip() for line in f]

    def extract_features(self, state: MatchState) -> pd.DataFrame:
        # Basic features (from state)
        features = {
            'inning_idx': state.innings,
            'score': state.runs[state.current_team_idx],
            # ... 12 basic features
        }

        # Player encoding
        features['batter_encoded'] = self.batter_encoder.transform(...)
        features['bowler_encoded'] = self.bowler_encoder.transform(...)

        # Player stats (TEMPORAL via StatsProvider)
        if self.stats_provider:
            batting = self.stats_provider.get_batting_stats(
                striker.player_id, state.match_date
            )
            bowling = self.stats_provider.get_bowling_stats(
                bowler.player_id, state.match_date
            )
            h2h = self.stats_provider.get_h2h_stats(
                striker.player_id, bowler.player_id, state.match_date
            )
            features.update({
                'batsman_avg': batting['avg'],
                'batsman_sr': batting['sr'],
                'bowler_avg': bowling['avg'],
                'bowler_econ': bowling['econ'],
                'h2h_avg': h2h['avg'],
                'h2h_sr': h2h['sr']
            })
        else:
            # Fallback to zeros
            features.update({'batsman_avg': 0, 'batsman_sr': 0, ...})

        # Momentum features (from state history)
        features.update(self._extract_momentum_features(state))

        # Pressure features
        features.update(self._extract_pressure_features(state))

        return pd.DataFrame([features])[self.feature_columns]
```

**Parallel Simulation**:

```python
config = SimulationConfig(
    n_simulations=1000,
    parallel=True,        # Use multiprocessing
    n_workers=4,          # CPU cores
    random_seed=42,
    verbose=True
)

results = engine.simulate_multiple(initial_state, config)
# Returns List[MatchResult] with 1000 simulated outcomes

summary = ResultAggregator.aggregate(results)
# Aggregates win probabilities, score distributions, etc.
```

**Output Structure**:

```python
MatchResult:
  - match_id: str
  - team1, team2: str
  - winner: str ("Team1", "Team2", or "Tie")
  - margin: str ("23 runs" or "5 wickets")
  - innings: List[InningsResult]
  - team1_score, team1_wickets: int
  - team2_score, team2_wickets: int

InningsResult:
  - batting_team, bowling_team: str
  - total_runs, total_wickets, total_balls: int
  - run_rate: float
  - batting_card: Dict[player_idx, (runs, balls, 4s, 6s)]
  - bowling_card: Dict[player_idx, (balls, runs, wickets)]
  - balls: List[BallResult]

AggregatedResults:
  - win_probability: {team1: 0.653, team2: 0.347}
  - score_stats: {team1: {mean: 167.4, std: 15.2, percentiles: {...}}}
  - wicket_stats: {team1: {mean: 6.2, distribution: {...}}}
```

**Usage**:

```python
# Load model with temporal stats
stats_provider = StatsProvider('models')
model = XGBoostModelV2(
    'models/xgb/xgboost_model_v2.pkl',
    'models/xgb/batter_encoder_v2.pkl',
    'models/xgb/bowler_encoder_v2.pkl',
    'models/xgb/feature_columns_v2.txt',
    stats_provider=stats_provider
)

# Create engine
engine = SimulationEngine(model, T20Rules())

# Create match state
state = MatchState(
    team1_lineup=india_lineup,
    team2_lineup=australia_lineup,
    batting_first="India",
    venue="MCG",
    match_date=datetime(2024, 6, 15)
)

# Run simulations
config = SimulationConfig(n_simulations=1000, parallel=True)
results = engine.simulate_multiple(state, config)

# Analyze
summary = ResultAggregator.aggregate(results)
print(f"India: {summary['win_probability']['India']:.1%}")
print(f"Australia: {summary['win_probability']['Australia']:.1%}")
```

---

### 5. `scripts/sim_eval/` - Evaluation Framework

**Purpose**: Evaluate match-level predictions against betting market odds.

#### 5.1 `loaders.py` - Data Loaders

**TestMatchLoader**:
- Loads JSON match files
- Creates `MatchState` objects for simulation
- Extracts players in batting order
- Handles incomplete lineups

**BettingOddsLoader**:
- Loads betting odds JSON
- Converts decimal odds to implied probabilities
- Removes bookmaker margin for fair comparison
- Calculates overround

**Usage**:
```python
# Load test matches
loader = TestMatchLoader()
matches = loader.load_matches('data/betting_test/')
# Returns: List[(match_id, MatchState)]

# Load odds
odds = BettingOddsLoader.load_odds('betting_odds_v3.json')
# Returns: Dict[match_id, odds_data]

# Get implied probabilities
market_probs = BettingOddsLoader.get_implied_probabilities(
    {'India': 2.10, 'Australia': 1.75}
)
# Returns: {'India': 0.476, 'Australia': 0.524}
```

#### 5.2 `match_evaluator.py` - Evaluation Logic

**MatchLevelEvaluator**:

Comprehensive evaluation framework comparing model predictions to betting markets.

**Metrics Calculated**:

1. **Log Loss** (per match):
   ```
   log_loss = -log(P(actual winner))
   Lower is better (0 = perfect)
   ```

2. **Brier Score** (per match):
   ```
   brier = (P(team1) - actual)²
   where actual = 1 if team1 won, else 0
   Lower is better (0 = perfect)
   ```

3. **Edge** (per team, per match):
   ```
   edge = model_prob - market_prob
   Positive edge = value bet opportunity
   ```

4. **Calibration** (across matches):
   - Bins predictions by probability
   - Measures: if model says 70%, do we win 70% of time?

5. **Betting Performance**:
   - **Total P&L**: Sum of realized profits/losses
   - **ROI**: Return on investment %
   - **Win Rate**: Percentage of winning bets
   - **Bets Placed**: Count of positive edge opportunities

**Key Methods**:

```python
evaluator = MatchLevelEvaluator(
    model=model,
    simulation_engine=engine,
    n_simulations=1000
)

# Evaluate all matches
results = evaluator.evaluate_all(matches, odds_lookup)

# Results structure:
OverallEvaluationResults:
  - n_matches: int
  - avg_log_loss: float
  - avg_brier_score: float
  - avg_edge: float (magnitude)
  - avg_signed_edge: float (positive if correct)
  - profitable_bets: int
  - total_pnl: float
  - roi: float
  - win_rate: float
  - bets_placed: int
  - calibration_bins: List[(predicted, actual, count)]
  - match_results: List[MatchEvaluationResult]
```

**Edge Interpretation**:
- **Unsigned edge**: Magnitude of disagreement with market
- **Signed edge**:
  - Positive = Correctly predicted winner with edge
  - Negative = Incorrectly predicted winner
  - Shows prediction quality, not just disagreement

**Betting Strategy**:
- Bet on team with highest positive edge
- Configurable threshold (`BET_EDGE_THRESHOLD = 0.0`)
- Unit stake (1.0) per bet
- Calculates realized P&L based on actual outcomes

#### 5.3 `run_sim_eval.py` - Main Evaluation Script

**Entry Point**: Command-line script for running full evaluation pipeline.

**Usage**:
```bash
python scripts/sim_eval/run_sim_eval.py \
    --test-dir data/betting_test \
    --odds betting_odds_v3.json \
    --n-sims 1000

# Output:
# - Console summary statistics
# - Optional: match_evaluation_results.json
```

**Flow**:
1. Load player stats cache
2. Load XGBoostModelV2 with stats provider
3. Create simulation engine
4. Load test matches and betting odds
5. Run evaluator
6. Print summary
7. Optionally save detailed results

**Output Summary**:
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

--- Calibration Analysis ---
Predicted probability vs Actual win rate:
(Perfect calibration: predicted = actual)
  Predicted: 45.2%, Actual: 42.1%, Diff: -3.1% (n=18)
  Predicted: 55.8%, Actual: 58.3%, Diff: +2.5% (n=24)
  Predicted: 67.3%, Actual: 65.0%, Diff: -2.3% (n=20)

--- Predictions by Signed Edge ---
(Top 10 matches by prediction quality)
...
```

---

## Data Flows

### Flow 1: Training Pipeline (End-to-End)

```
START
  │
  ├──► 1. Collect Raw Data
  │      Files: data/t20s_json/*.json (15,000+ matches)
  │      Format: Cricsheet JSON (ball-by-ball)
  │      Date range: 2010-2024
  │
  ├──► 2. Feature Engineering (parsing_v2.py)
  │      Process: PlayerStatsTracker accumulates stats chronologically
  │      │
  │      ├─► For each match (sorted by date):
  │      │     │
  │      │     ├─► Take snapshot of current stats (for simulations)
  │      │     │   Store in: stats_snapshots[match_date]
  │      │     │
  │      │     ├─► For each ball:
  │      │     │     ├─► Extract raw state (score, wickets, balls)
  │      │     │     ├─► Get player stats BEFORE ball (from tracker)
  │      │     │     ├─► Calculate basic features (run_rate, phase, etc.)
  │      │     │     ├─► Calculate momentum features (last N balls)
  │      │     │     ├─► Calculate pressure features (dots, boundaries)
  │      │     │     ├─► Create feature vector (29 features)
  │      │     │     ├─► Record outcome (0,1,2,4,6,wicket)
  │      │     │     └─► Update tracker AFTER ball
  │      │     │
  │      │     └─► Assign to temporal split:
  │      │           - train: < 2023
  │      │           - validation: 2022-2024
  │      │           - test: 2024
  │      │           - golden_test: Oct 2024+
  │      │
  │      └─► Outputs:
  │            ├─► data/xgb_data/cricket_data_v2_train.parquet
  │            ├─► data/xgb_data/cricket_data_v2_validation.parquet
  │            ├─► data/xgb_data/cricket_data_v2_test.parquet
  │            └─► models/cache_chunks/*.pkl (lazy-loaded)
  │
  ├──► 3. Model Training (xgboost_v2.py)
  │      │
  │      ├─► Load parquet files
  │      ├─► Encode players (fit on all unique IDs)
  │      ├─► Filter & remap classes (0,1,2,4,6,7 → 0,1,2,3,4,5)
  │      ├─► Calculate class weights (balanced)
  │      │
  │      ├─► Optuna Hyperparameter Tuning:
  │      │     ├─► For each trial (50 total):
  │      │     │     ├─► Sample hyperparameters
  │      │     │     ├─► Train XGBoost with sample weights
  │      │     │     ├─► Evaluate on validation set
  │      │     │     └─► Return validation log loss
  │      │     │
  │      │     └─► Select best parameters
  │      │
  │      ├─► Train final model with best params
  │      ├─► Evaluate on test set
  │      │
  │      └─► Save artifacts:
  │            ├─► models/xgb/xgboost_model_v2.pkl
  │            ├─► models/xgb/batter_encoder_v2.pkl
  │            ├─► models/xgb/bowler_encoder_v2.pkl
  │            ├─► models/xgb/feature_columns_v2.txt
  │            └─► models/xgb/optuna_study_v2.pkl
  │
END (Training Complete)
```

### Flow 2: Simulation Pipeline (Match Prediction)

```
START
  │
  ├──► 1. Load Model & Dependencies
  │      ├─► Load cache chunks + metadata (lazy loading)
  │      ├─► Initialize StatsProvider (binary search ready)
  │      ├─► Load XGBoostModelV2 with encoders
  │      ├─► Create T20Rules engine
  │      └─► Create SimulationEngine
  │
  ├──► 2. Prepare Match (loaders.py)
  │      ├─► Load match JSON
  │      ├─► Extract teams, venue, date
  │      ├─► Extract player lineups (batting order)
  │      └─► Create initial MatchState
  │
  ├──► 3. Run Monte Carlo Simulations (sim_v1_2.py)
  │      │
  │      Config: n_simulations=1000, parallel=True
  │      │
  │      ├─► For each simulation (parallelized):
  │      │     │
  │      │     ├─► Copy initial state
  │      │     ├─► Set random seed (for reproducibility)
  │      │     │
  │      │     ├─► INNINGS 1:
  │      │     │     │
  │      │     │     └─► While not innings_over:
  │      │     │           │
  │      │     │           ├─► Extract 29 features:
  │      │     │           │     ├─► Basic (from state)
  │      │     │           │     ├─► Player encodings
  │      │     │           │     ├─► Player stats (via StatsProvider)
  │      │     │           │     ├─► Momentum (from history)
  │      │     │           │     └─► Pressure (from history)
  │      │     │           │
  │      │     │           ├─► Model predicts probabilities:
  │      │     │           │     {'dot': 0.32, 'one': 0.39, ..., 'wicket': 0.05}
  │      │     │           │
  │      │     │           ├─► Sample outcome (weighted random)
  │      │     │           │
  │      │     │           ├─► Validate outcome (legal?)
  │      │     │           │
  │      │     │           ├─► Process ball:
  │      │     │           │     ├─► Update runs
  │      │     │           │     ├─► Update wickets
  │      │     │           │     ├─► Rotate strike (if odd runs)
  │      │     │           │     ├─► Handle wickets (new batsman)
  │      │     │           │     ├─► Track bowler overs
  │      │     │           │     └─► Add to history
  │      │     │           │
  │      │     │           ├─► Check end of over:
  │      │     │           │     ├─► Rotate strike
  │      │     │           │     └─► Select new bowler
  │      │     │           │
  │      │     │           └─► Check innings over:
  │      │     │                 - 10 wickets down?
  │      │     │                 - 120 balls bowled?
  │      │     │
  │      │     ├─► INNINGS 2:
  │      │     │     │
  │      │     │     ├─► Set target (innings1_runs + 1)
  │      │     │     ├─► Reset state (balls=0, new batsmen/bowler)
  │      │     │     │
  │      │     │     └─► While not innings_over:
  │      │     │           ├─► [Same ball loop as innings 1]
  │      │     │           └─► Additional check: target achieved?
  │      │     │
  │      │     ├─► Determine winner:
  │      │     │     - Compare final scores
  │      │     │     - Calculate margin
  │      │     │
  │      │     └─► Return MatchResult
  │      │
  │      └─► Aggregate results (1000 simulations):
  │            ├─► Win probability: team1_wins / n_sims
  │            ├─► Score distributions (mean, std, percentiles)
  │            ├─► Wicket distributions
  │            └─► Raw results for detailed analysis
  │
END (Predictions Generated)
```

### Flow 3: Evaluation Pipeline (Against Betting Markets)

```
START
  │
  ├──► 1. Load Evaluation Data
  │      ├─► Test matches JSON (data/betting_test/)
  │      ├─► Betting odds JSON (betting_odds_v3.json)
  │      ├─► Model & stats provider (from training)
  │      └─► Create evaluator
  │
  ├──► 2. For Each Match:
  │      │
  │      ├─► A. Run Simulations (1000x):
  │      │     ├─► Create MatchState
  │      │     ├─► Run Monte Carlo
  │      │     └─► Get win probabilities & score distributions
  │      │
  │      ├─► B. Extract Market Data:
  │      │     ├─► Decimal odds: {'India': 2.10, 'Australia': 1.75}
  │      │     ├─► Convert to probabilities: {'India': 0.476, 'Australia': 0.524}
  │      │     └─► Remove bookmaker margin (normalize to 1.0)
  │      │
  │      ├─► C. Calculate Metrics:
  │      │     │
  │      │     ├─► Log Loss:
  │      │     │     If actual_winner = 'India':
  │      │     │       log_loss = -log(simulated_prob['India'])
  │      │     │
  │      │     ├─► Brier Score:
  │      │     │     brier = (simulated_prob['India'] - actual)²
  │      │     │     where actual = 1 if India won, else 0
  │      │     │
  │      │     ├─► Edge (per team):
  │      │     │     edge['India'] = simulated_prob['India'] - market_prob['India']
  │      │     │     Positive = value bet opportunity
  │      │     │
  │      │     └─► Realized P&L (if betting):
  │      │           ├─► Find team with max positive edge
  │      │           ├─► If edge > threshold:
  │      │           │     - Place unit bet on that team
  │      │           │     - If won: return (odds - 1)
  │      │           │     - If lost: return -1
  │      │           └─► Else: no bet (return 0)
  │      │
  │      └─► Store MatchEvaluationResult
  │
  ├──► 3. Aggregate Results:
  │      │
  │      ├─► Average Metrics:
  │      │     ├─► Mean log loss across matches
  │      │     ├─► Mean Brier score
  │      │     ├─► Mean absolute edge (magnitude)
  │      │     └─► Mean signed edge (prediction quality)
  │      │
  │      ├─► Betting Performance:
  │      │     ├─► Total P&L = sum(realized_pnl)
  │      │     ├─► ROI = (total_pnl / bets_placed) × 100
  │      │     ├─► Win rate = winning_bets / bets_placed
  │      │     └─► Count profitable opportunities
  │      │
  │      └─► Calibration Analysis:
  │            ├─► Bin predictions (0-10%, 10-20%, ..., 90-100%)
  │            ├─► For each bin:
  │            │     ├─► Average predicted probability
  │            │     ├─► Actual win rate
  │            │     └─► Difference (calibration error)
  │            └─► Plot predicted vs actual
  │
  ├──► 4. Generate Reports:
  │      ├─► Console summary (formatted table)
  │      ├─► Top 10 predictions by signed edge
  │      ├─► Calibration plot (text)
  │      └─► Optional: JSON export (match_evaluation_results.json)
  │
END (Evaluation Complete)
```

---

## Key Classes & Interfaces

### Abstract Base Classes (ABCs)

#### `PredictionModel(ABC)`
```python
from abc import ABC, abstractmethod

class PredictionModel(ABC):
    @abstractmethod
    def extract_features(self, state: MatchState) -> Any:
        """Extract features from match state

        Returns:
            Features in format expected by predict_next_ball()
            (Can be np.ndarray or pd.DataFrame depending on implementation)
        """
        pass

    @abstractmethod
    def predict_next_ball(self, features: Any) -> Dict[str, float]:
        """Predict outcome probabilities

        Args:
            features: Output from extract_features()

        Returns:
            Dict mapping outcome names to probabilities:
            {'dot': 0.32, 'one': 0.39, 'two': 0.08, 'four': 0.10,
             'six': 0.04, 'wicket': 0.05, 'wide': 0.01, 'no_ball': 0.01}

            Must sum to ~1.0
        """
        pass
```

**Implementations**:
- `XGBoostModel`: V1 (9 features, no temporal stats)
- `XGBoostModelV2`: V2 (29 features, with StatsProvider)
- `DummyModel`: Testing (fixed probabilities)

**Usage Pattern**:
```python
# Any PredictionModel can be used with SimulationEngine
model = XGBoostModelV2(...)  # or XGBoostModel(...) or DummyModel()
engine = SimulationEngine(model, T20Rules())
results = engine.simulate_match(state)
```

#### `BowlerSelector(ABC)`
```python
class BowlerSelector(ABC):
    @abstractmethod
    def select_bowler(self, state: MatchState, available: List[int]) -> int:
        """Select bowler for next over

        Args:
            state: Current match state
            available: List of available bowler indices

        Returns:
            Index of selected bowler (0-10)
        """
        pass
```

**Implementations**:
- `RandomBowlerSelector`: Random choice (current default)

**Future Implementations** (TODOs):
- `SmartBowlerSelector`: Based on matchups, overs remaining, etc.
- `HistoricalBowlerSelector`: Based on actual bowling data

---

### Core Data Classes

#### `Player`
```python
@dataclass
class Player:
    player_id: str      # Unique ID (matches training data)
    name: str           # Display name
    team: str           # Team name
    role: str = "allrounder"  # "batsman", "bowler", "allrounder", "wicketkeeper"
```

#### `TeamLineup`
```python
@dataclass
class TeamLineup:
    team_name: str
    players: List[Player]  # Exactly 11 players in batting order

    def get_player_by_index(self, idx: int) -> Optional[Player]:
        """Get player at batting position idx (0-10)"""
```

#### `MatchState`
```python
@dataclass
class MatchState:
    # Immutable setup
    team1_lineup: TeamLineup
    team2_lineup: TeamLineup
    batting_first: str       # Must match team name
    venue: str
    match_date: datetime

    # Dynamic state (defaults)
    innings: int = 1         # 1 or 2
    balls: int = 0           # 0-119 for T20
    runs: np.ndarray = field(default_factory=lambda: np.zeros(2))
    wickets: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=int))

    # Player tracking
    striker_idx: int = 0     # 0-10 (batting order position)
    non_striker_idx: int = 1
    bowler_idx: int = 0
    last_bowler_idx: int = -1
    batsmen_out: Dict[int, List[int]]  # team_idx -> [dismissed_indices]

    # History tracking
    history: np.ndarray      # Ball-by-ball records (Nx9)
    history_idx: int = 0
    bowler_balls: Dict[Tuple[int, int], int]  # (team, player) -> balls bowled
    batsman_stats: Dict[Tuple[int, int], Tuple[int, int]]  # (team, player) -> (runs, balls)

    # Computed properties (many @property methods)
    @property
    def current_team_idx(self) -> int: ...
    @property
    def batting_team(self) -> str: ...
    @property
    def overs_completed(self) -> float: ...
    @property
    def target(self) -> Optional[int]: ...
    @property
    def required_run_rate(self) -> Optional[float]: ...
    # ... 10+ more properties

    # Key methods
    def update(self, outcome: Outcome, runs: int): ...
    def is_innings_over(self) -> bool: ...
    def is_match_over(self) -> bool: ...
    def get_next_batsman_idx(self) -> int: ...
    def get_available_bowlers(self) -> List[int]: ...
    def copy(self) -> MatchState: ...
```

**State Transitions**:
```
Initial State
    ↓
Ball Bowled → outcome sampled → state.update(outcome, runs)
    ↓
Update runs, wickets, strike rotation, history
    ↓
Check: Over complete? → select new bowler
    ↓
Check: Innings over? → start_new_innings() or end match
    ↓
Repeat
```

---

### Result Classes

#### `BallResult`
```python
@dataclass
class BallResult:
    innings: int          # 1 or 2
    over: int             # 0-19
    ball: int             # 0-5
    outcome: Outcome      # Enum value
    runs: int             # Runs scored
    striker_idx: int      # Who faced
    bowler_idx: int       # Who bowled
    team_runs: int        # Running total
    team_wickets: int     # Running wickets
```

#### `InningsResult`
```python
@dataclass
class InningsResult:
    batting_team: str
    bowling_team: str
    total_runs: int
    total_wickets: int
    total_balls: int
    run_rate: float

    # Performance cards
    batting_card: Dict[int, Tuple[int, int, int, int]]  # idx -> (runs, balls, 4s, 6s)
    bowling_card: Dict[int, Tuple[int, int, int]]       # idx -> (balls, runs, wickets)

    # Ball-by-ball
    balls: List[BallResult]
```

#### `MatchResult`
```python
@dataclass
class MatchResult:
    match_id: str
    team1: str
    team2: str
    winner: str          # "Team1", "Team2", or "Tie"
    margin: str          # "23 runs" or "5 wickets"

    innings: List[InningsResult]  # 1 or 2 innings

    # Quick access
    team1_score: int
    team1_wickets: int
    team2_score: int
    team2_wickets: int
```

---

### Evaluation Classes

#### `MatchEvaluationResult`
```python
@dataclass
class MatchEvaluationResult:
    match_id: str
    team1: str
    team2: str

    # Simulation outputs
    simulated_win_prob: Dict[str, float]
    simulated_scores: Dict[str, Dict[str, float]]

    # Market data
    market_win_prob: Dict[str, float]
    market_odds: Dict[str, float]

    # Actual outcome
    actual_winner: Optional[str]

    # Metrics
    log_loss: float
    brier_score: float
    edge: Dict[str, float]      # team -> edge
    realized_pnl: Optional[float]

    # Metadata
    n_simulations: int
    simulation_time: float
```

#### `OverallEvaluationResults`
```python
@dataclass
class OverallEvaluationResults:
    n_matches: int

    # Aggregate metrics
    avg_log_loss: float
    avg_brier_score: float
    avg_edge: float              # Unsigned (magnitude)
    avg_signed_edge: float       # Signed (quality)
    profitable_bets: int

    # Betting performance
    total_pnl: float
    roi: float
    win_rate: float
    bets_placed: int

    # Calibration
    calibration_bins: List[Tuple[float, float, int]]  # (predicted, actual, n)

    # Details
    match_results: List[MatchEvaluationResult]
    total_simulation_time: float
```

---

## Entry Points

### 1. Training Pipeline

**Step 1: Feature Engineering**
```bash
cd /Users/aryamangupta/CricML/Match_Prediction
python scripts/parsing_v2.py
```
- **Input**: `data/t20s_json/*.json` (15,000+ files)
- **Output**:
  - `data/xgb_data/*.parquet` (train/val/test splits)
  - `models/cache_chunks/*.pkl` (69 files, ~7.8GB total)
  - `models/player_stats_cache_metadata.pkl` (~10KB index)
- **Duration**: ~10-15 minutes
- **Memory**: ~4-8 GB

**Step 2: Model Training**
```bash
python scripts/xgboost_v2.py
```
- **Input**: `data/xgb_data/*.parquet`
- **Output**: `models/xgb/*` (model + encoders + metadata)
- **Duration**: ~30-60 minutes (with Optuna)
- **Memory**: ~8-16 GB

### 2. Simulation (Standalone)

```python
from scripts.sim_v1_2 import *
from scripts.stats_provider import StatsProvider

# Load components
stats_provider = StatsProvider('models')
model = XGBoostModelV2(
    'models/xgb/xgboost_model_v2.pkl',
    'models/xgb/batter_encoder_v2.pkl',
    'models/xgb/bowler_encoder_v2.pkl',
    'models/xgb/feature_columns_v2.txt',
    stats_provider=stats_provider
)
engine = SimulationEngine(model, T20Rules())

# Create match
india = TeamLineup("India", [Player(...), ...])  # 11 players
australia = TeamLineup("Australia", [Player(...), ...])

state = MatchState(
    team1_lineup=india,
    team2_lineup=australia,
    batting_first="India",
    venue="MCG",
    match_date=datetime(2024, 6, 15)
)

# Simulate
config = SimulationConfig(n_simulations=1000, parallel=True)
results = engine.simulate_multiple(state, config)
summary = ResultAggregator.aggregate(results)

print(f"India: {summary['win_probability']['India']:.1%}")
```

### 3. Evaluation Pipeline

```bash
python scripts/sim_eval/run_sim_eval.py \
    --test-dir data/betting_test \
    --odds betting_odds_v3.json \
    --n-sims 1000
```
- **Input**:
  - Test matches JSON
  - Betting odds JSON
  - Trained model
- **Output**:
  - Console summary
  - Optional: `match_evaluation_results.json`
- **Duration**: ~10-30 seconds per match

---

## Design Decisions

### 1. Temporal Data Integrity (CRITICAL)

**Problem**: Player stats in training vs simulation must match temporal reality.

**Solution**: Player stats cache with date-indexed snapshots.

**Implementation**:
- During training: Take snapshot BEFORE processing each match
- During simulation: Binary search for most recent snapshot ≤ match_date
- Ensures no data leakage: simulation uses only historically available info

**Code Location**: `parsing_v2.py` (line 481-486), `stats_provider.py` (line 57-78)

**Why This Matters**:
```
BAD (data leakage):
  Simulation on 2024-06-15 uses player's full 2024 stats
  → Model sees future performance

GOOD (temporal integrity):
  Simulation on 2024-06-15 uses player's stats as of 2024-06-14
  → Model sees only historical performance
```

### 2. Feature Engineering Order

**Decision**: Extract features BEFORE ball, update stats AFTER ball.

**Rationale**: Model should only know what's observable before prediction.

**Code**: `parsing_v2.py` (line 292-333)

```python
# 1. Extract features (uses tracker state BEFORE ball)
batting_features = tracker.get_batting_features(batter_id)

# 2. Record feature vector + outcome
ball_record = {...features..., 'ball_outcome': ...}

# 3. Update tracker (now tracker state is AFTER ball)
tracker.update_stats(batter_id, bowler_id, runs, is_wicket)
```

### 3. Class Remapping

**Decision**: Normalize rare outcomes (3,5,7+ runs) to common classes.

**Rationale**:
- 3-run balls: ~0.5% of data → combine with 2-run
- 5-run balls: ~0.1% of data → combine with 4-run
- 7+ run balls: ~0.01% of data → combine with 6-run
- Reduces class imbalance without losing semantic meaning

**Code**: `parsing_v2.py` (line 147-163), `xgboost_v2.py` (line 150-165)

### 4. Parallel Simulation

**Decision**: Use multiprocessing for Monte Carlo simulations.

**Rationale**:
- Simulations are independent (no shared state)
- ~1000 sims per match → 10-30 seconds single-threaded
- ~4x speedup with 4 cores

**Implementation**: Each worker gets:
- Copy of initial MatchState
- Unique random seed (for reproducibility)
- Independent prediction model (must be pickle-serializable)

**Code**: `sim_v1_2.py` (line 986-1010)

### 5. Betting Edge Calculation

**Decision**: Remove bookmaker margin for fair comparison.

**Rationale**: Bookmaker odds include ~5-10% margin (overround). To compare our probabilities to "true" market probabilities, we normalize implied probabilities to sum to 1.0.

**Example**:
```python
# Raw odds
odds = {'India': 2.10, 'Australia': 1.75}

# Implied probabilities (with margin)
raw_probs = {'India': 1/2.10=0.476, 'Australia': 1/1.75=0.571}
# Sum = 1.047 (4.7% margin)

# Normalized (margin-free)
fair_probs = {'India': 0.476/1.047=0.455, 'Australia': 0.571/1.047=0.545}
# Sum = 1.0
```

**Code**: `loaders.py` (line 181-216)

### 6. Model Output Handling

**Decision**: XGBoost outputs 6 classes, add small extras probability.

**Rationale**:
- Training data doesn't distinguish extra types well
- Extras are ~5% of balls but highly variable
- Add fixed 1% wide, 1% no-ball, normalize to 1.0

**Code**: `sim_v1_2.py` (line 636-644)

```python
# Model outputs probs for: dot, one, two, four, six, wicket
outcome_probs = {...}

# Add extras (not trained)
outcome_probs['wide'] = 0.01
outcome_probs['no_ball'] = 0.01

# Normalize to sum=1.0
total = sum(outcome_probs.values())
outcome_probs = {k: v/total for k, v in outcome_probs.items()}
```

### 7. Wicket Handling

**Decision**: No strike rotation on wickets.

**Rationale**: In cricket, when a batsman gets out, the new batsman takes the striker's position. Strike doesn't rotate.

**Code**: `sim_v1_2.py` (line 250-259)

```python
# Handle wicket
if outcome == Outcome.WICKET:
    self.wickets[self.current_team_idx] += 1
    self.batsmen_out[self.current_team_idx].append(self.striker_idx)
    self.striker_idx = self.get_next_batsman_idx()
    # NOTE: No strike rotation here

# Rotate strike (only for odd runs)
if runs % 2 == 1:
    self.striker_idx, self.non_striker_idx = self.non_striker_idx, self.striker_idx
```

### 8. Signed Edge Metric

**Decision**: Track "signed edge" in addition to absolute edge.

**Rationale**:
- Absolute edge: Magnitude of disagreement with market
- Signed edge: Prediction quality (positive if correct, negative if wrong)

**Example**:
```python
Model: India 70%, Market: India 55%
Edge: +15%

If India wins:
  Signed edge = +15% (correct + high confidence)

If India loses:
  Signed edge = -15% (wrong + high confidence)
```

**Code**: `match_evaluator.py` (line 463-502)

---

## Data Formats

### 1. Raw Match JSON (Cricsheet Format)

**Location**: `data/t20s_json/*.json`, `data/betting_test/*.json`

**Structure**:
```json
{
  "info": {
    "teams": ["India", "Australia"],
    "dates": ["2024-06-15"],
    "venue": "Melbourne Cricket Ground",
    "toss": {
      "winner": "India",
      "decision": "bat"
    },
    "registry": {
      "people": {
        "Rohit Sharma": "253802",
        "Virat Kohli": "253802",
        ...
      }
    }
  },
  "innings": [
    {
      "team": "India",
      "overs": [
        {
          "over": 0,
          "deliveries": [
            {
              "batter": "Rohit Sharma",
              "bowler": "Mitchell Starc",
              "non_striker": "Shubman Gill",
              "runs": {
                "batter": 4,
                "extras": 0,
                "total": 4
              },
              "wickets": null
            },
            ...
          ]
        },
        ...
      ]
    },
    {
      "team": "Australia",
      "overs": [...]
    }
  ]
}
```

### 2. Processed Parquet Files

**Location**: `data/xgb_data/cricket_data_v2_*.parquet`

**Columns** (29 features + metadata + target):
```
Metadata:
  - innings_id: str (unique innings identifier)
  - inning_idx: int (1 or 2)
  - over_idx: int (0-19)
  - ball_idx: int (0-119)

Raw State:
  - batter_id: str
  - non_striker_id: str
  - bowler_id: str
  - score: int (team total)
  - wickets: int (wickets fallen)
  - balls_bowled: int

Features:
  - [29 features as described in parsing_v2.py section]

Target:
  - ball_outcome: int (0,1,2,4,6,-1 for wicket)
```

**Row Count**:
- Train: ~3.5M balls
- Validation: ~400K balls
- Test: ~300K balls

### 3. Player Stats Cache

**Location**: `models/cache_chunks/` (69 chunk files) + `models/player_stats_cache_metadata.pkl`

**Structure** (per chunk file):
```python
{
    'snapshots': {
        '2020-01-15': {
            'batting': {
                'player_id_1': {'runs': 450, 'balls': 320, 'dismissals': 12},
                'player_id_2': {'runs': 890, 'balls': 650, 'dismissals': 22},
                ...
            },
            'bowling': {
                'player_id_1': {'runs_given': 1250, 'balls_bowled': 540, 'wickets': 18},
                ...
            },
            'h2h': {
                ('batter_1', 'bowler_1'): {'runs': 45, 'balls': 32, 'dismissals': 1},
                ...
            }
        },
        '2020-01-16': {...},
        # ... ~5000 date snapshots
    },
    'metadata': {
        'num_matches': 15000,
        'num_dates': 5000,
        'num_players_batting': 3000,
        'num_players_bowling': 2500,
        'num_h2h_matchups': 50000,
        'build_timestamp': '2024-09-27T10:41:00'
    }
}
```

**Total Size**: ~7.8GB (69 chunks × ~110MB each, lazy-loaded)

### 4. Betting Odds JSON

**Location**: `betting_odds_v3.json`

**Structure**:
```json
{
  "matches": [
    {
      "match_id": "2024-06-15_India_Australia_MCG",
      "date": "2024-06-15",
      "team1": "India",
      "team2": "Australia",
      "venue": "MCG",
      "odds": {
        "winner": {
          "India": 2.10,
          "Australia": 1.75,
          "timestamp": "2024-06-14T10:00:00Z"
        }
      },
      "actual_winner": "India"
    },
    ...
  ]
}
```

**Odds Format**: Decimal odds (e.g., 2.10 = bet $1 to win $2.10 total)

### 5. Evaluation Results JSON

**Location**: `match_evaluation_results.json`

**Structure**:
```json
{
  "summary": {
    "n_matches": 45,
    "avg_log_loss": 0.6234,
    "avg_brier_score": 0.1845,
    "avg_edge": 0.083,
    "profitable_bets": 28,
    "total_time": 450.2
  },
  "matches": [
    {
      "match_id": "2024-06-15_India_Australia_MCG",
      "teams": ["India", "Australia"],
      "simulated_prob": {"India": 0.653, "Australia": 0.347},
      "market_prob": {"India": 0.476, "Australia": 0.524},
      "edge": {"India": 0.177, "Australia": -0.177},
      "log_loss": 0.426,
      "brier_score": 0.121
    },
    ...
  ]
}
```

---

## Common Operations

### Operation 1: Re-train Model with New Data

```bash
# 1. Add new JSON files to data/t20s_json/
cp new_matches/*.json data/t20s_json/

# 2. Re-run feature engineering
python scripts/parsing_v2.py
# → Updates parquet files and cache chunks

# 3. Re-train model
python scripts/xgboost_v2.py
# → Updates models/xgb/* files

# 4. Verify performance
python scripts/sim_eval/run_sim_eval.py --test-dir data/betting_test --odds betting_odds_v3.json
```

### Operation 2: Simulate a New Match

```python
from scripts.sim_v1_2 import *
from scripts.stats_provider import StatsProvider
from datetime import datetime

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

# Define teams (ensure player_ids match training data)
india_players = [
    Player("253802", "Rohit Sharma", "India", "batsman"),
    Player("277906", "Virat Kohli", "India", "batsman"),
    # ... 11 total
]
australia_players = [
    Player("219889", "David Warner", "Australia", "batsman"),
    # ... 11 total
]

india = TeamLineup("India", india_players)
australia = TeamLineup("Australia", australia_players)

# Create match state
state = MatchState(
    team1_lineup=india,
    team2_lineup=australia,
    batting_first="India",
    venue="MCG",
    match_date=datetime(2024, 11, 15)  # Must be ≤ latest cache date
)

# Simulate
config = SimulationConfig(n_simulations=1000, parallel=True, verbose=True)
results = engine.simulate_multiple(state, config)

# Analyze
summary = ResultAggregator.aggregate(results)
print(f"\nWin Probabilities:")
print(f"  India: {summary['win_probability']['India']:.1%}")
print(f"  Australia: {summary['win_probability']['Australia']:.1%}")
print(f"\nExpected Scores:")
print(f"  India: {summary['score_stats']['India']['mean']:.0f} ± {summary['score_stats']['India']['std']:.0f}")
print(f"  Australia: {summary['score_stats']['Australia']['mean']:.0f} ± {summary['score_stats']['Australia']['std']:.0f}")
```

### Operation 3: Evaluate Single Match

```python
from scripts.sim_eval.match_evaluator import MatchLevelEvaluator
from scripts.sim_eval.loaders import TestMatchLoader, BettingOddsLoader

# Load model & engine (as in Operation 2)

# Load match
loader = TestMatchLoader()
matches = loader.load_matches('data/betting_test/')
match_id, match_state = matches[0]  # First match

# Load odds
odds_lookup = BettingOddsLoader.load_odds('betting_odds_v3.json')

# Evaluate
evaluator = MatchLevelEvaluator(model, engine, n_simulations=1000)
result = evaluator._evaluate_single_match(match_id, match_state, odds_lookup[match_id])

# Print results
print(f"Match: {match_id}")
print(f"Simulated: {result.simulated_win_prob}")
print(f"Market: {result.market_win_prob}")
print(f"Edge: {result.edge}")
print(f"Log Loss: {result.log_loss:.3f}")
print(f"Actual Winner: {result.actual_winner}")
```

### Operation 4: Inspect Player Stats Cache

```python
from scripts.stats_provider import StatsProvider

provider = StatsProvider('models')

# Query specific player
batting = provider.get_batting_stats('253802', '2024-06-15')
print(f"Batting Average: {batting['avg']:.2f}")
print(f"Strike Rate: {batting['sr']:.2f}")

bowling = provider.get_bowling_stats('253802', '2024-06-15')
print(f"Bowling Average: {bowling['avg']:.2f}")
print(f"Economy: {bowling['econ']:.2f}")

# Inspect metadata
print(f"\nCache Coverage:")
print(f"  Dates: {provider.dates[0]} to {provider.dates[-1]}")
print(f"  Total snapshots: {len(provider.dates):,}")
print(f"  Players: {provider.metadata['num_players_batting']:,}")
```

### Operation 5: Debug Simulation Issues

```python
# Enable verbose mode
config = SimulationConfig(
    n_simulations=10,  # Fewer for debugging
    parallel=False,    # Sequential for traceback
    verbose=True,      # Print progress
    random_seed=42     # Reproducible
)

# Run single simulation with debugging
try:
    result = engine.simulate_match(state, "debug_match")

    # Inspect ball-by-ball
    innings1 = result.innings[0]
    for i, ball in enumerate(innings1.balls[:20]):  # First 20 balls
        print(f"Ball {i}: {ball.outcome.name} → {ball.runs} runs "
              f"(Score: {ball.team_runs}/{ball.team_wickets})")

except Exception as e:
    import traceback
    traceback.print_exc()

    # Inspect state at failure
    print(f"\nState at failure:")
    print(f"  Innings: {state.innings}")
    print(f"  Balls: {state.balls}")
    print(f"  Score: {state.runs}")
    print(f"  Wickets: {state.wickets}")
```

---

## Module Dependencies

```
parsing_v2.py
  ├─► (no dependencies on other project modules)
  └─► External: pandas, numpy, pathlib, collections, datetime

xgboost_v2.py
  ├─► Reads: data/xgb_data/*.parquet (from parsing_v2.py)
  └─► External: xgboost, pandas, numpy, sklearn, joblib, optuna

stats_provider.py
  ├─► Reads: models/cache_chunks/*.pkl + metadata (from parsing_v2.py)
  └─► External: pickle, datetime, bisect

sim_v1_2.py
  ├─► Reads: models/xgb/* (from xgboost_v2.py)
  ├─► Reads: models/cache_chunks/*.pkl (lazy-loaded via stats_provider.py)
  ├─► Uses: stats_provider.py (optional, for XGBoostModelV2)
  └─► External: numpy, pandas, dataclasses, enum, multiprocessing, joblib

sim_eval/loaders.py
  ├─► Uses: sim_v1_2.py (MatchState, Player, TeamLineup)
  └─► External: json, pathlib, datetime

sim_eval/match_evaluator.py
  ├─► Uses: sim_v1_2.py (SimulationEngine, SimulationConfig, MatchState)
  ├─► Uses: sim_eval/loaders.py (BettingOddsLoader)
  └─► External: numpy, dataclasses

sim_eval/run_sim_eval.py
  ├─► Uses: sim_v1_2.py (all classes)
  ├─► Uses: stats_provider.py
  ├─► Uses: sim_eval/loaders.py
  ├─► Uses: sim_eval/match_evaluator.py
  └─► External: argparse, json
```

**Execution Order**:
1. `parsing_v2.py` (independent, creates data)
2. `xgboost_v2.py` (depends on parquet files from step 1)
3. `stats_provider.py` + `sim_v1_2.py` (depends on cache + model from steps 1-2)
4. `sim_eval/*` (depends on everything)

---

## Recent Changes (Git Log)

```
b7b3772 Fix critical metric calculation issues
        - Fixed signed edge calculation logic
        - Improved calibration binning

65b6231 Add signed edge sorting to show prediction quality
        - Show top 10 predictions by quality (correct + confident)
        - Distinguish correct/incorrect predictions with high edge

ee28717 Fix critical bugs and improve simulation evaluation metrics
        - Data type validation in simulation results
        - Improved metric aggregation

6606a93 using optuna for hyperparams
        - Integrated Optuna for hyperparameter tuning
        - 50 trial optimization

fb8c0d4 added a working xgboost model with more features implemented
        - Expanded to 29 features
        - Added player stats, momentum, pressure features

af553be working betting simulation and evalutation
        - Initial betting evaluation framework

b217697 added betting odds for evaluation
        - Betting odds scraping and storage
```

---

## Quick Reference

### File Paths Cheat Sheet
```
# Training
data/t20s_json/                              # Raw matches
data/xgb_data/cricket_data_v2_train.parquet  # Processed training data
models/xgb/xgboost_model_v2.pkl              # Trained model
models/cache_chunks/                         # Temporal stats (69 chunks, 7.8GB)
models/player_stats_cache_metadata.pkl       # Cache index

# Simulation
scripts/sim_v1_2.py                          # Main simulation engine
scripts/stats_provider.py                    # Stats access
scripts/sim_eval/run_sim_eval.py             # Evaluation script

# Evaluation
data/betting_test/                           # Test matches
betting_odds_v3.json                         # Market odds
match_evaluation_results.json                # Results output
```

### Key Constants
```python
# Simulation
T20_TOTAL_BALLS = 120  # 20 overs × 6 balls
MAX_WICKETS = 10
MAX_BOWLER_OVERS = 4   # 24 balls

# Evaluation
BET_EDGE_THRESHOLD = 0.0  # Minimum edge to bet

# Model
N_FEATURES_V2 = 29
N_CLASSES = 6  # After remapping
```

### Performance Benchmarks
```
Feature Engineering: ~10-15 min for 15K matches
Model Training: ~30-60 min with Optuna (50 trials)
Single Match Simulation: ~0.01-0.1 sec (1000 simulations, parallel)
Full Evaluation: ~10-30 sec per match
Stats Cache Load: ~2 seconds
```

---

**End of Reference Guide**

*This document is automatically updated with each major change. For the latest version, check git history.*
