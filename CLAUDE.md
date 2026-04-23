# CricML Match Prediction System

**Agent Guide** - Concise overview for AI agents working on this codebase

**Last Updated**: March 2025
**Branch**: features/transformer-model
**Purpose**: Portfolio project - ML system for T20 cricket match prediction

---

## Project Overview

A production-scale machine learning system that predicts T20 cricket match outcomes by:

1. **Predicting individual ball outcomes** (dot, 1, 2, 4, 6, wicket) using XGBoost or LSTM
2. **Simulating complete matches** via Monte Carlo methods (1000+ iterations)
3. **Evaluating against betting markets** to measure prediction quality

**Core Innovation**: Rather than directly predicting match winners (limited data), we predict individual balls (millions of examples) and simulate full matches to generate probabilistic forecasts.

**Pipeline**: Raw JSON → Feature Engineering (46+ features) → Model Training (XGBoost/LSTM) → Ball Predictions → Monte Carlo Simulation → Match Probabilities

**Model Types**:
- **XGBoost** (default): Gradient boosted trees, fast inference, ~55-60% accuracy
- **LSTM** (v1): PyTorch recurrent model with 10-ball sliding window sequence context
- **Transformer** (v1): PyTorch transformer with full 120-ball innings context

**Model Versions**:
- **v2** (legacy): 29 features, basic player stats
- **v3** (current): 46+ features with player metadata (hand, arm, age, matchup types, type-based stats)

---

## Quick Start

> **Important**: Use `uv run` to execute all scripts. This ensures the correct virtual environment and dependencies are used.

```bash
# Full training pipeline (v3 model with player metadata).
# Phase B split (2026-04-22): parsing is two scripts now — the stats cache is
# built once (schema v3 SQLite) and the parquet is materialized from it +
# the JSONs. run_experiment.py dispatches both automatically.

# --- Option A: run everything via run_experiment.py (recommended) ---
uv run python scripts/run_experiment.py experiments/configs/xgb_v3_baseline.yaml
# Dispatches cache-build + materialize + train + eval, skipping any step
# whose artifact is already current per check_smart_cache().

# --- Option B: run the new parsing scripts directly ---
uv run python scripts/build_stats_cache.py          # JSON → SQLite v3 (~6 min, 46 MB DB)
uv run python scripts/materialize_features.py       # SQLite + JSON → parquet (~3 min)
uv run python scripts/xgboost_v2.py                 # ~5-10 min
# build_stats_cache.py is idempotent — it no-ops if the JSONs haven't
# changed since the last build (compares _meta.source_json_mtime_max).
# Force with --force-rebuild if needed.

# --- Optuna hyperparameter tuning ---
uv run python scripts/xgboost_v2.py --tune --n-trials 50  # ~30-60 min

# Run evaluation (defaults to XGBoost v3 model)
uv run python scripts/sim_eval/run_sim_eval.py \
    --test-dir data/betting_test \
    --odds betting_odds_v3.json \
    --n-sims 1000

# Run evaluation with v2 legacy model
uv run python scripts/sim_eval/run_sim_eval.py \
    --model-version v2 \
    --test-dir data/betting_test \
    --odds betting_odds_v3.json

# --- LSTM Model ---
# Train LSTM model
uv run python scripts/lstm_v1.py --epochs 50 --batch-size 512

# Run evaluation with LSTM model
uv run python scripts/sim_eval/run_sim_eval.py \
    --model-type lstm \
    --test-dir data/betting_test \
    --odds betting_odds_v3.json \
    --n-sims 1000

# --- MLP Model ---
# Train MLP model (simple neural network baseline)
uv run python scripts/mlp_v1.py --epochs 30 --device cpu

# Run evaluation with MLP model
uv run python scripts/sim_eval/run_sim_eval.py \
    --model-type mlp \
    --test-dir data/betting_test \
    --odds betting_odds_v3.json \
    --n-sims 1000

# --- Transformer Model (Full 120-ball context) ---
# Train Transformer model (reduced batch size due to longer sequences)
uv run python scripts/transformer_v1.py --epochs 50 --batch-size 64

# Run evaluation with Transformer model
uv run python scripts/sim_eval/run_sim_eval.py \
    --model-type transformer \
    --test-dir data/betting_test \
    --odds betting_odds_v3.json \
    --n-sims 1000

# --- MLX Optimization (Apple Silicon M1/M2/M3/M4 only) ---
# Train Transformer with MLX backend (faster on Apple Silicon)
uv run python scripts/transformer_v1.py --mlx --epochs 50 --batch-size 128

# Run evaluation with MLX backend (unified memory, Metal GPU)
uv run python scripts/sim_eval/run_sim_eval.py \
    --model-type transformer \
    --mlx \
    --test-dir data/betting_test \
    --odds betting_odds_v3.json \
    --n-sims 1000

# Convert weights between PyTorch and MLX formats
uv run python scripts/convert_weights.py --input models/transformer_v1/transformer_model_v1.pt --to-mlx
```

**For detailed operations guide, see [docs/OPERATIONS.md](docs/OPERATIONS.md)**

### Experiment-Based Workflow (Recommended)

```bash
# Run a full experiment (parse → train → evaluate) with one command
uv run python scripts/run_experiment.py experiments/configs/xgb_v3_baseline.yaml

# Skip parsing if data hasn't changed
uv run python scripts/run_experiment.py experiments/configs/xgb_v3_baseline.yaml --skip-parsing

# Only re-run evaluation (e.g. after a bug fix)
uv run python scripts/run_experiment.py experiments/configs/xgb_v3_baseline.yaml --only-eval

# Preview what will run without executing
uv run python scripts/run_experiment.py experiments/configs/xgb_v3_baseline.yaml --dry-run

# Run a feature ablation (no code changes needed)
uv run python scripts/run_experiment.py experiments/configs/xgb_v3_no_metadata.yaml --skip-parsing

# List all past experiments
uv run python scripts/compare_experiments.py --list

# Compare two experiments side by side
uv run python scripts/compare_experiments.py <exp_id_1> <exp_id_2>
```

**For how to add new features, models, or data, see [docs/OPERATIONS.md#development-workflows](docs/OPERATIONS.md#development-workflows)**

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                        │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Raw JSON (15K+ matches)                                    │
│       │                                                       │
│       ▼                                                       │
│  build_stats_cache.py + materialize_features.py            │
│   - 46+ features (basic, player stats, momentum, metadata) │
│   - Temporal stats cache (v3: SQLite mmap, schema v3)      │
│       │                                                       │
│       ▼                                                       │
│  Parquet files (train/val/test splits)                     │
│       │                                                       │
│       ▼                                                       │
│  xgboost_v2.py                                              │
│   - 6-class classifier (dot, 1, 2, 4, 6, wicket)          │
│   - Optuna hyperparameter tuning (50 trials)               │
│   - ~55-60% ball-level accuracy                            │
│       │                                                       │
│       ▼                                                       │
│  Trained model artifacts                                    │
│                                                               │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   SIMULATION PIPELINE                        │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Match JSON + Team Lineups                                  │
│       │                                                       │
│       ▼                                                       │
│  sim_v1_2.py (SimulationEngine)                            │
│   - Loads XGBoostModelV2 + StatsProvider                   │
│   - Simulates ball-by-ball (120 balls per innings)         │
│   - Monte Carlo loop (1000+ iterations)                    │
│   - Parallel processing (multiprocessing)                   │
│       │                                                       │
│       ▼                                                       │
│  Match results (win probabilities + score distributions)    │
│                                                               │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   EVALUATION PIPELINE                        │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Test Matches + Betting Odds                                │
│       │                                                       │
│       ▼                                                       │
│  match_evaluator.py                                         │
│   - Runs simulations for each match                         │
│   - Compares to betting market probabilities                │
│   - Calculates: Log Loss, Brier Score, Edge, ROI           │
│       │                                                       │
│       ▼                                                       │
│  Evaluation metrics (JSON + console summary)                │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
Match_Prediction/
├── data/
│   ├── t20s_json/              # 15K+ raw match JSON files
│   ├── xgb_data/               # Processed parquet (train/val/test)
│   └── betting_test/           # T20 World Cup 2024 test matches
│
├── models/
│   ├── xgb/                    # Trained XGBoost model + encoders (v2)
│   ├── xgb_v3/                 # Trained XGBoost model + encoders (v3)
│   ├── lstm_v1/                # Trained LSTM model + encoders
│   ├── transformer_v1/         # Trained Transformer model + encoders
│   ├── cache_chunks/           # Player stats cache v2 (69 files, legacy)
│   └── player_stats_cache_v3.sqlite  # Player stats cache v3 (SQLite, ~46 MB)
│
├── scripts/
│   ├── build_stats_cache.py    # Phase B: JSON → SQLite schema v3
│   ├── materialize_features.py # Phase B: SQLite + JSON → parquet
│   ├── parsing_v2.py           # Core helpers (trackers, parse_match_data_v2, deep_copy_stats)
│   ├── tracker_rehydration.py  # Per-date tracker seeding from SQLite
│   ├── loaders_common.py       # Shared chronological match iterator
│   ├── stats_sqlite_backend.py # SQLite reader (schema v3)
│   ├── stats_provider.py       # Facade over the SQLite backend
│   ├── xgboost_v2.py           # XGBoost training with Optuna
│   ├── lstm_v1.py              # LSTM training script (PyTorch)
│   ├── transformer_v1.py       # Transformer training script (PyTorch/MLX)
│   ├── mlp_v1.py               # MLP baseline training script
│   ├── sim_v1_2.py             # Monte Carlo simulation engine
│   ├── player_metadata.py      # Player metadata provider
│   ├── feature_registry.py     # Central feature definitions (single source of truth)
│   ├── experiment_tracker.py   # Structured experiment result storage
│   ├── run_experiment.py       # Pipeline runner (cache → materialize → train → eval)
│   ├── compare_experiments.py  # Compare experiment results CLI
│   └── sim_eval/               # Evaluation framework
│       ├── run_sim_eval.py     # Main evaluation script
│       ├── match_evaluator.py  # Metrics calculation
│       └── loaders.py          # Data and odds loaders
│
├── experiments/
│   ├── configs/                # YAML experiment configs
│   │   ├── xgb_v3_baseline.yaml
│   │   ├── xgb_v3_no_metadata.yaml
│   │   ├── xgb_v3_no_type_based.yaml
│   │   ├── lstm_v1_baseline.yaml
│   │   └── transformer_v1_baseline.yaml
│   └── results/                # Auto-generated experiment results
│
├── docs/
│   ├── OPERATIONS.md           # How to run pipelines + development workflows
│   ├── DATA_FORMATS.md         # Data specifications
│   └── DESIGN_DECISIONS.md     # Architectural rationale
│
├── CLAUDE.md                   # This file (concise overview)
├── CLAUDE_REFERENCE.md         # Complete technical reference
├── IMPROVEMENTS.md             # Research findings + improvement roadmap
├── README.md                   # Public portfolio description
└── TODO.md                     # Task list
```

---

## Core Modules

### 1. Feature pipeline — `build_stats_cache.py` + `materialize_features.py`

**Purpose**: Transform raw JSON into ML-ready features with temporal integrity. Split into two scripts in Phase B (2026-04-22) so the stateful stats cache and the stateless per-match feature emission can evolve independently.

**`scripts/build_stats_cache.py`** — JSON → SQLite schema v3.
- Chronological walk over `data/t20s_json/` with `PlayerStatsTracker` + `PlayerEloTracker` + `VenueStatsTracker`.
- Writes delta-compressed snapshots + per-match `batting_match_log` / `bowling_match_log` rows.
- Output: `models/player_stats_cache_v3.sqlite` (~46 MB). Idempotent via `_meta.source_json_mtime_max` vs live JSON mtime.
- Runtime: ~6 min on full corpus.
- CLI: `uv run python scripts/build_stats_cache.py [--source-dir data/t20s_json] [--out models/player_stats_cache_v3.sqlite] [--force-rebuild]`

**`scripts/materialize_features.py`** — SQLite + JSON → parquet.
- Per-date batching: one rehydration per date from SQLite (via `tracker_rehydration.py`), then replays same-day matches in monolith order so the tracker drift inside a batch matches the reference pipeline.
- Splits from YAML `data.splits` block (optional; defaults preserve the hardcoded train/val/test/golden cutoffs).
- Output: `data/xgb_data_v3/{train,val,test,golden_test}.parquet` + `.feature_hash` marker.
- Runtime: ~3 min on full corpus (3.5× faster than the monolith — skips the tracker walk).
- CLI: `uv run python scripts/materialize_features.py [--config experiments/configs/xgb_v3_baseline.yaml]`

**Critical Design**: Features reflect state BEFORE ball, stats updated AFTER ball. Phase A harness (`scripts/tests/test_phase_a_parity.py`) proves bit-exact parity with the original monolith — 63/63 columns across all 9519 matches.

**Related**: the helper primitives (`parse_match_data_v2`, `deep_copy_stats`, tracker classes, `classify_match_k_factor`) still live in `scripts/parsing_v2.py` and are imported by both scripts. The old monolith orchestrator was removed in the Phase 5 cleanup.

---

### 2. `scripts/xgboost_v2.py` - XGBoost Training

**Purpose**: Train XGBoost classifier with hyperparameter tuning

**Model**:
- 6-class classifier (dot, 1, 2, 4, 6, wicket)
- 46+ input features (v3)
- Optuna tuning (50 trials)
- Balanced class weights
- ~55-60% accuracy

**Usage**: `python scripts/xgboost_v2.py` (~30-60 min)

---

### 3. `scripts/lstm_v1.py` - LSTM Training

**Purpose**: Train PyTorch LSTM model with sequence context

**Model**:
- 6-class classifier (dot, 1, 2, 4, 6, wicket)
- 63 input features (same as XGBoost v3)
- Sliding window of 10 balls for sequence context
- 2-layer LSTM (hidden=256) with embeddings for categorical features
- Embeddings: batter (64d), bowler (64d), venue (32d), matchup (16d)

**Usage**:
```bash
# Full training
python scripts/lstm_v1.py --epochs 50 --batch-size 512

# Quick test (5% data, 2 epochs)
python scripts/lstm_v1.py --quick
```

**Artifacts** (saved to `models/lstm_v1/`):
- `lstm_model_v1.pt` - Model weights
- `lstm_config_v1.json` - Architecture config
- `feature_scaler_v1.pkl` - StandardScaler for continuous features
- `*_encoder_v1.pkl` - Label encoders for categorical features

---

### 4. `scripts/stats_provider.py` - Temporal Stats Access

**Purpose**: Provide player statistics as of specific dates (prevents data leakage)

**Key Features**:
- Lazy loading: Loads chunks on-demand (69 chunks, ~110MB each)
- LRU cache: Keeps 5 most recent chunks (~550MB max)
- Binary search: O(log n) temporal lookups across 3,442 dates
- Returns: Batting avg/SR, bowling avg/econ, H2H stats

**Usage**:
```python
provider = StatsProvider('models')  # Loads in ~2 sec
stats = provider.get_batting_stats('player_id', '2024-06-15')
# Returns: {'avg': 31.4, 'sr': 140.2}
```

---

### 5. `scripts/sim_v1_2.py` - Monte Carlo Simulation

**Purpose**: Simulate complete T20 matches ball-by-ball

**Key Classes**:
- `MatchState`: Complete match state (teams, scores, players, history)
- `XGBoostModelV2`: XGBoost ball prediction model
- `LSTMModelV1`: LSTM ball prediction model with sequence context
- `SimulationEngine`: Orchestrates ball-by-ball simulation
- `T20Rules`: Enforces cricket rules
- `ResultAggregator`: Aggregates 1000+ simulations

**Usage**:
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

# Create match
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

**Performance**: ~0.01-0.1 sec per match (1000 sims, parallel)

---

### 6. `scripts/sim_eval/` - Evaluation Framework

**Purpose**: Evaluate model predictions against betting market odds

**Key Components**:
- `TestMatchLoader`: Loads match JSON and creates MatchState
- `BettingOddsLoader`: Loads odds, removes bookmaker margin
- `MatchLevelEvaluator`: Runs simulations, calculates metrics

**Metrics**:
- **Log Loss**: -log(P(actual winner)) - lower is better
- **Brier Score**: (P(team) - actual)² - lower is better
- **Edge**: model_prob - market_prob - positive = value opportunity
- **Betting Performance**: ROI, win rate, total P&L

**Usage**: `python scripts/sim_eval/run_sim_eval.py --test-dir data/betting_test --odds betting_odds_v3.json`

---

### 7. `scripts/feature_registry.py` - Feature Registry

**Purpose**: Central source of truth for all features. Eliminates duplicate feature lists across training scripts.

**Key Contents**:
- `FEATURE_GROUPS`: Dict mapping group names → feature column lists (10 groups, 63 total features)
- `resolve_feature_list(groups, exclude, include_extra)`: Build feature list from group names
- `get_feature_hash(feature_list)`: Deterministic hash for smart caching
- `V3_GROUPS` / `V2_GROUPS`: Convenience constants

**Usage**:
```python
from feature_registry import resolve_feature_list, V3_GROUPS
features = resolve_feature_list(V3_GROUPS)                          # All 63 features
features = resolve_feature_list(V3_GROUPS, exclude=['batter_age'])  # Ablation
```

**When to modify**: When adding new features to the system. Add them to an existing group or create a new group.

---

### 8. `scripts/run_experiment.py` - Pipeline Runner

**Purpose**: Run a complete experiment (parse → train → evaluate) from a single YAML config.

**Usage**:
```bash
uv run python scripts/run_experiment.py experiments/configs/xgb_v3_baseline.yaml
uv run python scripts/run_experiment.py experiments/configs/xgb_v3_baseline.yaml --skip-parsing
uv run python scripts/run_experiment.py experiments/configs/xgb_v3_baseline.yaml --only-eval
uv run python scripts/run_experiment.py experiments/configs/xgb_v3_baseline.yaml --dry-run
```

**Features**: Smart caching (skips parsing if feature hash unchanged), experiment tracking (saves config + git state + metrics), console output capture.

---

### 9. `scripts/compare_experiments.py` - Experiment Comparison

**Purpose**: List and compare experiment results.

**Usage**:
```bash
uv run python scripts/compare_experiments.py --list              # List all
uv run python scripts/compare_experiments.py --list --tag xgboost # Filter by tag
uv run python scripts/compare_experiments.py --show <exp_id>     # Show details
uv run python scripts/compare_experiments.py <id_1> <id_2>       # Side-by-side comparison
```

---

## Key Concepts

### 1. Temporal Integrity (CRITICAL)

Player stats must reflect only historical data available at match time.

**Implementation**:
- During training: Take snapshot BEFORE processing each match
- During simulation: Binary search for most recent snapshot ≤ match_date
- Ensures no data leakage

**Why it matters**: Using future data during simulation creates unrealistic accuracy estimates.

**Details**: [docs/DESIGN_DECISIONS.md#1-temporal-data-integrity](docs/DESIGN_DECISIONS.md#1-temporal-data-integrity)

---

### 2. Chunked Lazy Loading

**Problem**: 7.6GB stats cache is too large to load into memory

**Solution**: Split into 69 chunks (~110MB each), load on-demand with LRU cache

**Result**: 14x memory reduction (7.6GB → 550MB), ~2 sec startup

**Details**: [docs/DESIGN_DECISIONS.md#3-chunked-stats-cache](docs/DESIGN_DECISIONS.md#3-chunked-stats-cache)

---

### 3. Ball-Level Modeling

**Why not predict match outcomes directly?**
- Limited data: 15K matches
- Better approach: Predict 4M individual balls, simulate matches
- Result: 266x more training data, natural uncertainty quantification

**Details**: [docs/DESIGN_DECISIONS.md#5-ball-level-modeling](docs/DESIGN_DECISIONS.md#5-ball-level-modeling)

---

## Common Operations

### Retrain Model with New Data
```bash
# 1. Add new matches
cp new_matches/*.json data/t20s_json/

# 2. Re-run everything via run_experiment.py (detects SQLite/parquet staleness
#    from JSON mtime + feature hash, rebuilds only what's changed):
uv run python scripts/run_experiment.py experiments/configs/xgb_v3_baseline.yaml

# --- OR explicit step-by-step ---
# 2a. Rebuild the SQLite stats cache (new matches → new snapshots + match log)
uv run python scripts/build_stats_cache.py

# 2b. Rematerialize feature parquet
uv run python scripts/materialize_features.py

# 2c. Retrain XGBoost
uv run python scripts/xgboost_v2.py

# 3. Evaluate (betting_test = 44 T20 World Cup 2024 matches;
#    for the 261-match polymarket regression use --test-dir data/polymarket_test
#    --odds betting_odds_polymarket.json)
uv run python scripts/sim_eval/run_sim_eval.py \
    --test-dir data/betting_test \
    --odds betting_odds_v3.json \
    --n-sims 1000
```

### Simulate a Specific Match

See "Usage" section in Module 4 above. Key points:
- Player IDs must match training data
- Match date must be ≤ latest cache date
- Use `parallel=True` for 3-4x speedup

**Full example**: [docs/OPERATIONS.md#standalone-simulation](docs/OPERATIONS.md#standalone-simulation-python-api)

### Inspect Cache

```python
# Load v3 cache (with type-based stats)
provider = StatsProvider('models', version='v3')
print(f"Dates: {provider.dates[0]} to {provider.dates[-1]}")
stats = provider.get_batting_stats('253802', '2024-06-15')
print(f"Avg: {stats['avg']:.1f}, SR: {stats['sr']:.1f}")

# v3 also supports type-based stats
bat_vs_type = provider.get_batting_vs_type_stats('253802', '2024-06-15')
print(f"vs Pace: {bat_vs_type['avg_vs_pace']:.1f}, vs Spin: {bat_vs_type['avg_vs_spin']:.1f}")
```

---

## Data Flow Summary

**Training**:
```
Raw JSON → build_stats_cache.py → SQLite cache
         → materialize_features.py (uses SQLite + JSON) → Parquet → xgboost_v2.py → Model
```

**Simulation**:
```
Match JSON → MatchState → SimulationEngine → 1000 iterations → Win probabilities
                              ↑
                        StatsProvider (lazy-loads cache)
```

**Evaluation**:
```
Test Matches + Odds → Simulate each → Compare probabilities → Metrics
```

---

## Performance Benchmarks

| Operation | Duration | Memory |
|-----------|----------|--------|
| Feature Engineering | 10-15 min | 4-8 GB |
| Model Training | 30-60 min | 8-16 GB |
| Stats Cache Load | 1-2 sec | 300-550 MB |
| Single Match Sim | 0.01-0.1 sec | ~2 GB |
| Full Evaluation (45 matches) | 5-10 min | ~2-3 GB |

---

## Troubleshooting

### "Player not found in training data"
Player ID not in encoder vocabulary. Check if player exists or retrain with updated data.

### "KeyError: date not in cache"
Match date outside cache range. Check `provider.dates[0]` to `provider.dates[-1]`.

### Simulation very slow
Enable `parallel=True` in SimulationConfig. Reduces simulations for testing.

**Full troubleshooting guide**: [docs/OPERATIONS.md#troubleshooting](docs/OPERATIONS.md#troubleshooting)

---

## Key Files Reference

**Training (XGBoost v3 - current)**:
- `data/t20s_json/` - Raw matches
- `data/xgb_data_v3/*.parquet` - Processed training data (v3 with player metadata)
- `data/all_players_enriched.csv` - Player metadata (hand, arm, DOB, bowling style)
- `models/xgb_v3/xgboost_model_v3.pkl` - Trained model (v3)
- `models/player_stats_cache_v3.sqlite` - Player stats cache (SQLite schema v3, with match logs)
- `scripts/player_metadata.py` - Player metadata provider

**Training (LSTM v1)**:
- `scripts/lstm_v1.py` - LSTM training script
- `models/lstm_v1/lstm_model_v1.pt` - Trained LSTM model
- `models/lstm_v1/lstm_config_v1.json` - Model architecture config
- `models/lstm_v1/feature_scaler_v1.pkl` - Feature scaler
- `models/lstm_v1/*_encoder_v1.pkl` - Label encoders

**Training (XGBoost v2 - legacy)**:
- `data/xgb_data/*.parquet` - Processed training data (v2)
- `models/xgb/xgboost_model_v2.pkl` - Trained model (v2)
- `models/cache_chunks/` - Player stats cache (v2)

**Simulation**:
- `scripts/sim_v1_2.py` - Simulation engine (XGBoostModelV2, LSTMModelV1)
- `scripts/stats_provider.py` - Stats access (SQLite v3 only post-Phase-B)
- `scripts/stats_sqlite_backend.py` - SQLite reader (schema v3, match-log getters)
- `scripts/tracker_rehydration.py` - Rehydrates trackers from SQLite for replay

**Evaluation**:
- `data/betting_test/` - Test matches
- `betting_odds_v3.json` - Market odds
- `scripts/sim_eval/run_sim_eval.py` - Evaluation script (`--model-type xgboost|lstm`)

---

## Module Quick Reference

| Module | Purpose | Key Output | Documentation |
|--------|---------|------------|---------------|
| `build_stats_cache.py` | JSON → SQLite schema v3 | `models/player_stats_cache_v3.sqlite` | See section 1 above |
| `materialize_features.py` | SQLite + JSON → parquet (per-date batching) | `data/xgb_data_v3/*.parquet` + `.feature_hash` | See section 1 above |
| `parsing_v2.py` (helpers only) | Tracker primitives (`parse_match_data_v2`, `deep_copy_stats`, tracker classes, `classify_match_k_factor`) | Imported by the two scripts above | [CLAUDE_REFERENCE.md](CLAUDE_REFERENCE.md#1-scriptsparsing_v2py---feature-engineering-pipeline) |
| `tracker_rehydration.py` | Seed `PlayerStatsTracker`/`Elo`/`Venue` from SQLite at a date boundary | Per-date `temp_*` trackers | See section 1 above |
| `loaders_common.py` | `iter_matches_chronological(folder, gender)` | Shared chronological match iterator | — |
| `xgboost_v2.py` | XGBoost training | Trained XGBoost model | [CLAUDE_REFERENCE.md](CLAUDE_REFERENCE.md#2-scriptsxgboost_v2py---model-training) |
| `lstm_v1.py` | LSTM training | Trained LSTM model | See section 3 above |
| `transformer_v1.py` | Transformer training | Trained Transformer model | See section above |
| `mlp_v1.py` | MLP baseline training | Trained MLP model | — |
| `stats_provider.py` | Temporal stats access | Player statistics | [CLAUDE_REFERENCE.md](CLAUDE_REFERENCE.md#3-scriptsstats_providerpy---temporal-stats-access-with-lazy-loading) |
| `sim_v1_2.py` | Match simulation | Win probabilities | [CLAUDE_REFERENCE.md](CLAUDE_REFERENCE.md#4-scriptssim_v1_2py---monte-carlo-simulation-engine) |
| `sim_eval/` | Evaluation | Performance metrics | [CLAUDE_REFERENCE.md](CLAUDE_REFERENCE.md#5-scriptssim_eval---evaluation-framework) |
| `feature_registry.py` | Central feature defs | Feature lists + hashes | See section 7 above |
| `run_experiment.py` | Pipeline runner | Experiment results | See section 8 above |
| `compare_experiments.py` | Experiment comparison | Terminal tables | See section 9 above |
| `fetch_cricsheet.py` | Refresh match JSONs from Cricsheet | Updated `data/t20s_json/` + zip cache | [docs/OPERATIONS.md](docs/OPERATIONS.md#refreshing-cricsheet-data) |
| `enrich_players_cricketdata.py` | Fill player metadata via R cricketdata | Appended rows in `all_players_enriched.csv` | [docs/OPERATIONS.md](docs/OPERATIONS.md#enriching-player-metadata-r-cricketdata) |

---

## Documentation Structure

- **CLAUDE.md** (this file): Concise overview for agents
- **[CLAUDE_REFERENCE.md](CLAUDE_REFERENCE.md)**: Complete technical reference (15K tokens)
- **[docs/OPERATIONS.md](docs/OPERATIONS.md)**: How to run pipelines, common operations
- **[docs/DATA_FORMATS.md](docs/DATA_FORMATS.md)**: All data format specifications
- **[docs/DESIGN_DECISIONS.md](docs/DESIGN_DECISIONS.md)**: Architectural rationale

---

## Current Status

**Latest Models**:
- **XGBoost v3** (default): 46+ features, ~55-60% ball-level accuracy
- **LSTM v1**: PyTorch LSTM with 10-ball sequence context, same features as XGBoost v3

**Model Versions**:
| Model Type | Version | Features | Model Path |
|------------|---------|----------|------------|
| XGBoost | v3 (current) | 46+ | `models/xgb_v3/` |
| XGBoost | v2 (legacy) | 29 | `models/xgb/` |
| LSTM | v1 | 63 | `models/lstm_v1/` |
| Transformer | v1 | 63 | `models/transformer_v1/` |

**Recent Updates**:
- ✅ **Parsing pipeline split — Phase B / schema v3 (2026-04-22)**: the parsing monolith `process_folder_v2_with_splits` was retired in favor of two scripts with a clean concern split:
  - `scripts/build_stats_cache.py` — JSON → SQLite (schema v3, 46.5 MB, built from JSONs directly; replaces the pickle-chunk intermediate).
  - `scripts/materialize_features.py` — SQLite + JSON → parquet via per-date batching; 3.5× faster than the monolith (170 s vs ~600 s) because it skips the tracker walk.
  - Schema v3 adds `batting_match_log` / `bowling_match_log` tables — one row per `(player, match)` with an `intra_date_idx` — so `tracker_rehydration.py` can reconstruct the monolith's 5-entry recent-form deque bit-exactly, including eviction behavior on same-day secondaries. `get_batting_match_log_recent(pid, as_of_date, limit=5)` + bowling equivalent live on `_SQLiteBackend`; the denormalized `batting.recent_*` / `bowling.recent_*` sum columns (written by Phase A) stay as the fast hot-path cache for the 5 sim wrappers in `sim_v1_2.py`.
  - **Phase A parity harness (`scripts/tests/test_phase_a_parity.py`) passes 63/63 columns on all 9519 matches** (was 59/63 on 5855 same-day-secondaries under schema v2). Eval parity on 261 polymarket matches × 10 sims: flat betting P&L / ROI / win rate / bets-placed bit-identical to `eval_out_postfix/xgboost_20260421_220541.json`; log-loss + Brier within 0.001 (Monte Carlo noise).
  - **Phase 5 cleanup landed**: 12 GB of `models/cache_chunks_v3/` reclaimed, `scripts/build_stats_sqlite.py` deleted (its chunks→SQLite converter is obsolete), `_ChunkedBackend` + chunk fallback code removed from `scripts/stats_provider.py` (861 lines → 264, −583 lines), `models/player_stats_cache_v3_metadata.pkl` removed (metadata now lives in SQLite `_meta`). `scripts/parsing_v2.py` trimmed from 1446 → 1137 lines (orchestrator + `__main__` removed; helpers retained because the new scripts + sim pipeline still import `parse_match_data_v2`, `deep_copy_stats`, the tracker classes, and `classify_match_k_factor`).
  - **`run_experiment.py`**: `check_smart_cache` now returns `(sqlite_valid, parquet_valid)` — SQLite validity checks `_meta.schema_version==3` + `_meta.source_json_mtime_max ≥ max(JSON mtime)`; parquet validity unchanged. Dispatch runs `build_stats_cache.py` if SQLite stale, `materialize_features.py` if parquet stale; either is skipped independently.
  - YAML `data.splits` block optional — missing keys fall back to the hardcoded `DEFAULT_SPLITS` in `materialize_features.py`, so existing `experiments/configs/*.yaml` keep working unchanged.
  - New tests: `scripts/tests/test_schema_v3_match_log.py` (7 assertions on log queries: newest-first ordering, `limit` honor, strict `date_id<?`, log-sum ≡ denormalized sum columns, `USING INDEX` query plan, intra_date ordering).
- ✅ Recent-form cache fix — schema v2 (2026-04-21): SQLite stats cache bumped to schema v2. `batting` / `bowling` tables each gained 3 columns for last-5-match totals (`recent_runs/balls/dismissals`, `recent_runs_given/balls_bowled/wickets`). New getters `get_batting_recent` / `get_bowling_recent` on both backends, wired through `StatsProvider`/`StatsProviderCache` via `__getattr__`. `deep_copy_stats` in `parsing_v2.py` now pre-sums the recent deques at snapshot time; `build_stats_sqlite.py` delta-compresses on the 6-tuple. Fixed a silent-fallback bug: the 4 `*_recent_*` features were baked into training but inference paths substituted **0** (XGB `buf.fill`) or **career-avg** (3 PyTorch wrappers — LSTM, MLP, Transformer). All 5 wrappers in `sim_v1_2.py` now call the getters explicitly. `_ChunkedBackend` has a schema gate that raises on pre-v2 chunks (no silent `st.get(...)` defaults). DB size 39.7 → 41.7 MB, build 5:48; all Phase-1 bench gates still PASS (p50 2.96 µs, p99 6.58 µs, 4-worker RSS 222 MB), 78 stratified + 10k random equivalence green. Eval delta on 261-match polymarket × 10 sims: log_loss 0.7541 → 0.7518, flat_roi −1.37% → +6.51%, frac_kelly_roi +0.22% → +1.27%, 216/261 matches diverged (bug was live-wrong on 83%). Baseline at `eval_out_baseline/xgboost_20260421_213207.json`, post-fix at `eval_out_postfix/xgboost_20260421_220541.json`. **Parquet ↔ SQLite parity regression test** (`scripts/tests/test_recent_form_parquet_parity.py`) — parsing now writes `match_date` into each ball row; test samples single-match (player, date) pairs and asserts `get_batting_recent`/`get_bowling_recent` match `batsman_recent_*` / `bowler_recent_*` parquet columns bit-exactly (2 000 batting + 2 000 bowling rows, PASS).
- ✅ Duplicate-snapshot dedup (2026-04-21): building the parity test above surfaced a latent bug — 45/3 664 dates (1.2%) had TWO snapshots stored across chunks (chunk-boundary straddle: `stats_snapshots` dict was reset on chunk save, so a same-day match after the reset triggered a second, mid-day snapshot for that date). Both `_ChunkedBackend._build_date_index` and `build_stats_sqlite.py` used last-write-wins, returning the mid-day snapshot at inference — a temporal-integrity leak. Fixed at three layers: `parsing_v2.py` now tracks `snapshotted_dates: set` that persists across chunk-save resets (prevents future duplicates); `build_stats_sqlite.py` and `_ChunkedBackend` both switched to first-write-wins. Dates index now correctly 3 664 unique (was 3 709 with duplicates). Existing chunks on disk still contain both snapshots for those 45 dates — consumers just pick the first, so no parsing rebuild required. Verified: `fc4375ef` on `2026-02-10` now returns `{'avg': 11.4, 'sr': 83.82}` (pre-first-match snapshot, matches parquet) instead of `{'avg': 28.5, 'sr': 123.91}` (pre-match-2-of-day).
- ✅ SQLite stats-cache migration Phase 3+4 (2026-04-19): `StatsProvider` now auto-detects and uses `models/player_stats_cache_v3.sqlite` when present (logs `StatsProvider: using SQLite backend`), falls back to chunks with a log line otherwise. Staleness guard: `_meta.source_chunks_mtime_max` compared against live chunk mtimes — fails loud if SQLite is older than chunks. `_ChunkedBackend` split out of the facade; public API preserved (`__getattr__` delegation). Legacy tests (`test_stats_cache.py`, `test_sim_with_stats.py`, `validate_training_cache_match.py`) rewritten backend-agnostic and pass on both. **Phase 4 end-to-end**: 261×100 eval serial on polymarket_test — SQLite 36 min, chunks 38 min, `simulated_prob` **bit-identical to 16 dp across all 261 matches** (`scripts/tests/compare_phase4_evals.py`). 2-worktree parallel SQLite eval (the original crash scenario on chunks): peak combined RSS **1 736 MB** — no OOM, outputs bit-identical to serial. See `docs/SQLITE_MIGRATION_PROFILE.md` Phase 4 section. Parallel comparison tool: `scripts/tests/sample_rss_by_name.py`.
- ✅ Phase 1+2 SQLite stats-cache migration (2026-04-19): built `scripts/stats_sqlite_backend.py` (`_SQLiteBackend`, mmap read-only, lazy PID-aware connection, 8 getters + raw counter getters for validator parity) and `scripts/build_stats_sqlite.py` (one-shot chunk→SQLite converter with delta compression). Full DB: `models/player_stats_cache_v3.sqlite` = **39.7 MB** (vs 11 GB chunks = **276× smaller**), built in 5:43. Phase 1 bench (5-chunk POC, 4 workers): per-proc RSS 22.6 MB, combined 75.8 MB, query p50 2.2 µs, p99 4.8 µs — all 5 gates PASS. Phase 2 equivalence: 64 stratified cases + 10 000 random queries bit-exact vs chunked StatsProvider. Cross-date workload bench: 51 MB vs 3 225 MB (63× less RAM), 0.08 s vs 533 s (6581× faster). See `docs/SQLITE_MIGRATION_PROFILE.md`. Regression guards: `scripts/tests/bench_sqlite_backend.py`, `scripts/tests/test_sqlite_equivalence.py`, `scripts/tests/bench_sqlite_vs_chunks.py`.
- ✅ Phase 0 team-memo (2026-04-19): `StatsProviderCache` in `stats_provider.py` wraps a raw `StatsProvider` and memoizes the 5 lineup/venue-keyed methods (`get_team_batting_elo`, `get_team_bowling_elo`, `get_team_batting_strength`, `get_team_bowling_strength`, `get_venue_profile`). All 5 model classes (`XGBoostModelV2`, `LSTMModelV1`, `MLPModelV1`, `MLPModelV2`, `TransformerModelV1`) wrap via `wrap_with_cache()` on construction. Warm-chunk bench (100 sims): raw 9.12s → wrapped 7.57s = **1.20× speedup** on XGBoost v3. Bit-identical sim outputs. Also a prerequisite for the planned SQLite backend migration (reduces per-ball query count before swapping backends). Regression guards: `scripts/tests/test_team_memo_parity.py`, `scripts/tests/check_team_memo_e2e.py`, `scripts/tests/bench_team_memo_speedup.py`.
- ✅ Eval speedup Fix A + B (2026-04-18): `XGBoostModelV2` now (A) caches `LabelEncoder` classes as dicts to replace per-ball `transform()` calls, and (B) returns a preallocated float64 row to skip the per-ball DataFrame→XGBoost conversion. Combined: 2×20 profile bench 91.4 s → **2.25 s** (~40×); full 44×100 eval projected ~3.5 min (from 107.7 min). Bit-identical `simulated_prob`. See `docs/EVAL_PROFILING.md`; regression guards: `scripts/tests/test_xgboost_model_v2_encoder_cache.py`, `scripts/validate_numpy_predict.py`.
- ✅ Fixed XGBoost class_to_outcome bug (was 8-class, now correct 6-class)
- ✅ Fixed evaluation JSON output — now saves full betting metrics (P&L, ROI, win rate, Kelly) with timestamped filenames
- ✅ Re-evaluated all 4 models post-bug-fix — all lose money (best: XGBoost -43.9% flat ROI). Root cause: no team-level signal.
- ✅ Feature registry (`feature_registry.py`) — central feature definitions
- ✅ Experiment infrastructure (config → runner → tracker → comparison)
- ✅ Transformer model with MLX support (Apple Silicon)
- ✅ Player metadata features (Tier 1/2/3)
- ✅ Kelly Criterion betting evaluation
- ✅ Temporal stats cache with lazy loading
- ✅ Optuna hyperparameter tuning
- ✅ LSTM model with sequence context (PyTorch)

**Active Development** (see [TODO.md](TODO.md)):
- Team-level features (team_batting_avg, relative_strength, etc.) — #1 priority
- Calibration layer (isotonic regression, target ECE < 0.015)
- Expand test set to 200+ matches

---

## Quick Links

**For implementation details**: [CLAUDE_REFERENCE.md](CLAUDE_REFERENCE.md)
**For operations guide**: [docs/OPERATIONS.md](docs/OPERATIONS.md)
**For adding features/models/data**: [docs/OPERATIONS.md#development-workflows](docs/OPERATIONS.md#development-workflows)
**For data specifications**: [docs/DATA_FORMATS.md](docs/DATA_FORMATS.md)
**For design rationale**: [docs/DESIGN_DECISIONS.md](docs/DESIGN_DECISIONS.md)
**For research & improvements**: [IMPROVEMENTS.md](IMPROVEMENTS.md)
**For task list**: [TODO.md](TODO.md)

---

**Branch**: features/transformer-model
**Python**: 3.11+
**Dependencies**: See `requirements.txt`
**System**: 16GB RAM, 10GB disk, 4+ CPU cores recommended

---

*This is a concise overview. For complete technical documentation, see [CLAUDE_REFERENCE.md](CLAUDE_REFERENCE.md).*
