# CricML Match Prediction System

**Agent Guide** - Concise overview for AI agents working on this codebase

**Last Updated**: December 2024
**Branch**: features/cricinfo-features
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

**Model Versions**:
- **v2** (legacy): 29 features, basic player stats
- **v3** (current): 46+ features with player metadata (hand, arm, age, matchup types, type-based stats)

---

## Quick Start

> **Important**: Use `uv run` to execute all scripts. This ensures the correct virtual environment and dependencies are used.

```bash
# Full training pipeline (v3 model with player metadata)
uv run python scripts/parsing_v2.py          # ~10-15 min (generates v3 data + cache)
uv run python scripts/xgboost_v2.py          # ~5-10 min (uses saved hyperparameters)

# OR with Optuna hyperparameter tuning
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
```

**For detailed operations guide, see [docs/OPERATIONS.md](docs/OPERATIONS.md)**

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
│  parsing_v2.py                                              │
│   - 46+ features (basic, player stats, momentum, metadata) │
│   - Temporal stats cache (v3: cache_chunks_v3/, lazy-loaded)│
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
│   ├── cache_chunks/           # Player stats cache v2 (69 files)
│   └── cache_chunks_v3/        # Player stats cache v3 (69 files, 7.6GB)
│
├── scripts/
│   ├── parsing_v2.py           # Feature engineering pipeline
│   ├── xgboost_v2.py           # XGBoost training with Optuna
│   ├── lstm_v1.py              # LSTM training script (PyTorch)
│   ├── sim_v1_2.py             # Monte Carlo simulation engine
│   ├── stats_provider.py       # Lazy-loading stats access
│   ├── player_metadata.py      # Player metadata provider
│   └── sim_eval/               # Evaluation framework
│       ├── run_sim_eval.py     # Main evaluation script
│       ├── match_evaluator.py  # Metrics calculation
│       └── loaders.py          # Data and odds loaders
│
├── docs/
│   ├── OPERATIONS.md           # How to run pipelines
│   ├── DATA_FORMATS.md         # Data specifications
│   └── DESIGN_DECISIONS.md     # Architectural rationale
│
├── CLAUDE.md                   # This file (concise overview)
├── CLAUDE_REFERENCE.md         # Complete technical reference
├── README.md                   # Public portfolio description
└── TODO.md                     # Task list
```

---

## Core Modules

### 1. `scripts/parsing_v2.py` - Feature Engineering

**Purpose**: Transform raw JSON into ML-ready features with temporal integrity

**Key Features**:
- 29 features: basic state (12), player stats (6), H2H (2), momentum (5), pressure (4)
- Temporal stats cache: Snapshots player stats BEFORE each match (prevents data leakage)
- Outputs: Parquet files + 69 cache chunks (lazy-loaded)

**Critical Design**: Features reflect state BEFORE ball, stats updated AFTER ball

**Usage**: `python scripts/parsing_v2.py` (~10-15 min)

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

# 2. Re-run pipeline
python scripts/parsing_v2.py
python scripts/xgboost_v2.py

# 3. Evaluate
python scripts/sim_eval/run_sim_eval.py \
    --test-dir data/betting_test \
    --odds betting_odds_v3.json
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
Raw JSON → parsing_v2.py → Parquet + Cache → xgboost_v2.py → Model
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
- `models/cache_chunks_v3/` - Player stats cache (with type-based stats)
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
- `scripts/stats_provider.py` - Stats access (supports v2 and v3)

**Evaluation**:
- `data/betting_test/` - Test matches
- `betting_odds_v3.json` - Market odds
- `scripts/sim_eval/run_sim_eval.py` - Evaluation script (`--model-type xgboost|lstm`)

---

## Module Quick Reference

| Module | Purpose | Key Output | Documentation |
|--------|---------|------------|---------------|
| `parsing_v2.py` | Feature engineering | Parquet files + cache chunks | [CLAUDE_REFERENCE.md](CLAUDE_REFERENCE.md#1-scriptsparsing_v2py---feature-engineering-pipeline) |
| `xgboost_v2.py` | XGBoost training | Trained XGBoost model | [CLAUDE_REFERENCE.md](CLAUDE_REFERENCE.md#2-scriptsxgboost_v2py---model-training) |
| `lstm_v1.py` | LSTM training | Trained LSTM model | See section 3 above |
| `stats_provider.py` | Temporal stats access | Player statistics | [CLAUDE_REFERENCE.md](CLAUDE_REFERENCE.md#3-scriptsstats_providerpy---temporal-stats-access-with-lazy-loading) |
| `sim_v1_2.py` | Match simulation | Win probabilities | [CLAUDE_REFERENCE.md](CLAUDE_REFERENCE.md#4-scriptssim_v1_2py---monte-carlo-simulation-engine) |
| `sim_eval/` | Evaluation | Performance metrics | [CLAUDE_REFERENCE.md](CLAUDE_REFERENCE.md#5-scriptssim_eval---evaluation-framework) |

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

**Recent Updates**:
- ✅ Player metadata features (Tier 1: hand, arm, pace/spin, age)
- ✅ Matchup features (Tier 2: matchup_type, spin_advantage, same_arm)
- ✅ Type-based historical stats (Tier 3: avg/sr vs pace/spin, vs LHB/RHB)
- ✅ Kelly Criterion betting evaluation
- ✅ Temporal stats cache with lazy loading
- ✅ Optuna hyperparameter tuning
- ✅ LSTM model with sequence context (PyTorch)

**Active Development** (see [TODO.md](TODO.md)):
- Unknown player encoding
- Additional features (weather, dew factor)
- Full LSTM training and hyperparameter tuning
- Transformer architecture exploration

---

## Quick Links

**For implementation details**: [CLAUDE_REFERENCE.md](CLAUDE_REFERENCE.md)
**For operations guide**: [docs/OPERATIONS.md](docs/OPERATIONS.md)
**For data specifications**: [docs/DATA_FORMATS.md](docs/DATA_FORMATS.md)
**For design rationale**: [docs/DESIGN_DECISIONS.md](docs/DESIGN_DECISIONS.md)
**For task list**: [TODO.md](TODO.md)

---

**Branch**: features/cricinfo-features
**Python**: 3.11+
**Dependencies**: See `requirements.txt`
**System**: 16GB RAM, 10GB disk, 4+ CPU cores recommended

---

*This is a concise overview. For complete technical documentation, see [CLAUDE_REFERENCE.md](CLAUDE_REFERENCE.md).*
