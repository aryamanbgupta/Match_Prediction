# CricML: Ball-by-Ball T20 Match Prediction

An end-to-end ML system that predicts T20 cricket match outcomes by learning
individual ball outcomes (dot / 1 / 2 / 4 / 6 / wicket) and simulating complete
matches via Monte Carlo. Built as a portfolio project demonstrating production
ML engineering — data architecture, temporal-integrity features, model training,
simulation, and market-based evaluation.

---

## The Idea

Directly predicting "who wins" gives you ~11,000 training examples (one per
match). Predicting each ball gives you ~2.2 million. We model the ball, then
recover match probabilities by running 1000 simulations per match.

```
Raw Cricsheet JSON (11K matches)
        ↓
Feature engineering — schema-v4 SQLite stats cache + parquet splits
        ↓
XGBoost 6-class classifier (114 features, v6)
        ↓
Monte Carlo simulation (ball-by-ball, 1000 iterations/match)
        ↓
Win probabilities & score distributions → compared to Polymarket / bookmaker odds
```

---

## Current State

- **Data**: ~11,260 men's T20 match JSONs from Cricsheet (2005 → 2026-04) across
  14 leagues (IPL, BBL, PSL, SA20, ILT20, CPL, BPL, T20Is, …).
- **Feature pipeline**: two stages.
  - `build_stats_cache.py` — JSON → SQLite schema v4 (`models/player_stats_cache_v3.sqlite`, ~57 MB, mmap-read).
  - `materialize_features.py` — SQLite + JSON → four parquet splits
    (`data/xgb_data_v3/{train,validation,test,golden_test}.parquet`,
    2.2 M rows × 105 columns).
- **Model**: XGBoost v6 — 114 features, including 42 empirical-Bayes-shrunk
  outcome-distribution features (per batter / bowler / batter-vs-pace|spin /
  bowler-vs-LHB|RHB / venue). Other trainers (LSTM, MLP, Transformer) are kept
  for comparison; XGBoost is the production path.
- **Evaluation**: 261-match Polymarket test set (2025-07 → 2026-04) + legacy
  44-match T20 WC 2024 set. Metrics: log loss, Brier score, edge, flat /
  Kelly ROI, win rate.

**Latest numbers** (v6 outcome-dist, 261 matches × 100 sims, April 2026):
log loss **0.7122**, Brier **0.2562**, flat ROI **−7.1 %**, fractional-Kelly
ROI **+0.7 %**. Calibration vs the v4 baseline improved ~5–6 %; flat-betting
regressed (sharper probabilities compress betting edges — a known
calibration-vs-ROI tension). Market log loss on the same set is 0.6267.
See [IMPROVEMENTS.md](IMPROVEMENTS.md) for per-experiment breakdowns and
[TODO.md](TODO.md) for the `≥ $50 K` / `≥ $100 K` liquidity-sliced evaluation
still outstanding.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   TRAINING PIPELINE                         │
├─────────────────────────────────────────────────────────────┤
│ data/t20s_json/*.json                                       │
│        │                                                    │
│        ▼                                                    │
│ scripts/build_stats_cache.py      (JSON → SQLite v4, ~7 min)│
│        │                                                    │
│        ▼                                                    │
│ models/player_stats_cache_v3.sqlite (~57 MB)                │
│        │                                                    │
│        ▼                                                    │
│ scripts/materialize_features.py   (SQLite+JSON → parquet,   │
│        │                           per-date batching, ~5 min)│
│        ▼                                                    │
│ data/xgb_data_v3/{train,validation,test,golden_test}.parquet│
│        │                                                    │
│        ▼                                                    │
│ scripts/xgboost_v2.py             (Optuna hyperparam tune)  │
│        │                                                    │
│        ▼                                                    │
│ models/xgb_v3/ (model + encoders + feature_columns)         │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                 SIMULATION + EVAL PIPELINE                  │
├─────────────────────────────────────────────────────────────┤
│ Match JSON + lineup ─► MatchState                           │
│        │                                                    │
│        ▼                                                    │
│ scripts/sim_v1_2.py (SimulationEngine)                      │
│   ├─ XGBoostModelV2 / LSTMModelV1 / TransformerModelV1 / …  │
│   ├─ StatsProvider  (SQLite mmap, StatsProviderCache memo)  │
│   ├─ T20Rules       (strike rotation, bowler limits, chase) │
│   └─ Monte Carlo    (1000 iterations, multiprocessing)      │
│        │                                                    │
│        ▼                                                    │
│ Per-match win probs + score distributions                   │
│        │                                                    │
│        ▼                                                    │
│ scripts/sim_eval/                                           │
│   ├─ loaders.py         (TestMatchLoader, BettingOddsLoader) │
│   ├─ match_evaluator.py (log loss, Brier, edge, P&L, Kelly) │
│   └─ run_sim_eval.py    (CLI)                               │
└─────────────────────────────────────────────────────────────┘
```

---

## Quick Start

All scripts run via `uv run` to pick up the pinned environment.

```bash
# One-command run: cache → parquet → train → eval.
# Each step is skipped if its artifact is already current.
uv run python scripts/run_experiment.py \
    experiments/configs/xgb_v6_outcome_dist.yaml

# Or the same pipeline, step by step
uv run python scripts/build_stats_cache.py              # ~7 min → SQLite v4
uv run python scripts/materialize_features.py \
    --config experiments/configs/xgb_v6_outcome_dist.yaml   # ~5 min → parquet
uv run python scripts/xgboost_v2.py                     # ~5–10 min (no tune)
uv run python scripts/sim_eval/run_sim_eval.py \
    --test-dir data/polymarket_test \
    --odds betting_odds_polymarket.json \
    --n-sims 100                                        # ~40 min serial
```

`scripts/build_stats_cache.py` is idempotent: it no-ops when no JSON has
changed since the last build (compares `_meta.source_json_mtime_max`).

Alternative trainers:
```bash
uv run python scripts/lstm_v1.py --epochs 50 --batch-size 512
uv run python scripts/mlp_v1.py --epochs 30 --device cpu
uv run python scripts/transformer_v1.py --epochs 50 --batch-size 64
uv run python scripts/transformer_v1.py --mlx --epochs 50 --batch-size 128  # Apple Silicon
```

Select an alternative trainer at eval time:
```bash
uv run python scripts/sim_eval/run_sim_eval.py --model-type lstm  ...
uv run python scripts/sim_eval/run_sim_eval.py --model-type transformer --mlx  ...
```

> **Warning**: `--parallel` on `run_sim_eval.py` has crashed the 16 GB test
> machine; default is serial. Schedule long runs as background tasks.

---

## Key Engineering Decisions

**SQLite-backed temporal stats cache**. An earlier iteration used 69 pickle
chunks (~11 GB on disk, with LRU loading). Migrated to a single 57 MB mmap
SQLite file: **276 × smaller on disk**, `~3 µs` p50 query, safe to open
concurrently for multi-process evaluation. See
[docs/SQLITE_MIGRATION_PROFILE.md](docs/SQLITE_MIGRATION_PROFILE.md).

**Temporal integrity**. All features reflect state *before* each ball; stats
are updated *after*. `build_stats_cache.py` takes a first-write-wins snapshot
per date, so simulations for a 2024-06-15 match can never read a 2024-06-16
stat. Verified bit-exact against the reference monolith on all 9,519 matches
via `scripts/tests/test_phase_a_parity.py`.

**Ball-level modelling + Monte Carlo**. 2.2 M training balls vs 11 K matches is
a 200× data multiplier. Simulation variance gives us uncertainty for free.

**Per-match memoization** (`StatsProviderCache`). Team-strength / venue-profile
features are constant across every ball of a given match, so we memoize the
expensive 11-player-loop lookups once per match. ~1.2× speedup on XGBoost sim.

**Empirical-Bayes outcome distributions (v6, schema v4)**. For each batter /
bowler / batter-vs-bowler-type / bowler-vs-batter-hand / venue, emit
`P(0, 1, 2, 4, 6, W)` shrunk toward a global corpus prior π via
`p̂_c = (n_c + k·π_c) / (N + k)`. Computed over 2.19 M balls;
π = (0.304, 0.411, 0.076, 0.108, 0.047, 0.054). Adds the distributional
signal that label-encoded IDs structurally can't give XGBoost.

---

## Repository Layout

```
Match_Prediction/
├── CLAUDE.md              # Concise agent guide (current)
├── CLAUDE_REFERENCE.md    # Complete technical reference
├── IMPROVEMENTS.md        # Research log + per-experiment results
├── TODO.md                # Current workstreams and P0 gates
├── README.md              # This file
│
├── data/
│   ├── t20s_json/              # 11K+ cricsheet match JSONs
│   ├── all_players_enriched.csv   # Player metadata (hand, arm, DOB, bowling style)
│   ├── cricsheet_people.csv       # Cricsheet player register
│   ├── xgb_data_v3/               # Parquet splits (train/val/test/golden_test)
│   ├── betting_test/              # Legacy 44-match WC 2024 eval set
│   ├── polymarket_test/           # 261-match Polymarket eval set (2025-07 → 2026-04)
│   └── .cricsheet_zips/           # Cached Cricsheet downloads + manifest
│
├── models/
│   ├── player_stats_cache_v3.sqlite  # Schema-v4 stats cache (~57 MB)
│   ├── xgb_v3/                    # Current XGBoost artifacts
│   ├── lstm_v1/                   # LSTM artifacts
│   ├── mlp_v1/                    # MLP artifacts
│   └── transformer_v1/            # Transformer artifacts (PyTorch + MLX)
│
├── scripts/
│   ├── build_stats_cache.py       # JSON → SQLite v4
│   ├── materialize_features.py    # SQLite + JSON → parquet
│   ├── parsing_v2.py              # Tracker primitives + parse_match_data_v2
│   ├── stats_sqlite_backend.py    # _SQLiteBackend reader
│   ├── stats_provider.py          # Facade + StatsProviderCache memo
│   ├── tracker_rehydration.py     # Per-date tracker seeding from SQLite
│   ├── loaders_common.py          # iter_matches_chronological + DEFAULT_SPLITS
│   ├── feature_registry.py        # Central feature definitions
│   ├── player_metadata.py         # Player metadata provider
│   ├── xgboost_v2.py              # XGBoost training with Optuna
│   ├── lstm_v1.py                 # LSTM training
│   ├── transformer_v1.py          # Transformer training (PyTorch + MLX)
│   ├── mlp_v1.py                  # MLP baseline
│   ├── sim_v1_2.py                # Simulation engine + all model wrappers
│   ├── run_experiment.py          # YAML-driven pipeline runner
│   ├── compare_experiments.py     # List / compare experiment results
│   ├── experiment_tracker.py      # Per-experiment metrics + git state
│   ├── calibration.py             # Platt / isotonic calibrators (off by default)
│   ├── fetch_cricsheet.py         # Refresh Cricsheet data
│   ├── enrich_players_cricketdata.py  # Fill player metadata via R cricketdata
│   ├── build_polymarket_odds.py   # Match Polymarket markets → Cricsheet → odds JSON
│   ├── sim_eval/                  # Evaluation framework
│   └── tests/                     # Parity harness + benchmarks
│
├── experiments/
│   ├── configs/*.yaml             # Declarative experiment definitions
│   └── results/                   # Auto-generated per-run artifacts
│
└── docs/                          # Detailed reference documentation
    ├── ARCHITECTURE.md            # ★ canonical technical reference (modules,
    │                              #   data flow, classes, formats, design)
    ├── OPERATIONS.md              # How to run every pipeline
    ├── ADDING_NEW_MODELS.md       # Model-plugin guide
    ├── feature_roadmap.md         # Feature set + status
    ├── CLOUD_GPU_TESTING.md       # LLM cloud GPU notes
    └── archive/                   # historical design docs + one-time memos
                                   # (DATA_FORMATS, DESIGN_DECISIONS,
                                   #  CLAUDE_REFERENCE, SQLITE_MIGRATION_PROFILE,
                                   #  POLYMARKET_INTEGRATION, EVAL_PROFILING, ...)
```

`archive/` (top level, not shown above) is a separate, gitignored local
archive for non-doc artifacts (old logs, historical eval JSONs, superseded
scripts). See [archive/README.md](archive/README.md).

---

## Validation Discipline

Every experiment reports four benchmarks side by side: coinflip (0.6931 LL
floor), always-bet-favorite (~4 % ROI honest baseline), the current Polymarket
market (0.6267 LL ceiling), and our model. Before claiming skill:

1. Model log loss must beat market log loss on the ≥ $50 K liquidity slice.
2. Flat-ROI bootstrap CI on that slice must exclude zero.

Anything weaker is counterparty noise — see [TODO.md](TODO.md)
§ "Evaluation Discipline".

---

## Stack

Python 3.11+, XGBoost 2.x, PyTorch 2.x (+ MLX for Apple Silicon), Optuna,
scikit-learn, pandas + pyarrow (parquet), SQLite, `uv` for env + execution.

System recommended: 16 GB RAM, 10 GB disk, 4+ CPU cores.

---

**Author**: Aryaman Gupta · **Type**: Personal portfolio project · **Branch**: `main`
