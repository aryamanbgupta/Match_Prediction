# Operations Guide

Complete guide for running all pipelines and common operations in the CricML Match Prediction system.

---

## Quick Start

### Training Pipeline (XGBoost - Default)
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

### Training Pipeline (LSTM)
```bash
# Step 1: Feature engineering (same as XGBoost)
python scripts/parsing_v2.py

# Step 2: LSTM training (~30-60 min for full training)
python scripts/lstm_v1.py --epochs 50 --batch-size 512

# Quick test mode (5% data, 2 epochs, ~2 min)
python scripts/lstm_v1.py --quick

# Step 3: Evaluation with LSTM model
python scripts/sim_eval/run_sim_eval.py \
    --model-type lstm \
    --test-dir data/betting_test \
    --odds betting_odds_v3.json \
    --n-sims 1000
```

### Refreshing Cricsheet data

Use `scripts/fetch_cricsheet.py` to pull the latest men's T20 JSON archives from
Cricsheet and merge them into `data/t20s_json/`. The script is safe to re-run:
zip downloads go through `.staging/` and are only committed after a successful
SHA-256 compare, and match-JSON merges are **append-only** (Cricsheet JSONs are
immutable once published, so we only add new filenames).

```bash
# Full refresh (14 men's T20 leagues + player register)
uv run python scripts/fetch_cricsheet.py

# Preview with no writes
uv run python scripts/fetch_cricsheet.py --dry-run

# Single league(s)
uv run python scripts/fetch_cricsheet.py --only ipl,bbl

# List leagues + deliberate exclusions
uv run python scripts/fetch_cricsheet.py --list
```

**Artifacts** (all under the already-gitignored `data/` tree):
- `data/.cricsheet_zips/*.zip` — cached league archives
- `data/.cricsheet_zips/manifest.json` — per-slug sha256 + download timestamp + extracted match counts
- `data/.cricsheet_zips/.refresh.log` — append-only run log
- `data/t20s_json/*.json` — live match data (merge target)
- `data/cricsheet_people.csv` — Cricsheet player register

**Leagues included (14 men's T20):** IPL, BBL, CPL, T20 Blast, CSA T20 Challenge,
SA20, ILT20, MLC, men's T20Is, BPL, Super Smash (NZ), PSL, SMAT, LPL.

**Deliberately excluded:** The Hundred (100-ball format — see below), all women's
competitions (pipeline is men's only), associate-nation T20Is (data quality), and
the multi-format `all_json.zip` / Tests / ODIs (out of scope).

**After a successful refresh**, re-run the parser manually to regenerate the
training splits and stats cache:

```bash
uv run python scripts/parsing_v2.py   # ~10–15 min; rebuilds data/xgb_data_v3/ + models/cache_chunks_v3/
```

The fetcher will print this command at the end of any run that adds new matches.
It does **not** auto-chain to `parsing_v2.py` because the cache rebuild is
destructive.

**New unenriched players.** The fetcher compares cricsheet IDs in the newly-added
matches against `data/all_players_enriched.csv` and prints any IDs that aren't in
the enriched metadata. Running `cricinfo_scraper_v3.py` to fill those in is a
manual follow-up, not part of the fetcher.

### Enriching player metadata (R cricketdata)

After a fetch, `scripts/enrich_players_cricketdata.py` fills in biographical
metadata (country, DOB, batting/bowling style, full name) for every cricsheet ID
that appears in `data/t20s_json/*.json` but isn't yet in
`data/all_players_enriched.csv`.

It uses **only** the R `cricketdata` package via `rpy2` — no website scraping.
For each missing ID it looks up the cricinfo key in `cricsheet_people.csv`
(Cricsheet publishes these directly; `key_cricinfo` is present for ~100% of
players we care about) and calls `cricketdata::fetch_player_meta(playerid=...)`.
`cricketdata::find_player_id(name)` is only used as a fallback when the register
has no cricinfo key, which in practice never happens for our data.

```bash
# Preview counts (no R calls, no writes)
uv run python scripts/enrich_players_cricketdata.py --dry-run

# Smoke test one player
uv run python scripts/enrich_players_cricketdata.py --limit 1

# Full enrichment of everything missing
uv run python scripts/enrich_players_cricketdata.py
```

The script writes atomically (tmp file → rename), so a crash mid-run leaves
`all_players_enriched.csv` untouched. Expect ~90%+ fill on country and DOB;
batting/bowling style is sparser for associate-nation newcomers where Cricinfo
itself has no data.

Run order after new Cricsheet data lands:

```bash
uv run python scripts/fetch_cricsheet.py              # 1. pull new match JSONs + people.csv
uv run python scripts/enrich_players_cricketdata.py   # 2. fill metadata for any new cricsheet IDs
uv run python scripts/parsing_v2.py                   # 3. rebuild features + stats cache (destructive)
```

---

#### Future: include The Hundred

The Hundred (`hnd_json.zip`) is currently excluded because its 100-ball innings
is incompatible with the 120-ball hardcodes that appear in several places:

- `scripts/parsing_v2.py` — innings-length assumptions and feature windows
- `scripts/sim_v1_2.py` — `T20Rules` caps innings at 20 overs (120 balls)
- `scripts/transformer_v1.py` — `max_seq_len=120` for positional encoding

Supporting The Hundred would require making innings length data-driven across
those modules. Tracked in [TODO.md](../TODO.md).

---

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

### 3. LSTM Training (`lstm_v1.py`)

**Purpose**: Train PyTorch LSTM model with sequence context.

**Input**:
- `data/xgb_data_v3/cricket_data_v3_*.parquet` (same data as XGBoost)

**Output** (saved to `models/lstm_v1/`):
- `lstm_model_v1.pt` - Model weights
- `lstm_config_v1.json` - Architecture configuration
- `feature_scaler_v1.pkl` - StandardScaler for continuous features
- `batter_encoder_v1.pkl`, `bowler_encoder_v1.pkl`, `venue_encoder_v1.pkl`, `matchup_encoder_v1.pkl` - Label encoders
- `feature_columns_v1.txt`, `continuous_columns_v1.txt` - Feature lists
- `training_history_v1.json` - Training metrics

**Command**:
```bash
# Full training
python scripts/lstm_v1.py --epochs 50 --batch-size 512

# Quick test (5% data, 2 epochs)
python scripts/lstm_v1.py --quick

# Custom configuration
python scripts/lstm_v1.py \
    --epochs 30 \
    --batch-size 256 \
    --window-size 10 \
    --hidden-size 256 \
    --num-layers 2 \
    --learning-rate 0.001
```

**Performance**:
- Full training: ~30-60 minutes (CPU)
- Quick mode: ~2 minutes
- Memory: ~4-8 GB

**What It Does**:
1. Loads parquet files from v3 data
2. Creates sliding window sequences (default: 10 balls)
3. Fits StandardScaler on continuous features
4. Fits LabelEncoders on categorical features (batter, bowler, venue, matchup)
5. Creates PyTorch Dataset with sequence padding
6. Trains 2-layer LSTM with embeddings
7. Evaluates on test set
8. Saves all artifacts

**Model Architecture**:
```
Input per timestep:
├── Continuous features: 59 (normalized)
├── batter_encoded → Embedding(n_batters, 64)
├── bowler_encoded → Embedding(n_bowlers, 64)
├── venue_encoded → Embedding(n_venues, 32)
└── matchup_type_encoded → Embedding(n_matchups, 16)

LSTM:
├── Layer 1: LSTM(input_dim, 256), dropout=0.2
└── Layer 2: LSTM(256, 128), dropout=0.2

Output: Linear(128, 6) → Softmax (6 classes)
```

---

## Simulation

### Evaluation Pipeline (`run_sim_eval.py`)

**Purpose**: Evaluate model predictions against betting market odds.

**Command**:
```bash
# Evaluate with XGBoost (default)
python scripts/sim_eval/run_sim_eval.py \
    --test-dir data/betting_test \
    --odds betting_odds_v3.json \
    --n-sims 1000

# Evaluate with LSTM model
python scripts/sim_eval/run_sim_eval.py \
    --model-type lstm \
    --test-dir data/betting_test \
    --odds betting_odds_v3.json \
    --n-sims 1000
```

**Options**:
- `--model-type`: Model type to use (`xgboost` or `lstm`, default: `xgboost`)
- `--model-version`: XGBoost version (`v2` or `v3`, default: `v3`)
- `--test-dir`: Directory with test match JSONs
- `--odds`: Path to betting odds JSON file
- `--n-sims`: Number of simulations per match (default: 1000)
- `--parallel`: Enable parallel simulation (default: True)
- `--max-matches`: Limit number of matches (for testing)

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

## Experiment Infrastructure

The project includes a lightweight experiment system for running reproducible experiments, tracking results, and comparing runs. **This is the recommended way to test changes.**

### Core Components

| File | Purpose |
|------|---------|
| `scripts/feature_registry.py` | Central source of truth for all 63 features across 10 groups |
| `scripts/run_experiment.py` | Runs full pipeline (parse → train → eval) from a YAML config |
| `scripts/experiment_tracker.py` | Saves config + git state + metrics per experiment |
| `scripts/compare_experiments.py` | Lists and compares experiment results |
| `experiments/configs/*.yaml` | Declarative experiment definitions |
| `experiments/results/` | Auto-generated experiment result directories |

### Running an Experiment

```bash
# Full pipeline from config
uv run python scripts/run_experiment.py experiments/configs/xgb_v3_baseline.yaml

# Skip parsing (data hasn't changed)
uv run python scripts/run_experiment.py experiments/configs/xgb_v3_baseline.yaml --skip-parsing

# Only evaluation (model already trained)
uv run python scripts/run_experiment.py experiments/configs/xgb_v3_baseline.yaml --only-eval

# Preview commands without running
uv run python scripts/run_experiment.py experiments/configs/xgb_v3_baseline.yaml --dry-run
```

### Experiment Config Format

```yaml
experiment:
  name: "xgb_v3_baseline"
  description: "XGBoost v3 with all features"
  tags: ["xgboost", "v3", "baseline"]

data:
  version: "v3"
  test_dir: "data/betting_test"
  odds_file: "betting_odds_v3.json"

features:
  groups: [basic, player_stats, h2h, momentum, pressure, chase, medium,
           player_metadata, matchup, type_based]
  exclude: []           # Individual features to drop
  include_extra: []     # Extra features to add

model:
  type: "xgboost"       # xgboost | lstm | transformer | mlp
  hyperparameters:      # Model-specific params
    n_estimators: 444
    max_depth: 10
    learning_rate: 0.24
  tune: false
  tune_trials: 50

evaluation:
  n_sims: 1000
  parallel: false

pipeline:
  skip_parsing: false
  skip_training: false
  skip_evaluation: false
```

### Comparing Results

```bash
# List all experiments
uv run python scripts/compare_experiments.py --list

# Filter by tag
uv run python scripts/compare_experiments.py --list --tag xgboost

# Show single experiment details
uv run python scripts/compare_experiments.py --show <exp_id>

# Compare two experiments side by side
uv run python scripts/compare_experiments.py <exp_id_1> <exp_id_2>
```

### What Gets Tracked

Each experiment creates `experiments/results/{name}_{timestamp}_{git_hash}/` containing:
- `config.yaml` — exact config used
- `metadata.json` — git hash, branch, dirty flag, platform, step durations
- `metrics.json` — evaluation results (log loss, brier score, ROI)
- `training_metrics.json` — training accuracy/loss
- `console_output.log` — captured stdout/stderr

### Feature Registry

All features are defined in `scripts/feature_registry.py` in 10 groups:

| Group | Count | Examples |
|-------|-------|---------|
| `basic` | 16 | `score`, `wickets`, `run_rate`, `is_powerplay` |
| `player_stats` | 14 | `batsman_avg`, `bowler_econ`, `batter_encoded` |
| `h2h` | 2 | `h2h_avg`, `h2h_sr` |
| `momentum` | 6 | `last_5_balls_runs`, `balls_since_boundary` |
| `pressure` | 3 | `dot_percentage_recent`, `pressure_cooker_index` |
| `chase` | 3 | `chase_target`, `run_rate_required`, `lead_gap` |
| `medium` | 2 | `venue_avg_score`, `non_striker_sr` |
| `player_metadata` | 6 | `batter_hand`, `bowler_arm`, `is_pace`, `batter_age` |
| `matchup` | 3 | `spin_matchup_advantage`, `same_arm_matchup` |
| `type_based` | 8 | `batter_avg_vs_pace`, `bowler_econ_vs_lhb` |

```python
from feature_registry import resolve_feature_list, get_feature_hash, V3_GROUPS

# All v3 features (63)
features = resolve_feature_list(V3_GROUPS)

# Ablation: drop a group
features = resolve_feature_list([g for g in V3_GROUPS if g != 'player_metadata'])

# Drop individual features
features = resolve_feature_list(V3_GROUPS, exclude=['batter_age', 'bowler_age'])

# Get deterministic hash (for smart caching)
hash_val = get_feature_hash(features)
```

---

## Development Workflows

### How to Add a New Feature

**Step 1: Add the feature to `scripts/parsing_v2.py`**

This is where raw JSON is transformed into ML features. Find the ball-processing loop and add your feature computation. Respect temporal integrity — the feature must only use data available before the ball is bowled.

**Step 2: Register the feature in `scripts/feature_registry.py`**

Add it to the appropriate group in `FEATURE_GROUPS`, or create a new group:

```python
# Add to existing group
FEATURE_GROUPS['momentum'].append('new_momentum_feature')

# Or create a new group
FEATURE_GROUPS['team_strength'] = [
    'team_batting_avg', 'team_batting_sr',
    'opp_bowling_avg', 'opp_bowling_econ',
]
```

**Step 3: Re-run parsing and test**

```bash
# Re-parse data (generates new parquet files with the feature)
uv run python scripts/parsing_v2.py

# Create an experiment config that includes your new feature group
# Copy an existing config and add your group to features.groups

# Run experiment
uv run python scripts/run_experiment.py experiments/configs/your_config.yaml --skip-training

# Or test manually
uv run python scripts/xgboost_v2.py
```

**Step 4: Verify with ablation**

Create two configs — one with and one without your feature — and compare:

```bash
uv run python scripts/run_experiment.py experiments/configs/with_feature.yaml --skip-parsing
uv run python scripts/run_experiment.py experiments/configs/without_feature.yaml --skip-parsing
uv run python scripts/compare_experiments.py <with_id> <without_id>
```

**Important**: Training scripts automatically filter to features that exist in the parquet data (`feature_cols = [c for c in features if c in df.columns]`), so adding a feature to the registry before it exists in the data won't crash — it just gets ignored.

---

### How to Add a New Model

**Step 1: Create the training script**

Create `scripts/your_model_v1.py`. Follow the pattern of existing scripts:
- Accept `--config-json` CLI arg for experiment runner integration
- Load data from `data/xgb_data_v3/*.parquet`
- Use the same 6-class target mapping: `{0: dot, 1: one, 2: two, 3: four, 4: six, 5: wicket}`
- Save model artifacts to `models/your_model_v1/`

```python
# Config integration pattern (add to argparse)
parser.add_argument('--config-json', type=str, default=None,
                    help='JSON config from experiment runner')

# After parsing args:
if args.config_json:
    import json as _json
    _config = _json.loads(args.config_json)
    from feature_registry import resolve_feature_list
    feature_cols = resolve_feature_list(
        _config['features']['groups'],
        _config['features'].get('exclude'),
        _config['features'].get('include_extra'),
    )
else:
    # Hardcoded default features (backward compatible)
    feature_cols = [...]
```

**Step 2: Add a model wrapper to `scripts/sim_v1_2.py`**

Create a class that implements the `PredictionModel` interface:

```python
class YourModelV1(PredictionModel):
    def __init__(self, model_path, ...):
        # Load model
        # CRITICAL: use 6-class mapping
        self.class_to_outcome = {
            0: 'dot', 1: 'one', 2: 'two', 3: 'four', 4: 'six', 5: 'wicket'
        }

    def extract_features(self, state: MatchState) -> ...:
        # Build feature vector from current match state

    def predict_probabilities(self, features) -> np.ndarray:
        # Return 6-class probability distribution
```

**Step 3: Register in `scripts/sim_eval/run_sim_eval.py`**

Add your model type to the `--model-type` choices and the model loading logic.

**Step 4: Add to experiment runner**

In `scripts/run_experiment.py`, add your script to `build_training_cmd()`:

```python
script_map = {
    "xgboost": "scripts/xgboost_v2.py",
    "lstm": "scripts/lstm_v1.py",
    "transformer": "scripts/transformer_v1.py",
    "mlp": "scripts/mlp_v1.py",
    "your_model": "scripts/your_model_v1.py",  # Add this
}
```

**Step 5: Create experiment config and test**

```bash
# Create config
cp experiments/configs/xgb_v3_baseline.yaml experiments/configs/your_model_baseline.yaml
# Edit: change model.type to "your_model"

# Run
uv run python scripts/run_experiment.py experiments/configs/your_model_baseline.yaml --skip-parsing
```

---

### How to Add New Match Data

```bash
# 1. Add match JSON files
cp new_matches/*.json data/t20s_json/

# 2. Re-run parsing (rebuilds cache + parquet)
uv run python scripts/parsing_v2.py

# 3. Retrain and evaluate
uv run python scripts/run_experiment.py experiments/configs/xgb_v3_baseline.yaml
```

**For test/evaluation matches**: Add JSON files to `data/betting_test/` and update `betting_odds_v3.json` with corresponding odds.

---

### How to Run a Feature Ablation

No code changes needed. Just create a config with the feature group removed:

```bash
# Copy baseline config
cp experiments/configs/xgb_v3_baseline.yaml experiments/configs/xgb_v3_no_matchup.yaml

# Edit: remove 'matchup' from features.groups
# Edit: set pipeline.skip_parsing to true (data already exists)

# Run
uv run python scripts/run_experiment.py experiments/configs/xgb_v3_no_matchup.yaml

# Compare
uv run python scripts/compare_experiments.py <baseline_id> <ablation_id>
```

---

### How to Tune Hyperparameters

**XGBoost** (Optuna):
```bash
# Via experiment config: set model.tune: true and model.tune_trials: 50
# Or directly:
uv run python scripts/xgboost_v2.py --tune --n-trials 50
```

**Neural models** (LSTM/Transformer/MLP): Adjust hyperparameters in the experiment config under `model.hyperparameters`, or pass CLI args directly to the training script.

---

### Testing Checklist

When making changes, verify:

1. **Standalone scripts still work** (backward compatibility):
   ```bash
   uv run python scripts/xgboost_v2.py  # No --config-json
   ```

2. **Experiment runner works**:
   ```bash
   uv run python scripts/run_experiment.py experiments/configs/xgb_v3_baseline.yaml --dry-run
   ```

3. **Feature registry is consistent** (features resolve correctly):
   ```bash
   uv run python -c "from scripts.feature_registry import resolve_feature_list, V3_GROUPS; print(len(resolve_feature_list(V3_GROUPS)))"
   ```

4. **Simulated scores are realistic** (avg ~155-165 for T20s after bug fix):
   ```bash
   uv run python scripts/sim_eval/run_sim_eval.py --model-type xgboost --test-dir data/betting_test --odds betting_odds_v3.json --n-sims 10 --max-matches 3
   ```

5. **Class mapping is correct** (6 classes: dot, one, two, four, six, wicket):
   - In `sim_v1_2.py`, every `class_to_outcome` dict must have exactly 6 entries
   - Class 4 = 'six', Class 5 = 'wicket' (NOT the other way around)

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
| XGBoost Training | 30-60 min | 8-16 GB | With Optuna (50 trials) |
| LSTM Training | 30-60 min | 4-8 GB | Full dataset, 50 epochs |
| LSTM Training (quick) | ~2 min | 2-4 GB | 5% data, 2 epochs |
| Stats Cache Load | 1-2 sec | 300-550 MB | Lazy loading (5 chunks max) |
| Single Match Sim (XGBoost) | 0.01-0.1 sec | ~2 GB | 1000 simulations, parallel |
| Single Match Sim (LSTM) | ~35 sec | ~2 GB | 100 simulations, sequential |
| Full Evaluation (45 matches) | 5-10 min | ~2-3 GB | XGBoost, 1000 sims/match |

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
torch>=2.0.0          # For LSTM model
pyarrow>=12.0.0       # For parquet files
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
