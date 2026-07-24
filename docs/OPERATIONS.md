# Operations Guide

Complete guide for running all pipelines and common operations in the CricML Match Prediction system.

---

## Quick Start

### Training Pipeline (XGBoost - Default)

**Recommended: one command via `run_experiment.py`**
```bash
uv run python scripts/run_experiment.py experiments/configs/xgb_v3_baseline.yaml
```
Dispatches build-cache → materialize → train → eval. Each step is skipped if its artifact is current (`check_smart_cache` inspects SQLite `_meta.schema_version` + `_meta.source_json_mtime_max` and parquet `.feature_hash`).

**Explicit step-by-step (same four steps, lets you run them in isolation):**
```bash
# Step 1a: Build SQLite stats cache (~6 min on full corpus).
# Idempotent — no-ops if the JSONs haven't changed since the last build.
uv run python scripts/build_stats_cache.py
# Force a full rebuild with --force-rebuild (e.g. after a schema change).

# Step 1b: Materialize feature parquet (~3 min; reads SQLite + JSON).
# Per-date batching with tracker rehydration from SQLite.
uv run python scripts/materialize_features.py \
    --config experiments/configs/xgb_v3_baseline.yaml

# Step 2: Model training (~5-10 min with pinned hyperparameters; ~30-60 min with --tune).
uv run python scripts/xgboost_v2.py

# Step 3: Evaluation (~5-10 min for 45-match betting_test × 1000 sims;
# ~40 min for 261-match polymarket_test × 100 sims).
uv run python scripts/sim_eval/run_sim_eval.py \
    --test-dir data/polymarket_test \
    --odds betting_odds_polymarket.json \
    --n-sims 100
```

> **Phase B note (2026-04-22)**: `scripts/parsing_v2.py` was split into `scripts/build_stats_cache.py` + `scripts/materialize_features.py`. The monolithic orchestrator `process_folder_v2_with_splits` is gone; the helper primitives (tracker classes, `parse_match_data_v2`, `deep_copy_stats`) remain in `parsing_v2.py` and are imported by the new scripts.

> **`--parallel` on `run_sim_eval.py` works but isn't faster.** As of the
> 2026-05-08 perf pass it produces correct output (the pickle bug was
> fixed by `e4b97cc`), but multiprocessing.Pool over a single match's sims
> is dominated by IPC cost — 5×100 takes 76 s vs 44 s serial. The
> SQLite-unlocked parallelism the user wants is **multiple eval processes
> on disjoint match subsets**; see "Multi-process parallel eval" below.
> Budget for serial runs: ~34 min for 261×100 (post-perf-pass), ~5–10 min
> for 45×1000. Launch long runs as background tasks rather than polling.

### Multi-process parallel eval (2026-05-09)

The SQLite stats cache supports N concurrent readers via mmap (proven in
the Phase 1+2 migration with 1.7 GB combined RSS). The 2026-05-08 perf
pass added per-player memoization to `StatsProviderCache` that survives
fork/pickle, and a one-time `__getstate__` / `__setstate__` fix that
unblocked the multiprocessing path. Result: a 4-proc parallel eval on
disjoint match subsets gives **~2.3× throughput** at ~1.6 GB combined RSS
(per-process ~440 MB).

**Critical**: cap each child's BLAS/OMP threads or processes will
oversubscribe and serialize. Without the cap, 2 procs on 10 logical
cores took *longer* than serial.

```bash
# Driver + RSS sampling:
uv run python perf_runs/run_n_parallel.py <N_PROCS> <MATCHES_PER_PROC> <N_SIMS>

# Examples — 10 logical cores on the dev box:
uv run python perf_runs/run_n_parallel.py 2 -1 100   # 2 procs, full 261, OMP=5
uv run python perf_runs/run_n_parallel.py 4 -1 100   # 4 procs, full 261, OMP=2
# matches_per_proc == -1 splits the test set evenly; last proc takes the
# remainder.
```

Measured throughput (10 matches × 100 sims per proc, 10 logical cores):

| Config                  | Wall    | Throughput | Combined RSS |
|-------------------------|--------:|-----------:|-------------:|
| 1 proc, default OMP     | 86.3 s  | 1.00×      | 887 MB       |
| 2 procs, OMP=4 (no cap) | 173.7 s | **0.99×** (oversub!) | 1.1 GB |
| **2 procs, OMP=2**      | 96.3 s  | **1.79×**  | 890 MB       |
| **4 procs, OMP=2**      | 148 s   | **2.33×**  | 1.6 GB       |

End-to-end, full 261×100 eval (2026-05-09): 4 procs × OMP=2 ran in
**16.6 min** vs **41.9 min serial = 2.52× speedup**. Numerics within
Monte Carlo noise of the serial baseline (LL 0.7150 vs 0.7155).

The 16 GB box has comfortable headroom past 4 procs, but compute-bound
diminishing returns kick in (memory bandwidth, perf vs efficiency cores).
**Recommended**: 2 procs for routine ablation iteration, 4 procs for
biggest-experiment final runs. Do NOT mix `--parallel` (intra-match) with
multiprocess parallelism — the `OMP_NUM_THREADS` cap is the only safe knob.

### Sliced eval (Phase 1, 2026-04-24)

Use `--min-volume` to filter polymarket entries by `polymarket_volume_usd`. The plan's go/no-go gate is the **≥$50k slice** (170 matches); ≥$100k (110 matches) is a tighter sharp-market check.

```bash
# Single slice, with tournament-block 95% CIs on log loss + flat ROI.
uv run python scripts/sim_eval/run_sim_eval.py \
    --test-dir data/polymarket_test \
    --odds betting_odds_polymarket.json \
    --n-sims 100 --min-volume 50000 \
    --bootstrap-resamples 10000 \
    --output-dir eval_out/sliced

# All 3 slices in one shot (all / >=$50k / >=$100k):
bash scripts/run_sliced_eval.sh

# Cross-model comparison table:
uv run python scripts/sim_eval/compare_slices.py \
    --group "v6" eval_out/phase1_sliced/xgboost_*_*.json \
    --group "v7" eval_out/phase5_hier/*.json

# Post-hoc reslice an existing eval JSON (avoids re-running sims):
uv run python scripts/sim_eval/reslice_eval_json.py \
    --in  eval_out/postfix/xgboost_20260421_220541.json \
    --odds betting_odds_polymarket.json \
    --out-dir eval_out/phase1_sliced_v4 \
    --cluster-source-dir data/polymarket_test
```

YAML wiring: set `evaluation.min_volume` and `evaluation.bootstrap_resamples` in the experiment config to make the auto-eval inside `run_experiment.py` produce a single sliced result. To get all three slices, run `run_sliced_eval.sh` after training completes (use `--skip-training` or fresh `--only-eval` invocations).

I3 contract: `eval_statistics.py` groups contiguous fixtures sharing the
Cricsheet event name, splitting a later edition after more than 120 inactive
days. Missing event metadata falls back to unordered team pair plus cricket
season. The bootstrap samples whole blocks, uses 10,000 resamples and seed 42,
and records metadata coverage/effective cluster count in JSON. Fewer than 10
betting blocks is descriptive only. `realized_pnl` is never a bet-placement
sentinel; new output persists `bet_placed` and `bet_team`.

### Prop-bet backtest + bowler selector (2026-05-12)

The v7 sim doubles as a prop-bet engine. First build the phase-usage prior
the empirical bowler selector reads (idempotent; skip if current):

```bash
uv run python scripts/build_bowler_phase_usage.py \
    --source-dir data/t20s_json --out models/bowler_phase_usage.json
```

Backtest prop families against cricsheet actuals (Brier-skill + MAE +
bootstrap CIs), then render per-match views and an empirical-vs-random A/B:

```bash
# Aggregate prop report (writes report .md + detail .json sidecar).
uv run python scripts/sim_eval/prop_backtest.py \
    --test-dir data/polymarket_test --n-sims 100 \
    --out reports/prop_calibration_report_emp_n261.md

# Per-match drilldowns (one .md per match + index.md hit/miss table).
uv run python scripts/sim_eval/render_prop_per_match.py \
    --detail reports/prop_calibration_detail_emp_n261.json \
    --out-dir reports/prop_per_match/

# Selector A/B: run prop_backtest twice (--bowler-selector via run_sim_eval,
# or RandomBowlerSelector default in a comparison run) then diff:
uv run python scripts/sim_eval/compare_selector_eval.py \
    --left  reports/prop_calibration_detail_emp_n60.json \
    --right reports/prop_calibration_detail_rand_n60.json \
    --left-label empirical --right-label random \
    --out reports/prop_selector_comparison_n60.md

# Gate G5: bowler coverage (fraction with >=N historical balls as-of date).
uv run python scripts/sim_eval/check_bowler_coverage.py \
    --test-dir data/polymarket_test \
    --usage models/bowler_phase_usage.json --threshold 100
```

`run_sim_eval.py` takes `--bowler-selector {empirical,random}` (default
`empirical`) and `--bowler-usage-path` to point at a different prior.
`EmpiricalBowlerSelector` is also the default inside bare `T20Rules()` /
`SimulationEngine`; construct `T20Rules(RandomBowlerSelector())` to opt out.
Findings summary: `reports/prop_framework_summary.md`.

### Outcome-distribution k overrides (Phase 6, 2026-04-25)

The shrinkage strength on the 42 outcome-dist features is controlled via a top-level `outcome_dist:` block in the experiment YAML:

```yaml
outcome_dist:
  k_player: 30.0    # batter / bowler / vs-type / vs-hand cells
  k_venue:  200.0   # venue cells (more data → larger prior weight)
```

Defaults (30 / 200) are the Phase 6 sweep optimum and match the Phase 5 hierarchical-shrinkage baseline. The values get baked into the parquet `.feature_hash` payload (so a k change invalidates the parquet cache) and written to `models/xgb_v3/outcome_dist_config_v3.json` at train time. `XGBoostModelV2.__init__` reads the sidecar at sim time, so training and inference always use matching k values.

### Training Pipeline (LSTM)
```bash
# Step 1: Feature engineering (same two-step as XGBoost)
uv run python scripts/build_stats_cache.py
uv run python scripts/materialize_features.py

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

**After a successful refresh**, re-run the two parsing scripts to regenerate the
stats cache and training splits:

```bash
uv run python scripts/build_stats_cache.py       # ~7 min; rebuilds models/player_stats_cache_v3.sqlite (schema v4, ~57 MB)
uv run python scripts/materialize_features.py    # ~5 min; rebuilds data/xgb_data_v3/ (105 cols incl. 42 outcome-dist features)
```

Or run everything (including training + eval) via:

```bash
uv run python scripts/run_experiment.py experiments/configs/xgb_v3_baseline.yaml
```

The fetcher does **not** auto-chain because the cache rebuild is destructive
(overwrites the SQLite).

**New unenriched players.** The fetcher compares cricsheet IDs in the newly-added
matches against `data/all_players_enriched.csv` and prints any IDs that aren't in
the enriched metadata. Running `enrich_players_cricketdata.py` (R `cricketdata`
via `rpy2`, see next section) to fill those in is a manual follow-up, not part
of the fetcher. The legacy HTML scrapers (`cricinfo_scraper{,_v2,_v3}.py`) are
under `archive/scripts/` and are no longer the recommended path.

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
uv run python scripts/build_stats_cache.py            # 3a. rebuild SQLite stats cache (destructive)
uv run python scripts/materialize_features.py        # 3b. rematerialize feature parquet
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

# Load model (v3 / v6 paths)
stats_provider = StatsProvider('models')   # auto-uses SQLite v4
model = XGBoostModelV2(
    'models/xgb_v3/xgboost_model_v3.pkl',
    'models/xgb_v3/batter_encoder_v3.pkl',
    'models/xgb_v3/bowler_encoder_v3.pkl',
    'models/xgb_v3/feature_columns_v3.txt',
    stats_provider=stats_provider,
)
engine = SimulationEngine(model, T20Rules())

# Run simulation (see examples below for full code)
```

---

## Training Pipeline

### 1. Feature Engineering — Phase B split pipeline

**Post-Phase-B (2026-04-22)**: the monolith `parsing_v2.py` was split into two scripts. The stateful stats cache is built once; feature materialization reads from it and replays deliveries per-date. See CLAUDE.md §"Core Modules" for the full rationale.

**Input** (both scripts): `data/t20s_json/*.json` — 9,500+ raw match files.

**Outputs**:
- `models/player_stats_cache_v3.sqlite` (schema v4, ~57 MB) — from `build_stats_cache.py`. Filename retained for backwards compatibility; `_meta.schema_version` is the source of truth.
- `data/xgb_data_v3/{train,validation,test,golden_test}.parquet` + `.feature_hash` — from `materialize_features.py`. Includes 42 empirical-Bayes-shrunk outcome-distribution features under schema v4.

**Commands**:
```bash
# 1a. Build SQLite stats cache (chronological tracker walk).
uv run python scripts/build_stats_cache.py
# Options: --source-dir data/t20s_json, --out models/player_stats_cache_v3.sqlite,
#          --extra-source-dir (repeatable), --prior-source-sqlite,
#          --gender-filter male, --force-rebuild.
# Idempotent only when schema, deterministic ordering version, exact source
# directory list/file count, and max JSON mtime all match.

# 1b. Materialize feature parquet (per-date batching, SQLite rehydration).
uv run python scripts/materialize_features.py \
    --config experiments/configs/xgb_v3_baseline.yaml
# Options: --source-dir, --sqlite-dir, --out-dir, --version, --gender-filter.
# Ships serial; per-date ProcessPoolExecutor parallelism is a follow-up TODO.
```

**Performance (full corpus, schema v4, 2026-04-23)**:
- `build_stats_cache.py`: **~7 min**, ~4–8 GB RAM, 9,519 matches → 56.8 MB SQLite. Includes a one-pass global empirical-outcome-prior computation written to `_meta.prior_p*`.
- `materialize_features.py`: **~5 min**, ~2 GB RAM, 9,519 matches → 2.2M ball records across 4 parquets, 105 columns (63 + 42 outcome-dist). **~3× faster than the old monolith** because the tracker walk is skipped.

**What it does**:
1. **`build_stats_cache.py`**: Loads matches in versioned `(date, match_id)` order, runs `PlayerStatsTracker` + `PlayerEloTracker` + `VenueStatsTracker`, takes first-write-wins snapshots per date, emits delta-compressed rows to SQLite, and writes one `batting_match_log` / `bowling_match_log` row per (player, match) for recent-form reconstruction. Multiple non-overlapping source directories can be merged; duplicate match IDs fail closed. Schema v4 also writes 6 outcome-count columns (`c0..cw`) per row on `batting`, `bowling`, `batting_vs_type`, `bowling_vs_hand`, `venue`, plus the global prior π in `_meta`. `--prior-source-sqlite` freezes the global/phase priors from an earlier cache for forward state. Two integrity checks run before close: `_verify_log_denormalized_consistency` (deque-vs-sum) and `_verify_outcome_count_conservation` (Σ cX ≡ balls).
2. **`materialize_features.py`**: Groups matches by date, rehydrates trackers from SQLite once per date using the union of same-day players + venues, loads π from `_meta` once at startup, replays same-day matches in deterministic match-ID order, calls `parse_match_data_v2(..., prior=π)` per match, and writes per-split parquets. It requires the cache ordering metadata and fails on a legacy/unversioned cache. Splits come from the YAML `data.splits` block (falls back to the hardcoded defaults).

### Forward holdout state (never training input)

```bash
uv run python scripts/build_forward_state.py \
  --holdout-dir data/forward_holdout/2026-06-01_2026-07-13

uv run python scripts/verify_forward_state.py \
  data/forward_state/2026-06-01_2026-07-13
```

This builds an immutable sidecar under `data/forward_state/`, not the
production cache. It merges the historical corpus with the sealed holdout's
chronological context, freezes priors from the pre-holdout production SQLite,
materializes match features, verifies every selected fixture, and writes
`NO_MODEL_SCORING`. It does not import a model. See
`FORWARD_HOLDOUT.md` and `I6_SAME_DAY_ORDERING_AUDIT.md`.

The two prediction adapters remain blocked while the protocol is `DRAFT`:

```bash
uv run python scripts/score_forward_match_m7.py \
  evaluation/forward_protocol_2026-06-01_2026-07-13.yaml \
  --out forward_eval_out/2026-06-01_2026-07-13/match_m7_predictions.json

uv run python scripts/score_forward_ball_v7.py \
  evaluation/forward_protocol_2026-06-01_2026-07-13.yaml \
  --out forward_eval_out/2026-06-01_2026-07-13/ball_v7_predictions.json
```

The ball adapter rehydrates first-of-date state, builds lineups from
`info.players` only, locks each selected prediction before replaying its
completed match, and never writes to the sidecar SQLite.

**Parity guarantee**: `scripts/tests/test_phase_a_parity.py` validates that this two-step pipeline produces bit-exact parquet output vs the original monolith across all 9,519 matches. The harness now also passes π into both reference and candidate paths so the 42 outcome-distribution columns are checked column-by-column.

**Temporal Splits** (`scripts/loaders_common.py:DEFAULT_SPLITS`, override via
YAML `data.splits`):
- Train: matches with date < `train_end` (default `2024-12-31`)
- Validation: `train_end ≤ date < val_end` (default `< 2025-06-30`)
- Test: `val_end ≤ date < golden_start` (default `< 2026-04-17`)
- Golden test: `date ≥ golden_start` (default `≥ 2026-04-17`, currently sparse;
  it's the post-iteration holdout — see [TODO.md](../TODO.md) "Preserve a true
  holdout").

---

### 2. Model Training (`xgboost_v2.py`)

**Purpose**: Train XGBoost classifier with optional Optuna hyperparameter
tuning.

**Input**:
- `data/xgb_data_v3/cricket_data_v3_{train,validation,test,golden_test}.parquet`

**Output** (saved to `models/xgb_v3/`):
- `xgboost_model_v3.pkl`            — trained model
- `xgboost_model_v3_optimized.pkl`  — only when `--tune`
- `batter_encoder_v3.pkl`, `bowler_encoder_v3.pkl`
- `feature_columns_v3.txt`
- `optuna_study_v3.pkl`             — only when `--tune`

**Command**:
```bash
uv run python scripts/xgboost_v2.py                       # pinned hyperparams (~5-10 min)
uv run python scripts/xgboost_v2.py --tune --n-trials 50  # Optuna tune (~30-60 min)
```

**Performance**:
- Pinned hyperparameters: ~5–10 min, ~4–8 GB.
- With Optuna 50 trials: ~30–60 min, ~8–16 GB.
- Ball-level accuracy: ~55–60 % (6-class).

**What It Does**:
1. Loads the four parquet splits.
2. Resolves the feature list from `--config-json` (set by
   `run_experiment.py`) or falls back to `V6_GROUPS` defaults.
3. Encodes player IDs (fit LabelEncoders on union of all splits).
4. Remaps outcomes to 6 classes (`{0,1,2,4,6,-1} → {0,1,2,3,4,5}`).
5. Computes balanced class weights via `compute_class_weight('balanced')`.
6. Optionally: Optuna TPE sampler over 7 hyperparameters, optimize val log loss.
7. Trains final model with best params + early stopping (100 rounds).
8. Evaluates on test split and saves artifacts.

**Model Configuration**:
- 6-class classifier (dot, 1, 2, 4, 6, wicket)
- **114 input features** under `V6_GROUPS` (V3's 72 + 42 outcome-distribution).
  Trainers automatically filter to columns present in the parquet
  (`feature_cols = [c for c in features if c in df.columns]`).
- Balanced class weights
- Early stopping (100 rounds)

---

> **Note on legacy `data/xgb_data/` (v2)**: the older 29-feature parquet path
> is unmaintained. All current experiments use `data/xgb_data_v3/`.
> See [docs/archive/](archive/) for the v2 history.

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
- `--model-version`: XGBoost version (only `v3` is supported; v2 artifacts were deleted in the 2026-04-26 cleanup)
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

# Step 1: Load model components (v3 / v6)
stats_provider = StatsProvider('models')   # auto-uses SQLite v4
model = XGBoostModelV2(
    'models/xgb_v3/xgboost_model_v3.pkl',
    'models/xgb_v3/batter_encoder_v3.pkl',
    'models/xgb_v3/bowler_encoder_v3.pkl',
    'models/xgb_v3/feature_columns_v3.txt',
    stats_provider=stats_provider,
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

**Step 1: Add the feature to `scripts/parsing_v2.py` (helper module)**

Find `parse_match_data_v2` in `scripts/parsing_v2.py` — the ball-processing loop lives there, and both `build_stats_cache.py` and `materialize_features.py` import it. Add your feature computation respecting temporal integrity (the feature must only use data available before the ball is bowled).

If the feature requires a new kind of stats snapshot (e.g., a new `batting_vs_*` variant), also extend `PlayerStatsTracker` / `deep_copy_stats` in the same file and the SQLite schema in `scripts/stats_sqlite_backend.py` — then bump `SCHEMA_VERSION` and rebuild.

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
# If the feature added new tracker state / snapshot fields, rebuild the cache:
uv run python scripts/build_stats_cache.py --force-rebuild

# Otherwise just rematerialize the parquet (the feature is ball-context-only):
uv run python scripts/materialize_features.py

# Create an experiment config that includes your new feature group
# (copy an existing config and add your group to features.groups)

# Run experiment — run_experiment.py autodetects cache/parquet staleness and
# skips whichever step is already current.
uv run python scripts/run_experiment.py experiments/configs/your_config.yaml --skip-training

# Or test training manually
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

# 2. Retrain and evaluate — run_experiment.py autodetects the new JSONs
#    (via _meta.source_json_mtime_max) and rebuilds both SQLite + parquet.
uv run python scripts/run_experiment.py experiments/configs/xgb_v3_baseline.yaml

# --- OR run the parsing steps manually ---
# uv run python scripts/build_stats_cache.py        # SQLite (~6 min)
# uv run python scripts/materialize_features.py     # parquet (~3 min)
# uv run python scripts/xgboost_v2.py               # train (~5-10 min)
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

# 2. Re-run feature engineering (Phase B split: SQLite cache → parquet)
uv run python scripts/build_stats_cache.py
uv run python scripts/materialize_features.py

# 3. Re-train model
uv run python scripts/xgboost_v2.py

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

provider = StatsProvider('models')          # SQLite v4 backend

# Backend info
print(f"Backend: {provider.backend_name}, version: {provider.version}")
print(f"Date range: {provider.dates[0]} to {provider.dates[-1]}")
print(f"Snapshots: {len(provider.dates):,}")

# Inspect _meta (schema_version, build_timestamp, prior π, etc.)
meta = provider._backend.get_meta()
for k in sorted(meta):
    print(f"  {k}: {meta[k]}")

# Query specific player
player_id = "253802"  # Rohit Sharma
date = "2024-06-15"

batting = provider.get_batting_stats(player_id, date)
bowling = provider.get_bowling_stats(player_id, date)
recent  = provider.get_batting_recent(player_id, date)

print(f"\nPlayer {player_id} stats as of {date}:")
print(f"  Career: Avg={batting['avg']:.1f}, SR={batting['sr']:.1f}")
print(f"  Last 5: Avg={recent['avg']:.1f},  SR={recent['sr']:.1f}")
print(f"  Bowling: Avg={bowling['avg']:.1f}, Econ={bowling['econ']:.1f}")

# Schema-v4 outcome distributions
dist = provider.get_batter_outcome_dist(player_id, date)
print(f"  Outcome dist (shrunk): {dist}")

# H2H matchup
bowler_id = "290630"
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

### Operation 6: Predict an upcoming fixture (live)

For a genuine upcoming fixture, use the match-level direct model via
`predict_fixture.py`. Hand-write a fixture JSON (see
`fixtures/_template.json`), then:

```bash
uv run python scripts/predict_fixture.py --fixture fixtures/<id>.json
```

Lineup entries may be either cricsheet 8-char IDs or display names
(names are resolved against `data/all_players_enriched.csv`).
Polymarket odds are optional; when provided the script also reports
edge and bet recommendation. The standing A7 forward policy is flat one-unit
staking, betting all close fixtures (`|elo_diff| <= 5`) and requiring model
edge strictly above 10% on mismatch fixtures. Historical economic confidence
is unconfirmed under I3 competition-block resampling; keep this policy fixed
for forward evaluation rather than retuning it.

**The 2026-04-16 staleness caveat.** `predict_fixture.py` uses the
production SQLite cache (`models/player_stats_cache_v3.sqlite`) +
tracker snapshot from `data/t20s_json`, both deliberately frozen at
test_end (2026-04-16) to keep the iteration / golden eval sets
out-of-sample. For a fixture past that date, all per-player ELOs,
recent-form, and venue trackers see only pre-cutoff history. This
is fine for short-horizon fixtures but can materially shift the
number on longer-horizon ones (e.g. RR vs GT IPL 2026 Qualifier 2,
2026-05-29: stale cache gave RR 67.2%, post-refresh through
2026-05-24 gave RR 56.5% — recent-form was the missing signal).

**Refreshed-state prediction (non-destructive).** To predict with
post-cutoff data without contaminating the held-out train/eval
artifacts, use the duplicate workflow in `tmp/golden_inclusive/`:

```bash
# 1. Pull new T20 cricsheet JSONs from stat-generator into the
#    golden pool. Skips anything already in data/t20s_json (the
#    frozen local pool) — never mutates it.
uv run python scripts/extract_golden_cricsheet.py

# 2. Sync new golden files into the tmp combined pool that feeds
#    the duplicate SQLite cache.
cp -n data/golden/t20s_json/*.json tmp/golden_inclusive/t20s_combined/

# 3. Rebuild the duplicate SQLite cache (~7 min). The production
#    cache at models/player_stats_cache_v3.sqlite is left alone.
uv run python scripts/build_stats_cache.py \
    --source-dir tmp/golden_inclusive/t20s_combined \
    --out tmp/golden_inclusive/player_stats_cache_v3.sqlite \
    --metadata-csv data/all_players_enriched.csv \
    --force-rebuild

# 4. Predict, rebuilding the combined tracker snapshot in the
#    process. Same production model (xgb_match_v3_m7_production);
#    only the feature inputs are refreshed, no retrain.
uv run python tmp/golden_inclusive/predict_with_refreshed_state.py \
    --fixture fixtures/<id>.json --rebuild-snapshot
```

Outputs land in `tmp/golden_inclusive/predictions/`. Drop
`--rebuild-snapshot` on subsequent runs against the same data state.

The discipline: never add post-2026-04-16 files to `data/t20s_json`
or rebuild `models/player_stats_cache_v3.sqlite` in place. Doing
so contaminates the iteration + golden test sets that document the
production model's ROI numbers.

---

## Performance Benchmarks

| Operation | Duration | Memory | Notes |
|-----------|----------|--------|-------|
| Feature Engineering | 10-15 min | 4-8 GB | Full dataset (15K matches) |
| XGBoost Training | 30-60 min | 8-16 GB | With Optuna (50 trials) |
| LSTM Training | 30-60 min | 4-8 GB | Full dataset, 50 epochs |
| LSTM Training (quick) | ~2 min | 2-4 GB | 5% data, 2 epochs |
| Stats Cache Load | <100 ms | <50 MB resident | SQLite mmap, ~3 µs p50 query |
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
encoder = joblib.load('models/xgb_v3/batter_encoder_v3.pkl')
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
# Quick mitigation: subsample for iteration
import pandas as pd
df_train = pd.read_parquet(
    'data/xgb_data_v3/cricket_data_v3_train.parquet'
).sample(frac=0.1, random_state=0)
```

For chunked loading you'd swap pandas → pyarrow's `iter_batches`. The full
training parquet is ~1.7 M rows × 105 cols (~600 MB in memory) and fits
comfortably in 8 GB. If you OOM, the more likely culprit is XGBoost's
internal DMatrix doubling memory during training rather than parquet load.

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

**For architecture, key classes, data formats, and design rationale, see [ARCHITECTURE.md](ARCHITECTURE.md).**
**For adding a new model type, see [ADDING_NEW_MODELS.md](ADDING_NEW_MODELS.md).**
