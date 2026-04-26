# Architecture

Comprehensive technical reference for CricML. This is the canonical doc for how
the system is built, how data flows through it, and why each major decision was
made. Reflects the Phase B parsing split (2026-04-22), the v6 outcome-
distribution feature pass under schema v4 (2026-04-23), and the Phase 5
(hierarchical shrinkage, 2026-04-25) + Phase 6 (k-sweep, k=30 won) work that
together produced the active **v7** XGBoost model.

For day-to-day commands see [OPERATIONS.md](OPERATIONS.md). For adding new model
types see [ADDING_NEW_MODELS.md](ADDING_NEW_MODELS.md).

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Module Inventory](#2-module-inventory)
3. [Data Flows](#3-data-flows)
4. [Key Classes & Interfaces](#4-key-classes--interfaces)
5. [Data Formats](#5-data-formats)
6. [Design Decisions](#6-design-decisions)

---

## 1. System Overview

Three pipelines, three artifact tiers:

```
┌─────────────────────────────────────────────────────────────────────┐
│ TRAINING                                                            │
│                                                                     │
│  data/t20s_json/ ──► build_stats_cache.py ──► SQLite v4 (~57 MB)    │
│                                                  │                  │
│                                                  ▼                  │
│           materialize_features.py + JSON ──► parquet (4 splits)     │
│                                                  │                  │
│                                                  ▼                  │
│                  xgboost_v2.py / lstm_v1.py / ──► models/<m>_v*/    │
│                  transformer_v1.py / mlp_v1.py                      │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│ SIMULATION                                                          │
│                                                                     │
│  Match JSON + lineup ──► MatchState                                 │
│         │                                                           │
│         ▼                                                           │
│  SimulationEngine(model, T20Rules)                                  │
│    ├─ extract_features ──► predict_next_ball ──► sample outcome     │
│    ├─ T20Rules.process_ball ──► state.update                        │
│    └─ loop until innings/match end                                  │
│         │                                                           │
│         ▼                                                           │
│  ResultAggregator ──► win probabilities + score distributions       │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│ EVALUATION                                                          │
│                                                                     │
│  data/{betting,polymarket}_test/ + odds.json                        │
│         │                                                           │
│         ▼                                                           │
│  TestMatchLoader ──► MatchLevelEvaluator ──► OverallEvaluationResults│
│       │                     │                                       │
│       │                     ├─ log loss / Brier / edge              │
│       │                     ├─ flat / Kelly / fractional Kelly P&L  │
│       │                     └─ calibration bins                     │
│       ▼                                                             │
│  match_evaluation_results_<model>_<ts>.json                         │
└─────────────────────────────────────────────────────────────────────┘
```

**Why ball-level + Monte Carlo?** Direct match prediction has ~11 K training
matches; ball-level has ~2.2 M training balls (~200×). Simulation variance is
free uncertainty. See §6 for the full rationale.

**Critical invariants** (don't break these):
- Features reflect state **before** the ball; trackers update **after**.
- 6-class outcome space: `{0:dot, 1:one, 2:two, 3:four, 4:six, 5:wicket}`.
- SQLite snapshot for date `D` reflects only matches with date `< D`
  (first-write-wins).
- Schema-changing edits to the SQLite cache require bumping `SCHEMA_VERSION`
  in `stats_sqlite_backend.py`.
- `--parallel` on `run_sim_eval.py` has crashed the 16 GB test box; default
  is serial.

---

## 2. Module Inventory

All paths relative to repo root. Line counts are approximate, kept here as a
quick "scope sniff."

### 2.1 Parsing & feature pipeline

| File | Lines | Role |
|---|---:|---|
| `scripts/build_stats_cache.py` | ~600 | Chronological JSON walk → SQLite schema v4. Writes delta-compressed snapshots, outcome counts (`c0..cw`), match-log rows, and global prior π. Idempotent via `_meta.source_json_mtime_max`. |
| `scripts/materialize_features.py` | ~310 | Per-date batched parquet emission. Rehydrates `temp_*` trackers from SQLite once per date, replays same-day matches in monolith order, writes 4 splits + `.feature_hash` marker. |
| `scripts/parsing_v2.py` | ~1140 | Tracker primitives: `PlayerStatsTracker`, `PlayerEloTracker`, `VenueStatsTracker`, `InningsFeatureCalculator`, `parse_match_data_v2`, `deep_copy_stats`, `classify_match_k_factor`. The orchestrator was deleted in Phase B; only helpers remain. |
| `scripts/tracker_rehydration.py` | ~350 | Seeds `temp_*` trackers from SQLite at a date boundary using the new private accessors on `_SQLiteBackend`. |
| `scripts/loaders_common.py` | ~125 | `iter_matches_chronological`, `extract_match_metadata`, `DEFAULT_SPLITS`, `effective_splits`. Shared by parser + tests. |
| `scripts/feature_registry.py` | ~225 | Central feature catalog. 16 groups, `V3_GROUPS` / `V5_GROUPS` / **`V6_GROUPS`** convenience exports, `resolve_feature_list`, `get_feature_hash`. |
| `scripts/player_metadata.py` | ~290 | `PlayerMetadataProvider` — hand/arm/age/bowling-style lookups from `data/all_players_enriched.csv`. |

### 2.2 Stats cache backend

| File | Lines | Role |
|---|---:|---|
| `scripts/stats_sqlite_backend.py` | ~945 | `_SQLiteBackend` — mmap reader. 11 tables (see §5.3). Ten public getters, raw-counter accessors used by `tracker_rehydration.py`, fork-safe via `_ensure_conn` PID check. `SCHEMA_VERSION = 4`. |
| `scripts/stats_provider.py` | ~225 | `StatsProvider` — facade over the backend. `StatsProviderCache` — per-instance memo for the 5 lineup/venue-keyed methods (~1.2× sim speedup). `wrap_with_cache` helper. |

### 2.3 Training scripts

| File | Lines | Role |
|---|---:|---|
| `scripts/xgboost_v2.py` | ~430 | XGBoost 6-class classifier with optional Optuna tuning. Reads parquet, fits encoders, applies balanced class weights, saves to `models/xgb_v3/`. |
| `scripts/lstm_v1.py` | ~33 K bytes | PyTorch LSTM. Sliding-window sequence dataset, 2-layer LSTM with embeddings (batter 64d / bowler 64d / venue 32d / matchup 16d). Saves to `models/lstm_v1/`. |
| `scripts/mlp_v1.py` | ~15 K bytes | Simple MLP baseline. Saves to `models/mlp_v1/`. |
| `scripts/transformer_v1.py` | ~1180 | Transformer over full 120-ball innings context. PyTorch + MLX backend (`--mlx`) for Apple Silicon. Saves to `models/transformer_v1/`. |

### 2.4 Simulation engine

| File | Lines | Role |
|---|---:|---|
| `scripts/sim_v1_2.py` | ~3660 | One file holds: data classes (`Outcome`, `Player`, `TeamLineup`, `MatchState`), rules engine (`T20Rules`, `BowlerSelector`), the abstract `PredictionModel` interface, all five concrete model wrappers (`XGBoostModelV2`, `LSTMModelV1`, `MLPModelV1`, `MLPModelV2`, `TransformerModelV1`, `LLMModelV1`), the simulation orchestrator (`SimulationEngine`), and result types (`BallResult`, `InningsResult`, `MatchResult`, `ResultAggregator`). The 5-way duplicated feature-assembly blocks are tracked debt — see [TODO.md](../TODO.md). |

### 2.5 Evaluation framework (`scripts/sim_eval/`)

| File | Lines | Role |
|---|---:|---|
| `loaders.py` | ~310 | `TestMatchLoader` (JSON → `MatchState`), `BettingOddsLoader` (decimal odds → margin-free probabilities). |
| `match_evaluator.py` | ~1010 | `MatchLevelEvaluator` — runs sims per match, computes log loss / Brier / edge / calibration / flat & Kelly P&L. Optional Platt + isotonic calibration via `--calibrate` / `--ball-calibrate`. |
| `run_sim_eval.py` | ~22 K bytes | CLI entrypoint. `--model-type {xgboost,lstm,mlp,transformer}`, `--model-version`, `--n-sims`, `--max-matches`, `--mlx`, calibration flags. |

### 2.6 Experiment infrastructure

| File | Role |
|---|---|
| `scripts/run_experiment.py` | YAML-driven pipeline runner. Smart cache: SQLite valid iff `_meta.schema_version == 4` ∧ `source_json_mtime_max ≥ max(JSON mtime)`; parquet valid iff `feature_hash` + `splits` + `gender_filter` match and parquet mtime ≥ SQLite mtime. Dispatches `build_stats_cache.py` / `materialize_features.py` independently. |
| `scripts/experiment_tracker.py` | Per-experiment directory under `experiments/results/<name>_<ts>_<git>/` with `config.yaml`, `metadata.json`, `metrics.json`, `console_output.log`. |
| `scripts/compare_experiments.py` | List / filter / show / side-by-side compare experiments. |
| `experiments/configs/*.yaml` | Declarative experiment definitions. Active: `xgb_v3_baseline`, `xgb_v6_outcome_dist`, `lstm_v1_baseline`, `transformer_v1_baseline`. Others kept for reproducibility of past experiments. |

### 2.7 Data ingestion

| File | Role |
|---|---|
| `scripts/fetch_cricsheet.py` | Idempotent Cricsheet refresh. 14 men's T20 leagues + player register. SHA-256 manifest, append-only JSON merge. |
| `scripts/enrich_players_cricketdata.py` | Fills `data/all_players_enriched.csv` via R `cricketdata` package over `rpy2`. Uses `key_cricinfo` from Cricsheet's `people.csv`. |
| `scripts/build_polymarket_odds.py` | Matches Polymarket markets → Cricsheet JSONs → emits `betting_odds_polymarket.json` + copies the matched JSONs to `data/polymarket_test/`. Filters to senior men's T20 with volume ≥ $1 K. |

### 2.8 Calibration & analysis

| File | Role |
|---|---|
| `scripts/calibration.py` | Platt + isotonic + ball-level calibration utilities. **Off by default** — enabling it improved log loss but regressed flat-betting ROI in the March 2026 A/B (see `IMPROVEMENTS.md` §"Calibration System"). |
| `scripts/analyze_features.py` | XGBoost gain/weight/cover importances, Spearman correlations, per-group aggregation, redundancy detection. |
| `scripts/profile_eval.py` | cProfile harness for the eval hot-path. |

---

## 3. Data Flows

### 3.1 Training (full pipeline)

```
data/t20s_json/*.json (~11K files)
        │
        ▼  build_stats_cache.py
        │
        │  sort by date
        │  for each match (chronological):
        │    if first match on this date: deep_copy_stats() snapshot pre-match
        │    parse_match_data_v2 → ball-by-ball update of trackers + venue
        │    end_match: push to recent_batting / recent_bowling deques
        │    write match_log row for (player, match)
        │
        │  one-pass at end:
        │    π = global outcome distribution over all 2.19 M training balls
        │    write π to _meta.prior_p{0,1,2,4,6,w}
        │
        │  integrity gates:
        │    _verify_log_denormalized_consistency
        │    _verify_outcome_count_conservation
        ▼
models/player_stats_cache_v3.sqlite (~57 MB, schema v4)
        │
        ▼  materialize_features.py
        │
        │  load π from _meta once at startup
        │  group matches by date
        │  for each date:
        │    rehydrate temp_stats / temp_elo / temp_venue from SQLite
        │      (union of all same-day players + venues)
        │    for each same-day match (in monolith order):
        │      parse_match_data_v2(..., prior=π)
        │      → 105 columns/ball, including 42 outcome-dist features
        │      advance temp_venue post-match (matches monolith drift)
        │
        │  classify each match by date → train/validation/test/golden_test
        ▼
data/xgb_data_v3/{train,validation,test,golden_test}.parquet
data/xgb_data_v3/.feature_hash    # {hash, version, n_features, splits, gender_filter}
        │
        ▼  xgboost_v2.py (or lstm_v1 / mlp_v1 / transformer_v1)
        │
        │  load all 4 splits
        │  fit LabelEncoders (batter, bowler [, venue, matchup])
        │  apply 6-class remapping
        │  (Optuna: TPE sampler, 50 trials, optimize val log loss)
        │  train final model
        │  evaluate on test split
        ▼
models/xgb_v3/
  ├─ xgboost_model_v3.pkl
  ├─ batter_encoder_v3.pkl
  ├─ bowler_encoder_v3.pkl
  ├─ feature_columns_v3.txt
  └─ optuna_study_v3.pkl   (only if --tune)
```

### 3.2 Simulation (one match)

```
TestMatchLoader.load_matches('data/polymarket_test/')
        │
        ▼
List[(match_id, MatchState)]
        │
        ▼  SimulationEngine.simulate_multiple(state, config)
        │
        │  for each of n_simulations:
        │    state' = state.copy(); seed RNG
        │    while not match_over:
        │      features = model.extract_features(state')
        │      probs    = model.predict_next_ball(features)
        │      outcome  = sample(probs)
        │      if not T20Rules.is_legal_outcome: outcome = DOT
        │      runs     = T20Rules.process_ball(state', outcome)
        │      state'.update(outcome, runs)
        │      if balls % 6 == 0: state'.bowler_idx = T20Rules.select_next_bowler
        │      if state'.is_innings_over(): state'.start_new_innings()
        │
        ▼
List[MatchResult]
        │
        ▼  ResultAggregator.aggregate
        │
        ▼
{ win_probability, score_stats, wicket_stats }
```

`StatsProviderCache` (auto-applied by every model wrapper's `__init__`) memoizes
the 5 team-strength / venue-profile lookups per match — they're constant across
all balls of a given match, so the per-match work is amortized.

### 3.3 Evaluation

```
MatchLevelEvaluator.evaluate_all(matches, odds_lookup)
        │
        │  for each match:
        │    aggregate = simulate_multiple(state, config)
        │    market_probs = BettingOddsLoader.implied_probs(odds, normalize=True)
        │    log_loss = -log P(actual_winner)
        │    brier    = (P(team1) - 1{team1_won})²
        │    edge     = sim_prob - market_prob   (per team)
        │    flat / full Kelly / fractional Kelly P&L
        │    bucket into calibration bin
        ▼
OverallEvaluationResults
  ├─ avg_log_loss, avg_brier_score
  ├─ avg_edge, avg_signed_edge, profitable_bets
  ├─ flat_pnl, flat_roi, flat_win_rate
  ├─ full_kelly_pnl/roi/win_rate, frac_kelly_pnl/roi/win_rate
  ├─ calibration_bins[]
  └─ match_results[]
        │
        ▼
match_evaluation_results_<model>_<timestamp>.json
```

---

## 4. Key Classes & Interfaces

### 4.1 `PredictionModel` — model plugin interface

```python
class PredictionModel(ABC):
    @abstractmethod
    def extract_features(self, state: MatchState) -> Any: ...
    @abstractmethod
    def predict_next_ball(self, features) -> Dict[str, float]: ...
```

Returned dict must be a probability distribution over the **6 model classes**:
`{'dot', 'one', 'two', 'four', 'six', 'wicket'}`. Wrappers add small fixed
probabilities for `'wide'` and `'no_ball'` *after* the model output and
re-normalize — this happens in `predict_next_ball` itself, not the model.

Concrete implementations in `scripts/sim_v1_2.py`:
- `XGBoostModelV2` — production (active **v7**). 114 features (V6_GROUPS feature
  list; v7 keeps the same column names and uses hierarchical shrinkage on the
  vs-type / vs-hand cells — see § 6.7). Reads `outcome_dist_config_v3.json`
  sidecar at init for `k_player` / `k_venue` overrides.
- `LSTMModelV1` — sliding 10-ball window, embeddings.
- `MLPModelV1` / `MLPModelV2` — baseline + tuned MLP.
- `TransformerModelV1` — full 120-ball context, PyTorch + MLX.
- `LLMModelV1` — experimental.
- `DummyModel` — fixed probabilities for testing.

All wrappers receive a `stats_provider`; their `__init__` calls
`wrap_with_cache(provider)` so per-match memo is automatic.

### 4.2 Core data classes

```python
@dataclass
class Player:
    player_id: str        # must match training-data ID
    name: str
    team: str
    role: str = "allrounder"

@dataclass
class TeamLineup:
    team_name: str
    players: List[Player]   # 11 or 12 (Impact Player) in batting order

class MatchState:                 # mutable; sim copies before each iteration
    # immutable setup
    team1_lineup: TeamLineup
    team2_lineup: TeamLineup
    batting_first: str
    venue: str
    match_date: datetime
    # dynamic
    innings: int                  # 1 or 2
    balls: int                    # 0..119 (T20)
    runs: np.ndarray              # [t1, t2]
    wickets: np.ndarray           # [t1, t2]
    striker_idx, non_striker_idx, bowler_idx, last_bowler_idx: int
    batsmen_out: Dict[int, List[int]]
    history: np.ndarray           # ball-by-ball, dynamically grown
    bowler_balls: Dict            # (team, player) -> balls bowled (cap 24)
    batsman_stats: Dict           # in-match (runs, balls)
    # properties: current_team_idx, batting_team, overs_completed,
    # target, required_run_rate, ...
```

### 4.3 Rules engine

```python
class T20Rules:
    def select_next_bowler(self, state) -> int      # via BowlerSelector
    def is_legal_outcome(self, state, outcome) -> bool
    def process_ball(self, state, outcome) -> int   # returns runs

class BowlerSelector(ABC):
    @abstractmethod
    def select_bowler(self, state, available: List[int]) -> int

class RandomBowlerSelector(BowlerSelector): ...     # current default
```

State-transition invariants:
- Strike rotates on **odd runs only** and at end-of-over.
- On `WICKET`: striker_idx ← `get_next_batsman_idx()`; **no rotation**.
- Bowler caps at 24 balls (4 overs); `last_bowler_idx` blocks consecutive
  overs; selector picks from `available = T20Rules.get_available_bowlers(state)`
  (12-player squads supported via `range(len(lineup.players))`).
- Innings ends at 10 wickets OR 120 legal balls; innings 2 also ends when the
  target is achieved.

### 4.4 Simulation orchestrator

```python
class SimulationEngine:
    def __init__(self, model: PredictionModel, rules: T20Rules)
    def simulate_match(self, state, match_id=None) -> MatchResult
    def simulate_multiple(self, state, config: SimulationConfig) -> List[MatchResult]

@dataclass
class SimulationConfig:
    n_simulations: int = 1000
    parallel: bool = False        # see warning in §6
    n_workers: int = 4
    random_seed: Optional[int] = None
    verbose: bool = False
```

### 4.5 Stats access

```python
class StatsProvider:                  # facade over _SQLiteBackend
    def __init__(self, cache_dir='models', version='v3')
    # All getters delegate via __getattr__:
    def get_batting_stats(pid, as_of_date) -> {avg, sr}
    def get_bowling_stats(pid, as_of_date) -> {avg, econ}
    def get_h2h_stats(bid, bowid, as_of_date) -> {avg, sr}
    def get_batting_recent(pid, as_of_date) -> {avg, sr}
    def get_bowling_recent(pid, as_of_date) -> {avg, econ}
    def get_batting_vs_type_stats(pid, as_of_date) -> {avg_vs_pace, sr_vs_pace, avg_vs_spin, sr_vs_spin}
    def get_bowling_vs_hand_stats(pid, as_of_date) -> {...}
    def get_venue_profile(venue, as_of_date) -> {...}
    def get_team_batting_elo(player_ids, as_of_date) -> float
    def get_team_bowling_elo(player_ids, as_of_date) -> float
    def get_team_batting_strength(player_ids, as_of_date) -> {avg, sr}
    def get_team_bowling_strength(player_ids, as_of_date) -> {avg, econ}
    # Schema-v4 outcome-distribution getters (return shrunk P(0,1,2,4,6,w)):
    def get_batter_outcome_dist(pid, as_of_date) -> dict
    def get_bowler_outcome_dist(pid, as_of_date) -> dict
    def get_batter_vs_type_outcome_dist(pid, as_of_date, bowl_type) -> dict
    def get_bowler_vs_hand_outcome_dist(pid, as_of_date, bat_hand) -> dict
    def get_venue_outcome_dist(venue, as_of_date) -> dict
```

`StatsProviderCache(provider)` wraps the provider and memoizes the 5
team-strength / venue-profile methods per `(lineup_ids|venue, date)` key. All
model wrappers call `wrap_with_cache(provider)` in their `__init__`, so callers
just pass a plain `StatsProvider` and forget about it. The wrapper is
pickle-safe for `multiprocessing.Pool.starmap`.

### 4.6 Evaluation result types

```python
@dataclass
class MatchEvaluationResult:
    match_id, team1, team2: str
    simulated_win_prob: Dict[str, float]
    simulated_scores:   Dict[str, Dict[str, float]]   # mean/std/percentiles
    market_win_prob:    Dict[str, float]
    market_odds:        Dict[str, float]
    actual_winner:      Optional[str]
    log_loss, brier_score: float
    edge:               Dict[str, float]
    realized_pnl:       Optional[float]               # flat
    n_simulations:      int
    simulation_time:    float

@dataclass
class OverallEvaluationResults:
    n_matches: int
    avg_log_loss, avg_brier_score, avg_edge, avg_signed_edge: float
    profitable_bets: int
    # three betting strategies, all computed:
    flat_pnl, flat_roi, flat_win_rate, ...
    full_kelly_*  ... ; frac_kelly_*  ...
    calibration_bins: List[Tuple[float, float, int]]
    match_results:    List[MatchEvaluationResult]
    total_simulation_time: float
```

---

## 5. Data Formats

### 5.1 Cricsheet match JSON (raw input)

`data/t20s_json/*.json`. Structure (abridged):

```json
{
  "info": {
    "teams": ["India", "Australia"],
    "dates": ["2024-06-15"],
    "venue": "Melbourne Cricket Ground",
    "city": "Melbourne",
    "match_type": "T20",
    "gender": "male",
    "team_type": "international",
    "event": {"name": "ICC Men's T20 World Cup"},
    "toss": {"winner": "India", "decision": "bat"},
    "outcome": {"winner": "India", "by": {"runs": 23}},
    "players": {"India": ["Rohit Sharma", ...], "Australia": [...]},
    "registry": {"people": {"Rohit Sharma": "253802", ...}}
  },
  "innings": [
    {"team": "India", "overs": [{"over": 0, "deliveries": [
      {"batter": "Rohit Sharma", "bowler": "Mitchell Starc",
       "non_striker": "Shubman Gill",
       "runs": {"batter": 4, "extras": 0, "total": 4}},
      ...
    ]}]},
    {"team": "Australia", "overs": [...]}
  ]
}
```

Notes on quirks the parser handles:
- `info.players[team]` is the **authoritative roster** (12 entries when an
  Impact Player is named). The earlier `_extract_team_players` bug used the
  union of batters/bowlers in `innings`, which missed unused 11th men.
- `registry.people` maps display names → cricsheet IDs; we always work with
  IDs internally to survive name spellings.
- `info.event.name` + `info.team_type` + `info.teams` feed
  `classify_match_k_factor` — 4.0 (premium), 2.0 (standard), 1.0 (associate).

### 5.2 Parquet feature files

`data/xgb_data_v3/{train,validation,test,golden_test}.parquet`. 105 columns
under schema v4 (= 63 v3 columns + 42 outcome-distribution columns).

**Identifier columns** (not features, used for grouping):
```
match_id, innings_id, batter_id, non_striker_id, bowler_id,
match_date, venue, ball_outcome
```

**63 v3 feature columns** — see [feature_roadmap.md](feature_roadmap.md) for the
full catalog. Groups: `basic` (16), `player_stats` (14), `h2h` (2),
`momentum` (6), `pressure` (3), `chase` (3), `medium` (2),
`player_metadata` (6), `matchup` (3), `type_based` (8), `team_strength` (9).

(7 extra venue/match-context columns exist in the parquet but are excluded
from `V3_GROUPS` and `V6_GROUPS` — see `IMPROVEMENTS.md` §"Venue Profile +
Match Context Features"; they hurt every metric in the March 2026 A/B.)

**42 schema-v4 outcome-distribution columns** — empirical-Bayes-shrunk:

```
batter_p{0,1,2,4,6,w}                       # 6
bowler_p{0,1,2,4,6,w}                       # 6
batter_p{0,1,2,4,6,w}_vs_pace               # 6
batter_p{0,1,2,4,6,w}_vs_spin               # 6
bowler_p{0,1,2,4,6,w}_vs_lhb                # 6
bowler_p{0,1,2,4,6,w}_vs_rhb                # 6
venue_p{0,1,2,4,6,w}                        # 6
```

**Target column**: `ball_outcome ∈ {0, 1, 2, 4, 6, -1}` (raw data values).
Trainers remap to `{0,1,2,3,4,5}` via `class_mapping = {0:0, 1:1, 2:2, 4:3, 6:4, -1:5}`.

**Approximate row counts** (post-Phase-B corpus, ~11.3 K matches):
- train ≤ 2024-12-31: ~1.7 M balls
- validation 2025-01-01 → 2025-06-30: ~250 K
- test 2025-07-01 → 2026-04-16: ~250 K
- golden_test ≥ 2026-04-17: ~50 K (and growing)

**`.feature_hash` marker** (schema):
```json
{
  "hash": "<sha256[:12]>",
  "version": "v3",
  "n_features": 114,
  "splits": {"train_end":..., "val_end":..., "test_end":..., "golden_start":...},
  "gender_filter": "male"
}
```

`run_experiment.py:_check_parquet_cache` matches all four fields against the
current YAML to decide cache hit / miss.

### 5.3 SQLite stats cache (schema v4)

`models/player_stats_cache_v3.sqlite`. ~57 MB, mmap-read, fork-safe.

**Identity tables** (id assigned in stable order at build time):
```sql
players (id PK, player_id TEXT UNIQUE)
dates   (id PK, date TEXT UNIQUE)         -- id assigned in sorted order
venues  (id PK, venue TEXT UNIQUE)
```

**Snapshot tables** — one row per `(entity, date)` where the entity's
*career-cumulative* stats changed; reads use
`WHERE key=? AND date_id<=? ORDER BY date_id DESC LIMIT 1` to find the most
recent snapshot ≤ target date. All `WITHOUT ROWID` on composite PK except
`h2h` (rowid + UNIQUE INDEX — too big for `WITHOUT ROWID` page bloat).

```sql
batting (player_id, date_id, runs, balls, dismissals,
         recent_runs, recent_balls, recent_dismissals,    -- last-5-match sums
         c0, c1, c2, c4, c6, cw)                          -- schema-v4 outcome counts
bowling (player_id, date_id, runs_given, balls_bowled, wickets,
         recent_runs_given, recent_balls_bowled, recent_wickets,
         c0, c1, c2, c4, c6, cw)
h2h (batter_id, bowler_id, date_id, runs, balls, dismissals)
batting_vs_type (player_id, date_id, bowl_type, runs, balls, dismissals,
                 c0, c1, c2, c4, c6, cw)
bowling_vs_hand (player_id, date_id, bat_hand, runs_given, balls_bowled, wickets,
                 c0, c1, c2, c4, c6, cw)
venue (venue_id, date_id, total_runs, total_balls, total_wickets,
       boundary_runs, dot_balls, powerplay_runs, ...,
       c0, c1, c2, c4, c6, cw)
batting_elo (player_id, date_id, elo_rating)
bowling_elo (player_id, date_id, elo_rating)
```

**Match-log tables** (Phase B / schema v3) — one row per (player, match) for
recent-form deque reconstruction:
```sql
batting_match_log (player_id, date_id, intra_date_idx,
                   runs, balls, dismissals)        -- WITHOUT ROWID
bowling_match_log (player_id, date_id, intra_date_idx,
                   runs_given, balls_bowled, wickets)
```
`intra_date_idx` distinguishes same-day matches in monolith order; the recent
deque is reconstructed via
`ORDER BY date_id DESC, intra_date_idx DESC LIMIT N` with strict `date_id < ?`
at date boundaries.

**`_meta` table** (key/value text):
```
schema_version             = 4
build_timestamp            = 2026-04-23T18:25:00
source_json_mtime_max      = <max(mtime) of data/t20s_json/*.json at build>
prior_p0..prior_pw         = global empirical outcome distribution π
```

`_meta.schema_version == SCHEMA_VERSION` and `source_json_mtime_max ≥ live`
mtime are the gates for `StatsProvider` to accept the cache; `run_experiment`
uses them to decide whether to re-run `build_stats_cache.py`.

### 5.4 Betting odds JSON

Two formats, same shape:

**`betting_odds_v3.json`** (legacy 44-match WC 2024 set, bookmaker odds with
margin):
```json
{
  "matches": [
    {
      "match_id": "2024-06-15_India_Australia_MCG",
      "date": "2024-06-15",
      "team1": "India", "team2": "Australia", "venue": "MCG",
      "odds": {"winner": {"India": 2.10, "Australia": 1.75,
                          "timestamp": "2024-06-14T10:00:00Z"}},
      "actual_winner": "India",
      "actual_margin": {"runs": 23}
    }
  ]
}
```

**`betting_odds_polymarket.json`** (261-match Polymarket set, **margin-free**
prediction-market odds — `BettingOddsLoader` skips the normalize step for this
file). Adds `polymarket_event_slug`, `polymarket_volume_usd`, `tournament`
metadata.

`BettingOddsLoader.load_odds` returns `Dict[match_id → odds_data]` either way.

### 5.5 Evaluation results JSON

`match_evaluation_results_<model>_<timestamp>.json` written by
`scripts/sim_eval/run_sim_eval.py`. Top-level `summary` block + per-match
`matches[]` array; see `OverallEvaluationResults` and `MatchEvaluationResult`
dataclasses in §4.6 for field definitions.

### 5.6 Experiment artifacts

`experiments/results/<name>_<timestamp>_<git_hash>/`:
```
config.yaml              # exact YAML used
metadata.json            # git hash, branch, dirty flag, platform, durations
metrics.json             # extracted from eval stdout
training_metrics.json    # train/val accuracy/loss
console_output.log       # captured stdout/stderr from each step
```

---

## 6. Design Decisions

### 6.1 Temporal integrity (CRITICAL)

**Problem**: stats used during simulation must reflect only data available at
match time. Otherwise the model "sees the future" and evaluation is invalid.

**Mechanism**:
- Training: `build_stats_cache.py` walks matches chronologically and takes a
  `deep_copy_stats` snapshot **before** processing each new date — first-write-
  wins. The match itself is then parsed with the previously-snapshotted
  trackers. `parse_match_data_v2` extracts each ball's features from
  pre-ball state, then updates the tracker.
- Simulation: `_SQLiteBackend` uses `WHERE date_id <= ? ORDER BY date_id DESC
  LIMIT 1` for every getter — the most recent snapshot ≤ match date.

**Validation**: `scripts/tests/test_phase_a_parity.py` asserts the
materialized parquet is bit-exact (`check_exact=True`, `check_dtype=True`)
against the reference monolith on **all 9,519 matches × 63 columns** plus the
42 schema-v4 outcome-distribution columns. Same-day-secondary matches —
historically the leakage hotspot — are covered by the `batting_match_log` /
`bowling_match_log` rehydration via `tracker_rehydration.py`.

### 6.2 Feature ordering: read before write

Features describe state *as observed at delivery time*. Stats update *after*
the ball, so the tracker reflects the post-ball state for the next ball.
Reverse the order and you leak the outcome into its own features.

### 6.3 SQLite over chunked pickle

Earlier iteration: 69 pickle chunks (~110 MB each, ~7.6 GB on disk for v2;
11 GB for v3) with LRU caching, ~550 MB resident.

**Why we migrated** (April 2026):
- Parallel eval OOM'd the 16 GB box: each worker had its own LRU cache.
- Pickle deserialize cost: ~2 s startup × N workers.
- Chunk-boundary same-day-snapshot bug: `stats_snapshots` reset on chunk save
  could double-snapshot a date across the boundary, with last-write-wins
  serving the wrong snapshot at inference.

**Migration result**:
| | chunks (v3) | SQLite v3 → v4 |
|---|---:|---:|
| disk | 11 GB | 46.5 → 56.8 MB |
| build | ~15 min | 6:36 → 7:20 |
| query p50 | <0.01 ms (after chunk load) | ~3 µs |
| parallel-eval RSS (2 workers) | OOM | 1.7 GB combined |

Single mmap'd file means N readers share OS page cache, fork-safe via
PID-aware `_ensure_conn`. Schema-v4 added 6 outcome-count columns (`c0..cw`)
to 5 tables for the empirical-Bayes feature pass; +10 MB for +42 features.

### 6.4 Two-stage parsing pipeline (Phase B)

Before April 22, 2026 a single `parsing_v2.py:process_folder_v2_with_splits`
did three things in one ~600 s pass: chronological tracker walk, stats
snapshot emission, and per-ball feature materialization. We split it into
`build_stats_cache.py` (cross-date stateful) and `materialize_features.py`
(within-date stateful, cross-date stateless given the SQLite cache).

**Payoffs**:
- Re-materializing parquet after a feature change is **3.5× faster** (170 s
  vs 600 s) — the tracker walk is skipped.
- SQLite is now the single source of truth for stats; `models/cache_chunks_v3/`
  reclaimed (12 GB).
- `run_experiment.py:check_smart_cache` now returns
  `(sqlite_valid, parquet_valid)` independently — a JSON refresh re-runs only
  the cache, a feature-list change re-runs only the materializer.

**Insight**: same-day matches are the only intra-batch state dependency.
`materialize_features.py` rehydrates `temp_*` trackers once per date from
SQLite, then replays same-day matches in monolith order so within-date drift
matches. This is what makes the parity harness pass bit-exactly on all
secondaries.

### 6.5 Ball-level modelling + Monte Carlo

```
Direct match prediction: ~11 K matches.
Ball-level + simulation: ~2.2 M balls (200×).
```

We get richer features (momentum, phase, matchup, recent form), natural
uncertainty quantification from simulation variance, and the same engine works
for any ball-by-ball format. Cost: 100–1000 forward passes per match instead
of one. Acceptable given XGBoost inference is ~1 µs after Fix A+B.

### 6.6 6-class outcome remapping

Raw ball outcomes from Cricsheet have a long tail (3 runs, 5 runs, 7+ runs,
all-run boundaries, byes). We collapse to 6 semantic classes:

```
0 → 0  (dot)
1 → 1
2, 3 → 2  (running between wickets; 3 is rare overthrow/misfield)
4, 5 → 3  (boundary; 5 is rare overthrow boundary)
6, 7+ → 4  (max; 7+ are extreme overthrows + no-balls)
wicket → 5
```

3-/5-/7+-run balls together are <1% of the corpus. Combining preserves
semantic meaning and reduces the rare-class problem dramatically.

**Sim wrappers add `wide` and `no_ball` post-hoc** at fixed 1% each, then
re-normalize. Extras have high variance and small signal; we don't try to
predict them.

### 6.7 Empirical-Bayes outcome distributions (v6 → v7, schema v4)

XGBoost can't learn outcome distributions from label-encoded player IDs —
trees split on "id > N", which groups arbitrary players. Career averages
compress a player into 2 numbers (`avg`, `sr`), destroying distribution shape.

**Fix**: emit `P(0,1,2,4,6,W)` directly per (batter | bowler | batter-vs-pace
| batter-vs-spin | bowler-vs-LHB | bowler-vs-RHB | venue), shrunk toward a
global corpus prior π via Dirichlet-posterior-mean shrinkage:

```
p̂_c = (n_c + k · π_c) / (N + k)     where N = Σ n_c
```

- N → 0 ⇒ p̂ → π (full fallback to global prior)
- N → ∞ ⇒ p̂ → MLE
- N = k ⇒ half-and-half

`k = 30` for player cells, `k = 200` for venue cells (more data per venue).
π is computed during `build_stats_cache.py`'s walk on 2.19 M training balls
and stored in `_meta.prior_p{0,1,2,4,6,w}`. Final π =
(0.304, 0.411, 0.076, 0.108, 0.047, 0.054).

**v6 result** (261-match Polymarket × 100 sims): log loss 0.7518 → 0.7122
(−5.3%); Brier 0.2728 → 0.2562 (−6.1%); flat ROI +6.5% → −7.1% (calibration-vs-
ROI tension — sharper probabilities compress betting edges). The Phase 1
sliced eval (2026-04-24) showed the LL win **grows with liquidity**
(Δ LL −0.040 / −0.047 / −0.074 across all / ≥$50K / ≥$100K).

**v7 — hierarchical shrinkage (Phase 5, 2026-04-25)**: the four narrow cells
(`batter_p*_vs_pace`, `batter_p*_vs_spin`, `bowler_p*_vs_lhb`,
`bowler_p*_vs_rhb`) now shrink toward the player's *overall* distribution
rather than toward the global prior π directly. Closed-form, two-stage:

```
p̂_overall  = (n_overall + k · π) / (N_overall + k)         # stage 1
p̂_vs_pace  = (n_vs_pace + k · p̂_overall) / (N_vs_pace + k) # stage 2
```

This is mechanically more accurate than flat shrinkage when the player has
substantial overall data but limited cell data — e.g. a fringe spinner with
500 deliveries overall but only 80 vs LHB shrinks toward the 500-ball signal,
not the global average. Phase 6 swept `k_player ∈ {10, 30, 100, 300}`; **k=30
won on both LL and flat ROI**. Active config: `k_player=30`, `k_venue=200`,
written to `models/xgb_v3/outcome_dist_config_v3.json` and re-read by
`XGBoostModelV2.__init__` at sim time so training and inference always agree.

The hierarchical/flat switch is the `hierarchical=True` default kwarg on
`_SQLiteBackend.get_{batter_vs_type,bowler_vs_hand}_outcome_dist` and the
equivalent tracker getters; pass `hierarchical=False` to recover v6 flat-
shrunk values for ablations.

**Phase 3 negative result (2026-04-24)**: phase prior
P(outcome | PP / mid / death) was implemented end-to-end —
`prior_{pp,mid,death}_p*` are written to SQLite `_meta` and the per-phase
distribution is reachable via `_SQLiteBackend.get_phase_outcome_dist` — but
including the resulting `phase_outcome_dist` feature group **regresses** LL
(Δ +0.022 / +0.023 / +0.036 across all / ≥$50K / ≥$100K slices), almost
certainly because the phase signal is collinear with the existing
`is_powerplay` / `is_middle_overs` / `is_death_overs` indicators. The code
stays inert; resurrect via `experiments/configs/xgb_v7_phase_prior.yaml` if
you want to re-test.

**Phase 2 negative result (2026-04-24)**: dropping `batter_encoded` /
`bowler_encoded` (on the assumption XGBoost can't learn from label-encoded
IDs anyway, given outcome distributions are now first-class features)
*degrades* LL by +0.069 on the ≥$50K slice — 14× the threshold. Keep them.

See `IMPROVEMENTS.md` § "Empirical Outcome Distributions" and § "Phase 5 /
Phase 6" for the full per-experiment breakdowns.

### 6.8 Per-match memoization (`StatsProviderCache`)

Five getters are constant across every ball of a match: team batting/bowling
ELO, team batting/bowling strength, venue profile. Without memoization, each
ball re-runs an 11-player loop inside the provider. With memoization, each
match computes them once and the remaining ~24,000 calls (240 balls × 100
sims) are dict hits.

**Result**: 1.20× sim speedup (XGBoost v3, warm-chunk bench, 100 sims).
Bit-identical sim outputs. Wrapper is pickle-safe for `multiprocessing.Pool`.
All model classes apply it automatically in `__init__`.

### 6.9 Margin-free betting edge

Bookmaker odds carry 5–10% margin (overround). To compare model probabilities
to a "fair" market, normalize implied probabilities to sum to 1.0:

```
p_implied[t] = 1 / odds[t]
total = Σ p_implied[t]                # ~1.05 typical
p_fair[t] = p_implied[t] / total
edge[t]   = p_model[t] - p_fair[t]
```

Polymarket prices are **already margin-free** (prediction market, no vig), so
`BettingOddsLoader` skips the normalize step when `source == "polymarket"` —
otherwise we'd be over-correcting.

### 6.10 Signed edge

Absolute edge measures disagreement; signed edge measures *quality* of
disagreement:

```
if predicted_winner == actual_winner: signed_edge = +|edge|
else:                                  signed_edge = -|edge|
```

A high positive signed edge means "we confidently disagreed with the market
and were right." High negative means "we confidently disagreed and were
wrong." Useful as a leading indicator before P&L stabilizes.

### 6.11 Calibration disabled by default

We shipped Platt + isotonic + ball-level calibration in March 2026 and ran a
full A/B (4 experiments × 44 matches × 100 sims). Result:
- Match-level Platt: log loss 0.710 → 0.667 (−6%) but flat ROI −44.8% → −64.9%.
- Ball-level isotonic: hurt every metric.
- Both: in between.

The trade is mechanical — Platt compresses predictions toward 50%, which
reduces edge on every bet. Calibration is kept in `scripts/calibration.py`
behind `--calibrate` / `--ball-calibrate` flags; enable when the model has
better resolution. See `IMPROVEMENTS.md` § "Calibration System" for the full
A/B and root-cause discussion.

### 6.12 No `--parallel` on `run_sim_eval.py`

`SimulationConfig.parallel=True` works for in-process simulation but the
parallel eval path has crashed the 16 GB development box (RSS blowup;
multiple model + cache copies per worker). The migration to SQLite fixed
the underlying memory issue (1.7 GB combined for 2 workers), but the eval
parallelism path hasn't been re-validated end-to-end. **Default and
recommended: serial**. For long runs, schedule as a background task.

### 6.13 Strict same-day-stateful contract

Same-day matches in `materialize_features.py` are processed in monolith
order, with `temp_venue` updated post-match between siblings. Swapping M1
↔ M2 inside a same-day batch produces different features for both — Phase A
proved this on 5,855 same-day-secondary matches.

This has one architectural consequence: per-match parquet caching (Phase C,
not yet shipped) must key on `(match_id, position_in_same_day_batch,
feature_hash)`, not just `(match_id, feature_hash)`. Solo-date matches still
get clean cache hits; busy ICC days will partially invalidate.

### 6.14 What's deliberately NOT in the system

- **The Hundred** (`hnd_json.zip`): excluded — incompatible with 120-ball
  hardcodes in `parsing_v2.py`, `T20Rules`, and `transformer_v1.py`'s
  `max_seq_len`. Tracked in TODO.
- **Women's cricket**: `gender_filter=male` is the default in
  `iter_matches_chronological`. ELO and aggregate stats would otherwise
  cross-contaminate (no bridge players).
- **Associate-nation T20Is**: included; weighted via context-aware K-factor
  (K=1.0 vs K=2.0 vs K=4.0) so they don't dominate the ELO pool.
- **Direct match-winner training**: not implemented; `IMPROVEMENTS.md`
  considers it.
- **Real-time streaming**: out of scope. Eval on captured pre-match odds.

---

**For commands, see [OPERATIONS.md](OPERATIONS.md).**
**For adding a new model type, see [ADDING_NEW_MODELS.md](ADDING_NEW_MODELS.md).**
**For the feature catalog, see [feature_roadmap.md](feature_roadmap.md).**
**For research log + per-experiment results, see [../IMPROVEMENTS.md](../IMPROVEMENTS.md).**
**For active workstreams, see [../TODO.md](../TODO.md).**
