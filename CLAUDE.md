# CricML — Agent Guide

T20 cricket match prediction. Predict ball outcomes
(`{0:dot, 1:one, 2:two, 3:four, 4:six, 5:wicket}`), simulate matches via
Monte Carlo, evaluate vs market odds (Polymarket primary, bookmaker legacy).

**Active model**: XGBoost v7 — 114 features (V3 + 42 outcome-distribution),
**hierarchical shrinkage** on the 4 vs-type/vs-hand cells (Phase 5,
2026-04-25); k_player=30, k_venue=200 (Phase 6 swept k ∈ {10, 30, 100, 300},
k=30 won on both LL and flat ROI). Config:
`experiments/configs/xgb_v6_hierarchical_shrink.yaml`.

Schema-v4 SQLite stats cache at `models/player_stats_cache_v3.sqlite` —
schema unchanged from v6; v7 differs only in the shrinkage *composition*
(narrow cells shrink toward player overall, not π directly). Per-phase
priors (`prior_{pp,mid,death}_p*`) are written to `_meta` but the
`phase_outcome_dist` feature group is NOT in v7's feature list — Phase 3
ablation showed it regresses LL (collinear with is_powerplay/middle/death).

Eval set: `data/polymarket_test/` + `betting_odds_polymarket.json`
(261 matches). Use `--min-volume {50000,100000}` to slice for sharp markets.

---

## Tooling rule

Always run Python via `uv run` — e.g. `uv run python scripts/run_experiment.py ...`.
The `.venv` shipped with the repo is the source of truth.

---

## Quick start

```bash
# One command: cache → parquet → train → eval. Each step skipped if its
# artifact is already current (via _meta.schema_version + .feature_hash).
# v7 baseline (hierarchical shrinkage):
uv run python scripts/run_experiment.py \
    experiments/configs/xgb_v6_hierarchical_shrink.yaml

# v6 baseline (flat shrinkage, retained for reference):
uv run python scripts/run_experiment.py \
    experiments/configs/xgb_v6_outcome_dist.yaml

# Sliced eval (after eval has produced match_evaluation_results JSON):
bash scripts/run_sliced_eval.sh   # all / >=$50k / >=$100k
```

Step-by-step equivalent: `build_stats_cache.py` → `materialize_features.py`
→ `xgboost_v2.py` → `sim_eval/run_sim_eval.py`. See `docs/OPERATIONS.md`.

---

## Where to look

| Question | Doc |
|---|---|
| How is the system structured? Modules, data flow, classes, formats, design rationale | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| How do I run / retrain / evaluate / debug something? | [docs/OPERATIONS.md](docs/OPERATIONS.md) |
| How do I add a new model type? | [docs/ADDING_NEW_MODELS.md](docs/ADDING_NEW_MODELS.md) |
| What features exist? Implementation status? | [docs/feature_roadmap.md](docs/feature_roadmap.md) |
| What's the current research log / past experiment results? | [IMPROVEMENTS.md](IMPROVEMENTS.md) |
| What's actively being worked on? Eval gates? | [TODO.md](TODO.md) |
| What did the system look like historically? | [docs/archive/](docs/archive/) |

---

## Critical invariants (don't break)

1. **6 model classes only**: `{0:dot, 1:one, 2:two, 3:four, 4:six, 5:wicket}`.
   Every `class_to_outcome` dict in `sim_v1_2.py` must have exactly 6 entries.
   Class 4 = `'six'`, class 5 = `'wicket'` (NOT the reverse).
2. **Temporal integrity**: features reflect state **before** ball; trackers
   update **after**. SQLite snapshot for date `D` reflects only data from
   matches strictly before `D` (first-write-wins).
3. **Schema bumps**: any change to `_SQLiteBackend` table layout requires
   bumping `SCHEMA_VERSION` in `scripts/stats_sqlite_backend.py`. The
   provider refuses to open a file with the wrong version.
4. **No `--parallel`** on `run_sim_eval.py` — it has crashed the 16 GB box.
   Default is serial; budget ~40 min for 261×100 sims, ~5–10 min for 45×1000.
5. **Same-day match order matters** in `materialize_features.py`. Within-date
   trackers carry state across same-day siblings; reordering changes features
   for every match in the batch.

---

## Repo map

```
Match_Prediction/
├── CLAUDE.md, README.md, IMPROVEMENTS.md, TODO.md
├── data/                 # cricsheet JSONs, parquet splits, eval sets, polymarket
├── models/               # SQLite stats cache + per-model artifact dirs
├── scripts/              # parsing, training, simulation, evaluation, tests
│   ├── build_stats_cache.py + materialize_features.py    # parsing pipeline
│   ├── parsing_v2.py + tracker_rehydration.py            # tracker primitives
│   ├── stats_sqlite_backend.py + stats_provider.py       # cache backend
│   ├── feature_registry.py                               # central feature defs
│   ├── xgboost_v2.py / lstm_v1.py / mlp_v1.py / transformer_v1.py
│   ├── sim_v1_2.py                                       # sim engine + wrappers
│   ├── sim_eval/                                         # eval framework
│   ├── run_experiment.py + experiment_tracker.py         # YAML pipeline runner
│   └── tests/                                            # parity harness + benches
├── experiments/configs/  # YAML experiment definitions
├── experiments/results/  # per-run artifacts (auto-generated)
├── docs/                 # ARCHITECTURE / OPERATIONS / ADDING_NEW_MODELS / archive
└── archive/              # local-only; gitignored. Old logs / eval JSONs / superseded scripts
```

`archive/` is gitignored and **not** load-bearing for the current pipeline.
Use it for "I remember we had a script that did X" reference. Documentation
history lives in `docs/archive/` (tracked).

---

## When in doubt

Start with `docs/ARCHITECTURE.md` for the *what* and *why*; `docs/OPERATIONS.md`
for the *how*. Both stay in sync with `main`. The `docs/archive/` folder holds
the older design docs and one-off migration memos — useful for "how did this
get built?" questions, but not load-bearing for current work.

For environment, system requirements, and dependencies, see `pyproject.toml` /
`requirements.txt`. Python 3.11+, 16 GB RAM, 10 GB disk recommended.
