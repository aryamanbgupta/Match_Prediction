# CricML — Agent Guide

T20 cricket match prediction. Predict ball outcomes
(`{0:dot, 1:one, 2:two, 3:four, 4:six, 5:wicket}`), simulate matches via
Monte Carlo, evaluate vs market odds (Polymarket primary, bookmaker legacy).

**Active models** (two production models with complementary roles):

1. **Match-level direct (winner-market predictor)** — XGBoost binary classifier
   on ~47 match-level features (team strength, position-split ELOs, recent
   form, H2H, home/away, lineup mix). Trained on `team1_wins`. Variant of
   record: `models/xgb_match_v2_frozen/` (no-leakage diagnostic; see
   `reports/no_leakage_diagnostic.md`). Config:
   `experiments/configs/xgb_match_v1_baseline.yaml`. **Headline 2026-05-09**:
   ≥$50k slice LL **0.5004** (CI [0.45, 0.56]) vs market 0.6267, flat ROI
   **+53.67%** (CI [+36%, +74%]) — both go/no-go conditions cleared by wide
   margins; conservative early-test floor +33.5% on 124 bets. Use this for
   match-winner predictions.

2. **Ball-level sim** — XGBoost v7, 114 features (V3 + 42 outcome-dist),
   hierarchical shrinkage on the 4 vs-type/vs-hand cells (Phase 5
   2026-04-25); k_player=30, k_venue=200 (Phase 6 sweep). Config:
   `experiments/configs/xgb_v6_hierarchical_shrink.yaml`. v7 lost the
   winner-market race to the direct model by ~0.24 LL. **Use this for prop
   bets, score distributions, in-play scenarios** — anywhere ball-level
   resolution matters and match-level supervision can't help.

Schema-v4 SQLite stats cache at `models/player_stats_cache_v3.sqlite` —
schema unchanged from v6; v7 differs only in the shrinkage *composition*
(narrow cells shrink toward player overall, not π directly). Per-phase
priors (`prior_{pp,mid,death}_p*`) are written to `_meta` but the
`phase_outcome_dist` feature group is NOT in v7's feature list — Phase 3
ablation showed it regresses LL (collinear with is_powerplay/middle/death).

Eval set: `data/polymarket_test/` + `betting_odds_polymarket.json`
(261 matches; 255 join to the direct model after team-name aliasing).
Use `--min-volume {50000,100000}` to slice for sharp markets.

---

## Tooling rule

Always run Python via `uv run` — e.g. `uv run python scripts/run_experiment.py ...`.
The `.venv` shipped with the repo is the source of truth.

---

## Quick start

```bash
# === Match-level direct model (active winner-market predictor) ===
# Materialize one-row-per-match parquet, train, predict on test:
uv run python scripts/materialize_match_features.py \
    --out-dir data/xgb_match_data_v2_frozen \
    --freeze-trackers-after 2025-06-30
uv run python scripts/xgboost_match_v1.py \
    --cmd both \
    --data-dir data/xgb_match_data_v2_frozen \
    --model-dir models/xgb_match_v2_frozen
# Blend with v7 sim eval JSON + reslice + report:
uv run python scripts/sim_eval/blend_eval_json.py \
    --sim-json eval_out_phase5_hier/hier_all_20260425_165622.json \
    --direct-json models/xgb_match_v2_frozen/test_predictions.json \
    --w 0.0 0.2 0.5 0.8 1.0 \
    --out-dir eval_out_blend_a2_frozen
for w in w0p00 w0p20 w0p50 w0p80 w1p00; do
  uv run python scripts/sim_eval/reslice_eval_json.py \
    --in eval_out_blend_a2_frozen/hier_all_20260425_165622_${w}.json \
    --odds betting_odds_polymarket.json \
    --out-dir eval_out_blend_a2_frozen/sliced
done
uv run python scripts/sim_eval/blend_report.py \
    --sliced-dir eval_out_blend_a2_frozen/sliced \
    --direct-json models/xgb_match_v2_frozen/test_predictions.json \
    --out reports/blend_a2_frozen_report.md

# === Ball-level sim (v7 — for props / scores / in-play) ===
# One command: cache → parquet → train → eval. Each step skipped if its
# artifact is already current (via _meta.schema_version + .feature_hash).
uv run python scripts/run_experiment.py \
    experiments/configs/xgb_v6_hierarchical_shrink.yaml

# Sliced eval (after eval has produced match_evaluation_results JSON):
bash scripts/run_sliced_eval.sh   # all / >=$50k / >=$100k
```

Step-by-step v7 sim equivalent: `build_stats_cache.py` → `materialize_features.py`
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
4. **`--parallel` on `run_sim_eval.py` is correct but slower than serial**
   (intra-match `multiprocessing.Pool` is dominated by IPC cost). Use it
   only if you really mean it. For real parallelism, run N independent
   evals on disjoint match shards via `perf_runs/run_n_parallel.py`.
   Measured 2026-05-09: 4 procs at OMP=2 took **16.6 min for full 261×100
   eval vs 41.9 min serial = 2.52× speedup**, 1.5 GB combined RSS,
   numerics within MC noise of serial. Cap `OMP_NUM_THREADS` per process
   or BLAS oversubscribes and you serialize. See
   `docs/OPERATIONS.md` § "Multi-process parallel eval".
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
