# CricML — Agent Guide

T20 cricket match prediction. Predict ball outcomes
(`{0:dot, 1:one, 2:two, 3:four, 4:six, 5:wicket}`), simulate matches via
Monte Carlo, evaluate vs market odds (Polymarket primary, bookmaker legacy).

**Active models** (two production models with complementary roles):

1. **Match-level direct (winner-market predictor)** — XGBoost binary classifier
   on ~45 match-level features (team strength, position-split ELOs, recent
   form, H2H, home/away, lineup mix). Trained on `team1_wins`. Variant of
   record: **`models/xgb_match_v2_clean_unfrozen/`** — booster bytes
   bit-identical to the earlier `xgb_match_v2_clean` (train+val parquets
   are identical between frozen and unfrozen builds, so retraining
   produces the same model). Named for inference semantics: per-fixture
   unfrozen rehydration is the production path. The model was retrained
   2026-05-09 after a feature-leakage fix
   (see `reports/leakage_fix_comparison.md`); pre-fix `_build_match_record`
   was reading per-player ELOs *after* `parse_match_data_v2` had updated
   them ball-by-ball with this match's own outcomes. Patched in
   `materialize_match_features.py`. Config:
   `experiments/configs/xgb_match_v1_baseline.yaml`.
   **Honest headline (golden eval, 2026-05-07 cutoff, n=55 polymarket overlap)**:
   ≥$50k slice (n=50) LL **0.6747** (CI [0.64, 0.72]) vs market 0.6267 — does
   *not* beat market on log loss; flat ROI **+32.6%** (CI [-0.2%, +63.6%]) —
   ROI CI just barely includes zero. ≥$100k slice (n=45) LL 0.6698,
   flat ROI **+34.8%** (CI [+3.79%, +65.5%]) — narrowly clears the soft
   ROI-only gate. The earlier "+47-58% ROI" claim was inflated by the leakage
   and is retracted. Use this for match-winner predictions; treat the edge as
   modest, not dominant.

   **Frozen vs unfrozen tracker semantics**: previous diagnostic claimed
   frozen-mode trackers (snapshot at val/test boundary, no within-test
   updates) outperformed unfrozen on every slice. After the leakage fix
   that finding compressed roughly 2× and split: frozen still wins LL on
   the polymarket-overlap subset by ~0.01–0.02; unfrozen wins on the full
   782-match standalone test by 0.016 LL and wins ROI on the ≥$100k
   slice. Conclusion: the gap is small enough that picking either is
   defensible. We default to unfrozen — it matches real deployment
   semantics (each prediction sees state through the fixture date, just
   like a live bookmaker) without leaning on an empirical artifact. See
   `reports/no_leakage_diagnostic_clean.md`.

2. **Ball-level sim** — XGBoost v7, 114 features (V3 + 42 outcome-dist),
   hierarchical shrinkage on the 4 vs-type/vs-hand cells (Phase 5
   2026-04-25); k_player=30, k_venue=200 (Phase 6 sweep). Config:
   `experiments/configs/xgb_v6_hierarchical_shrink.yaml`. v7 lost the
   winner-market race to the clean direct model by ~0.07 LL on ≥$50k
   (v7 0.7402 vs clean direct golden 0.6747). Audited 2026-05-09 for
   the analogous leakage that hit the match-level model — clean
   structurally and empirically (`reports/v7_leakage_audit.md`). **Use
   this for prop bets, score distributions, in-play scenarios** —
   anywhere ball-level resolution matters and match-level supervision
   can't help.

Schema-v4 SQLite stats cache at `models/player_stats_cache_v3.sqlite` —
schema unchanged from v6; v7 differs only in the shrinkage *composition*
(narrow cells shrink toward player overall, not π directly). Per-phase
priors (`prior_{pp,mid,death}_p*`) are written to `_meta` but the
`phase_outcome_dist` feature group is NOT in v7's feature list — Phase 3
ablation showed it regresses LL (collinear with is_powerplay/middle/death).

Eval sets:
- **Iteration set**: `data/polymarket_test/` + `betting_odds_polymarket.json`
  (261 matches, 2025-07-01 → 2026-04-16). Used during model selection; not
  strictly out-of-sample.
- **Golden set** (2026-05-09): `data/golden/polymarket_test/` +
  `data/golden/betting_odds_golden.json` (55 matches, 2026-04-17 → 2026-05-07).
  Truly out-of-sample — never seen by training, validation, or selection.
  Built by `extract_golden_cricsheet.py` (cricsheet from
  `/Users/aryamangupta/Projects/stat-generator/data/cricsheet/`) and
  `build_polymarket_odds_golden.py` (polymarket from
  `/Users/aryamangupta/Projects/polymarket-cricket/data/polymarket_prematch_odds_<date>.json`).
  Per-match dashboard: `reports/ipl_2026_dashboard_clean.html`.

Use `--min-volume {50000,100000}` to slice either set for sharp markets.

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
    --out-dir data/xgb_match_data_v2_clean \
    --freeze-trackers-after 2025-06-30
uv run python scripts/xgboost_match_v1.py \
    --cmd both \
    --data-dir data/xgb_match_data_v2_clean \
    --model-dir models/xgb_match_v2_clean
# Blend with v7 sim eval JSON + reslice + report:
uv run python scripts/sim_eval/blend_eval_json.py \
    --sim-json eval_out_phase5_hier/hier_all_20260425_165622.json \
    --direct-json models/xgb_match_v2_clean/test_predictions.json \
    --w 0.0 0.2 0.5 0.8 1.0 \
    --out-dir eval_out_blend_a2_clean
for w in w0p00 w0p20 w0p50 w0p80 w1p00; do
  uv run python scripts/sim_eval/reslice_eval_json.py \
    --in eval_out_blend_a2_clean/hier_all_20260425_165622_${w}.json \
    --odds betting_odds_polymarket.json \
    --out-dir eval_out_blend_a2_clean/sliced
done
uv run python scripts/sim_eval/blend_report.py \
    --sliced-dir eval_out_blend_a2_clean/sliced \
    --direct-json models/xgb_match_v2_clean/test_predictions.json \
    --out reports/blend_a2_clean_report.md

# === Predict an upcoming fixture ===
# Hand-write fixtures/<match>.json (see fixtures/_template.json), then:
uv run python scripts/predict_fixture.py --fixture fixtures/<match>.json
# First run takes ~7s (builds tracker_snapshot_test_end.pkl from t20s_json
# only — golden data NOT included). Subsequent sub-second. Per-fixture
# unfrozen rehydration: SQLite + tracker queries use as-of fixture_date,
# falling back to test_end (2026-04-16) for fixtures past that. Pass
# --rebuild-snapshot if data/t20s_json grows.

# === Golden eval refresh (after new polymarket capture + cricsheet refresh) ===
# 1. Pull new T20 cricsheet JSONs from stat-generator (date >= 2026-04-17):
uv run python scripts/extract_golden_cricsheet.py
# 2. Build golden polymarket odds (set GOLDEN_POLYMARKET_PATH to latest file):
uv run python scripts/build_polymarket_odds_golden.py
# 3. Re-materialize parquet (clean, no leakage):
uv run python scripts/materialize_match_features.py \
    --source-dir data/t20s_json --extra-source-dir data/golden/t20s_json \
    --out-dir data/xgb_match_data_v2_clean --freeze-trackers-after 2025-06-30
# 4. Predict golden + reslice:
uv run python scripts/predict_golden.py \
    --model-dir models/xgb_match_v2_clean \
    --parquet data/xgb_match_data_v2_clean/golden_test.parquet \
    --out-json models/xgb_match_v2_clean/golden_predictions.json
uv run python scripts/synthesize_golden_envelope.py
uv run python scripts/sim_eval/blend_eval_json.py \
    --sim-json data/golden/golden_sim_envelope.json \
    --direct-json models/xgb_match_v2_clean/golden_predictions.json \
    --w 0.0 --out-dir data/golden/blended_clean_retrained
uv run python scripts/sim_eval/reslice_eval_json.py \
    --in data/golden/blended_clean_retrained/golden_sim_envelope_w0p00.json \
    --odds data/golden/betting_odds_golden.json \
    --out-dir data/golden/sliced_clean_retrained
# 5. Per-match audit:
uv run python scripts/build_ipl_dashboard.py

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
