# CricML — Agent Guide

T20 cricket match prediction. Predict ball outcomes
(`{0:dot, 1:one, 2:two, 3:four, 4:six, 5:wicket}`), simulate matches via
Monte Carlo, evaluate vs market odds (Polymarket primary, bookmaker legacy).

**Active models** (two production models with complementary roles):

1. **Match-level direct (winner-market predictor)** — XGBoost binary classifier
   on 49 match-level features (M1 baseline + 3 venue outcome-dist).
   Trained on `team1_wins`. **Production variant of record (post M7,
   2026-05-10): `models/xgb_match_v3_m7_production/`**. Same feature set as
   the prior `xgb_match_v3_m2_venue_only_unfrozen`; the only change at M7
   was the hyperparameter sweep result: lr 0.10 → 0.05, colsample 0.8 →
   0.9 (the prior config was over-aggressive). `predict_fixture.py` uses
   this model directly; raw probabilities (no Platt — Platt over-corrects
   on this config and kills iteration ROI). Per-fixture unfrozen
   rehydration semantics (chronological tracker walk through pre-match
   date). Earlier baselines preserved for reference: `xgb_match_v2_clean`,
   `xgb_match_v2_clean_unfrozen`, `xgb_match_v3_baseline` (M1 monotone),
   `xgb_match_v3_m2_venue_only`, `xgb_match_v3_m2_venue_only_unfrozen`.
   See `reports/m7_architecture_eval.md` for the sweep, M2 → M7
   improvement deltas, and the explanation of why M3–M6 feature work
   dropped.

   **Honest headline (iteration Polymarket eval; I3-revised 2026-07-23)**:
   raw LL **0.6299** vs market 0.6267 (essentially at market);
   flat ROI **+21.90%**, but the tournament-block CI is
   **[-10.79%, +50.18%]** across 19 competition blocks. ≥$100k:
   LL 0.5929, ROI **+26.39%**, block CI **[-17.36%, +46.42%]** across
   11 blocks. The earlier positive i.i.d. lower bounds, including the
   close-match and single-month claims, are superseded by I3 and no longer
   establish a production betting edge. M7 remains the direct probability
   model of record; A7 remains the predeclared forward betting policy, with
   economic performance unconfirmed. See
   `reports/i3_eval_statistics_hardening.md`.

   **Frozen forward result (2026-07-23):** on the preregistered ≥$50k slice,
   M7 LL is **0.6823** versus market 0.7445 and ball-v7 0.7015, so probability
   confirmation passed. M7 A7 returned +96.72%, but its block interval is
   [-3.29%, +623.85%] across only five betting blocks, so economic
   confirmation failed. Removing the only two winners priced below 25% leaves
   M7 LL 0.6654 versus market 0.6824 and M7 A7 ROI +20.04%. See
   `reports/forward_evaluation_2026-06-01_2026-07-13.md`.

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

   **Prop-bet framework (2026-05-12; REVISED 2026-07-24 by I13)**:
   `scripts/sim_eval/prop_backtest.py` backtests ~25 prop families against
   cricsheet actuals. **E2 v2 fair-baseline audit
   (`reports/e2_prop_fair_baselines.md`): no binary prop family beats an
   as-of fair baseline** (EB-shrunk career/venue/positional plus
   usage-aware `top_bowler`) — the
   2026-05-12 "ship as-is" list was a base-rate artifact; any prop claim
   must clear the versioned bar in
   `scripts/sim_eval/prop_fair_baselines.py`, not base rates.
   **E5 root cause of the tail-event overshoot
   (`reports/e5_class_weight_fix.md`)**: v7 trains with `balanced` class
   weights and the sim sampled the tilted probabilities raw (P(wkt) 2×
   actual per ball). Fixed by the val-fit `VectorScalingCalibrator`
   (`models/xgb_v3/vector_scaling_calibrator_v1.pkl`) — pass
   `--ball-calibrator vector` to `prop_backtest.py` for all prop/score
   work. Calibrated, the PP-total overshoot disappears, bowler-wicket
   overshoot halves, but the I13 usage-share baseline beats calibrated
   `top_bowler` CI-clean (sim − baseline +0.0038
   [+0.0026, +0.0051]). The old "first binary family with skill" claim is
   superseded. Known cost: per-batter runs MAE (read that from the raw sim
   or the career baseline). "Inverse plays" are dead — fair baselines beat
   both the raw sim and its inverse.

   **In-play win probability (E6, 2026-06-09)**: use
   `models/inplay_winprob_v1` (direct P(win|ball state), LL 0.5418 / AUC
   0.80 on 780 OOS matches) for win-prob worms / in-play scenarios — NOT
   the sim (crease/momentum extras add nothing over chase-math + rating;
   `reports/e6_inplay_winprob.md`).
   The sim now defaults to a phase-aware **`EmpiricalBowlerSelector`**
   (historical usage, EB-shrunk; `models/bowler_phase_usage.json` built
   by `scripts/build_bowler_phase_usage.py`) instead of the old
   `RandomBowlerSelector`. It passes winner-market LL parity (G1),
   top-batter no-regression (G3), and ≥90% bowler coverage (G5); halves
   the team-fours over-count bias. Pass `--bowler-selector random` to
   `run_sim_eval.py` for A/B baselines.

   **I5 experimental legal/off-bat stack (2026-07-24; not promoted):**
   `legal_off_bat_v1` trains only on legal off-bat outcomes and composes
   validation-fitted extras separately. It is isolated under the `i5` cache,
   parquet, model, calibrator, and evaluation paths; production v3/v7 remains
   the model of record. Raw ball and historical match LL improve slightly, but
   calibrated ball LL is worse than v3, historical ROI is long-shot-dependent,
   and the full paired n=261 prop gate fails on PP >55.5 plus all three
   innings-total lines despite improving bowler 2+/3+ wicket Brier. Raw threes
   are preserved but stay in the combined 2/3 class: they are 0.4224% of legal
   balls and range from 1.0587% at the MCG to 0.2464% at Mirpur. Do not
   prioritize a separate three-run draw until physical ground dimensions exist
   and an ablation justifies the added calibration, strike-rotation complexity,
   and simulation runtime. Do not score the consumed forward set with I5. See
   `reports/i5_legal_off_bat_evaluation_20260724.md` and `docs/OPERATIONS.md`
   § "I5 legal/off-bat experimental pipeline".

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
- **Sealed forward set** (2026-07-23):
  `data/forward_holdout/2026-06-01_2026-07-13/` (137 matches,
  matched 2026-06-02 → 2026-07-13; 61 at ≥$50k, 30 at ≥$100k).
  Constructed from the separate strict Polymarket extractor with every
  quote strictly before explicit scheduled start, exact H2H-only markets,
  male-T20 Cricsheet joins, outcome-blind selection, provenance hashes,
  and zero overlap with older pools. The protocol was frozen before scoring;
  both outcome-free prediction artifacts and the post-lock report are
  checksummed and committed. This set is now consumed and must never be used
  for fitting, calibration, threshold selection, or candidate selection. See
  `docs/FORWARD_HOLDOUT.md` and
  `reports/forward_evaluation_2026-06-01_2026-07-13.md`.

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
    --sim-json eval_out/phase5_hier/hier_all_20260425_165622.json \
    --direct-json models/xgb_match_v2_clean/test_predictions.json \
    --w 0.0 0.2 0.5 0.8 1.0 \
    --out-dir eval_out/blend_a2_clean
for w in w0p00 w0p20 w0p50 w0p80 w1p00; do
  uv run python scripts/sim_eval/reslice_eval_json.py \
    --in eval_out/blend_a2_clean/hier_all_20260425_165622_${w}.json \
    --odds betting_odds_polymarket.json \
    --out-dir eval_out/blend_a2_clean/sliced
done
uv run python scripts/sim_eval/blend_report.py \
    --sliced-dir eval_out/blend_a2_clean/sliced \
    --direct-json models/xgb_match_v2_clean/test_predictions.json \
    --out reports/blend_a2_clean_report.md

# === Predict an upcoming fixture ===
# Hand-write fixtures/<match>.json (see fixtures/_template.json), then:
uv run python scripts/predict_fixture.py \
  --fixture fixtures/<match>.json \
  --state-dir data/forward_state/2026-06-01_2026-07-13 \
  --tracker-snapshot tmp/live_state/tracker_snapshot_2026-07-13.pkl \
  --tracker-source-dir data/t20s_json \
  --tracker-source-dir \
    data/forward_holdout/2026-06-01_2026-07-13/context_t20s_json
# Add --rebuild-snapshot once if that matching snapshot does not exist.
# The default 2026-04-16 production state now fails when >14 days stale.
# A7 output is ≥$50k, exact-policy, shadow-only, and never authorizes
# execution. See docs/OPERATIONS.md § "Operation 6".

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

# === Prop-bet backtest (v7 sim) ===
# Build the bowler-usage prior once (skip if models/bowler_phase_usage.json
# is current), then backtest prop families vs cricsheet actuals:
uv run python scripts/build_bowler_phase_usage.py \
    --source-dir data/t20s_json --out models/bowler_phase_usage.json
uv run python scripts/sim_eval/prop_backtest.py \
    --test-dir data/polymarket_test --n-sims 100 \
    --out reports/prop_calibration_report_emp_n261.md
# Per-match drilldowns + A/B vs random selector:
uv run python scripts/sim_eval/render_prop_per_match.py \
    --detail reports/prop_calibration_detail_emp_n261.json \
    --out-dir reports/prop_per_match/
uv run python scripts/sim_eval/compare_selector_eval.py \
    --left reports/prop_calibration_detail_emp_n60.json \
    --right reports/prop_calibration_detail_rand_n60.json \
    --out reports/prop_selector_comparison_n60.md
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
| How is the sealed forward set built and protected? | [docs/FORWARD_HOLDOUT.md](docs/FORWARD_HOLDOUT.md) |
| What exactly changed in deterministic same-day state? | [docs/I6_SAME_DAY_ORDERING_AUDIT.md](docs/I6_SAME_DAY_ORDERING_AUDIT.md) |
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
5. **Same-day match order is a versioned data contract.** Every tracker walk
   must use `(match_date, Cricsheet match_id)` via `loaders_common.py`
   (`date_then_match_id_lexicographic_v1`). Within-date trackers carry state
   across siblings, so filesystem order is forbidden. Deterministic caches
   persist the version in `_meta`; match materializers fail closed when it is
   absent or different.
6. **Forward context is state, not training data.** Build it only under
   `data/forward_state/` with `build_forward_state.py`; never overwrite the
   production cache/parquets or recompute global/phase priors over the future
   context. The sidecar freezes those priors from the pre-holdout production
   cache and does not load a model. Direct match evaluation uses its
   deterministic match rows. Ball simulation additionally requires
   sequential same-day replay; date-only SQLite queries are insufficient.
7. **Match-winner ROI uncertainty uses I3 blocks, never match-level i.i.d.**
   Use `sim_eval/eval_statistics.py` contract
   `tournament_time_block_v1`: 10,000 seed-42 whole-event resamples, explicit
   bet placement, and `<10 blocks = descriptive`. Historical i.i.d. CI-clean
   claims are superseded.

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
├── eval_out/             # ALL eval run outputs (eval_out/<tag>/); gitignored, regenerable.
│                         #   Never write eval output anywhere else. Retired → archive/eval_results/
├── research/             # autonomous overnight loop: IDEAS.md queue, results.tsv verdict log,
│                         #   night.sh runner, reports/auto/<id>.md. Constitution: program.md
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
