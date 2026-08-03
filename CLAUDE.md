# CricML — Agent Guide

T20 cricket match prediction. Predict ball outcomes
(`{0:dot, 1:one, 2:two, 3:four, 4:six, 5:wicket}`), simulate matches via
Monte Carlo, evaluate vs market odds (Polymarket primary, bookmaker legacy).

**Active models** (two production models with complementary roles):

1. **Match-level direct (winner-market predictor)** — XGBoost binary classifier
   on 49 match-level features (M1 baseline + 3 venue outcome-dist).
   Trained on `team1_wins`. **Production variant of record (post I17/I18/I19
   promotion, 2026-07-31): `models/xgb_match_i7_swap_production/`** —
   the M7 config plus train-time team-swap symmetry augmentation
   (`--swap-augment`), seed 29, trained on the I7 identity frame
   (frame of record `data/xgb_match_data_i7_v2`, `cricsheet_primary_v1`).
   The artifact is the I17 audited swap arm promoted verbatim
   (`models/auto/i19/swap_seed29`, model.pkl byte-identical to the I18
   golden-audited `models/auto/i17/swap_seed29`). Evidence: D12 swap
   transfer confirmed on 5/5 paired seeds on the i7 frame (mean ≥$50k ΔLL
   −0.0144, floor 0.007, `docs/I17_I7_SWAP_SUCCESSOR.md`); golden audit —
   swap-i7 beats the slice-matched market LL on both sharp golden slices
   where the legacy line trails (`research/reports/auto/I18.md`). It trails
   the legacy line by ~0.005–0.009 LL on the iteration ≥$50k slice; the
   promotion rationale is operational — the legacy serving state ended
   2026-04-16 and is unregenerable, and only the i7 stack has a fresh-state
   build path. Serving state lives in `data/live_state_i7/` (see
   `docs/SWAP_I7_PROMOTION_CUTOVER_PLAN.md`). The previous production
   `models/xgb_match_v3_m7_swap_production/` (legacy identity, D12
   promotion 2026-07-30) and frozen `models/xgb_match_v3_m7_production/`
   are retained for rollback and reachable only through the legacy replay
   contract.
   M7 config: lr 0.05, colsample 0.9 (from the M7 sweep; prior config was
   over-aggressive). `predict_fixture.py` uses this model directly; raw
   probabilities (no Platt — Platt over-corrects on this config and kills
   iteration ROI). Per-fixture unfrozen rehydration semantics
   (chronological tracker walk through pre-match date). Earlier baselines
   preserved for reference: `xgb_match_v2_clean`,
   `xgb_match_v2_clean_unfrozen`, `xgb_match_v3_baseline` (M1 monotone),
   `xgb_match_v3_m2_venue_only`, `xgb_match_v3_m2_venue_only_unfrozen`.
   See `reports/m7_architecture_eval.md` for the sweep, M2 → M7
   improvement deltas, and the explanation of why M3–M6 feature work
   dropped.

   **Honest headline (iteration Polymarket eval; I3 blocks, updated
   2026-07-30 for the swap model)**: ≥$50k raw LL **0.6215** (base M7
   0.6299) vs slice-matched market 0.6482 — note the long-quoted 0.6267 is
   the all-261 market LL, where the swap model scores 0.6178; flat ROI
   **+24.53%** with tournament-block CI **[-1.98%, +46.37%]** across 19
   competition blocks (base +21.90% [-10.48, +49.94]). ≥$100k: LL 0.5796,
   ROI +26.60%, block CI [-17.21%, +45.42%] across 11 blocks. Golden audit
   (descriptive, 5–6 blocks): beats matched market LL on both slices where
   base does not. Adoption rests on D12's paired 5-seed evidence (ΔLL
   −0.0092 and ΔROI +3.39pp, both better on 5/5 seeds), not on a CI-clean
   ROI claim — block CIs still straddle zero everywhere, so no production
   betting edge is established. A7 remains the predeclared forward betting
   policy, with economic performance unconfirmed. See
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

   **The Hundred (2026-07-27)**: the match model runs on 100-ball fixtures
   via `--tracker-aux-dir` / `--team-aliases` / `--state-version`
   (`docs/OPERATIONS.md` § "Operation 7"). After the 2026-07-30 alias
   copy-fold fix and rerun, it picks the winner **61.0%** of the time over
   159 historical Hundred matches (i.i.d. p = 0.0034; season-block sign
   test p ≈ 0.03 — see the report's "Known limitations") but its
   probabilities compress into 0.37–0.62 (mean |p−0.5| 0.036 vs 0.105 on a
   401-match T20 control), so LL 0.6795 vs coinflip 0.6931, and it lands
   within ~3pp of the Polymarket line on every 2026 fixture. **Directional
   lean only — no edge, no betting.** The Hundred path deliberately uses
   `models/xgb_match_i7` with `--venue-identity-mode i7` and a matching
   canonical-venue cache. Frozen `xgb_match_v3_m7_production` remains
   available only through the temporary legacy replay contract; do not extend
   that mode with Hundred state. See `reports/hundred_2026_adaptation.md`.

   **Successor line (I17, 2026-07-30):** swap + M7 config trained on the
   I7 identity frame (`data/xgb_match_data_i7`) is the designated
   production-successor configuration — D12 swap transfer confirmed on
   5/5 paired seeds (mean ≥$50k ΔLL −0.0144, floor 0.007); beats the
   slice-matched market LL on 5/5 seeds where base does on 4/5. Promotion
   is a separate decision gated on an i7 golden-frame audit and the
   fresh-state serving cutover plan. See `docs/I17_I7_SWAP_SUCCESSOR.md`
   and `reports/i17_i7_swap_eval_20260730.md`.

2. **Ball-level sim** — **production of record (promoted 2026-08-02):
   `models/xgb_i7_noweights_production/`** — the D16 no-class-weights
   retrain on the i7 identity frame (`data/xgb_data_i7`,
   `venue_aliases_v1`, i7 stats cache), served **RAW — no ball
   calibrator**. The artifact is the D16 arm promoted verbatim
   (booster md5 `7ee1e180…`, sidecars byte-identical to the archived
   `models/xgb_i7` encoders). Evidence: D16 — no-weights RAW dominates
   the calibrated legacy-design stack (ball LL 1.5072 → 1.4253; pooled
   tail dBrier −0.0116 CI-clean; batter_runs_mae 14.435 → 13.891
   CI-clean; 11 favorable CI-clean movers, 0 regressions in 33
   families); D17 — calibrator-on-top is a decision-grade null (the
   E5→…→B8 marginal-calibration chain is CLOSED; never add a vector
   calibrator to this stack); D18 — hyperparameter re-tune fails to
   transfer (better val LL, prop-level noise), config stands at the
   swept lr 0.2404 / best_iteration 24. See
   `research/reports/auto/{D16,D17,D18}.md`.
   `scripts/sim_eval/prop_backtest.py` defaults now load this stack
   (`--stats-version i7`); `--ball-calibrator vector` is legacy-replay
   only and requires an explicit `--ball-calibrator-path`.
   **Use this for prop bets, score distributions, in-play scenarios** —
   anywhere ball-level resolution matters and match-level supervision
   can't help. No betting claim attaches: E2/I13 fair-baseline results
   are unchanged.

   **Legacy line (retired to replay)**: XGBoost v7, 114 features (V3 +
   42 outcome-dist), hierarchical shrinkage (Phase 5 2026-04-25);
   k_player=30, k_venue=200. Config:
   `experiments/configs/xgb_v6_hierarchical_shrink.yaml`. Lives on the
   pre-I7 frame (`data/xgb_data_v3`, 467 raw venue strings), which
   **fail-closes under the I7 identity contract and can no longer be
   trained** (D6). Replay needs explicit `--model-path models/xgb_v3/…
   --stats-version v3 --ball-calibrator vector --ball-calibrator-path
   models/xgb_v3/vector_scaling_calibrator_v1.pkl`. v7 lost the
   winner-market race to the clean direct model by ~0.07 LL on ≥$50k;
   leakage-audited clean (`reports/v7_leakage_audit.md`).

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
   actual per ball). Patched at the time by the val-fit
   `VectorScalingCalibrator`
   (`models/xgb_v3/vector_scaling_calibrator_v1.pkl`) — **legacy-replay
   only since the 2026-08-02 promotion**: the production i7 no-weights
   stack removes the tilt at training time and must run raw (D17). Calibrated, the PP-total overshoot disappears, bowler-wicket
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
- **Golden set** (2026-05-09; extended 2026-07-30): `data/golden/polymarket_test/` +
  `data/golden/betting_odds_golden.json` (**124** matches, 2026-04-17 →
  2026-06-17; the original 55 rows are preserved verbatim and the 137
  consumed forward fixtures are excluded). Truly out-of-sample — never
  seen by training, validation, or selection. Extended-window audit
  (WC-heavy, market-sharpest slice): swap beats base everywhere but both
  trail the market on LL; see
  `reports/golden_extension_eval_20260730.md`.
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

## Women's track (I12 / I12-L) — separate corpora, no edge

Two fully isolated women's families, neither production and neither carrying
any betting claim. They share no artifact with the men's line.

| family | corpus | model dirs | frame | cache |
|---|---|---|---|---|
| **w1** | 2,086 women's T20Is (`data/w_t20s_json/`) | `models/xgb_match_w1_{base,swap}` | `data/xgb_match_data_w1` | `player_stats_cache_w1.sqlite` |
| **w2** | 1,206 women's league T20s (`data/w_league_json/`) | `models/xgb_match_w2_{base,swap}` | `data/xgb_match_data_w2` | `player_stats_cache_w2.sqlite` |

**Market odds exist** (found 2026-08-01; the I12 memo's "no odds" premise was
an artifact of a men's-scoped pull). Women's internationals hide from
slug-based filters because Gamma carries the gender only in the *event
title* — see `docs/I12_WOMENS_TRACK_SCOPING.md`. Pull with
`extract_match_prematch_odds_strict.py --gender female --format t20+hundred`,
join with `scripts/build_womens_polymarket_odds.py`, evaluate with
`scripts/eval_womens_market.py`. Joined sets live in
`data/womens_polymarket/` (175 T20I fixtures) and
`data/womens_polymarket_leagues/` (51 league fixtures); both are gitignored
and regenerable, and neither touches any men's odds set.

**Honest state**: w1 clears the coinflip and ELO-baseline gates (test LL
0.4484, 81% acc) but **loses to the market on log loss on every liquidity
slice**. w2 **fails** the gates outright — the ELO-only baseline beats it on
both splits. The gap between them is the finding: the same pipeline scores
0.4484 on the associate-heavy T20I pool and 0.6858 on evenly-matched
franchise sides, so most of w1's headline is roster mismatch rather than
cricket modelling. A members-only w1 slice is the required next step.

When reading any women's market table, check the coinflip guard first: thin
league books frequently score *worse* than a coinflip, and "the model beat
the market" on such a slice means nothing. `eval_womens_market.py` flags
those slices with `!` and reports an informative-slice count.

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
uv run python scripts/predict_fixture.py --fixture fixtures/<match>.json
# Defaults (post 2026-07-31 promotion): models/xgb_match_i7_swap_production,
# i7 identity mode, state in data/live_state_i7/ (cache + tracker snapshot,
# currently through 2026-07-13). Add --rebuild-snapshot once after any cache
# refresh. State >14 days behind the fixture still fails loudly — refresh
# cricsheet + rebuild data/live_state_i7 before a live fixture (OPERATIONS
# § "Operation 6"). Legacy replay of the pre-I7 production family needs
# explicit --venue-identity-mode legacy + the old model/state paths.
# A7 output is ≥$50k, exact-policy, shadow-only, and never authorizes
# execution. See docs/OPERATIONS.md § "Operation 6".

# === Golden eval refresh — i7 production line (one command) ===
# Reproduces the I18 audit numbers for the current production model:
bash scripts/refresh_golden_i7.sh
# (legacy-line refresh below is retained for the retired v2_clean family)

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
├── program.md            # research-loop constitution (repo root, NOT research/)
├── research/             # autonomous overnight loop: IDEAS.md queue, results.tsv verdict log,
│                         #   night_v3.sh runner (v1/v2 superseded), reports/auto/<id>.md
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
