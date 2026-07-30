# TODO

> **Current evaluation note (I3, 2026-07-23):** match-winner headline
> intervals now use 10,000 seed-42 whole-competition resamples, not
> per-match i.i.d. bootstrap. The old positive ROI lower bounds in this
> historical task log are superseded: M7 ≥$50k is +21.90%
> `[-10.79%, +50.18%]`, A7 is +36.93% `[-1.52%, +59.81%]`, and ball v7
> is +6.11% `[-7.99%, +25.70%]`. See
> `reports/i3_eval_statistics_hardening.md`. The new forward holdout is
> sealed and remains unscored.

## Open decisions (human)
- [x] **Adopt D12 swap augmentation in production?** DONE 2026-07-30:
  archived D12 swap arm promoted as
  `models/xgb_match_v3_m7_swap_production`; `predict_fixture.py` switched.
  Iteration I3-block + golden confirmation both favorable (block ROI CIs
  still straddle zero — no CI-clean edge claim). See IDEAS.md **I16** and
  `reports/d12_swap_promotion_20260730.md`.

## Completed
- ✅ Make the outcome categories: squish/ round down to 0,1,2,4,6,W
- ✅ LSTM model architecture (scripts/lstm_v1.py, LSTMModelV1 in sim_v1_2.py)
- ✅ LSTM hyperparameter improvements (FocalLoss, WarmupCosine scheduler, LayerNorm)
- ✅ LSTM feature extraction alignment (59 continuous features)
- ✅ Root cause analysis: Both XGBoost and LSTM have weak match predictions
- ✅ Fix XGBoost class_to_outcome bug (sim_v1_2.py) — was 8-class, should be 6-class
- ✅ Feature registry (scripts/feature_registry.py) — central feature definitions
- ✅ Experiment tracking infrastructure (scripts/experiment_tracker.py)
- ✅ Experiment config schema (experiments/configs/*.yaml)
- ✅ Pipeline runner (scripts/run_experiment.py)
- ✅ Experiment comparison tool (scripts/compare_experiments.py)
- ✅ **Cricsheet data refresh** (scripts/fetch_cricsheet.py) — 14 men's T20 leagues (added BPL, PSL, SMAT, LPL, Super Smash); match corpus grew 8,341 → 11,264 (+2,923); latest match date 2025-06-15 → 2026-04-16. Append-only merge, SHA-256 manifest, idempotent re-runs.
- ✅ **Player enrichment via R cricketdata** (scripts/enrich_players_cricketdata.py) — filled 737 missing player bio rows in `all_players_enriched.csv` (10,519 → 11,256) using `cricketdata::fetch_player_meta` directly; no website scraping needed since Cricsheet's `people.csv` already carries `key_cricinfo` for every player we see.

## Next Steps (post data-refresh)
Sequential. Each step unblocks the next; the ultimate goal is a 500+ match betting eval set.

1. [ ] **Update hardcoded date splits in `parsing_v2.py`** (lines ~1180-1188). Current splits stop at 2024-09-30 / betting 2024-06-01..2024-06-29, but data now runs to 2026-04-16. Proposed: `train_end=2024-12-31`, `val_end=2025-06-30`, `test_end=2025-12-31`, `golden_start=2026-01-01`, `betting_start=2026-01-01`, `betting_end=2026-04-16`. That betting window is what becomes the expanded eval set.
2. [ ] **Parse-time gender filter in `parsing_v2.py`** — skip matches where `info.gender != 'male'`. Today only ELO is gender-segregated; women's ball outcomes still leak into training.
3. [x] **Rebuild features + stats cache**: now `uv run python scripts/build_stats_cache.py` (~7 min, JSON → SQLite) followed by `uv run python scripts/materialize_features.py` (~5 min, SQLite + JSON → parquet). The legacy `models/cache_chunks_v3/` chunked format was removed in the Phase B / Phase 5 cleanup (2026-04-22, IMPROVEMENTS.md §"Parsing Pipeline Split").
4. [ ] **Retrain XGBoost v3 baseline** on the expanded corpus: `uv run python scripts/run_experiment.py experiments/configs/xgb_v3_baseline.yaml --skip-parsing`. Expect team-strength and ELO features to sharpen now that we have ~33% more matches.
5. [ ] **Polymarket odds ingestion** — wire `/Users/aryamangupta/Projects/polymarket-cricket/data/polymarket_prematch_odds.json` (1,161 resolved markets) into `scripts/sim_eval/loaders.py::BettingOddsLoader`. Use the team-name mapping table in `docs/DATA_REFRESH_HANDOFF.md`. This is the step that actually moves the eval set from 44 → 500+.
6. [ ] **Re-run sim eval** on the expanded test + Polymarket odds: `uv run python scripts/sim_eval/run_sim_eval.py --test-dir data/betting_test --odds <new-odds-file>.json --n-sims 1000`. Compare log loss / Brier / ROI against the old 44-match baseline.

## Evaluation Discipline (P0 — post-polymarket-baseline, 2026-04-19)

Gate items before any new modeling work. Baseline is frozen at XGBoost v3 / clean-dedup Polymarket: model LL 0.7319, market LL 0.6267, coinflip LL 0.6931, flat ROI +0.06% (CI [−19%, +23%]), always-favorite baseline +4.15%. Model is worse than coinflip on log loss and worse than always-favorite on ROI. Close these gaps before claiming skill.

### Blockers (must fix before next eval run)
- [x] **Fix incomplete-lineup bug** (2026-04-19) — `_extract_team_players` rewritten to use `info.players[team]` as the authoritative roster; 104 dummy `Player` objects eliminated across 96 matches. Affected-slice ΔLL = −0.049 (paired-bootstrap 95% CI [−0.082, −0.021], excludes 0); unaffected slice bit-identical; flat ROI +0.06% → +2.78%. See `docs/POLYMARKET_INTEGRATION.md` → "Lineup-fix impact". Regression guards: `scripts/tests/test_lineup_extraction.py` (8 tests).
- [x] **Support Impact Player / 12-man squads (Option B)** (2026-04-19) — dropped `[:11]` slice in `sim_eval/loaders.py`; widened `range(11)` to `range(len(lineup.players))` at `sim_v1_2.py::get_next_batsman_idx` and `get_available_bowlers`. 93 team-match pairs (48 matches, ~18% of eval) now expose the 12th eligible player to the simulator. Eleven-only slice bit-identical (0 / 208 matches changed); impact slice ΔLL = +0.037, 95% CI [−0.003, +0.081] — **correctness fix, not a score improvement** (CI includes 0). Flat P&L on impact slice +0.61 across 47 matches. See `docs/POLYMARKET_INTEGRATION.md` → "Impact-Player support — Option B". Regression guards: 2 new unit tests (`test_twelve_man_squad_preserved`, `test_impact_sub_in_deliveries_and_roster`). Open follow-up: model the timed swap event (`delivery.replacements.match`) to match real substitution semantics.
- [ ] **Match-level calibration layer** — fit isotonic (primary) + Platt (sanity) on the 261 outcome-matches, LOOCV, predict-from-team1-prob. Reframed 2026-05-08 from "P0 blocker" to "measurement hygiene tool" after conceptual review (see IMPROVEMENTS.md §"Calibration vs. resolution"):
  - **What it does**: monotone redistribution of probability mass to match empirical bin frequencies. Cannot create resolution; can only redistribute it.
  - **Expected mechanical effect on us**: pulls over-dispersed bins (74%→51%, 16%→40%) toward the empirical rate, which is closer to 50%. LL improves by ~0.05–0.06 (0.7402 → ~0.68). Flat ROI likely *drops* — same pattern as the 2026-03 Platt experiments and the 2026-04-23 v6 work (LL −5.3%, flat ROI +6.5% → −7.1%). A perfectly calibrated coinflip has LL 0.6931, better than v7's 0.7402, with zero edge — calibration is trivially satisfiable and tells you nothing about edge.
  - **What it does NOT do**: close the go/no-go gate. The gate (`model LL < market LL on ≥$50k AND ROI CI > 0`) is a *resolution* problem; the LL gap of 0.114 vs market exists because the model under-discriminates strong vs weak teams, not because its bins are badly calibrated. Calibration alone doesn't clear it; feature/architecture work that adds discriminative signal does.
  - **When it IS worth doing**: (a) when feature ablations are getting hard to compare cleanly because of calibration drift across versions, (b) for honest Kelly sizing (Kelly assumes calibrated probs), (c) for any "only bet when edge > 3%" decision rule. Treat as supporting infrastructure, not a feature work prerequisite.

### Benchmark stack (mandatory columns for every experiment result)
- [ ] Wire a `benchmarks` helper into `scripts/sim_eval/run_sim_eval.py` that emits all four side-by-side:

  | Benchmark | Purpose | Current value |
  |---|---|---|
  | Coinflip (50/50) | Sanity floor — any model below this is broken | 0.6931 LL |
  | Always-bet-favorite | Honest ROI baseline (no model needed) | +4.15% flat ROI, 64% WR |
  | Polymarket market | Ceiling — closing this is the real win | 0.6267 LL |
  | Our model | Subject under test | 0.7319 LL, +0.06% ROI |

  Reject any experiment that improves model ROI but not model log loss vs market — that's counterparty noise, not skill.

### Reporting template (three slices, every time)
- [x] All 261 matches — maximum n, noisy markets
- [x] ≥$50K volume subset (170) — honest liquidity gate (`--min-volume 50000`)
- [x] ≥$100K volume subset (110) — tighter, market is sharp (`--min-volume 100000`)
- [x] Bootstrap 95% CIs on ROI (≥1,000 resamples) — landed Phase 1 (2026-04-24): `MatchLevelEvaluator._bootstrap_ci` emits to `OverallEvaluationResults.{avg_log_loss,flat_roi}_ci_{low,high}` and saved JSON. `scripts/sim_eval/compare_slices.py` for cross-model tables; `scripts/sim_eval/reslice_eval_json.py` for post-hoc reslicing. `scripts/run_sliced_eval.sh` runs all 3 slices in one shot.

### Go / no-go rule for any "we can bet this" claim
Both conditions required:
1. Model log loss < market log loss on the ≥$50K slice
2. Flat-ROI bootstrap CI on the ≥$50K slice excludes zero

Nothing weaker ships.

### Preserve a true holdout
- [ ] Treat `data/polymarket_test/` (2025-07-01 → 2026-04-16) as an iteration set. After ~2–3 iterations, stop tuning against it — overfitting risk is real at 261 matches.
- [ ] Ship a **golden** eval off matches played after **2026-04-17** (the current `golden_start`). Target 50+ matches before declaring any improvement real. Build a `betting_odds_golden.json` from fresh Polymarket snapshots captured genuinely pre-match (not scraped post-hoc), and keep the script in `scripts/fetch_polymarket_live.py` for ongoing appends.

### Odds-build hardening (post clean-dedup)
- [ ] Log the 7 residual winner-disagreements + 23 remaining `top_p > 0.92` entries as warnings in `scripts/build_polymarket_odds.py`, not just in the unmatched audit file. Make them grep-visible in eval runs so regressions in upstream Polymarket data are caught immediately.
- [x] Add `--min-volume` flag to `scripts/sim_eval/run_sim_eval.py` — landed Phase 1 (2026-04-24). Forwards from `BettingOddsLoader.load_odds(min_volume=...)`; null preserves non-polymarket odds files. Also threaded through `run_experiment.py` via YAML `evaluation.min_volume`.

## v6→v7 outcome-distribution follow-ups (2026-04-24/25)

The 6-phase plan to evaluate v6 + four feature/refactor experiments. Outcome:

- [x] **Phase 1**: Per-slice eval infrastructure shipped. v6 sliced metrics: LL win over v4 grows with liquidity (Δ −0.040 → −0.074); ROI noisy on both, all CIs straddle zero. See `project_v6_sliced_eval.md`.
- [x] **Phase 2**: Drop-encodings ablation → **KEEP** `batter_encoded`/`bowler_encoded`. Δ LL +0.069 on ≥$50k slice (14× threshold). See `project_phase2_drop_encodings.md`.
- [x] **Phase 3**: Phase prior → **DROP** `phase_outcome_dist`. Δ LL +0.022/+0.023/+0.036 across slices; collinear with is_powerplay/middle/death. Implementation kept inert in code; resurrect via `experiments/configs/xgb_v7_phase_prior.yaml`. See `project_phase3_phase_prior.md`.
- [ ] **Phase 4**: `build_ball_features` refactor — DEFERRED. Pure plumbing (consolidate the 5 sim wrappers' duplicated feature-build code into one helper); not a feature change. ~4h estimated; must keep Phase A parity at 9519/9519.
- [x] **Phase 5**: Hierarchical shrinkage → **SHIP**. vs-type/vs-hand cells shrink toward player overall (not π directly). LL ~flat (Δ ~+0.003 on all/≥$50k), but flat ROI swings +15pp (all) / +16pp (≥$50k); CIs still straddle zero. See `project_phase5_hierarchical_shrink.md`.
- [x] **Phase 6**: k_player sweep ∈ {10, 30, 100, 300} → **k=30 confirmed optimal**. k=10 noisier (worse on both LL and ROI); k=100/300 over-shrink toward π. v7 = Phase 5 hierarchical, k_player=30, k_venue=200. See `project_phase6_k_sweep.md`.

**Active model**: v7 = `experiments/configs/xgb_v6_hierarchical_shrink.yaml` (despite "v6" in the name).
**Sliced champion (≥$50k)**: v7 LL 0.7402, flat ROI +6.11% (CI [−10.7, +23.9]); v6 was 0.7370 / −9.81%; v4 was 0.7838 / +12.10%. All ROI CIs still straddle zero — none of these models clears the go/no-go bar above.

### Open follow-ups from the outcome-dist plan
- [ ] Phase 4 refactor (deferred above).
- [ ] Separate `k_narrow` sweep (Phase 5/6 held k_narrow = k_player = 30 throughout; an independent sweep could widen the hierarchical ROI win).
- [ ] Per-batter / per-bowler phase distributions (the Phase 3 plan's natural successor — those WOULDN'T be collinear with is_powerplay since they vary across players).

## Match-level direct + sim ensemble (LANDED 2026-05-09)

The v7 sim was hitting LL 0.7402 on the ≥$50k polymarket slice — 0.114 above market, and *worse than coinflip* on the sharpest matches. Diagnosed as a **resolution problem** (not calibration; see IMPROVEMENTS.md §"Calibration vs. Resolution"). Architectural fix: a direct match-level XGBoost binary classifier supervised on `team1_wins`, blended post-hoc with v7 sim outputs in logit space `logit(P_final) = w·logit(P_sim) + (1−w)·logit(P_direct)`. Plan: `~/.claude/plans/okay-let-s-go-ahead-reflective-sunrise.md`.

### Phase A1 — cheap-subset features (~26 features)
- Files: `scripts/materialize_match_features.py`, `scripts/xgboost_match_v1.py`, `scripts/sim_eval/blend_eval_json.py`, `scripts/sim_eval/blend_report.py`, `experiments/configs/xgb_match_v1_baseline.yaml`.
- Direct alone LL 0.6568 on the polymarket-overlap subset (255 of 261); v7 sim was 0.7158. LL-vs-w curve **monotone increasing** — sim adds no LL value. Direct beat sim on 119/261 matches per-match.
- Per-slice gate: NOT cleared (LL still > market 0.6267). Report: `reports/blend_a1_report.md`.

### Phase A2 — richer features (~47 features)
- Added `TeamFormTracker` (last-10 win rate), `H2HTracker` (Beta(1,1)-shrunk), `HomeVenueTracker` (3+ matches in 730d), lineup mix (LHB/pace/spinner counts), **top-6 batting / bottom-5 bowling ELO splits**.
- Top importance: `bottom5_bowling_elo_diff` (0.084), `top6_batting_elo_diff` (0.075). Position-split ELOs do more work than v7 sim's lineup-wide aggregates.
- Standalone polymarket-overlap LL: **0.5226** (vs A1 0.6568, vs market 0.6267, vs v7 0.7158).

### No-leakage diagnostic (added 2026-05-09)
- New `--freeze-trackers-after DATE` flag: per-match fresh SQLite rehydration as-of `freeze_date+1` in test period; A2 trackers stop updating past freeze. Prevents within-test cross-match contamination.
- **Counter-intuitive finding: FROZEN is BETTER than unfrozen** across every slice. Tracker contamination ruled out — unfrozen mode hurt by drifting test features past the temporal scope the model trained on.
- Temporal split (early vs late test) shows +33pp ROI gap in BOTH Phase A1 and A2-frozen → not contamination, **composition effect** from late-test T20 World Cup mismatch concentration (47/131 late matches are India/WI/Pakistan vs Namibia/Italy/Netherlands qualifiers).
- Audit pass clean: SQLite `_meta` priors used only by v7 sim, train-target correlations modest (max 0.33), binary-feature effects physically plausible (3.6pp home advantage, no toss-winner effect). Full report: `reports/no_leakage_diagnostic.md`.

### Headline ON ITERATION TEST (A2 frozen, w=0.0) — pre-leakage-fix, *retracted as inflated*

| Slice | LL (95% CI) | Flat ROI (95% CI) | n |
|---|---|---|---|
| all | 0.4944 [0.45, 0.53] | +50.73% [+32.4, +74.4] | 255 |
| ≥$50k | 0.5004 [0.45, 0.56] | +53.67% [+36.0, +73.8] | 168 |
| ≥$100k | 0.4361 [0.37, 0.50] | +58.03% [+33.4, +86.6] | 110 |

These were reported on 2026-05-09 and are **inflated by ~21-30pp ROI of feature leakage** discovered immediately after — see "ELO leakage discovered" subsection below.

### ELO leakage discovered + fixed (2026-05-09)

`materialize_match_features._build_match_record` was reading `temp_elo` AFTER `parse_match_data_v2` updated it with this match's own ball-by-ball outcomes. The `_split_elo` call produced post-match ELOs as features for predicting that same match. The 6 affected features include the model's two highest-importance features (`bottom5_bowling_elo_diff`, `top6_batting_elo_diff`).

Fixed by snapshotting `temp_elo` before parse runs (one-line patch in materialize_match_features.py:519). Retrained as `models/xgb_match_v2_clean/`. Full audit + comparison: `reports/leakage_fix_comparison.md`.

### HONEST headline ON GOLDEN SET (xgb_match_v2_clean, w=0.0)

Truly out-of-sample: 2026-04-17 → 2026-05-07, never seen by training/selection.

| Slice | LL (95% CI) | Flat ROI (95% CI) | n / win-rate |
|---|---|---|---|
| all | 0.6416 [0.59, 0.70] | +20.33% [-12, +49] | 55 / 53.7% |
| ≥$50k | 0.6747 [0.64, 0.72] | +32.61% [-0.20, +63.6] | 50 / 59.2% |
| ≥$100k | 0.6698 [0.63, 0.72] | **+34.75%** [+3.79, +65.5] | 45 / 61.4% |
| Reference: market | 0.6267 | — | — |
| Reference: coinflip | 0.6931 | — | — |

**Strict go/no-go (LL beats market AND ROI CI excludes 0): FAILS on every slice.** LL is approximately market-level on every slice; LL on ≥$50k/$100k is 0.04 *worse* than market.
**Soft go/no-go (ROI CI alone): clears only on ≥$100k slice.**

### Open follow-ups
- [x] **Audit `xgboost_v2.py` (v7 ball-level sim) for analogous leakage** (2026-05-09) — clean. Per-ball features computed before per-ball update_stats / elo_tracker.update; team-level constants computed at match start before any ball-loop mutation; SQLite snapshot per date is first-write-wins (pre-D state); no second-pass `_build_match_record` equivalent. Empirical pass: 10/10 solo-date first-of-day matches show bit-exact match between parquet `striker_elo` / `batting_team_elo` and SQLite pre-D rehydration. v7 also doesn't compute top-6/bottom-5 ELO splits, so the specific bug pattern can't apply. Full report: `reports/v7_leakage_audit.md`.
- [x] **Re-run no-leakage diagnostic with the clean model** (2026-05-09) — frozen-better-than-unfrozen finding survives, magnitude compressed roughly 2×. On polymarket-overlap clean: frozen LL 0.6271 / 0.6339 / 0.5877 vs unfrozen 0.6409 / 0.6437 / 0.6036 (all/≥$50k/≥$100k). On standalone full 782-match test, unfrozen now slightly better (0.6027 vs 0.6180). Late-vs-early temporal gap (~0.07 LL) persists in both variants — composition-effect explanation still holds. Full report: `reports/no_leakage_diagnostic_clean.md`.
- [ ] **Forward test**: capture polymarket pre-match snapshots starting now; in 30-60 days evaluate on 30-60 fresh matches.
- [ ] **Wait for IPL 2026 to complete (late May)**, re-evaluate on full ~70-match IPL-only slice for a sharper domestic-league sanity check.
- [x] **Reframe v7 sim's role**: direct beat sim on golden LL (0.67 vs ~0.74 on ≥$50k). v7 sim repurposed for prop bets — prop-backtest framework landed 2026-05-12 (`scripts/sim_eval/prop_backtest.py`, ~25 families, Brier-skill + bootstrap CIs). Sim has real skill on batter-fours / top-batter-bowler ranking / innings totals; over-states tail events (bowler wicket counts, PP totals) → profitable inverse plays. Required the phase-aware bowler-selector fix (see P2 below). Summary: `reports/prop_framework_summary.md`.
  - [ ] **Close G2 (top-bowler skill)**: improve the ball-level wicket-rate model (which bowler gets wickets *given* they bowl) — the selector fix can't move this. Candidate: bowler-vs-phase wicket priors.
  - [ ] **Productionize inverse plays**: the over-stated families (`bowler_wkts_*`, `pp_total_ou_*`) are consistent enough to fade; needs a sizing rule + live-odds wiring before deployment.
- [ ] **Address outlier sensitivity**: long-shot wins still have outsized PnL impact at small n. Consider an edge-cap or position-size rule before live deployment.
- [ ] **Feature work**: with leakage removed, the clean model is borderline-skilful. Real improvement likely requires features that capture intra-season form / current-IPL performance — the 2025-06-30 frozen state for A2 trackers is now visibly stale.

## Match-level v3 — feature engineering plan (2026-05-10)

Phased rollout to close the 0.05 LL resolution gap to market on the ≥$50k slice. Mirrors the v6→v7 outcome-dist phased shape: eval infra → highest-leverage feature → ablations → architecture sweep → sizing. Full catalog with rationale lives in `IMPROVEMENTS.md` § "Future improvements catalog — match-level direct model".

**Excluded from this plan** (per 2026-05-10 decision): market price as a feature. We don't have market prices on the training corpus, only on the eval set; revisit only if multi-bookmaker historical odds land (C3 in the catalog).

**Per-phase discipline**: each phase ends with a feature ablation run on the iteration set's `≥$50k` slice (170 matches). Keep additions iff Δ val LL < −0.005 *and* iteration ROI CI doesn't materially regress. Document each phase in IMPROVEMENTS.md and a memory file before moving on, exactly like Phases 1–6 of the v6→v7 sequence.

**Reference points** (clean baseline, golden ≥$50k): LL 0.6747, ROI +32.61% [-0.20, +63.6], market LL 0.6267.

### M1 — Eval infrastructure + sizing prep ✅ LANDED 2026-05-10
Eval lens in place for M2+ ablations. Full reference: `reports/m1_baseline_eval.md`.
- [x] **Stratified bootstrap** (D2): `_bootstrap_ci(strata=...)` in `match_evaluator.py` and `reslice_eval_json.py`. Tier×half stratum builder; verified narrows LL CI / widens ROI CI as expected.
- [x] **Adversarial slices** (D3): `--slice {ipl, international, mismatch, close}` in `reslice_eval_json.py` with feature-parquet join. Confirms composition effect: mismatch (n=33) LL 0.47 / ROI +61%; close (n=60) LL 0.72 / ROI -5.7%.
- [x] **Walk-forward eval** (D4): `scripts/sim_eval/eval_walk_forward.py` partitions by YYYY-MM. Iteration ≥$50k report at `reports/walk_forward_m1.md` shows the early/late temporal gap (2026-02 LL 0.53/+52% vs 2026-01 LL 0.67/-5%).
- [~] **CLV measurement** (D1): **BLOCKED ON DATA** (audited 2026-05-10). `betting_odds_polymarket.json` carries only opening-line timestamps; closing-line CLV requires forward capture (C2). Re-evaluate after ≥30 matches with closing snapshots land. See IMPROVEMENTS.md § D1.
- [x] **Calibration as sizing tool** (B4): `scripts/calibrate_match_predictions.py` — Platt LOOCV by default (isotonic regresses LL at val n=525). Writes `*_calibrated.json` adjacent to raw predictions; raw stays the LL-gate metric, calibrated is the sizing metric.
- [x] **Monotonic constraints** (B3): `_MONOTONE_SIGNS` dict in `xgboost_match_v1.py` covers 10 unambiguous directional features. New `--monotone` flag; off by default for backwards compat. New artifact `models/xgb_match_v3_baseline/`.

**Notable M1 outcome (positive surprise)**: M1+Platt is the first variant to clear iteration ≥$50k LL gate (0.6235 < market 0.6267) AND ROI CI > 0 simultaneously. On golden it is ~0.025 LL worse than `xgb_match_v2_clean` — within bootstrap noise on n=50, not promoted to production. M2 baseline is `xgb_match_v3_baseline` (raw, not calibrated) — calibrated layer is for sizing only.

### M2 — Phase/matchup outcome-dist transfer ✅ LANDED (venue-only) 2026-05-10
Mixed result. Full reference: `reports/m2_outcome_dist_eval.md`.
- [x] Added 21 outcome-dist features to `materialize_match_features.py` (top-6 batter pX_expected vs opp attack mix; bottom-5 bowler pX_expected vs opp batting hand mix; venue pX). Parquet at `data/xgb_match_data_v3_m2/`.
- [x] Trainer `--drop-features` substring filter for clean drop-one ablation.
- [x] Drop-one ablation across batter / bowler / venue groups → bowler group actively HURTS (test LL +0.006 when included; bottom-5 squad-order is not a clean bowling unit). Batter group neutral. Venue group helps.
- [~] Strict iteration ≥$50k LL Δ ≥ 0.01 gate **NOT cleared** (best M2 variant 0.6266 vs M1 0.6302 = Δ -0.0036; full M2 was Δ +0.0072 worse). Per drop-one rule, landed cleaner subset rather than full feature group.

**Landed: M2 venue-only (3 features added)** — `models/xgb_match_v3_m2_venue_only/`. Iteration ≥$50k raw LL 0.6347 / ROI +22.77% [+2.73, +43.46] (first variant with positive ROI CI lower bound). Golden ≥$50k raw LL 0.6885 vs M1's 0.7006 (Δ -0.012). The bowler outcome-dist follow-up is open: investigate metadata-based bowler set (`is_pace` filter) instead of bottom-5 squad-order; revisit in M5.

### M3 — Player-level rolling form ❌ DROPPED — but UNFROZEN materialization landed (2026-05-10)
Mixed outcome. Full reference: `reports/m3_rolling_form_eval.md`.
- [x] Added 18 rolling-form features in `_rolling_form_features` helper: top-6 batting avg/SR_recent + diffs, all-11 bowlers avg/econ_recent + diffs, in-form/out-of-form indicators + diffs. Parquet at `data/xgb_match_data_v3_m3{,_unfrozen}/`.
- [x] Drop-one ablation: bowling-recent HURTS (same pattern as M2 bowler outcome-dist); batting-recent + form indicators individually neutral.
- [x] Stale-tracker hypothesis test: re-materialized in UNFROZEN mode. Confirmed material feature drift (mean |Δ| ~7 batting avg points on golden).
- [~] **M3 features do not add value even unfrozen** — career-aggregate features already capture the predictive variance. M3 features stay in materializer (`--drop-features` excludes them at training time) but are NOT in production.

**Phase outcome — UNFROZEN MATERIALIZATION ADOPTED as new production mode**:
- New baseline: `models/xgb_match_v3_m2_venue_only_unfrozen/`. Same features as M2 v.o. but trained on unfrozen chronological-walk parquet.
- **First variant to clear iteration ≥$50k LL gate AND ROI CI gate simultaneously**: LL 0.6279 + Platt vs market 0.6267; ROI +26.69% [+6.26, +48.65].
- Golden ≥$50k + Platt: LL 0.6849, ROI **+31.29% [+1.92, +59.11]**, win 61.2%. First variant with golden ROI CI cleanly excluding 0.
- 2026-04 walk-forward (IPL): ROI **+43.67% [+10.36, +76.65]**, win 68.6%.

**Materialization mode going forward**: unfrozen (no `--freeze-trackers-after` flag). Frozen kept for diagnostic reference. Bowler-unit identification remains broken; revisit in M5 with metadata-based `is_pace` filter.

### M4 — Within-tournament features ❌ DROPPED 2026-05-10
Full reference: `reports/m4_within_tournament_eval.md`.
- [x] Added 15 features in `_within_tournament_features`: date-windowed form (60d), competition-filtered form (365d, same tier), scheduling proxies (days_since, back_to_back). Parquet at `data/xgb_match_data_v3_m4_unfrozen/`.
- [x] TeamFormTracker extended with `get_last_days_win_rate`, `get_competition_win_rate`, `get_days_since_last` queries; records now carry competition_tier.
- [x] Drop-one ablation: scheduling group HURTS test LL (Δ -0.0095 when dropped); competition-filter neutral; date-window slightly helps standalone test but doesn't translate to iteration ROI.
- [~] All variants regress on iteration ≥$50k. M4 full Platt ROI CI now includes 0; M4 window-only and M4-no-scheduling don't recover M2 v.o.'s ROI lower bound.

**Why M4 failed** (three converging problems, full data in report):
1. Form features (`win_rate_last_60d_diff`, `competition_win_rate_diff`) are 0.68-0.79 correlated with M1's `win_rate_diff`. All three diffs have target r ≈ 0.155 (same signal in 3 features).
2. Scheduling features have ~zero train target correlation (r = -0.001 to -0.024) but XGBoost still uses them at importance 0.011-0.014 via interactions → fits noise.
3. Net: M4 predictions ~10% more confident, LESS accurate at the tail (|p−0.5|>0.20: M2 89.6% acc → M4 86.0% acc).

**Discipline added** (memory file `feedback_correlation_check_before_features.md`): all M5+ feature groups must correlation-check vs M1+M2 baseline before training; |r|>0.5 against existing requires demonstrating orthogonal target signal. This check would have caught both M3 and M4.

**Discipline added** (memory file `feedback_iteration_only_decisions.md`): iteration test is the only set used for M-phase landing decisions; golden is held out for audit. Golden only re-enters the decision process at M7 / production-launch.

**M4 features stay in materializer code** (`_within_tournament_features`, TeamFormTracker extensions, FEATURE_COLUMNS entries) but excluded from production via `--drop-features` substring filter.

### M5 — Player × opposition / player × venue affinity ❌ DROPPED 2026-05-10 (at correlation check, pre-training)
Full reference: `reports/m5_player_affinity_eval.md`.
- [x] Added 8 player × opposition features in `_player_vs_opposition_features` helper. h2h-matrix aggregation, shrunk to career with k_balls. Parquet at `data/xgb_match_data_v3_m5_unfrozen/`.
- [x] **Correlation check (the post-M4 discipline) FAILED for all 8 features**: 7 have |r|>0.5 with an M1 baseline feature AND target r ≤ baseline's; the 1 borderline (`sr_vs_opp_diff`) has target r essentially identical to baseline (0.99×).
- [x] Sanity check at k_balls=10 (lower shrinkage): target correlations unchanged (~0.10-0.15). The redundancy is structural — lineup aggregation collapses per-player matchup signal toward team career means.
- [~] Player × venue (schema-bump half): SKIPPED. Same structural failure mode expected (per-player venue stats aggregate toward team venue stats already encoded by `is_team1_home` + `venue_p4/p6/pw`).

**Key insight (durable)**: M3 (player rolling form), M4 (within-tournament form), M5 (player × opposition) all failed because lineup-aggregate features collapse to team-level career aggregates. **Aggregated-player features don't beat team-level career features at the match level.**

**M6 strategy shift**: prioritize features that are *match-level by nature*. Captain identity, pitch conditions, day-of-match flags, month×venue interactions are per-match (not lineup-aggregated), so they don't suffer the aggregation collapse.

**Discipline win**: the M4 ablation report committed to a pre-training correlation check; this was its first application. Caught structural redundancy in minutes instead of after a full training+ablation cycle.

### M6 — Conditions / captain ❌ DROPPED 2026-05-10
Full reference: `reports/m6_conditions_captain_eval.md`.
- [x] Added 3 date-derived condition features in `_match_conditions_features`: `month_of_year`, `day_of_week`, `is_dew_prone_month`. Match-level scalars (no lineup aggregation).
- [~] **Captain features SKIPPED**: cricsheet doesn't tag captains; "first in lineup" heuristic too noisy. Defer until reliable identification source exists.
- [~] **`is_day_match` SKIPPED**: cricsheet has no start-time field.
- [x] Pre-training correlation check passed redundancy (clean orthogonality vs baseline) but **failed (newly-added) target-correlation-floor check** — all 3 features have |target r| ≤ 0.011, well below the new 0.03 floor.
- [x] Trained anyway to confirm M4-style over-confidence pattern. Confirmed: M6 month-only's standalone test LL improved -0.007 BUT iteration Platt ROI dropped to +15.14% [-4.19, +37.22] (M2 v.o. was +26.69% [+6.26, +48.65]); tail accuracy regressed -3.2pp at |p-0.5|>0.15.

**Discipline upgrade**: dual-condition correlation check now required (redundancy AND target-floor). Memory file `feedback_correlation_check_before_features.md` updated.

**5 of 5 named v3 feature phases now DROPPED**. The match-level model at M2 v.o. (49 features) is at a local LL optimum given the pre-match signal available. Feature-engineering frontier exhausted; M7 should be ARCHITECTURE work.

### M7 — Architecture sweep ✅ LANDED 2026-05-10
Full reference: `reports/m7_architecture_eval.md`.
- [x] **B1 Hyperparameter resweep** — 81-config grid (md × lr × ss × cs). Winner: md=4 lr=0.05 ss=0.8 cs=0.9 (baseline was lr=0.10 cs=0.8). Trainer defaults updated.
- [x] **B5 Per-tier specialization** explored (IPL-only hybrid). Marginal +3-5pp ROI on aggregate driven by 22 IPL matches with wide CI [-8, +77]. Not landed (complexity not justified). Preserved at `models/xgb_match_v3_m7b_ipl_only/` for future re-evaluation when IPL test n grows.
- [~] **B2 Stacking** SKIPPED — M7.A delivered a clean win; stacking unlikely to add. Keep architecture simple.
- [~] **B6 LightGBM/CatBoost** SKIPPED — low priority, no clear gap remaining.

**Production model**: `models/xgb_match_v3_m7_production/` (raw probabilities; Platt over-corrects on this config). `predict_fixture.py` switched from v2_clean_unfrozen → v3_m7_production.

**Iteration metrics (the gate)**:
- iter ≥$50k LL: 0.6299 vs M2 v.o. 0.6348 (Δ -0.005, closer to market 0.6267)
- iter ≥$50k ROI: +21.90% [+2.28, +43.83] (CI overlap with baseline)
- iter ≥$100k LL: 0.5929 vs M2 v.o. 0.6006
- iter ≥$100k ROI: +26.39% [+0.57, +58.78] (**CI now excludes 0**; baseline was [-0.93, +57.10])
- **Close-slice ROI**: +33.27% [+4.36, +61.53] (M2 v.o. was +26.12% [-2.34, +52.30] — CI now excludes 0 on the historically weak slice)
- 2026-04 IPL walk-forward: +34.87% ROI [+2.04, +68.06], win 65.7%

### M8 — Sizing / operational ✅ LANDED 2026-05-10
Full reference: `reports/m8_sizing_rules_eval.md`.
- [x] **Edge threshold + fractional Kelly (E1) tested**: sweep over thresholds {0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15}. **Threshold 0 wins on aggregate**: only config where iter ≥$50k ROI CI cleanly excludes 0 (+0.91 lower bound). Counter-intuitive but confirmed: the M7 model's low-edge bets are calibrated and carry real signal — filtering them hurts every metric.
- [x] **Outlier per-bet cap (E2) tooling shipped**: `scripts/sim_eval/sizing_rules.py` supports per-bet Kelly cap. Quarter Kelly + 2% cap is the documented Kelly default. No data-driven optimization (small n).
- [~] **Final golden re-evaluation DEFERRED to production-launch time** (per the iteration-only-decisions discipline). Single use only, not for sizing-rule selection.

**Production sizing rule LANDED**: flat 1-unit at edge threshold 0.

**Slice-conditional finding** (documented, not landed): on mismatch fixtures (|top6 ELO diff| ≥ 15), threshold=10% gives ROI +44% [+1.15, +78.30], win 72%. On close fixtures (|diff| ≤ 5), threshold=0 wins. Inference-time complexity not justified at small n; revisit after C2 forward capture provides more mismatch samples.

**Kelly variants preserved as tooling** (not production default):
- Quarter Kelly + 2% per-bet cap: conservative; tiny per-bet returns
- Full Kelly + 2% cap: moderate; per-bet ROI +0.38%
- Full Kelly no cap: aggressive; per-bet ROI +4.06% but max DD 2.52 of bank

### Parallel / continuous (not gated by M-sequence)
- [ ] **Forward polymarket capture** (C2 in catalog) — already in TODO. Continues independently; first 30–60-match read on real edge.
- [ ] **The Hundred ingestion** (C1 in catalog) — deferred; revisit if M2–M5 don't close the gate.
- [ ] **Multi-bookmaker historical odds** (C3 in catalog) — deferred; precondition for re-opening market-residual modeling (C4).

## Root Cause Analysis (Dec 2024)
**Why both models predict ~50% for all matches:**
1. **Weak correlation**: Player stats have r=0.06-0.11 with ball outcomes (explains <1.5% of variance)
2. **Missing team-level features**: Model only sees current batter/bowler, not aggregate team strength
3. **Ball-level noise**: Even Kohli hits dots 30% of balls; individual outcomes are random
4. **Signal dilution**: Small differences per ball don't compound meaningfully to match outcomes

## Critical (P0)
- [x] **Re-evaluate all models after bug fix** — Done (March 2025). All 4 models lose money:
  - XGBoost v3: -43.9% flat ROI, 26.8% win rate (best of 4)
  - LSTM v1: -63.4% flat ROI, 12.2% win rate
  - Transformer v1: -63.4% flat ROI, 12.2% win rate
  - MLP v1: -63.8% flat ROI, 12.2% win rate
  - Root cause: no team-level signal — models can't distinguish strong vs weak teams
- [x] **Fix evaluation JSON output** — now saves full betting metrics (P&L, ROI, win rate, Kelly, actual_winner per match) with timestamped filenames
- [ ] **Expand test set to 200+ matches** — 44 matches is too small for statistical significance

## High Priority (P1)
- [x] **Empirical outcome distributions as features** — **LANDED 2026-04-23 as `xgb_v6_outcome_dist`** (schema v4 bump, 42 new features). Empirical-Bayes shrinkage toward global corpus prior π = (0.304, 0.411, 0.076, 0.108, 0.047, 0.054), k=30 for player cells / k=200 for venue. Five hierarchies live: batter overall, bowler overall, batter-vs-pace/spin, bowler-vs-LHB/RHB, venue. Phase-prior (global PP/mid/death) deferred — see follow-ups.
  - **Files**: `scripts/stats_sqlite_backend.py` (SCHEMA_VERSION=4 + 5 new getters with `_shrink` helper), `scripts/parsing_v2.py` (tracker state + `update_stats` + 5 tracker getters + emission via `parse_match_data_v2`'s new `prior=` kwarg), `scripts/build_stats_cache.py` (count-tuple INSERTs + π computed from final tracker state, written to `_meta.prior_p*`), `scripts/tracker_rehydration.py` + `scripts/materialize_features.py` (count seeding + prior passthrough), `scripts/sim_v1_2.py` (`_fill_outcome_dists` helper wired into all 5 model wrappers), `scripts/feature_registry.py` (5 new groups + `V6_GROUPS`), `experiments/configs/xgb_v6_outcome_dist.yaml`. Plan file: `~/.claude/plans/yes-let-s-go-with-cheerful-waffle.md`.
  - **Validation**: Phase A parity 9519/9519 matches bit-exact incl. 42 new cols (251s); 7/7 shrinkage unit tests (`scripts/tests/test_outcome_dist_shrinkage.py`); schema-v4 conservation + query-plan + getter checks (`scripts/tests/test_schema_v4_outcome_dist.py`). SQLite cache 46.5 MB → 56.8 MB (+10 MB for 6 count cols × 5 tables); rebuild 440s.
  - **Eval (261 polymarket × 100 sims, exp `xgb_v6_outcome_dist_20260423_182515_7d92bfc`)**: log loss **0.7518 → 0.7122 (−5.3%)**, Brier **0.2728 → 0.2562 (−6.1%)**, flat ROI +6.51% → −7.1%, frac Kelly ROI +1.27% → +0.7%. Calibration improved; flat betting regressed (same calibration-vs-ROI tension as the 2026-03 Platt experiments). Per-slice liquidity eval not yet run.
  - **Follow-ups (deferred)**: phase prior P(outcome|PP/mid/death); drop `batter_encoded`/`bowler_encoded` ablation; per-hierarchy k sweep {10, 30, 100, 300}; hierarchical shrinkage (vs-pace toward overall, not π); per-slice ≥$50k / ≥$100k eval to confirm log-loss win on sharp markets.
- [ ] **Expand test set to 200+ matches** — 44 matches is too small for statistical significance
- [ ] **Add evaluation baselines** — 50-50 random, market odds passthrough, home team always wins
- [x] **Add team-level features** — ELO + aggregated team stats (9 features). 19% log loss improvement.
- [x] **Add calibration layer** — Implemented ball-level (isotonic) + match-level (Platt LOOCV). Tested scientifically: calibration improves probability metrics but hurts betting ROI. Disabled by default; enable via `--calibrate` / `--ball-calibrate` flags when model resolution improves.
- [x] **ELO rating system** — Player-level ball-by-ball ELO with context-aware K-factor. See IMPROVEMENTS.md.
- [x] **Feature importance analysis** — `scripts/analyze_features.py` (gain/weight/cover, correlation, redundancy, group-level)
- [x] **Tier 1 features: venue profile + match context** — 11 new features across 2 groups:
  - `venue_profile`: boundary_pct, dot_pct, wicket_rate, powerplay_avg, death_avg, first_innings_avg, chase_win_pct
  - `match_context`: chose_to_bat, match_importance, is_international, competition_tier

## Medium Priority (P2) — Tier 2 Features
- [ ] **Batting order position context**:
  - `batting_position` (1-11), `remaining_batting_quality` (avg ELO of batters to come), `top_order_wickets`
- [ ] **Bowler workload / spell context**:
  - `bowler_spell_balls`, `bowler_overs_left` (max 4 per T20), `new_bowler` (first ball of spell)
- [ ] **Enhanced momentum features**:
  - `scoring_acceleration` (last 10 vs last 30 run rate), `wicket_cluster` (wickets in last 10 balls), `boundaries_in_last_over`
- [x] **Home advantage (explicit)** — landed 2026-05-09 in the match-level direct model: `is_team1_home`, `is_team2_home` features (3+ matches at venue in last 730d) shipped via `HomeVenueTracker` in `materialize_match_features.py`. Modest signal — P(win | team1 home) = 0.509 vs 0.473 in train. See "Match-level direct + sim ensemble" section below.
- [ ] **Include The Hundred** (`hnd_json.zip`) in `scripts/fetch_cricsheet.py` once the pipeline supports variable innings length. Current 120-ball hardcodes live in `parsing_v2.py`, `sim_v1_2.py` (`T20Rules`), and `transformer_v1.py` (`max_seq_len=120`).
- [x] **Phase-aware bowler selection** — LANDED 2026-05-12. `EmpiricalBowlerSelector` (now the default in `T20Rules()`/`SimulationEngine`) samples each over's bowler proportional to historical phase-usage (PP/mid/death) shares, EB-shrunk (`k=30`) toward the year's league marginal, as-of match year. Prior built by `scripts/build_bowler_phase_usage.py` → `models/bowler_phase_usage.json`. Validated vs random (n=60): winner-market LL parity (G1), top-batter no-regression (G3), ≥90% coverage (G5); G2 (top-bowler skill) misses strict target — the limiter is the ball-level wicket model, not the selector. `--bowler-selector random` recovers the baseline. See `docs/ARCHITECTURE.md` §6.15.
- [ ] Second-innings aggression adjustment based on required run rate
- [ ] Unknown player encoding (bottom 5-10%) for new/unseen players
- [ ] Bootstrap confidence intervals on evaluation metrics
- [ ] CLV (Closing Line Value) tracking
- [ ] Minimum edge threshold (3-5%) before placing a bet

## Pipeline bugs found during E-series experiments (2026-06-09)

- [x] **`parsing_v2.py:1255` — `innings_id` is `hash(json) % 100000`** —
  **FIXED 2026-07-16 (B2, interactive)**: `parse_match_data_v2` gained a
  `match_ref=` kwarg (cricsheet filename stem; legacy hash fallback) and
  `materialize_features.py` threads the loader's match_id. Parquet rebuilt
  (v6 config, feature hash unchanged): all 140 non-id columns byte-equal;
  363 collisions removed (360 train / 1 val / 2 test; 716 train innings
  groups were silently merged); 100% of suffixes now join losslessly to
  `data/t20s_json/<stem>.json`. Old parquets: `archive/xgb_data_v3_pre_b2/`.
- [ ] **`sim_v1_2.py` XGBoostModelV2 never sets `venue_encoded`** — `_feat_buf`
  defaults missing keys to 0, so every simulated ball is scored as venue
  code 0 while training saw real codes. Teacher-forced deltas real-vs-0 are
  second-order relative to the class-weight tilt (see below), but it's an
  out-of-distribution input on every sim ball. Fix: save a venue encoder at
  training time and wire `venue_encoder_path` into the XGB wrapper (LSTM/
  Transformer wrappers already do this). Re-baseline sims after fixing.
- [x] **v7 sampled raw `balanced`-class-weight probabilities** (E5, the big
  one): per-ball P(wicket) ≈ 2× actual, boundaries +0.05 absolute. Root
  cause of the prop framework's tail-event overshoot. Fixed via val-fit
  `VectorScalingCalibrator` (`models/xgb_v3/vector_scaling_calibrator_v1.pkl`),
  `--ball-calibrator vector` in `prop_backtest.py`. See
  `reports/e5_teacher_forced_bias.md`.

## Prop-odds source landscape (researched 2026-07-11) — CONCLUSION

Every candidate source investigated. **No hobbyist-priced, ToS-clean, US-legal
source of cricket PROP odds exists.** The wall is structural and identical
across sources: rich-prop books either block US IPs, ban scraping in ToS, or
gate real props behind ~$5k+/mo enterprise aggregators.

| Source | Cricket props? | Legit US access | Verdict |
|---|---|---|---|
| **Polymarket** | Team-level only (top-batter, most-sixes; NO player lines/totals) | Yes (public API) | **USE — forward capture, free, built 2026-07-11** |
| Betfair (hist + API) | Yes (innings runs, top batsman) | **No — US barred from account** | SKIP |
| UK books (bet365 etc.) | Richest (top bat/bowl, player lines, totals) | **No — geo-block US at page load + ToS bans scraping** | SKIP (both walls) |
| **DraftKings** | **Deep & US-legal** (batter runs/fours/sixes, top batter/bowler, wickets, highest score — IPL & MLC) | Odds viewable in ~25 states, **but no public API + ToS bans automated collection + bot detection** | No legit programmatic path |
| FanDuel | Thin (mostly moneyline) | Same ToS/geo posture as DK | SKIP |
| the-odds-api | **Match-winner only** (cricket props = US-sports only) | Yes, cheap ($0–30/mo) | USE for winner odds; not props |
| OpticOdds / OddsJam | Yes (licenses DK/FD/UK cricket props) | Yes | ~$5k/mo enterprise — revisit only if prop edge is proven |
| Sportradar | Yes | Yes | ~$10k+/mo — out of scope |

**Notable**: DraftKings is the one US-legal book with genuinely deep cricket
props (the exact families we model). No API and ToS forbids scraping, so no
compliant automated capture — but the odds are viewable by a logged-in user in
legal states, which keeps a *manual/licensed* path theoretically open if prop
edge ever justifies the effort. **Standing decision: forward-capture Polymarket
(free) for calibration data now; treat OpticOdds trial as the gate to any paid
prop feed, contingent on the ball-model showing prop edge first (B3/B4).**

## Data acquisition — prop odds & eval expansion (2026-07-11)
- [ ] **Rebuild the odds-capture scraper as a respectful capture daemon**
  (replaces dead `run_scraper_cron.sh` → `src/bet_scraper.py`, whose crontab
  entry still fires daily at 18:00 and fails — remove that entry when this
  lands). Requirements: **official APIs first, GET-only, conservative rate
  limits, honor ToS/robots** (design goal: never banned). Target = **Polymarket
  props only** (the source landscape above ruled out all others for US-legal
  prop capture). `capture_props.py` PROTOTYPE built + test-run 2026-07-11 in
  `~/Projects/polymarket-cricket` (Gamma discovery + CLOB backfill; writes
  `data/polymarket_props_<date>.json`; 5 team-level families, YES/NO dedup +
  placeholder-price guard). Remaining: review the prototype, then wire the
  daily + T-60min launchd job (standing machine config → supervised, not
  overnight). Forward capture compounds — Polymarket historical prop odds
  can't be bought retroactively, so start the daily job soon even though
  current prop liquidity is thin (~$0–13k, mostly minor leagues off-IPL-season).
  Backtest wiring against captured props = good overnight idea once data exists.
- [x] ~~**Betfair historical data purchase decision**~~ — **resolved SKIP
  2026-07-11**: Betfair (historical + exchange API) is the only source
  confirmed to carry real cricket prop markets (Innings Runs, Top Batsman),
  but both are gated on a Betfair account and **US residents are
  contractually barred from registering** (ToS); VPN/proxy registration
  violates ToS — not pursuing. Pricing unpublished without login anyway.
  Consequence: **forward capture is the only viable prop-odds path from the
  US — start it early, every day counts.** Capture targets, by priority:
  (1) **Polymarket cricket props** — confirmed to exist beyond match winner
  (e.g. highest-individual-run-total on IPL; polymarket.com/sports/cricket/
  props; ~35 live markets, coverage inconsistent) — extend the existing
  polymarket capture repo; (2) probe **The Odds API** free tier for cricket
  player-prop coverage before paying (reportedly thin outside US sports);
  (3) **Smarkets** only if US eligibility verifies (£150 API gate, Malta
  license — unconfirmed). OddsPortal/BetExplorer are match-winner only, not
  prop sources.
- [ ] **Conditions eval pool** (`data/conditions_eval/`, separate from golden):
  all stat-generator T20s since 2026-04-17 (IPL + Blast + MLC + T20I + …),
  per-competition sliced readout (LL/accuracy/calibration; betting metrics
  where odds exist). Diagnostic set — allowed to look, not a selection gate.
  Reuses `extract_golden_cricsheet.py` / `extract_blast_golden.py` patterns.

## Infrastructure & Refactoring
- [x] ~~**Eval performance pass (3 phases)**~~ — **LANDED 2026-05-08 → 2026-05-09**. (1) `_SQLiteBackend._norm_date` cache: −5.6 % wall on 5×100 (`85d67bf`). (2) `StatsProviderCache` extended to memoize 12 per-player getters: cumulative −17.9 % wall, `_fill_outcome_dists` −80 % cumtime (`135514a`). (3) Pickle round-trip fixed via explicit `__getstate__`/`__setstate__` (`e4b97cc`); 4-process disjoint-shard parallel eval gives 2.33× throughput at 1.6 GB combined RSS. Bit-identical numerics verified at every stage. **End-to-end full 261×100 eval: 41.9 min → 16.6 min = 2.52× speedup**, with avg LL within 0.0005 of serial baseline. See IMPROVEMENTS.md §"Performance Pass".
- [x] ~~**Delete legacy v2 stats cache** (`models/cache_chunks/`, 7.4GB)~~ — done in the 2026-04-26 cleanup pass alongside `models/cache_chunks_v3_old/` (8.9 GB) and 9 stray `cache_chunk_*.pkl` chunks at `models/` root (2.1 GB). Repo went 30 GB → 11 GB; see `archive/README.md` for what was archived vs deleted.
- [x] ~~Rewrite v3 stats cache as per-entity timelines~~ — **superseded by SQLite migration (Phase 1+2, 2026-04-19)**. 11 GB chunks → 39.7 MB SQLite (276× smaller) with same public API. `StatsProvider` auto-detects `models/player_stats_cache_v3.sqlite`. See `docs/SQLITE_MIGRATION_PROFILE.md`.
- [x] ~~**Phase 5: rewrite `parsing_v2.py` to emit SQLite directly, delete chunks format**~~ — **LANDED 2026-04-22** as part of Phase B. `build_stats_cache.py` writes SQLite schema v3 directly from JSONs; `build_stats_sqlite.py` deleted; `_ChunkedBackend` removed; `models/cache_chunks_v3/` reclaimed (12 GB); `models/player_stats_cache_v3_metadata.pkl` removed. See IMPROVEMENTS.md §"Parsing Pipeline Split".
- [x] ~~**Split `parsing_v2.py` into cache-builder + feature-materializer**~~ — **LANDED 2026-04-22** as Phase B Option E. `scripts/build_stats_cache.py` (JSON → SQLite) + `scripts/materialize_features.py` (SQLite + JSON → parquet, per-date batching). Phase A harness passes 63/63 on all 9519 matches; eval parity bit-identical flat betting metrics on polymarket_test.
- [ ] **Promote `_SQLiteBackend` private accessors to public API** — `scripts/tracker_rehydration.py` currently reaches into `provider._backend._get_raw_batting` / `_get_raw_bowling` / `_get_raw_h2h` / `_venue_row` / `_player_id_map` / `_resolve_date_id` — underscore-prefixed private methods on the backend. This breaks the `StatsProvider` facade narrowing that Phase B landed (861 → 264 lines). Two options:
  - Strip the underscores and document the 6 accessors as public API on `_SQLiteBackend` (then rename that class to `SQLiteBackend`).
  - Add a batched `get_rehydration_snapshot(as_of_date, player_ids, venues) -> dict` method that returns all the raw rows in one call, and delete the cross-module private access.
  Follow-up to Phase B; not urgent. Noted in IMPROVEMENTS.md §"Parsing Pipeline Split → Deferred follow-ups".
- [ ] **Incremental cache refresh (`--since`) for `build_stats_cache.py`** — today the cache-builder is all-or-nothing (`out_path.unlink()` before every rebuild, ~6 min on full corpus). A real incremental path needs: checkpoint last processed date in `_meta`, reopen trackers from that snapshot on startup, append new rows instead of rebuilding from zero. Worth its own plan; small data corpora make this low urgency today.
- [ ] **Deduplicate feature-assembly blocks in `sim_v1_2.py`** — 4 near-identical blocks at lines ~601, ~1153, ~1596, ~2017 all call the same `stats_provider` methods and stitch the same feature dict. Drift risk when adding/modifying features. Extract one `build_ball_features(state, striker, bowler, stats_provider)` helper and have all model wrappers (XGBoostModelV2, LSTMModelV1, TransformerModelV1, MLPModelV1) call it.

## Low Priority (P3) — Tier 3 Features (External Data)
- [x] ~~**Weather / conditions data**~~ — **tested and closed 2026-07-11** (A6:
  no winner-market signal at match level; A12: dew-conditional ball calibrator
  also null). Open-meteo cache kept at `data/external/weather/`. See
  `research/reports/auto/A6.md`, `A12.md`.
- [ ] **Ground dimensions** (boundary distances — affects 4/6 rates)
- [ ] Ensemble stacking: 3-7 diverse models + logistic regression meta-learner
- [ ] Add time decay to features
- [ ] Consider regression model (predict E[runs]) instead of classification
- [x] **Match-level model: predict P(team1 wins) directly from lineups** — **LANDED 2026-05-09**. XGBoost binary classifier on ~47 match-level features dominates v7 sim on winner-market LL across every liquidity slice. See "Match-level direct + sim ensemble" section below.

## Research Notes
See [IMPROVEMENTS.md](IMPROVEMENTS.md) for detailed research findings:
- Calibration-optimized models generate 69.86% higher returns than accuracy-optimized (Walsh & Joshi 2024)
- Realistic market-beating edge is 1-3% ROI
- GBMs still outperform deep learning on tabular data
- Don't chase ball accuracy >60%, don't optimize for small test set
