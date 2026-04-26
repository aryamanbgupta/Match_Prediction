# TODO

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
- [ ] **Match-level calibration layer** — fit isotonic (primary) + Platt (sanity) on the 255 outcome-matches, LOOCV, predict-from-team1-prob. Target: post-calibration LL < market LL (0.6267) or at minimum < coinflip (0.6931). Expected: ~0.68–0.70 from calibration alone (the 74%→51%, 16%→40% bins are symmetric over-dispersion, mechanically fixable). Must land before any feature work — otherwise every "did X help?" is contaminated by miscalibration.

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
- [ ] **Home advantage (explicit)**:
  - `is_home_team` (3+ matches at venue in last 2 years), `team_venue_win_pct`
- [ ] **Include The Hundred** (`hnd_json.zip`) in `scripts/fetch_cricsheet.py` once the pipeline supports variable innings length. Current 120-ball hardcodes live in `parsing_v2.py`, `sim_v1_2.py` (`T20Rules`), and `transformer_v1.py` (`max_seq_len=120`).
- [ ] Phase-aware bowler selection (pace in powerplay/death, spin in middle overs)
- [ ] Second-innings aggression adjustment based on required run rate
- [ ] Unknown player encoding (bottom 5-10%) for new/unseen players
- [ ] Bootstrap confidence intervals on evaluation metrics
- [ ] CLV (Closing Line Value) tracking
- [ ] Minimum edge threshold (3-5%) before placing a bet

## Infrastructure & Refactoring
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
- [ ] **Weather / conditions data** (temperature, humidity, dew, wind via weather API)
- [ ] **Ground dimensions** (boundary distances — affects 4/6 rates)
- [ ] Ensemble stacking: 3-7 diverse models + logistic regression meta-learner
- [ ] Add time decay to features
- [ ] Consider regression model (predict E[runs]) instead of classification
- [ ] Match-level model: predict P(team1 wins) directly from lineups

## Research Notes
See [IMPROVEMENTS.md](IMPROVEMENTS.md) for detailed research findings:
- Calibration-optimized models generate 69.86% higher returns than accuracy-optimized (Walsh & Joshi 2024)
- Realistic market-beating edge is 1-3% ROI
- GBMs still outperform deep learning on tabular data
- Don't chase ball accuracy >60%, don't optimize for small test set
