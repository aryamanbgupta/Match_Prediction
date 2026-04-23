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
3. [ ] **Rebuild features + stats cache**: `uv run python scripts/parsing_v2.py` (~10-15 min, destructive — regenerates `data/xgb_data_v3/` and `models/cache_chunks_v3/`).
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
- [ ] All 261 matches — maximum n, noisy markets
- [ ] ≥$50K volume subset (170) — honest liquidity gate
- [ ] ≥$100K volume subset (110) — tighter, market is sharp
- [ ] Bootstrap 95% CIs on ROI (≥1,000 resamples). Point estimates without CIs at n≤300 are meaningless — current flat-ROI CI spans ±20 pp.

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
- [ ] Add `--min-volume` flag to `scripts/sim_eval/run_sim_eval.py` so liquidity-sliced runs don't need the ad-hoc filter script from `docs/POLYMARKET_INTEGRATION.md` lines 206–212.

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
- [ ] **Empirical outcome distributions as features** — Replace lossy summary stats (avg/SR) with full 6-class outcome distributions P(0,1,2,4,6,W) for each context. This is multi-class target encoding — directly aligned with the prediction target. Current summary stats have r=0.06-0.11 with ball outcomes; direct historical rates should correlate much more strongly. Implementation: expand `PlayerStatsTracker` to track outcome counts per class instead of just `{runs, balls, dismissals}`. Hierarchy of distributions (each adds 6 features):
  - **Batter overall**: P(0,1,2,4,6,W | this batter) — captures player shape (power hitter vs accumulator), not just mean
  - **Bowler overall**: P(0,1,2,4,6,W | this bowler) — same idea, from bowler's perspective
  - **Batter vs pace/spin**: P(0,1,2,4,6,W | batter, bowler type) — type-specific distributions (min 30 balls, fallback to overall)
  - **Bowler vs LHB/RHB**: P(0,1,2,4,6,W | bowler, batter hand) — handedness-specific (min 30 balls, fallback to overall)
  - **Venue**: P(0,1,2,4,6,W | venue) — venue-specific outcome rates
  - **Phase (global)**: P(0,1,2,4,6,W | powerplay/middle/death) — phase-specific priors with massive sample sizes
  - ~36-48 new features total. May allow dropping `batter_encoded`/`bowler_encoded` (XGBoost can't learn from label-encoded IDs anyway).
  - Temporal integrity already handled by stats cache. Not target leakage — uses only pre-match historical data.
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
- [ ] **Delete legacy v2 stats cache** (`models/cache_chunks/`, 7.4GB) — `StatsProvider` defaults to v3; no active code path loads v2 by default. Confirm `--model-version v2` usage is dead, then `rm -rf`. Instant -7.4GB disk.
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
