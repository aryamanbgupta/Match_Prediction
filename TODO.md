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
