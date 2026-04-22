# CricML Improvements & Research Findings

**Last Updated**: March 2025

---

## Honest Assessment

### What's Working
- **Ball-level simulation approach**: Validated by Kuo 2021 (beat Bet365 using similar methods). 4M+ training examples vs 15K matches.
- **Temporal integrity**: Stats snapshots prevent data leakage. Correct approach.
- **Chunked caching**: 14x memory reduction (7.6GB → 550MB), lazy loading works well.
- **Feature engineering**: 46+ features covering player stats, matchups, metadata, momentum.

### What's Broken
- **Critical bug (FIXED)**: XGBoost `class_to_outcome` mapping was 8-class instead of 6-class. Class 4 (six) mapped to 'four', class 5 (wicket) mapped to 'six'. **Wickets never occurred in XGBoost simulation.** All previous evaluation results (29.4% edge, 42/44 profitable bets) were artifacts. Fixed in `sim_v1_2.py` for both `XGBoostModelV2` and `XGBoostModel`.
- **Evaluation JSON was incomplete (FIXED)**: `run_sim_eval.py` only saved log loss, brier score, and a misleading `profitable_bets` count (always = n_matches by construction). Real betting metrics (P&L, ROI, win rate, Kelly) were computed but never persisted. Now saves all metrics with timestamped filenames.
- **No team-level signal**: Model can't distinguish India from Scotland. Only sees current batter/bowler, not aggregate team strength.
- **Miscalibrated evaluation**: 44-match test set is too small for meaningful confidence intervals.

### Post-Bug-Fix Evaluation Results (March 2025)
All 4 models evaluated on 44 T20 World Cup 2024 matches with 100 simulations each:

| Model | Avg Log Loss | Flat ROI | Win Rate | Frac Kelly ROI |
|-------|-------------|----------|----------|----------------|
| XGBoost v3 | 0.8754 | -43.9% | 26.8% | -5.7% |
| LSTM v1 | 0.7229 | -63.4% | 12.2% | -6.3% |
| Transformer v1 | 0.6880 | -63.4% | 12.2% | -5.7% |
| MLP v1 | 0.6997 | -63.8% | 12.2% | -6.1% |

**Conclusion**: All models lose money. The #1 blocker is missing team-level features — models predict ~50/50 for every match regardless of team strength, then always bet on the underdog (perceived "edge" is really just disagreement with market).

### Post-Team-Strength Evaluation Results (March 2026)
XGBoost v4 with player-level ELO + aggregated team strength (9 new features, 72 total):

| Model | Avg Log Loss | Avg Brier | Flat ROI | Win Rate | Frac Kelly ROI |
|-------|-------------|-----------|----------|----------|----------------|
| XGBoost v3 (baseline) | 0.8754 | 0.3168 | -43.9% | 26.8% | -5.7% |
| **XGBoost v4 (team strength)** | **0.7100** | **0.2554** | -44.8% | 24.4% | -5.5% |
| Change | **-18.9%** | **-19.4%** | ~same | ~same | ~same |

**Key findings**:
- **Log Loss improved 19%** — model now produces more differentiated win probabilities
- **Brier Score improved 19%** — probability calibration significantly better
- ROI unchanged — better probabilities haven't translated to profitable bets yet (needs calibration layer)
- 3 team features in XGBoost top 15: `striker_elo` (#9), `team_batting_sr` (#10), `batting_team_elo` (#14)

**ELO fixes applied** (vs initial broken implementation that had zero impact):
1. Fixed outcome scaling: dot ball was mapping to 0.0 (same as wicket), now maps to 0.4
2. Gender filter: women's cricket excluded from ELO (different competition pool)
3. Context-aware K-factor: premium matches (IPL/WC/full-member T20I) K=4.0, standard leagues K=2.0, associate bilaterals K=1.0

**Experiment**: `xgb_v4_team_strength_20260321_110247_40befa4`

---

## Research Findings

### Calibration > Accuracy
- Walsh & Joshi 2024: Calibration-optimized models generate **69.86% higher betting returns** than accuracy-optimized ones.
- Target ECE (Expected Calibration Error) < 0.015.
- Methods: Isotonic regression, temperature scaling, Platt scaling.

### Realistic Edge Sizes
- Market-beating edge is typically **1-3% ROI** (not 29%).
- Kuo 2021 achieved ~2-3% edge over Bet365 using ball-by-ball simulation.
- Any model showing >5% edge on a small sample is likely overfitting or has a bug.

### Decorrelation from Market
- Hubacek 2019: Profitability comes from **decorrelation** with market predictions, not just accuracy.
- Models that agree with the market on most games but disagree on a few high-confidence spots are most profitable.

### GBMs vs Deep Learning on Tabular Data
- GBMs (XGBoost, LightGBM) still outperform deep learning on structured/tabular data.
- Neural models (LSTM ~44%, Transformer ~44%) significantly underperform XGBoost (~55-60%) on ball prediction.
- Deep learning adds value primarily through sequence modeling and ensemble diversity.

### Ensemble Stacking
- Best approach: 3-7 diverse base models + logistic regression meta-learner.
- Diversity matters more than individual model accuracy.
- Candidate models: XGBoost, LightGBM, LSTM, Transformer (different feature subsets).

### Evaluation Best Practices
- **CLV (Closing Line Value)** is the gold standard metric for betting models.
- Minimum 200+ matches for statistically significant evaluation.
- Bootstrap confidence intervals on all metrics.
- Multiple baselines: 50-50, market odds, home team advantage.
- Minimum edge threshold of 3-5% before placing a bet (Kelly sizing).

---

## Feature Improvements Roadmap

### Team-Level Features (IMPLEMENTED — Phase 1)
Player-level ELO + aggregated team stats implemented in March 2026:
- `striker_elo` / `bowler_elo_rating` — individual player ELO ratings (ball-by-ball updates)
- `batting_team_elo` / `bowling_team_elo` — sum of 11 players' ELOs
- `elo_diff` — batting team ELO minus bowling team ELO
- `team_batting_avg` / `team_batting_sr` — aggregate of lineup batting stats
- `team_bowling_avg` / `team_bowling_econ` — aggregate of lineup bowling stats

See `docs/research/team_level_features.md` for full options analysis.

### Venue Profile + Match Context Features (IMPLEMENTED — Phase 2)
Tier 1 features implemented in March 2026. 11 new features across 2 groups:

**Venue Profile** (7 features — historical venue behavior with temporal integrity):
- `venue_boundary_pct` — % of runs from 4s+6s at this venue
- `venue_dot_pct` — % of balls that are dots
- `venue_wicket_rate` — wickets per ball
- `venue_powerplay_avg` — average powerplay score (overs 0-5)
- `venue_death_avg` — average death score (overs 16-19)
- `venue_first_innings_avg` — average 1st innings total
- `venue_chase_win_pct` — % of matches won by chasing team

**Match Context** (4 features):
- `chose_to_bat` — binary: toss winner chose to bat
- `match_importance` — ordinal 1-4 (associate bilateral → ICC World Cup)
- `is_international` — binary: international vs franchise/club
- `competition_tier` — ordinal 1-4 (associate → ICC/premium league)

**Feature importance analysis** (`scripts/analyze_features.py`):
- XGBoost gain/weight/cover importances
- Spearman correlation with target
- Feature-to-feature redundancy detection (|r| > 0.8)
- Per-group aggregated importance

**Experiment Results** (March 2026):

| Model | Log Loss | Brier | Flat ROI | Frac Kelly ROI | Win Rate |
|-------|----------|-------|----------|----------------|----------|
| v4 baseline (no venue) | **0.710** | **0.255** | **-44.8%** | **-5.5%** | **24.4%** |
| v5 (all 11 venue+context) | 0.923 | 0.337 | -48.9% | -6.2% | 24.4% |
| v5b (pruned: 5 features) | 0.965 | 0.354 | -68.1% | -9.0% | 14.6% |

**Conclusion: Venue features hurt all metrics.** Log loss +30%, Brier +32% for v5 vs v4.

**Root cause analysis:**
- 7 venue features are highly correlated with each other (r > 0.90) and with existing `venue_avg_score`
- Model became more overconfident (avg signed edge: -20.5% → -25.9%) without improving accuracy
- Venue features encode "high-scoring ground?" 7 different ways — redundant with `venue_avg_score`
- `match_importance` and `competition_tier` are correlated (r=0.88) — redundant
- T20 World Cup test venues have limited historical data → venue features don't generalize

**Decision: Venue features disabled.** v4 (team strength only) remains the best model. The `venue_profile` and `match_context` feature groups are kept in the code but excluded from active configs.

Experiments: `xgb_v5_venue_context_20260322_143913`, `xgb_v5b_venue_pruned_20260322_182632`

### Empirical Outcome Distributions (PLANNED — Next Priority)
Replace lossy summary stats (avg, SR, econ) with full 6-class outcome distributions as features. This is **multi-class target encoding** — the most direct possible signal for the prediction target.

**Why this is high-impact**:
- Current summary stats (avg=35, SR=140) compress a player into 2 numbers, destroying the distribution shape. Two batters with identical avg/SR can have completely different outcome profiles (power hitter vs accumulator).
- XGBoost **cannot learn** outcome distributions from label-encoded player IDs — trees can only split on "id > N", which groups arbitrary players together. Pre-computed distributions give the model exactly the signal it structurally can't extract.
- Direct target alignment: the model predicts P(outcome class), and these features provide historical P(outcome class | context). The learning task becomes "blend these priors" instead of "reconstruct distributions from scratch."
- Current player stats correlate r=0.06-0.11 with ball outcomes (<1.5% variance explained). Direct historical outcome rates (e.g., `batter_dot_pct`) should correlate much more strongly with the corresponding binary outcome.

**Proposed feature hierarchy** (each level adds 6 features — one per outcome class):

| Level | Features | ~Sample size | Sparsity risk |
|-------|----------|-------------|---------------|
| Batter overall | P(0,1,2,4,6,W \| batter) | 100+ balls | Low |
| Bowler overall | P(0,1,2,4,6,W \| bowler) | 100+ balls | Low |
| Batter vs pace/spin | P(0,1,2,4,6,W \| batter, type) | 50+ balls | Moderate |
| Bowler vs LHB/RHB | P(0,1,2,4,6,W \| bowler, hand) | 50+ balls | Moderate |
| Venue | P(0,1,2,4,6,W \| venue) | 100+ matches | Low |
| Phase (global) | P(0,1,2,4,6,W \| PP/mid/death) | Millions | None |

Total: ~36-48 new features. Manageable for XGBoost.

**Implementation notes**:
- Expand `PlayerStatsTracker` to track `outcome_counts: {0: N, 1: N, 2: N, 4: N, 6: N, W: N}` instead of just `{runs, balls, dismissals}`
- For sparse cross-cuts (< 30 balls), fall back to the broader distribution or apply shrinkage toward global prior
- Temporal integrity already handled by stats cache — no target leakage
- May allow dropping `batter_encoded` / `bowler_encoded` since distributions ARE the player representation
- Keep existing avg/SR features initially — let feature importance sort out redundancy

### Tier 2 Features (PLANNED — Future)
- Batting order position context (batting_position, remaining_batting_quality, top_order_wickets)
- Bowler workload/spell context (spell_balls, overs_left, new_bowler)
- Enhanced momentum (scoring_acceleration, wicket_cluster, boundaries_in_last_over)
- Explicit home advantage (is_home_team, team_venue_win_pct)

### Tier 3 Features (PLANNED — Requires External Data)
- Weather/conditions (temperature, humidity, dew, wind — needs weather API)
- Ground dimensions (boundary distances — needs manual curation)

### ELO Rating System — Implemented + Future Improvements
**Current implementation**:
- Ball-by-ball ELO with linear outcome scaling: wicket=0.0, dot=0.4, 1=0.5, 2=0.6, 4=0.8, 6=1.0
- Context-aware K-factor: premium matches (IPL/WC/full-member T20I) K=4.0, standard leagues K=2.0, associate bilaterals K=1.0
- Gender-separated: women's cricket excluded from ELO pool (no cross-gender bridge players)
- Follows players across leagues (implicit cross-league calibration)
- Team strength = sum of 11 player ELOs
- ELO range after fixes: batting ±70 points, bowling ±45 points per player

**Bugs fixed**:
- Outcome scaling: dot balls were mapping to 0.0 (same as wickets), causing systematic batting ELO deflation. Fixed to 0.4 (mild negative, not catastrophic).
- Flat K=1.0: all matches weighted equally regardless of quality. Fixed with 3-tier K-factor.
- Gender contamination: women's associate cricket bowlers dominated top bowling ELOs. Fixed by filtering women's matches from ELO computation.
- Feature list: team_strength features were in parquet but not in xgboost_v2.py's hardcoded feature list. Fixed.

**Potential improvements** (ordered by expected impact):
- **Phase-specific outcome scaling**: A dot in death overs is worse than a dot in the powerplay. Use game-phase-aware baseline (e.g., expected runs by over number) instead of flat 0.4 for dots. This would make ELO updates more contextually accurate.
- **Phase-specific K-factors**: Higher K in death overs (K=6-8) where outcomes are more decisive, lower K in middle overs (K=2-3) where outcomes are more routine. Stacks with match-importance K.
- **ELO decay for inactive players**: Rating deviation grows when a player hasn't played recently (Glicko-2 style). Inactive players regress toward global mean. Prevents stale ratings for retired/rested players.
- **Separate pace/spin bowling ELO**: A bowler may be elite with pace but mediocre with spin variations. Track separate ELOs by bowling type. Would need changes to feature_registry and sim_v1_2.py.
- **Glicko-2 adaptation**: Track uncertainty (sigma) alongside rating (mu). Higher uncertainty → larger updates. Naturally handles new players and returning players.
- **Wicket type weighting**: Bowled/LBW is more "bowler's doing" than a run-out or stumping. Could weight wicket types differently in the ELO update.
- **Bridge-player league normalization** (for aggregated stats): Compute explicit league difficulty factors from players who compete in multiple leagues. Normalize raw stats (batting avg, SR) before aggregating into team strength features. Would improve `team_batting_avg` / `team_bowling_econ` features but not ELO itself.
- **Time decay weighting**: Recent balls weighted more heavily in ELO calculation, with exponential decay. More recent form is more predictive.

### Phase-Aware Bowler Selection
- Current simulation selects bowlers randomly.
- Real teams use pace in powerplay/death, spin in middle overs.
- Implement phase-based bowling probability distributions.

### Second-Innings Aggression Adjustment
- Chase targets should affect batting aggression.
- Required run rate should influence outcome probabilities.
- Currently chase features exist but may not be sufficient.

---

## Calibration System (IMPLEMENTED & TESTED — March 2026)

### What Was Built
Full calibration pipeline with two independent techniques:
- **Ball-level calibration**: Per-class isotonic regression on XGBoost's 6-class outputs (dot, 1, 2, 4, 6, wicket). Fitted on 226K validation balls. Applied inside `predict_next_ball()` before simulation sampling.
- **Match-level calibration**: Platt scaling (2-param logistic) with LOOCV on match win probabilities after Monte Carlo simulation. Applied before [5%, 95%] probability clipping.

### Implementation
| File | What |
|------|------|
| `scripts/calibration.py` | PlattCalibrator, IsotonicCalibrator, BallLevelCalibrator, BallLevelCalibrationDiagnostics, ECE/reliability functions |
| `scripts/sim_v1_2.py` | `XGBoostModelV2` accepts optional `ball_calibrator` param |
| `scripts/sim_eval/match_evaluator.py` | `evaluate_all_with_calibration()` method with LOOCV two-pass approach |
| `scripts/sim_eval/run_sim_eval.py` | CLI flags: `--calibrate`, `--calibration-method`, `--ball-calibrate`, `--ball-diagnostics` |
| `scripts/run_experiment.py` | Calibration flags passthrough from YAML configs |
| `experiments/configs/xgb_v4_*_calibration.yaml` | 4 experiment configs for A/B testing |

All calibration is **off by default**. Enable via CLI flags only.

### Ball-Level Diagnostics (Pre-Calibration)
On 226,326 validation balls:
| Class | ECE | Quality |
|-------|-----|---------|
| dot | 0.089 | poor |
| one | 0.133 | poor |
| two | 0.054 | fair |
| four | 0.062 | poor |
| six | 0.022 | good |
| wicket | 0.063 | poor |
| **Overall (weighted)** | **0.097** | **poor** |

### Experiment Results (4-way A/B test, 100 sims, 44 matches)
| Experiment | Log Loss | Brier | Flat ROI | Frac Kelly ROI | Win Rate |
|------------|----------|-------|----------|----------------|----------|
| A: No calibration (baseline) | 0.710 | 0.255 | -44.8% | -5.5% | 24.4% |
| B: Ball-level only | 0.791 | 0.294 | -47.4% | -6.9% | 24.4% |
| C: Match-level Platt | **0.667** | **0.238** | -64.9% | -5.4% | 12.2% |
| D: Both | 0.676 | 0.242 | -60.3% | -5.9% | 14.6% |

Calibration comparison (C & D only):
| | ECE before | ECE after | Log Loss before→after | Brier before→after |
|--|-----------|-----------|----------------------|-------------------|
| C: Match-only | 0.186 | 0.182 | 0.710 → 0.718 | 0.255 → 0.262 |
| D: Both | 0.228 | 0.246 | 0.791 → 0.732 | 0.294 → 0.268 |

### Conclusions
1. **Ball-level calibration hurts** — isotonic correction at ball level distorted probability distributions, worsening all metrics (log loss +11%, Brier +15%).
2. **Match-level Platt improves calibration metrics** (log loss -6%, Brier -7%) but **destroys betting performance** (flat ROI -44.8% → -64.9%, win rate halved). Platt compresses predictions toward 50%, reducing edge on every bet.
3. **Calibration cannot create signal** — it can only fix the mapping from predicted→actual probabilities. When the model predicts Ireland > India (because ELO ratings are wrong), calibration can't fix that — it just makes the wrong prediction closer to 50%.
4. **Root cause confirmed**: The bottleneck is discriminative power (resolution), not calibration. ELO features need refinement before calibration adds value.

### Decision: Calibration Disabled by Default
All calibration code is kept but disabled. Enable later when the model has better resolution:
```bash
# Ball diagnostics (read-only)
uv run python scripts/sim_eval/run_sim_eval.py --ball-diagnostics ...
# Ball-level calibration
uv run python scripts/sim_eval/run_sim_eval.py --ball-calibrate ...
# Match-level Platt calibration
uv run python scripts/sim_eval/run_sim_eval.py --calibrate --calibration-method platt ...
# Both
uv run python scripts/sim_eval/run_sim_eval.py --ball-calibrate --calibrate ...
```

Experiments: `xgb_v4_no_calibration_20260321_215322`, `xgb_v4_ball_calibration_20260322_004600`, `xgb_v4_match_calibration_20260322_034532`, `xgb_v4_both_calibration_20260322_063841`

---

## Evaluation Improvements

1. **Expand test set**: 200+ matches minimum (currently 44).
2. **Add baselines**: 50-50 random, market odds passthrough, home team always wins.
3. **Bootstrap confidence intervals**: 1000 resamples on all metrics.
4. **CLV tracking**: Compare model predictions to closing line (not opening odds).
5. **Minimum edge threshold**: Only "bet" when model edge > 3-5%.

---

## Parsing Pipeline Split (PLANNED — bundle with Phase 5 SQLite migration)

**Planned 2026-04-19. Phase A lands this week; Phase B after the Phase-5 gate (≥ 2026-04-26).**

### Goal
Replace the monolithic `scripts/parsing_v2.py` (1409 lines, ~15 min per run) with two scripts whose boundaries map to the two real data artifacts:

1. `scripts/build_stats_cache.py` — chronological tracker walk → `models/player_stats_cache_v3.sqlite`
2. `scripts/materialize_features.py` — stateless per-match feature emission → `data/xgb_data_v3/*.parquet`

This subsumes Phase 5 of the SQLite migration (`TODO.md:127-134`) and the "split parsing" bullet (`TODO.md:136-139`). The two items were scoped separately but edit the same code; bundling halves the migration cost.

### Motivation
The parser today does three things in one pass, only one of which is truly stateful:

1. **Stateful (chronological)**: tracker updates (`PlayerStatsTracker`, `VenueStatsTracker`, `PlayerEloTracker`). SQLite is the serialized output of this.
2. **Snapshot emission**: frozen view per match-date → pickle chunks today, SQLite rows after Phase 5.
3. **Feature materialization** (`parse_match_data_v2`, `parsing_v2.py:744-1107`): per-ball feature row. Stateful only within an innings (`InningsFeatureCalculator` for momentum / partnership / per-batter balls faced). **Across matches it's a pure function of (ball context, cache snapshot at match_date, player metadata).**

Point 3 is the key insight: once the SQLite cache exists, feature materialization doesn't need chronological order, which unlocks incremental refresh, parallelization, and a per-match materialization cache.

### Proposed architecture

#### `build_stats_cache.py`
- Input: `data/t20s_json/`, `--gender-filter` (default `male`)
- Sorted JSON iteration → tracker updates → delta-compressed rows streamed into SQLite
- Reuses the streaming pattern from `scripts/build_stats_sqlite.py:58-328` but sources from JSONs instead of pickle chunks
- No parquet output, no feature materialization
- Output: `models/player_stats_cache_v3.sqlite` with `_meta.schema_version` and `_meta.source_json_mtime_max`

#### `materialize_features.py`
- Input: SQLite DB (read-only), `data/t20s_json/`, `--splits` config, `--feature-groups` / `--exclude` / `--include-extra`
- Per-match (any order): walks deliveries, runs `InningsFeatureCalculator` (per-innings state), queries SQLite for as-of-`match_date` historicals, joins `PlayerMetadataProvider`
- Writes split parquets + `.feature_hash`
- Output: `data/xgb_data_v3/{train,val,test,golden_test}.parquet`

### YAML config changes
Today `xgb_v3_baseline.yaml` has no parsing controls; splits are hardcoded at `parsing_v2.py:1181-1186`. After the split:

```yaml
data:
  version: "v3"
  splits:
    train_end: "2024-12-31"
    val_end: "2025-06-30"
    test_end: "2026-04-16"
    golden_start: "2026-04-17"
  gender_filter: "male"
  test_dir: "data/polymarket_test"
  odds_file: "betting_odds_polymarket.json"
```

`run_experiment.py`'s smart-cache check (lines 54-75) extends to two artifacts:
- **Cache valid** if `SQLite source_json_mtime_max ≥ latest JSON mtime`
- **Parquet valid** if `feature_hash matches AND cache mtime ≤ parquet mtime AND splits match`

### Phased rollout

**Phase A — Soft split** (bridge, low-risk, ~half day) — **pre-2026-04-26**
- Add `--materialize-only` flag to `parsing_v2.py`. Skips tracker updates + snapshot emission; reuses existing SQLite; regenerates parquet only.
- Validates the "cache stable, materialization stateless" assumption under the safe monolith before any extraction.
- Lets the split bullet move forward without touching the Phase-5 gate.
- **Exit gate**: `--materialize-only` parquet is bit-identical to a full `parsing_v2.py` run on all 4 splits.

**Phase B — Structural split** (bundled with Phase 5, ~2 days) — **after 2026-04-26**
- Extract `build_stats_cache.py` from `parsing_v2.py` (tracker half).
- Extract `materialize_features.py` from `parsing_v2.py` (feature half).
- Delete `scripts/parsing_v2.py`, `scripts/build_stats_sqlite.py`, `_ChunkedBackend` in `stats_provider.py`, `models/player_stats_cache_v3_metadata.pkl`.
- `rm -rf models/cache_chunks_v3/` (11 GB reclaimed).
- Update `run_experiment.py` to dispatch both scripts with the new YAML schema.
- Keep `parsing_v2.py` under `scripts/legacy/` for 30 days as rollback.

**Phase C — Per-match materialization cache** (optional follow-up, ~1-2 days) — **post-Phase B**
- Materializer writes per-match parquet to `data/balls_cache/<match_id>_<feature_hash>.parquet`.
- Final split parquet = concat of per-match parquets.
- Ablations changing a ball-context feature only re-materialize matches whose inputs changed.
- Ablation iteration time: 10-15 min → seconds.
- Storage cost: ~100-200 MB extra (delta from monolithic parquet is small).

### Validation strategy

**Primary gate (both phases): bit-identical parquet output vs reference monolith.**

New harness `scripts/tests/test_split_parity.py` (~150 lines):
1. Run `parsing_v2.py` on a pinned 500-match slice → reference parquet.
2. Run the new pipeline on the same slice → candidate parquet.
3. `pd.testing.assert_frame_equal(check_exact=True)` across all 4 splits. Column-level diff on failure.

**Secondary gates**:
- **Eval parity**: train XGBoost on both parquets, run `sim_eval/run_sim_eval.py`. `simulated_prob` bit-identical per match (same shape as `scripts/tests/compare_phase4_evals.py`).
- **Cache parity (Phase B only)**: the SQLite from `build_stats_cache.py` matches a `build_stats_sqlite.py` output when both read the same JSONs. Row-count + `_meta` checks + 100 random `_get_raw_batting` spot-checks.
- **Temporal integrity**: for each parquet row, assert `match_date > every cache snapshot date used in that row's feature lookups`. Catches any as-of-date leak.

**Rollback**:
- Phase A: revert the `--materialize-only` branch; `parsing_v2.py` is untouched.
- Phase B: `run_experiment.py` dispatch flip back to `scripts/legacy/parsing_v2.py` for 30 days.

### Pros
1. Phase 5 comes for free — SQLite becomes source of truth; 11 GB of chunks reclaimed.
2. Ball-context feature additions: re-materialize only (5-10 min) vs full re-parse (15 min).
3. YAML-driven splits and feature sets; today splits are buried in code.
4. Incremental data refresh — cache-builder appends new JSONs without replaying the corpus.
5. Unlocks Phase C per-match cache; ablation time → seconds.
6. Natural home for empirical outcome distributions (see §"Empirical Outcome Distributions" above) — expanded `PlayerStatsTracker` lives in one place.
7. Materializer is embarrassingly parallel across matches; SQLite mmap supports N readers (proven in Phase 4: 2 concurrent evals at 1.7 GB combined).
8. Cleaner concerns: trackers vs stateless feature emission vs XGBoost training.

### Cons / Risks
1. **Drift risk**: materializer's as-of-date SQLite lookup could diverge from the monolith's live-tracker lookup if the snapshot-boundary semantics are off by one ball. **Mitigation**: Phase A's bit-exact parity harness under the safe monolith catches any drift before extraction; Phase B re-runs the same harness.
2. **Two artifacts to version**: cache (`_meta.schema_version`) + parquet (`.feature_hash`). **Mitigation**: extend `run_experiment.py:check_smart_cache` to validate both and fail loudly on mismatch (already the pattern for the existing SQLite staleness check at `stats_provider.py:635-650`).
3. **JSON read twice** (cache-builder + materializer). ~11K matches × small JSON = <1 min extra; not a blocker. Premature to optimize.
4. **Immediate wall-clock win is small** — `skip_parsing: true` already makes most ablations free today. Structural wins compound over many experiments, not one.
5. **`PlayerMetadataProvider` needed by both scripts** — the cache-builder already needs `batter_hand` + `is_pace` for `batting_vs_type` / `bowling_vs_hand` tracker keys (`parsing_v2.py:1036-1038`). Loading a 100 KB CSV twice is free.

### Alternatives considered

**Pure soft-split only (no structural split)** — keep monolith, only ship `--materialize-only`.
- Captures ~70% of benefit, zero refactor risk.
- Leaves Phase 5 cleanup undone, keeps the tracker/feature entanglement, blocks the plugin-feature system.
- **Verdict**: adopted as Phase A stepping-stone; don't stop there.

**Feature plugin registry** — each feature group registers `compute(ball_context, stats_provider) -> dict`.
- Enables "drop a feature file, run" workflow.
- Big refactor, fights XGBoost's static-feature-list assumption, `feature_registry.py` would need to carry compute functions alongside column names.
- **Verdict**: defer until post-Phase-C. The materializer's structure makes this cheap to add later.

**Single undifferentiated parquet, split at train time** — materializer emits one ~11M-row parquet; `xgboost_v2.py` filters by date.
- Removes hardcoded split logic from the materializer.
- Training-memory hit; extra date-column filtering in every training script.
- **Verdict**: rejected. Split-by-time is a clean dataset boundary and keeps training simple.

### Open questions
1. **Cache rebuild trigger** — `build_stats_cache.py` should detect new JSONs (mtime-based) and append-only; force full rebuild only when tracker semantics change (e.g., new type-based stat). **Resolution**: `--force-rebuild` flag + mtime comparison in Phase B design.
2. **Innings-state in materializer** — per-match JSON walk stays; the materializer is not purely SQL. Accepted; JSON I/O is fast at this corpus size.
3. **Multi-JSON-same-date handling** — the monolith takes the first snapshot per date and ignores subsequent ones (`parsing_v2.py:1263`). SQLite dedup is already last-write-wins (`build_stats_sqlite.py:73-77`). **Resolution**: keep the existing semantics; the parity harness will catch any divergence.

### Recent-form features — train/inference mismatch (BLOCKER surfaced 2026-04-21)

Discovered while scoping Phase A. Four features in the training parquet are computed from state that the stats cache does not serialize — so any materializer that reads from the cache alone will diverge on these columns, and any simulator that reads from the cache alone already does.

**Features affected**:
- `batsman_recent_avg`, `batsman_recent_sr`
- `bowler_recent_avg`, `bowler_recent_econ`

**Where they come from today**:
- `PlayerStatsTracker.recent_batting` / `recent_bowling` are `defaultdict(lambda: deque(maxlen=5))` (parsing_v2.py:187-188).
- `start_match()` / `end_match()` (parsing_v2.py:194-205) push per-match aggregates into those deques.
- `get_batting_features()` / `get_bowling_features()` (parsing_v2.py:218-231, 243-255) sum the last-5-match deque contents to compute `*_recent_*`.
- Training parquet receives real values via `**batting_features` / `**bowling_features` at parsing_v2.py:971-972.

**What the cache stores**:
- `deep_copy_stats()` (parsing_v2.py:494-535) serializes only career totals: `batting_stats`, `bowling_stats`, `h2h_stats`, `batting_vs_type`, `bowling_vs_hand`, plus optional `venue` and `batting_elo`/`bowling_elo`. **The per-match deques are not in the snapshot.**
- SQLite schema (`stats_sqlite_backend.py:64-161`) has no `recent_batting` / `recent_bowling` tables — it's a direct serialization of the snapshot format.

**Consequence for inference (already live, already wrong)**:
- `sim_v1_2.py:1170-1173, 2033-2036, 2523-2526` all do `features['batsman_recent_avg'] = bat_stats.get('recent_avg', bat_stats.get('avg', 25.0))`. Since the StatsProvider-returned dict has no `recent_avg` key, this silently falls through to `avg` (career) or the literal default `25.0`.
- The simulator never supplies real recent form. XGBoost was trained on last-5-match rolling aggregates; at inference it sees career averages. Feature-shift on 4 of 63 features.
- Impact magnitude unmeasured. Candidate explanation for part of the train-vs-eval discrepancy; worth measuring with an ablation (drop the 4 columns, retrain, compare eval metrics).

**Consequence for Phase A**:
- The `--materialize-only` flag would read from SQLite via `StatsProvider`, so it cannot reproduce the 4 columns either. Bit-identical parity vs the full monolith is unachievable on these columns without a cache-schema change.

**Proposed fix** (folded into Phase A scope):
1. Extend `deep_copy_stats` to serialize `recent_batting` / `recent_bowling` deques (store as `list[dict]`, not deque — JSON/SQLite friendly).
2. Add `recent_batting` / `recent_bowling` tables to SQLite schema (keys: `(player_id, date_id)`, values: aggregated-last-5 blob — runs/balls/dismissals totals already summed, not the raw deque, to keep rows compact).
3. Add `StatsProvider.get_batting_recent(pid, date) -> {avg, sr}` / `get_bowling_recent(pid, date) -> {avg, econ}` methods.
4. Update `sim_v1_2.py` to call the new methods instead of the current `.get('recent_avg', ...)` fallback. Removes the train/inference mismatch.
5. Rebuild the SQLite cache once under the new schema. Since chunks still exist (Phase-5 gate), we have a fallback.

**Validation**:
- Parity gate (test_split_parity.py): now achievable on all 63 columns.
- Eval delta (2026-04-21, 261-match polymarket × 10 sims, XGBoost v3):

  | metric                  | pre-fix (silent 0 / career-avg) | post-fix (SQLite recent) | delta |
  |-------------------------|---------------------------------|--------------------------|-------|
  | avg_log_loss            | 0.7541                          | 0.7518                   | −0.0023 |
  | avg_brier_score         | 0.2743                          | 0.2728                   | −0.0014 |
  | avg_edge (\|model−mkt\|) | 0.1994                          | 0.1864                   | −0.0130 |
  | flat_betting_roi_pct    | −1.37 %                         | +6.51 %                  | +7.88 pp |
  | flat_betting_win_rate   | 41.18 %                         | 45.10 %                  | +3.92 pp |
  | frac_kelly_roi_pct      | +0.22 %                         | +1.27 %                  | +1.05 pp |
  | bets_placed             | 255                             | 255                      | 0 |

  216 / 261 matches diverged — silent-fallback was live on 83 % of matches. All metrics moved favorably without changing bet volume.
  Baseline: `eval_out_baseline/xgboost_20260421_213207.json`. Post-fix: `eval_out_postfix/xgboost_20260421_220541.json`. Diff via `scripts/tests/compare_phase4_evals.py`.
- Risk: fixing this feature-shift could improve eval metrics (model finally sees the features it was trained on) OR could expose a deeper miscalibration. Outcome: improvement (see table). Separate calibration work still warranted — pre-calibration ECE was already on the todo list.

**Scope impact**: ~1 extra day in Phase A. Phase A becomes "fix the recent-form mismatch + soft split" instead of just "soft split".

### Success criteria
- **Phase A**: `--materialize-only` parquet bit-identical to full monolith on all 4 splits. Landed before 2026-04-26.
- **Phase B**: split scripts produce parquet + SQLite bit-identical to Phase-A reference. `run_experiment.py` works unchanged for every `experiments/configs/*.yaml`. 11 GB reclaimed. Landed within 7 days of the Phase-5 gate.
- **Phase C**: ablation iteration ≤ 30 s for feature-only changes. Per-match cache hit rate ≥ 95 % across 3 consecutive ablation runs.

### Estimated effort
| Phase | Cost | Blocker |
|---|---|---|
| A — `--materialize-only` flag | ~half day | None |
| B — structural split + Phase 5 cleanup | ~2 days | Phase-5 gate (≥ 2026-04-26) |
| C — per-match cache | ~1-2 days | Phase B stable for 3+ ablations |

Critical path: ~3 days of focused work, spread over ~2 weeks.

---

## What NOT To Do

- Don't chase ball-level accuracy beyond ~60% — individual balls are inherently noisy.
- Don't optimize for the 44-match test set — too small, leads to overfitting.
- Don't increase simulations beyond 1000 — diminishing returns, 1000 is sufficient for stable estimates.
- Don't add features without validating that existing features are working correctly (fix bugs first).
- Don't trust any evaluation results that show >5% ROI on a small sample without extensive validation.
