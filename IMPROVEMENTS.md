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

### Calibration vs. Resolution — what the Walsh & Joshi finding does NOT say (added 2026-05-08)
A common misread of Walsh & Joshi is "always calibrate, calibration generates returns." That is not what their result shows, and applying it naively has hurt this project twice.

**The mechanical claim.** Calibration (isotonic, Platt) is a *monotone* mapping: it preserves the ranking of predictions and only redistributes probability mass to make per-bin frequencies match empirical rates. It cannot create resolution — it can only honour the resolution the underlying model already has.

**Why this hurts flat ROI on an under-resolving model.** When the underlying model is over-dispersed (predictions of 16% where events happen 40% of the time, predictions of 74% where events happen 51% of the time), calibration pulls those bins toward the empirical rate, which is closer to the base rate (50%). LL/Brier improve because per-bin frequencies match better. But our flat-betting decision rule (`model_prob > market_prob`) only cares about being on the correct side of the market line — and an over-confident-but-correctly-ranked prediction at 75% vs market 50% becomes a calibrated 60% vs market 50% (still bets), or worse, a 60% vs market 65% (no longer bets). At the margins, predictions that *barely* cleared the market line lose their edge as soon as they collapse toward 50%.

**The Walsh & Joshi finding holds when the underlying model has resolution and miscalibration is the binding constraint.** Ours doesn't — coinflip beats v7 on LL on the ≥$50k slice (0.6931 < 0.7402), which is the textbook signature of an under-resolving model rather than a miscalibrated one. A perfectly calibrated coinflip is trivially achievable and generates zero edge over market.

**Empirical evidence on this project.**
- 2026-03 Platt LOOCV experiments: LL improved, flat ROI dropped.
- 2026-04-23 v6 outcome-dist (which is implicitly a calibration improvement at the ball level): LL 0.7518 → 0.7122 (−5.3%), flat ROI +6.5% → −7.1%.
- Phase 5 hierarchical shrinkage walked some of that back (LL ~flat, flat ROI swung +15pp on all-261, +16pp on ≥$50k) — by adding *resolution* to the narrow cells, not by recalibrating.

**Implication for the go/no-go gate.** The gate (`model LL < market LL on ≥$50k AND ROI CI > 0`) is a resolution problem. Closing the 0.114 LL gap to market requires features or architecture that discriminate strong vs weak teams better; a calibration layer alone cannot move it and may anti-correlate with ROI. Use calibration as measurement infrastructure (clean ablations, Kelly correctness, edge-threshold rules) — not as a feature work prerequisite.

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

### Empirical Outcome Distributions (IMPLEMENTED — schema v4, 2026-04-23)
Direct multi-class target encoding for player/venue context. **42 new features across 5 hierarchies**, shipped as `xgb_v6_outcome_dist`. Replaces nothing — existing avg/SR/econ kept; feature importance sorts out redundancy in a follow-up ablation.

**Why this is high-impact**:
- Summary stats (avg=35, SR=140) compress a player into 2 numbers, destroying the distribution shape. Two batters with identical avg/SR can have completely different outcome profiles (power hitter vs accumulator).
- XGBoost **cannot learn** outcome distributions from label-encoded player IDs — trees can only split on "id > N", which groups arbitrary players together. Pre-computed distributions give the model exactly the signal it structurally can't extract.
- Direct target alignment: the model predicts P(outcome class), and these features provide historical P(outcome class | context). The learning task becomes "blend these priors" instead of "reconstruct distributions from scratch".
- Current player stats correlate r=0.06–0.11 with ball outcomes (<1.5% variance explained). Direct historical rates (e.g., `batter_p4`) carry stronger per-class signal.

**Hierarchy shipped** (6 features per cell, 42 total):

| Level | Features | k (shrinkage) |
|-------|----------|---------------|
| Batter overall | `batter_p{0,1,2,4,6,w}` | 30 |
| Bowler overall | `bowler_p{0,1,2,4,6,w}` | 30 |
| Batter vs pace/spin | `batter_p{...}_vs_{pace,spin}` (12) | 30 |
| Bowler vs LHB/RHB | `bowler_p{...}_vs_{lhb,rhb}` (12) | 30 |
| Venue | `venue_p{0,1,2,4,6,w}` | 200 |

**Sparsity handling — empirical Bayes shrinkage** (no hard thresholds):

    p̂_c = (n_c + k · π_c) / (N + k),   N = Σ n_c

- N → 0 ⇒ p̂ → π (full fallback to global prior)
- N → ∞ ⇒ p̂ → n / N (data dominates)
- N = k ⇒ half-and-half

π is the corpus-wide outcome distribution computed during `build_stats_cache.py`'s walk and stored in SQLite `_meta.prior_p*`. Final π = (0.304, 0.411, 0.076, 0.108, 0.047, 0.054) on 2.19M balls.

**Schema v4 changes**:
- 6 new INTEGER count columns (`c0, c1, c2, c4, c6, cw`) on `batting`, `bowling`, `batting_vs_type`, `bowling_vs_hand`, `venue` tables. h2h unchanged (sparse 2D cell).
- `_meta.prior_p{0,1,2,4,6,w}` rows store π.
- 5 new getters on `_SQLiteBackend` and `PlayerStatsTracker`/`VenueStatsTracker` with shared `_shrink` helpers (one in `parsing_v2.py` for the live-tracker path, one on `_SQLiteBackend` for SQLite reads).
- DB grew 46.5 MB → 56.8 MB (+10 MB). Build 396s → 440s.

**Validation**:
- Phase A parity harness: **9519/9519 matches bit-exact** including the 42 new columns (251s).
- 7/7 unit tests on shrinkage math (`scripts/tests/test_outcome_dist_shrinkage.py`) — empty counts → π exactly, large N → MLE, Σp̂=1, half-weight check, tracker–backend equivalence.
- Schema-v4 conservation + query-plan + getter sanity (`scripts/tests/test_schema_v4_outcome_dist.py`) on 1000-row samples per table — Σ(c0..cw) ≡ `balls`/`balls_bowled`/`total_balls`, planner still uses index on all 10 query shapes.
- `_verify_outcome_count_conservation` runs at build time (sample 500 per table).

**Eval results (261 polymarket × 100 sims, exp `xgb_v6_outcome_dist_20260423_182515_7d92bfc`)**:

| Metric | v4 baseline (2026-04-21) | **v6 (outcome_dist)** | Δ |
|---|---|---|---|
| avg_log_loss | 0.7518 | **0.7122** | **−5.3%** |
| avg_brier_score | 0.2728 | **0.2562** | **−6.1%** |
| flat_roi_pct | +6.51% | −7.1% | −13.6 pp |
| flat_win_rate_pct | 45.10% | 41.60% | −3.5 pp |
| frac_kelly_roi_pct | +1.27% | +0.7% | −0.6 pp |

Calibration improved (LL −5.3%, Brier −6.1%); flat-betting regressed. Same calibration-vs-ROI tension already documented in §"Calibration System (IMPLEMENTED & TESTED — March 2026)" — sharper probabilities near 50% compress betting edges. Per-slice (≥$50k / ≥$100k) eval not yet run; pending in follow-ups.

**Files touched**: `scripts/{stats_sqlite_backend,parsing_v2,build_stats_cache,tracker_rehydration,materialize_features,sim_v1_2,feature_registry}.py`, `experiments/configs/xgb_v6_outcome_dist.yaml`, plus 2 new tests. Plan file: `~/.claude/plans/yes-let-s-go-with-cheerful-waffle.md`. Memory note: `project_outcome_dist_v6.md`.

#### v6 follow-ups (LANDED 2026-04-24/25 — six-phase plan)

Plan file: `~/.claude/plans/yes-let-s-go-with-cheerful-waffle.md`. Five of the six phases shipped over two days; Phase 4 (refactor) deferred per user direction.

**Phase 1 — Per-slice eval infrastructure**: shipped. `--min-volume` flag on `run_sim_eval.py` filters polymarket odds by `polymarket_volume_usd`; bootstrap 95% CIs (1000 resamples, percentile method, seed=42) emitted for log loss + flat ROI; output JSON gains `slice` / `min_volume` / `n_matches_evaluated` / `*_ci_low|high`. `compare_slices.py` renders cross-model tables, `reslice_eval_json.py` post-hoc-slices legacy eval JSONs (no recompute), `run_sliced_eval.sh` runs all 3 slices in one shot. 16 unit tests pass. v4 baseline (post-fix) sliced via `reslice_eval_json.py` against `eval_out_postfix/xgboost_20260421_220541.json` — output to `eval_out_phase1_sliced_v4/`.

| Slice | v4 LL | v6 LL | Δ |
|---|---|---|---|
| all (261) | 0.7518 [0.696, 0.814] | 0.7122 [0.659, 0.765] | −0.040 |
| ≥$50k (170) | 0.7838 [0.711, 0.858] | 0.7370 [0.667, 0.807] | −0.047 |
| ≥$100k (110) | 0.7776 [0.689, 0.869] | 0.7041 [0.624, 0.790] | −0.074 |

v6's calibration win **grows with liquidity** — the outcome-dist features earn their keep on sharp markets. Flat-ROI CIs straddle zero on every model on every slice; no model clears the project's go/no-go bar (LL < market 0.6267 AND ROI CI excludes zero on ≥$50k). Memo: `project_v6_sliced_eval.md`.

**Phase 2 — Drop-encodings ablation → KEEP encodings**. New config `xgb_v6_no_encodings.yaml` excludes `batter_encoded` / `bowler_encoded` (112-feature subset). Training results: top features shifted dramatically (is_powerplay gain 0.225 → 0.017; gain spreads across remaining features). Sliced eval verdict:

| Slice | v6 LL | no_encodings LL | Δ |
|---|---|---|---|
| all (261) | 0.7122 | 0.7805 | **+0.068** |
| ≥$50k (170) | 0.7370 | 0.8059 | **+0.069** |
| ≥$100k (110) | 0.7041 | 0.8007 | **+0.097** |

Plan threshold was Δ > 0.005 on ≥$50k → result is 14× threshold and consistent across slices. **Encoded IDs are NOT redundant** with the 42 outcome-dist features. Pre-train gain rank (53/54, gain 0.0064) understated their value — XGBoost importance can mask correlated-but-essential features. Memo: `project_phase2_drop_encodings.md`.

**Phase 3 — Phase prior → DROP phase_p\***. End-to-end implementation landed: cache rebuild adds 18 `_meta` rows (`prior_{pp,mid,death}_p*`); per-phase priors land sane (PP dot 42% / death wicket 9.6% — distinct from global π). `parsing_v2._classify_phase_pre_ball` + `_phase_dist_from_priors` + 18 inn_agg keys + per-innings conservation guard + `_SQLiteBackend.get_phase_outcome_dist` + `sim_v1_2._fill_outcome_dists(balls_bowled=)` all wired. 11 unit tests pass. But:

| Slice | v6 LL | v7 (+phase prior) LL | Δ |
|---|---|---|---|
| all (261) | 0.7122 | 0.7346 | **+0.022** |
| ≥$50k (170) | 0.7370 | 0.7599 | **+0.023** |
| ≥$100k (110) | 0.7041 | 0.7398 | **+0.036** |

**Why it failed**: 6 phase_p* features take only 3 unique values each (one per phase), collinear with the existing `is_powerplay` / `is_middle_overs` / `is_death_overs` binary indicators. Trees can't extract orthogonal information; the new features only add overfit noise. The plan's premise — "add 6 features per ball selected by which phase the ball is in" — missed that those 6 features collapse to 3 values modulo the existing phase indicators. Implementation kept inert in code; resurrect via `experiments/configs/xgb_v7_phase_prior.yaml`. Memo: `project_phase3_phase_prior.md`.

**Phase 4 — `build_ball_features` refactor**: DEFERRED. Pure plumbing (consolidate the 5 `sim_v1_2.py` model wrappers' duplicated feature-build blocks at lines 578/978/1152/1620/2056/2547 into one helper). Plan estimate ~4h; must keep Phase A parity at 9519/9519 bit-exact. Not blocking subsequent phases.

**Phase 5 — Hierarchical shrinkage → SHIP**. Two-stage shrinkage on the 4 narrow cells (`batter_vs_pace`, `batter_vs_spin`, `bowler_vs_lhb`, `bowler_vs_rhb`): instead of `shrink(counts, π, k)`, compute `shrink(counts, parent, k)` where `parent = shrink(counts_overall, π, k)`. For sparse-data players, the narrow cells now fall back to the *player's own overall distribution* instead of toward the global prior π. `hierarchical=True` is the new default on `_SQLiteBackend.get_{batter_vs_type,bowler_vs_hand}_outcome_dist` and the equivalent tracker getters; pass `hierarchical=False` to recover v6 flat-shrunk values. 5 unit tests pass.

| Slice | v6 LL → v7 LL | v6 ROI → v7 ROI |
|---|---|---|
| all (261) | 0.7122 → 0.7158 | −7.08% → **+7.96%** |
| ≥$50k (170) | 0.7370 → 0.7402 | −9.81% → **+6.11%** |
| ≥$100k (110) | 0.7041 → 0.7311 | −4.80% → −2.86% |

Calibration essentially flat (Δ LL +0.003 / +0.003 / +0.027), but flat ROI swings positive on all and ≥$50k slices. Win rate also jumped 41.6% → 49.4%. ROI CIs still straddle zero so not statistically significant — but the cross-slice consistency is encouraging. Hierarchical shrinkage produces meaningfully different predictions for sparse-data players' vs-type cells, which apparently lands the model on better bet decisions even when aggregate LL barely moves. Memo: `project_phase5_hierarchical_shrink.md`.

**Phase 6 — k_player sweep → k=30 confirmed optimal**. Plumbed `k_player` / `k_venue` end-to-end: YAML `outcome_dist:` block → `materialize_features.materialize(k_player=, k_venue=)` → `parsing_v2.parse_match_data_v2(k_player=, k_venue=)` → `xgboost_v2.py` writes `models/xgb_v3/outcome_dist_config_v3.json` sidecar at train time → `sim_v1_2.XGBoostModelV2.__init__` reads sidecar → `_fill_outcome_dists` threads k to backend getters. `_check_parquet_cache` invalidates parquet on k change.

| k_player | LL | flat ROI |
|---|---|---|
| 10 | 0.7719 [0.716, 0.830] | −9.73% [−27.8, +15.8] |
| **30** | **0.7158 [0.661, 0.776]** | **+7.96% [−10.6, +29.1]** |
| 100 | 0.7357 [0.672, 0.802] | −2.21% [−19.8, +19.2] |
| 300 | 0.7562 [0.698, 0.818] | −2.35% [−21.0, +19.9] |

k=10 (less shrinkage) → noisier from sparse-data players, hurts both metrics. k=100/300 (more shrinkage) → distributions collapse toward π, losing per-player signal. **k=30 hits the sweet spot**: enough shrinkage to denoise sparse players, but enough player-specific signal to find +ROI bet decisions. Sliced eval skipped on k=10/100/300 — all-slice was decisive. Memo: `project_phase6_k_sweep.md`.

#### Final v7 = hierarchical shrinkage, k_player=30, k_venue=200

Active config: `experiments/configs/xgb_v6_hierarchical_shrink.yaml` (despite "v6" in the filename, this is the v7 path-of-record). Parquet at `data/xgb_data_v3/`, 114 features, hash `c520a3ba08ae`. Active model `models/xgb_v3/` carries the Phase 5 model + sidecar declaring `k_player=30.0, k_venue=200.0`.

**Backups in case of revert**:
- `models/xgb_v3_v6_backup/` — v6 flat shrinkage (the `xgb_v6_outcome_dist` baseline)
- `models/xgb_v3_phase5_k30/` — Phase 5 origin
- `models/xgb_v3_phase6_k{10,100,300}/` — k-sweep variants

**Open follow-ups from this wave**:

1. **Match-level calibration layer** (TODO.md §"Blockers" P0). v7's ≥$50k LL is 0.7402 vs market's 0.6267. Calibration alone (isotonic + Platt LOOCV on the 255 outcome-matches) is expected to close most of that gap (the 74%→51%, 16%→40% calibration bins are symmetric over-dispersion, mechanically fixable). This blocks any honest "did this feature help?" claim — without calibration, every comparison is contaminated by miscalibration noise.

2. **Per-player phase distributions**. Phase 3's natural successor: `P(outcome | phase, batter_id)` — a per-batter × 3 phases × 6 outcomes = 18-cell tensor, sparse, needs hierarchical shrinkage similar to Phase 5. These would NOT be collinear with `is_powerplay` because they vary across players — exactly the orthogonal signal Phase 3's global priors lacked.

3. **Separate `k_narrow` sweep**. Phase 5/6 held `k_narrow = k_player = 30` throughout. An independent sweep on `k_narrow` could widen the hierarchical ROI win — or reveal that k=30 was just the right point for both knobs. Plumbing in place; needs a 3-way YAML grid.

4. **Phase 4 build_ball_features refactor** (deferred). Worth doing before adding more features — the 5 wrappers are the future-proofing pain point.

5. **Larger eval set / golden holdout**. 261 polymarket matches sits at the edge for ROI signal — bootstrap CIs span ~±20pp. The corpus has matches after `golden_start = 2026-04-17`; building a `betting_odds_golden.json` from genuine pre-match Polymarket snapshots is the cleanest way to add evaluation power without overfitting risk. See TODO.md §"Preserve a true holdout".

6. **Benchmarks helper** (TODO.md §"Benchmark stack"). Wire coinflip / always-favorite / market into the standard eval output so every experiment has the same 4-row context.

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

## Performance Pass (LANDED 2026-05-08 → 2026-05-09)

Three-phase optimization driven by a cProfile of the v7 production eval
(41.9 min for 261 matches × 100 sims). The cProfile showed 55 % of wall
in XGBoost C-side `inplace_predict` (mostly unrecoverable) and 41 % in
feature extraction (very compressible).

### Phase 1 — `strftime` cache on `_SQLiteBackend._norm_date` (commit `85d67bf`)

1.77 M `strftime` calls in 38 s, all converting the same `match_date`.
Replaced the `@staticmethod` with an instance method backed by a tiny
dict cache (cap 16 entries, keyed on the datetime object). Strings hit
a passthrough fast path. Cache stripped on `__getstate__` for fork-safe
pickling.

| Metric | Δ |
|---|---:|
| 5×100 wall | 53.1 s → 50.1 s (**−5.6 %**) |
| `strftime` calls | 1.77 M → 340 K (−81 %) |
| `strftime` tottime | 2.12 s → 0.46 s (−78 %) |
| RSS | flat |
| Output | bit-identical |

### Phase 2 — per-player memoization on `StatsProviderCache` (commit `135514a`)

The wrapper memoized only 5 team-level methods. Extended with a tier-2
memo on the 12 per-ball getters: `get_batting_stats`, `get_bowling_stats`,
`get_h2h_stats`, the two `*_recent` and two `*_vs_type/hand_stats`, and
all 5 outcome-dist getters. Keys are `(player_id, date_str)` (or
`(p1, p2, date_str)` for h2h, with optional `k`/`hierarchical` for
outcome-dist). Within an innings the striker stays for many balls and
the bowler for 6 — cache locality was previously entirely unused.

| Metric | Δ |
|---|---:|
| **5×100 wall** | 53.1 s → **43.6 s** (**−17.9 %** total vs baseline) |
| Per-ball `extract_features` | 0.13 ms → 0.06 ms (**−54 %**) |
| `_fill_outcome_dists` cumtime | 4.47 s → 0.90 s (**−80 %**) |
| RSS overhead | +13 MB (memo footprint) |
| Output | bit-identical |

Memory bound is small: ~8K unique `(player, date)` pairs across 261
matches × 12 memos × ~16 KB ≈ 200 KB total. No invalidation needed.

### Phase 3 — multi-process parallel eval unlocked (commit `e4b97cc`)

Pickle round-trip on the wrapper was broken: pickle's `__setstate__`
probe fell through to `__getattr__` and returned the *backend's*
`__setstate__`, restoring the wrong `__dict__` shape on workers. Failed
silently with `Error evaluating match: _batting_stats`.

Fix: explicit `__getstate__`/`__setstate__` on both `StatsProvider` and
`StatsProviderCache`, plus a dunder short-circuit in `__getattr__` so
pickle/copy/etc. never get forwarded to the wrapped object.

With pickle correct, ran the experiment the SQLite migration was
intended to enable: N independent eval processes on disjoint match
shards, sharing the SQLite mmap. **Critical**: cap `OMP_NUM_THREADS`
per process or BLAS threads oversubscribe the cores and the parallel
runs serialize.

Measured throughput on 10 matches × 100 sims per process (10 logical
cores on the dev box):

| Config | Wall | Throughput | Combined RSS |
|---|---:|---:|---:|
| 1 proc, default OMP | 86.3 s | 1.00× | 887 MB |
| 2 procs, OMP=4 (no cap) | 173.7 s | **0.99×** (oversub!) | 1.1 GB |
| **2 procs, OMP=2** | 96.3 s | **1.79×** | 890 MB |
| **4 procs, OMP=2** | 148 s | **2.33×** | 1.6 GB |

Per-process RSS is ~440 MB; the 16 GB box has headroom for ~16 procs
if compute scaling didn't tail off (memory bandwidth, perf vs efficiency
cores limit usable N to ~4–5 in practice).

Driver: `perf_runs/run_n_parallel.py`. Operational recipe in
`docs/OPERATIONS.md` § "Multi-process parallel eval".

### End-to-end full-eval result (2026-05-09 — measured)

| Configuration | 261×100 wall | Speedup vs baseline |
|---|---:|---:|
| Pre-perf serial (2026-05-08) | 41.9 min | 1.00× |
| **Phase 1+2 + 4-proc parallel (OMP=2, 2026-05-09)** | **16.6 min** | **2.52×** |

Decomposition: ≈ 1.18× from Phase 1+2 single-proc gains + ≈ 2.1× from
4-process parallelism on top.

Numerics on the parallel run are within Monte Carlo noise of the serial
baseline (per-worker RNG seeds differ by construction, not a regression):

| Metric | Serial baseline (2026-05-08) | 4-proc parallel (2026-05-09) | Δ |
|---|---:|---:|---:|
| avg log loss | 0.7155 | 0.7150 | −0.0005 |
| avg Brier | 0.2530 | 0.2526 | −0.0004 |
| flat ROI | +8.0 % | +7.96 % | −0.04 pp |
| flat win rate | 49.4 % | 49.4 % | flat |

Per-shard balance (4 shards × ~65 matches × 100 sims):

| Shard | Matches | LL | Brier | flat ROI | Time |
|---|---:|---:|---:|---:|---:|
| 0 | 65 | 0.7552 | 0.2660 | +7.20 % | 16.6 min |
| 1 | 65 | 0.6675 | 0.2355 | +11.41 % | 16.4 min |
| 2 | 65 | 0.6903 | 0.2385 | −9.85 % | 16.5 min |
| 3 | 66 | 0.7466 | 0.2702 | +23.13 % | 16.4 min |

Per-shard variation in LL/ROI is the natural per-tournament-slice
difference (chronological splitting puts T20Is in one shard, BBL in
another, etc.); the n-weighted average over all 261 matches recovers
the serial baseline.

Bit-identical numerics across Phase 1 and Phase 2 verified by JSON diff
on 5 matches × 6 fields prior to this run.

### What was NOT promising

- `--parallel` on `run_sim_eval.py` (intra-match `multiprocessing.Pool`):
  now correct after `e4b97cc`, but slower than serial because IPC cost
  of the model+memos exceeds the gain at small sims/match. Documented
  as "do not use" in OPERATIONS.md.
- XGBoost batch prediction (vectorising all 240 balls of a sim): ~50 %
  predict-side savings possible, but each ball's features depend on the
  previous ball's outcome, so vectorising requires rewriting the sim
  loop. Architectural lift, ~2 days, easy to break temporal-integrity
  invariants. Not pursued.

---

## Parsing Pipeline Split (Phase A + B LANDED, Phase C pending)

**Planned 2026-04-19. Phase A + Phase B both LANDED 2026-04-22.**

### Goal
Replace the monolithic `scripts/parsing_v2.py` (1409 lines, ~15 min per run) with two scripts whose boundaries map to the two real data artifacts:

1. `scripts/build_stats_cache.py` — chronological tracker walk → `models/player_stats_cache_v3.sqlite`
2. `scripts/materialize_features.py` — stateless per-match feature emission → `data/xgb_data_v3/*.parquet`

This subsumes Phase 5 of the SQLite migration (`TODO.md:127-134`) and the "split parsing" bullet (`TODO.md:136-139`). The two items were scoped separately but edit the same code; bundling halves the migration cost.

### Motivation
The parser today does three things in one pass. They split along a **cross-date stateless, within-date stateful** boundary — not the "stateless across matches" framing an earlier draft of this section used:

1. **Cross-date chronological tracker walk**: `PlayerStatsTracker`, `VenueStatsTracker`, `PlayerEloTracker`. SQLite is the serialized output of this.
2. **Snapshot emission**: pre-match snapshot (first-write-wins per date). SQLite rows keyed on `(entity, date_id)`.
3. **Feature materialization** (`parse_match_data_v2`, `parsing_v2.py:744-1107`): per-ball feature row. Stateful within an innings (`InningsFeatureCalculator`) AND stateful across same-day matches within a batch — `end_match` advances the recent-form deque, `update_venue_stats_detailed` mutates venue counters, and ball-level `update_stats` / ELO updates advance career state mid-batch. **Across *dates* it's a pure function of (ball context, SQLite snapshot at match_date, player metadata); within a date, same-day matches must run serially in monolith order.**

Point 3 is the key insight — and its honest form: once the SQLite cache exists, feature materialization is cross-date parallelizable per date, not per match. The unit of independence is **the date**, not **the match**. This has three downstream consequences:

- **Phase C per-match cache**: the cache key must include position-in-same-day-batch, not just `<match_id, feature_hash>`. Swapping M1 ↔ M2 inside a same-day batch produces different features, so adding one match to a busy day invalidates every same-day sibling. The per-match cache is still useful (solo-date matches hit it cleanly; most dates in the corpus have one match), but the hit rate on busy-ICC days is low.
- **Parallelism unlock**: per-date, not per-match. `ProcessPoolExecutor(max_workers=N)` across dates is the future optimization.
- **Incremental refresh**: a new JSON for a date that already has matches forces re-materializing the full date (can't skip to "just this match"). Phase B ships a full-rebuild cache-builder anyway; a real `--since` path is a future add.

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

**Phase A — Parity harness** (LANDED 2026-04-22, ~1 day)
- Standalone test harness at `scripts/tests/test_phase_a_parity.py` proves the "per-date stateful, cross-date stateless" materializer assumption without editing the monolith. Replaces the `--materialize-only` soft-split proposal (which would have branched inside a script we plan to delete).
- New helpers: `scripts/loaders_common.py` (`iter_matches_chronological`), `scripts/tracker_rehydration.py` (`rehydrate_stats_tracker` / `rehydrate_elo_tracker` / `rehydrate_venue_tracker` — reads SQLite raw counters, reconstructs the three trackers per-date).
- Harness groups matches by date; rehydrates `temp_*` once per date from SQLite using the union of same-day players + venues; then for each match on the date runs BOTH the live-tracker reference path and the `temp_*` candidate path and `assert_frame_equal(check_exact=True, check_dtype=True)`. `temp_*` accumulates across same-day matches via `parse_match_data_v2`'s mutations + post-match venue updates — matching the monolith.
- **Result**: PASS on all **9519 male T20 matches** in 186s (~51 match/s).
  - 3664 first-of-date matches: **63/63 columns** bit-exact.
  - 5855 same-day-secondary matches: **59/63 columns** bit-exact. The 4 recent-form columns (`batsman_recent_avg/sr`, `bowler_recent_avg/econ`) are excluded (see schema-v2 limitation below).
- **Exit gate**: met. Becomes the reusable regression gate for Phase B.

**Schema-v2 limitation — resolved by Phase B schema v3 bump**
- Schema v2 stored 5-match recent-form as a single summed triple per (player, date) — `recent_runs/balls/dismissals` on the batting/bowling rows (stats_sqlite_backend.py:83-85, 95-97). The monolith's `PlayerStatsTracker.recent_batting` / `recent_bowling` are 5-entry `deque(maxlen=5)`; `end_match` pushes the current match's aggregates and evicts the oldest slot when full. Eviction identity was lost in the single-sum, so first `end_match` inside a same-day batch on players already at maxlen couldn't be reproduced from the schema-v2 seed.
- Blast radius: ~0.25% of same-day secondaries, 4 of 63 columns (`batsman_recent_avg/sr`, `bowler_recent_avg/econ`).
- **Schema v3 (Option E: per-match aggregate log)** closes the gap. See Phase B writeup below.

**Phase B — Structural split + schema v3** (LANDED 2026-04-22, 1 day)

Extracted the parsing monolith into two scripts, bumped SQLite to schema v3, deleted 12 GB of chunk remnants. Plan file: `~/.claude/plans/yes-go-ahead-and-structured-lecun.md`.

#### Landed state

| Deliverable | Where |
|---|---|
| `scripts/build_stats_cache.py` (~400 lines) | JSON → SQLite schema v3 directly. Replaces `scripts/build_stats_sqlite.py` (which was a chunks→SQLite converter; now deleted). Full corpus build: **9519 matches in 396 s → 46.5 MB DB** (+4.8 MB over schema v2 for match-log tables). |
| `scripts/materialize_features.py` (~320 lines) | SQLite + JSON → parquet via per-date batching (reuses Phase A's `tracker_rehydration.py` + `loaders_common.py`). **9519 matches in 170 s** — 3.5× faster than the monolith (~600 s) because the tracker-walk is skipped. Pandas shape bit-identical to monolith parquet: 92/93 columns match exactly; `innings_id` (93rd) has identical grouping semantics but different hash values due to Python `hash()` randomization (benign — only used as a `groupby` key in transformer dataset code, not as a training feature). |
| `scripts/stats_sqlite_backend.py` | `SCHEMA_VERSION = 3`. Added `batting_match_log` / `bowling_match_log` tables (composite PK `(player_id, date_id, intra_date_idx) WITHOUT ROWID`). Added `get_batting_match_log_recent(pid, as_of_date, limit=5)` / `get_bowling_match_log_recent(...)` with `_strict_before_bound` helper (uses `bisect_left` so strict `date_id < target` semantics work when `as_of_date` is exactly a snapshot date). |
| `scripts/tracker_rehydration.py` | `rehydrate_stats_tracker` reads from the match-log (newest-first, reversed into the deque) instead of seeding a single summed entry. Empty log → empty deque (covered by `get_batting_features`'s `balls==0` branch). |
| `scripts/tests/test_schema_v3_match_log.py` | 7 unit tests on the new getters: newest-first ordering, `limit` honored, strict `date_id<?`, log-sum consistency with the denormalized `batting.recent_*` / `bowling.recent_*` columns, `USING INDEX` query plan, intra_date_idx ordering within a date. All PASS. |
| `scripts/run_experiment.py` | `check_smart_cache` returns `(sqlite_valid, parquet_valid)`. SQLite validity: `_meta.schema_version == 3` AND `_meta.source_json_mtime_max >= max(data/t20s_json/*.json mtime)`. Parquet validity unchanged. Dispatch runs `build_stats_cache.py` / `materialize_features.py` independently based on each flag. |
| YAML `data.splits` block | Optional. Missing keys fall back to `DEFAULT_SPLITS` in `materialize_features.py`, so existing `experiments/configs/*.yaml` continue working unchanged. |
| `scripts/stats_provider.py` | Single-backend facade over SQLite. `_ChunkedBackend` class + chunks fallback deleted (861 → 264 lines, −69% / −583 lines). |
| `scripts/parsing_v2.py` | Orchestrator (`process_folder_v2_with_splits`) + `__main__` block removed (1446 → 1137 lines, −21% / −309 lines). Helper primitives (tracker classes, `deep_copy_stats`, `parse_match_data_v2`, `classify_match_k_factor`, `InningsFeatureCalculator`) retained — all still used by the new pipeline + simulation code. |

#### Schema v3 design — Option E (per-match aggregate log)

One row per `(player, match-they-played-in)`. Deque reconstructed at read time via `ORDER BY date_id DESC, intra_date_idx DESC LIMIT N`.

#### Schema v3 design — Option E (per-match aggregate log)

Instead of storing recent-form as a single summed triple per (player, snapshot-date), store **one row per (player, match-they-played-in)**. The deque is reconstructed at read time via `ORDER BY date_id DESC, intra_date_idx DESC LIMIT N`. Picks flexibility (variable N, venue/tier-filtered recency, corrections-friendly) over a minimal-change schema.

```sql
CREATE TABLE batting_match_log (
    player_id INTEGER NOT NULL,
    date_id INTEGER NOT NULL,
    intra_date_idx INTEGER NOT NULL,   -- 0,1,2... monolith order within date
    runs INTEGER NOT NULL,
    balls INTEGER NOT NULL,
    dismissals INTEGER NOT NULL,
    PRIMARY KEY (player_id, date_id, intra_date_idx)
) WITHOUT ROWID;

CREATE TABLE bowling_match_log (
    player_id INTEGER NOT NULL,
    date_id INTEGER NOT NULL,
    intra_date_idx INTEGER NOT NULL,
    runs_given INTEGER NOT NULL,
    balls_bowled INTEGER NOT NULL,
    wickets INTEGER NOT NULL,
    PRIMARY KEY (player_id, date_id, intra_date_idx)
) WITHOUT ROWID;
```

PK rows are structurally unique — **no delta compression needed**. Disk cost: ~30 log rows/match × 9500 matches × 28 bytes ≈ 8 MB. DB grows 41.7 → ~50 MB.

**Sum-columns decision (E1)**: keep existing `batting.recent_*` and `bowling.recent_*` columns AS-IS. They're the denormalized cache read on the simulation hot path by the 5 sim wrappers in `sim_v1_2.py` (the 2026-04-21 recent-form-fix patches). The log is additive, used only for rehydration and future flexible queries. Two derivations of the same underlying tracker state — consistent by construction.

#### Rehydration correctness (per-date batching from Phase A still works)

Phase A rehydrates `temp_*` trackers ONCE at each date boundary; `parse_match_data_v2` mutates `temp_stats.recent_batting[pid]` across same-day matches via `end_match`. With Option E:

- At start of date D: `get_batting_match_log_recent(pid, D, limit=5)` returns up to 5 matches strictly before D, newest-first. Reverse and append → `recent_batting[pid] = deque([oldest, ..., newest])`.
- Same-day match M1 runs; `end_match` pushes M1's aggregates; deque evicts oldest if at maxlen (matching monolith).
- Same-day match M2 runs; deque reflects post-M1 state.

Strict `date_id < ?` in the query (not `<=`) — at the start of D's batch, we want matches BEFORE D; the monolith's deque at that moment contains only pre-D entries.

#### Flexibility unlocked

These become trivial once schema v3 is live. All are marked NON-GOALS for Phase B itself (scope discipline) — they're the reason Option E was chosen over Option A:

1. **Variable window size** — `limit` is a getter parameter. Recent-10 instead of recent-5 = one parameter change.
2. **Venue-filtered recent form** — add a `venue_id` column to the log (additive), filter the query.
3. **Competition-tier filtered recent form** — same pattern with a tier column.
4. **Recency-weighted averages** — compute in Python from 5 returned rows with age-based weights. Zero schema change.
5. **Form trajectory features** — "last match runs vs avg of prior 4" — second query over the log.
6. **Cricsheet corrections** — `REPLACE INTO batting_match_log` for a revised match flows through downstream rehydration automatically. Recent-form self-heals; career-cumulative tables still need date-replay.

#### Validation gates — all passed

| Gate | Result |
|---|---|
| `scripts/tests/test_schema_v3_match_log.py` (7 assertions) | **PASS** (newest-first ordering, `limit` honored, strict `date_id<?`, log-sum ≡ denormalized `batting.recent_*` sum columns, `USING INDEX` query plan, intra_date_idx ordering) |
| **`scripts/tests/test_phase_a_parity.py` — primary exit criterion** | **PASS on all 9519 male T20 matches, 63/63 columns bit-exact, 194 s** (3664 first-of-date + 5855 same-day-secondary — schema v3 match log fully reproduces the monolith's deque eviction) |
| Parquet parity spot-check — new `materialize_features.py` output vs pre-Phase-B monolith parquet | **92/93 columns bit-identical** on all 2,187,930 rows across train/validation/test splits; `innings_id` differs due to Python `hash()` randomization (benign, same grouping semantics) |
| Cross-DB spot-check — new SQLite vs `models/player_stats_cache_v3.sqlite.pre_phase_b` | **Row counts identical** on all 8 v2-subset tables (batting=150 006, bowling=114 850, h2h=473 757, etc.); 500 random batting + 500 random bowling rows bit-identical by string-key (int-ID ordering differs because intern order depends on data source) |
| Eval parity — 261-match polymarket × 10 sims, new model vs `eval_out_postfix/xgboost_20260421_220541.json` | **Flat-betting P&L / ROI / win rate / bets-placed bit-identical** (255 bets, +16.60 PnL, +6.51% ROI, 45.10% win rate). Log loss 0.7528 (Δ +0.0010), Brier 0.2730 (Δ +0.0002), edge 0.1852 (Δ −0.0011) — all within Monte Carlo noise at n_sims=10. Kelly ROIs diverged more (frac -0.37 pp, full -1.50 pp) because Kelly sizing is non-linear in probability estimates; the *decisions* are preserved perfectly. |

#### Measured results

| Metric | Pre-Phase-B | Post-Phase-B | Notes |
|---|---|---|---|
| Stats cache disk | 11 GB (chunks) + 41.7 MB (SQLite v2) | 46.5 MB (SQLite v3 only) | **−12 GB reclaimed**; +4.8 MB over schema v2 for match-log tables |
| Cache rebuild walltime (full corpus) | 5 min 43 s (chunks) + 5 min 43 s (chunks→SQLite convert) | 6 min 36 s (JSON → SQLite v3 direct) | One-step build; chunk intermediate eliminated |
| Feature materialization walltime | ~600 s (monolith `process_folder_v2_with_splits`) | **170 s** (`materialize_features.py`) | **3.5× faster** — skips the tracker walk now that SQLite exists |
| Lines of code — parsing core | `parsing_v2.py` 1 446 | `parsing_v2.py` 1 137 (helpers only) + `build_stats_cache.py` ~400 + `materialize_features.py` ~320 | Concerns split; each script <500 lines |
| Lines of code — stats provider | `stats_provider.py` 861 (incl `_ChunkedBackend`) | `stats_provider.py` 264 | **−583 lines / −69%** |
| Phase A harness coverage | 59/63 cols on same-day secondaries | **63/63 cols on all matches** | Schema v3 resolves the recent-form gap |

#### Rollback

`scripts/parsing_v2.py` helper primitives retained, but the orchestrator + `PARSING_LEGACY=1` dispatch were removed in the Phase 5 cleanup — `git revert` / `git checkout` is the rollback path if we ever need the old pipeline. The pre-Phase-B SQLite (`player_stats_cache_v3.sqlite.pre_phase_b`) was retained through validation and then removed; re-obtainable by building at the parent commit if needed. `models/cache_chunks_v3/` is gone (12 GB reclaimed, not recoverable on this machine; re-derivable by a full `build_stats_cache.py` run in ~6 min).

#### Flexibility unlocked (future feature experiments)

Option E's key payoff — these become trivial additions rather than schema migrations. Each is its own feature-ablation experiment, out-of-scope for Phase B:

1. **Variable window size** — `limit` is a getter parameter. Recent-10 vs recent-5 = one call-site change.
2. **Venue-filtered recent form** — add a `venue_id` column to the log (additive ALTER), filter the query.
3. **Competition-tier filtered recent form** — same pattern with a tier column.
4. **Recency-weighted averages** — compute in Python from the returned rows with age-based weights. Zero schema change.
5. **Form trajectory features** — "last match runs vs avg of prior 4" — second query over the log.
6. **Cricsheet corrections self-heal** — `REPLACE INTO batting_match_log` for a revised match flows through downstream rehydration automatically. Career-cumulative tables still require date-replay, but recent-form is already self-consistent.

**Phase C — Per-match materialization cache** (optional follow-up, ~1-2 days) — **post-Phase B**
- Materializer writes per-match parquet to `data/balls_cache/<match_id>_<feature_hash>.parquet`.
- Final split parquet = concat of per-match parquets.
- Ablations changing a ball-context feature only re-materialize matches whose inputs changed.
- Ablation iteration time: 10-15 min → seconds.
- Storage cost: ~100-200 MB extra (delta from monolithic parquet is small).

### Validation strategy

**Primary gate (both phases): bit-identical parquet output vs reference monolith.**

`scripts/tests/test_phase_a_parity.py` (LANDED 2026-04-22):
1. Chronological walk; group matches by date.
2. Per date: rehydrate `temp_*` trackers from SQLite; iterate same-day matches accumulating temp state.
3. `assert_frame_equal(check_exact=True, check_dtype=True)` candidate vs live-tracker reference, per match.
4. Same-day secondaries skip the 4 recent-form columns until Phase B's schema-v3 bump.

**Secondary gates**:
- **Eval parity** (Phase B post-cleanup): train XGBoost on the new parquet, run `sim_eval/run_sim_eval.py`. `simulated_prob` bit-identical per match on the 261-match polymarket test (same shape as `scripts/tests/compare_phase4_evals.py`).
- **Cache parity (Phase B only)**: the SQLite from `build_stats_cache.py` matches a `build_stats_sqlite.py` output when both read the same JSONs. Row-count + `_meta` checks + 100 random `_get_raw_batting` spot-checks.
- **Temporal integrity**: for each parquet row, assert `match_date > every cache snapshot date used in that row's feature lookups`. Catches any as-of-date leak.

**Rollback**:
- Phase A: harness is read-only; nothing to revert (no monolith edits).
- Phase B: `run_experiment.py` dispatch flip back to `scripts/legacy/parsing_v2.py` for 30 days.

### Pros
1. Phase 5 comes for free — SQLite becomes source of truth; 12 GB of chunks reclaimed.
2. Ball-context feature additions: re-materialize only (~3 min) vs full re-parse (~10 min).
3. YAML-driven splits and feature sets (schema in place; `data.splits` block optional, defaults cover existing configs).
4. Unlocks Phase C per-match cache; ablation time → seconds. (Cache key must include the same-day prefix — see Motivation — so hit rate on busy-ICC days is lower than on solo-date days.)
5. Natural home for empirical outcome distributions (see §"Empirical Outcome Distributions" above) — expanded `PlayerStatsTracker` lives in one place.
6. Materializer is parallelizable **per date** (not per match); SQLite mmap supports N readers (proven in Phase 4: 2 concurrent evals at 1.7 GB combined).
7. Cleaner concerns: trackers vs per-date-batched feature emission vs XGBoost training.

### Deferred follow-ups (not delivered in Phase B)
- **Incremental cache refresh (`--since`)** — today `build_stats_cache.py` is all-or-nothing (`out_path.unlink()` before every rebuild). A real append/resume path needs: checkpoint last processed date in `_meta`, reopen trackers from that snapshot, and append new rows. Doable, but its own plan.
- **Per-match parallelism** — ruled out by the same-day-stateful contract. Per-date parallelism via `ProcessPoolExecutor` is the future optimization (the YAML already supports it; materialize_features ships serial).
- **Promote `_SQLiteBackend` private accessors to public API** — `tracker_rehydration.py` currently punches through the `StatsProvider` facade to `_get_raw_batting` / `_player_id_map` / etc. Architectural cleanup, tracked in TODO.md.

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
- **Phase A** (LANDED 2026-04-22): parity harness bit-exact on 9519 matches. 63/63 columns on 3664 first-of-date; 59/63 on 5855 same-day-secondary (4 recent-form cols excluded per schema-v2 limitation).
- **Phase B** (LANDED 2026-04-22): schema v3 + two-script split landed; Phase A harness **63/63 on all 9519 matches** (no exclusions); eval parity bit-identical flat-betting metrics on polymarket_test; **12 GB of chunks reclaimed**; monolith orchestrator retired.
- **Phase C**: ablation iteration ≤ 30 s for feature-only changes. Per-match cache hit rate ≥ 95 % across 3 consecutive ablation runs.

### Estimated effort (actual)
| Phase | Cost | Status |
|---|---|---|
| A — parity harness | ~1 day | LANDED 2026-04-22 |
| B — structural split + schema v3 + Phase 5 cleanup | ~1 day (beat the 4-day estimate, Phase A infrastructure reuse paid off) | LANDED 2026-04-22 |
| C — per-match cache | ~1-2 days | Pending — gated on Phase B stable for 3+ ablations |

Critical path complete for Phase A+B. Phase C remains the next opportunity.

---

## Match-Level Direct + Sim Ensemble (LANDED 2026-05-09)

The v7 ball-level simulator was hitting LL 0.7402 on the ≥$50k polymarket slice — 0.114 above market 0.6267, and worse than coinflip 0.6931 on the sharpest matches. Six phases of outcome-distribution work moved this only marginally on sharp markets. Diagnosed as a **resolution problem, not calibration** (see "Calibration vs. Resolution" above).

Architectural hypothesis: the sim estimates `P(team1 wins)` by aggregating 240 stochastic per-ball events at ~30% per-class accuracy; variance compounding swamps the marginal signal even at infinite sims. A direct match-level model is supervised on the binary `team1_wins` target with no aggregation noise — fewer training examples (~9.2k matches vs 4M balls), but each example provides supervision on the metric that actually matters.

Plan file: `~/.claude/plans/okay-let-s-go-ahead-reflective-sunrise.md`.

### Architecture
Direct XGBoost binary classifier on match-level features (`models/xgb_match_v2_frozen/`), blended post-hoc with v7 sim outputs in logit space:
```
logit(P_final) = w · logit(P_sim) + (1 − w) · logit(P_direct)
```
The blend is post-hoc — existing v7 eval JSONs are reweighted by `w` and resliced via `reslice_eval_json.py`; no need to re-run sim. New scripts: `scripts/sim_eval/blend_eval_json.py`, `scripts/sim_eval/blend_report.py`.

### Phase A1 — cheap-subset features (~26 features)
- **Files**: `scripts/materialize_match_features.py` (one-row-per-match parquet at `data/xgb_match_data_v1/`), `scripts/xgboost_match_v1.py` (binary trainer with auto-feature-detection + `--predict-test` subcommand), `experiments/configs/xgb_match_v1_baseline.yaml`.
- Features: per-team batting/bowling ELO, batting avg / SR, bowling avg / econ, ELO and stat differentials, venue (encoded + avg score / chase win pct / dot pct / boundary pct), match context (toss, batting first, competition tier, is_international).
- Target: `team1_wins` derived from cricsheet `info.outcome.winner`. 291 no-result/abandoned matches dropped.
- **Result**: direct alone LL **0.6568** on polymarket-overlap subset (255 of 261 matches; 6 missing due to team-name aliasing across cricsheet vs polymarket). v7 sim was 0.7158. LL-vs-w curve **monotone increasing** — sim adds no LL value at any weight. Direct beats sim per-match on 119/261 matches (45.6%).
- Per-slice: gate not yet cleared (LL 0.6644 on ≥$50k still > market 0.6267).
- Report: `reports/blend_a1_report.md`.

### Phase A2 — richer features (~47 features)
- Added in-process trackers in `materialize_match_features.py`:
  - `TeamFormTracker`: last-N win rate per team, queried strictly before match date
  - `H2HTracker`: pairwise head-to-head win rate, Beta(1,1) shrunk toward 0.5 (k=2 prior)
  - `HomeVenueTracker`: 3+ matches in prior 730 days at venue
- Lineup features via `PlayerMetadataProvider`: `team{1,2}_{lhb,pace,spinner}_count`.
- ELO splits via squad-list ordering: `team{1,2}_top6_batting_elo_avg`, `team{1,2}_bottom5_bowling_elo_avg`, plus diffs.
- **Top feature importances**: `bottom5_bowling_elo_diff` (0.084), `top6_batting_elo_diff` (0.075). Position-split ELOs do more work than v7 sim's lineup-wide aggregates — the sim sees `batting_team_elo` but not "batting unit's strength concentrated in the top 6".
- **Standalone polymarket-overlap LL: 0.5226** (vs A1 0.6568, vs market 0.6267, vs v7 0.7158). Below market by 0.10 — first time the model has cleared LL parity.
- Report: `reports/blend_a2_report.md`. Caveats first reported in `reports/blend_a2_caveats.md` (since superseded by the no-leakage diagnostic).

### No-leakage diagnostic (added 2026-05-09)
A2's headline was suspicious — +47% ROI is far above realistic 1-3% market edges. Two diagnostic findings prompted a deeper audit:
- **Temporal split** (early test 2025-09→2026-01 vs late test 2026-01→2026-04): late ROI was ~3× early ROI. Initial hypothesis: tracker contamination during test period (form/H2H/home trackers accumulate state as we walk the corpus chronologically).
- **Outlier sensitivity**: stripping France @ 20.0 + Zimbabwe @ 11.76 wins drops all-slice ROI from +43% to +32%.

Implemented `--freeze-trackers-after DATE` in `materialize_match_features.py`:
- For matches with `date > freeze_date`: per-match fresh SQLite rehydration as-of `freeze_date + 1 day`. Prevents within-test cross-match contamination.
- For matches with `date > freeze_date`: A2 trackers (form / H2H / home) are read-only — no updates from this match's outcome flow back into the trackers.

**Counter-intuitive finding: FROZEN is BETTER than unfrozen across every slice.** The tracker contamination hypothesis is ruled out — if anything, unfrozen mode hurt by drifting test features past the temporal scope the model trained on (test trackers had ~9 months of test-period data; train trackers only had data prior to each train match).

| Variant | ≥$50k LL | ≥$50k ROI | ≥$100k LL | ≥$100k ROI |
|---|---|---|---|---|
| A2 unfrozen | 0.5135 | +47.35% | 0.4554 | +51.04% |
| **A2 frozen** | **0.5004** | **+53.67%** | **0.4361** | **+58.03%** |

The temporal divergence persists in frozen mode (early ROI +33.54% vs late +67.01%, gap +33.46pp) — but Phase A1 has the SAME ~33pp gap with no A2 trackers at all. **Composition explains it**: late test has 47 T20 World Cup 2026 qualifying matches (India vs Namibia, France vs Portugal, etc.) where strength differentials are extreme and the model's confidence is well-justified.

Audit pass clean across every temporal data path: SQLite rehydration uses `date_id < as_of` semantics; `_meta` priors are global but only consumed by v7 sim; train-target correlations are modest (max 0.33); binary features show physically plausible effects (3.6pp home advantage, no toss-winner effect).

Full report: `reports/no_leakage_diagnostic.md`.

### Headline as initially reported (A2 frozen, w=0.0) — INFLATED, retracted 2026-05-09

| Slice | LL (95% CI) | Flat ROI (95% CI) | n |
|---|---|---|---|
| all | 0.4944 [0.45, 0.53] | +50.73% [+32.4, +74.4] | 255 |
| ≥$50k | **0.5004** [0.45, 0.56] | **+53.67%** [+36.0, +73.8] | 168 |
| ≥$100k | **0.4361** [0.37, 0.50] | **+58.03%** [+33.4, +86.6] | 110 |

⚠️ **These numbers are leakage-inflated. See "ELO leakage discovered" subsection below for the honest replacement.**

### ELO leakage discovered + fixed (2026-05-09)

Within hours of reporting the "+47-58% ROI" headline above, an audit
discovered a feature-engineering leak. `materialize_match_features._build_match_record`
calls `_split_elo(team1_lineup_ids, elo_tracker)` to compute the
top-6 batting / bottom-5 bowling ELO splits. The `elo_tracker` it
receives has been mutated by `parse_match_data_v2` ball-by-ball with
this match's own outcomes — so the resulting ELO averages reflect
**post-match** state, not pre-match. The 6 affected features include
the model's two highest-importance features:

- `bottom5_bowling_elo_diff` (importance 0.084 — #1)
- `top6_batting_elo_diff` (importance 0.075 — #2)

**Empirical audit** across all 62 golden matches confirmed only those
6 features drift on every match. Other features — `team_*_elo` sums,
`team_*_avg/sr/econ`, venue stats, A2 trackers, lineup mix — are clean
because parsing_v2.py:1063-1098 computes team aggregates ONCE before
the ball loop, and the materializer updates A2 trackers AFTER
`_build_match_record` returns.

**Fix**: snapshot `temp_elo` BEFORE `parse_match_data_v2` mutates it,
pass the snapshot to `_build_match_record`. Live tracker is still
mutated so subsequent same-day matches see post-this-match state
(maintains monolith chronological semantics for cross-match features).
Patch: `materialize_match_features.py:519-526`.

**Retrained model**: `models/xgb_match_v2_clean/` — same hyperparameters
as `xgb_match_v2_frozen`, trained from scratch on the cleaned parquet
(`data/xgb_match_data_v2_clean/`).

### Honest headline on golden set (xgb_match_v2_clean, w=0.0)

Golden set is truly out-of-sample: 2026-04-17 → 2026-05-07, 55 matches
matched against polymarket. Never seen by training, validation, or
selection.

| Slice | LL (95% CI) | Flat ROI (95% CI) | n / win-rate |
|---|---|---|---|
| all | 0.6416 [0.59, 0.70] | +20.33% [-12, +49] | 55 / 53.7% |
| ≥$50k | 0.6747 [0.64, 0.72] | +32.61% [-0.20, +63.6] | 50 / 59.2% |
| ≥$100k | **0.6698** [0.63, 0.72] | **+34.75%** [+3.79, +65.5] | 45 / 61.4% |
| Reference: market | 0.6267 | — | — |
| Reference: coinflip | 0.6931 | — | — |

Strict gate (LL beats market AND ROI CI excludes 0): **fails on every slice**.
LL is approximately market-level — even a touch worse — on every liquidity slice.
Soft gate (ROI CI alone): clears only on ≥$100k.

### Diff: leaky vs clean (golden ≥$50k)

| Metric | Leaky | Clean | Δ |
|---|---|---|---|
| LL | 0.5004 | 0.6747 | +0.17 (worse) |
| Flat ROI | +53.67% | +32.61% | -21pp |
| Win rate | 71.4% | 59.2% | -12pp |
| ROI CI lower bound | +36.0% | -0.20% | excludes ↦ includes 0 |

Roughly two-thirds of the previously reported ROI was leakage-driven.

### Implications for the project
- **`xgb_match_v2_clean` is now the production winner-market predictor.** `xgb_match_v2_frozen` is preserved on disk for reference only.
- **`predict_fixture.py` and `build_ipl_dashboard.py`** were switched to the clean model. predict_fixture's predictions now match the clean parquet bit-exactly (validated KKR-GT 2026-04-17: 41.6% on both paths).
- The **direct vs sim story still holds qualitatively**: clean direct model still beats v7 sim on LL (0.67 vs 0.74 on ≥$50k), just by less. Position-split ELOs are still the top features even pre-match — they carry real signal, just less than with the post-match drift on top.
- **Honest production pitch**: the model is borderline-skilful. Modest positive ROI on the most liquid slice with a CI lower bound around +4%. Don't claim more than that until a clean forward test confirms it.

### Caveats remaining
1. **No live forward test yet** — golden set is "fresh polymarket capture but model could still be slightly tuned to artifacts of the iteration set". The truly clean signal is 30-60 days of capture-then-evaluate.
2. **Eval composition** still leans on tournament-international + IPL — no PSL/SA20/domestic-league representation in golden.

Full audit and comparison: `reports/leakage_fix_comparison.md`.

### Future improvements catalog — match-level direct model (2026-05-10)

Diagnosis: golden ≥$50k LL 0.6747 vs market 0.6267 — a **resolution gap** of ~0.05 LL. The current ~45 features are dominated by slow-moving career aggregates (ELO sums, lifetime avg/SR/econ, lineup composition). The market knows things this stack does not — fast-moving player form, phase-/matchup-specific lineup quality, within-tournament dynamics, conditions on the day, late roster news. Closing the gap requires features (or architecture) that *add discriminative information*, not features that recalibrate what's already there. Calibration alone has been shown twice to anti-correlate with flat ROI on this project (2026-03 Platt; 2026-04-23 v6 outcome-dist) and is not a path through the gate.

The catalog below groups every candidate improvement by category, flags external-data feasibility, and estimates relative cost. **Excluded** from the catalog by design: market price as a feature (residual modeling) — we have polymarket prices only on the eval set, not on the ~12k training corpus, so we cannot train this without acquiring multi-bookmaker historical odds first. Listed as deferred under §C.

The phased rollout is mirrored in `TODO.md` § "Match-level v3 — feature engineering plan", following the same shape as the v6→v7 outcome-dist phases (eval infra → highest-leverage feature → ablations → architecture sweep → sizing).

#### A. Feature additions

**A1. Phase/matchup outcome-dist transfer.** Aggregate the v7 ball-level outcome-dist features up to match level. The ball-level pipeline already shrunk these via empirical Bayes (k=30 player / k=200 venue) under SCHEMA_VERSION=4; the match-level model uses none of them. Backend getters already exist (`get_batter_vs_type_outcome_dist`, `get_bowler_vs_hand_outcome_dist`, `get_batter_outcome_dist`, etc.), so this is a pure aggregation layer, no schema work.

Proposed features (~12–18):
- `team1_top6_p4_vs_opp_pace_mix`, `..._p6_...`, `..._pw_...` — top-6 batter mean P(class) given the opposition's pace_count / spinner_count weighting.
- Symmetric for bowlers vs the opposing batting hand mix (lhb/rhb counts).
- Aggregate venue outcome dist (`venue_p4`, `venue_p6`, `venue_pw`) — direct match-level addition.

Why this is the highest expected lift: XGBoost cannot reconstruct outcome distributions from `team1_lhb_count` + `top6_batting_elo`. Pre-aggregated and pre-shrunk, these directly encode "this set of batters vs this attack profile produces X distribution of outcomes," which is closer to the target than any team-level summary. Feasibility: **today, no extra data, ~1 day implementation**.

**A2. Player-level rolling form (lineup-aggregated).** Replace coarse `team1_win_rate_last_10` with player-level recency stats aggregated to the lineup. Backend getters `get_batting_recent`, `get_bowling_recent`, `get_batting_match_log_recent`, `get_bowling_match_log_recent` already exist and serve windowed stats.

Proposed features (~8–12):
- `team1_top6_batting_avg_recent`, `team1_top6_batting_sr_recent` (mean of top-6 batters' rolling-window avg/SR)
- `team1_bowling_econ_recent`, `team1_bowling_avg_recent` (mean over all bowlers in the 11)
- `team1_n_inform_batters` (count of top-6 with rolling avg > career avg + threshold)
- `team1_n_outofform_batters` (symmetric)
- Symmetric for team2; pairwise diffs.

Targets the IPL-2026-mid-tournament-form gap that team-level ELO smooths over. Feasibility: **today, no extra data, ~1 day**.

**A3. Within-tournament features.** Computed from match data + competition_tier:
- `team1_tournament_win_rate` (current competition only — e.g., IPL 2026 to date)
- `team1_tournament_n_matches` (sample-size flag for the rate)
- `team1_tournament_run_rate_for`, `team1_tournament_run_rate_against` (NRR proxy from match runs/balls totals)
- `days_since_team_last_match` per team (fatigue / rest)
- `is_back_to_back` flag

Cheap to compute; targets the tournament composition effects flagged in `reports/no_leakage_diagnostic_clean.md`. Feasibility: **today, no extra data, half day**.

**A4. Player × opposition / player × venue affinity.** Per-batter career stats *against the opposing team* and *at this venue*, shrunk to overall.
- Player × opposition: derivable from h2h table (per-bowler), aggregated to opposing-XI level. Schema-friendly.
- Player × venue: requires either (a) a new (player, venue) tracker / table, or (b) deriving it on-the-fly from cricsheet JSONs at materialization time. (a) is cleaner and parallels existing schema; estimate +0.5 day for schema bump and +0.5 day for materializer wiring.

Captures "Pollard at Wankhede" / "Warner vs CSK" effects that career means hide. Feasibility: **partial today (player×opp via existing h2h)**; player×venue needs schema work but no new external data.

**A5. Conditions / scheduling.** No external weather API initially:
- `is_day_match` (from match start time in cricsheet info, where present)
- `month_of_year` (encoded), `month × venue` interaction (proxy for seasonal dew/heat at known dew-affected grounds — e.g., Chennai/Mumbai evenings in April–May)
- Toss × venue interaction explicitly featurized: `toss_winner_chase_propensity` based on venue chase win pct.

Cheap, derivable from cricsheet `info` block. Weather-API integration deferred (operationally heavy, modest expected lift). Feasibility: **today, no extra data, half day**.

**A6. Captain proxy.** From `info.toss.winner` plus lineup metadata. Each team's nominal captain has a per-captain win-rate-as-captain track record:
- `team1_captain_win_rate_as_captain` (last-N matches)
- `team1_captain_chase_win_rate_as_captain`
- Pairwise diffs.

Tactical decision-making (field placement, bowling rotation) is partially captain-driven and not captured anywhere else in the feature set. Feasibility: **today, no extra data, ~1 day** (slight overhead to track captain identity per match).

#### B. Architecture / training improvements

**B1. Hyperparameter resweep.** Current config (300 × lr=0.1 × depth=4) was set before the leakage fix. Run a small grid: `n_estimators ∈ {400, 600, 1000}` × `lr ∈ {0.03, 0.05, 0.07}` × `max_depth ∈ {3, 4, 5, 6}` × `subsample ∈ {0.7, 0.8, 0.9}`, early-stopped on val. Expected lift: 1–3% LL. Cost: half day with early stopping. Feasibility: **today**.

**B2. Stacking with disjoint feature subsets.** Train 2–3 base learners on non-overlapping feature blocks (e.g., team-strength only / contextual-only / phase-matchup-only), then logistic-regression-stack on val. Phase A2's blend with the v7 sim showed w=0 was optimal — but that was because v7 sim is *wrong*, not because stacking is. Disjoint XGBs trained directly on `team1_wins` should give the meta-learner real diversity. Feasibility: **today**, ~1 day.

**B3. Monotonic constraints.** Force monotone increasing on directional ELO/strength diffs (`top6_batting_elo_diff`, `bottom5_bowling_elo_diff`, `elo_diff_batting`, `elo_diff_bowling`, `win_rate_diff`, `batting_avg_diff`, `bowling_econ_diff` — sign-flipped). XGBoost has `monotone_constraints` natively. Prevents small-sample inversions; usually worth 0.005–0.015 LL. Feasibility: **today, hours**.

**B4. Calibration as sizing tool, not headline metric.** Isotonic regression LOOCV on val, applied to test predictions for the *Kelly sizing* path only. Don't let it drive the LL gate (TODO.md "measurement hygiene tool" framing). Required for honest fractional Kelly and edge-threshold rules. Feasibility: **today, half day**.

**B5. Per-tier specialization.** Train a separate model for the top-tier cluster (IPL + international + premier franchises) vs the long tail (associate / lower-tier domestic). Tier-1 has 10× more data and very different feature distributions. Light specialization often beats one-size-fits-all. Worth doing only after the feature set stabilizes. Feasibility: **today**, ~1 day.

**B6. LightGBM / CatBoost.** Marginal at best on tabular but cheap. CatBoost handles categorical `venue` natively and may pick up venue interactions the label-encoding loses. Low priority — only revisit if we're squeezing the last 0.005 LL. Feasibility: **today**, half day.

#### C. Data / corpus (deferred — needs new external data)

**C1. The Hundred (`hnd_json`).** TODO already lists this. Match-level model doesn't care about variable innings length (only first-row-of-each-innings is used in `_build_match_record`); blocker is the parser path. ~150–200 additional matches/year of high-quality data, modest expected lift.

**C2. Forward polymarket capture.** TODO already lists this. Provides the only path to a clean forward-test of the +34.75% ROI claim. Operationally critical, not feature work.

**C3. Multi-bookmaker historical odds.** Required to enable the (deferred) market-residual modeling approach. Polymarket alone covers only ~261 matches; Bet365 / Betfair / Pinnacle archives would give the residual model real training corpus. Operationally heavy; defer until C2 forward test confirms there's edge to refine.

**C4. Market-residual modeling (DEFERRED — explicit user decision 2026-05-10).** Train to predict `team1_wins − market_implied_prob` and ship `final_prob = market_prob + residual`. Excluded from current planning because we lack market prices on the training corpus. Reconsider only if C3 lands.

#### D. Evaluation upgrades

**D1. CLV (closing line value).** CLV is the gold standard for betting models — a model with positive CLV is empirically picking bets the market revalues toward truth, which is the only durable measure of edge. Sample-size requirements are far gentler than for ROI significance. **Feasibility: BLOCKED on data** (audited 2026-05-10). The current `betting_odds_polymarket.json` carries only one opening-line timestamp per market (no order-book snapshots, no closing line). Recoverable only after forward capture (C2) starts ingesting periodic snapshots through to market resolution. Re-evaluate once we have ≥30 matches with both pre-match and closing snapshots.

**D2. Stratified bootstrap.** Current bootstrap CIs treat matches as exchangeable; stratify by `competition_tier` × early/late half before resampling. Feasibility: **today, hours**.

**D3. Adversarial slices.** IPL-only, international-only, mismatches (top6 ELO diff > X), close fixtures (top6 ELO diff < Y). Diagnoses where the +34.75% golden ROI actually concentrates and whether it's robust. Feasibility: **today, hours**.

**D4. Walk-forward eval.** Re-train on expanding train+val and evaluate at each test month. Catches whether edge is gaining or losing over time. Feasibility: **today**, half day.

#### E. Inference / sizing

**E1. Edge-threshold + fractional Kelly.** Bet only when calibrated edge > 3%, size at quarter-Kelly. From the blend report, large fractions of bets clear by very slim margins; cutting those should preserve most ROI while halving variance. Requires B4 first. Feasibility: **today (after B4)**, hours.

**E2. Outlier per-bet stake cap.** Already flagged in TODO follow-ups. Cap at 2% of bank regardless of Kelly — keeps tail risk bounded for live deployment. Feasibility: **today**, hours.

#### Phasing — see `TODO.md` § "Match-level v3 — feature engineering plan"

Eight numbered milestones (M1–M8) following the v6→v7 phased shape: eval infra first (M1), then highest-leverage feature work with per-phase ablation (M2–M5), then conditions/captain (M6), then architecture sweep on the stabilized feature set (M7), then sizing/operational (M8). Forward-test (C2) and deferred items run in parallel to M1–M8.

### Stage-1 audit follow-ups landed (2026-05-09)

**v7 ball-level sim audited for analogous leakage — clean.** Same parser as the match-level path, but the bug pattern doesn't apply: per-ball features are computed before per-ball `update_stats` / `elo_tracker.update`; team-level constants (lines 1061-1098 of `parsing_v2.py`) are computed at match start before any ball-loop mutation; SQLite snapshot per date is first-write-wins (pre-D state); there is no second-pass equivalent of `_build_match_record`. Empirical check: 10/10 solo-date first-of-day matches show bit-exact match between parquet `striker_elo` / `batting_team_elo` and SQLite pre-D rehydration. v7 also doesn't compute top-6/bottom-5 ELO splits at all, so the specific bug pattern can't symmetrically apply. The honest LL gap (v7 0.7402 ≥$50k vs clean direct golden 0.6747) reflects a real resolution problem in v7, not a fixable measurement artifact. Full report: `reports/v7_leakage_audit.md`.

**No-leakage diagnostic re-run on clean model — finding compressed, not flipped.** Frozen still beats unfrozen on polymarket-overlap LL across every slice (Δ −0.014 / −0.010 / −0.016 on all/≥$50k/≥$100k), but margins are roughly half what they were under the leaky regime. ROI: frozen wins on `all` and `≥$50k`; unfrozen narrowly wins `≥$100k` (+27.53 vs +25.31). On the full 782-match standalone test, unfrozen is slightly better (LL 0.6027 vs 0.6180) — the freeze advantage is specific to the polymarket-overlap subset. Late-vs-early temporal gap (~0.07 LL) persists in both variants, consistent with the original composition-effect explanation (T20 WC mismatches concentrated late). Take-aways: tracker contamination is not the dominant driver in either direction; frozen is the safer iteration-eval framing; unfrozen is the correct semantics for live deployment. Full report: `reports/no_leakage_diagnostic_clean.md`.

### Implications for prior conclusions
- **No-leakage diagnostic** ("frozen better than unfrozen"): SURVIVES on polymarket-overlap with halved magnitude (see Stage-1 follow-ups above). Reverses on the broader full-test standalone — frozen advantage is specific to high-liquidity matches.
- **Logistic-regression stacker still skip**: clean direct still beats clean sim on LL. The qualitative ranking holds even if the magnitudes shifted.
- **"Calibration vs resolution" framing**: still vindicated qualitatively — direct supervision adds resolution that calibrating the sim couldn't. The clean numbers are smaller but the directional story holds.

---

## E-series experiments (2026-06-09, branch `improvement-experiments`)

Six experiments across both models. Protocol: one change per experiment, fit/select
on val only, iteration/test readout, pre-registered keep rules, every outcome
committed (autoresearch-style). Reports: `reports/e{1..6}_*.md`.

| Exp | What | Verdict | Headline |
|---|---|---|---|
| E1 | Temperature sharpening (match) | ❌ discard | T=1.205 (val-fit) is the first transform to beat market LL on iter ≥$50k (0.6246 < 0.6267) but costs ~10pp flat ROI — *any* monotone recalibration that crosses market prices on near-coinflip mass destroys side-selection alpha. Kept as sizing layer only. |
| E2 | Fair baselines for prop families (ball) | 🔁 rewrites prop framework | **No binary prop family beats an as-of fair baseline** (career/venue/positional, EB-shrunk). Sim's real skill = continuous score forecasts (batter-runs MAE −0.71 vs career baseline, highest-individual −1.89). `scripts/sim_eval/prop_fair_baselines.py` is now the bar for any prop claim. |
| E3 | 10-seed ensemble (match) | ❌ discard | Ensemble *worse* than production seed on val and iteration. Seed 29 is the best of 10 on val → M7 headline contains seed luck. Tempered forward expectation: ~0.64 LL / ~+16% ROI on ≥$50k, not 0.6299 / +21.9%. |
| E4 | Quantile lineup pooling (match) | ❌ discard (val rule) | Bowling quantiles (`bowl_elo_top2_diff`, target r 0.158) genuinely beat bottom5-mean redundancy check — but val LL regresses; iteration readout favorable (≥$50k ROI CI [+3.6, +47.0]) recorded as **forward-test hypothesis only**. |
| E5 | Ball-model bias root cause | ✅ **landed** | v7 trains with `balanced` class weights; sim sampled the tilted probs raw → P(wkt) 2× actual per ball, boundaries +0.05 — THE mechanism behind tail-event overshoot. Val-fit `VectorScalingCalibrator` collapses teacher-forced deltas to ~0 (runs/ball +0.024, P(wkt) −0.002; test multiclass LL 1.608→1.520). `--ball-calibrator vector` in prop_backtest. Also found: sim feeds `venue_encoded=0` always; `innings_id` is an unstable hash (TODO). |
| E6 | Direct in-play win-prob (ball states) | ✅ **landed** (fair_blend) | P(win\|state) trained on 1.81M deliveries. Crease/momentum extras add nothing over chase-math + pre-match rating (Δ +0.0005, CI [−0.003, +0.004], 780 OOS matches) — replicates MLC at 20× sample. `models/inplay_winprob_v1` (LL 0.5418/AUC 0.80 OOS) supersedes the sim for in-play probabilities. |

Durable lessons added this session:
1. **Side-vs-market is the alpha; probability magnitude is not** (E1). Post-hoc
   recalibration helps LL and sizing, never side-selection.
2. **Headline metrics on a fixed eval set contain selection luck** (E3) — judge
   forward tests against ensemble-tempered expectations.
3. **Fair baselines first** (E2) — base-rate skill is a mirage; this is now
   enforced by `prop_fair_baselines.py`.
4. **Check the loss the model was actually trained on before interpreting its
   probabilities** (E5) — the class-weight tilt sat undetected through v4→v7
   because winner-market sims average it out while prop tails amplify it.

## What NOT To Do

- Don't chase ball-level accuracy beyond ~60% — individual balls are inherently noisy.
- Don't optimize for the 44-match test set — too small, leads to overfitting.
- Don't increase simulations beyond 1000 — diminishing returns, 1000 is sufficient for stable estimates.
- Don't add features without validating that existing features are working correctly (fix bugs first).
- Don't trust any evaluation results that show >5% ROI on a small sample without extensive validation.
