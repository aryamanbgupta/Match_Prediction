# M4 — within-tournament features (2026-05-10)

Phase 4 of match-level v3. Added 15 within-tournament / scheduling features. **Outcome: DROPPED.** All variants regress vs M2 v.o. unfrozen baseline on iteration evidence; the marginal LL improvement on standalone test is over-confidence-driven and doesn't translate to better bet selection.

**Production baseline remains** `models/xgb_match_v3_m2_venue_only_unfrozen/`.

## Features added (15)

A. **Date-windowed form** (5 features):
- `team{1,2}_win_rate_last_60d`, `team{1,2}_n_matches_last_60d`, `win_rate_last_60d_diff`
- Beta(1,1)-shrunk win rate over matches in the last 60 days.

B. **Competition-filtered form** (5 features):
- `team{1,2}_competition_win_rate`, `team{1,2}_competition_n_matches`, `competition_win_rate_diff`
- Beta-shrunk win rate over the last 365 days within the same `competition_tier` (int 1-4).

C. **Scheduling proxies** (5 features):
- `days_since_team{1,2}_last_match` (cap 365), `is_team{1,2}_back_to_back` (≤ 1 day), `days_since_diff`

## Iteration evidence — all variants

All on `data/xgb_match_data_v3_m4_unfrozen/` (chronological-walk unfrozen materialization, the M3-adopted mode):

| Variant | iter ≥$50k LL raw | iter ≥$50k ROI raw [CI] | iter ≥$50k LL Platt | iter ≥$50k ROI Platt [CI] |
|---|---|---|---|---|
| **M2 v.o. unfrozen (baseline)** | **0.6348** | **+25.40% [+4.75, +48.11]** | **0.6279** | **+26.69% [+6.26, +48.65]** |
| M4 full (all 15 features) | 0.6319 | +20.19% [+0.84, +42.22] | 0.6272 | +12.84% [-6.38, +34.95] |
| M4 window-only | 0.6402 | +21.92% [+1.59, +43.04] | 0.6334 | +20.01% [+1.02, +41.64] |
| M4 − scheduling | (not run on iter) | — | — | — |
| M4 − competition | (not run on iter) | — | — | — |

All M4 variants either match or regress on iteration. M4 full Platt ROI CI now INCLUDES 0 (`-6.38, +34.95`) where M2 v.o. Platt cleanly excluded it (`+6.26, +48.65`). On raw, M4 full ROI lower-bound drops from M2's +4.75 to +0.84.

Strict per-plan exit criterion: "any LL improvement OR clear ROI lift on IPL-only adversarial slice." M4 full LL is marginally better (Δ -0.003 raw) and IPL slice shows LL -0.020, ROI +6.1pp. So technically M4 PASSES the gate. But the gate doesn't catch the iteration-aggregate regression, and **iteration is the only set we tune against** (golden is held out for audit, not selection — see "Eval discipline" below).

## Drop-one ablation (standalone test LL)

| Variant | val LL | test LL | Δ test LL vs M4 full |
|---|---|---|---|
| M2 v.o. unfrozen | 0.6521 | 0.6015 | n/a |
| M4 full | 0.6463 | 0.5947 | — |
| M4 − scheduling | 0.6429 | **0.5852** | **−0.0095** (best, dropping scheduling helps a lot) |
| M4 − competition | 0.6435 | 0.5913 | −0.0034 |
| M4 − date-window | 0.6473 | 0.5995 | +0.0048 (worse → date-window helps) |
| M4 window-only | 0.6491 | 0.5983 | +0.0036 |

Scheduling features actively hurt. Competition-filter borderline. Date-window slightly helps. But the standalone test improvement from `date-window` doesn't translate to iteration ROI — the over-confidence cost outweighs.

## Why M4 fails — three converging problems

### 1. Form features are highly redundant with M1

Pearson correlations on train (n=7,912):

| Pair | r |
|---|---|
| `win_rate_diff` (M1) ↔ `win_rate_last_60d_diff` (M4) | 0.678 |
| `win_rate_diff` (M1) ↔ `competition_win_rate_diff` (M4) | **0.789** |
| `win_rate_last_60d_diff` ↔ `competition_win_rate_diff` | 0.705 |
| `team1_win_rate_last_10` (M1 A2) ↔ `team1_win_rate_last_60d` | 0.677 |
| `team1_win_rate_last_10` ↔ `team1_competition_win_rate` | **0.788** |

All three diffs correlate near-identically with target (~0.155). They measure the same thing — recent team form — at different time horizons. XGBoost splits the same signal across 3 features instead of 1 → "agrees with itself harder" → over-confidence without new information.

### 2. Scheduling features fit noise

| Feature | train r vs team1_wins | test r vs team1_wins |
|---|---|---|
| `days_since_team1_last_match` | -0.001 | -0.054 |
| `days_since_team2_last_match` | +0.004 | -0.016 |
| `days_since_diff` | -0.007 | -0.044 |
| `is_team1_back_to_back` | -0.024 | -0.024 |
| `is_team2_back_to_back` | -0.002 | +0.025 |

Train correlations are essentially zero. But XGBoost still uses these features (importance 0.011-0.014 except `back_to_back` at 0.0) through tree interactions — fitting noise.

Train vs test win rate by `days_since_team1_last_match` bin:

| bin | train win rate | test win rate | n train | n test |
|---|---|---|---|---|
| 0-2 days | 47.0% | 48.6% | 3555 | 387 |
| 2-5 days | 51.2% | 48.2% | 2219 | 199 |
| 5-10 days | 49.6% | 42.4% | 781 | 59 |
| **10-30 days** | **50.5%** | **71.0%** | **188** | **31** |
| 30-365 days | 48.8% | 36.5% | 1169 | 115 |

The 10-30 day bin shows 50.5% on train (basically noise) but 71% on test. The model learned no signal here on train and projects badly. The 30-365 bin shows the same kind of train→test instability.

`is_team*_back_to_back` has zero importance — base rate ~17-24% with negligible target correlation; XGBoost correctly ignores them.

### 3. Net effect: more confidence, less accuracy

Probability distribution on iteration test (n=782):

| Metric | M2 v.o. | M4 full |
|---|---|---|
| Mean \|p−0.5\| | 0.1139 | 0.1254 |
| Std of p | 0.152 | 0.166 |

M4 makes predictions ~10% more confident on average. Cumulative accuracy at confidence thresholds:

| Confidence | M2 v.o. n / acc | M4 full n / acc |
|---|---|---|
| \|p−0.5\| > 0.10 | 324 / 80.6% | 361 / 80.9% |
| \|p−0.5\| > 0.15 | 225 / **85.3%** | 251 / 84.1% |
| \|p−0.5\| > 0.20 | 154 / **89.6%** | 171 / 86.0% |

**M4 has MORE high-confidence predictions but each is LESS accurate.** At \|p−0.5\| > 0.20, accuracy drops 89.6% → 86.0% (-3.6pp absolute). With Kelly sizing this is the worst possible failure mode: bigger bets at lower hit rates.

This explains the Platt ROI regression: Platt amplifies whatever resolution is in the model. M4's resolution is *over-confident-resolution*, so Platt amplifies the wrong thing → ROI drops harder than the raw model's drop.

## Deeper lesson — feature-correlation discipline

Adding correlated features to a model that already has the signal hurts more than it helps. M3 (player-level recency) failed for the same reason — career aggregates already captured the variance. M4 (within-tournament form) failed because `win_rate_diff` and `team1_win_rate_last_10` already captured it.

**Going-forward discipline for M5+**: before training any new feature group, compute its correlation matrix against the existing M1 + M2 v.o. features. Any new feature with |r| > 0.5 vs an existing feature must demonstrate **orthogonal predictive signal** (e.g., target correlation must be meaningfully higher than the existing correlated feature's; or a sub-slice where the new feature improves discrimination beyond what the correlated existing feature does). Otherwise the new feature should be skipped — it'll add over-confidence without adding edge.

The M5 (player × opposition / venue affinity), M6 (conditions, captain) candidates should fare better because they target signals that aren't in the M1 baseline at all. But the correlation check is mandatory.

## Eval discipline correction (2026-05-10)

While writing this M4 report, user flagged that I had been letting golden-set numbers influence M-phase landing decisions. That contaminates golden (selection-against-it). **Going forward: iteration test drives all M-phase decisions. Golden is audit-only — measured but not selected against.** See `feedback_iteration_only_decisions.md` memory.

This is why M4 lands as DROP even though M4 full's golden Platt ROI was +44.23%. That number is now a held-aside audit observation, not evidence for landing M4.

## What this means for M5

- **M5 baseline is still `models/xgb_match_v3_m2_venue_only_unfrozen/` (raw)**.
- **Numbers to beat (iteration only)**: ≥$50k raw LL 0.6348 / ROI +25.40% [+4.75, +48.11]; ≥$50k Platt LL 0.6279 / ROI +26.69% [+6.26, +48.65]; ≥$100k raw ROI +26.21% / Platt ROI +26.61% [+0.63, +57.42].
- **Correlation check is mandatory** before training M5: player × opposition affinity should not correlate strongly with `h2h_team1_win_rate_shrunk`; player × venue affinity should not correlate strongly with `is_team{1,2}_home` or `venue_p4/6/w`.
- **M4 features stay in the materializer code path** (`_within_tournament_features` helper, FEATURE_COLUMNS entries, TeamFormTracker extensions). Excluded from production via `--drop-features` substring filter. Available for re-evaluation if we later figure out how to extract orthogonal signal (e.g., interactions with team_quality conditional on tournament intensity).

## Artifacts preserved

- `models/xgb_match_v3_m4_unfrozen/` — full M4 model
- `models/xgb_match_v3_m4_{no_sched,no_comp,no_window,window_only}/` — drop-one variants
- `data/xgb_match_data_v3_m4_unfrozen/` — 99-feature parquet (M2 + M3 + M4 columns available)
- `reports/walk_forward_m4.md`, `reports/walk_forward_m4_wo.md` — monthly breakdowns

## Headline (one-line)

M4 features have signal on standalone test (test LL 0.5947 vs M2's 0.6015) but the signal is **over-confidence on the same hypothesis M1 already encodes**, not new information. Iteration ROI regresses; drop.
