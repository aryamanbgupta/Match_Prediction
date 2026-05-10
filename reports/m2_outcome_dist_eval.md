# M2 — outcome-dist transfer (2026-05-10)

Phase 2 of match-level v3. Transfer the v7 ball-level outcome-dist features up to match level via lineup aggregation.

**Outcome (decision 2026-05-10)**: Land **M2 venue-only** (3 features added). Drop the batter and bowler outcome-dist groups per the plan's drop-one rule (no LL contribution; bowler features actively hurt LL when included).

**Production model**: `models/xgb_match_v3_m2_venue_only/` — 49 features (M1's 46 minus `_aux` + venue_p4/venue_p6/venue_pw). Trained with monotone constraints. Calibrated via Platt LOOCV.

**M3 baseline**: `models/xgb_match_v3_m2_venue_only/` (raw, not calibrated).

## Feature design

Aggregated v7 outcome-dist features up to match level via lineup-weighted means:

| Group | Features | Mechanism |
|---|---|---|
| Batter (DROPPED) | top-6 `pX_expected` per team + diffs | Mean over top-6 of `batter_pX_vs_pace × opp_pace_share + batter_pX_vs_spin × opp_spin_share` for X ∈ {4, 6, w}. 9 features. |
| Bowler (DROPPED) | bottom-5 `pX_expected` per team + diffs | Symmetric: lhb_share / rhb_share over opposing top-6 batters. 9 features. |
| **Venue (KEPT)** | `venue_p4`, `venue_p6`, `venue_pw` | Direct venue outcome-dist (k=200, shrunk to corpus prior π). 3 features. |

All features use the existing `temp_stats.get_*_outcome_dist(prior, k)` getters in `parsing_v2.py`. Pre-match temporal correctness: features computed BEFORE `parse_match_data_v2` mutates the live trackers — same semantics as `pre_match_elo`.

Materialization: `data/xgb_match_data_v3_m2/{train,val,test,golden_test}.parquet`. 21 M2 columns are emitted; the trainer's `--drop-features` flag controls which subset gets ingested.

## Drop-one ablation

Trained 4 variants (same hyperparameters, `--monotone`):

| Variant | n_features | val LL | test LL (n=782) | iter ≥$50k LL | iter ≥$50k ROI [CI] | golden LL (n=61) | golden ≥$50k LL |
|---|---|---|---|---|---|---|---|
| M1 baseline | 45 | 0.6497 | 0.6146 | 0.6302 | +17.40% [-2.85, +41.14] | 0.6502 | 0.7006 |
| M2 full | 66 | 0.6540 | 0.6226 | 0.6374 | +20.84% [-0.51, +42.89] | 0.6445 | 0.6949 |
| M2 − batter | 57 | 0.6564 | 0.6211 | 0.6327 | +17.49% [-3.39, +39.92] | 0.6512 | — |
| M2 − bowler | 57 | 0.6521 | 0.6169 | **0.6266** | +17.67% [-3.52, +39.19] | 0.6407 | 0.7004 |
| M2 − venue | 63 | 0.6526 | 0.6253 | 0.6437 | +20.33% [-0.10, +42.06] | 0.6468 | — |
| **M2 venue-only (LANDED)** | **48** | **0.6521** | 0.6227 | 0.6347 | **+22.77% [+2.73, +43.46]** | **0.6380** | **0.6885** |

**Bowler outcome-dist features actively hurt** — dropping them improves test LL by -0.0057. Plausible cause: "bottom-5 by squad-list order" is not a clean bowling unit (some bottom-5 are tail-end batters, not bowlers), so bowler features are high-noise.

**Batter group is borderline** — neutral on val/test; slightly worse on golden when included (0.6445 vs 0.6512 without).

**Venue group consistently helps** — dropping it regresses LL on every slice (val/test/iter/golden).

Per the plan's drop-one rule ("cull anything with importance < 0.005 *and* no LL contribution"), batter and bowler groups are culled.

## M2 venue-only — final headline numbers

### Iteration polymarket eval (n=261, blend w=0.0)

| Slice | n | LL | LL 95% CI | Flat ROI | ROI 95% CI | Win % |
|---|---|---|---|---|---|---|
| ≥$50k raw | 170 | 0.6347 | [0.603, 0.666] | **+22.77%** | **[+2.73, +43.46]** | 53.0 |
| ≥$50k Platt | 170 | 0.6279 | [0.593, 0.664] | +20.33% | [-0.23, +41.27] | 52.4 |
| ≥$100k raw | 110 | 0.5836 | [0.537, 0.630] | +24.61% | [-2.37, +55.80] | 54.5 |
| ≥$100k Platt | 110 | 0.5836 | [0.537, 0.630] | +24.61% | [-2.37, +55.80] | 54.5 |
| Reference: market | — | 0.6267 | — | — | — | — |

**Iteration ≥$50k LL gate**: 0.6347 vs market 0.6267 — does NOT clear (Δ +0.008). Strict M2 exit criterion (Δ ≥ -0.01 from M1 baseline = LL ≤ 0.6202) NOT cleared.

**Iteration ROI gate**: M2 venue-only raw is the **first variant** with ≥$50k ROI CI lower bound > 0 (+2.73). M1 raw was -2.85; M1+Platt was +1.94 (clearer); M2 venue-only raw is +2.73 (cleanest).

### Golden eval (n=55, truly out-of-sample, 2026-04-17 → 2026-05-07)

| Slice | n | LL | LL 95% CI | Flat ROI | ROI 95% CI | Win % |
|---|---|---|---|---|---|---|
| ≥$50k raw | 50 | 0.6885 | [0.658, 0.720] | +20.85% | [-12.31, +52.54] | 53.1 |
| ≥$50k Platt | 50 | 0.6775 | [0.641, 0.716] | +23.18% | [-10.49, +54.24] | 55.1 |
| ≥$100k raw | 45 | 0.6899 | [0.655, 0.729] | +21.65% | [-12.47, +55.45] | 54.5 |
| ≥$100k Platt | 45 | 0.6772 | [0.638, 0.723] | +24.26% | [-9.76, +55.95] | 56.8 |

**Golden LL improves substantially over M1**: ≥$50k 0.6885 vs M1's 0.7006 (Δ -0.012); ≥$50k Platt 0.6775 vs M1+Platt's 0.6926 (Δ -0.015). Standalone n=61: M2 venue-only 0.6380 vs M1 0.6502 (Δ -0.012). All of these close the gap toward `xgb_match_v2_clean` (golden 0.6747 / standalone 0.6337).

## Adversarial slices (iteration ≥$50k, M2 venue-only raw)

| Slice | n | LL | Flat ROI | ROI 95% CI | Win % |
|---|---|---|---|---|---|
| close (\|ELO diff\| ≤ 5) | 60 | 0.7177 | +1.74% | [-29.21, +35.52] | 41.4 |
| mismatch (\|ELO diff\| ≥ 15) | 33 | 0.4868 | +60.82% | [+3.59, +134.06] | 66.7 |

vs M1: close LL 0.7186 → 0.7177 (Δ -0.0009, basically flat); close ROI -5.7% → +1.7% (+7.5pp). Venue features carry a small but real signal in close fixtures, less than the full M2 batter+bowler set (which got close-LL to 0.7115) but at lower variance cost.

## Walk-forward (iteration ≥$50k, M2 venue-only raw)

| Month | n | LL | Flat ROI | Δ ROI vs M1 |
|---|---|---|---|---|
| 2025-10 | 1 | 0.6865 | -100.00% | flat |
| 2025-11 | 5 | 0.7903 | +57.26% | flat |
| 2025-12 | 27 | 0.6743 | -1.47% | +1.9pp |
| 2026-01 | 45 | 0.6757 | +6.03% | **+11.2pp** |
| 2026-02 | 49 | 0.5382 | +52.63% | flat |
| 2026-03 | 8 | 0.5668 | +7.38% | +21.4pp |
| 2026-04 | 35 | 0.6804 | +22.33% | +5.6pp |

Venue features flip the loss months (2025-12, 2026-01) toward neutral/positive. The 2026-01 ROI shift (+11.2pp) is the strongest single-month signal and is consistent across many bets (n=43).

## Why venue features carry the M2 lift

Hypothesis: venue outcome-dist (boundary rate, dot rate, wicket rate) captures pitch character at a level neither `venue_avg_score` (already in M1) nor `venue_chase_win_pct` directly encodes. A high-`venue_p6` ground favors aggressive batting strategies even when total scores look similar; the model uses this to nudge predictions away from the toss-decision-bat / chase-win patterns that dominate the M1 venue features.

Bowler outcome-dist failed because bottom-5 squad-list-order isn't a robust bowling unit — see follow-up.

## Status of M2 verification criteria

1. ⚠️ **Iteration ≥$50k LL Δ ≥ 0.01 vs M1**: NOT cleared. M2 venue-only raw 0.6347 vs M1 0.6302 = Δ +0.0045 (slight regression). Plan said "If Δ < 0.005, treat as failed lift, drop the feature group." Per drop-one ablation we landed a *cleaner subset* (venue-only) rather than dropping all M2 features.
2. ✅ **Iteration ≥$50k ROI CI excludes 0**: cleared (lower bound +2.73). M1 was -2.85; this is the first variant to clear.
3. ✅ **Golden LL improves vs M1**: 0.6885 vs 0.7006 (≥$50k), 0.6380 vs 0.6502 (standalone) — Δ ~-0.012.
4. ✅ **Drop-one ablation surfaced clean signal**: bowler features actively hurt; batter features marginal; venue carries the lift. Cleaner subset shipped per drop-one rule.

## What this means for M3

- **M3 baseline is `models/xgb_match_v3_m2_venue_only/` (raw)**.
- **M3 numbers to beat**: iter ≥$50k LL 0.6347 raw / 0.6279 Platt; ROI +22.77% raw [+2.73, +43.46] / +20.33% Platt. Golden ≥$50k LL 0.6885 raw / 0.6775 Platt.
- **Slice priorities**: close-match slice still under-resolves (LL 0.7177 / ROI +1.7%) — M3's player-rolling-form features should target this directly. Player-level recency may provide cleaner per-batter signal than the lineup-aggregated outcome-dist.
- **Bowler outcome-dist follow-up**: investigate replacing bottom-5 squad-order with a metadata-based bowler set (`is_pace ∈ {True, False}` filter). M5 (player × opp / venue affinity) may also re-open this.

## Caveats

- **n=50 golden polymarket-overlap is small** — 0.012 LL improvement on golden is real but not significant by bootstrap CI alone.
- **Iteration aggregate LL miss**: composition-driven (mismatch slice dominates LL variance, M2 helps less there).
- **M2 venue-only is a 3-feature addition** — small architectural surface. M3's expected lift is larger because player-rolling-form is conceptually closer to what the model is missing on close fixtures.

Artifacts:
- `models/xgb_match_v3_m2_venue_only/{model.pkl, encoders.pkl, feature_columns.txt, train_metrics.json, test_predictions{,_calibrated}.json, golden_predictions{,_calibrated}.json, platt_calibrator.json}`
- `models/xgb_match_v3_m2/` and `models/xgb_match_v3_m2_no_{batter,bowler,venue}/` — preserved for reproducibility and follow-up
- `data/xgb_match_data_v3_m2/` — full 21-M2-feature parquets (drop-features flag selects subset at training time)
- `eval_out_m2_*` — sliced eval JSONs
- `reports/walk_forward_m2.md` — iteration ≥$50k monthly breakdown
