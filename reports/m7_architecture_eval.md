# M7 — architecture sweep (2026-05-10)

> **I3 statistical revision (2026-07-23):** tables below retain the original
> per-match i.i.d. intervals used for the historical model-selection decision.
> They are superseded for current economic claims. Whole-competition block
> intervals are M7 ≥$50k **[-10.79%, +50.18%]** and ≥$100k
> **[-17.36%, +46.42%]**; neither excludes zero. See
> `reports/i3_eval_statistics_hardening.md`.

Phase 7 of match-level v3. **Outcome: LANDED M7.A best_val as new production baseline.** First v3 M-phase since M2 to actually clear gates beyond what M2 venue-only unfrozen delivered. Pivot from feature engineering to architecture tuning was the right move after M3–M6 all dropped.

**New production baseline**: `models/xgb_match_v3_m7_production/`. Same 49 features as M2 v.o. (M1 + 3 venue outcome-dist); only the hyperparameters changed.

**Hyperparameter change**:

| Hparam | M2 v.o. baseline | M7.A winner |
|---|---|---|
| `max_depth` | 4 | 4 |
| `learning_rate` | 0.10 | **0.05** |
| `subsample` | 0.8 | 0.8 |
| `colsample_bytree` | 0.8 | **0.9** |
| `n_estimators` (with early stop) | best_iter ~60 | best_iter ~81 |
| `reg_alpha` | 0.1 | 0.1 |
| `reg_lambda` | 1.0 | 1.0 |

Lower learning rate + slightly higher colsample = a less aggressive fit. Old config was running early-stopped at ~60 rounds; new config trains longer (~80 rounds) with smaller steps.

## M7.A — Hyperparameter sweep

Grid: max_depth ∈ {3,4,5} × lr ∈ {0.03,0.05,0.1} × subsample ∈ {0.7,0.8,0.9} × colsample ∈ {0.7,0.8,0.9} = 81 configs. Sweep took 13s in `/tmp/claude/m7a_sweep.py`. Selected top candidates by val LL, then evaluated each on iteration ≥$50k.

Top candidates (val LL ranking):

| Config | val LL | test LL | iter ≥$50k LL | iter ≥$50k ROI [CI] | iter ≥$100k ROI [CI] |
|---|---|---|---|---|---|
| **baseline (md=4 lr=0.10 ss=0.8 cs=0.8)** | 0.6521 | 0.6015 | 0.6348 | +25.40% [+4.75, +48.11] | +26.21% [-0.93, +57.10] |
| **best_val (md=4 lr=0.05 ss=0.8 cs=0.9) ← LANDED** | **0.6459** | **0.5924** | **0.6299** | +21.90% [+2.28, +43.83] | **+26.39% [+0.57, +58.78]** |
| best_test (md=4 lr=0.05 ss=0.7 cs=0.7) | 0.6530 | 0.5879 | 0.6372 | +14.24% [-6.77, +36.68] | +17.18% |
| alt_md5 (md=5 lr=0.10 ss=0.7 cs=0.7) | 0.6465 | 0.5988 | 0.6459 | +21.96% [+2.43, +43.71] | +27.32% |
| alt_md3 (md=3 lr=0.05 ss=0.8 cs=0.7) | 0.6466 | 0.5978 | 0.6382 | +16.48% [-4.04, +39.13] | +18.75% |
| alt_balanced (md=4 lr=0.05 ss=0.8 cs=0.8) | 0.6473 | 0.5919 | 0.6356 | +11.97% [-9.05, +34.30] | +17.18% |

**Why best_val and not best_test**: best_test has the lowest standalone test LL but the WORST iteration ROI (CI lower bound -6.77). Classic over-fitting; standalone test LL alone is misleading. Per the iteration-only-decisions discipline, iter ≥$50k ROI / LL drives selection.

**best_val improvements vs baseline**:
- iter ≥$50k LL: 0.6348 → **0.6299** (Δ -0.005, closer to market 0.6267)
- iter ≥$100k LL: 0.6006 → **0.5929** (Δ -0.008)
- iter ≥$100k ROI CI lower bound: -0.93 → **+0.57** (now cleanly excludes 0)
- iter ≥$50k ROI: +25.40% → +21.90% (CI overlap, no material regression)

## Adversarial slice deep-dive (the headline result)

The historically weak **close-match slice** (top6 ELO diff ≤ 5) finally clears the ROI gate:

| Slice | M2 v.o. baseline | M7 production | Δ |
|---|---|---|---|
| close (n=74) LL | 0.6982 | 0.6880 | -0.010 |
| close ROI [CI] | +26.12% [**-2.34**, +52.30] | **+33.27% [+4.36, +61.53]** | **+7.2pp; CI now excludes 0** |
| close win % | 52.7% | **56.8%** | +4.1pp |
| mismatch (n=24) LL | 0.3731 | 0.3565 | -0.017 |
| IPL slice (n=22) LL | 0.6770 | 0.6709 | -0.006 |

This is the slice that's stayed weak across M1 (LL 0.7186 / ROI -5.7%), M2 v.o. frozen (0.7177 / +1.7%), M2 v.o. unfrozen (0.6982 / +26.12%). M7 production gets it to LL 0.6880 / ROI +33.27% with positive ROI CI lower bound.

## Walk-forward (iteration ≥$50k, M7 production raw)

| Month | n | LL | Flat ROI [CI] | Win % |
|---|---|---|---|---|
| 2025-10 | 1 | 0.6468 | -100% | 0 |
| 2025-11 | 5 | 0.8872 | +57.26% [-100, +229] | 40 |
| 2025-12 | 27 | 0.7080 | -34.70% [-75, +10] | 25.9 |
| 2026-01 | 45 | 0.6676 | +13.14% [-24, +49] | 48.8 |
| 2026-02 | 49 | 0.5159 | +58.20% [+9.88, +118.31] | 63.3 |
| 2026-03 | 8 | 0.5874 | -25.95% [-79, +45] | 37.5 |
| 2026-04 | 35 | 0.6554 | **+34.87% [+2.04, +68.06]** | **65.7** |

2026-04 IPL period: +34.87% ROI with CI lower bound +2.04, win rate 65.7%. Strongest in-IPL single-month signal yet. 2025-12 internationals still weak (LL 0.708, ROI -35%), consistent with prior baselines.

## Platt calibration: kills iteration ROI

Calibration via Platt LOOCV improves LL but degrades iteration ROI:
- raw iter ≥$50k: LL 0.6299, ROI +21.90% [+2.28, +43.83]
- Platt iter ≥$50k: LL **0.6223** (better), ROI **+13.54% [-7.29, +34.94]** (CI now includes 0)

Lower-lr training produces predictions that are already closer to calibrated; Platt over-corrects. **Production uses RAW probabilities** (the `model.pkl`'s direct output). Platt is preserved as `platt_calibrator.json` for sizing-tool use only.

## M7.B — Per-tier specialization (explored, not landed)

Trained an IPL-only model (md=4 lr=0.05 ss=0.8 cs=0.9) on the 1076 IPL train matches. Hybrid prediction: IPL-only model on IPL matches, M7.A best_val on others.

| Slice | M7.A best_val | M7.B hybrid | Δ |
|---|---|---|---|
| iter ≥$50k LL | 0.6299 | 0.6308 | +0.001 (tie) |
| iter ≥$50k ROI | +21.90% [+2.28, +43.83] | +24.94% [+4.40, +46.93] | +3.0pp |
| iter ≥$100k ROI | +26.39% [+0.57, +58.78] | +31.04% [+4.04, +62.40] | +4.7pp |
| IPL slice (n=22) ROI | +12.79% [-30, +55] | +36.03% [-8, +77] | +23pp (but CI wide) |

Hybrid is a marginal Pareto improvement on aggregate iteration ROI. But the IPL slice CI is wide (n=22), and the aggregate gain is driven entirely by those 22 IPL matches. The +23pp IPL-slice lift is within noise (CI [-8, +77]). **Decision: not landing M7.B** — the modest aggregate gain doesn't justify the inference-time complexity of two models + hybrid logic. Documented for future re-evaluation when IPL test sample grows.

## M7.C — Stacking with disjoint feature subsets (skipped)

After M7.A landed a clean win and M7.B was marginal, stacking is unlikely to outperform. Skipped to keep the production architecture simple. Revisit if a future M-phase shows a clear case (e.g., two new feature groups with complementary signal — but the feature frontier is exhausted per M3–M6).

## Status of M7 verification criteria

1. ✅ **Pick best architecture+features combination as new production reference**: M7.A best_val (md=4 lr=0.05 ss=0.8 cs=0.9). Production model artifact: `models/xgb_match_v3_m7_production/`.
2. ✅ **Iteration ≥$50k LL improvement**: Δ -0.005 vs M2 v.o.
3. ✅ **Iteration ≥$100k ROI CI now excludes 0** (+0.57 lower bound vs M2 v.o.'s -0.93).
4. ✅ **Close-match slice ROI CI excludes 0** for the first time on this slice (+4.36 lower bound vs M2 v.o.'s -2.34).
5. ✅ **Production fixture predictor updated**: `predict_fixture.py` now points at `xgb_match_v3_m7_production/`.

## What this means for production deployment

- **Production model**: `models/xgb_match_v3_m7_production/`
- **Prediction path**: raw probabilities (no calibration). Platt calibrator stored for sizing-only use if downstream Kelly logic wants honest probabilities.
- **predict_fixture.py**: switched to new model. First-time use will reuse the existing `data/tracker_snapshot_test_end.pkl` (compatible with M2 v.o. features).
- **Trainer defaults**: `xgboost_match_v1.py` defaults updated to the M7.A winner (lr=0.05, cs=0.9, n_estimators=1000 + early stopping). Old defaults preserved via the artifact directories.

## Next steps (post-M7)

- **C2 forward polymarket capture** (in parallel since M1) — continue accumulating fresh pre-match snapshots through 2026-05+. After 30-60 days, validate the M7 production model on truly fresh data.
- **M8 sizing rules** (E1 + E2 in catalog): edge-threshold + fractional Kelly + per-bet outlier cap.
- **Golden eval (audit-only)**: not re-run for M-phase selection. Will be the final yes/no confirmation at production-launch time, not earlier.
- **Architecture follow-ups deferred**: M7.C stacking, B6 LightGBM/CatBoost.

## Artifacts

- `models/xgb_match_v3_m7_production/{model.pkl, encoders.pkl, feature_columns.txt, train_metrics.json, test_predictions.json, test_predictions_calibrated.json, golden_predictions.json, golden_predictions_calibrated.json, platt_calibrator.json}` — new production
- `models/xgb_match_v3_m7a_*/` — sweep candidates preserved for reproducibility
- `models/xgb_match_v3_m7b_ipl_only/` — IPL specialist + hybrid predictions
- `models/xgb_match_v3_m7_sweep/sweep_results.csv` — full 81-config sweep table + top_candidates_eval.csv
- `reports/walk_forward_m7.md` — monthly iteration breakdown

## Headline (one-line)

Lower learning rate (0.10 → 0.05) + slightly more column subsampling (0.8 → 0.9) closed the M2 baseline's residual fit-gap. Close-match slice ROI CI cleanly excludes 0 for the first time, and ≥$100k ROI CI also clears.
