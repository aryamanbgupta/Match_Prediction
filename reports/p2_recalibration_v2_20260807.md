# P2 re-look: recalibration on v2 prices (2026-08-07)

Closes the P2 open question from the 2026-08-05 market-benchmark correction
(TODO § "Market-benchmark correction follow-ups"): Platt (M7-era) and E1
temperature sharpening were rejected for "costing ROI", but that was measured
on the corrupt ROI surface, where the fake edges on toss-priced coin-flip
fixtures were exactly what recalibration destroyed. This re-look asks one
narrow question: **does the rejection survive on honest (v2) prices?** It is
explicitly NOT a claim that calibration helps — the repo has twice
established that calibration cannot create resolution, and the model's LL
deficit to the market is a resolution problem.

**This section (inputs, questions, decision rules) was written and committed
BEFORE any calibrator was fit.** Results below are filled in afterward.

## Inputs

- **Model**: `models/xgb_match_i7_swap_production` (production of record),
  raw `test_predictions.json` on the iteration set. Golden is audit-only and
  is not scored here (iteration-only decision discipline).
- **Calibrators**: Platt (primary — the exact method rejected at M7:
  "over-corrects on this config and kills iteration ROI"), fit by
  `scripts/calibrate_match_predictions.py` on the i7 frame's
  `validation.parquet` scored fresh with the production booster — fit on
  val, applied to test, no test leakage. Isotonic as sensitivity (expected
  to regress LL at this val size, per M1). The fitted Platt (a, b) are
  reported: a > 1 means the calibrator *sharpens* (E1's direction), a < 1
  means it flattens toward 0.5. The calibrated JSONs live under
  `eval_out/p2_recalibration/` — nothing is added to the production dir.
- **Eval pipeline**: identical to the audit's raw baseline —
  `blend_eval_json.py --w 0.0` against
  `eval_out/toss_defect_20260805/envN_i7_corrected.json`, then
  `reslice_eval_json.py --odds betting_odds_polymarket_v2.json
  --cluster-source-dir data/polymarket_test_v2`, slices all / ≥$50k /
  ≥$100k. Raw baseline (already published, reproduced by the same pipeline):
  all **0.6180 / +5.55%** [−15.39, +37.55]; ≥$50k **0.6249 / +3.38%**
  [−14.63, +37.06]; ≥$100k **0.5886 / −5.19%** [−28.73, +27.50].
- **Paired statistics**: per-match ΔLL (calibrated − raw) and paired
  per-match pnl delta (a fixture where one arm places no bet contributes 0
  for that arm), cluster-bootstrapped under the I3 contract (whole-event
  resamples, seed 42, 10,000). Note Platt is monotone, so probability
  *ranking* is preserved, but the bet side vs the market price can flip on
  fixtures where the model sits near the market — slice ROIs are therefore
  each arm's own bet set, and the paired delta is the like-for-like number.

## Pre-committed questions and decision rules

- **Q1 (LL)**: does Platt improve iteration LL? Verdict per slice: improved
  iff the paired ΔLL CI excludes 0 below; the 0.007 seed-noise floor does
  not apply (no retraining — fixed predictions, deterministic transform),
  so the CI is the only bar.
- **Q2 (ROI — the actual open question)**: the M7-era rejection ("Platt
  kills iteration ROI") **SURVIVES** on v2 only if calibrated flat ROI is
  worse than raw by more than 2.3pp on the ≥$50k slice AND the paired
  pnl-delta CI excludes 0 in the unfavorable direction. It is recorded as
  **UNSUPPORTED** if the deltas are within noise or favorable — phrased as
  "the rejection's evidence does not survive", never as "calibration
  helps".
- **Outcome mapping — no production change in any branch.** Serving stays
  raw regardless: the model does not beat the market on LL anywhere, no
  betting is authorized, and nothing downstream consumes a calibrated
  probability. The deliverable is the answer to the open question plus doc
  updates (CLAUDE.md's open-question sentence, TODO P2, the
  calibration-priority memory note).

## Results

**Fit diagnostics.** Platt on i7 val (n=528): **a = 1.107, b = −0.073**
(`eval_out/p2_recalibration/platt_calibrator.json`). a > 1 — the fitted
transform mildly **sharpens** this model's probabilities; it does not
squeeze them toward 0.5. The "calibration flattens an under-resolving
model" mechanism behind the historical warning does not apply to this
artifact. Val LOOCV LL moves 0.6504 → 0.6531 (slightly worse — the
transform is close to identity and not clearly beneficial even on its own
training distribution); full-798 standalone test LL 0.5859 → 0.5810
(−0.0049). Isotonic regresses the full test (+0.0045), exactly as M1
predicted at this val size.

### Slice metrics (v2 odds, I3 blocks)

| slice | raw LL | Platt LL | raw ROI | Platt ROI | raw win% | Platt win% |
|---|---|---|---|---|---|---|
| all (252) | 0.6180 | **0.6119** | +5.55 [−15.4,+37.6] | **+7.97** [−9.0,+35.5] | 42.5 | 45.2 |
| ≥$50k (167) | 0.6249 | **0.6191** | +3.38 [−14.6,+37.1] | **+7.07** [−6.0,+30.5] | 42.5 | 45.5 |
| ≥$100k (110) | 0.5886 | **0.5817** | −5.19 [−28.7,+27.5] | **+1.74** [−18.8,+26.1] | 38.2 | 42.7 |

Isotonic (sensitivity): LL 0.6159/0.6199/0.5823, ROI +5.06/+2.13/−8.00 —
dominated by Platt on every slice; the M1 "isotonic needs a bigger val"
finding stands.

### Paired deltas (Platt − raw, cluster bootstrap, seed 42 × 10,000)

| slice | ΔLL mean [CI] | Δpnl (pp ROI-equiv) [CI] | bet side-flips |
|---|---|---|---|
| all | −0.0061 [−0.0111, **+0.0003**] | +2.42 [−8.59, +9.09] | 27/252 |
| ≥$50k | −0.0058 [−0.0097, +0.0028] | +3.69 [−9.32, +11.08] | 17/167 |
| ≥$100k | −0.0069 [−0.0123, +0.0021] | +6.93 [−4.20, +13.07] | 7/110 |

### Verdicts (pre-committed rules)

- **Q1 (LL): NOT improved CI-clean.** All three ΔLL CIs straddle zero —
  favorable direction on every slice (and the all-slice upper bound sits
  at +0.0003, the boundary), but nothing clears. Platt is
  LL-neutral-to-slightly-favorable on this artifact, and in particular it
  does **not** over-correct: the M7-era "over-corrects on this config"
  characterization does not reproduce on the i7 swap production model.
- **Q2 (ROI): the rejection is UNSUPPORTED — and its evidence inverts.**
  The survival condition (calibrated ROI worse by >2.3pp at ≥$50k with an
  unfavorable CI-clean paired delta) fails in the strongest possible way:
  the point deltas are *favorable* on every slice (+3.69pp at ≥$50k,
  +6.93pp at ≥$100k, win rate +3.0pp) with CIs straddling zero. "Platt
  kills iteration ROI" was a property of the corrupt ROI surface: the
  toss-priced coin-flip fixtures manufactured fake edges that any
  probability shrinkage toward the (fake) market price destroyed. On
  honest prices the same transform is ROI-neutral with a favorable lean.

### What this does and does not change

- **No production change** (per the pre-committed outcome mapping).
  Serving stays raw: the model still loses to the market on LL on every
  slice (0.6191 vs 0.5940 at ≥$50k even calibrated — calibration cannot
  create resolution), no interval anywhere excludes zero, and no betting
  is authorized.
- The **settled-negative status of recalibration is rescinded**: the two
  historical rejections (M7 Platt, E1 temperature) cited evidence that
  does not survive the benchmark correction, and the re-look's point
  estimates lean the other way. Future model-selection work may treat a
  val-fit Platt layer as a legitimate candidate again — subject to the
  normal gates, and noting that on this artifact the fitted transform
  *sharpens* (a = 1.107), so the old "squeezes toward 50%" objection is
  moot.
- Any positive claim ("calibration helps ROI") would need its own
  pre-registered forward evidence; nothing here establishes it.

Artifacts: `eval_out/p2_recalibration/` (calibrators, calibrated
predictions, blend + sliced JSONs; regenerable). Paired tooling:
`scripts/auto/p2_recal_paired.py`.
