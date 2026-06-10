# E1 — Temperature sharpening of the match-level model ❌ DISCARDED (LL-only win)

**Date**: 2026-06-09 · **Branch**: `improvement-experiments`
**Model**: `models/xgb_match_v3_m7_production` (untouched) · **Harness**: `scripts/e1_temperature_sharpen.py`

## Hypothesis

The reliability diagnostic (2026-06-07, `reports/reliability_diagnostic.png`)
showed the production model is **under-confident** (calibration slope
1.34–1.75) with **market-level Brier resolution** on the iteration set.
Unlike the Platt-toward-50 corrections that failed in 2026-03/2026-04, an
*expansive* transform (T > 1 on the logit) should improve LL without the
shrink-to-coinflip failure mode — possibly clearing the strict iteration
≥$50k LL gate (market 0.6267) for the first time.

## Protocol

- Fit on **val only** (n=525, scored fresh from the production booster).
- Four candidates: temperature (free T), temperature (T≥1), Platt
  (reference), beta calibration.
- Apply to `test_predictions.json`; evaluate via the standard
  `blend_eval_json (w=0)` → `reslice_eval_json` pipeline (identical to the
  M7 baseline numbers; raw re-run through the same pipeline reproduces the
  documented 0.6299 / +21.90% exactly).
- Keep iff iteration ≥$50k LL improves AND flat-ROI CI does not materially
  regress (M-phase discipline).

## Val fits

| variant | params | val LL (raw 0.6459) |
|---|---|---|
| temp | T=1.205 | 0.6448 |
| temp_ge1 | T=1.205 (unclamped anyway) | 0.6448 |
| platt | a=1.200, b=−0.098 | 0.6437 |
| beta | a=2.519, b=0.095, c=1.648 | 0.6412 |

Note: val ΔLL for temp is −0.0011 — already **below** the −0.005 keep
floor used throughout the M-phases. The iteration readout below confirms
the discipline would have been right.

## Iteration-test results (the readout)

| variant | slice | n | LL | flat ROI % | ROI 95% CI |
|---|---|---:|---:|---:|---|
| **raw** | all | 261 | 0.6254 | +15.46 | [−4.62, +42.12] |
| **raw** | ≥$50k | 170 | **0.6299** | **+21.90** | **[+2.28, +43.83]** |
| **raw** | ≥$100k | 110 | 0.5929 | +26.39 | [+0.57, +58.78] |
| temp | all | 261 | 0.6192 | +13.51 | [−5.37, +39.93] |
| temp | ≥$50k | 170 | **0.6246** ✅ < market 0.6267 | +12.28 | [−6.69, +33.62] ❌ |
| temp | ≥$100k | 110 | 0.5826 | +19.64 | [−5.75, +51.30] ❌ |
| platt | ≥$50k | 170 | 0.6223 | +13.54 | [−7.29, +34.94] ❌ |
| platt | ≥$100k | 110 | 0.5818 | +24.52 | [−0.91, +55.72] ❌ |
| beta | ≥$50k | 170 | 0.6294 | +10.27 | [−6.06, +28.58] ❌ |

## Verdict: ❌ not production. LL gate finally clears; ROI co-gate breaks.

Temperature T=1.21 is the **first variant to beat the market on iteration
≥$50k LL with a pure post-hoc transform** (0.6246 < 0.6267, −0.005 vs
raw). But every transform costs ~8–12pp flat ROI and pushes every ROI CI
back across zero. Both production gates are required; raw stays.

## Why sharpening hurts flat ROI (mechanism, durable insight)

Flat betting picks the side where model prob > market prob. A monotone
expansive transform flips the bet side exactly on matches where the
market price lies **between the raw and sharpened probability** — i.e.,
matches where the raw model quietly disagreed with the market *toward*
50%. Those timid fades of the market favorite were disproportionately
**right** (they carry the model's counter-market signal); sharpening
converts them into bets *with* the market favorite and destroys that
profit pocket. LL, by contrast, rewards confident agreement with
outcomes regardless of the market — so LL improves while ROI degrades.

**Generalisation of the project's calibration lesson**: it is not
"shrinking toward 50% kills ROI" (2026-03/04 framing) — it is *any*
post-hoc monotone recalibration that crosses market prices on the
near-coinflip mass kills flat ROI. The raw probability's *side* relative
to the market is the alpha; its distance from 0.5 is mis-scaled for LL
but correctly scaled for side-selection.

## Legitimate residual use (mirrors M1's "calibration as sizing tool")

`temp` (T=1.205) is the best LL-per-parameter probability-quality layer
measured so far (beats Platt's 2 params on iteration LL parity with one
param, and beta's val win does not transfer). If a consumer needs honest
*probabilities* (Kelly sizing, scenario analytics, win-prob displays) the
temperature layer is preferable to raw or Platt. Not wired into
production; artifacts live in `eval_out_e1/`.

## Artifacts

- `scripts/e1_temperature_sharpen.py` (re-runnable end to end)
- `eval_out_e1/preds/*.json`, `eval_out_e1/sliced_*/`, `eval_out_e1/e1_summaries.json`
