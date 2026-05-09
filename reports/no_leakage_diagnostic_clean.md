# No-leakage diagnostic re-run on clean model (2026-05-09)

**TL;DR**: Re-ran the frozen-vs-unfrozen comparison using the post-fix
`xgb_match_v2_clean` model. The "frozen is BETTER than unfrozen" finding
**survives** on the polymarket-sliced metrics, but the magnitude shrinks
roughly by half. On the standalone full 782-match test, unfrozen is now
slightly better. Tracker contamination is still NOT the dominant signal
either way; the original composition-effect explanation (late test has
more T20 WC mismatches) holds. The fix did not flip the diagnostic — it
just compressed it.

## Setup

* Model: `models/xgb_match_v2_clean/model.pkl` (post-leakage-fix, retrained
  on snapshot-protected ELO features).
* Frozen test parquet: `data/xgb_match_data_v2_clean/test.parquet`
  (built with `--freeze-trackers-after 2025-06-30`).
* Unfrozen test parquet: `data/xgb_match_data_v2_clean_unfrozen/test.parquet`
  (newly materialized 2026-05-09, no freeze flag, otherwise identical
  source/sqlite/metadata config).
* Train + val rows are bit-identical between frozen and unfrozen builds,
  so the model is a fixed reference; only the test feature distribution
  differs between variants.
* Predictions: re-ran `xgboost_match_v1.py --cmd predict-test` on the
  unfrozen parquet using the same model artifacts, written to
  `models/xgb_match_v2_clean_unfrozen/test_predictions.json`.
* Eval: `blend_eval_json.py --w 0.0` (direct alone) → `reslice_eval_json.py`
  for all / ≥$50k / ≥$100k slices against `betting_odds_polymarket.json`.

## Headline — polymarket-overlap eval

| Variant | Slice | n | LL | LL 95% CI | Flat ROI | ROI 95% CI | Win % |
|---|---|---|---|---|---|---|---|
| Frozen | all | 261 | **0.6271** | [0.602, 0.651] | **+17.68%** | [-3.67, +44.20] | 48.6 |
| Frozen | ≥$50k | 170 | **0.6339** | [0.602, 0.665] | **+22.63%** | [+2.42, +45.04] | 52.4 |
| Frozen | ≥$100k | 110 | **0.5877** | [0.546, 0.628] | +25.31% | [-2.66, +57.71] | 53.6 |
| Unfrozen | all | 261 | 0.6409 | [0.616, 0.666] | +9.07% | [-11.36, +34.80] | 44.3 |
| Unfrozen | ≥$50k | 170 | 0.6437 | [0.613, 0.675] | +17.13% | [-3.03, +40.02] | 49.4 |
| Unfrozen | ≥$100k | 110 | 0.6036 | [0.560, 0.645] | **+27.53%** | [+1.90, +58.72] | 54.5 |
| Reference: market | — | — | 0.6267 | — | — | — | — |

* **LL**: frozen wins on every slice (Δ −0.014, −0.010, −0.016).
* **Flat ROI**: frozen wins on `all` and `≥$50k`; unfrozen narrowly wins
  on `≥$100k` (+27.53 vs +25.31). All ROI CIs except `≥$50k frozen` and
  `≥$100k unfrozen` straddle zero.
* **Win rate**: frozen +4.3pp (all), +3.0pp (≥$50k); unfrozen +0.9pp (≥$100k).

The frozen-better-than-unfrozen pattern persists. The headline numbers
are the same shape as the leaky diagnostic, just compressed.

## Compression vs the leaky diagnostic

Same comparison from the original (leaky) diagnostic for reference:

| Slice | Leaky frozen LL | Clean frozen LL | Δ | Leaky unfrozen LL | Clean unfrozen LL | Δ |
|---|---|---|---|---|---|---|
| all | 0.4944 | 0.6271 | +0.133 | 0.5226 | 0.6409 | +0.118 |
| ≥$50k | 0.5004 | 0.6339 | +0.134 | 0.5135 | 0.6437 | +0.130 |
| ≥$100k | 0.4361 | 0.5877 | +0.152 | 0.4554 | 0.6036 | +0.148 |

The leakage fix lifted LL by ~0.13–0.15 across the board — substantially
more than the frozen-vs-unfrozen gap itself. Most of the previous edge
was leakage; the residual frozen-vs-unfrozen gap is real but small
(~0.01–0.02 LL).

## Standalone full-test reverses

On the full 782-match standalone test (no polymarket filter):

| Variant | n | LL | Brier |
|---|---|---|---|
| Frozen | 782 | 0.6180 | 0.2154 |
| Unfrozen | 782 | **0.6027** | **0.2085** |

Unfrozen is BETTER on the full standalone set. So the frozen advantage is
specific to the polymarket-overlap subset, which skews toward
international/IPL/tournament matches. On routine domestic-league fixtures
not covered by polymarket markets, the within-test tracker updates appear
to *help* — likely because they keep the trackers' state semantics closer
to what the model trained on.

## Temporal split — composition effect still holds

| Variant | Early (124) | Late (131) | Late−Early |
|---|---|---|---|
| Frozen | LL 0.6604 | LL 0.5956 | -0.065 |
| Unfrozen | LL 0.6789 | LL 0.6050 | -0.074 |

Late beats early in both variants by ~0.07 LL. Same composition driver as
the original diagnostic (T20 World Cup qualifying mismatches concentrate
in late test, 47/131; the model exploits its confidence on lopsided
fixtures, and bookmakers/markets price them similarly lopsided). The gap
shape is unchanged from the leaky diagnostic; only the absolute LL levels
shifted up.

## Interpretation

1. **Tracker contamination is not the major issue either way.** Frozen
   protects against within-test cross-match contamination, but the
   protection is worth ~0.01–0.02 LL on polymarket-overlap and slightly
   *negative* on the broader test set. The leakage fix removed the
   dominant feature drift, exposing the underlying small effect.

2. **Choice between frozen and unfrozen for production**: frozen is
   marginally safer on the high-liquidity slices we care about for
   betting; unfrozen is closer to real-world deployment semantics
   (trackers keep evolving as new matches resolve) and wins on the broad
   test set. For the iteration eval headline, frozen is the more
   conservative, less-leak-vulnerable framing — keep using it. For an
   actual live deployment, unfrozen / per-match-fresh rehydration is the
   correct semantics.

3. **The previous "frozen is BETTER" claim does not retract** — it just
   shrinks. The composition-effect explanation for the early/late ROI
   gap (T20 WC mismatches in late test) still holds and is the dominant
   driver.

4. **No additional leakage uncovered.** The frozen-vs-unfrozen LL
   compression aligns with what the leakage fix should have produced. If
   the diagnostic had flipped sharply (say frozen now much worse), it
   would suggest the freeze flag was masking a different bug. It didn't
   flip — the compression is consistent with the explanation that the
   leakage fix removed the dominant drift, leaving a small-and-real
   tracker-state effect that's mostly washing out.

## Status of the open follow-up

`reports/leakage_fix_comparison.md` listed:

> Re-run no-leakage diagnostic with the clean model — the previous
> "frozen is BETTER than unfrozen" finding may flip or change shape now
> that the dominant feature drift is removed

Outcome: **shape mostly preserved, magnitude compressed**. Frozen retains
the LL lead on polymarket-overlap; ROI lead persists on `all` and `≥$50k`
slices but flips narrowly on `≥$100k`. The shift is consistent with the
post-fix model being borderline-skilful rather than artificially
dominant. No new diagnostic action recommended.
