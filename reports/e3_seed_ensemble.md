# E3 — Seed ensemble for the match-level model ❌ DISCARDED (val + iteration both regress)

**Date**: 2026-06-09 · **Branch**: `improvement-experiments`
**Harness**: `scripts/e3_seed_ensemble.py` · Production model untouched.

## Hypothesis

10-seed averaging of the exact M7 production config (depth 4, lr 0.05,
subsample 0.8, colsample 0.9, early stop on val) is pure variance
reduction and typically buys 0.003–0.008 LL on tabular problems.

## Result: ensemble is WORSE on val and on iteration

Per-seed (production encoders + feature list reused verbatim, trained on
`data/xgb_match_data_v3_m3_unfrozen`):

| seed | best_iter | val LL | standalone test LL |
|---|---:|---:|---:|
| **29 (prod seed)** | 81 | **0.6432** | 0.5941 |
| 7 | 92 | 0.6484 | 0.5913 |
| 42 | 71 | 0.6506 | 0.5992 |
| 101 | 56 | 0.6477 | 0.6044 |
| 271 | 91 | 0.6460 | 0.5978 |
| 314 | 56 | 0.6510 | 0.6027 |
| 555 | 63 | 0.6501 | 0.6017 |
| 1337 | 84 | 0.6488 | 0.5952 |
| 2026 | 56 | 0.6475 | 0.6051 |
| 90210 | 70 | 0.6467 | 0.5996 |

Combiners (selection on val only): prob-mean val LL 0.6469, logit-mean
0.6468 — both **worse than single seed 29's 0.6432** → discard at the
val stage already (Δ +0.0036). Iteration readout confirms:

| variant | slice | n | LL | flat ROI % | ROI 95% CI |
|---|---|---:|---:|---:|---|
| raw single (prod) | ≥$50k | 170 | 0.6299 | +21.90 | [+2.28, +43.83] |
| raw single (prod) | ≥$100k | 110 | 0.5929 | +26.39 | [+0.57, +58.78] |
| ens logit-mean | ≥$50k | 170 | 0.6397 | +16.03 | [−4.26, +38.53] |
| ens logit-mean | ≥$100k | 110 | 0.6028 | +20.19 | [−7.21, +52.76] |

## Why — and the durable insight

Seed 29 is the **best val seed of all 10** (0.6432 vs pack 0.646–0.651).
The M7 sweep selected its winning config *by val LL with seed fixed at
29*, so part of the production model's val (and, it turns out,
iteration) advantage is favorable seed variance, not config quality. An
ensemble is a lower-variance estimator of the config's true skill — and
it reads ~0.01 LL and ~6pp ROI worse than the production point estimate.

**Implication for forward expectations**: the honest central estimate of
the production config's skill on fresh data is closer to the ensemble
reading (iteration ≥$50k LL ≈ 0.64, ROI ≈ +16%) than to the headline
(0.6299 / +21.9%). The forward test (C2) should be judged against the
tempered number, not the headline.

**Why we still keep the single model in production**: the gates were
always point-estimates on the fixed iteration set; the production
artifact IS seed 29's draw, and replacing it with the ensemble would
make both gate metrics worse on the set we measure. But quoting the
headline as the expected forward edge would be self-deception — this
experiment quantifies the gap.

## Reproduction caveat

Retraining seed 29 today gives val LL 0.6432 / test LL 0.5941 vs the
production artifact's recorded 0.6459 / 0.5924 — close but not
bit-exact. Environment drift since 2026-05-10 (xgboost version or
parquet re-materialization). The iteration comparison above uses the
*actual* production `test_predictions.json` for the baseline row, so the
discard decision is unaffected.

## Artifacts

- `scripts/e3_seed_ensemble.py`
- `eval_out_e3/` (per-variant sliced JSONs + `e3_summaries.json`)
