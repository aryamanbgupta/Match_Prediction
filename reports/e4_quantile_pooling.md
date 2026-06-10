# E4 — Quantile lineup pooling ❌ DISCARDED on the val rule (iteration favorable — forward-test candidate)

**Date**: 2026-06-09 · **Branch**: `improvement-experiments`
**Materializer**: `_quantile_elo_features` added to `materialize_match_features.py`
(stays in code, excluded from production training — same convention as M3/M4 features).
**Parquet**: `data/xgb_match_data_v3_e4_unfrozen` (unfrozen walk; parity check on the
48 existing features vs `v3_m3_unfrozen`: bit-identical, drift NONE).

## Hypothesis

M3/M4/M5 all failed because **mean**-pooled player features collapse to team
career aggregates. Quantile pooling (max / spread / best-k) preserves
within-lineup structure a mean destroys: one elite batter ≠ six average ones;
one strike bowler ≠ a flat attack. Specific target: the bowling unit, where
`bottom5_bowling_elo_avg` (squad-order proxy) is known-noisy since M2.

## Pre-training correlation check (the post-M4/M6 discipline)

12 candidates; dual condition = (redundancy: |r|>0.5 vs an existing feature
requires higher |target r| than that feature) AND (target floor |r| ≥ 0.03).

| feature | target r | max\|r\| vs prod | verdict |
|---|---:|---:|---|
| team1_top6_bat_elo_max | +0.091 | 0.883 (top6_avg) | FAIL redundancy |
| team1_top6_bat_elo_spread | +0.063 | 0.575 | FAIL redundancy |
| team2_top6_bat_elo_max | −0.043 | 0.887 | pass (marginal) |
| team2_top6_bat_elo_spread | −0.030 | 0.591 | FAIL redundancy |
| top6_bat_elo_max_diff | +0.138 | 0.758 (top6_diff r=0.174) | FAIL redundancy |
| top6_bat_elo_spread_diff | +0.080 | 0.384 | **pass** |
| team1/2_bowl_elo_max, _top2 | +0.06…+0.10 | 0.75–0.82 | **pass** (target r > bottom5's) |
| bowl_elo_max_diff | +0.143 | 0.613 | **pass** |
| bowl_elo_top2_diff | **+0.158** | 0.703 (bottom5_diff r=0.143) | **pass** |

The bowling quantiles are the real finding: `bowl_elo_top2_diff` carries more
target signal (0.158) than the feature it correlates with (0.143) — best-2
bowling ELO over the XI is a cleaner "strike bowler quality" measure than the
bottom-5-by-squad-order mean.

## Training (M7 config, seed 29, same env for all three)

| variant | features | val LL | standalone test LL |
|---|---:|---:|---:|
| base (re-trained prod set) | 48 | **0.6432** | 0.5941 |
| all8 (+8 survivors) | 56 | 0.6477 | **0.5897** |
| bowl6 (+6 bowling quantiles) | 54 | 0.6458 | 0.5938 |

**Pre-registered keep rule (M-phase): Δ val LL < −0.005 AND iteration ROI CI
not materially regressed. Both variants FAIL the val half** (+0.0045 / +0.0026
— val LL gets *worse*). → DISCARD.

## Iteration readout (transparency; NOT the decision basis)

| variant | slice | LL | flat ROI % | ROI CI |
|---|---|---:|---:|---|
| base | ≥$50k | 0.6312 | +15.38 | [−5.59, +38.13] |
| base | ≥$100k | 0.5884 | +25.25 | [−2.03, +58.12] |
| all8 | ≥$50k | 0.6288 | +24.35 | **[+3.62, +47.01]** |
| all8 | ≥$100k | 0.5880 | +32.92 | **[+6.19, +63.77]** |
| bowl6 | ≥$50k | 0.6316 | +19.27 | [−2.05, +40.92] |

all8 is directionally better than base on *every* iteration slice — but
landing it on that observation would be selecting on the readout set after
the val rule already said no. The conflict (val worse, iteration better, at
n=525 val both deltas within noise) is recorded as an open hypothesis:
**re-test all8 on forward-captured data (C2) when ≥50 fresh matches exist.**

## Side observation (consistent with E3)

The re-trained base — identical features and config to production, current
environment — reads weaker on iteration than the production artifact
(0.6312/+15.4 vs 0.6299/+21.9 @≥$50k). Second independent sighting of the
seed/env-luck component in the production headline quantified by E3.

## Artifacts

- `models/xgb_match_e4_{base,all8,bowl6}/`, `eval_out_e4/e4_summaries.json`
- Drop-lists generated programmatically (exact-name, substring-safety asserted)
