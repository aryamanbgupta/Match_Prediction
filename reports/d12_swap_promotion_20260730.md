# D12 swap-augmentation production promotion — 2026-07-30

## Decision

`models/xgb_match_v3_m7_swap_production` is the new production winner-market
model. It is the **archived D12 swap arm promoted verbatim** — not a fresh
retrain — because the current trainer correctly refuses to train new models
on pre-I7 legacy-identity frames, and the production line *is* the pre-I7
legacy line. The artifact was trained 2026-07-17 by the D12 harness at
commit `8a03cd9` on the exact production 48-feature frame
(`data/auto/d12`, m3-unfrozen subset), seed 29, `--monotone
--swap-augment`, trainer defaults (= M7 config). `predict_fixture.py` now
defaults to it, still under the legacy venue-identity serving contract,
exactly as `xgb_match_v3_m7_production` was served. The frozen M7 dir is
retained unchanged for reference and rollback.

## Control

The D12 base sibling (`models/auto/d12/base_seed29`) reproduces
`models/xgb_match_v3_m7_production/test_predictions.json` with
**max |Δp| = 0.0 over all 782 matches** (re-verified 2026-07-30), so the
swap arm differs from production by exactly one change: train-time team-swap
symmetry augmentation (train 7,912 → 15,824 rows, base rate exactly 0.5).

## Evidence

Primary evidence is D12's paired 5-seed result on this exact frame
(LANDED 2026-07-17, `research/reports/auto/D12.md`): iteration ≥$50k
ΔLL −0.0092 (better 5/5 seeds, floor 0.007) and ΔROI +3.39pp (up 5/5
seeds, floor 2.3), consistent at ≥$100k (ΔLL −0.0123), ROI seed-std
halved.

Seed-29 readouts recomputed 2026-07-30 under the I3
`tournament_time_block_v1` contract (D12's original CIs were pre-I3
i.i.d. and are superseded):

| Iteration slice | swap LL | base LL | market LL (slice-matched) | swap ROI | swap ROI block CI | base ROI | blocks |
|---|---:|---:|---:|---:|---|---:|---:|
| all (255 scored) | **0.6178** | 0.6254 | 0.6267 | +19.67% | [−4.44, +45.80] | +15.46% | 25 |
| ≥$50k (168 bets) | **0.6215** | 0.6299 | 0.6482 | +24.53% | [−1.98, +46.37] | +21.90% | 19 |
| ≥$100k (110 bets) | **0.5796** | 0.5929 | 0.6224 | +26.60% | [−17.21, +45.42] | +26.39% | 11 |

Note the frozen 0.6267 constant is the **all-261 market LL**; the
slice-matched ≥$50k market LL is 0.6482. On matched slices the swap model
beats the market everywhere. Block ROI CIs straddle zero on every slice —
for base as well — which is the honest I3 state of the iteration set; the
adoption case rests on the paired 5/5-seed deltas, not on a CI-clean ROI
claim.

Golden audit (out-of-sample, descriptive only at 5–6 blocks):

| Golden slice | swap LL | base LL | market LL | swap ROI | base ROI | swap win | base win |
|---|---:|---:|---:|---:|---:|---:|---:|
| all (54 bets) | **0.6576** | 0.6680 | 0.6617 | +12.52% | −0.64% | 0.537 | 0.463 |
| ≥$50k (49 bets) | **0.7009** | 0.7078 | 0.7085 | +19.24% | +7.15% | 0.551 | 0.490 |

Standalone golden (62-match m3-unfrozen parquet): LL 0.6430 vs 0.6611,
Brier 0.2282 vs 0.2364. Direction confirmed on every golden metric; the
swap model beats the matched market LL on both golden slices where base
does not.

## Pipeline fix made during this evaluation

`blend_eval_json._persist_bet_contract` stamped a **team-pair fallback**
`competition_cluster_id` on every blended row; downstream reslice treats an
explicit id as authoritative, silently replacing the I3 event-time blocks
with near-per-match clusters (observed: 134 "blocks" at ≥$50k instead of
19, ROI CI narrowed to [+0.98, +44.22] from the true [−10.48, +49.94]).
The stamping is removed; blend output now leaves the field absent unless
the source carried a real one, and the D10 characterization pins were
updated to the corrected contract. Any direct-model ROI CI computed
through the current-code blend path between I3 (2026-07-23) and this fix
understates block correlation and should be recomputed if quoted.

## Economic status

Unchanged from I3/forward: probability improvement is the confirmed part;
no CI-clean production betting edge is claimed. A7 remains shadow-only.
Economic confirmation still requires accumulating independently-clustered
post-2026-07-30 forward competitions.

## Reproduce

```bash
# Control equivalence (archived arms)
uv run python - <<'PY'
import json
prod = json.load(open('models/xgb_match_v3_m7_production/test_predictions.json'))
base = json.load(open('models/auto/d12/base_seed29/test_predictions.json'))
print(max(abs(prod[k]['p_team1'] - base[k]['p_team1']) for k in prod))
PY
# Iteration readout (envelope → blend w0 → I3 reslice)
uv run python scripts/synthesize_golden_envelope.py --odds betting_odds_polymarket.json --out /tmp/iter_env.json
uv run python scripts/sim_eval/blend_eval_json.py --sim-json /tmp/iter_env.json \
  --direct-json models/xgb_match_v3_m7_swap_production/test_predictions.json --w 0.0 --out-dir /tmp/swap
uv run python scripts/sim_eval/reslice_eval_json.py --in /tmp/swap/iter_env_w0p00.json \
  --odds betting_odds_polymarket.json --out-dir /tmp/swap_sliced
# Golden audit
uv run python scripts/predict_golden.py --model-dir models/xgb_match_v3_m7_swap_production \
  --parquet data/xgb_match_data_v3_m3_unfrozen/golden_test.parquet --out-json /tmp/golden_swap.json
```
