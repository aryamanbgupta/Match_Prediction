# Extended golden audit — 2026-07-30

## What this is

The golden audit set grew 55 → **124** odds-matched fixtures (new slice
2026-05-10 → 2026-06-17: 45 T20 World Cup 2026, 19 IPL playoff, 5 other
international fixtures; the 137 consumed forward fixtures are excluded and
`verify_forward_holdout` still reports zero overlap). Features come from
the frozen forward-state sidecar
(`data/forward_state/2026-06-01_2026-07-13/match_features/golden_test.parquet`,
401 deterministic legacy-semantics rows through 2026-07-13) — the same
pre-I7 state contract the production model serves under, so no venue
canonicalization mismatch. Five minor associate-nation venues were unseen
by the encoders and fell back (Botswana, two Japanese grounds, Old Deer
Park, Pomona) — all low-liquidity fixtures.

Both arms scored: production `xgb_match_v3_m7_swap_production` (swap) and
frozen `xgb_match_v3_m7_production` (base). Golden remains audit-only;
nothing here was used for selection.

## Results (I3 block bootstrap; ≤10 blocks = descriptive)

| Slice | n | swap LL | base LL | market LL | swap ROI | swap ROI CI | base ROI | swap win | blocks |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| all | 124 | **0.5831** | 0.5916 | 0.5513 | −4.35% | [−39.4, +13.0] | −13.38% | 0.472 | 11 |
| ≥$50k | 75 | **0.6685** | 0.6736 | 0.6573 | +9.69% | [−46.9, +19.5] | +1.81% | 0.527 | 9 |
| ≥$100k | 66 | **0.6938** | 0.6964 | 0.6843 | +14.38% | [−85.1, +20.3] | +3.27% | 0.554 | 5 |

## Reading

1. **The swap model beats base on every slice** — log loss, ROI, and win
   rate — extending the D12/promotion pattern to a fourth independent
   readout (iteration, original golden, and now the extended golden).
2. **Both models trail the market on this window.** The extension is
   dominated by T20 World Cup markets — the deepest, sharpest books of the
   year — and the market's log loss (0.5513 all-slice) is well ahead of
   either model. Compare the original 55-match golden, where swap beat the
   matched market on both slices: the model's relative standing depends
   heavily on how sharp the market is, which is consistent with the I3-era
   conclusion that side-vs-market selection, not absolute probability
   quality, is where past ROI came from.
3. **No betting edge is claimed.** Every ROI interval straddles zero and
   the liquid slices are descriptive (≤10 blocks). The positive swap point
   estimates on the liquid slices (+9.7% / +14.4%) are noted, not relied
   on.
4. Practical read for A7: nothing here authorizes execution. If anything,
   the WC result argues for caution precisely where liquidity is best.

## Reproduce

```bash
PARQ=data/forward_state/2026-06-01_2026-07-13/match_features/golden_test.parquet
uv run python scripts/predict_golden.py --model-dir models/xgb_match_v3_m7_swap_production \
  --parquet $PARQ --out-json /tmp/pred_swap.json
uv run python scripts/synthesize_golden_envelope.py \
  --odds data/golden/betting_odds_golden.json --out /tmp/env.json
uv run python scripts/sim_eval/blend_eval_json.py --sim-json /tmp/env.json \
  --direct-json /tmp/pred_swap.json --w 0.0 --out-dir /tmp/swap
uv run python scripts/sim_eval/reslice_eval_json.py --in /tmp/swap/env_w0p00.json \
  --odds data/golden/betting_odds_golden.json --out-dir /tmp/swap_sliced \
  --cluster-source-dir data/golden/t20s_json
```
