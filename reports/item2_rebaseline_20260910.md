# Item 2 step 6 — production-model rebaseline (2026-09-10)

## Verdict

**Not identical.** The mandated current-code pipeline does not reproduce the
restated production headline. The largest absolute difference among the
headline fields stated in `CLAUDE.md` is **27.1411 percentage points of ROI**
on the >=$100k slice: the run produced **+21.9511%**, versus the restated
**-5.1900%**. The model log losses and slice counts match that headline to
four decimals, but the market log losses, PnL-derived returns, and block CIs
do not.

The reason visible in the artifacts is mechanical: the mandated source
`eval_out/i17/hier_all_cricsheet.json` contains the pre-v2 `market_prob` and
`market_odds`. `blend_eval_json.py --w 0.0` carries those prices into the
blended JSON, while `reslice_eval_json.py --odds
betting_odds_polymarket_v2.json` uses the supplied odds file for slice
membership only; it does not reprice the rows. `blend_report.py` independently
confirmed this provenance and recomputed the old market lines (0.6267,
0.6482, 0.6224). Therefore the scripts, with the inputs required by this
task, cannot produce the corrected-price headline. No code was modified.

The production artifact itself is the requested I17 swap seed-29 model:
`model.pkl` MD5 `54faf58638a799468d551beb3493b22d`. Its pulled
`test_predictions.json` MD5 is `9209919063941841ce6d3411ccd6a279`.

## Commands run (verbatim)

The sandbox could not initialize the default user uv cache, so each successful
command used the transient `/tmp/cricml-item2-uv-cache`. Every Python command
was run through `uv run --no-sync`.

```bash
UV_CACHE_DIR=/tmp/cricml-item2-uv-cache uv run --no-sync python scripts/sim_eval/blend_eval_json.py \
  --sim-json eval_out/i17/hier_all_cricsheet.json \
  --direct-json models/xgb_match_i7_swap_production/test_predictions.json \
  --w 0.0 \
  --out-dir eval_out/item2_rebaseline/blend | tee eval_out/item2_rebaseline/blend.log

UV_CACHE_DIR=/tmp/cricml-item2-uv-cache uv run --no-sync python scripts/sim_eval/reslice_eval_json.py \
  --in eval_out/item2_rebaseline/blend/hier_all_cricsheet_w0p00.json \
  --odds betting_odds_polymarket_v2.json \
  --cluster-source-dir data/polymarket_test_v2 \
  --out-dir eval_out/item2_rebaseline/sliced | tee eval_out/item2_rebaseline/reslice_zero_event.log

UV_CACHE_DIR=/tmp/cricml-item2-uv-cache uv run --no-sync python scripts/sim_eval/reslice_eval_json.py \
  --in eval_out/item2_rebaseline/blend/hier_all_cricsheet_w0p00.json \
  --odds betting_odds_polymarket_v2.json \
  --cluster-source-dir data/polymarket_test_v2 \
  --spread-bps 100 --fee-bps 0 --fee-basis winnings \
  --out-dir eval_out/item2_rebaseline/cost_s100_f0_winnings | tee eval_out/item2_rebaseline/reslice_s100_f0_winnings.log

UV_CACHE_DIR=/tmp/cricml-item2-uv-cache uv run --no-sync python scripts/sim_eval/reslice_eval_json.py \
  --in eval_out/item2_rebaseline/blend/hier_all_cricsheet_w0p00.json \
  --odds betting_odds_polymarket_v2.json \
  --cluster-source-dir data/polymarket_test_v2 \
  --spread-bps 200 --fee-bps 0 --fee-basis winnings \
  --out-dir eval_out/item2_rebaseline/cost_s200_f0_winnings | tee eval_out/item2_rebaseline/reslice_s200_f0_winnings.log

UV_CACHE_DIR=/tmp/cricml-item2-uv-cache uv run --no-sync python scripts/sim_eval/reslice_eval_json.py \
  --in eval_out/item2_rebaseline/blend/hier_all_cricsheet_w0p00.json \
  --odds betting_odds_polymarket_v2.json \
  --cluster-source-dir data/polymarket_test_v2 \
  --spread-bps 100 --fee-bps 100 --fee-basis winnings \
  --out-dir eval_out/item2_rebaseline/cost_s100_f100_winnings | tee eval_out/item2_rebaseline/reslice_s100_f100_winnings.log

UV_CACHE_DIR=/tmp/cricml-item2-uv-cache uv run --no-sync python scripts/sim_eval/reslice_eval_json.py \
  --in eval_out/item2_rebaseline/blend/hier_all_cricsheet_w0p00.json \
  --odds betting_odds_polymarket_v2.json \
  --cluster-source-dir data/polymarket_test_v2 \
  --spread-bps 100 --fee-bps 100 --fee-basis stake \
  --out-dir eval_out/item2_rebaseline/cost_s100_f100_stake | tee eval_out/item2_rebaseline/reslice_s100_f100_stake.log

UV_CACHE_DIR=/tmp/cricml-item2-uv-cache uv run --no-sync python scripts/sim_eval/reslice_eval_json.py \
  --in eval_out/item2_rebaseline/blend/hier_all_cricsheet_w0p00.json \
  --odds betting_odds_polymarket_v2.json \
  --cluster-source-dir data/polymarket_test_v2 \
  --volume-basis market \
  --out-dir eval_out/item2_rebaseline/volume_market_zero | tee eval_out/item2_rebaseline/reslice_zero_market.log

UV_CACHE_DIR=/tmp/cricml-item2-uv-cache uv run --no-sync python scripts/sim_eval/blend_report.py \
  --sliced-dir eval_out/item2_rebaseline/sliced \
  --direct-json models/xgb_match_i7_swap_production/test_predictions.json \
  --out eval_out/item2_rebaseline/blend_report.md | tee eval_out/item2_rebaseline/blend_report.log

rg -n "Cost scenarios|gate safety check|0/0 bps" eval_out/item2_rebaseline/blend_report.md
```

## Zero-cost parity

Market LL below is recomputed from each sliced JSON's own `market_prob` rows.
PnL is in flat one-unit stakes; ROI and CI are percentages. Counts are shown
to four decimals solely to apply the requested numeric presentation uniformly.
An em dash means the comparison source did not publish that field.

| Slice | Source | Model LL | Market LL | n | Bets | Total PnL | ROI | Block CI (blocks) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| all | current code, v2 event-volume membership | 0.6187 | 0.6267 | 261.0000 | 255.0000 | 44.8756 | +17.5983% | [-6.8449%, +45.2816%] (27.0000; fallback) |
| all | pre-item-2 I17 recorded JSON | 0.6187 | 0.6267 | 261.0000 | 255.0000 | 44.8756 | +17.5983% | [-7.4823%, +45.9995%] (25.0000) |
| all | `CLAUDE.md` honest-headline paragraph | — | — | — | — | — | — | — |
| >=$50k | current code, v2 event-volume membership | 0.6249 | 0.6482 | 168.0000 | 167.0000 | 35.5067 | +21.2615% | [-4.9529%, +44.5122%] (18.0000) |
| >=$50k | pre-item-2 I17 recorded JSON | 0.6262 | 0.6482 | 170.0000 | 168.0000 | 34.5067 | +20.5397% | [-5.5608%, +43.4749%] (19.0000) |
| >=$50k | `CLAUDE.md` restated headline | 0.6249 | 0.5940 | 168.0000 | — | — | +3.3800% | [-14.6300%, +37.0600%] (18.0000) |
| >=$100k | current code, v2 event-volume membership | 0.5886 | 0.6224 | 110.0000 | 110.0000 | 24.1462 | +21.9511% | [-19.7672%, +40.5060%] (11.0000) |
| >=$100k | pre-item-2 I17 recorded JSON | 0.5886 | 0.6224 | 110.0000 | 110.0000 | 24.1462 | +21.9511% | [-19.7672%, +40.5060%] (11.0000) |
| >=$100k | `CLAUDE.md` restated headline | 0.5886 | 0.5377 | 110.0000 | — | — | -5.1900% | [-28.7300%, +27.5000%] (11.0000) |

Exact current-minus-recorded differences at four decimals:

| Slice / comparison | Delta model LL | Delta market LL | Delta n | Delta bets | Delta PnL | Delta ROI | Delta CI low | Delta CI high |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| all vs pre-item-2 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000pp | +0.6375pp | -0.7179pp |
| >=$50k vs pre-item-2 | -0.0013 | -0.0000 | -2.0000 | -1.0000 | +1.0000 | +0.7218pp | +0.6078pp | +1.0373pp |
| >=$100k vs pre-item-2 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000pp | +0.0000pp | +0.0000pp |
| >=$50k vs restated headline | +0.0000 | +0.0542 | +0.0000 | — | — | +17.8815pp | +9.6771pp | +7.4522pp |
| >=$100k vs restated headline | +0.0000 | +0.0847 | +0.0000 | — | — | +27.1411pp | +8.9628pp | +13.0060pp |

The new zero-cost stamps are identical across all three current slices:

| Stamp | Value |
|---|---|
| `cost_model` | `spread_bps=0.0000`, `fee_bps=0.0000`, `fee_basis=winnings` |
| `price_basis` | `mid` |
| `volume_basis` | `event` |
| `pnl_unrecomputable` | 0.0000 |
| `price_rejected` | 0.0000 |

The all-slice current CI is descriptive only: 6.0000 of 261.0000 rows lacked
tournament metadata in `data/polymarket_test_v2` and used pair-block fallback,
changing the contract to `tournament_time_block_v1_fallback_pair_blocks` and
the reported block count to 27.0000. The >=$50k and >=$100k runs retain the
`tournament_time_block_v1` contract and reliable-CI stamp.

## Cost scenarios

Each cell is `total PnL / ROI`; all scenarios retain the same placed-bet set
within a slice. The zero-cost column is the theoretical safety-check column.

| Slice | 0/0 winnings | 100/0 winnings | 200/0 winnings | 100/100 winnings | 100/100 stake |
|---|---:|---:|---:|---:|---:|
| all | 44.8756 / +17.5983% | 39.3037 / +15.4132% | 34.1795 / +13.4037% | 37.6006 / +14.7453% | 36.7537 / +14.4132% |
| >=$50k | 35.5067 / +21.2615% | 32.6016 / +19.5219% | 29.8216 / +17.8572% | 31.4656 / +18.8416% | 30.9316 / +18.5219% |
| >=$100k | 24.1462 / +21.9511% | 22.0846 / +20.0769% | 20.1263 / +18.2966% | 21.3338 / +19.3944% | 20.9846 / +19.0769% |

## Volume-basis comparison at zero cost

| Slice | Volume basis | Model LL | Market LL | n | Bets | Total PnL | ROI | Block CI |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| all | event | 0.6187 | 0.6267 | 261.0000 | 255.0000 | 44.8756 | +17.5983% | [-6.8449%, +45.2816%] |
| all | market | 0.6187 | 0.6267 | 261.0000 | 255.0000 | 44.8756 | +17.5983% | [-6.8449%, +45.2816%] |
| >=$50k | event | 0.6249 | 0.6482 | 168.0000 | 167.0000 | 35.5067 | +21.2615% | [-4.9529%, +44.5122%] |
| >=$50k | market | 0.6242 | 0.6519 | 156.0000 | 155.0000 | 40.8251 | +26.3388% | [+0.4571%, +54.2114%] |
| >=$100k | event | 0.5886 | 0.6224 | 110.0000 | 110.0000 | 24.1462 | +21.9511% | [-19.7672%, +40.5060%] |
| >=$100k | market | 0.5896 | 0.6252 | 108.0000 | 108.0000 | 24.5837 | +22.7627% | [-19.6013%, +40.6621%] |

The market-volume outputs stamp `cost_model={spread_bps: 0.0000, fee_bps:
0.0000, fee_basis: winnings}`, `price_basis=mid`, `volume_basis=market`,
`pnl_unrecomputable=0.0000`, and `price_rejected=0.0000` on every slice.

## `blend_report.py` safety-check ordering

Confirmed. With the default `--cost-scenarios`, every slice prints the
zero-cost column first and labels it exactly `0/0 bps winnings (gate safety
check)`, followed by 100/0 winnings, 200/0 winnings, 100/100 winnings, and
100/100 stake. The generated check report is
`eval_out/item2_rebaseline/blend_report.md`.

## Plain-language reading

This exercise does not establish a betting edge. It shows that the current
cost machinery behaves monotonically on a fixed set of old embedded prices:
adding the placeholder spread and fee assumptions lowers both total PnL and
ROI, while changing from event volume to market volume changes which fixtures
qualify. More importantly, the prescribed I17 envelope cannot reproduce the
corrected v2 headline because it still carries the retracted pre-v2 market
prices, and reslicing does not replace them. Consistent with plan principle
0.4, log loss is the decision metric and returns are only a safety check; the
reported return intervals are noisy, and no betting claim is made.

## Addendum (Claude, 2026-09-10): the parity check the step actually requires

The verdict above compares the current pipeline against the restated
CLAUDE.md headline. That comparison is confounded twice: the I17 envelope
embeds the retracted pre-v2 prices, and the I17 sliced outputs of record
were built with `betting_odds_polymarket.json` and `data/polymarket_test`
(261/170/110), not the v2 files (255/168/110). Neither is a property of
item 2.

Plan §2 step 6 asks whether the item 2 code reproduces the pre-item-2
numbers on the same inputs. Re-running `reslice_eval_json.py` on the same
blended JSON with the same old inputs the I17 run used
(`eval_out/item2_rebaseline/parity_old_inputs/`, retracted odds file used
here as a code-parity input only, not as a market benchmark):

| Slice | numeric summary fields differing | non-stamp fields differing |
|---|---:|---:|
| all | 0 | 0 |
| ≥$50k | 0 | 0 |
| ≥$100k | 0 | 0 |

Per-record `realized_pnl` over all 261 records: 0 mismatches. The only new
fields are the item 2 stamps (`cost_model`, `price_basis`, `volume_basis`,
`pnl_unrecomputable`, `price_rejected`, `cluster_resolution`).

**Verdict: identical at zero cost; numbers moved only in the cost-scenario
columns, as the plan expected.**

Finding carried to item 4: reslice takes prices from the stored records, so
an eval JSON built on retracted prices keeps them however it is resliced.
This is exactly why the claim gate must recompute `market_prob` and profit
from the registered odds file (plan §4 [v4] evidence check). Under the v2
files six envelope fixtures have no cluster in `data/polymarket_test_v2`,
which drops the `all` slice to the pair-block fallback
(`bootstrap_reliable: false`); the ≥$50k and ≥$100k slices resolve fully.
