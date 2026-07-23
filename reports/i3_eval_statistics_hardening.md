# I3 evaluation-statistics hardening

## Decision

I3 is complete as of 2026-07-23. Headline match-winner log-loss and flat-ROI
intervals now use the shared `tournament_time_block_v1` contract rather than
i.i.d. per-match resampling.

The change does not alter model probabilities, realized P&L, or point ROI.
It changes uncertainty estimates and therefore changes which historical
economic claims are supportable.

## Problems fixed

### P&L was an invalid bet-placement sentinel

The old aggregation paths treated `realized_pnl != 0` as “a bet was placed.”
A winning bet at decimal odds 1.0 has exactly zero P&L, so it disappeared from
the bet count, ROI denominator, and win rate.

New evaluation records persist:

- `bet_placed`;
- `bet_team`;
- `competition_cluster_id`.

Legacy records are reconstructed from known outcome, maximum edge strictly
above the configured threshold, and valid odds for the selected team. P&L is
never used to decide placement. A zero-return win now remains a placed,
winning bet.

The consumed production files contain no zero-return placed bets, so their
historical point estimates and bet counts are unchanged.

### Per-match bootstrap treated correlated fixtures as independent

The iteration set contains long tournament runs: 47 T20 World Cup matches,
44 BBL matches, 30 SA20 matches, 25 ILT20 matches, and so on. Market/model
errors within one competition share teams, conditions, scheduling, and market
regimes. Resampling individual matches understated this dependence.

## Frozen bootstrap contract

`scripts/sim_eval/eval_statistics.py` is now the single implementation:

1. Prefer `info.event.name` from the exact Cricsheet fixture.
2. Keep consecutive matches from that event in one block. Start a new edition
   only after more than 120 inactive days.
3. If event metadata is unavailable, fall back to unordered team pair plus
   July-to-June cricket season.
4. Draw the observed number of blocks with replacement.
5. Carry every selected observation from each drawn block into the replicate.
6. Recompute the observation-weighted mean. For flat one-unit betting this is
   sampled total P&L divided by sampled bet count.
7. Report the 2.5th and 97.5th percentiles from 10,000 resamples with seed 42.

Every summary records the contract, seed, resample count, metadata coverage,
and effective number of betting blocks. Fewer than 10 blocks is marked
descriptive rather than confirmatory.

## Re-reported consumed results

| Model/policy | Slice | Bets | ROI | Old i.i.d. 95% CI | I3 block 95% CI | Blocks |
|---|---|---:|---:|---:|---:|---:|
| Match M7 flat | ≥$50k | 168 | +21.90% | [+2.28%, +43.83%] | **[-10.79%, +50.18%]** | 19 |
| Match M7 flat | ≥$100k | 110 | +26.39% | [+0.57%, +58.78%] | **[-17.36%, +46.42%]** | 11 |
| Match M7 + A7 | ≥$50k | 109 | +36.93% | [+12.06%, +68.89%] | **[-1.52%, +59.81%]** | 17 |
| Match M7 + A7 | ≥$100k | 72 | +35.86% | [-0.99%, +80.58%] | **[-36.70%, +58.16%]** | 10 |
| Ball v7 flat | ≥$50k | 168 | +6.11% | [-10.72%, +23.87%] | **[-7.99%, +25.70%]** | 19 |
| Ball v7 flat | ≥$100k | 110 | -2.86% | [-23.03%, +19.29%] | **[-28.60%, +17.32%]** | 11 |

Corresponding competition-block log-loss intervals:

| Model | Slice | Log loss | I3 95% CI |
|---|---|---:|---:|
| Match M7 | ≥$50k | 0.6299 | [0.5714, 0.7004] |
| Match M7 | ≥$100k | 0.5929 | [0.5421, 0.6823] |
| Ball v7 | ≥$50k | 0.7402 | [0.7101, 0.8028] |
| Ball v7 | ≥$100k | 0.7311 | [0.6893, 0.8227] |

The historical point profitability remains encouraging, and A7 still improves
the point ROI and drawdown. However, no standing historical match-winner ROI
claim now has a block-bootstrap lower bound above zero. Prior “CI-clean”
economic claims are superseded.

Generated consumed-data summaries are under `reports/i3_consumed_eval/`.

## Historical inputs

The I3 re-report consumes existing predictions only; it does not retrain or
rescore either model:

| Input | Purpose | SHA-256 |
|---|---|---|
| `archive/eval_results/eval_out_m7prod/hier_all_20260425_165622_w0p00.json` | Match M7 predictions, 261-match iteration set | `022ba32c84e6997512beae1516a3669f156be64837765a900d53f5c9b17a1031` |
| `eval_out/v7_full_20260508/xgboost_all_20260508_163131.json` | Ball v7 winner probabilities on the same 261 matches | `89403b41daa1b6184c80971eaa5613405fbd508fc2fa86f518afb032137f26f7` |
| `betting_odds_polymarket.json` | Historical close-line prices and volume slices | `cd889b8a893c5c80e1386a31ee10c516f315090c2d677b23625343cd4ee7ec65` |
| `data/xgb_match_data_v3_m6_unfrozen/test.parquet` | Pre-existing ELO slice used by the fixed A7 rule | `48e93dfbcc7671adec1d2510895015ae9fedf16e3ca3ed14ca5f455d1b641c9b` |

Competition blocks come from the exact 261 Cricsheet fixtures in
`data/polymarket_test`; all 261 evaluation IDs matched that metadata.

## Forward-set implications

The sealed forward dataset remains unscored. Its fixture metadata alone shows:

| Slice | Fixtures | Competition blocks |
|---|---:|---:|
| All | 137 | 13 |
| ≥$50k primary | 61 | 7 |
| ≥$100k secondary | 30 | 6 |

In the primary slice, 25 fixtures are MLC and 24 are the Vitality Blast.
Therefore the forward set can still provide a strong probability comparison,
but its ROI interval is low-power. Under the I3 minimum-cluster guard, this
window cannot by itself confirm a production betting edge, even if its
descriptive lower bound is positive. More independently clustered forward
competitions must accumulate for economic confirmation.

No forward model probability or ROI was computed during I3.

## Verification

- Synthetic block tests prove entire competitions move together.
- Cross-calendar events stay together; later editions split after the
  inactivity gap.
- Zero-return wins count as placed wins in both evaluator and reslice paths.
- Evaluator and reslice wrappers remain deterministic at fixed seed.
- Existing Kelly, edge-threshold, liquidity-boundary, blend, and legacy
  arithmetic tests remain covered.
