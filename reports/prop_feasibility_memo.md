# v7 sim → prop bets — Phase 1 feasibility memo

**Date**: 2026-05-09 · **Compute**: 30 matches × 100 sims, ~4.3 min serial
**Detail**: `reports/prop_calibration_detail.json`
**Tables**: `reports/prop_calibration_report.md`

## TL;DR

The v7 ball-by-ball sim is well-calibrated on a few specific prop families and
miscalibrated on others. The pattern is consistent enough to suggest a real
prop-edge play — but **only on the props the sim is good at**. Two prop
families are genuinely clean signal; two are systematically biased (which is
itself a betting opportunity, just on the *opposite* side); two have no usable
signal.

| Prop family | Sample n | Verdict | Direction to bet |
|---|---:|---|---|
| **Highest individual score** O/U | 30 | ✅ ship | Both sides — sim mean is unbiased, P10–P90 coverage hits 80% (ideal) |
| **Batter ≥1 six** (Y/N) | 486 | ✅ ship | Reliability lines up across bins; Brier 6.4% better than base |
| **Batter total runs** (O/U) | 486 | ✅ ship | MAE 16 runs, bias 0.10, coverage 77% (close to ideal 80%) |
| **Team total sixes** O/U | 60 | 🟡 ok | Unbiased, MAE 3.7, but coverage 71% (slightly over-confident) |
| **Top batter** (per team) | 660 | 🟡 small edge | Brier 4.2% better than base; reliability good in low-prob bins, weakening above 0.25 |
| **Team total fours** O/U | 60 | ⚠️ inverse | **Sim systematically over-predicts by +2.9 fours** — bet UNDER when sim says OVER |
| **Batter 50+** (Y/N) | 486 | ⚠️ inverse | **Sim is over-confident; Brier WORSE than base** — bet NO when sim flags YES |
| **First over runs** O/U | 60 | ❌ skip | Bias -1.4 + coverage 62% (under-disperses); too noisy |
| **Top bowler** (per team) | 660 | ❌ skip | Brier essentially equal to base (0.0823 vs 0.0826) — no signal |

## Read of the result

1. **Sim is best where ball-level granularity matters most.** Highest
   individual score and batter sixes both depend on the full 6-class outcome
   distribution rather than averages — which is exactly what the sim outputs
   per ball. Calibration falls apart for "top bowler" because v7 doesn't
   model dismissal-method or who's bowling at the death (bowler selection
   is currently `RandomBowlerSelector`, see `sim_v1_2.py:404`). That bowler
   selector is the smoking gun for the top-bowler miss; this is a fix, not a
   blocker.

2. **Two systematic biases are exploitable.** The sim **over-counts fours
   by +2.9 per team** and **over-states 50+ probabilities** vs realised
   hit rate. Both biases are large enough relative to typical DK juice
   (-110/-115) that, even before considering market sharpness, a flat strategy
   of fading the sim on these two families likely shows positive ROI in a
   backtest. This needs Phase 2 confirmation against actual lines.

3. **Markets are fragmented enough to matter.** Across 112 prop-rich DK
   captures (Feb–Aug 2025) we already have 82 highest-individual-score
   captures, 49 top-batter captures, 25 50+ captures, and 18 team most-fours
   captures. So Phase 2 has real ground for a sim-vs-line comparison even
   without restoring the scraper.

## Phase 2 plan

1. **Fix the cron** (cheap): repoint `run_scraper_cron.sh` from the dead
   `src/bet_scraper.py` to `archive/scripts/bet_scraper.py`, or restore the
   `src/` symlink. Today the scraper has been dark since 2025-08-25.
2. **Retrofit `prop_backtest.py` to score against DK lines.** For each DK
   capture, look up the cricsheet match in `data/t20s_json/` by team-name
   alias + date, run the sim, compute prop probabilities, convert DK American
   odds to implied p, compute edge, and tally flat ROI per prop family.
   Restrict to matches dated **after 2025-06-30** (tracker freeze) so player
   stats are out-of-sample. That gives ~50 prop-rich captures from Jul–Aug
   2025.
3. **Prioritise the four green-flagged families** (highest individual score,
   batter sixes, batter total runs, team total sixes) and the two
   inverse-bias families (team fours, batter 50+). Don't bet top bowler /
   first over until those are fixed.
4. **Open question**: should we replace `RandomBowlerSelector` with a usage-
   pattern-aware selector (most-bowled-at-this-phase, captain's matchup
   history)? That's the single biggest known sim defect for prop bets and
   would unlock both top-bowler and bowler-match-bet markets.

## What this study did NOT do

- **No DK line comparison.** Phase 1 only checks whether the sim is
  internally accurate vs cricsheet ground truth, not whether DK's prices
  leave room for an edge.
- **No coverage of more exotic props** (1st wicket method, most run-outs,
  1st-ball runs) — these need engine changes (modeling dismissal type or
  run-outs separately) before we can evaluate them.
- **Sample is 30 matches.** Mostly Indian/SA/NZ/WI internationals plus the
  first chunk of BBL 2025–26. Findings need to hold up on the larger 261
  Polymarket matches (and ideally on independent IPL data) before we size
  bets seriously.
