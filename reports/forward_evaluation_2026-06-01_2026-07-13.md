# Forward evaluation: 2026-06-01 through 2026-07-13

## Result

The frozen one-time protocol completed on 2026-07-23.

- **Probability confirmation: PASS.**
- **Economic confirmation: NOT CONFIRMED.**

On the preregistered primary slice (Polymarket volume at least $50,000),
match-level M7 has log loss 0.6823, better than the market's 0.7445 and
ball-v7's 0.7015. This satisfies the frozen probability rule: M7 is no more
than 0.01 worse than the market and is better than ball-v7.

M7's frozen A7 policy returned +96.72% on the primary slice, but the 33 bets
span only five competition blocks and the 95% block-bootstrap interval is
-3.29% to +623.85%. The protocol requires at least ten blocks and a lower
bound above zero. The economic result is therefore descriptive, not
confirmatory.

The operational interpretation is:

- retain M7 as the direct match-winner probability model;
- retain ball-v7 as the score/prop distribution engine and winner benchmark;
- do not claim a confirmed betting edge from this window;
- do not train, calibrate, tune thresholds, or select a new candidate using
  these 137 outcomes.

## Frozen evaluation set

- 137 resolved men's T20 fixtures from 2026-06-02 through 2026-07-13;
- 61 fixtures at volume at least $50,000 (primary);
- 30 fixtures at volume at least $100,000 (secondary, low-power);
- 13 competition/time blocks overall, seven in the primary slice, and six in
  the secondary slice;
- no unresolved outcomes.

Holdout fingerprint:
`82ccde16cf2b7e5f13a9236f2788f3c8be1582f312f5c028ec44a6ab76561028`.

Frozen protocol fingerprint:
`a7c1648ea058b3cbccf550a77dde95d5d95b473c010fa62cc32e5432af9d387f`.

## Probability metrics

Lower is better.

| Slice | Candidate | Log loss | Brier |
|---|---:|---:|---:|
| All (137) | Market | **0.6052** | **0.2088** |
| All (137) | Match M7 | 0.6358 | 0.2239 |
| All (137) | Ball v7 | 0.6539 | 0.2335 |
| Volume ≥$50k (61) | Market | 0.7445 | 0.2651 |
| Volume ≥$50k (61) | Match M7 | **0.6823** | **0.2449** |
| Volume ≥$50k (61) | Ball v7 | 0.7015 | 0.2533 |
| Volume ≥$100k (30) | Market | 0.8066 | 0.2839 |
| Volume ≥$100k (30) | Match M7 | 0.7136 | 0.2599 |
| Volume ≥$100k (30) | Ball v7 | **0.6743** | **0.2403** |

Primary-slice log-loss intervals are broad because the 61 fixtures contain
only seven blocks:

- market: 0.7445 [0.6228, 1.0752];
- M7: 0.6823 [0.5789, 0.7527];
- ball-v7: 0.7015 [0.4478, 0.7621].

The frozen decision uses the preregistered point comparison; the individual
intervals above are not a paired interval for the difference.

The all-match market result is better than both models, while M7 wins on the
primary liquidity slice. The ≥$100k result favors ball-v7, but that secondary
slice has only 30 fixtures and six blocks and cannot override the primary
decision.

## Comparison with the earlier iteration set

M7 did not improve in absolute log loss. On the older ≥$50k iteration set its
log loss was 0.6299; on this forward slice it is 0.6823, a deterioration of
0.0524. The important transfer result is relative:

- earlier iteration: M7 0.6299 versus market 0.6267, so M7 was 0.0032 worse;
- frozen forward: M7 0.6823 versus market 0.7445, so M7 is 0.0621 better;
- frozen forward without the two major upsets: M7 0.6654 versus market
  0.6824, so M7 remains 0.0170 better.

The older and forward sets have different fixture and competition mixes, so
these values are not a claim that the model itself improved between runs.
They show that the frozen M7 transferred better than the contemporaneous
market on the preregistered high-liquidity slice.

The historical I3-corrected point ROIs were +21.90% for M7 flat and +36.93%
for M7 A7. The frozen point ROIs are much higher, but the upset-removed values
below are +20.58% and +20.04%. The apparent ROI jump is therefore mainly a
consequence of two long-shot wins, not evidence that expected ROI tripled.

## Betting metrics

ROI intervals are 95% tournament/time-block bootstrap intervals in percentage
points. One unit is staked per placed bet.

### All fixtures

| Candidate/policy | Bets | P&L | ROI | ROI 95% interval | Blocks |
|---|---:|---:|---:|---:|---:|
| M7 flat | 137 | +6.77 | +4.94% | [-42.30%, +88.19%] | 13 |
| M7 A7 | 86 | +10.62 | +12.35% | [-45.71%, +144.26%] | 12 |
| Ball-v7 flat | 137 | +2.72 | +1.98% | [-40.27%, +85.32%] | 13 |
| Ball-v7 A7 | 110 | +16.81 | +15.28% | [-34.03%, +123.63%] | 13 |

### Volume at least $50,000 (primary)

| Candidate/policy | Bets | P&L | ROI | ROI 95% interval | Blocks |
|---|---:|---:|---:|---:|---:|
| M7 flat | 61 | +37.85 | +62.05% | [+9.54%, +346.57%] | 7 |
| M7 A7 | 33 | +31.92 | +96.72% | [-3.29%, +623.85%] | 5 |
| Ball-v7 flat | 61 | +32.08 | +52.58% | [+1.22%, +358.50%] | 7 |
| Ball-v7 A7 | 50 | +37.15 | +74.31% | [+14.64%, +487.17%] | 6 |

Positive lower bounds for some rows do not make them confirmatory: every
primary betting row has fewer than the frozen minimum of ten independent
blocks. The decision policy is M7 A7, whose interval also crosses zero.

### Volume at least $100,000 (secondary)

| Candidate/policy | Bets | P&L | ROI | ROI 95% interval | Blocks |
|---|---:|---:|---:|---:|---:|
| M7 flat | 30 | +29.60 | +98.68% | [+1.63%, +487.33%] | 6 |
| M7 A7 | 16 | +31.94 | +199.63% | [+26.71%, +773.27%] | 5 |
| Ball-v7 flat | 30 | +35.68 | +118.94% | [+22.76%, +513.10%] | 6 |
| Ball-v7 A7 | 26 | +35.96 | +138.30% | [+25.54%, +573.44%] | 6 |

These estimates are especially unstable and are descriptive only.

## Concentration and liquidity diagnostics

This section is post-hoc diagnosis, not a new decision rule.

Two Ireland wins against India, at decimal odds 18.1818 and 9.5238, contribute
+25.71 units of M7 A7's +31.92 primary-slice P&L. They are both part of the
same `India tour of Ireland` block. They are also the only primary-slice
matches whose actual winner had market probability below 25%, so that cutoff
removes exactly these two fixtures.

Removing them gives:

| Candidate/policy | Bets | P&L | ROI |
|---|---:|---:|---:|
| M7 flat | 59 | +12.14 | +20.58% |
| M7 A7 | 31 | +6.21 | +20.04% |
| Ball-v7 flat | 59 | +6.37 | +10.80% |
| Ball-v7 A7 | 48 | +11.45 | +23.85% |

The corresponding 59-match probability metrics are:

| Candidate | Log loss | Brier |
|---|---:|---:|
| M7 | **0.6654** | **0.2369** |
| Market | 0.6824 | 0.2454 |
| Ball-v7 | 0.6953 | 0.2503 |

Thus the probability ordering survives this sensitivity, while the ROI
magnitude falls sharply.

If the same two wins are removed from the all-liquidity results, every
betting row becomes negative: M7 flat -14.03%, M7 A7 -17.96%, ball-v7 flat
-17.03%, and ball-v7 A7 -8.24%. This reinforces that the defensible result is
specific to the frozen ≥$50k primary slice.

The stored provenance for both outliers was audited after evaluation:

| Date | Polymarket event/market | Price time | Scheduled start | Lag | Volume | Cricsheet result |
|---|---|---|---|---:|---:|---|
| 2026-06-26 | 626207 / 2652647 | 12:00:04Z | 12:30:00Z | 1,796s | $1,273,352 | Ireland by 34 runs |
| 2026-06-28 | 639388 / 2701986 | 12:00:22Z | 12:30:00Z | 1,778s | $507,750 | Ireland by 1 run |

Both source rows are exact-title `Ireland vs India` head-to-head markets,
male T20 scope, and exact date/team/venue joins. Their price ticks are roughly
30 minutes before scheduled start and are not low-liquidity observations.
There is no evidence in the sealed provenance that either win is an in-play,
wrong-market, duplicate-selection, or result-matching artifact.

A separate derived diagnostic for the 76 fixtures below $50,000 shows M7
flat ROI of -40.90% and M7 A7 ROI of -40.19%. This was not a preregistered
reporting slice, so it must not be used to tune a new threshold on this
holdout. It does explain why the all-fixture betting result is much weaker
than the primary-slice result and supports continuing to report liquidity
regimes separately.

## Locked artifacts

| Artifact | SHA-256 |
|---|---|
| `match_m7_predictions.json` | `1d433f598fa7f85618c3d96edab6b44b69a820fd84d5fd7e74a9ab8e906fda5e` |
| `ball_v7_predictions.json` | `0cf9bfc6e37a13ec90686ec98ef81de90d2a903642f6d7b78ad9c9193e3fec12` |
| `evaluation_report.json` | `a1c09c269c62abf844040f05ab4c37c20c2fa5c7172aa7c13a20645eeb4d80ba` |

The outcome-free M7 artifact was locked and committed before ball-v7 ran.
The outcome-free ball-v7 artifact was then locked and committed before the
evaluator read realized outcomes and odds. The raw report is
`forward_eval_out/2026-06-01_2026-07-13/evaluation_report.json`.

## Next evaluation step

Continue shadow capture with the same locked M7/A7 policy until a new,
unseen forward window contains at least ten independent betting blocks. The
current 137 outcomes remain evaluation-only forever. Any calibration,
simulation-count change, feature work, or betting-policy change must be
developed without these labels and compared on a future holdout under a new
protocol.
