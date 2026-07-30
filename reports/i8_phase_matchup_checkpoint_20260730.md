# I8 phase and matchup checkpoint — 2026-07-30

## Decision

I8 is implemented, reproducible, and retained as an isolated candidate, but
it is **not promoted**. The 18 hierarchical phase/H2H features produce a
small, consistent ball-level improvement. On the consumed Polymarket
diagnostic they improve the full-slice match log-loss and Brier point
estimates, but the competition-block confidence intervals cross zero and
flat-bet ROI is worse. A new untouched post-2026-07-30 terminal window is
required for a promotion decision.

This decision is separate from I7 venue identity. I7 remains the mandatory
identity contract for all new training and artifacts even when a particular
I7/I8 model does not beat the frozen production model. The legacy live mode
exists only to serve frozen artifacts without falsely relabeling them.

## Frozen experiment contract

I8 keeps all 114 I7 features and adds exactly 18 pre-ball probabilities:

- six current-phase batter outcome probabilities;
- six current-phase bowler outcome probabilities; and
- six exact batter-vs-bowler outcome probabilities.

Player-phase cells shrink to the player's already-shrunk overall distribution
with `k_player=30` and `k_phase=30`. H2H cells shrink to the arithmetic mean
of the batter and bowler profiles with `k_h2h=60`. Venue shrinkage remains
`k_venue=200`. The outcome order is `0, 1, 2, 4, 6, W`.

The run used the unchanged 9,519-match male corpus through 2026-04-16. It
copied the global priors from the frozen I7 SQLite cache rather than
recomputing them. The source prior SHA-256 was
`9b87622a39e19ffc08adcc843780e0958484766a1beedebadb8ef1674bbab2a0`.
No refreshed forward fixture entered training, validation, or test.

Artifacts are isolated under schema v5:

- `models/player_stats_cache_i8.sqlite`;
- `data/xgb_data_i8/`;
- `models/xgb_i8/`; and
- experiment `xgb_i8_phase_matchup_20260730_092615_a4160ba`.

The simulator uses the separate fail-closed
`scripts/sim_eval/run_sim_eval_i8.py` path. It requires schema 5, all 18 new
columns, the four shrinkage values, and the active I7 venue contract. It
cannot silently substitute zeros, legacy state, or the demonstration model.

## Build and materialization

The schema-v5 cache contains 7,516 players, 373 canonical venues, and 3,664
match dates. Materialized rows exactly match I7:

| split | rows |
|---|---:|
| train | 1,876,971 |
| validation | 124,292 |
| test | 186,667 |
| post-test golden | 0 |

The 132-feature build is more expensive because the cache stores and queries
player-phase and batter-bowler cells:

| step | I7 seconds | I8 seconds | ratio |
|---|---:|---:|---:|
| SQLite cache | 455.6 | 838.3 | 1.84x |
| parquet materialization | 532.5 | 938.4 | 1.76x |
| XGBoost training | 363.8 | 442.3 | 1.22x |
| total | 1,351.9 | 2,219.0 | 1.64x |

## Paired ball-level result

Deltas are I8 minus I7; lower log loss and Brier are better. Confidence
intervals use match-cluster bootstrap resampling.

| split | balls / matches | I7 LL | I8 LL | delta LL, 95% CI | I7 Brier | I8 Brier | delta Brier, 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|
| validation | 124,292 / 545 | 1.637592 | **1.634469** | -0.003123 [-0.006706,+0.000506] | 0.782692 | **0.781540** | -0.001152 [-0.002645,+0.000396] |
| test | 186,667 / 822 | 1.631620 | **1.628690** | -0.002931 [-0.005822,+0.000102] | 0.781231 | **0.779532** | **-0.001699 [-0.002854,-0.000457]** |

Accuracy moves from 32.17% to 32.20% on validation and from 32.24% to
32.35% on test. The effect is small: only the test Brier interval excludes
zero. `batter_phase_p6` ranks sixth in XGBoost feature importance, showing
that at least one new feature is being used materially.

I8 modestly reduces the existing six and wicket overprediction and improves
dot/one calibration. It makes the two and four probability gaps slightly
worse. This is a useful but incomplete calibration change, not a decisive
ball-model promotion signal.

## Paired match-simulation diagnostic

This is diagnostic only: the 261-fixture Polymarket set was already consumed
by earlier work. Both arms use the same current simulator, 100 simulations
per match, seed 42, and market/outcome rows. Six unresolved fixtures leave
255 paired scores. Intervals resample 25 competition-time blocks and recompute
each arm's own flat P&L/bet-count ratio.

| slice | scored / blocks | I7 LL | I8 LL | delta LL, 95% CI | I7 Brier | I8 Brier | delta Brier, 95% CI | I7 ROI | I8 ROI | delta ROI, 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 255 / 25 | 0.7042 | **0.6825** | -0.0217 [-0.0521,+0.0179] | 0.2530 | **0.2456** | -0.0074 [-0.0216,+0.0101] | +0.46% | -1.49% | -1.95pp [-16.31,+12.61] |
| >=$50k | 168 / 19 | 0.7176 | **0.7017** | -0.0160 [-0.0445,+0.0219] | 0.2607 | **0.2550** | -0.0057 [-0.0198,+0.0114] | +1.38% | -0.38% | -1.75pp [-19.15,+14.84] |
| >=$100k | 110 / 11 | **0.6770** | 0.6780 | +0.0010 [-0.0294,+0.0658] | **0.2418** | 0.2446 | +0.0028 [-0.0119,+0.0333] | +6.08% | -1.55% | -7.64pp [-43.05,+7.82] |

The full-set probability point estimates favor I8, but the highest-liquidity
slice does not and every interval includes zero. The flat strategy loses
3.80 units under I8 versus gaining 1.17 units under I7.

### Upset sensitivity

The largest candidate win is France over Portugal at a 5% market probability,
worth +19 flat units. Removing that fixture changes:

- I7: +0.46% to -7.02% ROI over 254 bets;
- I8: -1.49% to -8.98% ROI over 254 bets; and
- the relative I8-minus-I7 gap remains approximately -1.96 percentage points.

Removing both winning candidate bets whose market probability was at most
10% leaves I7 at -11.30% and I8 at -13.27%. The probability-loss advantage
remains directionally similar, but neither betting result is robust without
rare upsets.

## Next gate

Do not tune `k_phase`, `k_h2h`, model hyperparameters, or feature subsets
against the frozen test or consumed Polymarket results. Keep the schema-v5
implementation and candidate artifacts intact. Accumulate fixtures strictly
after 2026-07-30, seal their market snapshots before outcomes, and run the
precommitted raw I8 candidate once against I7 and the frozen production
control. Promotion should require:

1. a probability improvement whose block confidence interval is convincing;
2. no material regression on the highest-liquidity slice;
3. ROI reported with and without the largest upset, but not used alone; and
4. unchanged I7 identity, schema-v5, prior-hash, and feature-order checks.

Until that terminal evaluation exists, production remains on the frozen
model through explicit legacy compatibility, while I7 is the required
forward data identity and I8 remains an unpromoted candidate.
