# Forward evaluation protocol — draft, not yet frozen

## Status

**DRAFT / DO NOT SCORE THE FORWARD HOLDOUT YET.**

This document is the proposed one-time evaluation contract for
`data/forward_holdout/2026-06-01_2026-07-13/`. It must be reviewed and changed
only before any model probabilities are produced for that directory.

The machine-readable companion is
`evaluation/forward_protocol_2026-06-01_2026-07-13.yaml`. It remains `DRAFT`
and fails closed if a scoring command asks for frozen authorization.

Run the model-free preflight at any time:

```bash
uv run python scripts/forward_eval_contract.py \
  evaluation/forward_protocol_2026-06-01_2026-07-13.yaml
```

Adding `--require-frozen` must fail until scorer tests, ball same-day replay,
scoring-code hashes, and explicit user approval are all recorded.

Sealed dataset facts:

- 137 matched men's T20 fixtures, 2026-06-02 through 2026-07-13.
- 61 fixtures at Polymarket volume ≥$50k; 30 at ≥$100k.
- Full sealed-directory SHA-256 fingerprint:
  `82ccde16cf2b7e5f13a9236f2788f3c8be1582f312f5c028ec44a6ab76561028`.
- No model scoring was performed during extraction, construction, or
  verification.

I6 state prerequisite (complete 2026-07-23):

- deterministic order: `date_then_match_id_lexicographic_v1`;
- sidecar: `data/forward_state/2026-06-01_2026-07-13/`;
- sidecar SQLite SHA-256:
  `c95524db29f1680c5fff9fa97a3f7f5d882ceef41075e7477d30698a730edb7c`;
- priors frozen from the pre-holdout production SQLite with SHA-256
  `6c26c1bd6aff82eb88a37828143dcf5b4e6ddffecd1f078f9c54d56d17896fbd`;
- 401 context feature rows materialized and all 137 selected fixtures
  verified; no model was loaded or scored.

I3 statistics prerequisite (complete 2026-07-23):

- bootstrap contract: `tournament_time_block_v1`;
- 10,000 whole-competition resamples, seed 42, percentile 95% interval;
- contiguous Cricsheet events remain one block until an inactivity gap over
  120 days; missing event metadata falls back to unordered team pair/season;
- bet placement is explicit and independent of realized P&L;
- fewer than 10 betting blocks is descriptive, not confirmatory.

## Candidates

### Primary winner-market model

Match-level M7 XGBoost, raw probabilities:
`models/xgb_match_v3_m7_production/`.

| Artifact | SHA-256 |
|---|---|
| `model.pkl` | `5274a9ed52bf67a4916614bb9c16c88252fc3dbaf6c989840864661d30ca7ac6` |
| `encoders.pkl` | `a09411274d3885e7e980ad049a5709d795d472223ac68b23ddbea4ea61402e93` |
| `feature_columns.txt` | `5891d03bbc6694a5dc92c455355102cca06794ff9d20873b3738295abf4ef873` |

No Platt or isotonic calibrator is applied.

### Ball-level comparison model

XGBoost v7 hierarchical-shrinkage simulation. This is a winner-market
benchmark and the production engine for score/prop distributions; it is not
expected to replace the direct match model on match-winner log loss.

| Artifact | SHA-256 |
|---|---|
| `xgboost_model_v3.pkl` | `5400df329221d8a85f36eea793821459c39bd9fbd35a30d72e6f3900d3d491ac` |
| `batter_encoder_v3.pkl` | `94ed4c061945e0d6fd055dd46f48c2f349b8ab2651f2a482560e99c44b610e90` |
| `bowler_encoder_v3.pkl` | `554c1987b5d13d4d94a84b0f118cb85ba5d41d837b0f9861861f1dbb571f89ca` |
| `venue_encoder_v3.pkl` | `b451bb5a8c9ca1177e39eaceefc8977e2cb20dfd31d966c27b55fbeb21c18892` |
| `matchup_encoder_v3.pkl` | `051f45780937859ef6981d3e1d3afcf84277e2d02f655409b1d043a53d384ab2` |
| `feature_columns_v3.txt` | `5f400222192caf2c7b6617558ef521e18d3429f842e714647d27e2f6c1c4b176` |
| `outcome_dist_config_v3.json` | `5ecf91976a74f0ff9b68bbf32838ef18400406a7adc298f93a24362a595eac17` |
| `vector_scaling_calibrator_v1.pkl` | `e7f80123cb2460befdab2e414b7d06794892d7f2652746a2992b376961a9ed73` |
| `models/bowler_phase_usage.json` | `0ed98417e215a27343867fa3f2fb3fb7a2e65e4e6c4f1d472d053c077774e80a` |
| experiment config | `46a8ec24771648a89c689b74ca06aa31f909b33e8900884e7ec29b2912f1ff99` |
| `scripts/sim_v1_2.py` | `f20aef46e9a1d6511f28fa8ec84b061406a9c868444b578234a49ba4205c6433` |

Proposed simulation recipe: 100 simulations per fixture, fixed seed 43,
phase-aware empirical bowler selector, vector ball calibration, and current
venue-aware/default D15 simulation behavior.

## Historical reference only

These are context, not gates to be silently reused as forward results:

- Match M7 iteration ≥$50k: LL 0.6299; flat ROI +21.90%
  `[-10.79%, +50.18%]`, 168 bets and 19 competition blocks.
- A7 iteration ≥$50k: 109 bets, ROI +36.93%
  `[-1.52%, +59.81%]`, 17 competition blocks.
- Match clean-direct golden ≥$50k: LL 0.6747; ROI +32.61% with a CI that
  included zero.
- Ball v7 iteration ≥$50k winner LL: 0.7402; ROI +6.11%
  `[-7.99%, +25.70%]`, 19 competition blocks.

The 261-match iteration odds use the legacy extractor and are not equivalent
to the new strict forward selection contract. These are the I3 block
intervals; earlier i.i.d. lower bounds above zero are superseded.

## Pre-registered slices and metrics

Report every candidate on exactly:

1. all 137 matched fixtures;
2. volume ≥$50k (61 fixtures; primary);
3. volume ≥$100k (30 fixtures; secondary and explicitly low-power).

Probability metrics:

- mean binary log loss (primary);
- mean Brier score;
- Polymarket normalized implied-probability log loss/Brier as the market
  baseline;
- reliability table/plot as descriptive output only.

Betting metrics, flat one-unit stakes:

- number of bets, total P&L, ROI, win rate, and maximum drawdown;
- I3 block-bootstrap interval, effective block count, and metadata coverage;
- no Kelly headline and no threshold sweep.

## Frozen betting policies

Report both, without choosing between them after seeing the holdout:

1. **Flat:** bet the model-preferred side only when model edge is positive.
2. **A7 conditional:** when `|top6_batting_elo_diff| ≤ 5`, require positive
   edge; when it is `> 5`, require edge strictly greater than 10 percentage
   points.

No odds threshold, ELO boundary, or liquidity cutoff may be changed after
scoring.

## Proposed one-time interpretation rule

This section requires user approval before the draft becomes frozen.

- **Probability confirmation:** on the primary ≥$50k slice, M7 log loss is
  no more than 0.01 worse than the market baseline and is better than the
  ball-sim winner probability.
- **Economic confirmation:** A7 ≥$50k requires at least 10 competition blocks
  and a 95% block-bootstrap lower bound above zero.
- If probability confirms but ROI does not, keep M7 as the winner model but
  label the betting edge unconfirmed and continue forward capture without
  threshold tuning.
- If ROI confirms but probability does not, do not claim superior
  probabilities; keep the betting result as a hypothesis requiring a new
  forward window.
- The ≥$100k slice is descriptive because 30 fixtures cannot carry a
  promotion decision alone.

The sealed ≥$50k slice contains only seven competition blocks (49 of 61
fixtures are MLC or Vitality Blast). It therefore cannot satisfy the
pre-registered economic-confirmation rule in this window, regardless of its
point ROI. Report the ROI and interval descriptively; probability confirmation
remains actionable.

## Opening conditions

All must be complete before scoring:

1. **COMPLETE:** I6 fixes and tests deterministic same-day ordering and builds
   the hashed holdout-specific sidecar. Production caches remain untouched.
2. **COMPLETE:** I3 fixes zero-P&L bet placement and defines/tests the shared
   tournament/team fallback block bootstrap used for every reported CI.
3. The scorer is tested on synthetic and legacy data only, including reversed
   team order, aliases, ties/no-results, missing odds, and all three liquidity
   boundaries. The ball-simulation path must also prove that it replays an
   earlier same-day fixture in `date_then_match_id_lexicographic_v1` order;
   its current date-only SQLite query is insufficient by itself.
4. The sidecar cache is hashed. The scoring code and final protocol must be
   hashed after I3. Any transient/per-match state artifact introduced for the
   ball-simulation replay must be generated from the sealed context, verified,
   and hashed as well.
5. The user explicitly approves this protocol. At that point change this
   status from DRAFT to FROZEN before the first model run.

## Prohibited after opening

- fitting, retraining, calibration, feature selection, or threshold tuning on
  any forward row;
- replacing or deleting difficult fixtures;
- changing aliases/deduplication based on model errors;
- reporting only a favorable liquidity, competition, or model slice;
- treating the same consumed window as a fresh test for a later iteration.
