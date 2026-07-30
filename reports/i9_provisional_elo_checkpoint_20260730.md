# I9 provisional-ELO checkpoint — 2026-07-30

## Decision

`FAILED`. The fixed provisional schedule produced encouraging point
improvements in the ball model, but the precommitted provisional-event
interval crossed zero. The five-seed direct-model guardrail also regressed.
No production, serving, betting, or forward-holdout artifact was changed.

## Frozen change

I9 changed only player-ELO K during the first 120 rated role deliveries:

```text
multiplier(n) = 1 + 3 * max(0, 1 - n / 120)
K_role(n) = K_base * multiplier(n)
```

Batting and bowling counts are independent. Exposure advances exactly when
the baseline ELO update runs. The I7 identity stack, inclusive-total-run
delivery semantics, 114-feature ball recipe, splits, priors, XGBoost
hyperparameters, and direct M7 feature/architecture contract stayed fixed.
I8 was disabled.

## Implementation and contracts

- `PlayerEloTracker` carries independent role exposure and applies exact
  multipliers 4.0, 2.5, 1.0, and 1.0 at exposures 0, 60, 120, and 121.
- SQLite rehydration derives role exposure from as-of batting/bowling
  delivery counts.
- Same-day replay retains transient exposure and matches one uninterrupted
  chronological pass.
- SQLite, ball parquet, match parquet, ball model, direct model, tracker
  snapshot, and serving loaders carry or validate `elo_update_version`.
- Legacy-missing provenance resolves only to `fixed_competition_k_v1`;
  it cannot be served as provisional state.
- Candidate artifacts are isolated under the `i9` namespace. The paired
  fixed-K control is isolated under `i9_baseline`.

Focused schedule/state/provenance/evaluation tests: 22 passed. The broader
suite reported 259 passed plus four pre-existing collection errors in
`test_xgboost_model_v2_encoder_cache.py`, whose functions request undefined
pytest fixtures.

## Paired ball-model rebuild

Both arms used 9,519 male matches, 7,516 players, 373 canonical venues, and
the exact same rows:

| split | deliveries |
|---|---:|
| train | 1,876,971 |
| validation | 124,292 |
| test | 186,667 |

The fixed-K control reproduced the earlier I7 checkpoint exactly:

| arm | validation LL | test LL |
|---|---:|---:|
| fixed-K control | 1.63759 | 1.63162 |
| provisional candidate | **1.63585** | **1.62762** |
| candidate − control | -0.00174 | -0.00400 |

The test split is diagnostic only.

## Frozen validation gate

Deltas are candidate minus fixed-K control; negative is better. The interval
uses 10,000 paired match-block resamples with seed 29.

| slice / metric | control | candidate | delta | result |
|---|---:|---:|---:|---|
| provisional LL, 30,241 balls / 506 matches | 1.60937 | 1.60827 | -0.00110, CI [-0.00759,+0.00541] | **primary failed** |
| all LL | 1.63759 | 1.63585 | -0.00174 | guardrail passed |
| all multiclass Brier | 0.78269 | 0.78204 | -0.00065 | guardrail passed |
| established-vs-established LL, 94,051 balls | 1.64667 | 1.64472 | -0.00195 | guardrail passed |

The candidate is directionally better everywhere, but the primary
provisional-event uncertainty interval includes zero.

## Direct-model five-seed guardrail

Both arms used the exact M7 48-feature model contract on 7,972 train, 528
validation, and 798 test matches.

| seed | fixed-K validation LL | provisional validation LL | delta |
|---:|---:|---:|---:|
| 7 | 0.65440 | 0.65556 | +0.00115 |
| 13 | 0.65320 | 0.65515 | +0.00194 |
| 29 | 0.65316 | 0.65574 | +0.00258 |
| 42 | 0.65182 | 0.65348 | +0.00166 |
| 101 | 0.65120 | 0.65571 | +0.00451 |
| **mean** | **0.65276** | **0.65513** | **+0.00237** |

The candidate was worse at every seed, so the direct guardrail failed.

## Artifacts

- Machine-readable decision:
  `reports/i9_development_gate_20260730.json`
- Ball control:
  `models/player_stats_cache_i9_baseline.sqlite`,
  `data/xgb_data_i9_baseline/`, `models/xgb_i9_baseline/`
- Ball candidate:
  `models/player_stats_cache_i9.sqlite`, `data/xgb_data_i9/`,
  `models/xgb_i9/`
- Direct frames:
  `data/xgb_match_data_i9_baseline/`, `data/xgb_match_data_i9/`
- Direct models:
  `models/xgb_match_i9_baseline_seed{29,7,13,42,101}/` and
  `models/xgb_match_i9_seed{29,7,13,42,101}/`

The exact recipe and CLI commands are in
`docs/I9_PROVISIONAL_ELO_EXPERIMENT.md`.

## Review note (2026-07-30): exposure-contract inconsistency, disclosed

Under the legacy delivery semantics this experiment used, the live ELO
update runs on every delivery (wides/no-balls included), and exposure
advances with it — but SQLite rehydration seeds exposure from the
legal-ball-only batting/bowling counters. Rehydrated exposure is therefore
lower than an uninterrupted chronological pass by each player's career
extras faced (~4–5% of deliveries), so the "same-day replay matches one
chronological pass" property holds only on extras-free fixtures — which is
exactly what the parity test uses. This does **not** overturn the verdict:
both arms shared the same inconsistency in a paired design, exposure only
affects updates made after rehydration, and the direct-model guardrail
failed at all five seeds regardless. But any re-opening of I9 must first
pick one side of the contract (most plausibly: advance exposure only on
legal deliveries, matching the rehydration counters), add an extras-bearing
fixture to the parity test, and re-precommit.

## Disposition

Do not promote I9 and do not tune the 120-delivery threshold or 4× multiplier
against this validation result. A different schedule is a new hypothesis
that requires a new precommitment. Current production and live-serving
artifacts remain fixed-K.
