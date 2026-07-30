# I9 provisional ELO experiment

Status: implemented and evaluated 2026-07-30; development gate failed.

## Question

Does a fixed provisional K schedule let a new batter or bowler reach a useful
rating faster without degrading established-player or overall probability
calibration?

This experiment changes only the ELO update schedule. It does not change
identity, delivery labels, features, model hyperparameters, splits, priors, or
betting rules.

## Precommitted candidate

Maintain independent rated-delivery exposure counts for each player's batting
and bowling roles. Before an update, for role exposure `n` and the existing
competition K factor `K_base`:

```text
provisional_multiplier(n) = 1 + 3 * max(0, 1 - n / 120)
K_role(n) = K_base * provisional_multiplier(n)
```

Therefore the multiplier is 4.0 on the first rated delivery, decreases
linearly, and is exactly 1.0 from the 121st rated delivery onward. Batting and
bowling K values are calculated separately; an established bowler facing a
debutant batter does not become provisional again.

An exposure count advances exactly when the corresponding baseline
`PlayerEloTracker.update` would run. This deliberately preserves the selected
delivery-semantics contract rather than mixing an extras/legal-ball change
into I9.

> **Contract verified on extras (2026-07-30, superseding a same-day review
> note):** a review note first claimed live updates (every delivery) and
> rehydration (the `balls` counters) diverge on wides/no-balls. Writing the
> extras-bearing parity test refuted that claim: under legacy semantics the
> stats path never passes `is_legal`, so the `balls` counters are inclusive
> of extras and match the all-delivery updates exactly; under i5 semantics
> both sides are legal-only. The contract is consistent in both modes and
> is now pinned by
> `test_i9_provisional_elo.py::test_exposure_parity_holds_on_extras_bearing_matches`.

No threshold or multiplier sweep is allowed in the first experiment. If this
fixed candidate fails, changing 120 or 4.0 is a new hypothesis.

## State and artifact contract

Implementation must:

- add batting and bowling exposure maps to `PlayerEloTracker`;
- rehydrate their as-of counts from the matching batting/bowling statistics;
- preserve them across same-day transient updates;
- add `elo_update_version = provisional_linear_120_x4_v1` to SQLite,
  parquet, model, tracker-snapshot, and bundle manifests;
- reject an I9 model paired with baseline ELO state, and vice versa; and
- rebuild in isolated I9 cache/data/model directories.

The baseline is the canonical I7 identity stack with its existing delivery
semantics and feature recipe. I8 features remain disabled for the first I9
comparison so only the ELO schedule changes.

## Tests before model training

- exact K values at exposures 0, 60, 120, and 121;
- batting and bowling counts advance independently;
- an established opponent retains baseline K;
- rehydrated state produces the same next update as uninterrupted state;
- same-day replay produces byte-identical state to one chronological pass;
- baseline mode remains numerically unchanged; and
- missing or mismatched ELO-version provenance fails closed.

## Development evaluation

All choices above are frozen before rebuilding.

Use the existing training split for fitting and validation split for the
development decision. Define a provisional event using only pre-ball state:
the batter has fewer than 120 prior rated batting deliveries or the bowler has
fewer than 120 prior rated bowling deliveries.

Primary development gate:

- validation ball-level log loss on provisional events must improve, with the
  upper bound of a paired match-block bootstrap interval for
  `candidate - baseline` below zero.

Guardrails:

- overall validation ball log loss and Brier may each regress by at most
  0.001 absolute;
- established-vs-established events, where both role counts are at least 120,
  may regress by at most 0.001 log loss; and
- the direct match model, trained at the fixed A1 seeds
  `{29, 7, 13, 42, 101}`, must not have worse mean validation log loss than
  its paired baseline.

The repeatedly used test, 261-match Polymarket, and consumed forward-holdout
sets may be reported only as diagnostics; they cannot promote or tune I9.

## Promotion gate

Passing the development gate makes I9 an eligible candidate, not a promoted
model. Promotion requires a new untouched post-2026-07-30 forward window:

- probability performance must satisfy that window's frozen decision rule;
- the provisional-fixture slice must not show a confidence-clean regression;
- no economic claim is made unless its tournament-block interval clears the
  predeclared economic gate; and
- failure leaves current serving artifacts unchanged.

## Result

The isolated fixed-K control exactly reproduced the earlier I7 ball
checkpoint (validation LL 1.6376, test LL 1.6316), confirming that baseline
numerics did not drift. The provisional candidate improved aggregate
validation LL to 1.6358 and diagnostic test LL to 1.6276.

The binding provisional-event gate did not clear:

| validation slice | baseline LL | candidate LL | candidate − baseline | paired match-block 95% CI |
|---|---:|---:|---:|---:|
| provisional, 30,241 balls | 1.60937 | 1.60827 | -0.00110 | [-0.00759, +0.00541] |
| established, 94,051 balls | 1.64667 | 1.64472 | -0.00195 | guardrail passed |
| all, 124,292 balls | 1.63759 | 1.63585 | -0.00174 | guardrail passed |

Overall multiclass Brier also improved by 0.00065, so every ball-model
guardrail passed. However, the primary interval includes zero.

The fixed five-seed direct-model guardrail also failed. Mean validation LL
moved from 0.65276 for the fixed-K control to 0.65513 for the provisional
candidate, a regression of 0.00237; the candidate was worse at every seed.

Decision: `FAILED`. Keep the implementation and isolated evidence artifacts
for reproducibility, do not tune 120 or 4× in response to this result, and do
not change production or live-serving defaults.

The machine-readable gate is
`reports/i9_development_gate_20260730.json`; the narrative checkpoint is
`reports/i9_provisional_elo_checkpoint_20260730.md`.

## Reproduce

```bash
# Paired ball control and candidate.
uv run python scripts/run_experiment.py \
  experiments/configs/xgb_i9_baseline.yaml
uv run python scripts/run_experiment.py \
  experiments/configs/xgb_i9_provisional_elo.yaml

# Paired direct M7 frames.
uv run python scripts/materialize_match_features.py \
  --version i9_baseline \
  --out-dir data/xgb_match_data_i9_baseline_full \
  --elo-update-version fixed_competition_k_v1
uv run python scripts/build_i7_match_frame.py \
  --source-dir data/xgb_match_data_i9_baseline_full \
  --out-dir data/xgb_match_data_i9_baseline
uv run python scripts/materialize_match_features.py \
  --version i9 \
  --out-dir data/xgb_match_data_i9_full \
  --elo-update-version provisional_linear_120_x4_v1
uv run python scripts/build_i9_match_frame.py
uv run python scripts/run_i9_direct_seeds.py

# Final frozen gate. Pass the five seed-paired train_metrics.json files
# through --direct-pair as shown in the checkpoint report.
uv run python scripts/evaluate_i9_provisional_elo.py \
  --baseline \
    models/xgb_i9_baseline/validation_predictions_i9_baseline.parquet \
  --candidate models/xgb_i9/validation_predictions_i9.parquet \
  --out reports/i9_development_gate_20260730.json
```
