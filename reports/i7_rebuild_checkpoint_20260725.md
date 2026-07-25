# I7 venue identity rebuild checkpoint — 2026-07-25

## Scope

This is an isolated rebuild from the unchanged 9,519-match male modeling
corpus through 2026-04-16. Later refreshed context/holdout matches were not
added to training, validation, or test. Production v3/M7 artifacts and the
consumed forward holdout were not modified.

Identity contract:

- version: `venue_aliases_v1`
- active aliases: 94
- SHA-256:
  `853b32b0ce3098dd8c0f33ba1437846f5505d50d9a425fbd37bff9c9f76745d8`

## Cache and ball materialization

- Matches preserved: 9,519
- Players preserved: 7,516
- Venue identities: 467 raw labels → **373 canonical labels**
- SQLite integrity: batting/bowling match-log parity passed; all sampled
  outcome-count conservation checks passed.
- Ball rows: 1,876,971 train; 124,292 validation; 186,667 test; zero invalid
  target rows removed.
- Feature recipe: exact current 114-feature hierarchy-shrink model,
  feature hash `c520a3ba08ae`, k-player 30 / k-venue 200.

## Raw ball-model checkpoint

| model | validation LL | test LL |
|---|---:|---:|
| Frozen v3/v7 reference | 1.638883 | **1.629655** |
| I7 canonical venues | **1.63759** | 1.6316 |
| I7 − reference | -0.00129 | +0.00195 |

The raw per-delivery result is effectively neutral and mixed: a small
validation improvement with a similarly small test regression. This is not a
promotion decision. Venue canonicalization is intended to improve historical
state quality and inference consistency; the match-winner simulation and ROI
evaluation still matter.

## Pending

1. Materialize the direct match frame from the same I7 cache.
2. Retrain the unchanged monotone M7 architecture on its exact 48-feature
   production order.
3. Compare paired direct-model predictions on the iteration set.
4. Run the I7 ball simulation on the existing iteration set, then report
   block-bootstrap LL/Brier/ROI and alias-affected slices.
5. Keep the consumed forward holdout closed; use a future untouched window
   for terminal confirmation.
