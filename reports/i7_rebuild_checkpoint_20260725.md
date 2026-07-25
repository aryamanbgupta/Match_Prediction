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

1. Run the I7 ball simulation on the existing iteration set, then report
   block-bootstrap LL/Brier/ROI and alias-affected slices.
2. Keep the consumed forward holdout closed; use a future untouched window
   for terminal confirmation.

## Direct M7 checkpoint

The direct model was rebuilt with seed 29, the deployed M7 hyperparameters,
the exact 48-feature production order, and the same 10 monotonic constraints.

Standalone rows are not perfectly paired because the current materializer
also includes later deterministic-order/tie-handling fixes:

| model | train | validation | test | validation LL | test LL |
|---|---:|---:|---:|---:|---:|
| Frozen M7 | 7,912 | 525 | 791 | **0.6459** | **0.5924** |
| I7/current pipeline | 7,972 | 528 | 798 | 0.6532 | 0.5974 |

For the honest comparison, legacy prediction IDs were canonicalized and only
the 255 common resolved Polymarket fixtures were scored. The old-model row
exactly reproduces the published ≥$50k LL 0.6299 and ROI +21.90%.

| slice | n / blocks | old LL | I7 LL | ΔLL (I7−old), block 95% CI | old ROI | I7 ROI | ΔROI, block 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|
| all | 255 / 25 | **0.6254** | 0.6356 | +0.0102 [-0.0013,+0.0210] | **+15.46%** | +10.34% | -5.13pp [-10.88,+0.04] |
| ≥$50k | 168 / 19 | **0.6299** | 0.6421 | +0.0123 [-0.0022,+0.0229] | **+21.90%** | +17.49% | -4.41pp [-14.12,+2.78] |
| ≥$100k | 110 / 11 | **0.5929** | 0.6067 | +0.0138 [-0.0100,+0.0249] | **+26.39%** | +24.27% | -2.12pp [-11.34,+6.85] |

Brier changes tell the same story (all +0.00449
[-0.00065,+0.00930]); none of the probability or ROI deltas is
competition-block CI-clean. Mean absolute probability movement is 2.99
percentage points overall and 3.22 points at ≥$50k.

No evaluation JSON used an alias spelling directly: every venue was already
on its city-qualified canonical label. However, 131/255 fixtures were at
canonical targets whose history was enriched by merged aliases. Their ΔLL was
+0.0118 [-0.0024,+0.0255], versus +0.0085
[-0.0055,+0.0245] elsewhere. There is no evidence that the I7 retrain helped
the intended venue-enriched slice.

**Direct-model decision:** do not promote `models/xgb_match_i7`. The point
estimates are consistently worse, even though uncertainty does not establish
a clean regression. Keep the identity implementation and isolated artifacts;
finish the ball-simulation evaluation before closing I7.

## Match-ID collision discovered

The I7 test parquet has 798 rows but only 788 synthetic
`date_team1_team2_venue` IDs. Ten same-day doubleheaders share the same teams
and venue, so `test_predictions.json` is last-write-wins. The frozen parquet
already had nine such collisions; I7 exposed one more under the current
corpus. None overlaps the 261-match Polymarket set, so the paired evaluation
above is unaffected. I15 tracks the required durable fix: use Cricsheet ID as
the primary key and keep the synthetic string as display/join metadata.
