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

## Ball-simulation evaluation

Both arms were evaluated on the same already-consumed 261-match Polymarket
iteration set with 100 simulations per fixture, deterministic seed 42, the
empirical bowler selector, the current D15 simulator, and 10,000
competition-time-block bootstrap resamples. Six unresolved fixtures have no
winner score, leaving 255 paired probability rows.

The previously published raw-v7 headline (LL 0.7158, Brier 0.2529, ROI
+7.96%) was generated before later simulator correctness changes. Comparing
I7 only to that artifact would mix model/identity changes with simulator-era
changes. A fresh control therefore ran the frozen v3 ball model and frozen
v3 stats cache through the exact current simulator used by I7. The old cache's
missing I6 metadata was retained deliberately; rebuilding it would cease to
be the frozen model control.

### Same-simulator paired result

Deltas are I7 minus the fresh frozen-v3/v7 control, so lower probability
deltas and higher ROI deltas are better. The ROI delta bootstrap resamples
the same competition blocks and recomputes each strategy's own P&L/bet-count
ratio; one exact model-market tie makes the control place one fewer bet in the
all and >=$50k slices.

| slice | scored / blocks | frozen LL | I7 LL | delta LL, block 95% CI | frozen Brier | I7 Brier | delta Brier, block 95% CI | frozen ROI (bets) | I7 ROI (bets) | delta ROI, block 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 255 / 25 | **0.6845** | 0.7042 | +0.0197 [-0.0124,+0.0506] | **0.2433** | 0.2530 | +0.0097 [-0.0046,+0.0238] | **+9.07%** (254) | +0.46% (255) | -8.61pp [-29.06,+7.73] |
| >=$50k | 168 / 19 | **0.6970** | 0.7176 | +0.0207 [-0.0076,+0.0581] | **0.2499** | 0.2607 | +0.0107 [-0.0027,+0.0281] | **+9.79%** (167) | +1.38% (168) | -8.41pp [-32.86,+10.30] |
| >=$100k | 110 / 11 | **0.6688** | 0.6770 | +0.0082 [-0.0270,+0.0403] | **0.2378** | 0.2418 | +0.0040 [-0.0127,+0.0187] | **+11.39%** (110) | +6.08% (110) | -5.31pp [-15.57,+13.37] |

Every I7 point estimate is worse than the same-simulator frozen control.
None is confidence-clean: all block intervals cross zero. Mean absolute
match-probability movement is 10.35 percentage points overall, 11.01 points
at >=$50k, and 11.33 points at >=$100k.

### Venue-enriched diagnostic

A fixture is "venue enriched" when its canonical ground is the target of at
least one active alias and therefore received additional merged history in
I7. Evaluation rows already use canonical city-qualified venue labels; this
slice tests the downstream effect of the newly pooled training history.

| slice | scored / blocks | frozen LL | I7 LL | delta LL, block 95% CI | frozen ROI (bets) | I7 ROI (bets) | delta ROI, block 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|
| venue enriched | 131 / 16 | **0.6959** | 0.7160 | +0.0201 [-0.0442,+0.0718] | **+15.01%** (130) | -0.30% (131) | -15.31pp [-45.44,+19.20] |
| other venues | 124 / 18 | **0.6725** | 0.6918 | +0.0193 [-0.0148,+0.0513] | **+2.84%** (124) | +1.26% (124) | -1.58pp [-17.69,+12.81] |

The intended venue-enriched slice does not improve and its LL movement is
nearly identical to the other-venue slice. The I7 all-slice profit is also
upset-fragile: +1.17 units becomes -17.83 units, or -7.02% ROI over the
remaining 254 bets, when the single France win at 5% market probability is
removed. The fresh control falls from +9.07% to +1.59% under the same removal.

**Ball-model decision:** do not promote `models/xgb_i7`. The ball-level
validation/test checkpoint was neutral, while the same-simulator match
simulation is directionally worse on every headline slice and shows no
special gain where merged venue history should help.

The consumed 2026-06-01 through 2026-07-13 forward holdout remained closed.
A future untouched window is still required before any later model candidate
can receive terminal confirmation.

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
the completed ball-simulation result above also rejects model promotion.

## Final I7 disposition

The reviewed exact identity map, shared canonicalization code, provenance
contracts, rebuild recipe, and fail-closed artifact checks are retained. The
player-merge premise remains rejected. Neither retrained model is promoted:
production v3/v7 and M7 model artifacts remain frozen, and the isolated I7
artifacts remain diagnostic only.

Because the active identity contract deliberately rejects legacy live state
or model artifacts that lack its provenance, the current live prediction path
must not silently relabel the frozen production models as I7-compatible. The
next operational decision is either to keep I7 opt-in until a model passes its
gate, or implement an explicit, separately evaluated legacy-compatibility
mode. Do not bypass the guard by copying I7 metadata onto old artifacts.

## Match-ID collision discovered

The I7 test parquet has 798 rows but only 788 synthetic
`date_team1_team2_venue` IDs. Ten same-day doubleheaders share the same teams
and venue, so `test_predictions.json` is last-write-wins. The frozen parquet
already had nine such collisions; I7 exposed one more under the current
corpus. None overlaps the 261-match Polymarket set, so the paired evaluation
above is unaffected. I15 tracks the required durable fix: use Cricsheet ID as
the primary key and keep the synthetic string as display/join metadata.
