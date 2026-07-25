# I7 venue identity contract

## Decision

The reviewed `venue_aliases_v1` map is active from 2026-07-25. It contains
94 exact aliases across 92 canonical venue components. Canonical labels favor
the most specific city-qualified Cricsheet spelling. Raw Cricsheet JSON is
never rewritten.

The separate player-identity audit did not support any merges. The 94 repeated
display names have no shared Cricinfo ID or full-name/date-of-birth signature
and are treated as distinct homonyms. There is no player merge map.

## Runtime behavior

`scripts/identity_maps.py` is the only venue canonicalization implementation.
It:

- consumes only rows marked `active`;
- performs exact-string matching (no fuzzy, substring, or city inference);
- rejects blank labels, self-aliases, duplicate aliases, unsupported versions
  or statuses, and active alias chains;
- leaves unknown venue strings unchanged; and
- exposes the map version, SHA-256, and active-row count as an artifact
  contract.

The same function is used by the SQLite cache walk, ball parser, direct
match-feature materializer, live-fixture path, same-day replay, Polymarket
builders, forward-holdout builder, and evaluation match-ID generation.
Legacy odds/evaluation IDs are normalized at load time by replacing an exact
venue suffix; raw historical odds files remain unchanged.

## Artifact compatibility

Venue histories and categorical encoders depend on the canonical labels.
Artifacts built before activation must not be mixed with code or artifacts
built after activation. The following outputs now record the venue identity
contract and fail closed on a missing or different contract:

- SQLite `_meta`;
- ball parquet `.feature_hash`;
- match parquet `venue_identity.json`;
- ball-model training contract;
- match-model `venue_identity.json`;
- live tracker snapshots; and
- new strict Polymarket holdout manifests and odds files.

Smart-cache checks treat an old map hash as a cache miss. Live inference
requires both refreshed state and a retrained model carrying the current
contract.

## Required rebuild order

1. Rebuild the SQLite cache from the approved corpus.
2. Re-materialize ball and match features from that cache.
3. Retrain the ball-level and match-level models, including their venue
   encoders.
4. Run the existing iteration evaluation and compare paired predictions,
   calibration, log loss, Brier score, and ROI against the frozen baseline.
5. Do not use the consumed 2026-06-01 to 2026-07-13 forward holdout for model
   selection. A new post-decision terminal window is required for final
   confirmation.

The isolated reproducible commands are:

```bash
# Cache + ball parquet + ball model; simulation evaluation intentionally skips.
uv run python scripts/run_experiment.py \
  experiments/configs/xgb_i7_venue_identity.yaml

# Full match materialization from the same I7 cache.
uv run python scripts/materialize_match_features.py \
  --version i7 \
  --out-dir data/xgb_match_data_i7_full

# Restrict to the exact current M7 production feature order.
uv run python scripts/build_i7_match_frame.py

# Retrain the unchanged M7 architecture in an isolated directory.
uv run python scripts/xgboost_match_v1.py \
  --cmd both \
  --data-dir data/xgb_match_data_i7 \
  --model-dir models/xgb_match_i7 \
  --monotone
```

The modeling corpus remains the existing 9,519 male matches through
2026-04-16. Later refreshed matches are held-out evaluation/context data and
must not enter this I7 retrain.

The already-consumed forward holdout and its locked predictions remain
immutable historical artifacts. Do not rebuild them merely to attach the new
identity contract.

## Polymarket-specific rule

`build_polymarket_odds.py` and `build_forward_holdout.py` canonicalize the
Cricsheet venue before constructing
`{date}_{team1}_{team2}_{venue-with-underscores}`. Market/team matching is
unchanged. The venue map is not used to fuzzy-match Polymarket markets, inspect
outcomes, or select duplicates. New strict holdouts persist the identity
contract in both `manifest.json` and `betting_odds.json`.
