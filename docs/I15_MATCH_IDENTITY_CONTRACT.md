# I15 stable match identity contract

## Decision

All newly built match artifacts use the immutable Cricsheet JSON file stem as
their primary `match_id`.

The historical
`{date}_{team1}_{team2}_{canonical_venue}` value remains useful to people and
for compatibility with frozen artifacts, but is now
`display_match_id`. It is not a primary key because two fixtures can share it.

The versioned row contract is:

```json
{
  "match_id": "1477609",
  "cricsheet_id": "1477609",
  "display_match_id": "2026-01-27_West_Indies_South_Africa_Boland_Park,_Paarl",
  "match_identity_version": "cricsheet_primary_v1"
}
```

For `cricsheet_primary_v1`:

- `match_id` and `cricsheet_id` are required, non-empty, and equal;
- `display_match_id` is required but is not assumed unique;
- duplicate primary IDs are fatal; and
- any one-to-many compatibility alias is fatal when a caller attempts to use
  it.

The machine-readable artifact contract is:

```json
{
  "match_identity_version": "cricsheet_primary_v1",
  "primary_key": "cricsheet_id",
  "display_key": "display_match_id"
}
```

## Frozen-artifact compatibility

Frozen odds, prediction, evaluation, and consumed forward-holdout artifacts
are not rewritten. Rows with no match-identity version remain
`synthetic_fixture_v1`:

- a present `cricsheet_id` becomes the resolved primary ID;
- the old `match_id` remains the display alias;
- a row with only the old `match_id` can be joined through that alias only
  when the alias is unique; and
- a collision raises an error rather than selecting the last row.

This preserves the published 261-match and consumed forward results. None of
the known legacy-key collisions occurs in the 261-match Polymarket subset.

## Propagation

The contract now flows through:

- match feature materialization and future match-model identity sidecars;
- match-model test prediction JSON;
- Polymarket odds construction;
- simulation match loading, evaluation, blending, reslicing, walk-forward,
  and sizing analysis;
- prop-backtest match details; and
- future forward-holdout manifests, odds, scorers, verifier, and outcome
  evaluator.

Future forward holdouts use manifest schema 2. The already-consumed
2026-06-01 through 2026-07-13 holdout remains schema 1 and readable through
the legacy path.

## Collision audit

Both current I7 test parquets contain:

| artifact | rows | legacy IDs | Cricsheet IDs | rows lost by legacy keying |
|---|---:|---:|---:|---:|
| `data/xgb_match_data_i7/test.parquet` | 798 | 788 | 798 | 10 |
| `data/xgb_match_data_i7_full/test.parquet` | 798 | 788 | 798 | 10 |

The old prediction dictionary used last-write-wins behavior for those ten
rows. New prediction construction rejects a duplicate primary before writing.

## Verification

Regression coverage includes:

- unique primaries for a synthetic doubleheader;
- rejection of an ambiguous display alias;
- rejection when a new row has `match_id != cricsheet_id`;
- duplicate odds-primary rejection;
- use of the input filename stem by the simulation loader; and
- a new evaluation row joining a frozen legacy odds row by its unique display
  alias.

The real compatibility smoke loaded all 261 frozen odds rows and resolved
Cricsheet match `1477609` through its legacy display ID.

