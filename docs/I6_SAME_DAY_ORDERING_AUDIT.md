# I6 same-day ordering audit

## Decision

I6 is complete as of 2026-07-23. All chronological tracker walks now order
matches by:

```text
(Cricsheet info.dates[0], filename stem / Cricsheet match ID)
```

The contract is versioned as
`date_then_match_id_lexicographic_v1`. Cricsheet does not expose a universal
scheduled-start timestamp, so the match ID is the stable secondary key.
Single- and multi-directory iteration use the same implementation, and
duplicate IDs across directories fail closed.

Production model, parquet, SQLite, and tracker-snapshot artifacts were not
overwritten. A new forward-only sidecar was built at:

```text
data/forward_state/2026-06-01_2026-07-13/
```

## Why this mattered

Trackers advance between matches on the same date. The old loader sorted only
by date, so its secondary ordering inherited filesystem enumeration order.
Copying the same files to another machine could therefore change later
same-day ELO and recent-form features.

The refreshed audit found:

| Source | Same-day groups | Matches in those groups | Groups whose legacy filesystem order differed |
|---|---:|---:|---:|
| Historical male corpus through 2026-04-16 | 2,191 | 8,046 | 1,616 |
| Forward context, 2026-04-17 through 2026-07-13 | 74 | 393 | 67 |

The old I6 backlog estimate of 77 groups was therefore too narrow; it was
based on an earlier/smaller scope.

## Forward-state guardrails

The sidecar combines 9,519 historical male matches and 401 forward-context
matches in one deterministic chronological walk. It contains 9,699 resolved
match-level feature rows; ties with an eliminator are treated as resolved.
All 137 sealed evaluation fixtures are present.

Global and phase outcome priors are deliberately copied from the production
cache frozen through 2026-04-16. Recomputing those priors over all 401 context
matches would let later fixtures influence earlier feature rows. The rejected
first build that exposed this issue was not promoted as the canonical state.

Key provenance:

| Artifact/contract | SHA-256 or value |
|---|---|
| Sealed holdout fingerprint | `82ccde16cf2b7e5f13a9236f2788f3c8be1582f312f5c028ec44a6ab76561028` |
| Frozen-prior source SQLite | `6c26c1bd6aff82eb88a37828143dcf5b4e6ddffecd1f078f9c54d56d17896fbd` |
| Forward sidecar SQLite | `c95524db29f1680c5fff9fa97a3f7f5d882ceef41075e7477d30698a730edb7c` |
| Forward golden/context parquet | `233d7b221c1e41b06399b6f4953a8dc5a4617cc4e9ab0a7a8b45bf678a3e5fc6` |
| Complete forward-state directory | `725719e6ab1951cb0fa36544ea2e710ce22f0625e5934bd3dae992e21f9b792f` |
| Same-day order | `date_then_match_id_lexicographic_v1` |
| Prior contract | `frozen_external_sqlite_v1` |

The sidecar builder never imports or scores a model and writes a
`NO_MODEL_SCORING` marker. The independent verifier rechecks the artifact
inventory and hashes, SQLite metadata, frozen-prior provenance, sealed
holdout fingerprint, and all selected feature IDs.

The materialized match rows already contain the deterministic within-day
advancement required by the direct model. The current ball-simulation
evaluator reads SQLite at date granularity; it must not be pointed at this
sidecar and assumed to have equivalent same-day state.

`scripts/sim_eval/same_day_stats.py` now supplies the missing transient layer.
It rehydrates the same production trackers once per date, forces evaluated
predictions to lock before replay, then advances actual deliveries only for a
later same-day fixture. It is an in-memory adapter around the read-only
sidecar, not a change to the simulation engine or SQLite schema. Synthetic
tests cover chronology, ordering, stale-cache invalidation, and pre-prediction
field access; a model-free smoke test replayed a real three-fixture context
date while leaving the sidecar SQLite at its recorded SHA-256.

`scripts/score_forward_ball_v7.py` now connects this provider to ball v7
behind the same frozen-protocol gate as M7. It builds lineups only from
ordered `info.players` rosters, walks all 401 context matches, and requires
the `simulate → lock → replay` order for each of the 137 selected fixtures.
The model-free context inventory and synthetic scorer tests pass, so
`ball_same_day_replay_complete` is recorded. No sealed model probability was
produced during implementation or testing.

## Historical impact audit

To quantify the consequence without opening the forward holdout, M7 was
scored on the 791 already-consumed test rows shared by the production
filesystem-ordered features and the deterministic rebuild.

| Metric | Existing features | Deterministic features | Change |
|---|---:|---:|---:|
| Log loss | 0.592629 | 0.592474 | -0.000155 |
| Brier score | 0.203845 | 0.203765 | -0.000081 |

Prediction movement was small:

- mean absolute probability change: 0.000788 (0.079 percentage points);
- 95th percentile: 0.004198 (0.420 percentage points);
- maximum: 0.051045 (5.105 percentage points);
- three of 791 predictions crossed the 0.50 side boundary.

Feature drift is widespread because an early same-day reorder propagates
through ELO histories, but it is generally numerically small. On the shared
test rows, mean absolute changes were 0.071 for batting ELO difference and
0.066 for bowling ELO difference.

This is a compatibility audit on an already-consumed set, not a new model
selection result. It does not score or reveal the sealed 137-match forward
holdout.

## Commands

```bash
uv run python scripts/build_forward_state.py \
  --holdout-dir data/forward_holdout/2026-06-01_2026-07-13

uv run python scripts/verify_forward_state.py \
  data/forward_state/2026-06-01_2026-07-13
```

The build is immutable and refuses to overwrite an existing state directory.
