# Repository consolidation and model lifecycle

## Decision

The long-term repository should have **one supported current pipeline**.
`legacy`, `i7`, and `i8` are not three products that should remain selectable
forever:

- **I7 venue identity is a correctness invariant** for all newly built and
  future data, state, training, evaluation, and inference.
- **I8 is a candidate feature bundle**, not a second identity mode or a
  permanent serving branch.
- **Legacy is a temporary replay/rollback bridge** for artifacts built before
  I7. It must not keep expanding the main pipeline.

Historical reproducibility does not require every historical behavior to stay
live in the core code. It requires an immutable experiment config, artifact
manifest, evaluation report, source commit/tag, and rebuild command.

## Why the repository currently feels more complicated

The recent work solved different problems but exposed them through similar
version labels:

| label | what it actually means | long-term treatment |
|---|---|---|
| frozen v3/v7 and M7 | current pre-I7 production artifacts | temporary legacy replay only |
| I6 | deterministic same-day ordering and held-out context construction | universal invariant |
| I7 | canonical venue identity and provenance | universal invariant |
| I8 | 18 phase/H2H ball features on schema v5 | candidate model bundle |

The complexity comes from treating artifact compatibility as if it were a
domain choice. A fixture does not meaningfully have a “legacy venue” and an
“I7 venue”; it has one canonical venue. The old spelling behavior exists only
because an older model and cache were trained with fragmented identities.

## Target architecture

The main path should eventually be:

```text
fixture
  -> canonical identity (I7 invariant)
  -> load one declared model bundle
  -> validate its manifest
  -> load the matching state/schema
  -> predict or simulate
```

There should be no `--venue-identity-mode` on the final current command.
Compatibility should be resolved once when the model bundle is loaded, not
through repeated feature-level branches.

The repository should expose three lifecycle areas:

| area | support level | allowed behavior |
|---|---|---|
| current | fully tested and operational | one identity contract, one default bundle |
| experiments | isolated and fail-closed | may add schemas/features; never silently serve |
| archive/replay | frozen, read-only, temporary | reproduce a named historical artifact only |

An experiment is either promoted and replaces the current bundle, or rejected
and removed from active runtime registration. It does not become another
permanent mode.

## One model-bundle manifest

The current path should load a single generic manifest instead of checking
generation names in several places. At minimum it should declare:

- model family and artifact paths;
- feature count, ordered feature hash, and outcome semantics;
- stats schema version;
- venue identity map version, SHA-256, and row count;
- same-day ordering version;
- delivery semantics;
- shrinkage/prior contract and prior hash;
- training-data cutoff; and
- source commit/release identifier.

The loader should validate these fields generically and fail closed. The
runtime should not need to know whether the bundle was historically called
I7, I8, or something later.

## Consolidation sequence

### 1. Make canonical identity the only main-path behavior

After approval, change the primary live command so I7 canonicalization is the
default and only identity behavior. Move the frozen pre-I7 invocation behind
a clearly named temporary replay command such as
`scripts/legacy/predict_fixture_v3.py`.

This is a behavioral migration and should be a separate reviewed commit. The
current compatibility implementation remains necessary until that decision
is made; it should not receive additional features.

### 2. Replace mode flags with bundle validation

Create the generic bundle manifest/loader, migrate the canonical live command
to it, and remove identity-generation checks from feature code. Keep
canonicalization at ingestion boundaries and validate the resulting contract
at artifact load.

### 3. Resolve I8 once a new terminal window exists

I8 remains isolated while post-2026-07-30 fixtures accumulate:

- if promoted, its schema-v5/132-feature bundle becomes `current`; callers do
  not select `i8`;
- if rejected, remove I8 from active runner registration and retain only its
  config, report, source commit, and rebuild instructions.

Do not preserve I7 and I8 as two indefinite serving modes.

### 4. Retire legacy replay

Once a canonical bundle has an operational smoke test, documented rollback,
and accepted live behavior, tag the final legacy source/artifacts and remove
the temporary replay path from the main branch. Rollback then means selecting
the tagged release, not carrying old semantics in every new release.

The retirement decision does not require pretending the new model performs
better. Predictive performance and data correctness are separate: the
performance change should be measured and disclosed, while canonical identity
remains the forward standard.

## Venue evolution over time

A stadium changing over time is real, but duplicate names are not the right
representation. Keep one stable canonical venue ID and attach as-of,
time-varying state such as:

- boundary dimensions and configuration;
- altitude;
- season/month weather normals;
- surface or renovation era;
- rolling scoring and chase characteristics; and
- an effective-date interval for each physical configuration.

Feature queries must select the state known before the match date. If a ground
is materially rebuilt, represent that as a dated venue-state/era record linked
to the same venue identity. Do not manufacture a second categorical venue by
changing punctuation or appending/removing the city.

## Work completed through 2026-07-30

- I6 made multi-directory cache walks deterministic by
  `(match_date, Cricsheet match_id)` and kept refreshed fixtures out of model
  training/test splits.
- I7 reviewed 94 exact venue aliases, rejected unsupported player merges,
  propagated identity provenance across artifacts, and added fail-closed live
  compatibility.
- The I7 retrained models were not promoted, but I7 remains the required
  forward identity contract.
- I8 added schema-v5 player-phase and batter-bowler counts/readers, 18
  hierarchical pre-ball features, isolated training artifacts, and a
  fail-closed simulator.
- I8 improved ball-level test log loss from 1.631620 to 1.628690 and Brier
  from 0.781231 to 0.779532. On the consumed match diagnostic, log loss
  improved from 0.7042 to 0.6825 while flat ROI fell from +0.46% to -1.49%.
  It remains unpromoted pending a new untouched post-2026-07-30 window.

Detailed contracts and results:

- `docs/I6_SAME_DAY_ORDERING_AUDIT.md`
- `docs/I7_VENUE_IDENTITY_CONTRACT.md`
- `docs/I7_LIVE_COMPATIBILITY.md`
- `docs/I8_FEATURE_CONTRACT.md`
- `reports/i7_rebuild_checkpoint_20260725.md`
- `reports/i8_phase_matchup_checkpoint_20260730.md`

## Rules for future work

1. Do not add another identity mode.
2. Do not make experiment IDs part of public runtime semantics.
3. New experiments must use isolated artifacts and fail closed.
4. Promotion replaces the current bundle; rejection removes active hooks.
5. Preserve reproducibility through manifests, reports, configs, and source
   history rather than permanent compatibility branches.
