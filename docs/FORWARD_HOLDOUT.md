# Forward holdout: construction and use

## Purpose

The forward holdout is a locked, terminal evaluation set for model and betting
policy decisions made after the repeatedly-used 261-match iteration set. It is
not training data and it is not a validation set. Building it must not trigger
model scoring.

Current contract:

- Holdout window: 2026-06-01 through the latest date covered by both the
  strict Polymarket pull and refreshed Cricsheet archives.
- Chronological context begins 2026-04-17 so June/July features can see
  legitimate earlier April/May results.
- Model parameters remain frozen. Historical state may advance only through
  matches strictly earlier than the fixture being predicted.
- Existing train, validation, test, iteration, golden, production-cache, and
  production-model artifacts are never modified by the builder.

## Source preparation

Refresh the Cricsheet archives in `stat-generator`, then use the separate
strict extractor in `polymarket-cricket`:

```bash
uv run python extract_match_prematch_odds_strict.py \
  --cutoff 2026-06-01 --through 2026-07-23
```

The strict extractor is intentionally separate from the original incremental
extractor. It selects only exact-title head-to-head markets, excludes women's
events for the male model, requires explicit scheduled start time, and refuses
any quote that is not strictly pre-match.

The builder joins on the local fixture date persisted at the end of the
Polymarket event slug, not blindly on the UTC date of `gameStartTime`.
Cricsheet stores local match dates, so late-evening matches in the Americas
can otherwise be shifted onto the following fixture in a back-to-back series.
The builder permits at most a one-day UTC/local difference and fails closed
when the slug date is missing or inconsistent.

## Build

From this repository:

```bash
uv run python scripts/build_forward_holdout.py \
  --market-json /Users/aryamangupta/Projects/polymarket-cricket/data/polymarket_match_odds_strict_2026-06-01_2026-07-23.json \
  --start 2026-06-01 \
  --end 2026-07-13 \
  --dry-run

# Remove --dry-run only after the integrity report passes.
```

Output:

```text
data/forward_holdout/<start>_<end>/
├── SEALED
├── betting_odds.json
├── context_t20s_json/
├── diagnostics.json
├── integrity_report.json
├── manifest.json
├── polymarket_test/
└── raw/
```

Verify the sealed artifact at any time without loading a model:

```bash
uv run python scripts/verify_forward_holdout.py \
  data/forward_holdout/2026-06-01_2026-07-13
```

The verifier checks every selected Cricsheet payload hash, the copied raw
market-source hash, manifest/odds IDs, all quote-before-start timestamps,
current overlap with older pools, and prints a fingerprint for the complete
sealed directory.

`context_t20s_json/` is bridge/live-state input only. Its contents must never
be appended to the existing training corpus or used to refit a model. The
matched `polymarket_test/` and `betting_odds.json` remain sealed until the
candidate models, betting rules, metrics, and slices are frozen.

## Guardrails

Construction fails on:

- post-start or missing quote/start timestamps;
- missing or inconsistent local fixture dates in market slugs;
- props, non-male events, unresolved markets, malformed probabilities;
- ambiguous date/team matches;
- selected market winner disagreement with Cricsheet after outcome-blind
  market selection;
- overlap with existing training or evaluated match pools;
- Cricsheet ID collisions with different payloads;
- attempts to overwrite an existing sealed dataset.

Duplicate fixture selection never uses the realized winner. It ranks only by
integrity eligibility, volume, quote recency, and stable market ID.
Team-name joins use an explicit alias table only; the builder never performs
fuzzy or result-assisted matching. Ambiguous same-day doubleheaders remain
unmatched.

## Chronological feature state

I6 completed this prerequisite on 2026-07-23. The later scoring workflow must
use the dedicated sidecar at
`data/forward_state/2026-06-01_2026-07-13/`, built from:

1. the existing corpus through 2026-04-16;
2. `context_t20s_json/` from 2026-04-17 onward.

Queries remain as-of the fixture date. Same-day matches use the versioned
`(match_date, Cricsheet match_id)` contract
`date_then_match_id_lexicographic_v1`. Global and phase outcome priors are
frozen from `models/player_stats_cache_v3.sqlite` (the pre-holdout cache);
recomputing them over all context fixtures would let later outcomes influence
earlier feature rows. Production cache, parquet, tracker snapshot, and model
artifacts remain untouched.

The older `tmp/golden_inclusive/` cache is not an input to this evaluation:
it predates the refreshed corpus and its source directory is incomplete.
Rebuilding state is allowed; retraining on the forward context or holdout is
not.

Build and verify without loading a model:

```bash
uv run python scripts/build_forward_state.py \
  --holdout-dir data/forward_holdout/2026-06-01_2026-07-13

uv run python scripts/verify_forward_state.py \
  data/forward_state/2026-06-01_2026-07-13
```

The current immutable sidecar contains 401 golden/context feature rows and
verifies all 137 selected fixtures. Its SQLite SHA-256 is
`c95524db29f1680c5fff9fa97a3f7f5d882ceef41075e7477d30698a730edb7c`.
See `docs/I6_SAME_DAY_ORDERING_AUDIT.md` for the ordering and compatibility
audit.

The direct match model must read its selected rows from the materialized
golden/context parquet. The current ball-simulation evaluator queries SQLite
at date granularity, so it cannot recover an earlier same-day sibling from
SQLite alone. Before opening the holdout, its scorer must walk the sealed
context in the same versioned order and advance transient state only after
each earlier match. Pointing the existing evaluator at the sidecar SQLite
without that replay is prohibited.

`scripts/sim_eval/same_day_stats.py` implements the transient stats half of
that contract. It reads only `info` fields before prediction, requires an
evaluated prediction to be locked before delivery replay, advances the
rehydrated trackers in memory, and invalidates the simulation stats memo after
every match. It never writes the sidecar. The ball scorer that orchestrates
this layer is still pending, so the machine-readable replay opening condition
remains false.

`scripts/score_forward_match_m7.py` implements the direct-model half behind
the frozen-protocol gate. Its parquet projection excludes target/outcome
fields, validates all 137 identities and team orientations, and emits a
write-once prediction artifact plus SHA-256 sidecar. Do not run it on the
sealed set while the machine-readable protocol remains `DRAFT`.

## Opening the holdout

Before any model probabilities are scored, freeze and document:

- incumbent and challenger model artifact hashes;
- whether the A7 conditional betting filter is included;
- odds/edge thresholds and sizing;
- LL/Brier/ROI metrics and uncertainty method;
- all/≥$50k/≥$100k slices;
- the one-time decision rule.

The proposed choices and current artifact hashes are in
`docs/FORWARD_EVALUATION_PROTOCOL_DRAFT.md`. It is deliberately marked DRAFT:
I3 and I6 are complete, but review and freeze it only after the scorer tests
and ball-simulation same-day replay guard are complete.

The tracked machine-readable contract is
`evaluation/forward_protocol_2026-06-01_2026-07-13.yaml`. Its model-free
preflight verifies both sealed fingerprints, all candidate artifact hashes,
the three liquidity counts, and every opening condition:

```bash
uv run python scripts/forward_eval_contract.py \
  evaluation/forward_protocol_2026-06-01_2026-07-13.yaml
```

After scoring, this holdout is consumed. Subsequent tuning requires a new
forward window.
