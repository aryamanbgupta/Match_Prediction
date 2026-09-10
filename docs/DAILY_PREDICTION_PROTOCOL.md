# Daily prediction protocol

Status: fixed before collection  
Protocol version: 1  
Governing cohort: `daily_v1`  
Effective date: 2026-09-10

## Purpose and seal

This protocol governs the next prospective daily match-prediction cohort. It
is fixed before the first collection run and before any line under `daily/`
is committed. Collection, prediction, settlement, and scoring are
append-only: a correction or retry creates another record and never rewrites
prior evidence.

The single scored observation for a fixture is based on its T-60 run. This
keeps the toss-blind model and the market on the same information set. The
toss is typically called about 30 minutes before the start and has moved The
Hundred market price by a mean 3.5 percentage points; the last quote before
the start is therefore normally post-toss and is diagnostic, not the
pre-toss headline.

## Cohort and record identity

Every prediction line has the identity:

`(cohort_id, fixture_id, run_kind, attempt_ts)`

For this version, `cohort_id` is `daily_v1`. Any change to this protocol,
including its collection window, eligibility rule, or scored information
set, starts a new `cohort_id`; old records retain their original cohort.

`attempt_ts` is the creation time stamped by the writer from its own clock at
the moment the line is appended. The writer API has no caller parameter for
`attempt_ts`, so callers cannot supply or backdate it. Every line also records
`quote_ts`, the capture time of the quote it used, and `scheduled_start`.
The `quote_ts` field is present but null when no quote exists. Retries are new
lines with a new writer-stamped `attempt_ts`. Nothing is ever rewritten, and
no line contains a pointer to another line.

An already-present four-part identity is rejected rather than overwritten.
Timestamps are timezone-aware instants and are compared in UTC.

## Run kinds and information modes

| Field | Value | Contract |
|---|---|---|
| `run_kind` | `t60` | Scheduled run with toss unknown and either a projected or confirmed lineup. This is the only run kind eligible for scoring. |
| `run_kind` | `toss` | Run only when both a toss result and a confirmed eleven have been supplied. It is never scheduled blindly and is always diagnostic. |
| `lineup_mode` | `projected` | Each team's XI is taken from that team's last match. |
| `lineup_mode` | `confirmed` | The XI comes from an optional hand-supplied confirmed-lineup file. |
| `toss_mode` | recorded | Records whether the toss is unknown or supplied, together with the supplied toss facts when present. |

Toss results and confirmed elevens are typically public 15 to 30 minutes
before the start. A future scraper, with its source still to be chosen (likely
ESPNcricinfo), may supply the `toss` run kind. It does not alter, replace, or
change selection of the scored `t60` line.

## Deterministic scored-line function

Selection is a pure function implemented in `score_daily.py`.

Inputs are the complete sequence of immutable prediction lines and a target
`(cohort_id, fixture_id)`. The output is either exactly one selected line or
no eligible line, plus deterministic exclusion/coverage counts. The function
does not read results, settlements, file order, operator choices, or any
other mutable state.

Let `T` be a line's `scheduled_start`. A line is eligible if and only if all
four clauses hold:

1. `run_kind == "t60"` and `attempt_ts` is in the inclusive window
   `[T - 75 minutes, T - 45 minutes]`.
2. `quote_ts <= attempt_ts`.
3. `quote_ts` is in the same inclusive window
   `[T - 75 minutes, T - 45 minutes]`.
4. `attempt_ts < scheduled_start`.

From the eligible lines, the function returns the line with the latest
`attempt_ts`. This timestamp ordering is the sole tie-breaker. Two candidates
with the same maximum `attempt_ts` would have the same four-part identity and
are invalid input; the function fails closed instead of breaking that tie by
file order or content. Selection is never manual, including after outcomes
are known.

A line whose `attempt_ts` is on or after `scheduled_start` is ineligible even
when it carries an old quote. Lines outside the window, all `toss` lines, and
all other ineligible lines remain diagnostic and never enter the scored set.
`score_daily.py` has a test for each clause, order independence, two
in-window attempts scoring once, and the fail-closed tie case.

## Daily commands and append-only artifacts

| Step | Command | Append-only output and required stamp |
|---|---|---|
| a | `refresh_state.py` | Fetches Cricsheet into a new dated context directory; builds a new cache and tracker state under `data/live_state_i7/`; records `state_as_of`. An existing context or completed state build is never edited. |
| b | `fetch_fixtures.py` | Appends to `daily/fixtures/<date>.jsonl`: teams, venue, `scheduled_start`, market id, quote/price, market volume, and capture `quote_ts`. |
| c | `resolve_lineups.py` | Appends lineup resolution records: projected XI from each team's last match or an optional hand-supplied confirmed XI, with `lineup_mode`. |
| d | `predict_daily.py` | Appends one JSONL line per `(cohort_id, fixture_id, run_kind, attempt_ts)`: writer timestamp, model role and MD5, `state_as_of`, `lineup_mode`, `toss_mode`, `p_team1`, quote, `quote_ts`, `scheduled_start`, and minutes to start. |
| e | `settle_daily.py` | Appends outcome records to `daily/settled.jsonl`. Corrections append `revision: n`; no existing line changes. |
| f | `score_daily.py` | Appends a versioned score artifact from the standard evaluator for the deterministically designated line per fixture, including block-bootstrap metadata and the stamped cost model. |

The driver runs a-f in order for the scheduled `t60` job. A command may
observe existing identities and no-op, reject a collision, or append a new
record; it may not edit a prior record. The `toss` job is event-driven only.

### State refresh constraints

The refresh in step a wraps the live-state procedure in `docs/OPERATIONS.md`
Operation 6. Newly completed Cricsheet matches go into a separate,
non-overlapping dated context directory. That context is serving state, not
training data: it is never added to the frozen training corpus, never used to
refit the model, and never used to recompute global or phase priors. Those
priors remain frozen from the pre-context production cache.

The SQLite cache and tracker snapshot are rebuilt as a matched pair, their
source-match counts must agree, and the resulting `state_as_of` is stamped on
every prediction. State queries remain strictly as-of the fixture. Tracker
walks use `(match_date, Cricsheet match_id)` under
`date_then_match_id_lexicographic_v1`; within-date state advances only after
the earlier match. Frozen production caches, parquets, model artifacts, and
consumed forward evidence are never overwritten.

## Quotes, coverage, and voids

When no eligible quote is available, the prediction line is still written
with `quote: null` and `quote_ts: null`. It is excluded from market comparison
and counted in coverage. The prediction remains diagnostic; the selector
returns no market-scored line unless another attempt satisfies all four
eligibility clauses.

Postponed and no-result fixtures are voided at settlement. They remain in the
inventory and coverage counts but are excluded from binary scoring and bets.

## Settlement and revisions

Settlement is separated from prediction. Each settlement revision is a new
line with `revision: n`; scoring uses the highest revision for the fixture.
A revision may only add an outcome or void an outcome. It may never change a
prediction, quote, timestamp, lineup, toss fact, model identity, or earlier
settlement line. Conflicting records at the same highest revision fail
closed.

## Scoring and reporting

The settled cohort is scored through the standard evaluator. Log loss is the
primary decision metric; returns are a safety check, not a decision metric.
Market log loss and return comparisons use only the deterministic selected
line and exclude `quote: null` records as described above.

Match-winner uncertainty follows the I3
`tournament_time_block_v1` contract: 10,000 seed-42 whole-event resamples
with explicit bet placement. Fewer than 10 blocks is stamped
`descriptive` and cannot support an edge claim. Every score artifact stamps
the exact `CostModel` (`spread_bps`, `fee_bps`, and `fee_basis`), price basis,
block contract, block count, coverage, void count, and protocol/cohort
version. Zero cost is reported as the theoretical default; configured cost
scenarios are diagnostics beside it.

The acceptance criterion is: “ten consecutive days of `t60` lines with no
manual edits.” The full item acceptance additionally requires the protocol
to precede the first line in history, the identity/window/revision tests to
pass, and a settled cohort to score through the standard evaluator with the
cost model stamped.

## What this protocol does not cover

- Selecting or implementing the toss and confirmed-lineup scraper or its
  source; it will feed `toss` without changing the scored line (plan §9
  backlog context).
- Selecting an automated confirmed-lineup source; projected-XI and optional
  hand-supplied confirmed files are the present contract (plan §9 backlog
  context).
- Closing-line value or a market-residual model; these use intraday snapshots
  and remain in plan §9.

v1 2026-09-10 initial, precedes first collection run
