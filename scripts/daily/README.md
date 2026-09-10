# Daily prediction job

This directory implements protocol version 1 in
`docs/DAILY_PREDICTION_PROTOCOL.md`. `run_daily.sh` runs refresh, fixture
capture, lineup resolution, T-60 prediction, settlement, and scoring in that
order. All evidence files are append-only; score files and state builds are
versioned and never replaced.

## Fix pass 3 invariants

Contexts live at `daily/context/<date>/`. A newly fetched context retains only
Cricsheet IDs absent from `data/t20s_json` and every published dated context,
including contexts newer than an out-of-order run. Fetches prepare under
`daily/context/.tmp-<date>-<pid>/` and publish with one rename; the next run
removes abandoned temporary directories. `context_date` must be an exact
`YYYY-MM-DD`, and its resolved destination must remain under the context root
and outside the base corpus and sealed builds. One non-blocking
`daily/refresh.lock` covers the entire refresh.
Every state rebuild consumes the base followed by all dated contexts, so an
empty new context still produces a non-regressing state. `refresh_state.py`
requires the state build parent to be the base corpus's `data/` parent and
refuses destinations inside the base corpus or below any `BUILT` marker. A
build is promoted only when both its `state_as_of` and source match count do
not regress the current target; otherwise it is sealed and logged as
`stale build not promoted`.

Fixture capture owns `quote_ts`; callers cannot supply it, and a missing quote
is stamped with a null timestamp. Each fixture line carries
`writer: fetch_fixtures` and an HMAC-SHA-256 over the canonical `quote`,
`quote_ts`, `market_volume_usd`, `scheduled_start`, `fixture_id`, and
`market_id` fields. The 32-byte machine-local key is created once by
`fetch_fixtures` at `daily/.writer_key` with mode 0600; `DAILY_WRITER_KEY`
overrides that location. Prediction only reads the key and refuses a missing,
forged, or altered HMAC and any other writer. It selects the latest capture per
fixture and counts already-started fixtures as `expired_skipped`. Lineup
resolution also filters and counts expired fixtures before attempting any
historical-XI lookup, so one expired fixture cannot stop later upcoming work.
Prediction append owns `attempt_ts`; only after preparing content and scanning
identities does it sample the clock and enforce `attempt_ts < scheduled_start`.
Predictions and settlements each use exactly one `<file>.lock` sidecar around
their complete read/allocate/append transaction, then flush and `fsync` before
unlocking. Settlement reads the accumulated `daily/fixtures/*.jsonl`
inventory, dedupes captures per fixture, and appends only new or changed
highest revisions.

Run from the repository root:

```bash
scripts/daily/run_daily.sh
```

For an offline replay, set `GAMMA_RESPONSE` to a recorded Gamma JSON file and
`CRICSHEET_FETCH_CMD` to a command that writes fetched match JSON into the
`CRICML_CONTEXT_DIR` environment path. The default wraps the Cricsheet fetcher
documented in Operation 6 with its destinations redirected into that context;
it never merges into `data/t20s_json`. `toss` predictions are invoked
directly through `predict_daily.py`; they require both a confirmed-XI file and
a supplied toss file and never replace the scored `t60` record.
