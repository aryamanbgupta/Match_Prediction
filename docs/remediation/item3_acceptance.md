# Item 3 acceptance checks (daily prediction job)

Written 2026-09-10 by Claude. Only step 1 (protocol doc) is in scope until
item 2 lands (3f depends on `market_math`) and the Mac mini is ready (3.8).
Checks for steps 2–8 will be appended before each is started.

## Step 1: `docs/DAILY_PREDICTION_PROTOCOL.md`

| # | Check | Pass condition |
|---|---|---|
| C1 | Precedes data | committed before any file under `daily/` exists in git history (`git log --diff-filter=A -- daily/` is empty at the doc's commit) |
| C2 | Identity | states the four-part identity `(cohort_id, fixture_id, run_kind, attempt_ts)`; `attempt_ts` is stamped by the writer's own clock and has no caller parameter; lines also carry `quote_ts` and `scheduled_start`; retries are new lines; no line references another |
| C3 | Run kinds | `t60` and `toss` defined exactly as plan §3; `toss` never scheduled blindly; the future toss scraper plugs into `toss` without changing the scored line |
| C4 | Scored-line rule | pure-function statement of: latest `t60` line with `attempt_ts` in `[T-75, T-45]`, `quote_ts <= attempt_ts`, `quote_ts` in the window, `attempt_ts < scheduled_start`; everything else diagnostic; the rule will live in `score_daily.py` with a test per clause |
| C5 | Settlement | revisions are new lines with `revision: n`, highest wins, may only add or void an outcome; postponed/no-result voided; missing quote → `quote: null`, excluded from market comparison, counted in coverage |
| C6 | Cohorts | a protocol change starts a new `cohort_id`; the doc carries its own version and the cohort id it governs |
| C7 | Scoring | scored through the standard evaluator, I3 block bootstrap (`tournament_time_block_v1`), cost model stamped, log loss primary and returns a safety check (plan §0.4); `<10 blocks = descriptive` |
| C8 | Lineups | `lineup_mode` ∈ {projected, confirmed} with the projected-XI rule (each team's last match); `toss_mode` recorded |
| C9 | Outputs | the six command table (a–f) with their append-only outputs and `state_as_of` |
| C10 | No contradictions | nothing in the doc conflicts with plan §3 or CLAUDE.md invariants 2, 5, 6, 7; the acceptance "ten consecutive days of t60 lines with no manual edits" is quoted |

## Steps 2–7 (written 2026-09-10 before implementation; step 8 scheduler is mini-only and later)

| # | Check | Pass condition |
|---|---|---|
| C11 | `scripts/daily/refresh_state.py` | wraps OPERATIONS Operation 6: fetches cricsheet into a dated context dir (never into `data/t20s_json`), builds cache + tracker snapshot into a NEW `data/live_state_i7_<as_of>_<build_utc>/`, writes `BUILT` last, moves the symlink via `artifacts.promote_live_state`, records `state_as_of`; refuses a dir with `BUILT`; test on the mini corpus with a stub fetcher |
| C12 | `predict_fixture.predict_record(fixture) -> dict` | the CLI is a thin wrapper; all 30 existing predict_fixture tests pass unchanged; the function has no side effects on state |
| C13 | `scripts/daily/fetch_fixtures.py` | Gamma query for open male T20 head-to-head markets; appends `daily/fixtures/<date>.jsonl` with teams, venue, `scheduled_start`, market id, price, market volume, `quote_ts` (capture time from the writer's clock); recorded-response test; strict moneyline/H2H filter reusing the sealed-set extractor's rules (never the toss market: regression test on a toss-market fixture) |
| C14 | `scripts/daily/resolve_lineups.py` | projected XI = each team's last match XI from the current state (`lineup_mode: projected`), optional hand-supplied confirmed file (`confirmed`); mini-corpus test |
| C15 | `scripts/daily/predict_daily.py` | one JSONL line per `(cohort_id, fixture_id, run_kind, attempt_ts)`; `attempt_ts` stamped by the writer, no caller parameter; identical identity refused, retry with new `attempt_ts` accepted; carries model role + md5 (from the manifest), `state_as_of`, `lineup_mode`, `toss_mode`, `p_team1`, quote, `quote_ts`, `scheduled_start`, minutes to start; `quote: null` when missing; schema test |
| C16 | `scripts/daily/settle_daily.py` | joins cricsheet winners into `daily/settled.jsonl`; revisions append `revision: n`; postponed/no-result → void; test proves no existing line changes |
| C17 | `scripts/daily/score_daily.py` | pure `select_scored_line(lines, cohort_id, fixture_id)` with a test per window clause (T−75/T−45 inclusive, `quote_ts <= attempt_ts`, quote in window, `attempt_ts < scheduled_start`), order independence, two in-window attempts score once, fail-closed tie; scoring through `market_math` + `eval_statistics` block bootstrap with the cost model stamped; `<10 blocks = descriptive` |
| C18 | `scripts/daily/run_daily.sh` | runs a→f idempotently for the `t60` schedule; a dry run on the mini corpus + a recorded Gamma response produces a fixtures file, a prediction line and a scored (descriptive) output without network |
| C19 | Protocol conformance | every field name matches `docs/DAILY_PREDICTION_PROTOCOL.md`; `cohort_id` is `daily_v1`; the doc precedes every `daily/` file in git history |
| C20 | Suite 0 failures; frozen evidence untouched; no edits to program.md/research/ |

## Fix pass (2026-09-10)

| Finding | Fixed contract | Regression evidence |
|---|---|---|
| K1 + K2 | Dated contexts contain only IDs absent from the base and earlier contexts; every build receives base plus all contexts in date order, including an empty current context. | `test_refresh_accumulates_contexts_filters_duplicates_and_promotes` |
| K3 | Refresh rejects context/state destinations inside the base corpus, a state parent other than the base's `data/` parent, and destinations below a `BUILT` marker before mutation. | `test_refresh_path_guards_run_before_mutation` |
| K4 | Fixture captures are deduped per `fixture_id`; revision allocation observes existing and same-call allocations; unchanged outcomes no-op. | `test_repeated_fixture_captures_allocate_only_one_revision_one` |
| K5 | Settlement consumes all `daily/fixtures/*.jsonl`; the driver supplies every dated context as an outcome source. | `test_day_one_fixture_settles_when_result_arrives_on_day_three` |
| K6 | Prediction append locks the JSONL exclusively, reads identities while locked, stamps and checks `attempt_ts` at the write boundary, writes, flushes and `fsync`s. | `test_concurrent_identical_appends_are_serialized`; `test_append_refuses_at_or_after_start` |
| K7 | `records_from_gamma` emits no timestamp; `append_fixtures` owns `quote_ts`, rejects caller timestamps, and stamps null for a null quote. | `test_append_owns_quote_timestamp_and_null_quote_has_null_timestamp` |
| K8 | Settlement maxima are determined before conflicts are checked, so superseded lower-revision conflicts are ignored regardless of input order. | `test_settlement_conflicts_only_apply_to_highest_revision_order_independent` |
| K9 | The structural H2H classifier has one repository home in `scripts/sim_eval/market_selection.py`; daily fixture selection imports it and the toss conflict remains fail-closed. The separately operated strict extractor is not present in this repository checkout. | `test_recorded_gamma_selects_moneyline_and_rejects_toss`; `test_moneyline_identity_conflict_fails_closed` |
| K10 | The offline test executes `run_daily.sh`; Gamma fetch is recorded and expensive refresh/model work is stubbed while real daily writers, settlement and scoring run. It asserts fixtures, one prediction, one settlement, and a descriptive score with protocol fields. | `test_daily_driver_executes_offline_end_to_end` |

## Fix pass 2 (2026-09-10)

| Finding | Fixed contract | Regression evidence |
|---|---|---|
| P2-1 | Context dedupe checks the base and all published dates, including later dates during an out-of-order run. | `test_refresh_accumulates_contexts_filters_duplicates_and_promotes` |
| P2-2 | One non-blocking `daily/refresh.lock` covers a refresh; promotion requires non-regressing `state_as_of` and source match count. | `test_refresh_refuses_held_global_lock`; `test_stale_build_is_sealed_but_not_promoted` |
| P2-3 | Context files are prepared in `.tmp-<date>-<pid>` and the directory is atomically renamed; interrupted and abandoned attempts are cleaned. | `test_interrupted_fetch_never_publishes_partial_context`; `test_refresh_accumulates_contexts_filters_duplicates_and_promotes` |
| P2-4 | `context_date` is exact ISO-date syntax and the resolved final destination is guarded before mutation. | `test_refresh_path_guards_run_before_mutation` |
| P2-5 | Settlement uses the single `<jsonl>.lock` across read, revision allocation, append, flush, and `fsync`. | `test_concurrent_settlement_allocates_one_revision` |
| P2-6 | Prediction prepares serialization and scans identities before sampling its writer clock; the sidecar lock covers the complete transaction. | `test_clock_is_sampled_after_identity_scan_and_serialization`; `test_concurrent_identical_appends_are_serialized` |
| P2-7 | Fixture lines stamp `writer: fetch_fixtures` and a recomputed SHA-256; prediction rejects missing or altered provenance at both the file reader and line-construction API, and preserves null quote timestamps. | `test_append_owns_quote_timestamp_and_null_quote_has_null_timestamp`; `test_unsigned_or_tampered_fixture_is_refused_before_inference`; `test_prediction_line_refuses_unsigned_fixture_quote_timestamp` |
| P2-8 | Prediction selects the latest capture per fixture and skips already-started fixtures with an `expired_skipped` count. | `test_fixture_provenance_latest_capture_and_expired_skip` |

## Fix pass 3 (2026-09-10)

| Finding | Fixed contract | Regression evidence |
|---|---|---|
| L1 | Fixture provenance is an HMAC-SHA-256 over the canonical writer-owned `quote`, `quote_ts`, `market_volume_usd`, `scheduled_start`, `fixture_id`, and `market_id` fields. `fetch_fixtures` alone creates the 32-byte mode-0600 machine-local key at `daily/.writer_key` (`DAILY_WRITER_KEY` overrides it); prediction only reads the key and rejects absent/forged HMACs, altered timestamps, and non-`fetch_fixtures` writers before inference or append. | `test_append_owns_quote_timestamp_and_null_quote_has_null_timestamp`; `test_writer_key_is_created_once_and_genuine_line_verifies`; `test_fixture_hmac_is_mandatory_and_authenticates_quote_timestamp`; `test_fixture_with_forged_writer_is_refused`; `test_append_cannot_bypass_fixture_hmac_with_modified_quote_timestamp` |
| L2 | Lineup resolution removes `scheduled_start <= now` fixtures before any XI lookup and reports `expired_skipped`; later upcoming fixtures continue through resolution. Prediction retains its independent expiry guard at inference and append time. | `test_expired_missing_xi_is_skipped_before_upcoming_lineup_resolution`; `test_fixture_provenance_latest_capture_and_expired_skip`; `test_append_refuses_at_or_after_start` |

## Step 8 (scheduler on the Mac mini; written before implementation)

| # | Check | Pass condition |
|---|---|---|
| C21 | Branch on the mini | `embeddings-ladder` checked out at the commit that contains steps 2–7; `artifacts.py verify` OK there (live state resolved through the symlink; the mini's current flat `data/live_state_i7/` is migrated the same way as here, with its own `BUILT` dir) |
| C22 | Plist | `launchd` agent runs `scripts/daily/run_daily.sh` at fixed UTC times covering T−60 for the day's fixtures (a single daily fetch at 00:30 UTC plus per-fixture `t60` attempts scheduled from `scheduled_start`, or a fixed 30-minute cadence if per-fixture scheduling is out of scope; the choice is recorded); logs to `daily/logs/`; never runs the `toss` kind |
| C23 | Resource limits | `OMP_NUM_THREADS=2`; refresh runs once a day; no simulator eval; disk check before the cricsheet fetch (refuse under 5 GB free) |
| C24 | Dry day | one full run on the mini in dry mode (recorded Gamma response) before the first live day; outputs under `daily/` reviewed |
| C25 | Ten-day acceptance | ten consecutive days of `t60` lines with no manual edits (`git log -- daily/` shows only the job's commits, if `daily/` is committed; otherwise file mtimes and the append-only structure) |

### Review closure, steps 2–7 (Claude, 2026-09-10)

Three Astra rounds: ten, eight and two findings, all fixed with locking
tests (context accumulation and atomic publication, path containment,
per-file and global locks, non-regressive promotion, serialize-then-lock-
then-clock appends, HMAC-signed fixture lines under a machine-local key,
expiry filtering before lineups, real offline driver test). Suite 545
passed, 0 failed. C11–C20 met; step 8 next.

### Step 8 progress (2026-09-10)

C21 met (branch on the mini at ff53c0a+, `artifacts.py verify` identical to
the laptop after the canonical-source correction; live state migrated to
the versioned layout). C24 dry day run 1 exposed a real defect: the
promotion guard only ran when the current target carried `state.json`, so
the migrated production state (bare `BUILT`) was replaced by a 2026-04-16
build. Symlink restored by hand within minutes (the old dir was intact);
`refresh_state` now derives the current state's as-of and count from the
cache `_meta` and the directory name and fails closed when it cannot;
regression tests reproduce the mini's layout. Dry day re-run follows.
Before going live the two context dirs the production state was built
from (sealed forward context and `live_context_20260801`) must be seeded
into `daily/context/` under their dates, or no daily build can ever be
non-regressive.
