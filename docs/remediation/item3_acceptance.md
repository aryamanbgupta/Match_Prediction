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
