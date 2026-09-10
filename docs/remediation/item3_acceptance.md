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
