# CricML remediation and simplification plan (2026-09-09)

Status: v4 AGREED between Claude Fable 5.1 and Codex GPT-6 Astra (light) on 2026-09-09 after four review rounds (Astra
corrections folded in below,
marked "[v2]", "[v3]", "[v4]", plus "[v5]" user decisions). Awaiting user approval; implementation starts only when the user triggers it.

Scope: the nine items agreed on 2026-09-09 after the August integrity review
and the September atlas backlog were reconciled, plus the backlog decisions
taken in that discussion.

## 0. Principles that every item follows

1. **One home per job.** Almost every bug in the August review was one piece
   of logic copied by hand into several files, with one copy drifting. Each
   item either creates a single owner module or protects one with tests.
   No new hand-mirrored copies.
2. **Tests before refactors.** Nothing that changes behaviour lands until the
   tests that would catch a regression run automatically (item 1).
3. **Frozen evidence stays frozen.** `betting_odds_polymarket_v2.json`,
   `data/golden/betting_odds_golden_v2.json`, the sealed forward set, and
   every committed report are never edited in place. New results go to new
   versioned files; errata are appended to old reports.
4. **Log loss decides; returns are a safety check.** Sample-size analysis
   (2026-09-09): confirming a true 10% return edge needs roughly 1,100 to
   1,800 priced bets, a 5% edge 3,500 to 7,000, and the current +3.4%
   headline over 10,000. At ~300 sharp-market fixtures a year, returns cannot
   decide anything on a project timescale.
5. **Every fix ships with its regression test.** Every behaviour-changing fix
   records a before/after on a fixed-seed sample in IMPROVEMENTS.md.
6. **Intentional behaviour changes are named.** When a consolidation also
   corrects a bug (for example the dashboard's missing de-vig), the change
   is listed under "intentional changes" in the item, with its own test,
   rather than hidden inside a parity claim. [v2]
7. **Two-model review** (section 11).

## 1. CI and test hygiene

**Problem.** Two suites exist. `scripts/tests/` (58 files, 299 tests, from
the project's start) has only ever run by hand. `tests/` (11 files, 53 tests)
was added on this branch in August with
`.github/workflows/embeddings-contracts.yml`, which runs only `pytest tests`
and only when transformer or simulator files change. The winner model, odds
builders and evaluator therefore have zero automated coverage. Every test
file patches `sys.path` itself (63 call sites, six spellings). All but one
resolve to `scripts/`; `scripts/tests/test_prop_fair_baselines.py:7` adds
`scripts/sim_eval` and imports `prop_fair_baselines` bare. [v2] Eight test
files each build their own fake cricsheet corpus. One test hard-fails
without a local artifact (`test_build_consistency.py:74`), three "soft-skip"
by printing and returning (pytest reports PASSED), and
`test_min_volume_filter.py:60-70` asserts 261/170/110 against the retracted
`betting_odds_polymarket.json`.

**Design.** One workflow, one import mechanism, one fake corpus, honest
skips.

**Steps.**
1. [v2] Exactly one path mechanism: `[tool.pytest.ini_options]` in
   `pyproject.toml` with `testpaths = ["tests", "scripts/tests"]` and
   `pythonpath = ["scripts"]`. No `conftest.py` path hacks. Remove the
   `sys.path` blocks from the 69 test files. Normalise the one exceptional
   import to `from sim_eval.prop_fair_baselines import ...`. CLI scripts keep
   their bootstraps (packaging is a backlog item).
2. Create `scripts/tests/cricsheet_fixtures.py` exposing `delivery()`,
   `innings()`, `match_json()`, `write_corpus()`, `write_metadata_csv()`,
   lifted from `test_train_serve_parity.py`; migrate the other seven
   builders. Commit a small generated corpus under
   `tests/fixtures/cricsheet_mini/` (about 20 matches, two competitions, one
   same-day pair, one super over, one D/L result) plus its generator and
   seed.
3. Replace the three soft-skips and the hard assert with
   `pytest.mark.skipif(..., reason=...)`; register markers `needs_artifacts`
   and `slow`.
4. [v2] Volume-filter semantics get a synthetic test on the mini corpus
   (event vs market volume, missing volume, boundary at exactly 50,000);
   the real-file count check moves to `betting_odds_polymarket_v2.json`
   with 255/168/110 and is marked as a regression pin, not the semantic
   test. Add `scripts/tests/test_shipped_odds_contract.py`: every row in
   both v2 files is head-to-head (`market_question == event_title` or
   `sports_market_type == moneyline`), implied probabilities sum to
   [0.98, 1.02], no quote at or after scheduled start, no duplicate
   cricsheet id.
5. Rename the workflow to `ci.yml`; run on every push and pull request with
   no path filters; `uv run --no-sync pytest -q -m "not needs_artifacts"`.
   Keep the uv/Python setup and the excluded-packages list; 20-minute
   timeout.
6. [v2] Verify collection in an artifact-free checkout before merging: a
   local run with `models/` and `data/` temporarily renamed must collect
   the same set CI does, with zero errors.
7. Astra light reviews the workflow, `pyproject.toml` block and the fixture
   module.

**Acceptance.** CI green with 0 failures and every skip carrying a reason;
the artifact-free local run and CI collect identical test ids; no test file
contains `sys.path`; the fixture module is the only cricsheet builder.

## 2. One betting-math module with a cost model

**Problem.** Six places compute bet placement and profit with their own
copies: `match_evaluator._calculate_realized_pnl`,
`blend_eval_json._recompute_realized_pnl`, `predict_fixture.compute_bet`,
`hundred_roi_eval.settle`, `sizing_rules._compute_pnl`,
`build_ipl_dashboard.compute_bet` (which does not remove the bookmaker
margin). All returns use a spread-free, fee-free mid. Slices use event-level
volume although the head-to-head market's volume is stored (168 vs 156 at
≥$50k).

**Design.** [v2] Two layers, deliberately separate:

- `scripts/sim_eval/market_math.py` owns the **numeric primitives** only:
  `implied_probs(odds, remove_margin)`, `edge(model_p, market_p)`,
  `settle_flat(bet_team, odds, winner, cost)`, `kelly_fraction`,
  `settle_kelly`, `expected_value`, and `CostModel`.
- **Placement policies stay where they are** and call the primitives:
  `eval_statistics.flat_bet_team` (threshold-0 flat rule of record),
  `predict_fixture.compute_bet` (retired A7 policy with its liquidity,
  freshness and scope contracts; its 23 tests stay untouched),
  `hundred_roi_eval.settle` (predeclared no-threshold rule),
  `sizing_rules` (capped fractional Kelly).

`CostModel(spread_bps, fee_bps, fee_basis)` with the accounting written
down once [v3]:

- `spread = spread_bps / 10_000` and `fee = fee_bps / 10_000`, both
  absolute probability units on a price in (0, 1).
- Market price `q = 1 / o` must satisfy `0 < q < 1`; otherwise the record is
  rejected (no silent clipping). The bettor buys at `q_eff = min(q +
  spread / 2, 1 - 1e-9)`, so effective odds `o_eff = 1 / q_eff`.
- `fee_basis = "winnings"`: win returns `(o_eff - 1) * (1 - fee)`, loss
  returns `-1`. `fee_basis = "stake"`: win returns `(o_eff - 1) - fee`, loss
  returns `-(1 + fee)`.
- Expected value and Kelly use the same `o_eff` and the same fee rule:
  `EV = p * win_return + (1 - p) * loss_return`; Kelly fraction
  `f = (p * b - (1 - p)) / b` with `b = win_return` when `loss_return = -1`,
  and the general two-outcome formula `f = (p * b - (1 - p) * l) / (b * l)`
  with `l = -loss_return` otherwise. [v4] Stakes are non-negative; the
  bankroll constraint is `f * l <= 1` (so `f * (1 + fee) <= 1` under the
  stake basis), and `f = 0` (no bet) whenever `win_return <= 0` or the
  Kelly numerator is non-positive. Valid cost ranges: `0 <= spread < 1`,
  `0 <= fee < 1`; anything else is a constructor error. Boundary tests
  cover `q + spread/2` at the cap, `win_return = 0`, and the stake-basis
  bankroll bound.
- Default scenario set: (0, 0, winnings), (100, 0), (200, 0), (100, 100,
  winnings), (100, 100, stake). `CostModel.none()` is (0, 0, winnings) and
  reproduces today's arithmetic exactly.
- Tests cover: each formula at both fee bases, a losing bet under `stake`,
  price-bound rejection, the cap, and `none()` parity.

**Intentional changes** (each with its own test, listed in IMPROVEMENTS.md):
the dashboard starts de-vigging, so its zero-cost bet decisions may change
on fixtures where the raw and de-vigged edge straddle zero.

**Steps.**
1. `market_math.py` with unit tests per primitive, plus a parity fixture of
   50 recorded evaluator records proving `settle_flat(..., CostModel.none())`,
   `kelly_fraction`, `settle_kelly` and `expected_value` reproduce
   `match_evaluator` bit-for-bit.
2. Route the six call sites' arithmetic through it; delete the private
   copies; policies unchanged.
3. Add `--spread-bps`, `--fee-bps`, `--fee-basis` to `run_sim_eval.py`,
   `reslice_eval_json.py`, `blend_report.py`, `predict_golden.py`,
   `hundred_roi_eval.py`. Reslice recomputes profit from stored
   `market_odds` under the requested cost model instead of copying
   `realized_pnl`; summaries stamp `cost_model`, `price_basis` and
   `volume_basis`.
4. `--volume-basis {event,market}` on reslice.
5. `blend_report.py` prints every configured scenario side by side. [v5,
   user decision] **Theoretical profitability first:** the gate's safety
   check runs at zero cost by default (`CostModel.none()`), and the cost
   scenarios are reported next to it, never in place of it. The scenario
   list is configuration (`--cost-scenarios`), not code; the 1-cent spread
   and fee values are placeholders until the real Polymarket execution
   constraints (spread, depth, fee schedule, slippage on size) are measured
   from captured order books, at which point the measured scenario replaces
   the placeholder and both continue to be reported.
6. Re-baseline on the production model's `test_predictions.json` (pulled
   from the Mac mini, item 5 step 3 is a prerequisite [v2]): record which
   numbers moved (expected: cost columns only, and the dashboard's
   intentional change).

**Acceptance.** Parity fixture passes; zero-cost evaluator and reslice
numbers identical to today's to four decimals; each policy's existing tests
pass unchanged; cost columns present in every report; the dashboard change
is documented with its test.

## 3. Daily prediction job (the next forward set)

**Problem.** The sealed forward set is consumed. Live state on the Mac mini
ends 2026-07-30; the 14-day guard refuses later fixtures. Fixtures are typed
by hand. The sibling repo captures open markets (`collect_data.py`,
`collect_ipl_odds.py`, launchd plists) but nothing is loaded on either
machine. [v2] No source anywhere supplies toss results or confirmed
lineups; only the Polymarket capture and cricsheet exist.

**Design.** Small idempotent commands under `scripts/daily/`, append-only
outputs, one driver. [v2] The protocol is written and committed **before**
any collection run, and the T-60 line is the one scored line per fixture;
everything else is diagnostic.

| Step | Command | Output |
|---|---|---|
| a | `refresh_state.py` [v2: explicit step] | cricsheet fetch into a dated context dir, cache + tracker rebuild in `data/live_state_i7/`, `state_as_of` recorded |
| b | `fetch_fixtures.py` | `daily/fixtures/<date>.jsonl`: teams, venue, start time, market id, price, market volume, capture timestamp |
| c | `resolve_lineups.py` | projected XI from each team's last match (`lineup_mode: projected`), or confirmed XI from an optional hand-supplied file (`confirmed`) |
| d | `predict_daily.py` | one JSONL line per (cohort, fixture id, run kind): timestamp, model role + md5, `state_as_of`, `lineup_mode`, `toss_mode`, p_team1, quote, minutes to start |
| e | `settle_daily.py` | joins cricsheet winners into `daily/settled.jsonl`; a later revision writes a new line with `revision: n`, never edits |
| f | `score_daily.py` | standard evaluator, block bootstrap, cost model, over the designated line per fixture |

[v3] Run identity and scoring rules, fixed in
`docs/DAILY_PREDICTION_PROTOCOL.md`:

- Every prediction line has identity `(cohort_id, fixture_id, run_kind,
  attempt_ts)`. [v4] `attempt_ts` is the **creation time stamped by the
  writer from its own clock** at the moment the line is appended; the
  writer API has no parameter for it, so a caller cannot backdate. Each
  line also records `quote_ts` (capture time of the quote it used) and
  `scheduled_start`. Retries are new lines; nothing is ever rewritten, and
  no line carries a pointer to another line.
- `run_kind` is `t60` (scheduled, toss unknown, projected or confirmed
  lineup) or `toss` (only when a toss result and confirmed eleven were
  supplied; never scheduled blindly). [v5] Toss results and confirmed
  elevens are typically public 15 to 30 minutes before the start; a scraper
  for them (source to be chosen, likely ESPNcricinfo) is a backlog item that
  plugs into the `toss` run kind without changing the scored `t60` line.
- **Deterministic scored line [v4]:** for each `(cohort_id, fixture_id)`
  the scored line is the `t60` line with the latest `attempt_ts` that
  satisfies all of: `attempt_ts` in `[T-75 min, T-45 min]` relative to
  `scheduled_start`; `quote_ts <= attempt_ts`; `quote_ts` within the same
  window. A line whose `attempt_ts` is on or after `scheduled_start` is
  ineligible regardless of its quote, which closes the post-result retry
  with an old quote. Lines outside the window, `toss` lines, and any
  others are diagnostic and never enter the scored set. The rule is a pure
  function in `score_daily.py` with tests for each clause, so selection
  cannot be done by hand after outcomes are known.
- Settlement revisions are new lines with `revision: n`; scoring uses the
  highest revision per fixture. A revision may only add or void an outcome,
  never change a prediction.
- Missing quote: line written with `quote: null`, excluded from market
  comparison, counted in coverage. Postponed or no-result fixtures are
  voided at settlement. A protocol change starts a new `cohort_id`.

**Steps.** (1) protocol doc committed first [v2]; (2) `refresh_state.py`
wrapping OPERATIONS § Operation 6 with a test on the mini corpus [v2];
(3) refactor `predict_fixture.py` into `predict_record(fixture) -> result`
plus the CLI, existing 18 tests unchanged; (4) `fetch_fixtures.py` against
Gamma with a recorded-response test; (5) `resolve_lineups.py` with the
projected-XI rule and a mini-corpus test; (6) `predict_daily.py` + schema
test (identical four-part identity refused; a retry with a new
`attempt_ts` accepted); (7) `settle_daily.py` + `score_daily.py` on
`market_math` (dependency [v3]) and `eval_statistics`, with tests that the
window rule picks the same line regardless of line order and that two
in-window attempts score once; (8) `run_daily.sh` + launchd plist on the Mac mini for
the `t60` schedule only.

**Acceptance.** Protocol doc precedes the first line in git history; ten
consecutive days of `t60` lines with no manual edits; the identity,
window-selection and revision tests pass; a settled cohort scores through
the standard evaluator with the cost model stamped.

## 4. Verdict rule and gate function

**Problem.** `program.md` lands an idea only if log loss AND returns both
improve beyond a floor. Returns on 168 bets are noise (0.4), so the rule
vetoes real gains (D8: ΔLL −0.0093 on 5/5 seeds, TABLED). `log_verdict.py`
validates bookkeeping, requires RUNNING status, and refuses to overwrite a
non-placeholder result, so re-assessments have no supported path. [v2] No
single function computes a verdict; sixteen gate scripts in `scripts/auto/`
import `a8_gate_analysis.py` and do it by hand.

**Design.** `scripts/sim_eval/claim_gate.py` exposes
`decide(candidate, baseline, kind, cost) -> Verdict`. [v2] Its evidence
contract: inputs are standard sliced eval JSONs; match ids must match
exactly (mismatch is an error, not a filter). [v3] **Seed rule, shared with
item 6:** with 5 or more aligned seeds per arm the estimator is
`seed_mean_match_cluster_ci`; with 2 to 4 seeds the same estimator is used
and the verdict is stamped `provisional`; with 1 seed the estimator is
`match_cluster_ci` and the verdict is stamped `provisional`. [v5, user decision] A
provisional verdict can never be LANDED: if its ΔLL interval excludes zero
favourably and the point estimate clears the seed floor of 0.007 (the A1
measured seed standard deviation), the verdict is `PROMISING`, which keeps
the code on a branch and queues a full five-seed confirmation run; only
that run can produce LANDED. All
intervals are 95%, 10,000 resamples, seed 42. [v4] **Cluster contract:**
the gate strips any `competition_cluster_id` / `cluster_id` stamped on the
records before resolution (today `cluster_id_with_resolution` at
`eval_statistics.py:189-194` trusts a stamped label first, which the gate
must not inherit) and resolves every block id from the registered cluster
source dir only; any fallback resolution makes the verdict `DESCRIPTIVE`;
callers cannot pass labels. [v4] **Evidence check:** the gate loads the
registered `_v2` odds file (sha256 in `docs/registered_odds.json`), checks
each record's raw `market_odds` against it, and **recomputes** `market_prob`,
bet placement and cost-adjusted profit from the registered odds under the
requested cost model. Stored `market_prob` and `realized_pnl` are ignored,
so a JSON built on other prices cannot pass by declaring a hash. Records with missing outcome or market are dropped from
both arms symmetrically and counted; different bet sets between arms are
expected and the profit delta is computed on the union with zero stake
where no bet; `bootstrap_reliable: false` or fewer than 10 blocks makes the
verdict `DESCRIPTIVE` (never LANDED); non-finite intervals are an error. Verdict rules for
match-model ideas: LANDED if the paired ΔLL interval excludes zero
favourably and the Δprofit interval does not exclude zero unfavourably,
where Δprofit is computed under the gate's configured cost model
(default: zero cost, per item 2 step 5; every verdict also reports the
configured cost scenarios as diagnostics); TABLED if ΔLL clears but profit is clearly harmful; FAILED
otherwise. Betting-layer and sim/prop ideas keep their gate pairs through
`kind`. The verdict object carries seed count, estimator, cost model,
odds-file sha256, cluster contract, and block count.

**Steps.**
1. `claim_gate.py` on `eval_statistics.bootstrap_mean_ci` and
   `registered_experiment` estimators; unit tests for every branch above
   with synthetic paired records.
2. [v2] `log_verdict.py`: `verdict` requires `--gate-json` and checks the
   verdict equals the gate's, the gate's odds sha256 is in
   `docs/registered_odds.json`, and the gate's cluster resolution has zero
   fallbacks. New subcommand `reassess <ID>` appends a dated
   `**Reassessment:**` block under the idea and a new `results.tsv` row with
   idea id `<ID>-r2`, leaving the original result line untouched.
3. `program.md` VERDICT RULE replaced; old text kept in a dated
   "superseded" block; `night_v3.sh` prompts call the gate.
4. Move the helpers `claim_gate.py` needs out of `a8_gate_analysis.py` into
   `eval_statistics.py`; `a8` becomes a shim.
5. Re-run the twelve TABLED ideas (A4, B1, B5, B7, B8, B10, B13, B17, D2,
   D3, D8, D14) through `reassess`, using archived artifacts on the Mac mini
   where they exist, otherwise re-train under item 6's harness.
6. [v2] **Checkpoint (review, not trigger):** after the TABLED batch and at
   least six new ideas have run, the user reviews a one-page comparison on
   the **iteration set only** (golden is never used to choose the rule):
   ideas landed under each rule, their paired ΔLL, and production log loss
   before and after. Reverting to the old rule is a human decision recorded
   in program.md. [v5, user decision] **This is a standing review, not a
   one-off.** The rule carries a `review_due` date in program.md; the
   first review is the checkpoint above, and thereafter the rule is
   re-reviewed every 90 days or every ten verdicts, whichever comes first.
   `log_verdict.py` prints a reminder when the review is overdue. The
   review asks three questions: did the ideas landed under this rule
   improve production log loss; did any landed idea lose money under the
   measured cost scenario; is the rule's noise floor still the right
   number.

**Acceptance.** Every verdict row after the change carries the gate JSON's
sha256; the night runner cannot write a verdict without one; a test proves
`reassess` leaves the original lines byte-identical; a test proves an
unreliable-bootstrap input cannot produce LANDED.

## 5. Artifact manifest, reproducibility, and the engine gate re-run

**Problem.** This machine is a light checkout. The Mac mini has all
artifacts (verified 2026-09-09: ball model md5 `7ee1e180…`, match model md5
`54faf586…`, live state through 2026-07-30). Rebuild commands are spread
across four docs. `run_br2_gates.sh:47` runs G1 with no `--model`, so
`run_sim_eval.py` would load the retired `models/xgb_v3`. [v3] Hard-coded
artifact defaults exist in two tiers. **Production and evaluation paths
(must migrate):** `predict_fixture.py` (`--model-dir`, `--state-dir`,
`--tracker-snapshot`), `prop_backtest.py:884-903`, `predict_golden.py:29-31`,
`run_sim_eval.py:172` and `--model-version`, `refresh_golden_i7.sh`,
`run_br2_gates.sh`, `build_bowler_phase_usage.py`, and the simulator
sidecar defaults in `sim_v1_2.py` (bowler usage, extras graft, run-out
rates). **Diagnostic one-offs (exempt with a marker):** `analyze_features.py:28`,
`profile_eval.py:67`, `e1_temperature_sharpen.py:185`, and any other
script found by the scan in step 4, each carrying a
`# manifest-exempt: <reason>` comment line. G1 and the prop A/B have
never run on the fixed engine; G5 sits at 89.7% against a 90% bar.

**Design.** `models/MANIFEST.yaml`, checked in: for each artifact of record
its role name, path, kind (`file` or `dir`), hash (file md5; for a dir, the
md5 of the sorted list of `relative_path md5` lines), producing command,
input roles, and promoting doc. [v3] Mutable live state is versioned:
`data/live_state_i7/` is a symlink to
`data/live_state_i7_<as_of>_<build_utc_ts>/`; each build dir is immutable
once written (a `BUILT` marker file is the last write and refresh refuses
to touch a dir that has one), so two refreshes on the same `as_of` produce
two dirs and the link moves to the newer; the manifest records the current
target. `scripts/artifacts.py`: `verify [role]`, `pull --from
mac-mini [role]` (rsync then verify), `rebuild role` (topological order over
input roles; refuses if an input's hash mismatches), `path role` (prints the
path; scripts use this). [v2] Precedence: an explicit CLI path wins, then
`--role`, then the manifest default. [v4] Step 4 migrates the production
tier and adds `scripts/tests/test_manifest_defaults.py`, an AST scan of
every `.py` under `scripts/` plus a regex scan of every `.sh`: any string
literal or constant beginning with `models/`, `data/live_state`,
`data/xgb_`, `data/golden/polymarket_test`, `data/polymarket_test` or
`eval_out/` anywhere (argparse defaults, function and constructor defaults
such as `sim_v1_2.py:714`, module constants, positional literals, shell
commands) fails the test unless the enclosing line or the file header
carries `manifest-exempt: <reason>`. Exemptions are listed in the test's
expected set and reviewed in the PR, so the inventory is enforced rather
than asserted.

**Steps.**
1. Manifest from the 2026-09-09 inventory.
2. `artifacts.py` with tests on a temp manifest (hash, topological rebuild
   order, precedence).
3. `pull` to this machine; `verify` on both.
4. Migrate the listed loaders to `--role` with manifest defaults; fix
   `run_br2_gates.sh`.
5. [v2] Gate matrix, with required outcomes fixed before running:

   | Gate | What | Required |
   |---|---|---|
   | prop A/B | fixed-seed `prop_backtest` new vs recorded engine, paired per family | no CI-clean regression in any family; movements only where SIM1/SIM2/PROP3/engine changes fire |
   | G1 | winner-market LL parity, empirical selector, i7 ball model | ΔLL vs recorded within 0.002 |
   | G3 | `top_batter` paired delta | interval includes zero or favourable |
   | G5 | bowler coverage ≥90% on v2 set | ≥90% on the declared population; if a debutant fallback is added to the usage prior, it is a separate change with its own before/after, and the bar is not reinterpreted |
   | E2 | `highest_individual_mae` re-derived against the corrected match-level baseline | verdict restated either way |

   Recorded in `research/reports/auto/BR2.md`. [v3] prop A/B, G1, G3 and
   G5 are **mandatory**: the merge in step 6 does not happen until each
   meets its required outcome, or the user records a written exception in
   BR2.md. E2 is a restatement and cannot block.
6. Human review, merge `embeddings-ladder` to main, re-baseline the seeded
   numbers listed in TODO.md Phase 3b.
7. `scripts/daily/refresh_state.py` (item 3) is the one-command refresh;
   OPERATIONS § Operation 6 points to it.

**Acceptance.** `artifacts.py verify` passes on both machines; each gate in
the matrix has a pass/fail line in BR2.md and the four mandatory ones pass
(or carry a written exception); main carries the fixed engine; the
manifest-defaults scan test passes.

## 6. Paired multi-seed experiment harness and four candidates

**Problem.** Seed noise is 0.007 LL. Paired 5-seed runs exist only as
bespoke scripts (`d7_run.py`, `d12_run.py`, `run_i9_direct_seeds.py`) and a
bash loop in a report. The validation set does early stopping, tuning and
calibration at once; `calibrate_match_predictions.py:47` reads all of
`validation.parquet`. Serving uses one seed. Seed averaging was rejected
twice (E3, A2) under the old rule with LL deltas inside the noise floor, so
it is an open question, not a free win.

**Design.** `scripts/experiment_harness.py run <config.yaml>`: config names
the frame, baseline and candidate trainer args, seed list (default 29, 7,
13, 42, 101), slices, cost model, and `expected_minutes_per_seed`. Trains
both arms per seed (skipping seeds whose artifacts exist with a matching
config hash), evaluates through `market_math` and reslice, calls
`claim_gate.decide`, writes one JSON with per-seed rows and the verdict.
[v3] Escape hatch: `seeds: [29]` or `seeds: 3` allowed; the seed rule is
item 4's, applied by the gate, not by the harness (1 seed: match-cluster
estimator; 2 to 4: seed-mean estimator; both stamped `provisional` and
subject to the 0.007 point floor). Budget guard: refuses a run whose
seeds × expected minutes exceeds the configured ceiling without
`--allow-long`. [v3] Ensemble estimand: the candidate is the single
ensemble prediction set; the baseline is the deployed single-seed
production model (seed 29). The deciding statistic is the paired per-match
ΔLL, aggregated by mean, with a tournament-block bootstrap interval, under
the 1-seed rule (so `provisional`, and the 0.007 point floor applies).
Per-seed comparisons against the other four baseline seeds are reported as
diagnostics only. Profit uses the same pairing. [v2] The profit safety
check applies to every candidate including A. [v3] **Fold-local
preprocessing:** the trainer fits its label encoders across train,
validation and test (`xgboost_match_v1.py:269`); under the harness every
fold or split refits encoders on its training rows only, via a
`--fit-encoders-on {train,all}` flag defaulting to `train` inside the
harness and `all` for the legacy CLI path until the production config is
re-selected. [v4] **Unseen-category policy:** `_apply_encoders`
(`xgboost_match_v1.py:286`) currently calls `LabelEncoder.transform`,
which raises on a value absent from the fit set. Under `train` fitting,
unseen venues and competition tiers map to the reserved code `-1` at
transform time in training-fold evaluation, in test scoring, and in
serving alike (serving already flags unseen categories as a degraded
prediction; that flag stays and now carries the same code). A test trains
on rows without one venue and predicts a row at that venue successfully
with the `-1` code and the degraded flag set.

**Steps.**
1. Harness + schema + tests: dry run on the mini corpus, resume after a
   killed run reuses completed seeds, config hash mismatch forces retrain,
   provisional stamping, budget refusal.
2. Port `d12_run.py`'s frame checks as optional `verify` hooks.
3. Candidate A: logit-mean of five seed models under the ensemble estimand.
4. [v2] Candidate B: Platt on an isolated slice. `calibrate_match_predictions.py`
   gains `--calib-after <date>`: early stopping uses validation rows before
   the date, calibration rows on or after; the trainer receives the same
   split so no row is used twice.
5. [v2] Candidate C: rolling-origin folds for hyperparameter and early-stop
   selection. Feature rows are as-of by construction (CLAUDE.md critical
   invariant 2, temporal integrity [v3]), so folds are row masks by
   `match_date` over the existing frame, with encoders refit per fold: train ≤2022 /
   select 2023; ≤2023 / 2024; ≤2024 / 2025-H1. The production config is
   re-selected only if the fold-mean LL improves, then confirmed on the
   iteration set through the gate.
6. Candidate D: E4 quantile lineup pooling.

**Acceptance.** Harness tests above pass, plus [v3]: a pairing test (both
arms' prediction files carry identical match-id sets or the harness
refuses), a leakage test (no calibration row id appears in the early-stop
set, and no test-fold venue appears in a fold's encoder vocabulary unless
it appears in that fold's training rows), and the ensemble test (deciding
statistic is the mean paired ΔLL vs seed 29). Every candidate has a
harness JSON and a `results.tsv` row.

## 7. Simulator feature-builder consolidation

**Problem.** `scripts/sim_v1_2.py` has nine `extract_features` bodies. The
production one is `XGBoostModelV2` (1618-1916). `LSTMModelV1`,
`MLPModelV1`, `MLPModelV2`, `TransformerModelV1` carry 220-236 line copies.
[v2] They are not value-identical by design: the NN copies use encoder
code + 1 with unknown = 0 and `is_toss_winner` fixed at 0, while
`XGBoostModelV2` uses unshifted codes, unknown = −1, and the real toss
identity; `XGBoostModelV2` deliberately zero-fills and warns once on any
trained column it cannot produce.

**Design.** [v2] Two layers: `build_ball_context(state, ctx) -> dict` owns
the shared cricket facts (ids, counts, rates, phase flags, chase math,
h2h, momentum, pressure, toss identity); each wrapper applies its own small
encoding adapter (`encode_xgb`, `encode_nn`) that maps ids to codes with
its own unknown sentinel and offset. `XGBoostModelV2` keeps its buffer fill
and the warn-once zero-fill behaviour (a strict assertion would change
compatibility); new wrappers may opt into strict mode. User decisions:
keep LSTM and MLP wrappers; `TransformerModelV1` retirement is a separate
decision; `XGBoostModel` v1 moves to `archive/`.

**Steps.**
1. Golden-value test first: replay 2,000 balls from the mini corpus and one
   real match through all five builders; save dicts and the XGB array
   (copied, since the buffer is reused).
2. Extract `build_ball_context` and `encode_xgb`; route `XGBoostModelV2`;
   arrays bit-identical.
3. `encode_nn`; route the four NN wrappers; dicts identical, including the
   +1/0 conventions and the constant toss field.
4. Delete the copies; run the T1 parity suite and the BR2 prop A/B sample.

**Acceptance.** `sim_v1_2.py` shrinks by roughly 800 lines; golden-value
tests pass for all five; one context builder, two encoders.

## 8. Auto-research tidy and docs

**Problem.** `scripts/auto/` holds 72 scripts. 68 are referenced, mostly
only by their own report; 6 are live code dependencies; `a8_gate_analysis.py`
is imported by sixteen siblings. [v2] Scripts compute the repo root by
parent traversal (`d12_run.py:32`, `d7_run.py:41`: three parents), so
moving them one level deeper breaks their data and model paths. Docs drift
("49 features" vs 48; ARCHITECTURE counts; CLAUDE.md quick start).

**Design (user decision: archive inside the folder, never delete).**
`scripts/auto/archive/` is flat (one level deeper only). `archive_closed.py`
reads IDEAS.md status, `git mv`s scripts of closed ideas that no live code
imports, rewrites the root traversal in each moved file to one more parent,
and **rewrites every reference** to the old path in `*.sh`, `*.md`,
`*.yaml` and the manifest (it fails if any reference cannot be rewritten).
[v4] Relocation validation is **static only; archived scripts are never
imported or executed by any test**, because importing can write (for
example `b1_build_venue_encoder.py:28` creates a directory at import). A
test parses each moved file's AST and (a) evaluates every
`Path(__file__)...parent` chain and every path expression built from it,
asserting each resolves to the same absolute path recorded before the
move; (b) lists every string literal beginning with `scripts/`, `data/`,
`models/`, `research/`, `eval_out/`, `reports/` or a root-level odds file
name and classifies it as ROOT-anchored or cwd-relative; cwd-relative
literals are allowed only because the archive `README.md` states, as the
original folder did, that these scripts run from the repo root, and the
test asserts that each such literal still exists or matches a known
output pattern; (c) refuses with a named failure on any path expression it
cannot evaluate statically, unless that line carries an
`archive-path-exempt: <reason>` marker reviewed in the PR; (d) checks
cross-imports among moved scripts by resolving `import X` / `from X
import` names against the archive listing, statically. The six live dependencies and `a8`
stay; a `README.md` in the archive maps idea to report.

**Steps.** (1) `archive_closed.py` dry-run listing with pre-move path
resolutions recorded; (2) run it; (3) fix cross-imports among moved gate
scripts; (4) relocation test + static cross-import check; (5) docs pass
(CLAUDE.md quick start, ARCHITECTURE §5.2 and §6.14, feature count, TODO.md
pointer to this plan); (6) root clutter (two stray eval JSONs,
`night.sh`/`night_v2.sh`) to `archive/`.

**Acceptance.** `scripts/auto/` top level holds only live dependencies and
open ideas; relocation and static cross-import checks pass; no stale reference to a
moved path anywhere in the repo; `pytest` green; docs quote 48 features.

## 9. Backlog (recorded, not scheduled)

- **Toss- and lineup-blind model (data point, not primary).** Train and
  evaluate a match model that sees neither the toss nor the exact eleven
  (projected XI, four-branch toss average via
  `predict_fixture.toss_branches` weighted by the historical bat-first rate
  for venue and competition) against the market on the actual toss and
  lineup. The daily job logs both modes, so this becomes measurable.
- **Revisit every rejected or failed idea** (FAILED, TABLED, DISCARDED,
  SUPERSEDED in IDEAS.md and the E/I series in IMPROVEMENTS.md) with one
  question each: has new data (women's and Hundred odds now exist; intraday
  snapshots; the new ball-by-ball capture), better models, or web search
  changed the premise? Output: a table of idea, original blocker, what
  changed, re-run or stay closed.
- **Closing-line value and market-residual model** using the sibling repo's
  173 intraday snapshots; capture inputs now via item 3.
- **New ball-by-ball data (line, length, wagon wheel).** Availability and
  coverage audit, registry entry with a coverage stamp so missing is a state
  not a zero, then experiments through item 6.
- **Free-hit modelling** after the BR2 re-baseline.
- **Historical report sweep** with errata pointing to the toss-defect audit.
- **CLI packaging** (remove the 158 script-level `sys.path` bootstraps).
- **Women's members-only w1 slice; Hundred settlement; WBBL 2026** as
  listed in TODO.md.

## 10. Order and dependencies

```
1 CI ─┬─► 2 market_math ─┬─► 4 claim_gate ──► 6 harness ──► candidates A-D
      │        ▲         │                        │
      │        │         │                        └──► 4.5 TABLED reassess
      │        │         │                                (retrains via 6)
      │        │         └─► 3f score_daily                    │
      │   5.3 pull (needed for 2.6 re-baseline)           4.6 review
      ├─► 3 daily job: 3.1 protocol ──► 3a-e ──► 3f (after 2)
      └─► 5 manifest ──► 5.4 loaders ──► 5.5 BR2 gates ──► merge ──► 7 sim
                                                                  └──► 8 tidy
```

Items 1, 3.1 (protocol) and 5.1-5.3 can start in parallel. Item 7 waits
for the merge. Step 4.5 runs after item 6 exists, since TABLED ideas that
need retraining go through the harness.

## 11. Who implements and who reviews

| Item | Implementer | Reviewer |
|---|---|---|
| 1 CI, pytest config, fixtures | Codex Sol (medium) | Claude + Astra light |
| 2 market_math | Opus 5 | Astra light |
| 3 daily job | Sol for fetch/settle/refresh; Opus 5 for the `predict_fixture` refactor | Astra light + Claude |
| 4 claim_gate, log_verdict, program.md | Opus 5 | Astra light |
| 5 manifest, pull, BR2 | Sol | Claude |
| 6 harness | Opus 5 | Astra light |
| 7 sim consolidation | Opus 5 | Astra light + Claude |
| 8 tidy + docs | Sol | Claude |

Claude Fable coordinates, writes each item's acceptance checks before
implementation starts, and reads every diff. Note (2026-09-09): the Codex
model requested as `gpt-5.6-sol` identified itself as "GPT-5.4 Codex" on a
ping; confirm the routing before the first implementation task.
