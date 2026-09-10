# Item 2 acceptance checks (one betting-math module with a cost model)

Written 2026-09-10 by Claude before implementation, per plan section 11.
Step 6 (re-baseline on the production model's `test_predictions.json`) is
deferred until the Mac-mini pull (item 5 step 3) exists; steps 1–5 are in
scope here.

| # | Check | Command / method | Pass condition |
|---|---|---|---|
| B1 | Module exists with exactly the named primitives | `grep -n "^def \|^class " scripts/sim_eval/market_math.py` | `implied_probs`, `edge`, `settle_flat`, `kelly_fraction`, `settle_kelly`, `expected_value`, `CostModel` (with `none()`); no placement policy inside |
| B2 | CostModel accounting matches plan §2 line by line | `uv run --no-sync pytest -q scripts/tests/test_market_math.py -v` | tests for: spread/fee unit conversion; `0<q<1` rejection (no clipping); `q_eff` cap at `1-1e-9`; win/loss returns at both fee bases; losing bet under `stake`; EV and Kelly on the same `o_eff`; general two-outcome Kelly when `loss_return != -1`; `f=0` when `win_return<=0` or numerator non-positive; bankroll bound `f*l<=1`; constructor rejects `spread>=1`, `fee>=1`, negatives, unknown basis; `none()` == (0,0,winnings) |
| B3 | Parity fixture frozen BEFORE refactor | `git log --follow scripts/tests/fixtures/market_math_parity.json` (or chosen path) and the test | fixture of ≥50 records produced by the unrefactored `MatchEvaluator` on synthetic inputs, committed in a commit that precedes any change to the six call sites; the test proves `settle_flat(..., CostModel.none())`, `kelly_fraction`, `settle_kelly`, `expected_value` reproduce the recorded `realized_pnl`, `full_kelly_fraction`, `full_kelly_pnl`, `fractional_kelly_pnl`, `expected_value` bit-for-bit (`==`, not `isclose`) |
| B4 | Six copies deleted, arithmetic routed | `grep -n "odds - 1.0\|odds-1.0\|1.0 / price - 1.0" scripts/sim_eval/match_evaluator.py scripts/sim_eval/blend_eval_json.py scripts/predict_fixture.py scripts/hundred_roi_eval.py scripts/sim_eval/sizing_rules.py scripts/build_ipl_dashboard.py` | zero hits; each file imports from `sim_eval.market_math`; `blend_eval_json._recompute_realized_pnl` and `sizing_rules._compute_pnl` arithmetic bodies gone |
| B5 | Policies unchanged | `uv run --no-sync pytest -q scripts/tests/test_predict_fixture.py scripts/tests/test_predict_fixture_identity_mode.py scripts/tests/test_eval_math.py -k "not needs"` and any hundred/sizing tests | all existing tests pass with their assertions unmodified (`git diff` on those test files shows no assertion change) |
| B6 | Zero-cost numbers identical | synthetic eval JSON fixture (built from the mini corpus or hand-written, ≥30 matches, some with edge≤0, some odds<1, some missing winner) run through `reslice_eval_json.py` before and after | every summary number identical to four decimals; `cost_model`, `price_basis`, `volume_basis` stamped |
| B7 | Cost flags present | `--help` on `run_sim_eval.py`, `reslice_eval_json.py`, `blend_report.py`, `predict_golden.py`, `hundred_roi_eval.py` | `--spread-bps`, `--fee-bps`, `--fee-basis` on all five; reslice recomputes profit from stored `market_odds` under the cost model (test: a nonzero spread changes profit; `realized_pnl` field in the input is ignored, proven by corrupting it) |
| B8 | Volume basis | `reslice_eval_json.py --volume-basis market` on the synthetic fixture | event vs market slices differ where the fixture makes them differ; default remains `event`; summary stamps the basis |
| B9 | Scenario reporting | `blend_report.py --cost-scenarios ...` | zero-cost column first and labelled as the gate's safety check; the five default scenarios reported next to it; scenario list is configuration not code |
| B10 | Dashboard intentional change | new test in `scripts/tests/` | `build_ipl_dashboard.compute_bet` now de-vigs; a test fixture where raw and de-vigged edge straddle zero shows the decision flip; change listed in IMPROVEMENTS.md under "intentional changes" |
| B11 | Frozen evidence untouched | `git diff --stat main -- betting_odds_polymarket.json betting_odds_polymarket_v2.json data/golden/ data/forward_holdout/ reports/` | empty |
| B12 | Full suite | `uv run --no-sync pytest -q` | 0 failures |
| B13 | Reviewer passes | Astra light on `market_math.py` + its tests; Claude on the full diff | written, fixes re-run through B1–B12 |

## Result (2026-09-10, steps 1–5)

Implemented by Codex Sol (medium). Sol's sandbox could not write `.git`, so
Claude made the two ordered commits after regenerating the parity fixture in
a clean worktree at the pre-refactor HEAD: sha256 identical
(`babb2be2…`), which is the B3 proof.

Suite: 367 passed, 10 skipped (352 → 367: +15 item-2 tests).

Flagged disagreements between today's arithmetic and plan §2 (parity wins,
behaviour unchanged, listed here because Sol's report omitted them):

1. Legacy `match_evaluator._calculate_realized_pnl` and
   `sizing_rules._compute_pnl` accept decimal odds of exactly 1.0 (q = 1,
   zero win return). `market_math` rejects q ≥ 1 per the plan, so both
   callers keep a two-line boundary case for `odds == 1.0` under
   `CostModel.none()`. This is the only arithmetic left outside the module.
2. `predict_fixture.compute_bet` previously accepted `d == 1.0` and now
   returns `suppression_reasons: ["invalid_odds"]` for it; all 29 tests pass
   and the A7 policy is retired, so this is recorded, not reverted.
3. Claude patch: at zero spread the effective odds returned the raw decimal
   odds even when `q` exceeded the `1 - 1e-9` cap; now the raw odds are used
   only when the cap did not bind.

Step 6 (re-baseline on the production `test_predictions.json`) remains
deferred to the Mac-mini pull (item 5 step 3).

## Fix pass (Astra review)

1. **Reslice missing-price and corrupted-P&L handling.** New-format records
   with explicit placement always recompute from `market_odds`, so numeric
   and null stored-P&L corruption produce identical output. A placed record
   without `market_odds` retains stored P&L only under `CostModel.none()`;
   any nonzero cost makes P&L null. Both paths stamp `pnl_unrecomputable`.
   Legacy rows without explicit placement still use non-null P&L solely as
   the settled-ness sentinel needed by `flat_bet_team`; it is not reused as
   profit when a price is available.
2. **Evaluator price boundary.** `settle_flat_policy` and
   `settle_kelly_policy` in `eval_statistics.py` are now the sole home for
   the legacy zero-cost odds-1.0 behavior. With any nonzero cost, the strict
   market primitive rejects the price; the evaluator excludes the record
   from returns and stamps `price_rejected` in its summary.
3. **Sizing unresolved outcomes.** `_compute_pnl` now returns null for an
   unresolved winner, and `evaluate` stamps `unresolved_excluded`. This is
   the documented intentional change from the legacy lost-bet treatment.
4. **Blend parity.** Realized flat P&L at odds 1.0 and zero-fraction Kelly
   P&L now use the shared policy helpers and match the legacy evaluator
   exactly. The parity generator now includes positive-edge/non-positive-
   Kelly overround-gap cases. Claude regenerated the frozen fixture from the
   pre-refactor commit `ec4e59a` in a clean worktree: 72 records (60 → 72);
   the original 60 differ only in their `id` numbering.
5. **Numeric coverage.** Market-math tests now pin the zero-spread effective-
   price cap, winnings-basis fee EV/Kelly values, and rejection of non-finite
   cost constructor inputs.
6. **Prediction boundary.** A regression test pins `compute_bet` odds 1.0
   suppression as `invalid_odds`, and the intentional change is recorded in
   `IMPROVEMENTS.md`.

Second Astra pass: five of six resolved; one residual (evaluator with an
unresolved winner skipped price validation, so a positive-edge bet at odds
1.0 under a nonzero cost model raised inside the EV calculation). Fixed by
Claude at both evaluator sites: the EV/Kelly block now catches the price
rejection, stamps `price_rejected`, and zeroes the sizing outputs; regression
in `scripts/tests/test_eval_math.py`.

## Step 6 result (2026-09-10)

Production `test_predictions.json` pulled from the Mac mini (model md5
`54faf586…`). Re-baseline in `reports/item2_rebaseline_20260910.md`: on the
same inputs as the recorded I17 evaluation, zero-cost numbers are identical
on all three slices (0 numeric differences, 0 per-record profit
mismatches); cost-scenario columns and stamps are the only new output. Sol's
headline comparison in the report body is confounded by stored pre-v2 prices
in the I17 envelope; see the addendum. **Item 2 is complete.**
