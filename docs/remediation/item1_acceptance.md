# Item 1 acceptance checks (CI and test hygiene)

Written 2026-09-10 by Claude before implementation, per plan section 11.
Each check is a command plus the exact pass condition. All must pass.

| # | Check | Command | Pass condition |
|---|---|---|---|
| A1 | One path mechanism | `grep -ln "sys.path" scripts/tests/test_*.py tests/test_*.py` | zero matches (the `bench_*`/`check_*` scripts in `scripts/tests/` are standalone CLIs, not collected tests, and keep their bootstraps per plan step 1); `pyproject.toml` has `[tool.pytest.ini_options]` with `testpaths = ["tests", "scripts/tests"]` and `pythonpath = ["scripts"]`; no `conftest.py` outside `.venv`/`cricWAR` |
| A2 | Exceptional import normalised | `grep -n "^from\|^import" scripts/tests/test_prop_fair_baselines.py` | imports `from sim_eval.prop_fair_baselines import ...` |
| A3 | Single fixture module | `grep -ln "def _\?delivery\|def _\?innings\|def _\?match_json\|def _\?write_corpus" scripts/tests tests -r` | only `scripts/tests/cricsheet_fixtures.py` (test files import from it) |
| A4 | Mini corpus committed with generator | `ls tests/fixtures/cricsheet_mini/ \| wc -l`; generator re-run | ~20 match JSONs; two competitions, one same-day pair, one super over, one D/L result present; re-running the generator with the recorded seed reproduces byte-identical files |
| A5 | Honest skips | `uv run --no-sync pytest -q -rs` | every skipped test lists a reason; no `[SKIP]` print-and-return remains (`grep -rn "\[SKIP\]" scripts/tests tests` empty); `test_build_consistency` skips rather than fails without the cache |
| A6 | Markers registered | `uv run --no-sync pytest -q --strict-markers -m "not needs_artifacts"` | runs clean; `needs_artifacts` and `slow` in `[tool.pytest.ini_options] markers` |
| A7 | Volume-filter semantic test | `uv run --no-sync pytest -q scripts/tests/test_min_volume_filter.py -v` | synthetic cases: event vs market volume basis, missing volume, boundary at exactly 50,000; real-file pin asserts 255/168/110 on `betting_odds_polymarket_v2.json`, docstring says "regression pin" |
| A8 | Shipped-odds contract | `uv run --no-sync pytest -q scripts/tests/test_shipped_odds_contract.py -v` | both `_v2` files: head-to-head only, implied-prob sum in [0.98, 1.02], no quote at/after start, no duplicate cricsheet id; passes |
| A9 | Workflow | `cat .github/workflows/ci.yml` | old workflow deleted; triggers `push` and `pull_request` with no `paths`; step runs `uv run --no-sync pytest -q -m "not needs_artifacts"`; uv/python setup and the excluded-package list kept; `timeout-minutes: 20` |
| A10 | Artifact-free collection parity | rename `models/` and `data/` (and `betting_odds_polymarket.json` stays), `pytest --collect-only -q -m "not needs_artifacts"` before and after | identical test-id sets, zero collection errors, `cricWAR/` not collected. Note: renaming `data/` also hides the tracked `data/golden/betting_odds_golden_v2.json`, so the artifact-free *run* shows exactly one failure in the shipped-odds contract, by design (a missing tracked evidence file must fail, never skip) |
| A11 | Full suite | `uv run --no-sync pytest -q` (with artifacts) | 0 failures; count ≥ 352 minus nothing removed without a stated reason |
| A12 | Frozen evidence untouched | `git diff --stat main -- betting_odds_polymarket.json betting_odds_polymarket_v2.json data/golden/ reports/` | empty |
| A13 | Reviewer passes | Astra light on workflow + pyproject block + fixture module; Claude on the full diff | both written and any fixes re-run through A1–A12 |

## Result (2026-09-10)

Implemented by Codex Sol (medium); reviewed by Astra light and Claude.

| Check | Result |
|---|---|
| A1–A9, A11, A12 | pass on Claude's independent run |
| A10 | 331 identical ids with and without `models/`+`data/`; zero collection errors; 0 `cricWAR` ids |
| Suite with artifacts | 344 passed, 10 skipped (all reasoned), 354 collected (352 + 2 new) |
| A13 | Astra: 4 findings; Claude: 2 findings. All resolved below. |

Review findings and resolutions:

1. Astra, blocking: CI ran Python 3.12 but `uv.lock` pins numpy 1.24.3 with no cp312 wheel. Fixed: `ci.yml` uses 3.9 (matches `.python-version`). The inherited workflow had never run on GitHub, so this was latent.
2. Astra, blocking: `cricsheet_fixtures.match_json` keeps several partial-shape variants (appearance-list deliveries, outcome-free forward stub, `top_scores`). Downgraded to minor by Claude: those shapes are what the migrated tests deliberately exercise (lineup extraction from deliveries without runs; an outcome-blind forward stub). One home is achieved; tidying the keyword modes is deferred to item 8.
3. Astra, minor: `from tests.cricsheet_fixtures` relies on a namespace package named `tests` that an installed package could shadow. Fixed: bare `from cricsheet_fixtures import` (the test's own directory, under pytest's default prepend import mode).
4. Astra, minor: head-to-head check passed when both `market_question` and `event_title` were absent. Fixed: requires a non-empty question.
5. Claude: `bench_*`/`check_*` standalone scripts under `scripts/tests/` lost their bootstraps. Restored (plan step 1: CLI scripts keep theirs).
6. Claude: the shipped-odds contract skipped on a missing tracked file. Changed to a hard failure.

Deviations from plan text: CI Python 3.9 not 3.12 (finding 1); fixture module keeps mode keywords (finding 2). No frozen evidence touched.

CI: first run on the pushed branch, 2026-09-10, run 34443297003, green in
55 s (tests job). A9 and the "CI green" clause of the plan's acceptance are
met. Annotation only: GitHub's Node 20 deprecation notice on the action
versions, no effect on results.
