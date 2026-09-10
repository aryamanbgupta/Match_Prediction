# Item 1 acceptance checks (CI and test hygiene)

Written 2026-09-10 by Claude before implementation, per plan section 11.
Each check is a command plus the exact pass condition. All must pass.

| # | Check | Command | Pass condition |
|---|---|---|---|
| A1 | One path mechanism | `grep -rn "sys.path" scripts/tests tests` | zero matches; `pyproject.toml` has `[tool.pytest.ini_options]` with `testpaths = ["tests", "scripts/tests"]` and `pythonpath = ["scripts"]`; no `conftest.py` outside `.venv`/`cricWAR` |
| A2 | Exceptional import normalised | `grep -n "^from\|^import" scripts/tests/test_prop_fair_baselines.py` | imports `from sim_eval.prop_fair_baselines import ...` |
| A3 | Single fixture module | `grep -ln "def _\?delivery\|def _\?innings\|def _\?match_json\|def _\?write_corpus" scripts/tests tests -r` | only `scripts/tests/cricsheet_fixtures.py` (test files import from it) |
| A4 | Mini corpus committed with generator | `ls tests/fixtures/cricsheet_mini/ \| wc -l`; generator re-run | ~20 match JSONs; two competitions, one same-day pair, one super over, one D/L result present; re-running the generator with the recorded seed reproduces byte-identical files |
| A5 | Honest skips | `uv run --no-sync pytest -q -rs` | every skipped test lists a reason; no `[SKIP]` print-and-return remains (`grep -rn "\[SKIP\]" scripts/tests tests` empty); `test_build_consistency` skips rather than fails without the cache |
| A6 | Markers registered | `uv run --no-sync pytest -q --strict-markers -m "not needs_artifacts"` | runs clean; `needs_artifacts` and `slow` in `[tool.pytest.ini_options] markers` |
| A7 | Volume-filter semantic test | `uv run --no-sync pytest -q scripts/tests/test_min_volume_filter.py -v` | synthetic cases: event vs market volume basis, missing volume, boundary at exactly 50,000; real-file pin asserts 255/168/110 on `betting_odds_polymarket_v2.json`, docstring says "regression pin" |
| A8 | Shipped-odds contract | `uv run --no-sync pytest -q scripts/tests/test_shipped_odds_contract.py -v` | both `_v2` files: head-to-head only, implied-prob sum in [0.98, 1.02], no quote at/after start, no duplicate cricsheet id; passes |
| A9 | Workflow | `cat .github/workflows/ci.yml` | old workflow deleted; triggers `push` and `pull_request` with no `paths`; step runs `uv run --no-sync pytest -q -m "not needs_artifacts"`; uv/python setup and the excluded-package list kept; `timeout-minutes: 20` |
| A10 | Artifact-free collection parity | rename `models/` and `data/` (and `betting_odds_polymarket.json` stays), `pytest --collect-only -q -m "not needs_artifacts"` before and after | identical test-id sets, zero collection errors, `cricWAR/` not collected |
| A11 | Full suite | `uv run --no-sync pytest -q` (with artifacts) | 0 failures; count ≥ 352 minus nothing removed without a stated reason |
| A12 | Frozen evidence untouched | `git diff --stat main -- betting_odds_polymarket.json betting_odds_polymarket_v2.json data/golden/ reports/` | empty |
| A13 | Reviewer passes | Astra light on workflow + pyproject block + fixture module; Claude on the full diff | both written and any fixes re-run through A1–A12 |
