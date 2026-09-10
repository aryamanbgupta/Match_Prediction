# Item 5 acceptance checks (artifact manifest, reproducibility, BR2 gates)

Written 2026-09-10 by Claude before implementation. Steps 1–4 are in scope
now (artifacts pulled from the Mac mini 2026-09-10; md5s verified against
the plan: ball model `7ee1e180…`, match model `54faf586…`). Step 5 (gate
matrix) and 6 (merge) get their own checks appended before they start.

## Steps 1–2: `models/MANIFEST.yaml` and `scripts/artifacts.py`

| # | Check | Pass condition |
|---|---|---|
| E1 | Manifest schema | every entry has role, path, kind (`file`/`dir`), hash, producing command, input roles, promoting doc; a dir hash is the md5 of the sorted `relative_path md5` lines; the two production models, the i7 stats cache, the i7 ball frame, the i7 match frame of record, live state, bowler usage prior, prop fair-baseline corpus, and the three evidence sets (iteration v2, golden v2, sealed forward) are present |
| E2 | Recorded hashes match the plan and CLAUDE.md | ball booster md5 begins `7ee1e180`, match model.pkl md5 begins `54faf586`; forward-holdout directory sha256 begins `725719e6` if that scheme is retained alongside md5 |
| E3 | `artifacts.py verify` | passes on this checkout for every pulled role; a role whose path is absent reports MISSING, not an exception; a deliberately corrupted temp copy reports MISMATCH with the role name |
| E4 | `artifacts.py path <role>` | prints the manifest path; `--role` on a loader overrides the default; an explicit CLI path overrides `--role` (precedence test) |
| E5 | `artifacts.py rebuild <role>` | on a temp manifest with a diamond dependency, prints the topological order; refuses when an input's recorded hash mismatches; dry-run flag prints commands without running |
| E6 | `artifacts.py pull --from mac-mini <role>` | rsync then verify; refuses to overwrite a role whose local hash already matches unless `--force`; never touches paths outside the role |
| E7 | Live-state versioning | `data/live_state_i7` is a symlink to `data/live_state_i7_<as_of>_<build_utc_ts>/`; the build dir carries a `BUILT` marker written last; a refresh refuses a dir that has one; a second refresh on the same `as_of` makes a new dir and moves the link; the manifest records the current target. Test on a temp tree |
| E8 | Tests | `scripts/tests/test_artifacts.py` covers E3–E7 on temp manifests/trees; no test reads a real artifact (mark `needs_artifacts` if one must) |

## Step 4: loader migration and the manifest-defaults scan

| # | Check | Pass condition |
|---|---|---|
| E9 | Production tier migrated | `predict_fixture.py`, `prop_backtest.py`, `predict_golden.py`, `run_sim_eval.py` (+ `--model-version`), `refresh_golden_i7.sh`, `run_br2_gates.sh`, `build_bowler_phase_usage.py`, and the `sim_v1_2.py` sidecar defaults resolve through `artifacts.path(role)`; behaviour with no flags unchanged (existing tests pass) |
| E10 | `run_br2_gates.sh` G1 | passes an explicit `--model` for the i7 ball model role |
| E11 | Scan test | `scripts/tests/test_manifest_defaults.py`: AST scan of `scripts/**/*.py` string constants and defaults + regex scan of `*.sh`; literals starting `models/`, `data/live_state`, `data/xgb_`, `data/golden/polymarket_test`, `data/polymarket_test`, `eval_out/` fail unless the line or file header carries `manifest-exempt: <reason>`; the expected exemption set is listed in the test and reviewed here |
| E12 | Exemption inventory | the three named diagnostics carry markers; every other exemption is listed in this file's Result section with its reason |
| E13 | Frozen evidence untouched | `git diff --stat main -- betting_odds_polymarket*.json data/golden/ data/forward_holdout/ reports/` empty; pulled evidence under `data/golden` and `data/forward_holdout` is byte-identical to the tracked versions where tracked |
| E14 | Suite | `uv run --no-sync pytest -q` 0 failures; artifact-free collection still identical |
| E15 | Review | Claude reviews the full diff (plan §11: item 5 reviewer is Claude) |

## Result, steps 1–4 (2026-09-10)

Implemented by Codex Sol; reviewed by Claude (E15). `artifacts.py verify`
reports OK for all 24 roles on this checkout; the two production hashes match
the plan (ball booster `7ee1e180…`, match model `54faf586…`). Live state
migrated to `data/live_state_i7_2026-07-30_20260801T050221Z/` with a `BUILT`
marker and the `data/live_state_i7` symlink; `predict_fixture` resolves
through it. Loader migration: precedence explicit path > `--role` > manifest
default, verified in the diffs of the eight listed production loaders;
`run_br2_gates.sh` G1 now passes explicit i7 model paths. Item 5 tests: 8
passed; artifact-free collection identical (414 ids).

Exemption inventory (E12): 72 `scripts/auto/` scripts (closed-idea scripts,
archived by item 8), the three named diagnostics, and 33 further scripts
grouped in the test as diagnostics/replay tools, trainers and frame builders
that own configurable namespaces, legacy model tooling, embeddings-ladder
experiments, isolated research paths, and shell reproduction scripts. Claude's
review note: the trainer/builder group is broader than the plan's "diagnostic
one-offs" wording; it is accepted because those scripts take their paths as
required CLI inputs in every documented use, and item 6's harness passes
explicit paths. Any new production default must migrate, not join the list.

E14 carries three known failures in `test_forward_eval_contract.py`, an
environment fact rather than a code defect: this checkout's `data/t20s_json`
is a fresh cricsheet 1.2.0 export that contains the 137 sealed forward
fixtures, so the sealed-set preflight correctly fails closed; the same tests
pass on the Mac mini whose pool is the frozen corpus of record. Resolution
(replace the local pool with the mini's, keeping the fresh export aside) is
awaiting the user's decision. Not on the mini: `models/bowler_roster_policy.json`
(local-only, recorded in the manifest).

Steps 5–6 (gate matrix, merge) remain open.
