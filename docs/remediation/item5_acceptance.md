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

Update (2026-09-10, later): with the user's approval the local fresh cricsheet
export was renamed `data/t20s_json_fresh_export_20260805/` and the mini's
frozen corpus (11,264 files, 1.0.0 export) synced into `data/t20s_json`;
the legacy `models/xgb_v3` and `models/xgb_match_v3_m7_production` rollback
artifacts were pulled because the sealed-set preflight verifies them. Full
suite on this checkout: **443 passed, 5 skipped, 0 failed** (E14 met).

## Step 5: BR2 gate matrix checks (written before running; run on the laptop, 2026-09-10)

Required outcomes are fixed by plan §5 step 5. Inputs: recorded pre-fix detail
`reports/prop_calibration_detail_emp_n261.json` (pulled from the mini,
untracked evidence), `run_br2_gates.sh` with manifest roles, seed 42,
100 sims, empirical selector, OMP threads capped.

| Gate | Required | How read |
|---|---|---|
| prop A/B | no CI-clean regression in any family; movements only where SIM1/SIM2/PROP3/engine changes fire | `compare_selector_eval.py` paired output per family |
| G1 | winner-market ΔLL vs recorded within 0.002 | `run_sim_eval` sliced summary vs the recorded ≥$50k line |
| G3 | `top_batter` paired delta interval includes zero or favourable | prop A/B row |
| G5 | bowler coverage ≥90% on the v2 set | `check_bowler_coverage.py` |
| E2 | `highest_individual_mae` restated against the corrected baseline | verdict restated either way; cannot block |

Recorded in `research/reports/auto/BR2.md`; the four mandatory gates must
pass or carry a written user exception before the merge (step 6).

### Gate-matrix run 1 (2026-09-10): two defects found before any gate could be read

1. **Zero bets in every simulator eval since 2026-08-14.** The one-sided-book
   guard in `BettingOddsLoader.get_implied_probabilities` (commit 094532a)
   compares parsed sides to `len(odds)`, and the v2 odds rows carry a
   `timestamp` inside `odds.winner`, so every row returned `{}`: no market
   probabilities, zero edge, zero bets, G1 unreadable. Fixed by excluding
   metadata keys before the count; regression
   `scripts/tests/test_odds_metadata_keys.py`. Not an item 2 regression (the
   loader was untouched by item 2); a pre-existing branch defect the gate
   matrix exposed.
2. **Prop A/B ran on 30 matches and paired nothing.** `prop_backtest.py`
   defaults `--n-matches 30` and `run_br2_gates.sh` did not override it; the
   recorded detail is keyed by the legacy display id while the new detail is
   cricsheet-keyed, so the paired join was empty. `compare_selector_eval.py`
   gains `--join-key display_match_id`; the full 255-match backtest is
   re-run.

G1's recorded reference (`reports/prop_selector_comparison_n60.md`) is an
n=30 empirical-vs-random delta on the old engine, not an absolute LL on this
population, so G1 is read as the plan defines it (empirical selector not
worse than random by more than 0.002) on the full v2 set under the fixed
engine, both arms re-run. G5 read 89.7% on 3,061 slots (2025: 90.5%, 2026:
89.3%), below the 90% bar as the plan anticipated; that requires a written
user exception or a separate debutant-fallback change, never a
reinterpreted bar.
