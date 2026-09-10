# Stage 0 acceptance checks (sequence and embeddings track)

Written 2026-09-10 by Claude Fable before any stage 0 work started, per
`docs/SEQUENCE_TRACK_PLAN.md` § "Stage 0" and § "Stage 0 kickoff brief".
Each table is written before its deliverable begins; the "Result" block under
each table is filled in only when the deliverable is done, with numbers copied
verbatim from files. A run that did not finish has no number.

Standing rules for every deliverable: no read of `data/golden/` or
`data/forward_holdout/`; nothing written under `models/` outside
`models/embeddings/seq_stage1/`; no edit to frozen evidence, committed
reports, `TODO.md`, `IMPROVEMENTS.md`, `program.md`, `research/results.tsv`,
`models/MANIFEST.yaml`, `scripts/daily/`, `daily/`, the live-state build dirs,
or the moved-aside `*_local_20260910` / `*_export_*` dirs. Python runs through
`uv run --no-sync`. Everything runs on the laptop.

Design facts established while reading (2026-09-10), which the checks below
rely on:

- `SameDayReplayStatsProvider`'s tracker view implements every provider
  method `XGBoostModelV2`'s feature builder calls, so the production ball
  model can be driven through the same replay lifecycle as the T1 wrapper.
  Stage 1 therefore runs every arm through one runner with one chronology,
  which is what makes identical prediction-time cutoffs possible.
- `sim_v1_2.SimulationEngine` seeds one global `random` / `np.random` stream
  per simulation (`random_seed + i`); outcome sampling, extras, selector
  draws and run-outs all draw from it. Separate named streams need an
  engine-wide RNG refactor of the BR2-gated engine. Stage 0 registers the
  per-fixture seed derivation and the three named sub-seeds, and records
  stream separation as a deviation (D5 check 5.9).
- The ablation checkpoints (`models/embeddings/t1_ablation_v1_mps/`) were
  trained on `data/xgb_data_v3`, and `sim_t1.py` refuses any other frame, so
  B and C serve from the legacy v3 stats cache while A and A50 serve from
  the i7 cache. This is a registered cross-arm asymmetry, not something
  stage 0 can remove (D5 check 5.10).
- The run-out rate is a code constant (`sim_v1_2.RUNOUT_P = 0.075077`); the
  `models/auto/d15/runout_rates.json` artifact it cites is absent on this
  checkout. The pin records the constant and the source line.
- `prop_backtest.py` constructs `XGBoostModelV2` directly and has no T1 seam;
  the stage 1 realism and prop scorers therefore consume the arm runner's
  own per-simulation output rather than re-simulating.

---

## D1. This file

| # | Check | Pass condition |
|---|---|---|
| 1.1 | Written first | this file exists in the commit history before any file under `research/sequence_track/`, `scripts/sequence_track/`, or `models/embeddings/seq_stage1/` (single stage-0 commit; the file's own tables precede their Result blocks) |
| 1.2 | Coverage | one table each for D2–D7, numbered check ids, a pass condition per row, and a Result block per table filled after the fact |

---

## D2. Queue runner (`research/sequence_track/run_queue.sh`, `queue.yaml`)

| # | Check | Pass condition |
|---|---|---|
| 2.1 | Schema | `queue.yaml` has top-level `defaults` (`memory_floor_gb`, `memory_cap_gb`, `poll_seconds`, `stop_file`) and a `jobs` list; each job has `id`, `config`, `command`, `output_dir`, `expected_hours`, `machine`; a job missing any field or with a duplicate `id` makes the runner exit non-zero before launching anything |
| 2.2 | Order and machine tag | jobs run in listed order; a job whose `machine` does not match `--machine` (default `laptop`) is skipped with a log line |
| 2.3 | Skip-if-complete | a job is skipped when `output_dir/COMPLETE` exists and its recorded `config_sha256` equals sha256 of the job's `config` file; a changed config re-runs the job and a passing run rewrites `COMPLETE` |
| 2.4 | Timeout | each job runs under `caffeinate -i` and `perl -e 'alarm shift; exec @ARGV'` with alarm = `expected_hours * 2 * 3600` (integer, minimum 1 s); a job exceeding it is killed and `output_dir/TIMEOUT` is written |
| 2.5 | STOP file | if the configured stop file exists when a job would start, the runner exits 0 without launching it; the stop file is checked between jobs, never mid-job |
| 2.6 | Logs | every job's stdout and stderr tee to `output_dir/run.log` (appended across retries), with start/exit-code/end stamps |
| 2.7 | Retry | a non-zero exit (including timeout and memory kill) retries once; a second failure writes `output_dir/FAILED` and the runner continues to the next job (does not abort the queue) |
| 2.8 | Memory floor | before each launch the runner reads available memory from `vm_stat` (free + inactive + speculative pages × page size) and refuses to launch, writing `output_dir/REFUSED_MEMORY`, when it is below `memory_floor_gb`; the reading and threshold are logged |
| 2.9 | Memory watchdog | while a job runs, the runner polls the job's process-tree RSS every `poll_seconds` (default 30) and kills the tree when it exceeds `memory_cap_gb`, writing `output_dir/KILLED_MEMORY` with the observed RSS |
| 2.10 | No side channels | the runner contains no `claude`, `codex`, `git`, or `log_verdict` invocation (grep is the test) |
| 2.11 | Dry run | `--dry-run` prints the resolved job list, the skip/run decision for each, and the memory reading, and launches nothing |
| 2.12 | Tests | `scripts/tests/test_run_queue.py` exercises 2.1, 2.3, 2.4, 2.5, 2.7, 2.8, 2.9 and resume-after-STOP on 10-second dummy jobs (a sleeper, a non-zero exiter, and a small allocator with a low cap); passes under `uv run --no-sync pytest -q scripts/tests/test_run_queue.py`; no test needs artifacts |
| 2.13 | Resume | after a STOP mid-queue, removing the stop file and re-running skips the completed jobs (2.3) and runs the rest |

### Result (D2)

Recorded 2026-09-10. Implemented by an Opus subagent; reviewed by Fable, one
review finding fixed (below). Files: `research/sequence_track/run_queue.sh`,
`research/sequence_track/queue.yaml` (example queue with one placeholder
stage-1 job), `research/sequence_track/README.md`,
`scripts/sequence_track/queue_lib.py` (schema/TSV resolve, `vm_stat`
reading, process-group RSS + deadline watchdog),
`scripts/tests/test_run_queue.py`.

| # | Result |
|---|---|
| 2.1 | PASS — missing field, duplicate id, or missing `defaults` → exit 2 before any launch. A config path that does not exist yet is recorded as `absent:<path>` (so the shipped placeholder job runs; the signature changes when the real config lands) |
| 2.2 | PASS — `--machine` (default laptop) mismatch → skip with a log line |
| 2.3 | PASS — skip on `COMPLETE` with matching `config_sha256`; changed config re-runs and rewrites the marker |
| 2.4 | PASS after fix — `caffeinate -i` + perl alarm at `int(expected_hours*2*3600)` (min 1). Review finding: the alarm signals only the exec'd process, so a job with a backgrounded child outlived it and `wait` blocked on the tee pipe (measured 61 s per attempt, no markers). Fix: the watchdog also enforces the deadline on the process group and writes `TIMEOUT` (`killed_by: watchdog`); post-fix 10 s for both attempts, no survivors. Regression test verified failing on the pre-fix runner |
| 2.5 | PASS — pre-existing STOP → exit 0 with nothing launched; STOP written by job 1 → job 2 not launched; checked only between jobs |
| 2.6 | PASS — `run.log` appended across attempts with start / command / exit / end stamps |
| 2.7 | PASS — one retry, then `FAILED`; the queue continues. `REFUSED_MEMORY` does not consume a retry (the job never ran) |
| 2.8 | PASS (after Astra MUST-FIX 1) — available = (free + inactive + speculative) × page size from `vm_stat`, GiB units; the reading is taken before **every** attempt, a below-floor reading or a failed reading (`QUEUE_VM_STAT_CMD` seam for tests; rc 3 / 4) refuses the launch with `REFUSED_MEMORY` and no `FAILED`; regression tests verified failing on the pre-fix runner (`23 passed`) |
| 2.9 | PASS — RSS polled per process group every `poll_seconds`; 300 MB allocator (also as a grandchild) under a 0.1 GiB cap → `KILLED_MEMORY` with observed RSS, no survivors |
| 2.10 | PASS — grep for `git`, `claude`, `codex`, `log_verdict` empty in runner, helper, and queue.yaml (asserted by test) |
| 2.11 | PASS — `--dry-run` prints jobs, decisions, timeout, sha, and the memory reading; launches nothing |
| 2.12 | PASS — `uv run --no-sync pytest -q scripts/tests/test_run_queue.py`: `20 passed in 27.25s` (repeats 25.7–26.9 s); no artifacts needed |
| 2.13 | PASS — after STOP, removing it and re-running skips job 1 (COMPLETE) and runs job 2 |

---

## D3. Readiness record (`docs/sequence_track/stage0_readiness.md`)

| # | Check | Pass condition |
|---|---|---|
| 3.1 | Artifact verify | full output of `uv run --no-sync python scripts/artifacts.py verify` pasted; every role OK except `bowler_roster_policy` may not be MISSING here (it is laptop-only by record) |
| 3.2 | BR2 status | the gate table from `research/reports/auto/BR2.md` quoted verbatim: prop A/B PASS, G1 PASS, G3 PASS, G5 PASS BY WRITTEN EXCEPTION, E2 restated |
| 3.3 | T1 tests | `tests/test_run_sim_eval_t1_lifecycle.py`, `tests/test_sim_t1_snapshot_guard.py`, `tests/test_sim_t1_prefix_cache.py`, `tests/test_sim_t1_parity.py` run and pass; counts recorded |
| 3.4 | Gate exercised | `scripts/sim_eval/claim_gate.py --kind match_model` run on the HA0 swap-smoke evidence (`experiments/results/item6_swap_smoke/{candidate,baseline}_seed*/sliced_50000.json`, odds role `odds_iteration_v2`) to a scratch output; verdict line quoted; the regenerated payload's verdict equals the committed `gate.json` verdict |
| 3.5 | Verdict exercised | `research/log_verdict.py` run end to end (claim then verdict, `--dry-run`) against a scratch copy of `research/IDEAS.md` carrying a synthetic HA0 entry, a scratch results.tsv copy, and the 3.4 gate JSON; the printed diff and results row are captured; `git diff --stat -- research/results.tsv research/IDEAS.md` is empty afterwards |
| 3.6 | Full suite | `uv run --no-sync pytest -q` passed/failed/skipped counts recorded; 0 failures (a pre-existing failure unrelated to stage 0 is recorded by test id with the commit that introduced it, and does not block) |
| 3.7 | Machine | `sysctl -n hw.memsize`, `sysctl -n hw.ncpu`, free-disk, and the vm_stat available-memory reading recorded with timestamps |

### Result (D3)

Recorded 2026-09-10 (full text in `docs/sequence_track/stage0_readiness.md`).

| # | Result |
|---|---|
| 3.1 | PASS — `artifacts.py verify` 19:19:55 IST: all 24 roles OK (including `bowler_roster_policy`) |
| 3.2 | PASS — BR2 gate table quoted in the readiness file: prop A/B PASS, G1 PASS, G3 PASS, G5 PASS BY WRITTEN EXCEPTION, E2 RESTATED: parity |
| 3.3 | PASS — the four T1 test files: `23 passed, 4 warnings in 1.12s` |
| 3.4 | PASS — `claim_gate.py --kind match_model` on the ten item6 swap-smoke `sliced_50000.json` files, odds role `odds_iteration_v2`: printed `LANDED 23f840a5…`; regenerated verdict LANDED = committed `gate.json` verdict LANDED; `inputs` block identical |
| 3.5 | PASS — `log_verdict.py claim HA0` on a scratch IDEAS copy (PENDING → RUNNING), then `verdict HA0 LANDED --dry-run` with the 3.4 gate JSON: printed the IDEAS diff and the results.tsv row (`gate_sha256=23f840a5…`), ended `DRY RUN — nothing written`; `git diff --stat -- research/results.tsv research/IDEAS.md` empty. Note: `--result-text-file` is required, not optional as the docstring example implies |
| 3.6 | PASS — final, after every stage 0 file landed (2026-09-11 00:28 IST): `uv run --no-sync pytest -q` → `828 passed, 5 skipped, 9 warnings in 85.80s`; artifact-free `-m "not needs_artifacts"` → `801 passed, 32 deselected`. An interim run had one failure (`test_manifest_defaults`, hard-coded artifact literals in `pin_stage1.py`), fixed by resolving manifest roles through `artifact_path` with no exemption added | Historical counts on the way: 678 passed (2026-09-10 20:11, before the Astra rounds), 758 and 815 passed after rounds 1 and 2 |
| 3.7 | PASS — hw.memsize 51,539,607,552 (48 GiB), hw.ncpu 15, disk 533 GiB free, vm_stat available (free+inactive+speculative) 15.31 GiB at 19:20:44 IST |

---

## D4. A50 ball model (`models/embeddings/seq_stage1/a50/`)

| # | Check | Pass condition |
|---|---|---|
| 4.1 | Column list | the 50 columns are `embeddings_e1.EB_BAT_COLS + EB_BOWL_COLS + VENUE_COLS + CTX_COLS + transformer_t1.STATE_COLS`, obtained by import in the trainer, length 50, no duplicates, all present in `data/xgb_data_i7/cricket_data_i7_{train,validation,test}.parquet` |
| 4.2 | Frame | `a50/data/a50_{train,validation,test}.parquet` are column selections (50 features + label + `match_id`/`match_date` bookkeeping) of the i7 frame with identical row counts (1,876,971 / 124,292 / 186,667) and row order; `a50/data/feature_hash.json` records the column list, its sha256, and the source parquet md5s |
| 4.3 | Config | hyperparameters equal `experiments/configs/xgb_v6_hierarchical_shrink.yaml` `model.hyperparameters` (n_estimators 444, max_depth 10, lr 0.24036372383981375, subsample 0.8776663421127178, colsample_bytree 0.7424085095268674, reg_alpha 0.8503122682099661, reg_lambda 0.18186045420525845), `random_state` 29, `eval_metric` mlogloss, `early_stopping_rounds` 100 on the validation split, no `sample_weight` (the `--no-class-weights` semantics of `xgboost_v2.py`), label encoding identical to `xgboost_v2.py`; all recorded in `a50/training_contract.json` |
| 4.4 | Production untouched | `xgboost_v2.py` unchanged (`git diff --stat scripts/xgboost_v2.py` empty); `artifacts.py verify` still OK afterwards |
| 4.5 | Artifacts | `a50/xgboost_model_a50.pkl` (md5 recorded), `feature_columns_a50.txt` (the 50 names in order), `metrics.json` with train/validation/test ball log loss and `best_iteration`, and copies of the production encoders/outcome-dist config byte-identical to `models/xgb_i7_noweights_production/` so the serving wrapper loads the same sidecars |
| 4.6 | Serves | `XGBoostModelV2` loads the A50 directory and returns a 6-class probability vector on one synthetic state (test), using only the 50 columns |
| 4.7 | Reported | train/val/test LL recorded here verbatim; the production 114-feature test LL 1.4253 is quoted for context only, no claim |

### Result (D4)

Recorded 2026-09-10 from `models/embeddings/seq_stage1/a50/metrics.json` and
`training_contract.json` (implemented by an Opus subagent; reviewed by Fable).

| # | Result |
|---|---|
| 4.1 | PASS — 50 columns from `EB_BAT_COLS(18)+EB_BOWL_COLS(18)+VENUE_COLS(6)+CTX_COLS(4)+STATE_COLS(4)`, imported, unique, all present in the three i7 parquets |
| 4.2 | PASS — rows 1,876,971 / 124,292 / 186,667; `DataFrame.equals` against the source columns True on all three splits; after Astra rounds 1–2 any frame reuse hashes exactly the files the invocation reads, requires the recorded paths to resolve to those files (a copied frame dir with its old manifest is refused), and refuses on any md5/row/column mismatch (`--verify-frames-only` on the real frames: all six md5s equal the recorded ones; artifacts untouched, `8 passed`); the i7 frame has no `match_id`, so bookkeeping is `innings_id`, `ball_idx`, `match_date` (recorded in `data/feature_hash.json`); feature-columns sha256 `defaf0306649a81ee7987da98beeb1b85ca561686fd9a5cc3bfa9b845823972e` |
| 4.3 | PASS — hyperparameters read from `xgb_v6_hierarchical_shrink.yaml` at run time; `random_state` 29, mlogloss, `early_stopping_rounds` 100 on validation, no `sample_weight` kwarg, label map `{0:0,1:1,2:2,4:3,6:4,7:5}` with −1 → 7 (wicket), identical to `xgboost_v2.py`; `n_jobs=8` is an addition recorded in the contract (a rerun at another thread count is not guaranteed bit-identical) |
| 4.4 | PASS — `git diff --stat scripts/xgboost_v2.py` empty; `artifacts.py verify` all 24 roles OK after the run |
| 4.5 | PASS — `xgboost_model_a50.pkl` md5 `7705b330bac547674d8e569f4b28a278`; `feature_columns_a50.txt`; `metrics.json` with `best_iteration` 24; the five production sidecars copied byte-identical under both `_i7` and `_a50` names; an extra `training_contract_a50.json` so the wrapper resolves delivery semantics explicitly |
| 4.6 | PASS — `XGBoostModelV2` loads the directory unchanged ("50 features"), builds all 50 columns without the zero-fill warning, returns a 6-vector summing to 1 (`scripts/tests/test_train_a50.py`: 3 passed, 2 marked `needs_artifacts`) |
| 4.7 | Ball log loss, verbatim: train 1.3449013907578484, validation 1.441408870047355, test 1.4332497087527303 (accuracy 0.4728 / 0.4391 / 0.4356); fit 32.3 s, wall 37.3 s, early stop fired at round 124. Context only, no claim: production 114-feature test LL of record is 1.4253 |

---

## D5. Stage 1 config (`experiments/configs/seq_stage1_sim_v1.yaml`, `scripts/sequence_track/pin_stage1.py`)

| # | Check | Pass condition |
|---|---|---|
| 5.1 | Pin script | `pin_stage1.py --write` fills every hash field in the YAML from the files; `pin_stage1.py --verify` recomputes and exits non-zero on any mismatch; `--verify` passes on the committed file |
| 5.2 | Per-arm artifacts | for A, A50, B, C: model dir path + dir hash (manifest contract `md5_file_or_sorted_relative_path_md5_lines_v1`, computed with `artifacts.md5_directory`) and booster/checkpoint file md5; stats version and cache md5 (`models/player_stats_cache_i7.sqlite` for A/A50, `models/player_stats_cache_v3.sqlite` for B/C); `--t1-context-dir data/t20s_json` with its dir hash; `data/all_players_enriched.csv` sha256; `models/bowler_phase_usage.json`, `models/b10_usage_corpus.pkl`, `models/bowler_roster_policy.json` hashes; `models/auto/b18/extras_graft_v1.json` sha256 (must equal `ad6e863b…` from the PPC report) applied to every arm via `T1_EXTRAS_GRAFT_PATH` and the equivalent XGBoost-arm seam; run-out rule = `sim_v1_2.RUNOUT_P` value and line; odds role `odds_iteration_v2` with the registry sha256; `models/prop_fair_baseline_corpus_v2.pkl` md5; C114 listed with `status: deferred` and no artifacts |
| 5.3 | Checkpoint choice | B = `t1_ablation_v1_mps/mlp/seed_<s>`, C = `t1_ablation_v1_mps/full/seed_<s>` where `<s>` minimises `splits.validation.arms.<arm>.per_seed[].ll` in `summary.yaml` (ties → lowest seed number); the script prints the per-seed table and the YAML records the chosen seed, its validation LL, and the rule; test LL is not consulted |
| 5.4 | Seeds | `base_seed` registered; per-fixture seed = low 31 bits of sha256(f"{cricsheet_id}:{base_seed}"); three named sub-seeds (outcome, extras, selector) derived as sha256 of the per-fixture seed with the stream name; the runner passes the per-fixture seed as `SimulationConfig.random_seed`; a second base seed is registered for the 1b variability rerun |
| 5.5 | Clipping | match win probability clipped to [0.01, 0.99] at one seam in the evaluator before log loss, edge, and bet placement, default off so existing runners and BR2 behaviour are unchanged; regression test covers on and off |
| 5.6 | Convergence protocol | candidates 100, 200, 400, … with no ceiling; three independent simulation batches per arm per candidate on the 1b timing shard; stop at the smallest n at which the 95% range of the paired C−B and B−A winner LL across batches is below 0.002; the chosen n and table fields are placeholders marked `to_be_filled_before_1d` |
| 5.7 | Decision rule | primary slice ≥$50k; contrasts C−B, B−A, C−A, A50−A, all candidate minus reference in log loss; 0.007 equivalence margin with the four outcomes (parity / favourable / adverse / inconclusive) spelled out; advancement rule; Holm adjustment over the three confirmatory contrasts; A50−A labelled exploratory system contrast |
| 5.8 | Commands | one complete command line per arm (A, A50, B, C), runnable from the repo root, with every flag and env var explicit; the 1a smoke commands differ from the full-run commands only in the fixture dir, `--n-sims`, and output dir |
| 5.9 | Deviations recorded | the YAML has a `deviations` block naming (i) single global RNG stream in the engine (stream separation deferred, with the reason) and (ii) any other place the implementation falls short of the plan text |
| 5.10 | Asymmetries recorded | the YAML has a `known_asymmetries` block: stats cache i7 vs v3 across arms; training frame i7 (A, A50) vs v3 (B, C); A's stage-1 path is the replay lifecycle, not the BR2 snapshot runner, so BR2's G1 line is not a stage-1 reference number |
| 5.11 | Forbidden data | `forbidden_data: [data/golden, data/forward_holdout]`; no path under either appears anywhere in the YAML |

### Result (D5)

Recorded 2026-09-10. `experiments/configs/seq_stage1_sim_v1.yaml` (generated,
936 lines), `scripts/sequence_track/pin_stage1.py`,
`scripts/tests/test_pin_stage1.py` (58 tests) by an Opus subagent; reviewed
by Fable, who also re-ran `--verify`.

| # | Result |
|---|---|
| 5.1 | PASS — `--write` regenerates every hash and selection field; after Astra MUST-FIX 4 every registered command passes `--config`, and `run_arm.py` runs the verify function before launching, refuses on any mismatch between its effective flags and the arm block, and records `config_sha256` / `config_verified` in provenance (this fired for real once when the runner was edited after the last `--write`); `--verify` rebuilds the payload from disk and diffs key by key, exit 1 naming the key on mismatch; on the committed file: `pin_stage1: OK — … matches every recomputed fact` (only `provenance.pins_generated_at` and `provenance.git_head_short` are recorded-not-compared, and that exclusion list is itself compared) |
| 5.2 | PASS — provenance pins 34 source files on the execution path (runner, evaluator, gate, reducers, engine and model stack, replay/rehydration/identity/settlement modules; absent list empty) plus runtime versions (python 3.9.25, xgboost 2.1.4, torch 2.5.1, numpy 1.24.3, pandas 2.3.2, scikit-learn 1.6.1); per arm: model dir + `md5_directory` hash, checkpoint md5, stats version + cache md5, `data/t20s_json` dir hash (11,264 JSONs), player metadata sha256, usage / B10 / roster md5s, extras graft sha256 asserted `= ad6e863b…bfa1` (PPC report line 59) applied via `T1_EXTRAS_GRAFT_PATH` (B, C) and the `extras_graft` attribute (A, A50), run-out `sim_v1_2.RUNOUT_P = 0.075077` line 106 read by import, odds role `odds_iteration_v2` re-hashed against `docs/registered_odds.json`, prop corpus v2 md5, fixture set 255, prefix cache off, cpu, 4 threads; C114 `status: deferred`, `artifacts: null` |
| 5.3 | PASS — B = `mlp/seed_101` (validation LL 1.437755; seeds 7/13/29/42/101 = 1.438134 / 1.438039 / 1.438053 / 1.438600 / 1.437755); C = `full/seed_101` (1.436634; 1.436972 / 1.438353 / 1.437451 / 1.437245 / 1.436634). Computed by the script; the tie-break and "test split unused" are covered by synthetic tests |
| 5.4 | PASS — `base_seed` 20260910, batch base seeds 20260910 / 20260911 / 20260912 registered for the 1b convergence batches with a per-arm `timing_1b` block (fixture dir absent until 1b, re-pin after it is built) and a `variability_rerun` note; the preflight accepts only a registered block's fixture dir, base seed, n_sims and device (round-2 fix: n_sims 999 and device mps are refused); formulas match `run_arm.fixture_seed` / `sub_seed` (tested across 6 ids × 4 base seeds); the runner passes the fixture seed to `SimulationConfig.random_seed` (changing only `--base-seed` changes the draws) |
| 5.5 | PASS — `MatchLevelEvaluator.prob_clip` seam, default `None` = legacy; registered [0.01, 0.99]. Finding: the evaluator already clipped unconditionally to [0.05, 0.95], so the seam *replaces* those bounds and the registered clip is wider than every previous run, BR2 G1 included; `prob_clip=None` is byte-identical to before (test) |
| 5.6 | PASS — candidates 100, 200, 400, … no ceiling; three batches per arm; stop rule 95% range of paired C−B and B−A < 0.002; `chosen_n_sims: to_be_filled_before_1d`; the full-run `--n-sims` renders from the same constant so command and protocol cannot disagree |
| 5.7 | PASS — decision rule block as specified; A50−A exploratory |
| 5.8 | PASS with a stated exception — the smoke commands are complete and were executed verbatim for D7 (the four arms ran from the YAML's command strings); the full-run commands are complete except the single `<chosen_n_sims>` token, deliberately unresolved until 1d (recorded in `full_run.command_completeness`) |
| 5.9 | PASS — deviations recorded (post-Astra: the RNG overlap is screened, not argued away — `seeds.overlap_check` covers each base seed alone and the **joint** seed line over 20260910 / 20260911 / 20260912 that the 1b batches share: joint minimum gap 3,290, so `timing_1b.n_sims` (the permitted list enforced by runner and audit) stops at 3,200 and 6,400 stays in `candidates_screened` marked `not_permitted_without_new_seeds` (a 6,400 launch is refused even though the 10-fixture shard alone would not collide) (the colliding pair 1493246@20260910 / 1512733@20260911 is recorded); `run_arm` screens the block's whole seed cohort in preflight and refuses on any overlap; `--config` now checks `--n-sims` and the resolved device too; `--allow-unregistered` never permits mixing registered and unregistered arms; single global RNG stream unchanged) (single RNG stream and the additive `seed+i` overlap point; extras-graft attribute seam; run-out sidecar absent; scorers consume raw sims; no parallel; clip replaces; `resolved` reads cricsheet; odds-less as-of stamp; roster selector by factory swap) |
| 5.10 | PASS — four asymmetries recorded: i7 vs v3 stats cache; i7 vs v3 training frame; replay path vs BR2 snapshot; BR2 G1 ran with the plain selector, snapshot provider and 0.05/0.95 clip, so it is not a stage-1 number |
| 5.11 | PASS — `data/golden` and `data/forward_holdout` appear only under `forbidden_data`; a build-time guard rejects any other occurrence |

---

## D6. Cross-arm audit (`scripts/sequence_track/audit_cross_arm.py`)

| # | Check | Pass condition |
|---|---|---|
| 6.1 | Provenance source | every arm run writes `arm_provenance.json` in its output dir with, per fixture: cricsheet id, match date, as-of stamp (date plus the ordered list of same-day fixture ids advanced before the prediction lock, plus `matches_advanced` at lock), eligibility (`odds_row_found`, `resolved` = actual winner present, `male_t20` from cricsheet `info.gender`/`match_type`), per-fixture seed, n_sims; and per run: arm id, model dir hash, stats cache md5, selector class + usage/roster hashes, extras sidecar sha256, run-out constant, clip bounds, engine md5 (`sim_v1_2.py`), runner md5 |
| 6.2 | Fixture sets | the audit exits non-zero unless the fixture id sets are identical across all given arm dirs |
| 6.3 | As-of stamps | identical per fixture across arms (date, ordered predecessor list, matches_advanced) |
| 6.4 | Eligibility | identical per fixture across arms for all three flags |
| 6.5 | Selector and sidecars | identical selector class, usage hash, roster hash, extras sidecar sha256, run-out constant, clip bounds, n_sims across arms; model dir hash and stats cache md5 are *expected* to differ and are listed, not asserted |
| 6.6 | Negative tests | `scripts/tests/test_audit_cross_arm.py`: a fixture removed from one arm, a changed stamp, a flipped eligibility flag, and a changed sidecar hash each fail with a message naming the arm and fixture; the identical case passes |
| 6.7 | Smoke run | the audit passes on the D7 smoke output dirs for A, A50, B, C; its output is pasted in the Result block |

### Result (D6)

Recorded 2026-09-10. `scripts/sequence_track/audit_cross_arm.py` and
`scripts/tests/test_audit_cross_arm.py` (12 tests) implemented by an Opus
subagent alongside the runner; reviewed by Fable.

| # | Result |
|---|---|
| 6.1 | PASS — `arm_provenance.json` (contract `sequence_track_arm_provenance_v2` after Astra MUST-FIX 3) with every listed field plus, per fixture, the resolved odds row identity and sha256, `odds_actual_winner`, `cricsheet_resolved`, `cricsheet_winner`, `scored`, `skip_reason`, and per run `config_sha256`, `config_verified`, `odds_sha256`, `context_dir_hash`. The as-of stamp of an odds-less fixture is taken at `begin_match` since no lock exists for it |
| 6.2–6.5 | PASS — after Astra rounds 1–2: a typed mandatory schema is enforced on every arm before comparison (a field missing from all arms fails), empty fixture sets fail, the fixture set must equal the JSON stems of the recorded fixture dir (`--expected-fixtures`), each arm's provenance is validated against its registered block in the config file whose sha256 it recorded (non-null identity required; model dir, checkpoint md5, stats cache, graft, usage/roster, odds, context, clip, threads, device, engine/runner md5 vs `provenance.source_md5`, block name, base seed and n_sims — Astra's uniformly-wrong-hash bypass now fails per arm and field), the claimed block is bound to the run (provenance fixture dir, `fixture_dir_hash` and `fixture_count` must equal the block's pins; a block with no pinned inventory such as `timing_1b` before 1b cannot be claimed), per-fixture seeds and sub-seeds are recomputed from the base seed and cricsheet id with the runner's own functions, per-fixture and run n_sims must agree with the block's permitted list, player-metadata sha256 and run-out probability are compared to the pins, and the asserted-identical fields are: fixture set, as-of stamps, all nine eligibility/scoring fields, selector class, usage md5, roster-policy md5, extras sidecar sha256, run-out constant, clip bounds, n_sims, base seed and per-fixture seeds, `config_sha256`, `odds_sha256`, `context_dir_hash`, `engine_md5`, `runner_md5`; `config_verified: false` fails unless `--allow-unregistered`. Model dir hash, checkpoint md5, stats version and cache md5 are listed, not asserted |
| 6.6 | PASS — synthetic negative cases (fixture removed, stamp changed, eligibility flipped, sidecar changed, seed changed, odds row / winner / scored / skip_reason changed, empty sets, 19 uniformly-missing-field cases, wrong types, fixture-dir mismatch both ways, unregistered run, shared-input hash drift) each exit 1 naming arm/fixture/field; identical case passes. `test_run_arm.py` + `test_audit_cross_arm.py` + `test_pin_stage1.py` + `test_manifest_defaults.py`: 155 passed |
| 6.7 | PASS — on the final D7 smoke dirs: `CROSS-ARM AUDIT PASSED: 4 arms (A, A50, B, C), 4 fixtures, schema complete, checks 6.2-6.5 identical.` 4/4 scored on every arm; `config_sha256`, `odds_sha256` (`dc36aef3…`), `context_dir_hash`, engine and runner md5 identical; Selector `RosterEmpiricalBowlerSelector` on all arms; usage md5 `2e650423…`, roster md5 `f2a5efa2…`, graft sha256 `ad6e863b…`, run-out 0.075077, clip 0.01/0.99, base seed 20260910 identical; listed-only fields as expected (A booster `7ee1e180…`, A50 `7705b330…`, B `5cfb36ab…`, C `504c3b64…`; caches i7 `671ac820…` / v3 `3faf24a6…`) |

---

## D7. 1a smoke

| # | Check | Pass condition |
|---|---|---|
| 7.1 | Fixture choice | 4 fixtures from `data/polymarket_test_v2` chosen by a stated deterministic rule and recorded by cricsheet id: one same-day pair (both evaluated, so the second sees the first replayed), one fixture at ≥$50k volume, one further fixture; copied as JSON into `models/embeddings/seq_stage1/smoke/fixtures/` |
| 7.2 | Every arm loads | A, A50, B, C each start, print their model banner, and finish all 4 fixtures at `--n-sims 10` with exit code 0; wall time per arm recorded |
| 7.3 | Sliced JSON | each arm's output dir holds the evaluator JSON and a reslice (`reslice_eval_json.py`, `--odds` = the `odds_iteration_v2` path, `--cluster-source-dir data/polymarket_test_v2`) with `all`, `50000`, `100000` slice files present |
| 7.4 | Realism scorer | `scripts/sequence_track/score_realism.py` runs on each arm's per-simulation output and writes `realism.json` with innings-score P10/P50/P90 coverage, first-innings bias, extras per innings, wickets, unique bowlers; numbers not read |
| 7.5 | Prop scorer | `scripts/sequence_track/score_props.py` runs on each arm's per-simulation output against `prop_fair_baselines` (corpus v2) and writes `props.json` with per-family Brier for sim and baseline; the family list is recorded; numbers not read |
| 7.6 | Audit | D6 passes on the four arm dirs (6.7) |
| 7.7 | Odds resolution | on a single fixture chosen from the BR2 G1 record as the one with the largest model-vs-market edge, arm A at 10 sims yields a record with non-empty market probabilities, `bet_placed: true`, and non-null `realized_pnl`; if the chosen fixture does not place a bet at 10 sims, the check is repeated on a scratch odds file with an extreme price for the same fixture, and that fallback is recorded |
| 7.8 | No numbers read | no log loss, Brier, ROI, or edge from the smoke appears in this file, the readiness file, the report to the user, or any commit message; the per-arm JSONs are kept only under `models/embeddings/seq_stage1/smoke/` (gitignored) |
| 7.9 | Stop | nothing beyond the smoke (no 1b) runs before the user's explicit go |

### Result (D7)

Recorded 2026-09-10. Fixtures, commands and outputs live under
`models/embeddings/seq_stage1/smoke/` (gitignored). No log loss, Brier,
ROI, or edge value from these runs was read or is reported anywhere.

| # | Result |
|---|---|
| 7.1 | PASS — rule: all fixtures on the earliest date in `data/polymarket_test_v2` that carries ≥2 fixtures (2025-10-31: `1478908`, `1501896`, `1505127`; `1478908` is ≥$50k), plus the lowest cricsheet id at ≥$50k not on that date (`1477609`). All four are male T20 with odds rows |
| 7.2 | PASS — every arm exit 0 with its banner (A "114 features", A50 "50 features", B `mlp/seed_101` 272,006 params, C `full/seed_101` 298,758 params, context_capacity 200, prefix cache off); B18 graft and the causal roster policy ACTIVE on all four. Final run (2026-09-11 00:2x IST, after Astra round 3) from the pinned YAML's registered `smoke_1a.command` strings (config sha256 `f1898f19d50c3d79e75e7872d14cd80e8ef063a77a2e488746cb416cdf02dd26`; every arm `config_verified: true`; the audit validates each arm's provenance against its registered block, including the block's pinned fixture-directory hash and count). Wall time per arm (4 fixtures × 10 sims, warm context, including the pin verification preflight): A 9 s, A50 9 s, B 8 s, C 10 s; reported simulation time A 1.3 s, A50 1.2 s, B 1.4 s, C 3.5 s; wall A 8 s, A50 9 s, B 8 s, C 10 s. Note the YAML sha256 moves on every `--write` (`pins_generated_at`), so the stable integrity check is `pin_stage1.py --verify`, not the quoted sha. A first cold pass took 431 s in the context-corpus scan alone (disk cache), not in simulation |
| 7.3 | PASS — evaluator JSON + `eval.json` per arm; `reslice_eval_json.py` wrote `eval_all.json`, `eval_min_volume_50000.json`, `eval_min_volume_100000.json` for every arm (exit 0) |
| 7.4 | PASS — `score_realism.py` (reuses `t1_ppc_common` metrics) exit 0 on all four arms, 4/4 fixtures scored each, `realism.json` with first/second-innings bias, P10–P90 coverage, extras events, wickets, unique bowlers, completed-chase block, and the batting-first minus chasing summary; numbers not read |
| 7.5 | PASS — `score_props.py` (reuses `prop_backtest` family builders and `prop_fair_baselines`, corpus role `prop_fair_baseline_corpus_v2`; both scorers reject sealed paths and validate every fixture path derived from a match id — pattern check, resolve inside the fixture dir, symlink targets — before opening; prop rows are sorted so `props.json` is byte-reproducible across `PYTHONHASHSEED`) exit 0 on all four arms; 15 families scored (`top_batter`, `batter_50plus`, `batter_runs_mae`, `innings_runs_ou_{160_5,170_5,180_5}`, `pp_total_ou_{45_5,50_5,55_5}`, `team_highest_individual_ou_{29_5,34_5,39_5}`, `bowler_wkts_{1,2,3}plus`), 18 skipped with the missing raw field named per family (per-sim fours/sixes, bowler runs conceded, the innings ball log, no fair baseline for `p_tie` and economy). `highest_individual_mae` is skipped because the corpus of record predates the match-level top-score column; an unregistered `prop_fair_baseline_corpus_v3.pkl` exists on disk but is not in the manifest — a stage 1 decision, not changed here. Numbers not read |
| 7.6 | PASS — see 6.7 |
| 7.7 | PASS — fixture `1512749` (largest model-vs-market edge in the BR2 G1 record, no synthetic odds needed; run with the registered arm A command minus `--config`, since the odds-check dir is not a registered block): arm A at 10 sims produced a record with non-empty `market_prob` for both teams, `bet_placed: true`, `bet_team` set, `realized_pnl` non-null, `price_rejected: false` |
| 7.8 | PASS — this file, the readiness file, and the report carry no smoke metric |
| 7.9 | PASS — nothing beyond the smoke has run |

---

## D8. Astra review and commit

| # | Check | Pass condition |
|---|---|---|
| 8.1 | Rounds | each `codex exec -m gpt-6-astra` round's output file path and verdict recorded below; iteration continues until a round returns SIGN-OFF with no MUST-FIX |
| 8.2 | Commit | one commit of only files created by stage 0 (plus the minimal runner/evaluator edits listed in D5/D6), plain summary + body, no AI-attribution trailer; hash recorded in the final report |

### Result (D8)

Astra rounds (`codex exec -m gpt-6-astra -c model_reasoning_effort="low" -s read-only`, prompt covering D1–D7 and every stage 0 file; output files in the session scratchpad, verdict lines quoted):

| round | output file | verdict | disposition |
|---|---|---|---|
| 1 | `<scratchpad>/astra_stage0_1.md` (2026-09-10 20:14–20:16 IST) | **NO SIGN-OFF** — 5 MUST-FIX, 3 SHOULD, 4 NOTE | MUST-FIX 1 (memory check outside the retry loop, launch on measurement error) → queue agent; 2 (audit passes on empty sets / uniformly missing fields) , 3 (`resolved` from cricsheet is not the scoring eligibility; record odds-row identity, winner, scored set), 4 (pins not enforced at launch; audit must assert config/odds/context/engine/runner hashes), 5 (RNG overlap prose wrong; add an interval-overlap guard) → runner agent; SHOULD 8 (scorer sealed-path guards) → scorer agent; SHOULD 6 (pin the source closure) and 7 (validate reused A50 frames) → second wave; NOTE 11 (qualify the "no sealed read" claim) fixed in the readiness file; NOTE 9, 10, 12 accepted as stated |
| 2 | `<scratchpad>/astra_stage0_2.md` (20:43–20:46 IST) | **NO SIGN-OFF** — round-1 items 1, 2, 3, 9–12 confirmed fixed/accepted; MUST-FIX 4 residual (n_sims and device escape the `--config` check; audit trusts the `config_verified` boolean instead of validating provenance against the registered pins; no registered block for the 1b convergence batches so the second batch seed is refused) and MUST-FIX 5 residual (overlap screened per base seed, not jointly across batch seeds; cross-seed gap 3,290 collides at 6,400 sims); SHOULD 6 (four execution dependencies missing from the source closure), 7 (frame verification not bound to the paths actually trained on), 8 (scorer filenames derived from raw match ids are unguarded) | 4, 5, 6 → runner agent; 7 → A50 agent; 8 → scorer agent |
| 3 | `<scratchpad>/astra_stage0_3.md` (2026-09-11 00:16–00:19 IST) | **NO SIGN-OFF** — round-2 items 4 (runner side), 5 (arithmetic), 6, 7, 8 confirmed; residual MUST-FIX: (1) the audit never binds the selected block's fixture directory/inventory to the provenance (relabelled smoke records passed as `timing_1b`); (2) `n_sims_permitted_jointly` is recorded but not enforced by runner or audit; (3) per-fixture seeds/sub-seeds/counts are compared only across arms, and player-metadata hash and run-out probability are not in the registration comparison | all three → runner agent |
| 4 | `<scratchpad>/astra_stage0_4.md` (2026-09-11 00:31–00:34 IST) | **SIGN-OFF** — round-3 items 1, 2, 3 confirmed fixed with file:line; no outstanding MUST-FIX; one SHOULD (the D3 table's 3.6 check row had been overwritten by its result and the result block quoted a stale count — fixed in this file, earlier counts labelled historical); notes: the 828-pass suite was not independently rerun by the reviewer, and sign-off covers stage 0 only, the explicit-go requirement for 1b stands | closed |
