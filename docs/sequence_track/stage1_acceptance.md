# Stage 1 acceptance checks (sequence and embeddings track)

Written 2026-09-11 by Claude Fable before any stage 1 work started, per
`docs/sequence_track/stage1_kickoff_brief.md` § 1 and
`docs/SEQUENCE_TRACK_PLAN.md` § "Stage 1". Each table is written before its
deliverable begins; the "Result" block under each table is filled in only
when the deliverable is done, with numbers copied verbatim from files. A run
that did not finish has no number. Deliverables D9–D12 (1b, 1c, 1d, gate) get
their tables written immediately before each of those steps starts, after the
user's explicit go for that step, and never before.

Standing rules for every deliverable (unchanged from stage 0): no read of
`data/golden/` or `data/forward_holdout/`; nothing written under `models/`
outside `models/embeddings/seq_stage1/`; no edit to frozen evidence,
committed reports, `IMPROVEMENTS.md`, `program.md`, `research/results.tsv`,
`models/MANIFEST.yaml`, `scripts/daily/`, `daily/`, the live-state build
dirs, the ablation checkpoints under `models/embeddings/t1_ablation_v1_mps/`,
or the moved-aside `*_local_20260910` / `*_export_*` dirs (`TODO.md` backlog
entries are editable on the user's instruction, as done in D2). Python runs
through `uv run --no-sync`. Everything runs on the laptop. No log loss,
Brier, ROI or edge from the 1a smoke or the 1b timing shard is read or
reported anywhere. Opus subagents implement; Fable reviews every diff;
Codex Astra reviews to sign-off before any commit that lands results; Fable
is the only committer and the only caller of `research/log_verdict.py`.

Design facts established while reading (2026-09-11), which the checks below
rely on:

- `data/xgb_data_i7/.feature_hash` declares `delivery_semantics:
  inclusive_total_runs_v1`, `venue_alias_version: venue_aliases_v1`,
  `venue_alias_sha256: 853b32b0…`, `venue_alias_active_count: 94`, split
  boundaries and `k_player` / `k_venue`. `data/xgb_data_v3/.feature_hash`
  declares none of the identity or semantics fields. Both carry the same
  114-feature `hash` `c520a3ba08ae`, so that hash identifies the feature
  list, not the frame.
- `scripts/transformer_t1.py` hardcodes the `cricket_data_v3_{split}` stem
  (`load_split`), unconditionally loads `models/embeddings/eval_kit/`
  masks and probes after training and before saving, and rounds every
  stored log loss to four decimals (`round(mean_ll, 4)`).
- `scripts/sim_t1.py` refuses a checkpoint whose recorded `config.data_dir`
  lacks the substring `xgb_data_v3`. It never inspects the serving cache.
- `scripts/sequence_track/pin_stage1.py` resolves B and C checkpoints from
  the hard-coded `ABLATION_DIR` and selects seeds from
  `t1_ablation_v1_mps/summary.yaml` (`splits.validation.arms.<arm>.per_seed`).
- The stats-cache backend exposes `_meta` through `get_meta()`; both the i7
  and legacy caches store the same global prior π (identical
  `prior_p*` rows) because both were built from the same 9,519-match corpus.
- The iteration set (255 fixtures, 2025-07-01 → 2026-04-16) lies entirely in
  the i7 **test** split (train ends 2024-12-30, validation 2024-12-31 →
  2025-06-29); checked by fixture id during the Astra cache review. Training
  B and C on train with validation selection touches no iteration label,
  exactly as for A and A50.

---

## D1. This file

| # | Check | Pass condition |
|---|---|---|
| 1.1 | Written first | this file exists before any code change or training run for stage 1; its tables precede their Result blocks |
| 1.2 | Coverage | one table each for D2–D8 now; D9–D12 tables added before their step starts and after the user's go for that step |

---

## D2. Cache decision record

| # | Check | Pass condition |
|---|---|---|
| 2.1 | Decision | the user's decision (2026-09-11) is recorded verbatim in substance: i7 cache for all four arms; B and C retrained on `data/xgb_data_i7`, five seeds (7, 13, 29, 42, 101), selection by validation log loss only, ties to the lowest seed; legacy v3 rejected; venue recency deferred to backlog |
| 2.2 | Astra round | the Codex Astra review of the decision (`gpt-6-astra`, low, read-only) is recorded with its verdict line and every MUST-FIX mapped to a check id in D3–D6 |
| 2.3 | Corrected claims | the record states that B−A and C−A remain *system* contrasts (114 vs 50 features) after the cache unification, that the i7 checkpoint report's winner comparison is paired on 255 fixtures and directionally adverse to i7 without any interval excluding zero, and that "clean identity-only comparison" is withdrawn as a description of it |
| 2.4 | Corpus-prior limitation | the global outcome prior's whole-corpus computation is recorded as an inherited limitation of the iteration screen with the measured table (train-only vs whole-corpus π; max feature shift by k/(n+k)) from a reproducible script; the record states shared exposure across arms and an unknown, unmeasured differential effect on each model |
| 2.5 | Backlog | `TODO.md` carries (a) the corrected fold statement (81 disjoint pairs, 63 cutovers in 2020-21, 13 overlapping), (b) the "recency weight subsumes a hard era split; one registered candidate set" framing, (c) the new "Global outcome prior is not as-of-date" item with the measured table |

### Result (D2)

Recorded 2026-09-11.

| # | Result |
|---|---|
| 2.1 | Decision by the user, 2026-09-11, in conversation with Fable: "We can go ahead with i seven and retraining of b and c … for a comparison right now, just a single one should be fine." i7 for A, A50, B, C; B and C retrained on `data/xgb_data_i7`, seeds 7/13/29/42/101, validation-LL selection, ties to lowest seed; legacy v3 rejected; venue recency and the as-of prior deferred to the backlog |
| 2.2 | Astra review, prompt `<scratchpad>/astra_cache_decision_prompt.md`, output `<scratchpad>/astra_cache_decision_1.md`, verdict line: `VERDICT: AGREE WITH CHANGES — MUST-FIX: 1. Record corpus-prior leakage and correct comparison claims. 2. Decouple training from old eval-kit artifacts. 3. Rewire pinning to new checkpoints and full-precision seed selection. 4. Require complete provenance and fail-closed compatibility tests.` Mapping: MUST-FIX 1 → 2.3, 2.4, 2.5, 6.6; MUST-FIX 2 → 3.2; MUST-FIX 3 → 3.4, 6.1, 6.2; MUST-FIX 4 → 3.5, 4.1–4.6. Its four file:line claims (`transformer_t1.py:343` kit load, `:358` rounding; `pin_stage1.py:386` selection source, `:625` `ABLATION_DIR`; `build_stats_cache.py` final-state prior) were each verified by Fable against the source before acceptance |
| 2.3 | Recorded: B−A and C−A are system contrasts (A: 114 features; B, C: 50) even on one cache; A50−A isolates the feature set at equal information within one model family; C−B is the registered full-sequence T1 versus token-MLP contrast (the architectures differ in outcome-history embedding, attention and parameter count, so it is not a pure sequence-memory isolation). The i7 checkpoint report (`reports/i7_rebuild_checkpoint_20260725.md`, "Same-simulator paired result") is paired on 255 fixtures, every i7 point estimate is worse, no block interval excludes zero (≥$50k ΔLL +0.0207 [−0.0076, +0.0581]); the brief's phrase "the one clean identity-only comparison" is withdrawn because that control ran the legacy cache without same-day ordering metadata, so identity and ordering are confounded. The adverse direction is retained as evidence, not dismissed as Monte Carlo noise |
| 2.4 | Recorded. `_meta.prior_p*` is summed over the tracker's final-state counts (`scripts/build_stats_cache.py`, "global prior π" block, "Fixed constant — not rolling, not as-of-date"). Measured 2026-09-11 on `data/xgb_data_i7` (train 2005-02-17 → 2024-12-30; corpus → 2026-04-16), π from train only vs whole corpus: wicket 0.054037 / 0.054361 (+0.000324), dot 0.303601 / 0.304017 (+0.000415), single 0.413227 / 0.411313 (−0.001915), two 0.076086 / 0.075796 (−0.000291), four 0.107686 / 0.107803 (+0.000117), six 0.045362 / 0.046711 (+0.001349). Max abs 0.001915. Feature shift = diff × k/(n+k): venue (k 200) 0.001915 at n 0, 0.000957 at n 200, 0.000174 at n 2,000, 0.000038 at n 10,000; player (k 30) 0.001915 / 0.000957 / 0.000174 at n 0 / 30 / 300. Source: `scripts/sequence_track/measure_global_prior_asof.py` (output retained at `retrain_i7/evidence/global_prior_asof_i7.{txt,json}`; its whole-corpus prior equals the cache `_meta` prior to six decimals). This is feature displacement only: every arm sharing the cache has the same exposure, and the downstream effect on each model's log loss is unknown and unmeasured (Astra round-1 MUST-FIX 2 withdrew the earlier "cannot favour an arm" wording). Per-player, per-venue and per-match counts remain as-of-date (invariant 2) |
| 2.5 | `TODO.md` edited 2026-09-11 (+53 / −5): fold statement corrected; framing paragraph added under "Do not un-fold the aliases"; new section "Global outcome prior is not as-of-date (recorded 2026-09-11)" with the table and a backlog checkbox for as-of priors (schema bump, materialization change, production retrain) |

---

## D3. Trainer changes (`scripts/transformer_t1.py`) and retrain runner

| # | Check | Pass condition |
|---|---|---|
| 3.1 | Parquet stem | `load_split` derives the stem from the frame directory's `.feature_hash` `version` field (`cricket_data_{version}_{split}.parquet`), with an explicit `--frame-version` override; `--data-dir data/xgb_data_v3` still resolves to the old files byte-for-byte (existing tests unchanged) |
| 3.2 | Eval kit optional | a `--no-kit` flag (or `--kit-dir none`) skips loading `unseen_pair_masks.npz` and `probe_labels.parquet`; with it, training, validation scoring, checkpoint save and `metrics.json` complete without any read of `models/embeddings/eval_kit/`; `validation_ll_unseen_pairs` / probe fields are absent, not fabricated; the default path (kit present) is unchanged |
| 3.3 | Test-split silence | with `--no-kit` the test split is loaded only if `--score-test` is passed; the retrain does not pass it, so no test log loss exists for the retrained checkpoints at selection time (test rows are never scored before the seed is chosen) |
| 3.4 | Full precision | `metrics.json` stores `validation_ll` unrounded (the pin script's decimal-count check in 6.2 is only a rounded-value rejection heuristic) (repr of the float64 mean); the rounded four-decimal value may be kept alongside under a distinct key; a unit test asserts two runs whose means differ in the sixth decimal store distinct values |
| 3.5 | Training contract | `metrics.json` gains a `training_contract` block: frame dir, the frame's `.feature_hash` JSON verbatim (delivery semantics, alias version and sha256, splits, k_player, k_venue), md5 of each split parquet read, row counts and min/max `match_date` per split, the ordered 50 feature names and their sha256, the state-column normalisers (120 / 200 / 12 / clip 36), `CLASS_MAPPING` verbatim, architecture (dmodel, layers, heads, arm), optimiser settings, seed, best epoch, epochs run, device, torch/numpy/pandas versions, git HEAD short, the stats-cache role given by `--stats-cache-role` resolved through `scripts/artifacts.artifact_path` with its md5 and its `_meta` `venue_alias_version` and `same_day_order_version`, which must equal the frame's declared alias version (else the trainer refuses to start). *Amended after Astra round 1 (SHOULD 2):* a contract is written iff the frame declares `delivery_semantics`; a contracted frame without `--stats-cache-role` refuses to start; an undeclared (v3) frame with a role refuses; an undeclared frame without a role writes **no** contract and prints a warning that the checkpoint serves only through the legacy route. The original wording ("records `null` … refused by D4") is superseded |
| 3.6 | Determinism | seeds set for torch and numpy as before; `torch.use_deterministic_algorithms` not required (MPS), but the contract records device and the fact that MPS runs are not guaranteed bit-reproducible |
| 3.7 | Retrain runner | `scripts/sequence_track/retrain_i7.py` reads `experiments/configs/seq_stage1_retrain_i7_v1.yaml` (arms `mlp`, `full`; seeds 7, 13, 29, 42, 101; dmodel 128, layers 2, heads 4, batch 128, epochs 30, lr 0.0003, patience 3, device mps, data dir `data/xgb_data_i7`, stats-cache role `stats_cache_i7`, `--no-kit`, no `--score-test`, no aux), launches `transformer_t1.py` per (arm, seed) in order, writes `models/embeddings/seq_stage1/retrain_i7/<arm>/seed_<s>/`, refuses to overwrite a completed checkpoint, records wall seconds per run, and writes `summary.yaml` with `splits.validation.arms.<arm>.per_seed[]` rows `{seed, ll}` (full precision), `n_rows`, `n_matches`, the config sha256, and the frame md5s; it never scores the test split; exactly the five seeds, each once |
| 3.8 | Ablation checkpoints untouched | md5 of every file under `models/embeddings/t1_ablation_v1_mps/{mlp,full}/seed_*/` recorded before and after; identical |
| 3.9 | Tests | `scripts/tests/test_transformer_t1_contract.py` (new): stem resolution for v3 and i7, `--no-kit` skips the kit and the test split, full-precision storage, contract completeness on a tiny synthetic frame with a synthetic `.feature_hash`, refusal when the cache's alias version differs from the frame's; `tests/test_transformer_t1_ablations.py` and `tests/test_transformer_t1_load_aux.py` still pass |
| 3.10 | Runs complete | ten runs, exit 0, `metrics.json` + `model.pt` + contract per run; per-seed validation LL table for both arms printed by the runner and pasted here verbatim; wall seconds per run recorded |
| 3.11 | What is not compared | no retrained-checkpoint number is set against the `t1_ablation_v1_mps` table (different frame, different rows); the ablation's MLP−logistic and T1−MLP findings are not restated as describing these checkpoints; the selected C−B validation gap on i7 is recorded as a selection-conditioned diagnostic, not evidence |

### Result (D3)

Recorded 2026-09-11. `scripts/transformer_t1.py` (+331/−30), new
`scripts/sequence_track/retrain_i7.py`, new
`experiments/configs/seq_stage1_retrain_i7_v1.yaml` (sha256
`1be256d2ef61cbc7822529139e91a37e09ff8adb1d8c005e36c625a70f2c6871`), new
`scripts/tests/test_transformer_t1_contract.py`; implemented by an Opus
subagent, reviewed by Fable, who launched the runs.

| # | Result |
|---|---|
| 3.1 | PASS — stem from `.feature_hash` `version` (`resolve_frame_version`), `--frame-version` override; an undeclared frame refuses instead of falling back to v3; `data/xgb_data_v3` resolves to the same files |
| 3.2 | PASS — `--no-kit` (or `--kit-dir none`): no read of the kit; unseen-pair / probe fields absent; default path unchanged |
| 3.3 | PASS — with `--no-kit` the test parquet is not even loaded unless `--score-test`; the retrain config does not pass it; every retrained `metrics.json` records `test_split_scored: false` |
| 3.4 | PASS — `validation_ll` stored unrounded, `validation_ll_rounded4` alongside (retained one-epoch runner smoke `retrain_i7_smoke/mlp/seed_7/metrics.json`: `1.4502653968864392` / `1.4503`). Additional fix beyond the letter: `eval_split` accumulated the mean in float32 (≈1e-6 summation noise over 124k rows, the decimal the selection reads); now float64. The implementer reported a 1.07e-8 movement on the smoke; that comparison run was not retained, so it is reported, not evidenced. Because the float64 loss also feeds early stopping, v3-frame training after this change is not byte-identical to earlier runs (feature arithmetic unchanged; loss reduction changed); historical recorded values were rounded to 4 dp and do not move, but unchanged selection on a rerun is not proven |
| 3.5 | PASS (re-verified after the round-1 amendment: contract iff the frame declares semantics; three refusal/omission cases tested) — `training_contract` (`seq_stage1_t1_training_contract_v1`) with every listed field; `--stats-cache-role stats_cache_i7` resolved through `artifact_path`, cache md5 `671ac8200b275fa3d11d848e609f5132`, `_meta` alias `venue_aliases_v1` = frame's, ordering `date_then_match_id_lexicographic_v1`; mismatch refuses before the splits load (tested with a synthetic cache via `--stats-cache-path`) |
| 3.6 | PASS — `device: mps`, `mps_bit_reproducible: false` recorded |
| 3.7 | PASS (after round-1 MUST-FIX 3 and round-2 MUST-FIX 1: reuse of a completed run now also compares architecture, optimiser incl. epochs, seed, cache md5, the `run_record.json` overrides, the frame's complete `.feature_hash` key-by-key and the live train/validation parquet md5, row count and date range (measured once in preflight, 0.9 s) against the effective config; the summary's frame block is built from the live measurement and re-asserted against every reused checkpoint; same-path parquet byte replacement, changed declaration key, changed row count each refused by test; a final no-arg rerun over the ten runs changed `summary.yaml` only by `generated_at`; pre-pass over every completed run before the first launch, mismatch refuses and names the field; the real one-epoch smoke run is refused by a default resume, exit 2; a no-arg rerun over the ten runs of record skipped all ten, touched no checkpoint, and changed `summary.yaml` only by `generated_at` and an `overrides: {}` per seed) — runner validates the YAML (arms, exactly the five seeds once each, `no_kit` true, `score_test` false, `aux` false), preflights frame and cache identity, refuses to overwrite a completed run without `--force`, writes `run_record.json` per run and `summary.yaml` (`complete: true`, validation `n_rows` 124292, `n_matches` 545); `--dry-run` printed ten command lines, every one with `--no-kit`, none with `--score-test` or `--aux` |
| 3.8 | **PARTIALLY MET, recorded honestly** — the check asked for a before/after hash of every file in the ten ablation directories. The before-manifest Fable wrote (41 files) was overwritten during the session by the implementer's own ten-line `model.pt`-only manifest at the same scratch path, so a complete before-list no longer exists. What is evidenced: (a) all ten `model.pt` md5s identical before and after (`retrain_i7/evidence/ablation_model_pt_md5_before.txt` vs `ablation_all_files_md5_after.txt`: `5cfb36ab…`, `3fc373b2…`, `57fa3f65…`, `e873e6f2…`, `ab8f80c6…`, `504c3b64…`, `c25b9958…`, `933ba4f4…`, `ed345631…`, `36178a34…`); (b) `superseded_checkpoints` in the config pins `md5_directory` of both seed-101 dirs, so any later change there fails `--verify`; (c) every file under `t1_ablation_v1_mps/` and `eval_kit/` carries a modification time of 2026-08-08 or 2026-08-12 (`evidence/ablation_evalkit_mtimes.txt`, 93 files), supporting but not proving non-modification. The `metrics.json` / `predictions_*.npz` files have no retained pre-session hash |
| 3.9 | PASS — historical count as reported by the implementer at the time (output not retained): `40 passed, 7 warnings in 1.82s` for the three files; the retained, traceable evidence is the final full-suite run in `evidence/pytest_full_round2.txt` (992 passed), which includes these files |
| 3.10 | PASS — ten runs, exit 0 (`retrain_i7/run.log`, 08:4x–08:57 IST). Per-seed validation LL verbatim from `summary.yaml` (seed / ll / best_epoch / wall s / checkpoint md5): **mlp** 7 1.4381307431731392 / 15 / 53.5 / `72c565d6…`; 13 1.4380360131566159 / 16 / 68.1 / `b371b109…`; 29 1.438049555829733 / 13 / 92.7 / `8ad3b5b4…`; 42 1.438596617959276 / 10 / 69.7 / `31451aef…`; 101 1.4377511308142412 / 16 / 87.2 / `dd240cca…`. **full** 7 1.4369648845208556 / 23 / 174.5 / `ba586ca0…`; 13 1.4383511253032655 / 16 / 121.8 / `ebf559b2…`; 29 1.4374436296092699 / 19 / 167.2 / `cf650b70…`; 42 1.4372224749152225 / 28 / 225.9 / `b0a07f8d…`; 101 1.4366172113916893 / 24 / 211.6 / `aa79292c…` |
| 3.11 | Observed — no comparison to the `t1_ablation_v1_mps` table is made or implied. Selected C−B validation LL on i7 = 1.4366172113916893 − 1.4377511308142412 = −0.0011339194225519 (the config's printout rounds this to −0.001134), recorded as `selection_conditioned_diagnostic_not_evidence`. A one-epoch runner smoke under `retrain_i7_smoke/` (gitignored) is retained; it is not an arm |

---

## D4. Serving guard (`scripts/sim_t1.py`)

| # | Check | Pass condition |
|---|---|---|
| 4.1 | Contract path | if the checkpoint's `metrics.json` carries `training_contract`, the wrapper requires `delivery_semantics == "inclusive_total_runs_v1"`, a non-null `venue_alias_version`, and a non-null stats-cache md5; any missing or mismatched field refuses with a message naming the field |
| 4.2 | Cache identity at serve time | the wrapper obtains the serving cache's `_meta` through the provider it was given (`SameDayReplayStatsProvider` → `StatsProvider` → backend `get_meta()`; a provider that cannot expose `_meta` is refused) and requires `_meta.venue_alias_version == training_contract.venue_alias_version` and `_meta.same_day_order_version` present; mismatch refuses |
| 4.3 | Legacy path preserved | a checkpoint **without** `training_contract` keeps the existing rule (`config.data_dir` must contain `xgb_data_v3`) **and** additionally requires the serving cache's `_meta` to carry no `venue_alias_version`, so an uncontracted v3 checkpoint served from the i7 cache is refused. The stage 0 smoke configuration (v3 checkpoint + v3 cache) still serves |
| 4.4 | Feature list | the contract's 50 feature names must equal the wrapper's `N_FEATS` construction order (EB_BAT + EB_BOWL + VENUE + CTX + STATE), else refuse |
| 4.5 | Tests | `tests/test_sim_t1_contract_guard.py` (new), using a synthetic checkpoint dir and a fake provider exposing `_meta`: (i) i7 contract + i7 meta → loads; (ii) uncontracted v3 + meta without alias → loads; (iii) uncontracted v3 + i7 meta → refused; (iv) i7 contract + meta without alias → refused; (v) contract with `delivery_semantics: legal_off_bat_v1` → refused; (vi) contract missing `venue_alias_version` → refused; (vii) contract with a permuted feature list → refused; (viii) provider without `_meta` → refused |
| 4.6 | Existing tests | `tests/test_run_sim_eval_t1_lifecycle.py`, `test_sim_t1_snapshot_guard.py`, `test_sim_t1_prefix_cache.py`, `test_sim_t1_parity.py` pass unchanged |
| 4.7 | No other behaviour change | `git diff scripts/sim_t1.py` touches only the guard block and the meta lookup helper; forward pass, prefix cache and outcome-dist code untouched |

### Result (D4)

Recorded 2026-09-11. `scripts/sim_t1.py` (+171/−13, two hunks),
`scripts/stats_provider.py` (+12, `get_cache_meta` passthrough),
`scripts/sim_eval/same_day_stats.py` (+9, `_TrackerStatsView.get_cache_meta`,
readable before `begin_date`), new `tests/test_sim_t1_contract_guard.py`;
Opus subagent, reviewed by Fable.

| # | Result |
|---|---|
| 4.1 | PASS — `_verify_training_contract`: semantics must equal `inclusive_total_runs_v1`, alias version non-null, `stats_cache.md5` non-null; each refusal names the field and both values |
| 4.2 | PASS — `_meta` reached via `SameDayReplayStatsProvider.__getattr__ → _TrackerStatsView.get_cache_meta → StatsProvider.get_cache_meta → _SQLiteBackend.get_meta()`; alias equality and `same_day_order_version` presence required; a provider without `get_cache_meta` (or a non-mapping return) refuses on both paths. Live check: i7 cache exposes alias `venue_aliases_v1` + ordering; v3 cache exposes neither |
| 4.3 | PASS — uncontracted checkpoint keeps the `xgb_data_v3` substring rule and additionally requires no `venue_alias_version` in the serving `_meta`. Real-chain check on copied ablation weights: v3+v3 LOADED; v3+i7 REFUSED; i7-contract+v3 REFUSED; i7-contract+i7 LOADED |
| 4.4 | PASS — `feature_names` must equal `EXPECTED_FEATURE_NAMES` (EB_BAT+EB_BOWL+VENUE+CTX+STATE, import-time `== N_FEATS` assert); message names the first differing position |
| 4.5 | PASS — all eight cases plus a `stats_cache.md5`-null case, real tiny `T1Model` state dicts through the production load path; the expected order is rebuilt in the test, not imported from `sim_t1` |
| 4.6 | PASS — historical count as reported by the implementer (output not retained): `33 passed, 14 warnings in 0.69s` for the five files; no existing test file touched; traceable evidence is the retained final full-suite run (`evidence/pytest_full_round2.txt`) |
| 4.7 | PASS — diff is the new guard block (two `# noqa: E402` local imports kept inside it) and a one-line call replacing the old inline check; forward pass, prefix cache and outcome-dist code untouched |

---

## D5. Retrain runs

Covered by D3 checks 3.7–3.11; the run record lands in D3's Result block.
Order of operations: D3 code + tests → D4 code + tests → D3 runs → D6.

---

## D6. Pin rewiring (`scripts/sequence_track/pin_stage1.py`, `run_arm.py`) and re-pin

| # | Check | Pass condition |
|---|---|---|
| 6.1 | Checkpoint source | B and C resolve from `SEQ_STAGE1_ROOT / "retrain_i7" / <arm> / seed_<s>` (composed, no literal); `ABLATION_DIR` / `ABLATION_SUMMARY` are removed from the stage 1 path (they may remain only in a `superseded_checkpoints` record naming the old dirs, their hashes and the reason) |
| 6.2 | Selection | `selection_table` reads `retrain_i7/summary.yaml`; per-seed rows must be exactly the five registered seeds, finite, full precision; the chosen seed, its LL, the rule and the summary sha256 are recorded; a synthetic-summary test covers ties to the lowest seed, a missing seed, a duplicate seed, a non-finite LL, and a rounded (four-decimal) LL being refused |
| 6.3 | Arm blocks | B and C `stats_version: i7`, `stats_cache_role: stats_cache_i7`, cache md5 equal to A's; `training_frame` recorded as `data/xgb_data_i7` with the frame md5s from the checkpoint contract; the checkpoint contract's alias version and delivery semantics pinned and verified by `--verify`; `run_arm.py` `ARM_SPEC` B/C `stats_version` changed to `i7` in lockstep (the runner refuses drift between its spec and the config) |
| 6.4 | Asymmetries | `known_asymmetries` no longer lists `stats_cache_i7_vs_v3` or `training_frame_i7_vs_v3`; a `removed_asymmetries` block records both ids, the date, and "removed by the 2026-09-11 retrain on the i7 frame"; the two remaining entries (`arm_a_path_is_the_replay_lifecycle`, `br2_g1_is_not_a_stage_1_number`) unchanged |
| 6.5 | Feature-count asymmetry stated | a `known_asymmetries` entry `feature_set_114_vs_50` is added: A carries 114 features, A50/B/C carry 50; B−A and C−A are system contrasts; A50−A isolates the feature set at equal information within one model family; C−B is the registered full-sequence-versus-token-MLP contrast |
| 6.6 | Limitation recorded | a `known_limitations` block carries `global_prior_not_as_of` with the D2.4 numbers and "identical across arms" |
| 6.7 | Teacher-forced diagnostic | the config records `selected_validation_ll` for B and C and their difference `C_minus_B_validation_ll` labelled `selection_conditioned_diagnostic_not_evidence` |
| 6.8 | Source closure | `provenance.source_md5` re-pinned for the edited `transformer_t1.py`, `sim_t1.py`, `pin_stage1.py`, `run_arm.py` and the new `retrain_i7.py`; `--verify` fails on the pre-edit config (proof the closure moved) and passes after `--write` |
| 6.9 | Tests | `scripts/tests/test_pin_stage1.py`, `test_run_arm.py`, `test_manifest_defaults.py`, `test_audit_cross_arm.py` pass after the change; new tests for 6.2 |
| 6.10 | Verify | `pin_stage1.py --write` then `--verify`: `OK`; the full-run `<chosen_n_sims>` token still present (untouched until 1d) |

### Result (D6)

Recorded 2026-09-11. `scripts/sequence_track/pin_stage1.py` (+677),
`run_arm.py` (+16), tests updated; Opus subagent (code, synthetic tests),
Fable ran `--write` / `--verify` on the real tree after the retrain.

| # | Result |
|---|---|
| 6.1 | PASS (after round-1 MUST-FIX 1 the pin RECOMPUTES every training fact from disk: md5, row count and `match_date` range of the train and validation parquets via pyarrow, the whole `.feature_hash` key-by-key, the 50 feature names position-by-position and their sha256 from the imported column lists, the retrain config sha256 from disk, `stats_cache` role/path/md5 against the live cache, validation `n_matches` from `innings_id`, and all ten `per_seed.ll` against each seed's own `metrics.json`; 20 drift tests; `--verify` wall 4.1 s warm, so the earlier 830 MB concern is moot; negative control: a tampered train md5 in a YAML copy → `CHANGED arms.C.training_frame.split_md5s.train`, exit 1) — `RETRAIN_DIR = SEQ_STAGE1_ROOT/"retrain_i7"`; `ABLATION_DIR`/`ABLATION_SUMMARY` removed (test asserts `not hasattr`); `superseded_checkpoints` records `t1_ablation_v1_mps/mlp/seed_101` (`md5_directory` `a7047d948f8a4de7e53345afedd158a6`) and `full/seed_101` (`2675b034e2e1b9a62d3e847e2fca365c`) with the reason |
| 6.2 | PASS — `selection_table` reads `retrain_i7/summary.yaml` (absent → clean PinError naming the runner); exactly seeds 7/13/29/42/101 once each, finite; a rounded-value rejection heuristic (`FULL_PRECISION_RULE`: refused when `ll == round(ll, 4)` and repr has < 6 decimals — it rejects values indistinguishable from a four-decimal rounding; it cannot prove a value was stored at full precision, which is instead established by the trainer's unrounded writer, 3.4, and by 6.1's comparison of every summary ll to its `metrics.json`); tests for tie → lowest seed, missing, duplicate, non-finite, rounded |
| 6.3 | PASS — B/C `stats_version: i7`, role `stats_cache_i7`, md5 `671ac820…` = A's; `training_frame` block pins frame dir (role `ball_frame_i7`), split md5s/rows/date ranges, semantics, alias version + sha256, 50 names sha256 `14f40e4a…`, architecture, seed, best epoch, contract version, training-time cache identity; `--verify` compares the contract's cache md5 to the live cache and its seed to the selected seed; `run_arm.py` `ARM_SPEC` B/C → `i7` with drift test |
| 6.4 | PASS — `known_asymmetries` drops the two cache/frame entries; `removed_asymmetries` records both ids, `removed_on: 2026-09-11`, reason; the two remaining entries verbatim |
| 6.5 | PASS — `feature_set_114_vs_50` added |
| 6.6 | PASS — `known_limitations.global_prior_not_as_of` with the D2.4 table and `identical_across_arms: true` |
| 6.7 | PASS — `checkpoint_selection.teacher_forced_diagnostic`: B 1.4377511308142412, C 1.4366172113916893, `C_minus_B_validation_ll` −0.001134, label `selection_conditioned_diagnostic_not_evidence` |
| 6.8 | PASS — before the edits `--verify` on the committed YAML: `pin_stage1: OK`; after the edits and before the retrain: `pin_stage1: ERROR: missing retrain summary …` (exit 1, no traceback); source closure now includes `retrain_i7.py` and the edited `transformer_t1.py`, `sim_t1.py`, `stats_provider.py`, `same_day_stats.py`, `run_arm.py`, `pin_stage1.py` |
| 6.9 | PASS — final full suite after the round-2 fixes and the 10:24 re-pin (`evidence/pytest_full_round3.txt`): `1002 passed, 5 skipped, 28 warnings in 107.12s` and artifact-free `967 passed, 40 deselected`. Earlier: `scripts/tests/test_pin_stage1.py test_run_arm.py test_manifest_defaults.py test_audit_cross_arm.py`: `287 passed, 8 skipped` before the retrain and full suite `947 passed, 5 skipped, 22 warnings in 105.92s` after the first re-pin — both historical, reported at the time, outputs not retained (the 8 skips carry a `needs_repin` marker that skips only while the retrain summary is absent); after the round-1 fixes and final re-pin (`evidence/pytest_full_round2.txt`): `992 passed, 5 skipped, 28 warnings in 108.08s` and artifact-free `957 passed, 40 deselected` |
| 6.10 | PASS — 2026-09-11 ≈09:00 IST (first re-pin) and ≈10:02 IST (final re-pin after the round-1 fixes; the smoke in D7 was re-run against this one): `--write` → selection printout (B `mlp/seed_101` chosen at 1.437751; C `full/seed_101` at 1.436617; per-seed table matches D3.10 to six decimals) then `--verify` → `pin_stage1: OK — … matches every recomputed fact`; `<chosen_n_sims>` token untouched |

---

## D7. Re-smoke (1a) and cross-arm audit

| # | Check | Pass condition |
|---|---|---|
| 7.1 | Commands | the four smoke commands are executed verbatim from the re-pinned YAML's `smoke_1a.command` strings, all with `--config`; every arm `config_verified: true` |
| 7.2 | Every arm loads | A, A50, B, C exit 0 on the same four smoke fixtures at 10 sims; B and C banners name the `retrain_i7/<arm>/seed_<s>` dirs; the D4 guard passes on both (contract path); wall time per arm recorded |
| 7.3 | Audit | `audit_cross_arm.py` passes on the four dirs; the stats cache md5 is now **identical** across all four arms (listed field, asserted here by hand from the audit output), model dir hashes differ as expected |
| 7.4 | Scorers | `score_realism.py` and `score_props.py` exit 0 on every arm; numbers not read |
| 7.5 | Odds resolution | not repeated (unchanged engine and odds path since stage 0 D7.7); recorded as inherited |
| 7.6 | No numbers | no log loss, Brier, ROI or edge from the smoke appears in this file, the report to the user, or any commit message |
| 7.7 | Stop | nothing beyond the smoke runs before the user's explicit go for 1b |

### Result (D7)

Recorded 2026-09-11 09:03–09:05 IST (first run, now under
`smoke/round1_archive/`), re-run 10:04–10:05 IST after the round-1 fixes
(now under `smoke/round2_archive/`), and **re-run 10:25–10:26 IST against the
final re-pinned config** after the round-2 fixes; the rows below describe
the 10:25 run (`evidence/smoke_round3_*`), with earlier runs' values noted
where they differ. Stage 0 smoke outputs moved to `smoke/stage0_archive/`;
fixtures dir unchanged (md5 `703f522d…`, 4 files).
No log loss, Brier, ROI or edge value was read: runner output went to
`run.log` files and only banner, exit, timing and provenance lines were
grepped.

| # | Result |
|---|---|
| 7.1 | PASS — the four `arms.<arm>.smoke_1a.command` strings executed verbatim from the final re-pinned YAML (`evidence/smoke_round3_commands_executed.sh`; config sha256 `f3ecd29b…` recorded identically by all four arms; the 10:04 run recorded `a156a2f2…` and the 09:03 run `19e0f4f5…`, both pre-fix configs); `config_verified: true` on every arm |
| 7.2 | PASS — exit 0 ×4; banners: A "114 features", A50 "50 features", B `retrain_i7/mlp/seed_101` (272,006 params, `delivery_semantics=inclusive_total_runs_v1`), C `retrain_i7/full/seed_101` (298,758 params, context_capacity 200, prefix cache off); B18 graft, run-out channel, B10 selector and causal roster policy ACTIVE on all four; outer wall per arm from the shell timestamps around each command (`evidence/smoke_round3_shell.txt`; 10:25:04→:14, :14→:24, :24→:33, :33→:44) A 10 s, A50 10 s, B 9 s, C 11 s (10:04 run identical), which includes `uv` start-up and the pin-verification preflight (now ~4 s because it hashes the two training parquets); first run 8/10/8/10 s; the runner's own `Total simulation time` lines in each `run.log` are read only as banners, not as results |
| 7.3 | PASS (all three runs; excerpt of the 10:26 audit in `evidence/smoke_round3_audit_excerpt.txt`) — `CROSS-ARM AUDIT PASSED: 4 arms (A, A50, B, C), 4 fixtures, schema complete, checks 6.2-6.5 identical.` Listed fields: `stats_cache_md5` **`671ac8200b275fa3d11d848e609f5132` on all four arms** (the stage 0 audit listed i7 for A/A50 and `3faf24a6…` v3 for B/C); model dir hashes A `a6ac7671…`, A50 `1ae7bbab…`, B `4b5e28aa…`, C `e8195af8…`; checkpoint md5s `7ee1e180…` / `7705b330…` / `dd240cca…` / `aa79292c…` (the B/C values equal the D3.10 seed-101 md5s) |
| 7.4 | PASS (10:26 run: exit codes retained per arm in `evidence/smoke_round3_postprocess_exits.txt`, `reslice=0 realism=0 props=0` ×4, likewise for the 10:05 run; the 09:03 run's codes were read from the shell but only its output inventory was retained) — `reslice_eval_json.py` (odds `betting_odds_polymarket_v2.json`, cluster source `data/polymarket_test_v2`) wrote `_all`, `_min_volume_50000`, `_min_volume_100000` per arm; `score_realism.py` "fixtures scored: 4 of 4" ×4; `score_props.py` "families scored (15)" ×4; all exit 0 |
| 7.5 | Inherited from stage 0 D7.7 (engine `d7e7a70d…` and odds path unchanged) |
| 7.6 | PASS — no smoke metric appears in this file or in the report to the user |
| 7.7 | PASS — nothing beyond the smoke has run; 1b awaits the user's go |

---

## D8. Astra review and commit (retrain and re-pin)

| # | Check | Pass condition |
|---|---|---|
| 8.1 | Rounds | `codex exec -m gpt-6-astra -c model_reasoning_effort="low" -s read-only` over D2–D7 and every changed file; each round's output path and verdict recorded; iterate until SIGN-OFF with no MUST-FIX |
| 8.2 | Commit | one commit of the stage 1 retrain/re-pin files plus `TODO.md` and this file; plain summary + body per `CLAUDE.md`, no AI-attribution trailer; hash recorded here |

### Result (D8)

Astra rounds (`codex exec -m gpt-6-astra -c model_reasoning_effort="low" -s read-only`; prompts and outputs in the session scratchpad, verdict lines quoted):

| round | output file | verdict | disposition |
|---|---|---|---|
| 1 | `<scratchpad>/astra_stage1_d8_1.md` (2026-09-11 ≈09:10 IST) | **NO SIGN-OFF** — 4 MUST-FIX, 4 SHOULD, 3 NOTE | MUST-FIX 1 (pin copies frame facts from the contract instead of recomputing) → recompute layer + 20 drift tests (6.1, Opus); MUST-FIX 2 ("cannot favour an arm" / 0.007 comparison unsupported; 10k-balls typo; no reproducible source) → wording withdrawn in the config, TODO.md and this file (2.4), `scripts/sequence_track/measure_global_prior_asof.py` written and its output retained under `retrain_i7/evidence/` (Fable, inline); MUST-FIX 3 (runner reuse ignores architecture/optimiser/epochs/overrides) → effective-config comparison, pre-pass, override provenance per seed, one-epoch-then-default-resume test (3.7, Opus); MUST-FIX 4 (3.8 narrower than its check; unsourced timings and test claims) → 3.8 rewritten as PARTIALLY MET with what is and is not evidenced, timing sources and retained evidence files named in 3.4, 3.11, 7.2, 7.4 (Fable, inline). SHOULD 1 (guard boundary wording + same-alias/different-cache and ordering tests) → done; SHOULD 2 (v3-frame checkpoints got a null contract and were then refused) → contract iff the frame declares semantics, three cases tested, 3.5 amended; SHOULD 3 (test the trainer's own writer; "isolates sequence memory" overstated) → done in tests, config, and this file; SHOULD 4 (stale audit docstring) → done. NOTE 2 (unrelated untracked files) → the commit is scoped to the reviewed files only. Follow-through: re-pin 10:02 IST (`--verify` OK), smoke re-run 10:04 IST against it, full suite 992 passed |
| 2 | `<scratchpad>/astra_stage1_d8_2.md` (≈10:12 IST) | **NO SIGN-OFF** — 1 MUST-FIX, 2 SHOULD, 3 NOTE; round-1 items 1, 2, 4 and SHOULD 1–4 confirmed closed with file:line; MUST-FIX 3 partially closed | MUST-FIX 1 (runner reuse compared frame path and version only, not the parquet identity or the full declaration) → live frame measured once in preflight, key-by-key and split-by-split comparison before reuse, summary frame block from the live measurement, four drift tests (3.7, Opus); SHOULD 1 (historical test counts unsourced) → 3.9, 4.6, 6.9 relabelled as reported-at-the-time with the retained final suite as the traceable evidence (Fable, inline); SHOULD 2 ("full precision" shorthand stronger than the heuristic proves) → pin-script prose and 3.4/6.2 reworded to "rounded-value rejection heuristic" (Fable, inline). Astra noted it could not execute `uv` inside its sandbox (cache dir permission), so its verification is code inspection plus retained evidence. Follow-through: final re-pin 10:24 IST (`--verify` OK), smoke re-run 10:25 IST against it (config sha256 `f3ecd29b…`, `evidence/smoke_round3_*`), audit passed, full suite in `evidence/pytest_full_round3.txt` |
| 3 | `<scratchpad>/astra_stage1_d8_3.md` (≈10:35 IST) | **SIGN-OFF** — MUST-FIX none, SHOULD none; round-2 MUST-FIX 1 closed (`retrain_i7.py:202`, `:334`, `:713`, `:532`, `:764`; tests `:716–875`), SHOULD 1–2 closed; Astra independently confirmed all 35 pinned source md5s, the ten checkpoint declarations, the live train/validation parquet md5s, config sha256 `f3ecd29b99cc6793ddd314fd5afe621ed0947b239c2145e40a29a699b6864947`, and that all four smoke provenance files bind to it; notes: tests and `--verify` not independently rerun by the reviewer (sandbox cannot run `uv`); 3.8 stays partially met; 1b unrun and outside this sign-off | closed |

8.2: committed as `504c17f` on `embeddings-ladder` (18 files, the reviewed set only; plain summary + body, no attribution trailer, per `CLAUDE.md`). `pin_stage1.py --verify` OK at the committed state. This D8 note is the only change after the commit and is folded into the next commit.

---

## D9. 1b timing and convergence (user go: 2026-09-11, "okay sounds good let's run it")

Written before the shard was simulated. Nothing in this deliverable reads
or reports a log loss, Brier, ROI or edge from the shard: the convergence
statistic is a *spread* (range and SD across batches of a paired contrast),
never a level.

| # | Check | Pass condition |
|---|---|---|
| 9.1 | Shard rule | ten fixtures from `data/polymarket_test_v2` by a stated deterministic rule biased to long innings (largest total delivery count, male T20, recorded winner, odds row, ties to the lower cricsheet id), copied to `models/embeddings/seq_stage1/timing/fixtures/`, rule and table recorded in `timing/SHARD_RULE.md` |
| 9.2 | Candidate 50 | `CONVERGENCE_CANDIDATES` gains 50; `timing_1b.n_sims` permits it; the joint overlap screen re-run over the three batch seeds includes 50 and still caps at 3,200 (joint min gap 3,290); the protocol text labels 50 a noise-curve point only (plug-in bias ≈0.01 at 50 sims), never a full-run candidate |
| 9.3 | Re-pin | `pin_stage1.py --write` then `--verify` OK with the shard present (`fixture_count` 10, `fixture_dir_md5` recorded); `stop_rule_reading` recorded as `to_be_decided_by_the_user_before_1d` |
| 9.4 | Driver | `scripts/sequence_track/run_timing_1b.py` renders every command from the registered `timing_1b.command_template` per arm, refuses any unpermitted `n_sims`, runs the four arms concurrently per (n_sims, batch seed) so memory pressure is real, skips complete runs, honours a STOP file, records wall seconds / exit code / parsed simulation time / peak RSS per run in `timing/timing_record.json`, and never parses a metric line |
| 9.5 | Runs | 5 candidates × 3 batch seeds × 4 arms = 60 runs, all exit 0, every arm's `config_verified: true`; the cross-arm audit passes on every (n_sims, seed) group with `--expected-fixtures` the shard dir |
| 9.6 | Variability rerun | the brief's "second base seed rerun" is satisfied by the batch-seed design: batch seeds 20260910 and 20260911 are the same shard under two seeds at every count; the per-fixture paired-difference SD between them is reported per contrast and count (no extra runs) |
| 9.7 | Convergence table | `scripts/sequence_track/convergence_1b.py` writes `timing/convergence_1b.{md,json}` with, per (n_sims, contrast ∈ {C−B, B−A, C−A, A50−A}): the range (max−min) and SD of the paired shard-mean delta across the three batches, the below-0.002 flag on the raw range, and the full-set-equivalent range × sqrt(10/255); plus the noise-curve fit a·n^(−1/2) with the crossing points; the code asserts no output field carries a log-loss level, and a test greps the outputs for leaked levels |
| 9.8 | Timing table | per arm and count: wall seconds per match, seconds per simulation, extrapolation to 255 fixtures (serial per arm, and concurrent wall = max over arms), peak RSS; brought to the user with the convergence table so they choose the 1d count and the stop-rule reading |
| 9.9 | Numbers not read | no log loss / Brier / ROI / edge from the shard appears in this file, the report, the convergence outputs, or any commit message |
| 9.10 | Stop | 1c and 1d do not start before the user chooses the count and the stop-rule reading, both recorded in the config by `pin_stage1.py` before 1d |

### Result (D9)

Recorded 2026-09-11 12:30 IST from `models/embeddings/seq_stage1/timing/`
(`SHARD_RULE.md`, `timing_record.json`, `driver.log`, `driver_attempt1.log`,
`convergence_1b.{md,json}`). No log loss, Brier, ROI or edge from the shard
was read; the convergence script asserts that no output field carries a
level, and the report was read only after that assertion.

| # | Result |
|---|---|
| 9.1 | PASS — rule and table in `timing/SHARD_RULE.md`: the ten largest fixtures by total delivery count among male T20, winner-recorded, odds-carrying fixtures (257–264 deliveries; iteration median 240 under this count; 252 of 255 eligible), ids 1493243, 1529380, 1527693, 1529379, 1512727, 1494272, 1501330, 1525160, 1512766, 1528306; 1525160 and 1512766 share 2026-02-26 (a same-day pair) |
| 9.2 | PASS — `CONVERGENCE_CANDIDATES` = [50, 100, 200, 400, 800, 1600, 3200, 6400], `TIMING_1B_CANDIDATES` = [50, 100, 200, 400, 800]; `timing_1b.n_sims` = [50, 100, 200, 400, 800, 1600, 3200]; joint screen rows: 50 disjoint, joint min gap 3,290, largest permitted 3,200, 6400 `not_permitted_without_new_seeds`; protocol text labels 50 a noise-curve point only |
| 9.3 | PASS — `--write` / `--verify` OK with `fixture_count` 10, `fixture_dir_md5` `857d03aa63c3f0114d8983a1e6f5a35e`, `fixture_dir_status: present`; `stop_rule_reading: to_be_decided_by_the_user_before_1d`; `test_pin_stage1.py` updated for the present shard (177 passed) |
| 9.4 | PASS — `scripts/sequence_track/run_timing_1b.py` (+ `scripts/tests/test_timing_1b.py`, 38 passed): commands rendered from `timing_1b.command_template` with only the four tokens substituted, unpermitted `n_sims` and unregistered base seeds refused, four arms concurrent per group, skip-if-complete, STOP file honoured between groups (exercised for real: `driver_attempt1.log` stopped before `n50/seed20260912` with exit 3 when the driver was re-launched under `nohup` to survive the session; the two completed groups were skipped on relaunch), per-run wall / exit / `Total simulation time` / process-tree peak RSS in `timing_record.json`; only the timing line is parsed from the logs |
| 9.5 | PASS — 60 runs recorded, every exit code 0, `[audit] pass` on all 15 (n_sims, seed) groups (`grep -c` on `driver.log` = 15 after the two groups of attempt 1, which also passed); wall 10:55–12:26 IST across the two attempts |
| 9.6 | PASS — per-fixture paired-difference SD between batch seeds 20260910 and 20260911, per contrast and count, in the "Variability rerun" table of `convergence_1b.md`; no extra runs |
| 9.7 | PASS — `scripts/sequence_track/convergence_1b.py` wrote `convergence_1b.{md,json}`; slice used: **all** (the shard has no sliced files; the ≥$50k semantics is not applied at 1b — recorded in the report header); `range_95` = max−min of the three batches (with n = 3 the empirical range is the 95% range, no wider estimator); spread rows for C−B, B−A, C−A, A50−A at 50/100/200/400/800; noise-curve fit a·n^(−1/2) with crossing points; leak assertion passed |
| 9.8 | PASS — timing table per arm and count and the concurrent 255-fixture projection are in the report; brought to the user 2026-09-11 12:30 IST with the spread table |
| 9.9 | PASS — no level appears in this file, the report or the outputs |
| 9.10 | CLOSED 2026-09-11 ≈13:10 IST — user decisions: **n_sims = 1,600** for 1d; **stop-rule reading** = the full-set-equivalent Monte Carlo SD of the paired primary contrasts (two-seed paired difference ÷ sqrt(10) × sqrt(10/255)) at or below about 0.002 at the chosen count, recorded in `convergence_protocol.stop_rule_reading` as a change of statistic from the registered range rule, which was not met at any permitted count (fit crossing ≈5,264 for C−B, above the 3,200 seed cap); `CHOSEN_N_SIMS = 1600`, every full-run command now carries `--n-sims 1600`; the spread table (spreads only) is pinned into `convergence_protocol.spread_table` from `convergence_1b.json` and recomputed by `--verify` |

**Addendum — thread cap and sharding probe (2026-09-11 12:32–13:05 IST,
timing only, no metric read).** Twelve concurrent full-T1 processes at the
registered 4 threads ran 5.45× slower per process than the 1b rate
(0.6712 vs 0.1232 s/sim). Twelve at `--threads 1` were no better (0.6503
s/sim, aggregate 2.27× one process); 4 concurrent one-thread processes gave
2.05× aggregate, 8 gave 1.62×. Cause: `scripts/sim_t1.py` hard-coded
`torch.set_num_threads(4)` on CPU after the runner's cap, so every
"one-thread" T1 process ran four torch threads (`ps`: ≈350–400 % CPU each,
38 % system time; `torch.get_num_threads()` reports 1 under the env cap
outside the wrapper). Threads buy nothing per process (solo 4-thread 0.0888
s/sim vs the 4-thread smoke rate 0.0875). The XGBoost arms honour the cap:
four concurrent one-thread arm-A processes ran at 0.032 s/sim each with no
contention. Machine: Apple M5 Pro, 10 performance + 5 efficiency cores, 48 GB.
**User decision:** fix the wrapper to honour `OMP_NUM_THREADS` (default 4
when uncapped, tests in `tests/test_sim_t1_contract_guard.py`), register
`threads: 1` for every arm (`pin_stage1.THREADS`, `run_arm.py` default and
cap), re-pin, re-smoke, and confirm scaling with a ten-process probe before
1c. Floating-point reductions can differ across torch thread counts, so no
1d number is expected to be bit-identical to a four-thread run; nothing
scored has run at four threads except the timing shard, which is timing
only. Probe outputs under `timing/probe*/` are unregistered and are not
results.

**Headline numbers from `convergence_1b.md`, verbatim (spreads and timings
only).** Raw 95% range of the paired shard-mean contrast across the three
batches, and the full-set equivalent (× sqrt(10/255) = 0.1980):

| n_sims | C−B range / scaled | B−A range / scaled | C−A range / scaled | A50−A range / scaled |
|---|---|---|---|---|
| 50 | 0.101046 / 0.02001 | 0.043837 / 0.008681 | 0.057209 / 0.011329 | 0.100475 / 0.019897 |
| 100 | 0.070026 / 0.013867 | 0.105539 / 0.0209 | 0.051868 / 0.010271 | 0.058657 / 0.011616 |
| 200 | 0.044304 / 0.008773 | 0.055086 / 0.010909 | 0.016345 / 0.003237 | 0.039585 / 0.007839 |
| 400 | 0.05468 / 0.010828 | 0.081055 / 0.016051 | 0.03966 / 0.007854 | 0.064831 / 0.012839 |
| 800 | 0.034961 / 0.006923 | 0.043664 / 0.008647 | 0.008703 / 0.001724 | 0.027953 / 0.005536 |

No cell meets the literal rule (raw range < 0.002); on the scaled reading
only C−A at 800 does (0.001724). Variability-rerun scaled shard-mean SD at
800: C−B 0.002513, B−A 0.003221, C−A 0.00264, A50−A 0.002915 (at 400:
0.004182 / 0.005237 / 0.004439 / 0.004155). Noise-curve crossing of 0.002 on
the scaled reading: C−B n ≈ 5,264, B−A ≈ 4,497, C−A ≈ 1,826, A50−A ≈ 4,797
(relative RMSE 0.16–0.56, an order-of-magnitude guide). Timing: seconds per
simulation A 0.049–0.060, A50 0.047–0.060, B 0.047–0.056, C 0.108–0.123;
peak RSS A/A50 ≈ 0.8 GB, B/C ≈ 0.5 GB; concurrent-arms wall for one 255-fixture
batch: 400 → 3.39 h, 800 → 6.09 h (T1 sets it; sharding the T1 arm divides
that).

---

## D10. 1c shard consistency (user go: 2026-09-11 ≈13:10 IST, together with 1d)

Written before any 1c run. 1c is a consistency check, not a result: no
metric is read; the comparison is byte equality of per-fixture outputs.

| # | Check | Pass condition |
|---|---|---|
| 10.1 | Cases | a fixture set covering all five registered boundary cases, each named by cricsheet id and the rule that chose it: (a) a same-day pair split across two shards; (b) a fixture on the first and on the last date of the iteration set; (c) a fixture without an odds row (a male T20 from the context corpus inside the window that is not in the odds file); (d) a fixture whose same-day sibling is in `data/t20s_json` but not in the evaluated set; (e) the first fixture of every shard (process initialisation) |
| 10.2 | Runs | every arm (A, A50, B, C) simulated on the case set serially in one process and sharded across ≥2 processes, each shard keeping the full replay context (`--context-dir data/t20s_json`), same base seed, same n_sims (small), threads 1; unregistered runs (no `--config`), because the case set is not a registered block |
| 10.3 | Identity | for every arm and every case fixture: the per-fixture eval record, the per-simulation raw rows (`raw_sims.jsonl`) and the provenance as-of stamp (date, ordered predecessor list, `matches_advanced`), eligibility fields and per-fixture seed are byte-identical between the serial run and the shard that contains the fixture, after dropping run-level fields (timestamps, output paths, shard membership); any difference fails naming arm, fixture and field |
| 10.4 | Odds-less fixture | case (c) is stamped, unscored, `odds_row_found: false`, and identical between serial and sharded |
| 10.5 | Post-run assertions | `scripts/sequence_track/merge_shards.py` (or equivalent) asserts, over a set of shard outputs: union of fixture ids equals the expected set, no duplicate fixture across shards, every odds record claimed by exactly one fixture, coverage counts (scored / unscored / skipped) sum to the totals; each assertion has a negative test |
| 10.6 | Tests and record | tests for the comparison and the assertions on synthetic outputs; the 1c run's command lines, case table and PASS/FAIL per (arm, case) recorded here; no metric read |

### Result (D10)

Recorded 2026-09-11 (first pass ≈13:40 IST; final pass appended below once
the runner fix landed). Files: `scripts/sequence_track/shard_consistency_1c.py`,
`scripts/sequence_track/merge_shards.py`, tests
`scripts/tests/test_shard_consistency_1c.py` (33) and
`scripts/tests/test_merge_shards.py` (27): `60 passed`. Outputs under
`models/embeddings/seq_stage1/consistency_1c/`. Both scripts refuse to write
any number that is not an allow-listed count; no metric was read.

| # | Result (first pass, before the runner fix) |
|---|---|
| 10.1 | PASS — 8 fixtures, 3 shards by `round_robin_over_date_then_id_v1`: (a) 1512766 / 1525160 (2026-02-26) split across shards 1 / 2, plus 1527575 / 1527576 (2026-04-16) split 0 / 1 and 1496921 / 1448357 (2025-09-10) split 1 / 0; (b) 1496921 (first date on disk, **2025-09-10** — the docs' 2025-07-01 is the window bound, not the first fixture) and 1527575, 1527576 (last date 2026-04-16); (c) 1448357, a male T20 from `data/t20s_json` in the window with no odds row; (d) 1477610 (2026-01-29; siblings 1519139, 1519636 context-only) and 1496921 (sibling 1477997 context-only); (e) shard firsts 1448357, 1496921, 1477609; control 1477609 (alone on its date) |
| 10.2 | PASS — 16 runs (4 arms × serial + 3 shards), unregistered (registered smoke commands minus `--config`, only fixture dir / `--n-sims 20` / output dir substituted; `--threads 1`, base seed 20260910, full context), all exit 0, 6–20 s each |
| 10.3 | **FAIL on one field, everything else identical** — 96 (arm, fixture, artefact) rows: `raw_sims` rows 32/32 identical, provenance per-fixture 32/32 identical, eval records 28/32 identical; the 4 failures are fixture 1477610 on every arm, and the only differing field in the whole run is `competition_cluster_id` (all differing keys were listed, not just the first). Run identity serial vs shard 12/12 identical. Excluded from the comparison: `as_of.matches_advanced` only, a run-cumulative process counter; in its place the ordered `same_day_advanced_before` list is compared in full and the decomposition (advanced − same-day count constant per date, non-decreasing) is asserted, PASS on 16/16 |
| 10.4 | PASS — 1448357 stamped on all arms and both paths, `odds_row_found: false`, `scored: false`, `skip_reason: no_odds_row`, absent from `eval.json` and `raw_sims.jsonl`, provenance identical |
| 10.5 | PASS — per arm: union == 8, no duplicates, every odds record claimed exactly once, coverage 7 scored + 1 unscored + 0 skipped; merged == serial byte-for-byte after the cluster re-stamp; negative tests for every assertion |
| 10.6 | recorded here; commands in `consistency_1c.md` |

**Diagnosis of the 10.3 failure.** `scripts/sim_eval/run_sim_eval.py` builds
the competition-cluster lookup from the run's own fixture directory, so the
I3 block id `event:<name>|block_start:<date>` takes `block_start` from the
first event member *in that directory*: serial (holding 1477609, same event,
2026-01-27) stamped 1477610 with `block_start:2026-01-27`; shard 0 (holding
only 1477610) stamped `2026-01-29`. This is the registered block key of
invariant 7, and `reslice_eval_json.py` prefers a stamped id, so reslicing
does not repair it. `merge_shards.py` gained an explicit, opt-in
`--cluster-source-dir` re-stamp from the registered set (fail-closed on an
uncovered fixture; `n_changed` recorded; 1 per arm here). **Decision
(Fable, 2026-09-11):** a post-hoc repair does not satisfy "serial and sharded
outputs must be identical"; the runner is fixed to build the lookup from the
registered fixture set for every shard (`run_arm --cluster-source-dir`, pinned
per block, asserted identical across arms by the audit), and 1c is re-run;
the merge re-stamp is retained as a belt-and-braces assertion expected to
change zero records.

**Final pass (≈13:55 IST, after the runner fix).** `run_arm.py` gained
`--cluster-source-dir` (required with `--config`; the pinned value is the
`iteration_set_v2` role path, `data/polymarket_test_v2`, dir hash
`8005cad7…`), installed as a seam over the frozen runner's
`load_competition_clusters` that redirects the run's own fixture dir to the
registered set and fails closed on an uncovered fixture; pinned in every
block and shard and rendered into every command; provenance records the
dir and hash; the audit asserts the hash identical across arms and equal to
the pin. 14 new tests; suites `438 passed` (sequence-track files), full
`1184 passed, 5 skipped`. Re-pin `--verify` OK. **1c re-run
(`shard_consistency_1c.py --force`): `PASS: 117 checks, 0 failing`** —
eval records 32/32, raw-sims rows 32/32, provenance 32/32 identical serial
vs shard on all four arms; run identity 12/12; decomposition 0 problems;
merge assertions 8/8 with merged == serial; cluster re-stamp `n_changed = 0`
on every arm; fixture 1477610 now stamped
`event:West Indies tour of South Africa|block_start:2026-01-27` on both
paths. **D10: PASS.** The 1c case set was stamped from its own registered
set (`consistency_1c/serial/fixtures`) because it holds a fixture outside the
iteration set; 1d stamps from `data/polymarket_test_v2`.

---

## D11. 1d full run (user go: 2026-09-11 ≈13:10 IST, conditional on 1c passing)

| # | Check | Pass condition |
|---|---|---|
| 11.1 | Shard registration | the full-run block registers a deterministic partition of the 255 registered fixtures into 10 shards (rule stated, e.g. round-robin over the `(match_date, cricsheet id)` order so long and short matches balance and each shard spans the window), materialised as per-shard fixture dirs under `models/embeddings/seq_stage1/full/shards/<k>/fixtures/` with md5 and count pinned per shard; the union of shard inventories equals the registered 255-fixture set (asserted by `--verify`); every per-shard command differs from the registered full-run command only in `--fixture-dir` and `--output-dir` |
| 11.2 | Settings | n_sims 1,600 (`CHOSEN_N_SIMS`), base seed 20260910, threads 1, clip [0.01, 0.99], roster selector, B18 graft, i7 cache, the retrained B/C checkpoints; `run_arm --config` verifies each shard against its registered shard block and records `config_verified: true` |
| 11.3 | Driver | 40 jobs (4 arms × 10 shards) run at a concurrency of 10 one-thread processes, detached (nohup), skip-if-complete, STOP file, per-job wall / exit / simulation time / peak RSS recorded; the T1 arm's shards are scheduled first so the slowest arm sets the wall clock |
| 11.4 | Completion | 40 jobs exit 0; per shard the cross-arm audit passes across the four arms with `--expected-fixtures` the shard dir; the D10.5 post-run assertions pass on the union for every arm (255 fixtures, no duplicates, every odds record claimed once, coverage counts) |
| 11.5 | Merge | per arm, one merged eval JSON (the union of the ten shards' per-fixture records in the registered chronological order) plus a merged provenance, with the shard of origin recorded per fixture; reslice to all / ≥$50k / ≥$100k; realism and prop scorers run on the merged raw simulations |
| 11.6 | Numbers not read | no winner log loss, Brier, ROI or edge is read from any shard or merged output before the gate runs; the merge and assertions print counts only |
| 11.7 | Stop | the gate (D12) runs only after 11.4–11.6 pass and its table is written |

### Result (D11)

**Launched 2026-09-11 14:01 IST** (`nohup caffeinate -i … run_full_1d.py
--concurrency 10`, log `models/embeddings/seq_stage1/full/driver.log`,
record `full_record.json`), after the final re-pin (`--verify` OK) and a
smoke re-run against that config (config sha256 `b41ceb34…`, audit PASS,
threads 1, cluster source hash `8005cad7…` identical on all arms;
`evidence/smoke_round5_*`). Shards: 10, round-robin over
`date_then_match_id_lexicographic_v1`, sizes 26×5 + 25×5 = 255, dir md5s
`63d8b5cd…`, `18a52752…`, `3e0e1d26…`, `8040bcd6…`, `2ac21d13…`,
`e85d917e…`, `5b01fba2…`, `38f4cca6…`, `e9ae15b3…`, `01d497ef…`. Jobs 40,
order C → B → A50 → A, n_sims 1,600, base seed 20260910. Expected wall
≈3.5 h from the 1b/probe rates (T1 ≈0.166 s/sim at ten concurrent).

**Completed 2026-09-11 18:36 IST** (driver exit 0; `full_record.json`,
config sha256 `b41ceb34…`). Per arm: C: 10 shards, exit codes [0], wall min/mean/max 8327/8511/8691 s, peak RSS max 512 MB; B: 10 shards, exit codes [0], wall min/mean/max 2429/2490/2565 s, peak RSS max 629 MB; A50: 10 shards, exit codes [0], wall min/mean/max 2379/2429/2489 s, peak RSS max 888 MB; A: 10 shards, exit codes [0], wall min/mean/max 2137/2198/2261 s, peak RSS max 792 MB.
Per-shard cross-arm audits: `[audit shard k] pass` for k = 0…9 (11.4).
Merges (`merge_shards.py`, per arm, `--cluster-source-dir
data/polymarket_test_v2 --odds betting_odds_polymarket_v2.json`): every arm
`[pass]` on run_identity, union_equals_expected (255), no_duplicate_fixture,
odds_claimed_once, coverage_counts (255 scored + 0 unscored + 0 skipped);
cluster re-derivation changed 0 records on every arm; merged outputs under
`full/merged/<arm>/` (11.5). Reslice: 255 / 168 / 110 fixtures on
all / ≥$50k / ≥$100k for every arm. Realism scorer 247 of 255 per arm (3
no-decided-winner and 5 D/L fixtures excluded by design), prop scorer 250
fixtures, 15 families, per arm. No winner log loss, Brier, ROI or edge was
read before the gate (11.6). Observed wall per shard (ten concurrent): T1 139–145 min, B 40–43 min, A50 40–41 min, A 36–38 min (`full_record.json`); total 14:01 → 18:36 IST.

---

Note on test counts in D9–D12: every pytest count quoted in these sections
is the count reported by the implementing agent or by Fable's shell at the
time; the retained, traceable evidence is the full-suite output saved at the
end of the stage (D12.9 result block), not the individual invocations.

## D12. Gate and verdict

Written 2026-09-11 ≈17:30 IST, after the C and B merges and before any gate
run or any read of a winner log loss. The registered rule
(`decision_rule` in the config) is applied as written and is not restated
after the numbers are seen.

| # | Check | Pass condition |
|---|---|---|
| 12.1 | Inputs | per arm, the merged full-run output (D11.5) resliced into `all` / `50000` / `100000`; the gate reads the `50000` slice for the primary contrasts and the other two for the secondary report; every input file's sha256 recorded |
| 12.2 | Gate runs | `claim_gate.py --kind match_model --odds-role odds_iteration_v2 --cluster-source-dir data/polymarket_test_v2` for each confirmatory contrast (C−B, B−A, C−A) and the exploratory A50−A on the primary slice, and for the same four on `all` and `100000` (secondary); each gate JSON kept unchanged under `models/embeddings/seq_stage1/full/gate/`; the gate recomputes prices, placement and profit from the registered odds role |
| 12.3 | Uncertainty | `tournament_time_block_v1`, 10,000 seed-42 whole-event resamples; block count per slice recorded; a slice with fewer than 10 blocks is labelled descriptive |
| 12.4 | Multiplicity | Holm adjustment across the three confirmatory contrasts, applied to the block-bootstrap p-values (or equivalently to the interval level) exactly as the gate exposes them; A50−A not adjusted; the adjustment script and its inputs recorded |
| 12.5 | Classification | each contrast labelled parity / favourable / adverse / inconclusive by the registered outcomes (interval inside ±0.007 → parity; interval excludes zero and point beyond ∓0.007 → favourable/adverse; else inconclusive), on the Holm-adjusted intervals for the confirmatory family; the pre-registered expectation ("parity everywhere, no arm advancing") quoted beside the result |
| 12.6 | Advancement | B advances only if B−A is favourable; C only if C−A is favourable; a favourable C−B alone is recorded as evidence about the pair for stage 2, not an advancement; any market claim additionally needs the gate's market comparison under the five registered cost scenarios and a paired Δprofit/ΔROI interval, and is otherwise not made |
| 12.7 | Exploratory | realism and prop outputs (D11.5) read only after 12.5 is recorded; reported as exploratory with no advancement or claim attached; prop families reported paired against the fair baseline |
| 12.8 | Report | `research/reports/embeddings/SEQ_STAGE1_REPORT.md` (new file): question, arms, settings, 1b/1c facts, the four contrasts on three slices with intervals and labels, realism and props, every registered deviation and asymmetry and known limitation restated, the single-checkpoint caveat, and the D9 stop-rule change; all numbers verbatim from the gate JSONs |
| 12.9 | Astra | end-of-stage review over D9–D12 and the report, rounds to SIGN-OFF, recorded in D8-style table; then one commit by Fable |
| 12.10 | Verdict | after the user's advancement decision, `research/log_verdict.py` with the unchanged gate JSON(s); the user makes the call; Fable is the only caller |

### Result (D12)

Recorded 2026-09-11 18:45 IST. Gate JSONs under
`models/embeddings/seq_stage1/full/gate/` (`<contrast>_<slice>.json`, twelve
files, unchanged since written; sha256 prefixes below), Holm outputs
`holm_<slice>.{json,md}` from `scripts/sequence_track/holm_stage1.py`
(`--verify-gate`; 60 tests after the step-down fix). Every number below is copied from those files.

| # | Result |
|---|---|
| 12.1 | PASS — inputs are the merged, resliced outputs of D11.5 (`full/merged/<arm>/sliced/eval_min_volume_50000.json` etc.); each sliced file's canonical sha256 is recorded in its gate JSON and re-verified by the Holm script |
| 12.2 | PASS — twelve gate runs (4 contrasts × 3 slices), all exit 0; `--odds-role odds_iteration_v2 --cluster-source-dir data/polymarket_test_v2`; the gate recomputed prices, placement and profit from the registered odds role |
| 12.3 | PASS — `tournament_time_block_v1`, 10,000 seed-42 resamples; blocks 18 (≥$50k, 167 paired records: one of the 168 slice fixtures dropped symmetrically by the gate's pairing rule), 11 (≥$100k, 110), 25 (all, 252); no slice below the 10-block floor. Every gate is `provisional: true` because each arm is a single checkpoint (the gate's ≥5-seed rule), so per invariant 9 nothing here can be LANDED — consistent with the plan's single-checkpoint screen |
| 12.4 | PASS — Holm step-down over C−B, B−A, C−A on the bootstrap p-values (`p = 2·min(P(δ*≤0), P(δ*≥0))`); the script reproduced every gate `ci95` to 1e-9 from the same draws before adjusting; A50−A unadjusted |
| 12.5 | **Primary slice ≥$50k, registered outcomes.** Favourable/adverse require the Holm step-down rejection (adjusted p ≤ 0.05) and a point beyond ±0.007; the intervals quoted are rank-local percentile intervals, not simultaneous Holm intervals (Astra end-of-stage MUST-FIX 1; the classifier was corrected to enforce the step-down and the labels did not change). C−B point −0.0257, rank-local interval [−0.0346, −0.0102], raw p 0.0006, Holm p 0.0018 → **favourable**; B−A +0.0247, [−0.0027, +0.0554], Holm p 0.1640 → **inconclusive**; C−A −0.0009, [−0.0277, +0.0288], Holm p 0.8936 → **inconclusive**; A50−A (exploratory) +0.0276, [+0.0095, +0.0419], raw p 0.0002 → **adverse**. Pre-registered expectation was "parity everywhere, no arm advancing": C−B departs from it in the favourable direction; B−A and C−A are inconclusive rather than parity (intervals wider than ±0.007) |
| 12.6 | **No arm advances.** B−A is not favourable, C−A is not favourable. C−B favourable is recorded as evidence about the full-T1 vs token-MLP pair for stage 2, per the registered rule, not as an advancement and not as a measurement of sequence memory. No market claim is made: every gate is provisional (single checkpoint), and the Δprofit intervals are reported below as descriptive |

Gate table, all slices (Δ = candidate − baseline; Δprofit in units per flat 1-unit bet, descriptive, cost scenario 0/0 winnings):

| contrast | slice | n | blocks | ΔLL point | ΔLL ci95 | Δprofit point | Δprofit ci95 | gate verdict | gate sha256 |
|---|---|---|---|---|---|---|---|---|---|
| C-B | 50000 | 167 | 18 | -0.02569 | [-0.03307, -0.01346] | +0.1859 | [+0.0493, +0.3255] | PROMISING | `2f43c19b88c2…` |
| B-A | 50000 | 167 | 18 | +0.02475 | [-0.00118, +0.05257] | -0.0687 | [-0.1489, -0.0070] | FAILED | `aa13ad29f269…` |
| C-A | 50000 | 167 | 18 | -0.00095 | [-0.02772, +0.02877] | +0.1173 | [-0.0690, +0.2887] | FAILED | `038fdfac04f2…` |
| A50-A | 50000 | 167 | 18 | +0.02762 | [+0.00945, +0.04194] | -0.0918 | [-0.1741, -0.0081] | FAILED | `f62d1c197151…` |
| C-B | 100000 | 110 | 11 | -0.02672 | [-0.03780, -0.01382] | +0.1696 | [+0.0297, +0.3646] | PROMISING | `27dab32a4691…` |
| B-A | 100000 | 110 | 11 | +0.04005 | [+0.00879, +0.06267] | -0.1288 | [-0.3103, +0.0000] | FAILED | `3e5177e628b4…` |
| C-A | 100000 | 110 | 11 | +0.01334 | [-0.02302, +0.03801] | +0.0408 | [-0.1516, +0.3087] | FAILED | `28c63b055010…` |
| A50-A | 100000 | 110 | 11 | +0.04235 | [+0.02602, +0.04957] | -0.1955 | [-0.3176, -0.0880] | FAILED | `8117301ae2a8…` |
| C-B | all | 252 | 25 | -0.02149 | [-0.03053, -0.01004] | +0.1266 | [+0.0233, +0.2251] | PROMISING | `39f277ce7100…` |
| B-A | all | 252 | 25 | +0.02695 | [+0.00707, +0.04907] | -0.0794 | [-0.1418, -0.0250] | FAILED | `013b5bb1d399…` |
| C-A | all | 252 | 25 | +0.00546 | [-0.01578, +0.02942] | +0.0472 | [-0.0748, +0.1686] | FAILED | `9e713e7947d7…` |
| A50-A | all | 252 | 25 | +0.02584 | [+0.01296, +0.03751] | -0.1081 | [-0.1707, -0.0417] | FAILED | `77d698ee9662…` |

Secondary slices (outside the confirmatory decision; the same Holm step was run for reporting, and the numbers below are the `gate ci95` / `point` columns of `holm_100000.md` / `holm_all.md`, i.e. the unadjusted gate intervals):
C−B favourable on both (≥$100k −0.0267 [−0.0378, −0.0138]; all −0.0215
[−0.0305, −0.0100]); B−A adverse on both (≥$100k +0.0401 [+0.0088, +0.0627];
all +0.0270 [+0.0071, +0.0491]); C−A inconclusive on both; A50−A adverse on
both. The gate's own labels (`PROMISING` / `FAILED`) are its LANDED-rule
readout and are not the stage-1 classification.

| # | Result (continued) |
|---|---|
| 12.7 | PASS — realism and props read after 12.5 was recorded, from `full/merged/<arm>/realism.json` and `props.json`; reported in the stage report § 6 as exploratory. Realism first innings (247 fixtures per arm): bias A +2.12, A50 +0.50, B +3.01, C +5.06 runs; P10–P90 coverage 0.753 / 0.745 / 0.761 / 0.773 (nominal 0.80); sim wickets 6.86 / 6.82 / 6.68 / 6.68 vs actual 7.00. Props (250 fixtures, 15 families): every arm beats the as-of fair baseline with an interval excluding zero on `innings_runs_ou_{160_5,170_5,180_5}`, `pp_total_ou_{45_5,50_5,55_5}`, `team_highest_individual_ou_{29_5,34_5,39_5}`, `batter_50plus` and `batter_runs_mae`, and loses on `bowler_wkts_{1,2,3}plus`; `top_batter` straddles zero. Exploratory; differs from E2 v2 (retired v7 stack, 100 sims); no claim |
| 12.9 | Astra end-of-stage rounds (`codex exec -m gpt-6-astra -c model_reasoning_effort="low" -s read-only`, prompt `<scratchpad>/astra_stage1_end.prompt.md`): **round 1** `<scratchpad>/astra_stage1_end_1.md` (≈18:55 IST) → **NO SIGN-OFF**, 5 MUST-FIX, 2 SHOULD, 3 NOTE. Dispositions: MUST-FIX 1 (Holm classifier used rank-local intervals without enforcing the step-down) → `holm_stage1.py` now rejects only via the cumulative step-down, parity decided on the gate ci95, intervals renamed `rank_local_*` with an explicit "not simultaneous" note, regression cases 0.020/0.021/0.022 and rank-3-cannot-reject added (60 tests); the three real Holm runs re-executed: every point, interval and p byte-identical, **no label changed**. MUST-FIX 2 ("parity-or-better", "T1 recovers ~0.028 from sequence history") → withdrawn; report now says C−A inconclusive with a near-zero point, A50−A adverse, neither establishing parity nor quantifying recovery. MUST-FIX 3 (teacher-forcing vs rollout "disagreement" on "the same pair") → restated as historical motivation on different checkpoints; a matched teacher-forced score named as the follow-up. MUST-FIX 4 (convergence numbers presented as measured primary-slice MC SD) → relabelled extrapolated two-batch-difference spreads (√2 factor, all ten fixtures not the primary slice, four threads, never measured at 1,600), range_95 called an empirical range. MUST-FIX 5 (deviation consequences dropped; "byte-identical" overstated) → consequences restated per id from the config; 1c wording qualified with the `matches_advanced` exclusion. SHOULD 6 → B shard wall corrected to 40–43 min; secondary-slice source columns named; p at the resolution floor shown as `<0.0002`; "none retyped" narrowed to table numbers; test-count note added above D12. SHOULD 7 → `merge_shards.py` identity keys now include `cluster_source_dir` and its hash (4 tests); `--check-only` re-run on all four arms: PASS (the merged provenance written earlier does not carry the two keys in its run block; the shard provenance does, all 40 identical). NOTE 9 (E2 vs BR2 restatement) → report cites both. NOTE 10 (verdict vocabulary) → adopted, see 12.10. Full suite after the fixes (`evidence/pytest_full_stage_end.txt`): `1248 passed, 5 skipped, 29 warnings in 124.48s`; artifact-free `1211 passed, 42 deselected` |
| 12.9 (round 2) | `<scratchpad>/astra_stage1_end_2.md` (≈19:20 IST) → **SIGN-OFF**; MUST-FIX 1–5 closed with file:line, SHOULD 6 substantially closed (three non-blocking wording items: renderer docstring, the § 7 "95% range" introduction, and that the retained pytest evidence is two summary lines — the first two fixed after the round, the third recorded as is), SHOULD 7 partially closed (analysis-source manifest requested → written to `full/gate/analysis_manifest.json`: sha256 of 21 post-processing sources and 30 inputs plus the report), NOTE 10 approved the verdict approach (see 12.10). A plain-language section (§ 10) was added to the report after sign-off at the user's request; it introduces no number not already in the tables |
| 12.8 | PASS — `research/reports/embeddings/SEQ_STAGE1_REPORT.md` rendered by `scripts/sequence_track/render_stage1_report.py` from the gate / Holm / realism / props / convergence / full-record files and the pinned config; sections: question, arms and settings, decision rule, primary-slice Holm table, secondary slices, full gate table with sha256 prefixes, uncertainty and non-claims, exploratory readouts, 1b spreads and timings, 1c, registered deviations/asymmetries/limitations restated, next |

Tables written before each step starts, after the user's explicit go for
that step. Known content they must cover (from the brief § 5 and § 6): the
1b shard rule (ten fixtures including long innings), candidates 50 / 100 /
200 / 400 / 800 with 50 added to the permitted list and the joint overlap
screen re-run, three batches per arm per count, the second base seed
rerun, seconds per match per arm and the extrapolation to 255, the user's
chosen reading of the stop rule recorded in the config before 1d; the five
1c boundary cases and the post-run union/duplicate/odds-claim assertions;
1d at the user's chosen count; `claim_gate --kind match_model` against
`odds_iteration_v2`, realism and prop scorers, verdict through
`research/log_verdict.py` with the unchanged gate JSON.
