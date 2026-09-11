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
| 3 | `<scratchpad>/astra_stage1_d8_3.md` (≈10:35 IST) | **SIGN-OFF** — MUST-FIX none, SHOULD none; round-2 MUST-FIX 1 closed (`retrain_i7.py:202`, `:334`, `:713`, `:532`, `:764`; tests `:716–875`), SHOULD 1–2 closed; Astra independently confirmed all 35 pinned source md5s, the ten checkpoint declarations, the live train/validation parquet md5s, config sha256 `f3ecd29b99cc6793ddd314fd5afe621ed0947b239c2145e40a29a699b6864947`, and that all four smoke provenance files bind to it; notes: tests and `--verify` not independently rerun by the reviewer (sandbox cannot run `uv`); 3.8 stays partially met; 1b unrun and outside this sign-off | closed; commit follows (8.2) |

---

## D9–D12. 1b timing and convergence, 1c shard consistency, 1d full run, gate and verdict

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
