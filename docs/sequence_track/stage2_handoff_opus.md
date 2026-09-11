# Stage 2 handoff: from Fable to the Opus orchestrator (2026-09-11, ≈23:15 IST; updated ≈00:25 IST 2026-09-12)

Fable orchestrated stage 2 through the design decisions, the acceptance
file and the review of the three implementation deliverables. This file
hands the rest of the stage to an Opus session. Read, in this order:
`docs/sequence_track/stage2_kickoff_brief.md` (in full),
`docs/sequence_track/stage2_acceptance.md` (in full: it is the contract),
`docs/SEQUENCE_TRACK_PLAN.md` § "Stage 2" and § "Statistical rules", then
this file.

## 1. Roles after the handover (user decision 2026-09-11)

- **Opus orchestrates and executes.** Run the steps in the order the
  acceptance file gives; fill every Result block verbatim from files; a run
  that did not finish has no number. Spawn an Opus implementer for anything
  that breaks; Sonnet for mechanical work with a case list. No more than
  three agents at once. You are the only committer and the only caller of
  `research/log_verdict.py`.
- **Astra thinks and reviews.** Do not invent design. Whenever a step needs
  a judgment the acceptance file does not already make, write the question
  to a prompt file and ask Codex Astra; record the prompt, output path and
  verdict line in the acceptance file. Astra also reviews at every
  registered gate: before the code commit (after 2c), before the results
  commit (after 2f), and at stage end. Iterate rounds to SIGN-OFF. Per the
  user, use **medium** reasoning from here:

  ```bash
  codex exec -m gpt-6-astra -c model_reasoning_effort="medium" -s read-only \
      "$(cat prompt.md)" </dev/null > out.md 2>err.md
  ```

  Astra runs on the laptop, never the mini. Use `</dev/null` or it hangs.
- **The user decides**: every advancement, the `k`, every verdict, and any
  change to the rules below. The user is near their Fable limit; bring
  decisions in plain language, point by point, and wait.
- Rules that override convenience: never read or evaluate against
  `data/golden/` or `data/forward_holdout/` (the two id-only reads in D2.3
  are done and frozen; do not repeat them); write only under
  `models/embeddings/seq_stage2/` (plus code/config/test/doc paths in their
  usual places); stage 1 artifacts, reports and configs are frozen, errata
  are new files; Python only via `uv run --no-sync`; commits per
  `CLAUDE.md` (summary + body, **no AI-attribution trailers**); D9, D10
  and D11 tables are written before those steps and after the user's go.

## 2. State at handover

Done and reviewed by Fable (results recorded in the acceptance file):

| deliverable | state |
|---|---|
| D2 decisions | recorded, Astra round 1 mapped |
| D3 arm code (`scripts/transformer_t1.py` +707/−51, `transformer_xr.py` 1 line, `scripts/tests/test_stage2_arms.py`, `test_stage2_masks.py`) | implemented; Fable switched the custom feed-forward to ReLU; 179 tests pass across the arm/mask/ablation/contract/recurrent files; **one redesign still required before smoke (§ 3.1)** |
| D3.10 recurrent arms (`scripts/sequence_track/recurrent_arms.py`, `test_stage2_recurrent.py`) | implemented, 15 tests pass; interface `build_recurrent_model(arm, n_feats, dmodel, dropout)` → module with `forward(feats, prev_y, pad_mask) -> (logits, {})`; parameter counts at n_feats=7: lstm 266,886, xlstm 351,158; xLSTM simplifications in `simplifications()` |
| D4 base logits | built and recorded (`models/embeddings/seq_stage2/base_logits/{train,validation,test}.npz` + `.json`; keys `logp`, `n_rows`, `parquet_md5`) |
| D5 cohort | built, verified, **frozen** (`models/embeddings/seq_stage2/cohort/FROZEN.json`, 210 matches, 48,406 rows). Do not touch it until D10.10 |
| D3.1 config, D3.2 pin, D3.12 driver, D8.2 queue | implemented and reviewed by Fable: `experiments/configs/seq_stage2_v1.yaml`, `retrain_stage2.py`, `pin_stage2.py` (`--verify`: OK), `queue.yaml` (16 mini jobs), tests 40 + 39 + 8 pass; full artifact-free suite 1 failed (the stage 1 closure test, § 3.2) / 1433 passed |

Nothing is committed. `git status` shows the modified `TODO.md`,
`scripts/transformer_t1.py`, `scripts/transformer_xr.py`, plus the new files
above and the untracked pre-existing `docs/cricml_atlas_updated.html` and
`experiments/results/item6_*` (not stage 2; leave them out of the commit).

## 3. Decisions Fable made on the implementers' findings (apply these)

### 3.1 Masked-arm keys carry their own outcome (redesign before smoke)

The implementer found that with the current token construction the target
of a masked arm also depends on the one-step history input of every
allowed token (for `same_entity`, an allowed token j that shares only the
batter carries its bowler's previous ball, a non-owned row). D7 as written
would fail, and the plan (A12.2) says rebuild, not footnote. Decision:

- For `recency`, `same_entity` and `aligned_hist_rf` (the relay-free arms),
  the key/value token for an earlier row j < i is
  `e_j = feat_proj(feat_j) + out_emb(y_j) + pos_j`, where `y_j` is row j's
  **own realised outcome** (known before ball i). The target's own token
  (the query stream's initial state and its self key/value) is
  `e_i^self = feat_proj(feat_i) + hist(i) + pos_i` with `hist(i)` the arm's
  registered history input (aligned pair for `same_entity` /
  `aligned_hist_rf`; innings-previous for `recency`). Use a separate
  embedding table for own outcomes (`own_out_emb`, 6 classes, no BOS).
- Then the dependency set is exactly `S(i) = attention_set(i) ∪ {history
  source rows of i}`, as the arm register defines it, and D7 certifies
  that. Update `outcome_dependency()` in `test_stage2_arms.py` and the
  3.7 invariance tests accordingly; keep the standard-wiring arms unchanged.
- Record as a `known_asymmetries` entry in the config: relay-free arms read
  earlier rows as (state, own outcome) pairs, standard arms as (state,
  shifted previous outcome) pairs; so `aligned_hist_rf − aligned_hist`
  measures wiring plus key construction, not wiring alone. Ask Astra to
  confirm this decision in the pre-commit round; if Astra prefers keeping
  the current construction and widening S(i), bring both to the user.

### 3.2 Stage 1 pin closure test

Editing the trainer makes `scripts/tests/test_pin_stage1.py::test_source_closure_is_hashed_in_full`
fail (`transformer_t1.py` md5 is stale) — by design, since the stage 1
config is frozen. Decision: write `docs/sequence_track/stage1_erratum_trainer_closure.md`
(the trainer changed for stage 2; the four stage 1 arms were verified
bitwise identical in logits, state-dict keys and parameters under the same
seed against `git show HEAD:scripts/transformer_t1.py`; stage 1 evidence
stands), and change the test to accept either the live md5 or the md5 of
`scripts/transformer_t1.py` at the stage 1 commit `2c7ea6f` (via
`git show`; skip if git is unavailable). Do not edit
`experiments/configs/seq_stage1_sim_v1.yaml`.

### 3.3 Accepted as implemented

- Custom layers use ReLU (Fable's edit) so cross-wiring contrasts carry no
  activation difference; `TokenMLPBlock` (stage 1 `mlp`) keeps GELU.
- `arm_params.base_logits_md5` is one md5 over the sorted
  `<split>=<npz md5>` lines, with `base_logits_md5_by_split` alongside.
- `collate` returns a `Batch` NamedTuple whose first five fields are the
  old tuple; `transformer_xr.py` updated with `[:5]`.
- `arm_params` is a top-level `metrics.json` key; `training_contract` is
  untouched.
- xLSTM simplifications (no conv pre-blocks, dense single-head sLSTM, no
  up/down projection, SwiGLU FFN per block, exp gates on both) are recorded
  and go into the config's `deviations`.
- Residual OOF refits at fixed 25 rounds, no early stopping, encoders
  reused: recorded deviation.
- Cohort: merged symlink corpus; two non-feature I9 exposure columns in the
  rebuild; `materialized_full/` train/validation pruned after parity.

## 4. Interface contracts the pieces share

- Trainer flags: `--arm` ∈ {mlp, full, fixed_decay, fox, aligned_hist,
  aligned_hist_rf, recency, same_entity, residual_mlp, residual_t1, lstm,
  xlstm}; `--k <int|unr>` required for recency/same_entity, refused
  elsewhere; `--base-logits-dir` required for residual_*, refused
  elsewhere; `--residual-l2` default 0.001; `--save-predictions` on for
  every run. Training block: d 128, 2 layers, 4 heads, batch 128, 30
  epochs, lr 3e-4, patience 3, no aux, `--no-kit`, no `--score-test`,
  `--stats-cache-role stats_cache_i7`, `--data-dir data/xgb_data_i7`.
- Driver: `scripts/sequence_track/retrain_stage2.py --config-ids <id>
  [--seeds 7,13] [--epochs N --out-root PATH]`; output
  `models/embeddings/seq_stage2/runs/<config_id>/seed_<s>/`; reuse check
  includes `arm_params`.
- Base logits npz schema: `logp` float32 (n_rows, 6) in T1 class order,
  `n_rows`, `parquet_md5` (must equal the live split parquet md5).
- Cohort: `cohort/cricket_data_i7_cohort.parquet` (48,406 rows, 158
  columns, same 114 features), sha256 in `FROZEN.json`; cohort base logits
  for the residual arms are produced later with
  `build_base_logits.py --split cohort --parquet <that file>` (never before
  D10.10).
- Bootstrap clusters: tournament blocks from
  `scripts/sim_eval/eval_statistics.load_competition_clusters("data/t20s_json")`
  keyed by match id (`registered_experiment.match_ids` strips the innings
  prefix); the validation split has 47 blocks, 0 unmapped. For the cohort,
  the same function over `cohort/raw_json/` (its matches are not in
  `data/t20s_json`); check the block count and apply the ≥10 rule.

## 4b. Update at ≈00:25 IST 2026-09-12 (Fable's last two agents landed)

- § 3.1 and § 3.2 are DONE (acceptance D3 addendum). Artifact-free suite
  1458 passed; full suite 1 failed (`test_pin_stage1.py::
  test_verify_passes_on_the_committed_config`, artifact-gated) — ask Astra
  whether `pin_stage1.verify()` gets the same stage-1-commit fallback or an
  accepted exception. Do not touch the stage 1 config.
- D7 is DONE and PASSED for all four masked arms (exactly 0), with the
  positive-control interpretation in D7.4 to put to Astra at gate 1.
- D6 is PARTIAL: six of sixteen smoked; the other ten still need the
  one-epoch smoke (step 2 below), then fill 6.3/6.4.
- Step 1 below is therefore reduced to: confirm `git status`, run the two
  suites once more, and proceed to step 2.

## 5. Execution order from here

1. **Finish 2a.** Apply § 3.1 and § 3.2 (Opus implementer, tests to the
   check ids). Review the config/driver/pin agent's report against 3.1,
   3.2, 3.12, 3.13, 8.2. Run `pin_stage2.py --write` then `--verify`
   (expect `pin_stage2: OK`). Full artifact-free suite must pass with zero
   failures; record the count in D3's Result block.
2. **2b smoke (D6).** `retrain_stage2.py --seeds 7 --epochs 1 --out-root
   models/embeddings/seq_stage2/smoke/` for all 16 configurations on the
   laptop. Record exit codes, artefact presence and wall seconds only.
   Never read a smoke log loss.
3. **2c dependency test (D7).** Write
   `scripts/sequence_track/ownership_dependency_test.py` per D7.1–7.4
   (perturb raw outcomes and features outside S(i), rebuild derived
   inputs, 2,000 targets seed 29, CPU). Run on the smoke checkpoints of
   `same_entity_k0`, `same_entity_k30`, `same_entity_unr`, `recency_k30`
   (pass ≤ 1e-6) and `full`, `aligned_hist` (positive controls > 1e-3).
   On failure: one-layer rebuild of that arm, re-smoke, re-test (D7.5).
   Paste the table into D7's Result block.
4. **Astra gate 1 (code).** Prompt covers D2–D8 tables, every changed
   file, the § 3.1 decision, and the deviations. Iterate to SIGN-OFF;
   record rounds in a D11 table written before the round. Then **commit**
   the stage 2 code, config, tests, acceptance file, TODO.md, erratum and
   this handoff (one commit, plain summary + body, no trailers). Do not
   commit anything under `models/`.
5. **2d overnight (D8).** `git push mini embeddings-ladder` and check the
   mini out at that commit; rsync `models/embeddings/seq_stage2/base_logits/`
   to the mini (`rsync -a --stats`); re-verify frame and cache md5 on the
   mini; overwrite each job's `expected_hours` from the D6.3 wall times
   (× 3 for the mini × 2 seeds, rounded up); `run_queue.sh --machine mini
   --dry-run` then the real run under `caffeinate` in `nohup`/`screen`;
   record the start stamp and confirm it is after `FROZEN.json`'s
   `frozen_at_utc` (2026-09-11T17:23:20Z). Mini rules: one job at a time,
   `OMP_NUM_THREADS=4`, nothing else running. In the morning rsync
   `models/embeddings/seq_stage2/runs/` back (`--stats`), fill D8's Result.
6. **2e k sweep (D9).** Write the D9 table first (after the user's go):
   per k, the two-seed mean validation LL and each seed's value, verbatim
   from `summary.yaml`; rule: best k unless no k beats k = 30 by more than
   0.002, then keep 30; the user reads the table and confirms. If the
   chosen k ≠ 30, queue `recency_k<chosen>` at the same seeds before 2f.
7. **2f statistics and report (D10).** Write the D10 table first (after
   the user's go), then `scripts/sequence_track/stage2_stats.py`:
   per-row LL from the saved validation probabilities; tournament-block
   bootstrap (2000 reps, seed 29, ball-weighted, ≥10 blocks per slice else
   descriptive); estimand (i) per checkpoint and (ii) joint seed+block,
   with two seeds labelled descriptive; contrasts and families from the
   config's `statistics` block (each arm − mlp; fox − fixed_decay;
   same_entity_k − recency_k; same_entity_unr − aligned_hist_rf;
   aligned_hist_rf − aligned_hist; aligned_hist − full; residual_t1 −
   residual_mlp; family per candidate = {primary vs matched control, death
   gate, chase gate}, Holm); non-inferiority = upper 95% bound of
   (arm − mlp) on death overs and on chases below +0.002; exploratory
   slices labelled. **Family freeze** (D10.9): write the candidate list,
   `k`, controls and families into the config, pin, and get the user's
   go before the cohort is opened; then score the frozen set on the cohort
   exactly once (D10.10), residual arms after building cohort base logits.
   Report: `research/reports/embeddings/SEQ_STAGE2_REPORT.md` from a
   generator script that reads every number from files, with § 10 in plain
   language (what was tested, what came out, which model is best and why,
   what to do next) as the stage 1 report does; restate every deviation,
   asymmetry and limitation, dropping none.
8. **Astra gate 2 and stage end (D11).** Iterate to SIGN-OFF, commit the
   results (reports, acceptance file, config provenance; never `models/`),
   then the verdict through `research/log_verdict.py` with the unchanged
   gate JSON, on the user's decision only. Two-seed evidence is provisional
   and can never be LANDED (invariant 9); say so in the report.

## 6. Traps

- The mini is at `dfa45ad`, six commits behind, with no `models/embeddings/`.
  It needs the code commit before the night, and only the base logits
  under `models/`.
- `queue.yaml` defaults must be mini-sized (floor 3 GB, cap 12 GB) or the
  runner refuses every job on a 16 GB machine.
- Never launch the queue before D7 passes for the masked arms and before
  the code commit exists; the queue's config hash is what makes a night
  resumable.
- MPS is not bit-reproducible; record `mps_bit_reproducible: false` as
  stage 1 did. CPU re-scoring of a checkpoint differs from its recorded
  validation LL in the last decimals; report both, select on neither.
- The test split is exploratory; the cohort is the only confirmatory read
  and it happens once. Two seeds cannot confirm anything.
- Every number in the acceptance file and the report is copied from a
  file. If a run did not finish, it has no number.
- Do not re-run Fable's earlier facts: the acceptance file's "Design
  facts" block is current as of 2026-09-11.
