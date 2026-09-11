You are Codex Astra reviewing, read-only, the stage 2 CODE GATE of the sequence/embeddings research track in this repository (/Users/aryamangupta/CricML/Match_Prediction, branch embeddings-ladder). This is the review that must be passed before the stage 2 code is committed. You reviewed the stage 2 execution PLAN earlier and returned "AGREE WITH CHANGES" with 11 MUST-FIX items; this review checks the implementation against the acceptance contract and closes or re-opens those items.

## What to read

Contract and context (read these first, in this order):
- docs/sequence_track/stage2_acceptance.md — THE CONTRACT. Every check id (D1-D8) and its recorded Result. Your review is against this file.
- docs/sequence_track/stage2_handoff_opus.md — the orchestration handoff; § 3 holds the three decisions made on implementer findings, § 4 the interface contracts, § 4b the current state.
- docs/SEQUENCE_TRACK_PLAN.md — § "Statistical rules for the whole track" and § "Stage 2" (the plan of record; the acceptance file must not contradict it).
- docs/sequence_track/stage2_kickoff_brief.md — background on why the stage exists.
- docs/sequence_track/stage1_erratum_trainer_closure.md — the erratum written because the shared trainer changed.

Implementation (all uncommitted in the working tree; `git status` lists them; `git diff` shows the modified ones against commit 2c7ea6f):
- scripts/transformer_t1.py (MODIFIED, ~+818/-50): the stage 2 arms. Key additions: `aligned_history`, `attention_set`, `history_source_rows`, `dependency_set`, `attention_mask`, `alibi_slopes`/`alibi_bias`, `_masked_attention`, `_FeedForward`, `BiasedEncoderLayer`, `RelayFreeLayer`, `T1Model.__init__`/`_bias`/`forward`/`_mix`, `Batch`, `collate`, `parse_k`, `load_base_logits`, `arm_params_block`.
- scripts/sequence_track/recurrent_arms.py (NEW): lstm and xlstm arms.
- scripts/sequence_track/build_base_logits.py (NEW): residual arms' base log-probabilities incl. out-of-fold train refits.
- scripts/sequence_track/build_cohort_stage2.py (NEW): the untouched cohort builder.
- scripts/sequence_track/retrain_stage2.py (NEW): the multi-seed training driver.
- scripts/sequence_track/pin_stage2.py (NEW): the provenance pin.
- scripts/sequence_track/ownership_dependency_test.py (NEW): the D7 certification.
- experiments/configs/seq_stage2_v1.yaml (NEW): the registered config, sixteen configurations.
- research/sequence_track/queue.yaml (MODIFIED): sixteen mini jobs.
- scripts/tests/test_stage2_arms.py, test_stage2_masks.py, test_stage2_recurrent.py, test_build_base_logits.py, test_build_cohort_stage2.py, test_retrain_stage2.py, test_pin_stage2.py, test_queue_stage2.py, test_ownership_dependency_test.py (NEW); scripts/tests/test_pin_stage1.py and scripts/tests/test_run_queue.py (MODIFIED).
- scripts/transformer_xr.py (MODIFIED, 1 line: `collate(...)[:5]` because `collate` now returns a NamedTuple).
- TODO.md (MODIFIED): one new backlog section.

Evidence on disk you may inspect (do NOT open anything under data/golden/ or data/forward_holdout/ — the access rule forbids it and two id-only reads already happened and are recorded in D2.3 and in the cohort's FROZEN.json):
- models/embeddings/seq_stage2/base_logits/{train,validation,test}.{npz,json}
- models/embeddings/seq_stage2/cohort/{FROZEN.json,eligibility_audit.csv,raw_manifest.json,notes_consumers.md,notes_context_case.md,parity_test_split*.json,materialize_report.json,.feature_hash}
- models/embeddings/seq_stage2/dependency/*.json
- models/embeddings/seq_stage2/smoke/*/metrics.json (one-epoch smoke checkpoints; their log-loss values are NOT to be read as results by anyone, including you — check structure, not numbers)

## The question

Does this implementation satisfy the acceptance contract well enough to commit, and does it preserve the statistical and access guarantees the plan requires? Specifically:

1. **Check-by-check.** For D3 (3.1-3.13), D4 (4.1-4.4), D5 (5.1-5.9), D6 (6.1-6.4), D7 (7.1-7.6) and D8.2/8.3: does the code actually do what the recorded Result claims? Name any check whose Result overstates what the code does. Verify by reading the code, not by trusting the table.
2. **Your 11 plan MUST-FIX items.** The acceptance file maps them to check ids (see the mapping line under the D2 table). For each, say CLOSED (with file:line evidence) or STILL OPEN.
3. **The relay-free key construction decision** (handoff § 3.1, acceptance D3 addendum). The masked arms now read an earlier row j<i as `feat_proj(feat_j) + own_out_emb(y_j) + pos_j` (row j's own realised outcome), while the target's own token carries its registered history input. This was done so that the dependency set is exactly `S(i) = attention_set(i) ∪ history_source_rows(i)` and the D7 certification can pass. Questions: (a) is this construction sound and leak-free — in particular is `y_i` itself unreachable from row i, and is every `y_j` used strictly in the past of i; (b) does it damage the interpretability of the registered contrasts, given that `aligned_hist_rf − aligned_hist` now measures wiring PLUS key construction and is recorded as an asymmetry; (c) is the alternative (keep shifted-history keys and widen S(i) to `⋃_{j∈A(i)} history_source_rows(j)`) preferable, and if so why? If you judge the current choice wrong, say so plainly — the orchestrator will bring both options to the user.
4. **D7.4's positive controls.** A standard-wiring arm scored against its OWN S(i) is trivially 0, because its attention set is the whole causal prefix and the only excluded rows are future rows that no causal model reads. So each positive control is scored against the S(i) of the masked arm sharing its history input at k=30 (`full` against recency's set, `aligned_hist` against same_entity's set), recorded as `dependency_set_arm`/`dependency_set_k`; `--set-arm` is refused for relay-free arms so a masked arm's certificate is always its own set. Is that a faithful reading of D7.4's "under the same perturbation", or does it weaken the control? Propose the exact wording for D7.4's Result if you would state it differently.
5. **The stage 1 pin test.** `scripts/tests/test_pin_stage1.py::test_source_closure_is_hashed_in_full` was loosened to accept either the live md5 of a closure source file or its md5 at the stage 1 commit 2c7ea6f, because the shared trainer had to change for stage 2 and the stage 1 config is frozen. The erratum records that the four stage 1 arms were verified bitwise identical (logits, state-dict key order, parameters, same seed) against the pre-edit file. A second, artifact-gated test still fails: `test_pin_stage1.py::test_verify_passes_on_the_committed_config`, because `pin_stage1.verify()` itself recomputes the live trainer md5. Which of these is right: (a) give `pin_stage1.verify()` the same stage-1-commit fallback; (b) leave the test failing and record an accepted exception with a named owner and expiry; (c) something else? Note the constraint: `experiments/configs/seq_stage1_sim_v1.yaml` is frozen evidence and must not be re-pinned. Also say whether loosening the closure test at all was acceptable, or whether the stage 1 closure should instead have been made to hash a snapshot copy of the trainer.
6. **Access and leakage.** Confirm: no stage 2 code path reads data/golden/ or data/forward_holdout/ beyond the two recorded id-only helpers in build_cohort_stage2.py; the cohort's priors are frozen from the production cache and not recomputed over post-2026-04-16 data (invariant 6); the cohort parquet is not loaded by any trainer, smoke or selection path (5.9); the test split is not scored during selection (`--score-test` off); the residual arms' train base logits are genuinely out-of-fold. Flag anything that could let future data or the cohort influence selection.
7. **Anything that will break the overnight run.** The next step after this commit is: push the commit to the Mac mini (16 GB, 10 cores, ~21 GB free, MPS), rsync only the base logits, and run `research/sequence_track/run_queue.sh --machine mini` — sixteen jobs, each `retrain_stage2.py --config-ids <id>` at seeds 7 and 13, `OMP_NUM_THREADS=4`, memory floor 3 GB, cap 12 GB, per-job timeout = 2 × expected_hours. Laptop one-epoch smoke wall times are in D6's Result. Name anything in the driver, queue, or arm code that will fail, hang, blow memory or silently skip on that machine, and anything about `expected_hours` that the orchestrator should set differently.

## Output format

Numbered MUST-FIX items first (things that must change before the commit), then SHOULD, then NOTE; each one line of reason plus file:line where it applies. Then a short section "MUST-FIX closure" listing your 11 plan items as CLOSED or STILL OPEN. End with exactly one line: "VERDICT: SIGN-OFF" or "VERDICT: AGREE WITH CHANGES" or "VERDICT: NO SIGN-OFF". Be concrete and brief; do not restate the plan back to me.

## Addendum: the D6 smoke timings (for question 7)

All sixteen configurations trained seed 7 for one epoch on the laptop (MPS),
exit 0, 4/4 artefacts each. Wall seconds (load + one epoch): mlp 6.5 · lstm 6.9
· residual_mlp 7.4 · full 8.2 · aligned_hist 8.7 · fixed_decay 10.0 ·
same_entity_k0 11.0 · same_entity_k30 11.0 · fox 11.1 · same_entity_unr 11.1 ·
recency_k30 11.2 · residual_t1 11.4 · aligned_hist_rf 13.9 · same_entity_k6
14.2 · same_entity_k12 14.4 · **xlstm 62.7**.

Stage 1 reference for extrapolation: a registered 30-epoch run with patience 3
early-stopped between epochs 10 and 28 and took 53-93 s (token MLP) or
122-226 s (full T1) per seed on this laptop; the mini is assumed 2-3x slower.
The artifact-free test suite on this tree: 1458 passed, 54 deselected, 0 failed.

Question 7 therefore includes: what `expected_hours` should each of the sixteen
jobs carry (the per-job timeout is 2x that value, so too small a number kills a
legitimate job and too large a number lets a hung job burn the night), and is
the whole sixteen-job, two-seed night realistic on the mini in one night, or
should the orchestrator split it and tell the user which jobs will not run?
