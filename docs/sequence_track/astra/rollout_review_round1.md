**MUST-FIX**

1. **The proposed protocol does not match Stage 1’s pins.** `docs/sequence_track/rollout_protocol.md:76` specifies clipping at `1e-4`; Stage 1 used `[0.01, 0.99]` (`experiments/configs/seq_stage1_sim_v1.yaml:112`). The draft’s 2,000 bootstrap replicates with seed 29 also differ from the pinned claim-gate calculation: 10,000 with seed 42 (`…yaml:3686`). Preserve Stage 1’s simulation settings for import. Version statistical calculations separately so changing an interval calculation requires rescoring stored outputs, not rerunning matches.

   `pin_stage1.py --verify` is not a generic protocol verifier: it compares the supplied file against `build_config()`’s Stage 1 schema (`scripts/sequence_track/pin_stage1.py:3111`). Specify a new verifier and an explicit historical-import compatibility check.

2. **“From its run directory alone” is currently false.** `docs/sequence_track/rollout_protocol.md:41` omits essential dependencies:
   - C114 stores category vocabulary **sizes**, not the category-to-code mappings (`scripts/transformer_t1.py:1952`). Its saved means and standard deviations must also be applied (`…py:1991`).
   - Residual checkpoints identify saved base predictions; those predictions cannot generate base probabilities for simulated states.
   - Configuration identity, overrides and training signatures require `run_record.json`; completion integrity requires the applicable `COMPLETE.json` checks (`scripts/sequence_track/retrain_stage2.py:2314`).

   Define a versioned serving manifest containing, or resolving by content hash, every dependency. Existing C114 encoders can be reconstructed deterministically from the pinned training frame and exported without retraining, then verified by parity. Do not silently substitute production encoders.

3. **Prediction parity cannot be the sole admission gate.** Replace `docs/sequence_track/rollout_protocol.md:58`–64 with three requirements: artifact/serving-contract verification, per-checkpoint prediction parity, and certification of the actual state-to-input serving path. Preserve D4 and the launcher’s actual cache-hash check (`scripts/sim_t1.py:89`; `scripts/sequence_track/run_arm.py:945`).

   The cited precedent tested **live feature parity at `1e-6`**, plus state/storage behavior; it was not simply a `1e-5` saved-probability comparison (`research/reports/embeddings/T1_SIM_PARITY_PPC_V1.md:21`). A checkpoint can reproduce saved predictions while its simulation feature builder, reset behavior or extras handling is wrong.

4. **Define reuse and ledger integrity precisely enough to prevent another rerun cycle.** `docs/sequence_track/rollout_protocol.md:35`, `:80`, `:88` needs:
   - An immutable simulation contract, separately versioned adapter implementations and analysis registrations.
   - A compatibility procedure for an adapter extension that leaves existing serving paths unchanged. Stage 1 hashes the entire source closure, including the adapter and runner (`experiments/configs/seq_stage1_sim_v1.yaml:4024`).
   - Explicit retention of `raw_sims.jsonl`. Evaluator summaries alone cannot support the promised realism and prop analyses (`scripts/sequence_track/run_arm.py:38`).
   - Result hashes, atomic completion, resumable failed shards, duplicate detection and fail-closed fixture pairing.
   - Artifact identities covering preprocessing/base models, with typed identities for A/A50 rather than assuming every entry has `model.pt`.

   Promise “once per checkpoint and compatible simulation contract,” with explicit invalidation for defects. “Nothing is ever re-simulated” is unsupportable.

5. **The promotion rule is incomplete and conflicts with the registered comparisons.** `docs/sequence_track/rollout_protocol.md:117` requires beating B, but several proposed families contain no candidate-versus-B comparison. Beating B is also insufficient for production advancement: Stage 1 explicitly required improvement against A (`experiments/configs/seq_stage1_sim_v1.yaml:3657`).

   Register the exact contrast, reference, slice, multiplicity treatment, effect threshold and decision for each candidate. Preserve or explicitly amend the `0.007` practical threshold. For five-seed comparisons, specify all five seeds of each learned comparator, paired seed/block resampling, seed spread and the 4/5 direction count (`docs/SEQUENCE_TRACK_PLAN.md:67`). Reuse already completed seed rows.

   Separate a favourable research screen from the actual `claim_gate` verdict. `match_model` also checks profit harm; five seeds plus a cohort read do not automatically confer LANDED (`scripts/sim_eval/claim_gate.py:362`).

6. **The C114 family omits a central question, and the expected-outcome statement misidentifies the seed.** At `docs/sequence_track/rollout_protocol.md:147`, register `full_114 − full_50` if the batch is intended to resolve the production-contract question. Add `mlp_114 − A` if the stronger teacher-forced arm is a potential production candidate.

   At `:138`, “new seed” is incorrect under the proposed selection rule: batch-2 `full_50` selects **101**, and imported Stage 1 C also uses **101** (`SEQ_STAGE3_BATCH2_REPORT.md:239`; `seq_stage1_sim_v1.yaml:3028`). Describe this as a fresh checkpoint under the batch-2 training run, subject to checking its training signature and artifact identity—not an independent training-seed replication.

7. **Correct the motivating evidence.** `docs/sequence_track/rollout_protocol.md:25` makes stage-specific storage sound like mathematical non-comparability; compatible stored fixture outputs can already be compared. The missing piece is a standing compatibility and reuse policy.

   At `:30`, C114 is not a second observed teacher-forced/rollout disagreement. Its report explicitly says no matches were simulated and that comparison with Stage 1 does not establish a contradiction (`research/reports/embeddings/SEQ_STAGE3_BATCH2_REPORT.md:262`, `:322`). State the honest motivation: teacher-forced scores do not answer the rollout question, which remains unmeasured for these checkpoints.

**SHOULD**

1. **Make support an explicit capability registry.** `docs/sequence_track/rollout_protocol.md:47` groups materially different serving paths and promises “any” checkpoint while omitting capabilities such as tier conditioning and auxiliary-head loading. Register supported combinations of wiring, history, features and sidecars; refuse unsupported combinations. Reuse the training model implementations rather than implementing another attention stack.

2. **Register machine qualification before moving arms.** `docs/sequence_track/rollout_protocol.md:175`: recording “mini” and using one thread does not prove equivalence. Pin runtime versions and numerical settings, then run a small cross-machine comparison of predictions and fixture outputs before qualifying that machine. Stage 1’s serial/shard result established equivalence on five boundary cases, not across hardware (`SEQ_STAGE1_REPORT.md:264`).

3. **Carry forward the simulation-noise limitation and qualify runtime estimates.** `docs/sequence_track/rollout_protocol.md:75`, `:133`: 1,600 simulations was an operational compromise extrapolated from earlier measurements, not a measured universal convergence guarantee (`SEQ_STAGE1_REPORT.md:219`). Keep it for the directional batch, distinguish simulation noise from training-seed spread, and predeclare how a borderline result is handled. The 45-minute/2.5-hour figures are measurements for B/C, not established costs for residual or 114-feature serving.

**NOTE**

1. **A was already a 114-feature model.** Correct the Stage 1 table’s blanket “50 features” description (`docs/sequence_track/rollout_protocol.md:15`; `experiments/configs/seq_stage1_sim_v1.yaml:36`).

2. **Retain inherited limitations in ledger reports.** In particular, the shared global prior was not as-of-date, and its differential effect across arms was unknown (`experiments/configs/seq_stage1_sim_v1.yaml:3922`). Importing results must not erase that qualification.

3. **This was a read-only design review.** I inspected the working tree on `embeddings-ladder`, made no changes, ran no simulations, and did not open the excluded directories. Import eligibility remains to be verified against the actual output artifacts.

**Direct answers to questions 1–7**

1. **Does the root-cause analysis match the code?**

   Broadly, the lack of a general serving interface is real. Several details make this more than checkpoint loading:

   - **Features from simulated state:** T1 combines live same-day counts with simulated player updates; venue distributions remain pre-match (`scripts/sim_t1.py:282`). Historical pre-ball features cannot be reconstructed from outcome history alone (`…py:604`). A universal builder must preserve those distinct update rules.
   - **Serving guard:** preserve delivery semantics, venue identity, ordered features and cache identity. The current guard checks cache-hash presence; the launcher establishes equality. Generalize feature-contract validation rather than removing it.
   - **Hidden capacity bug:** current `T1Model` can instantiate fixed-decay/FoX and recurrent models, but the wrapper assigns `max_seq_len = 1` when there is no positional embedding (`scripts/sim_t1.py:494`). These arms deliberately lack that embedding (`scripts/transformer_t1.py:239`). Some checkpoints can load and still fail on their second delivery.
   - **Prefix caches:** Stage 1 explicitly disabled them (`run_arm.py:1103`). The existing incremental implementation mirrors ordinary transformer layers and is restricted to `full`/`no_history`; it is not a general attention cache.
   - **Thread caps:** set BLAS/OMP caps before heavy imports and preserve Torch’s effective cap (`run_arm.py:151`). A launcher refactor must preserve this ordering.
   - **Residuals:** obtain raw production probabilities on the residual arm’s current state, apply the training class permutation, floor, normalization and logarithm, then add the residual. Training uses `log(p_base)`, not an arbitrary postprocessed delivery distribution (`build_base_logits.py:154`).
   - **114 features:** reproduce all numeric formulas, training encoders, missing-value handling and saved standardization. Production’s feature builder is useful infrastructure, not proof of training parity.
   - **Masked/aligned inputs:** capture the participants before each delivery. Aligned history and ownership masks are different mechanisms; relay-free arms additionally require previous rows’ **own outcomes** (`transformer_t1.py:787`, `:1268`). Windows count delivery rows, including extras.
   - **Recurrence:** there is no exposed persistent-step interface. LSTM discards returned state; both xLSTM cells initialize state inside each full forward (`recurrent_arms.py:153`, `:211`, `:331`). Efficient stepping needs implementation and equivalence checks.
   - **Launcher:** `run_arm.py` admits only A/A50/B/C and binds launches to Stage 1 arm blocks. Returning a `PredictionModel` alone does not make a new arm launchable.

2. **Is per-checkpoint teacher-forced parity sufficient? What should change?**

   **Necessary, but insufficient.** Keep full validation prediction parity, with explicit absolute tolerance, finite normalized six-class probabilities, class order, validation-file identity and row alignment. The saved archive contains `probs`, `y` and `innings_id`, so preserve within-innings row order rather than joining on innings alone (`transformer_t1.py:2508`).

   Add serving-feature parity and a small reusable lifecycle suite covering opening extras, strike/bowler changes, wickets, innings transitions, new simulations, repeated/skipped calls, copied states, long innings, unknown identities and same-day cutoffs. Certify extras composition separately from six-class prediction parity.

   For recurrent or optimized attention implementations, compare stepped predictions with full-prefix predictions. Drop the claim that checkpoint parity replaces all adapter review: certify a serving implementation once, then automate admission for each checkpoint.

3. **Is Stage 1 import legitimate, and what must be verified?**

   **Yes, conditionally.** Verify:

   - Exact A/A50 boosters and dependencies; B/C checkpoint hashes and seed-101 selection provenance.
   - Original config hash, source closure and runtime, using historical source verification where appropriate—not pretending today’s source files produced old outputs.
   - Exact fixture inventory and eligibility, odds identity/labels, replay cutoffs, cache, context, metadata, selector, usage/roster artifacts, extras, run-out setting, clipping, seeds and simulation count.
   - Ten disjoint shard inventories whose union is the registered set; successful merge and audit; unchanged tournament-block identities.
   - Per-fixture agreement between evaluator records, provenance and raw simulations, with no duplicates or unexplained omissions.
   - Original result/analysis registrations and limitations.

   The report has 255 fixtures but **252 scored all-slice records**, and 167 on the primary slice; import must preserve the documented eligibility differences, not force 255 scored rows. Give production imports an appropriate historical admission record instead of inventing neural prediction-parity results.

4. **Any objection to one selected seed, then five for a candidate?**

   **No objection.** It answers the user’s cost concern. Select from a fixed, complete seed roster using full-precision validation LL, ties to the lowest seed, and record the selection before rollout. This estimates the selected checkpoint’s performance; it does not represent an average seed.

   The trigger is **not tight enough**. Define research-follow-up eligibility against the registered scientific control, and production eligibility against A. Require the registered adjusted interval/effect rule and relevant operational guards. Five-seed expansion should add only missing checkpoint runs; learned controls also need the registered five-seed coverage. The iteration set remains selection evidence even after five seeds.

5. **Are the first-batch families and anchors right? Is the expected outcome honest?**

   The three topics are sensible, but the registrations need changes:

   - **C114:** retain `full_114 − mlp_114`; add `full_114 − full_50`. Treat `mlp_114 − B` as a comparison of these specific systems unless training differences are controlled. Add the relevant production comparisons before using them to advance candidates.
   - **Forgetting:** `fox − fixed_decay` is a useful comparison. `fixed_decay − C` is usable as a selected-system contrast; a mechanism claim needs a control with verified matching training settings. `xlstm − B` compares architectures, not forgetting alone. Add `xlstm − lstm` if the question concerns xLSTM’s particular recurrence.
   - **Residual:** retain both proposed contrasts; add `residual_t1 − A` if that arm can be advanced.

   A and B are useful standing anchors, but neither replaces every matched control. Splitting tests into three families must reflect distinct decisions; an “advance whichever wins anywhere” decision needs multiplicity treatment across that selection.

   “May or may not survive rollout” is honest uncertainty. Correct the fresh-seed claim and acknowledge that the hypotheses are informed by earlier results. “No arm advances on one seed” is appropriate.

6. **Answers to the five open questions in §5**

   1. **Keep D4 and actual cache-hash verification**, plus the additional serving-path checks above.
   2. **Use a short full-prefix-versus-step test for early debugging**, but do not substitute it for full per-checkpoint prediction parity. Full-prefix replay is a cheaper engineering implementation than immediately optimizing recurrence.
   3. **Sharing a loaded production scorer is acceptable within a process.** Share a computed result only for the exact same pre-ball state and dependencies. A’s independently simulated trajectory cannot supply another arm’s logits. Keep mutable feature buffers and caches isolated; `XGBoostModelV2.extract_features` returns a reused, non-thread-safe buffer (`sim_v1_2.py:1619`).
   4. **Allow the mini after qualification**, not merely after recording its name. Whole-arm placement is a sensible initial operating rule, but it does not establish numerical comparability by itself.
   5. **Leave T4 results in place and link them as historical evidence.** A separate historical index is optional; importing them adds no comparable anchor to this batch.

7. **Engineering estimate and deferrals**

   Estimates are hands-on engineering hours, including focused correctness checks, excluding training and full rollout wall time. Family estimates are incremental after shared infrastructure.

   | Component/family | Hours | Recommendation |
   |---|---:|---|
   | Shared serving manifest, launcher, admission harness, ledger/import checks | 16–24 | First |
   | Standard 50-feature MLP/full/no-history | 4–6 | First |
   | C114 encoder export, numeric feature parity, scaling; MLP/full | 20–36 | First priority |
   | Fixed-decay | 4–8 | First batch |
   | FoX, using full-prefix inference | 4–8 | After fixed-decay |
   | Participant-aligned standard history | 6–10 | Later |
   | Relay-free/aligned-RF/ownership-window family | 12–20 | Defer initially |
   | LSTM with persistent stepping | 6–10 | Before xLSTM if needed |
   | xLSTM with persistent stepping | 12–20 | Defer initially |
   | Production residual MLP/T1 pair | 12–20 | Second batch |
   | Exposure counts/spread/recency | 8–14 | Defer |
   | Identity-residual family and frozen-reference serving | 12–20 | Defer until admissible training artifacts exist |

   I would first deliver Stage 1 imports, C114 and fixed-decay, then add FoX and production residuals. The larger costs are feature fidelity and artifact completeness, not model dispatch. This sequencing should not change registrations after seeing results; register deferred work as deferred.

This design can stop most unnecessary repeat runs: save each model’s simulated matches once, then answer later questions from those saved matches. It cannot promise that a broken simulator, changed match rules or a genuinely new model will never need another run. With the changes above, the normal process becomes what you asked for: test one chosen version of each model, save everything useful, and test the other four versions only when the first result justifies it.

VERDICT: AGREE WITH CHANGES
