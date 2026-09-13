# Handoff to Codex Astra — orchestrator of the sequence and embeddings track (2026-09-13)

You are Codex Astra (`gpt-6-astra`). From this point you **orchestrate** the
sequence and embeddings track in `/Users/aryamangupta/CricML/Match_Prediction`,
branch `embeddings-ladder`. Until now you reviewed every stage read-only and
Claude (Fable, then Opus) executed. That chain is inverted: you plan, assign,
execute or delegate, and the user talks to you directly. Claude is no longer in
the loop unless the user brings it back.

Read, in this order, before doing anything: `CLAUDE.md` (all of it),
`docs/SEQUENCE_TRACK_PLAN.md`, `docs/sequence_track/rollout_protocol.md`,
`research/reports/embeddings/README.md` items 13–18, and the three most recent
reports it points to. Then this file again.

## 1. House rules that do not change

- Python runs through `uv run --no-sync python`. Dependencies through `uv add`.
- Commits are authored solely by the user: summary plus body, **no**
  `Co-Authored-By`, no session trailers. Never rebase, squash or force-push.
  The branch is `embeddings-ladder`; `main` is untouched until Stage 6.
- **Never read** `data/golden/`, `data/forward_holdout/`,
  `data/xgb_data_i7/test.parquet`, or `models/embeddings/seq_stage2/cohort/`.
  The untouched cohort is `DEFERRED_UNOPENED` on your own ruling.
- Every experiment is registered in a frozen YAML **before** any result is
  read; consumed configs are never edited, they are superseded (`predecessor:`).
- Statistics: `tournament_time_block_v1`; teacher-forced screens use 2,000
  replicates at seed 29 (`stage2_stats.py`), rollout ledger rows use the
  claim gate's 10,000 at seed 42 (`analysis_v1`); Holm within a registered
  family, `<10 blocks = descriptive`, five complete paired
  seeds for a family, 4/5 favourable per-seed directions on the primary.
- Verdicts go only through `research/log_verdict.py` with a gate JSON.
  Teacher-forced validation screens are DESCRIPTIVE by construction. Rollout
  contrasts use `claim_gate --kind match_model` on `odds_iteration_v2` and can
  be PROMISING (one seed) or LANDED (five seeds, non-provisional). Provisional
  evidence is never LANDED (invariant 9).
- Frozen reports are never edited; append errata. Living docs (`CLAUDE.md`
  lookup row, `TODO.md`, `IMPROVEMENTS.md`, the plan, the embeddings README)
  are updated at each stage close.
- **Framing the user wants:** lead with the finding. Never open with "no arm
  advances" or "stage failed". The ledger word FAILED means *not promoted*.
- Every stage records an acceptance file in `docs/sequence_track/` with
  checks written before the work and results filled in after; the review that
  used to be yours must now be done by someone other than the implementer
  (Claude on request, or a second Codex session in read-only mode).

## 2. Machines and tooling facts

- Laptop: Apple M5 Pro, 48 GB, 15 cores. Mini: `ssh mac-mini`, Apple M4,
  16 GB, 10 cores, ~22 GB disk; repo at the same path; git remote `mini`.
- Multi-seed training splits **by seed**: laptop 7, 29, 42; mini 13, 101. Every
  arm on both machines. Never split an arm across machines.
- Queue runner: `research/sequence_track/run_queue.sh --queue <yaml> --machine
  <laptop|mini>`; markers COMPLETE / FAILED / TIMEOUT / KILLED_MEMORY /
  REFUSED_MEMORY in each output dir; STOP file honoured between jobs.
- `uv` is not on the PATH for non-interactive `ssh mac-mini`; `export
  PATH="$HOME/.local/bin:$PATH"` first, and scp a launch script rather than
  quoting `nohup` through ssh. macOS openrsync: no `***`, use `--stats`; pass
  `</dev/null` to rsync inside `while read`. `pgrep -f x | wc -l`, never
  `pgrep -c`. zsh aborts on an empty glob: use `find`.
- Simulator cost measured in Stage 1 (255 fixtures × 1,600 sims, ten
  one-thread shards, laptop): token-MLP arm ≈ 41 min, transformer arm ≈ 142
  min, production XGBoost ≈ 37 min. Residual and 114-feature serving costs
  are unmeasured. The mini is not qualified for ledger rollouts until the
  protocol § 2.8 cross-machine check passes. MPS kernels are not bit-reproducible;
  every checkpoint records `mps_bit_reproducible: false`.
- Both machines may still have `caffeinate` running from night 3; check and
  kill what is not needed.

## 3. Where the track stands (ledger rows SQ1–SQ4, all logged)

| stage | status | finding |
|---|---|---|
| 1 | closed, SQ1 | in rollout on 50 features the transformer beats the token MLP (C−B −0.0257) and ties production; not promoted |
| 2 | closed, SQ2 DESCRIPTIVE | forgetting helps: fixed decay, xLSTM, production-residual control beat the no-memory MLP CI-clean 5/5; death-over harm reproduced; no mechanism isolated; `aligned_hist − full` withdrawn |
| 3a | closed, SQ3 | negative transfer not detected on two targets; P family not evaluable (6 blocks); equal-epoch replication backlog |
| 3b | not started | contract in `delphyne_mapping.md` § 2–4; TODO backlog |
| 3c | CLOSED | leakage audit: masked outcome recovered exactly from scoreboard/EB deltas |
| 4 refs | built | `models/embeddings/stage4/refs/` eb_ctx 1.4500, raw 1.4534, lin_50 1.4433 |
| 4d | closed, SQ4 | exposure + spread + recency null (−0.0000 [−0.0007, +0.0008]) |
| C114 | closed, SQ4 | on 114 features the token MLP (1.4277) beats the transformer (1.4306) CI-clean 0/5, teacher-forced; the 64 hand-built features are worth 0.0068 |
| 4b | re-registered | v1 runs refused by the driver; v2 config, driver fix and queues ready (§ 5) |
| 4a | audit only | `STAGE4_PAIR_GRAPH_AUDIT.md`: one component, 41% of validation pairs seen, 13% with an unseen player |
| 4c, 4e | not started | |
| cohort | DEFERRED_UNOPENED | condition 0 (§ 6) still open |

## 4. Your first job: the standing rollout ledger

The user's decision: **no more per-stage rollout reruns.** Every ball
checkpoint goes through one fixed simulator protocol once, its per-fixture
records are stored, and comparisons are computed from storage. One seed per
configuration first (lowest validation LL); five seeds only for a promotion
candidate. The design is `docs/sequence_track/rollout_protocol.md` (revision 3);
your round-1 to round-3 reviews are `docs/sequence_track/astra/
rollout_review_round{1,2,3}.md` and every item is folded in and tagged.
Build it in this order (your own dependency order from round 2):

1. **Acceptance first.** Write `docs/sequence_track/rollout_acceptance.md`
   with the checks for every step below before any code; register the
   families, decisions and expected outcome from protocol § 3 in it.
2. **Protocol generator and verifier**, developed early, hashes frozen last:
   `scripts/sequence_track/rollout_protocol.py --write/--verify` producing
   `experiments/configs/rollout_protocol_v1.yaml` from the Stage 1 pins with
   exactly Stage 1's simulation settings (clip [0.01, 0.99], 1,600 sims, base
   seed 20260910, cpu / threads 1 / prefix cache off, typed pins: extras
   graft sha256, run-out as a source-covered constant), the eligibility
   inventory `eligibility_v1.json` (252 / 167 with exclusion reasons),
   `analysis_v1` (claim gate, 10,000 seed-42 block resamples, Stage 1 reading
   rule), and the historical-import compatibility check. Freeze the engine
   and adapter source-closure hashes only after step 5.
3. **Ledger writer** (`entry.json`, `raw_sims.jsonl` retention, atomic
   completion, duplicate detection, fail-closed pairing against the
   eligibility inventory), `compare_ledger.py`, `ROLLOUT_LEDGER.md` renderer.
4. **Serving manifest and `serve_checkpoint.py`** as a capability registry
   (protocol § 2.1–2.2) with the three-part admission (§ 2.3). Certify the
   standard 50-feature `mlp` / `full` path first (1e-6 feature parity plus
   the lifecycle suite).
5. **Launcher generalisation.** `run_arm.py` admits only A / A50 / B / C and
   calls the Stage 1 verifier (`run_arm.py` ~790, ~950); add a ledger
   launcher that takes an entry's serving manifest and the protocol file,
   keeps the thread-cap-before-import ordering, shards ten ways, and merges
   with `merge_shards.py` **given the full cluster-source directory** so
   shard-local block ids are replaced (`merge_shards.py` ~728). Record the
   exact source config / output locations and the launch, merge and audit
   commands in the acceptance file.
6. **Import Stage 1's four runs** after the § 2.6 checks and after step 4
   has certified the path B and C were served on (A / A50 typed as boosters
   with a historical admission record).
7. **Capabilities**: `v7_114` (export the four encoders from the pinned
   frame, verify by parity), then `fixed_decay` (fix the
   no-positional-embedding `max_seq_len = 1` wrapper path). FoX and the
   production residuals are the second batch; xLSTM, masked/aligned, sidecar
   and identity-residual serving are registered as deferred. Implement the
   typed fixed-reference contrast against A in `analysis_v1` before any
   five-seed production reading.
8. **First batch**, one selected seed each (lowest validation LL, ties to the
   lowest seed, recorded before launch), laptop: `mlp_114`, `full_114`,
   `full_50`, `fixed_decay`. Time 114-feature serving on one shard before
   committing a night. Log one ledger row, idea id **SQ5**, gated on
   `family_rollout_c114`'s primary (`full_114 − mlp_114`, ≥$50k) with the
   other members as supporting gates; one-seed readings are PROMISING at
   most.

## 5. Rung 4b: launch tonight alongside the ledger build

Ready on disk (uncommitted at handoff unless the commit log says otherwise):
the driver fix in `scripts/sequence_track/retrain_stage2.py` (checkpoint
verification derives the expected parameter set from the arm), fixed-reference
support in `stage2_stats.py` (`statistics.fixed_references`), the frozen
config `experiments/configs/seq_stage3_batch2_4b_v2.yaml` registering
`family_4b_l3` and `family_4b_l2` against `ref_eb_ctx` **before** any v2 run,
output root `models/embeddings/seq_stage3/batch2_4b_v2/runs/`, and queues
`research/sequence_track/queue_{laptop,mini}_batch2_4b_v2.yaml`. Steps: run
the tests, smoke one epoch on each machine into a scratch dir, launch both
queues, rsync the mini's seeds home in the morning, consolidate, run
`stage2_stats.py` under the v2 config, render, write the acceptance, log SQ6.
The v1 runs stay excluded from evidence.

## 6. The cohort, in plain words

The "untouched cohort" is 471 men's T20 matches from 2026-04-17 to
2026-08-05, set aside as a final exam that no sequence-track model has been
trained on or judged against. Before it can be opened, two things must be
true and written down: (a) no training data came from that window — proven
(`stage2_cohort_consumer_audit.md`); (b) no *evaluation* in this repo ever
scored those matches at ball level — **not yet proven**. The match-level
golden set (124 matches) and the sealed forward set (137) fall inside that
window and were scored by the match model; they are excluded from the cohort,
but the proof that nothing else in the window was ever used as a test set has
not been written. Until (b) is written, the exam cannot be trusted as unseen.
The cohort is opened once, for every frozen candidate at once, and only for
a candidate that has cleared a rollout contrast. Do not spend it on a screen.

## 7. Stage 4 in full — what remains and how it was designed

From `docs/SEQUENCE_TRACK_PLAN.md` § Stage 4, order 4d → 4b → 4a → 4c → 4e,
each against its named control **and** the two frozen references.

- **4b identity residual** — § 5 above. Difference from the failed E4 rung:
  the base cannot move. Two λ arms, the family never selects between them.
- **4a interaction factorisation** — low-rank batter × bowler term on top of
  the frozen EB-linear model. The exposure-graph audit is done. Still to
  register: rank and shrinkage, centring of each factor against its EB
  marginal, zero fallback for unseen players, and separate reporting for
  seen pairs / unseen pairs of known players / unseen players (13.4% of
  validation balls). Ball LL first readout, then the ledger.
- **4c style vectors** (DeepCrease line/length/shot/control labels) — start
  with the coverage audit; player-grouped holdouts; context-residualised;
  provenance settled before anything leaves the repo. Control: EB outcome
  vectors. Not a ball-LL-first idea until coverage is known.
- **4e team-as-set readout** — a **match-level** gate through
  `scripts/experiment_harness.py` and `claim_gate` against `match_model_prod`
  with identical lineup information; pooled-EB controls; role- and
  order-aware pooling variants beside the permutation-invariant one. Separate
  workstream; does not touch the ball ledger.
- **4d follow-on** (unregistered half of 4d): Glicko-style or learned-variance
  ratings against a counts-only control. Low priority given the null.

Design records: `docs/sequence_track/night3_design_draft.md` § Block E,
`docs/sequence_track/batch2_acceptance.md` § "The 4b decision",
`scripts/sequence_track/stage4_references.py`, `stage4_exposure_sidecar.py`,
`stage4_guard.py` (validation-only guard: keep using it), `stage4_pair_graph_audit.py`.

## 8. Backlog (do not start without the user)

- 3a equal-epoch / early-stopped replication of the six night-3 arms.
- 3b over-level patching (contract in `delphyne_mapping.md`).
- Block A aligned history plus decay (`night3_design_draft.md` § Block A).
- Golden-set redefinition to 2026-08-06 onward (TODO.md).
- Stage 5 doc alignment and Stage 6 merge to `main`.

## 9. What the user expects from you each morning

One message: what ran, the finding first in plain words, what did not run and
why, the deviation list, the ledger rows logged, and the next launch. Numbers
in tables, not prose. Ask only for decisions that are the user's: verdict
labels, opening the cohort, promotion, anything that changes the protocol
version.
