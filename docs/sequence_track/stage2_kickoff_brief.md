# Stage 2 kickoff brief (sequence and embeddings track)

Written 2026-09-11 by the stage 1 orchestrator, for the agent that picks up
stage 2. Stage 1 is complete, reviewed to sign-off, committed (`4109fc1`)
and given its verdict (`84cc1a8`). Nothing of stage 2 has been started.

Read this file first, then `docs/SEQUENCE_TRACK_PLAN.md` § "Stage 2" (v5,
the plan of record and the only place the stage 2-6 design lives), then
`docs/sequence_track/stage1_acceptance.md` (every stage 1 check with its
recorded result) and `research/reports/embeddings/SEQ_STAGE1_REPORT.md`.

---

## 1. Your role and the rules

You orchestrate. Opus subagents implement logic; Sonnet does mechanical
work with a specified case list; you do small wording edits inline
yourself. No more than three agents at once. Codex Astra
(`gpt-6-astra`, low effort, read-only sandbox) reviews at the end of the
stage and before any commit that lands results; iterate rounds until
SIGN-OFF and record every round. You are the only one who commits and the
only one who calls `research/log_verdict.py`. The user decides every
advancement, every verdict, the arm priority, `k`, and the cohort question
in § 5.

Invoke Astra as:

```bash
codex exec -m gpt-6-astra -c model_reasoning_effort="low" -s read-only \
    "$(cat prompt.md)" </dev/null > out.md 2>err.md
```

Rules that override convenience:

- Never read or evaluate against `data/golden/` or `data/forward_holdout/`.
  Hashing them through `scripts/artifacts.py verify` is integrity checking
  and is fine; opening their contents in a model, scorer or runner is not.
  See § 5.3 for the one place this needs a decision.
- Never overwrite a production artifact or anything a manifest role points
  at. Stage 2 writes only under `models/embeddings/seq_stage2/` (and a
  mini-local scratch dir, § 4).
- **Stage 1 artifacts are frozen.** Do not modify anything under
  `models/embeddings/seq_stage1/`, the committed stage 1 report, its
  addendum, or `experiments/configs/seq_stage1_sim_v1.yaml`. Errata are new
  files.
- Frozen evidence and committed reports are never edited.
- Python only through `uv run --no-sync`. Never `pip`; `uv add` for deps.
- Report numbers verbatim from files. If a run did not finish, the number
  does not exist.
- Commit convention is in `CLAUDE.md`: summary plus body, **no
  AI-attribution trailers**. Read it before your first commit.
- Write acceptance checks before the work, in
  `docs/sequence_track/stage2_acceptance.md`, one table per deliverable
  with numbered check ids and pass conditions, and record results under
  each table. That is how stages 0 and 1 were run and it is what Astra
  reviews against.

---

## 2. What stage 2 is for

The 2026-08 ablation found that the full T1 transformer's gain over a token
MLP was −0.0004 with an interval crossing zero, and that full T1 **hurt the
death overs on 0 of 5 seeds** (+0.0053). Three hypotheses for the harm were
left open: stale history is noise at the death; the history belongs to
other batters and bowlers; one attention head serves a regime that changes
at the death.

**Stage 1 changed the picture and sharpened the question.** On the i7 frame,
in rollout through the simulator, full T1 beat the token MLP by −0.0257
[−0.0346, −0.0102] on the primary slice, and beat the 50-feature XGBoost by
−0.0286 [−0.0462, −0.0101]. So the history the transformer reads is
carrying something real. What is unknown is **which part of that history**,
and whether the death-over harm survives on this frame.

Stage 2's question, stated precisely: **does the death-over harm come from
stale history, from history that belongs to other players, or from
neither?** It is answered at ball level (teacher-forced validation log
loss), five seeds per arm, with each mechanism isolated by its own matched
control.

Two framing rules the plan fixes and you must not relax:

- **Mechanism is answered by direct contrasts, never by comparing
  significance.** "Arm X is significant and arm Y is not" is never treated
  as a difference. The two mechanism contrasts are FoX minus fixed-decay
  (learned forgetting) and same-entity minus recency-only (ownership beyond
  recency), each with its own paired interval.
- **If no sequence arm beats the MLP by a CI-clean margin**, the conclusion
  is that this model family, at this resolution, shows no further sequence
  gain. It is not a proof that the scoreboard summary is sufficient.

Stage 2 also functions as a partial confirmation of stage 1's headline: the
vanilla-T1-versus-token-MLP contrast is re-measured here on five seeds, at
ball level rather than in rollout. That is deliberate and worth saying in
the report.

---

## 3. What stage 1 built that you inherit

Everything below is committed and tested. Do not rebuild any of it.

| capability | where | what it does for you |
|---|---|---|
| Training on the i7 frame | `scripts/transformer_t1.py` | `--data-dir data/xgb_data_i7` works: the parquet stem comes from the frame's `.feature_hash` `version`, `--frame-version` overrides. `--no-kit` trains and scores validation without touching the v3-pinned eval kit. `--score-test` is off by default, so **selection never sees test rows unless you ask**. `--stats-cache-role` records and verifies cache identity. `metrics.json` carries a full `training_contract` (frame hashes, split rows and dates, feature names and sha, architecture, optimiser, seed, best epoch, device, versions, cache identity). `validation_ll` is stored unrounded (float64 mean) |
| Multi-seed retrain runner | `scripts/sequence_track/retrain_i7.py` + `experiments/configs/seq_stage1_retrain_i7_v1.yaml` | Runs (arm × seed) in order, validates the config, preflights frame and cache identity once, refuses to reuse a completed run whose effective config, frame declaration or parquet identity differs, records wall seconds and override provenance, writes `summary.yaml` with full-precision per-seed validation LL. **This is the model for your stage 2 training driver; extend or copy it, do not start from scratch** |
| Overnight queue runner | `research/sequence_track/run_queue.sh`, `queue.yaml`, `scripts/sequence_track/queue_lib.py` | Ordered jobs, skip-if-complete by config sha256, `caffeinate`, per-job timeout enforced by both a perl alarm and the watchdog, STOP file, tee'd logs, one retry, free-memory floor re-read before every attempt, process-group RSS watchdog. **Built in stage 0 for exactly this stage and never used.** `queue.yaml` still holds one placeholder job; replace it. Run with `--machine mini` |
| Serving guard | `scripts/sim_t1.py` | A checkpoint's `training_contract` is checked against the serving cache's `_meta` (delivery semantics, venue alias version, ordering version, feature order). Matters only if a stage 2 survivor goes to the simulator |
| Cluster-source fix | `scripts/sequence_track/run_arm.py --cluster-source-dir` | The I3 block id is stamped from the registered fixture set, not the run's own shard. Matters only for a rollout run |
| Gate and multiplicity | `scripts/sim_eval/claim_gate.py`, `scripts/sequence_track/holm_stage1.py` | The Holm script enforces the step-down rejection, reproduces the gate's interval to 1e-9 before forming p-values, and labels rank-local intervals as such. Stage 2's statistics are ball-level, not the match gate, so you will need a **ball-level** paired-interval tool; `scripts/registered_experiment.py` has `match_cluster_ci` and `seed_mean_match_cluster_ci`, which is what the ablation used |
| Shard consistency and merge | `scripts/sequence_track/shard_consistency_1c.py`, `merge_shards.py` | Only needed if you shard a rollout run |

The stage 1 model checkpoints live at
`models/embeddings/seq_stage1/retrain_i7/{mlp,full}/seed_{7,13,29,42,101}/`
(gitignored). Both arms selected seed 101 by validation LL. **These are the
i7-frame token MLP and vanilla T1 at five seeds already trained** — stage 2
can reuse them as its `token MLP` and `vanilla T1` arms if the training
config matches, which saves two of the eleven arms. Verify the contract
matches your stage 2 config before reusing; `retrain_i7.py`'s reuse check
does exactly this comparison and refuses on drift.

---

## 4. The machine, verified 2026-09-11

Stage 2 multi-seed training goes on the **Mac mini**, one job at a time,
overnight, per the plan's machine split. Checked today over `ssh mac-mini`:

| fact | value |
|---|---|
| host | `Aryamans-Mac-mini.local`, reachable over `ssh mac-mini`, `BatchMode` works (no password prompt) |
| memory / cores | 16 GB unified, 10 cores |
| free disk | **20 GB** on the data volume — the binding constraint |
| repo | `~/CricML/Match_Prediction`, on branch `embeddings-ladder` at **`dfa45ad`**, i.e. 6 commits behind the laptop; `daily/` is untracked there |
| torch | 2.5.1, **MPS available** |
| i7 frame | present and **byte-identical to the laptop**: train md5 `fac0b7be…`, validation `32643631…`, `.feature_hash` version `i7` |
| i7 stats cache | present, md5 `671ac820…`, identical to the laptop |
| `models/embeddings/` | **does not exist on the mini** — no eval kit, no ablation checkpoints, no stage 1 checkpoints |

Consequences:

- The frame and cache are already there and correct, so **no large data
  transfer is needed**. Sync the branch (`git push mini embeddings-ladder`
  from the laptop, or pull on the mini) so the mini has the stage 1
  trainer changes; that is code only.
- `--no-kit` is not optional on the mini: the eval kit is not there, and it
  is v3-pinned anyway.
- 20 GB free means: keep raw checkpoints in a mini-local scratch dir, keep
  only what you need per seed, and rsync summaries to the laptop each
  morning (`--stats`, not `--info=stats1`; macOS ships openrsync).
- Cap torch and BLAS threads at 4-6 on the mini and never run training and
  anything else at once. `run_queue.sh`'s memory floor and RSS watchdog are
  there for this.
- `codex exec` must run on the laptop, not the mini.

Laptop measurements from stage 1, for extrapolation (MPS, 30 epochs, i7
frame, `--no-kit`): token MLP **53-93 s per seed**, full T1 **122-226 s per
seed** (early stop fired between epochs 10 and 28). The mini is slower;
budget 2-3× and measure once before committing to a night. Eleven arms ×
5 seeds ≈ 55 runs; at 5 minutes per run that is under 5 hours, so the whole
stage plausibly fits one night, but the recurrent and masked arms are new
code with unknown cost.

---

## 5. The open decisions — put these to the user before building

### 5.1 Arm priority and how many arms in the first night

The plan registers **eleven arms**. That is a lot of new model code, and
several arms are only meaningful in pairs. A defensible first night is the
mechanism core, which answers the stage question:

`token MLP` (control) · `vanilla T1` (reference) · `recency-only mask`
(recency control) · `fixed-decay T1` (control for FoX's learned gate) ·
`FoX T1` (learned forgetting) · `participant-aligned history T1` (control
for aligned input alone) · `same-entity mask T1` (ownership)

That is seven arms, of which the first two may be reusable from stage 1.
The remaining four (`LSTM`, `xLSTM`, `residual T1`, `residual MLP`) test
different questions — classic forget gates, exponential gating, and
sequence over the production model — and can be a second night. **Put the
split to the user; do not decide it yourself.**

### 5.2 Reuse of the stage 1 checkpoints

The stage 1 `mlp` and `full` runs are five seeds each on the i7 frame with
the registered architecture. Reusing them as stage 2's control and
reference saves ~10 runs and makes stage 1 and stage 2 directly
comparable. The risk is that any change to the training config (epochs,
patience, batch, learning rate, early-stopping rule) silently invalidates
the comparison. `retrain_i7.py`'s reuse check catches exactly that. **Ask
the user; recommend reuse if and only if the stage 2 config is identical.**

### 5.3 The untouched cohort — a genuine precondition, and it needs care

The plan's statistical rules say: the ablation test split has been read by
the 2026-08 program and shaped the death-over hypothesis, so **stage 2
test-split results are exploratory**, and a confirmatory sequence claim
needs an untouched temporal cohort, **frozen with a hash before stage 2
starts**. The user's 2026-09-10 decision: the cohort is ball rows from
matches dated **2026-04-17 onward** (the i7 frame's `golden_start`), to
2026-08-05, excluding the sealed forward fixtures (2026-06-02 to
2026-07-13, 137 matches) and any match previously used for training,
tuning or inspected evaluation.

Two things to flag to the user:

1. **The ball-level "golden" region is not `data/golden/`.** The i7 frame's
   `.feature_hash` declares `golden_start: 2026-04-17`; that is a date
   boundary in the ball frame. `data/golden/` is the match-level Polymarket
   evaluation set and stays closed. Building the cohort means selecting
   cricsheet matches by date from the context corpus, not opening
   `data/golden/`. Say this explicitly in the acceptance file so nobody
   later reads a rule violation into it.
2. **Excluding the forward fixtures requires knowing their ids.** That is a
   directory listing, not an evaluation, and it is the same kind of
   integrity-only touch that `artifacts.py verify` makes. Get the user's
   explicit written note that an id-only read of
   `data/forward_holdout/` for exclusion purposes is permitted, and record
   it. If the user prefers, the exclusion can instead be done by date
   range, which is coarser but touches nothing.

The cohort is a precondition for *confirmatory* claims, not for running the
arms. A reasonable sequencing is: build and freeze the cohort while the
first night's training runs, then score the surviving arms on it. Put that
to the user.

### 5.4 Where the ownership dependency test runs

The plan requires a computational dependency test before `k = 0` may be
called "ownership": for a sample of 2,000 target positions, perturb the
features of every token outside the ownership set and assert the target's
output is unchanged to 1e-6. With two layers, allowed tokens can relay
excluded history, so the mask must be applied at **every** layer, and if
any dependence remains the arm is rebuilt (one layer, or per-layer masks
that also restrict what the allowed tokens themselves saw). This is fast
and interactive; run it on the laptop, and treat a failure as a rebuild,
not a footnote.

---

## 6. The eleven arms, from the plan

Every arm is described by four access columns: current 50 features /
shifted outcome history / identity relationships (which earlier tokens
share batter or bowler) / frozen production logits. Each addition has a
matched control.

| arm | features | history | identity | prod logits | what it tests |
|---|---|---|---|---|---|
| token MLP | yes | no | no | no | control |
| vanilla T1 | yes | yes | no | no | reference |
| recency-only mask (last k rows, any player) | yes | yes | no | no | control for the mask's recency component |
| fixed-decay T1 (ALiBi-style, no learned gate) | yes | yes | no | no | control for FoX's learned gate; identical to FoX in every other setting including no positional embedding, QK-norm, width, depth, dropout, optimiser |
| FoX T1 | yes | yes | no | no | learned data-dependent forgetting (per-head sigmoid gate, cumulative log-gate bias added before softmax, no positional embedding) |
| participant-aligned history T1 (no mask) | yes | yes, participant-aligned | yes | no | control for the aligned-history input alone |
| same-entity mask T1 | yes | yes, participant-aligned | yes | no | ownership |
| LSTM | yes | yes | no | no | classic forget gate |
| xLSTM | yes | yes | no | no | exponential gating |
| residual T1 | yes | yes | no | yes | sequence over the production model |
| residual MLP | yes | no | no | yes | control for the residual arm |

**Same-entity mask details.** Windows are in delivery rows (inclusive of
extras), not overs; innings reach 138-144 rows. `k = 30` rows is the
registered default; the sweep is `k ∈ {0, 6, 12, 30, unrestricted}`. In the
mask arm the history input is participant-aligned: the previous outcome of
the same batter, and of the same bowler, as separate embeddings, instead of
the innings-previous outcome. Window selection is on validation only:
choose the `k` with the best validation LL; **if no `k` beats `k = 30` by
more than 0.002, keep 30.** Only the chosen `k` is scored on test, and test
is exploratory.

**Residual arm.** `p = softmax(log p_base + r_theta)`, with `p_base` from
the production ball model floored at 1e-4, `r_theta` zero-initialised with
L2 shrinkage to zero, base logits produced temporally out-of-fold (the
production model is trained on rows before the validation split, so
validation and test logits are out-of-sample; train logits are refit
out-of-fold by date blocks). Residual T1's confirmatory contrast is
**residual T1 minus residual MLP** — beating the plain token MLP does not
qualify it, because that gain could come from the production logits alone.

**Recurrent arms.** Block composition frozen in the config: LSTM 2 layers,
hidden matched to T1's 128-d; xLSTM 2 blocks, sLSTM then mLSTM, 128-d.
State reset at innings start, full-innings backpropagation, equal tuning
budget (learning rate and dropout grid of the same size as T1's).
Parameters and compute reported.

**Not chosen, recorded:** Gated DeltaNet, Mamba-2, Titans. Linear-time
recurrence buys nothing at 120-144 tokens per innings, and FoX keeps the
existing T1 code and ablation arms.

**Where the code goes.** `scripts/transformer_t1.py` currently validates
`arm not in {"full", "mlp", "no_attention", "no_history"}` in `T1Model.__init__`
and branches on `self.arm` in `forward`. Every new arm needs to be added to
that set, to the argparse `--arm` choices, and to the forward branch. The
LSTM and xLSTM arms are not transformers at all and probably want their own
module rather than a branch in `T1Model`; decide that deliberately and say
so in the acceptance file.

---

## 7. How the arms are tested

| element | setting |
|---|---|
| lineage | `experiments/configs/t1_ablation_v1_mps.yaml` is the ancestor; write a new registered config for stage 2, do not edit that one |
| frame | `data/xgb_data_i7` (train 1,876,971 / validation 124,292 / test 186,667 rows), `--no-kit`, no `--score-test` during selection |
| seeds | 5 per arm: 7, 13, 29, 42, 101 |
| primary contrast | per arm vs the **token MLP** on the full validation split |
| uncertainty | seed-mean plus match bootstrap; `scripts/registered_experiment.py` `seed_mean_match_cluster_ci`; report seed spread and the 4/5-direction count alongside the seed-mean interval |
| confirmatory family | per candidate arm: the primary contrast vs its matched control, **plus** the death-over and chase non-inferiority gates, since those gates decide advancement. Holm-adjusted within the family |
| non-inferiority gates | upper confidence bound of the harm below **+0.002** on each of death overs and chases |
| exploratory | phase, innings, thin-player buckets, secondary slices — labelled exploratory in the report |
| mechanism contrasts | FoX − fixed-decay (learned forgetting); same-entity − recency-only (ownership beyond recency). Each with its own paired interval. Never infer a difference from one being significant and the other not |
| advancement | only an arm that beats the MLP CI-clean, passes **both** non-inferiority gates, and survives the untouched cohort goes to the simulator under the stage 1 protocol |

Two estimands are always reported separately: (i) the registered
checkpoint's own performance, and (ii) across-training-seed robustness.
Five seeds speak to (ii), never to (i).

---

## 8. Suggested deliverable structure

Write the acceptance table for each before the work starts.

- **2a — config and code.** A registered stage 2 config pinning every arm,
  frame, seeds, statistics and command lines, with a `--verify` pinning
  script in the `pin_stage1.py` mould; the new arm implementations; unit
  tests per arm asserting the access matrix holds (e.g. the no-history arm
  is invariant to shifted outcomes, the masked arm is invariant to tokens
  outside its set).
- **2b — smoke.** Every arm trains one seed for one or two epochs on the
  laptop, writes a contract and a validation LL, and the arm-access unit
  tests pass. No number is read as a result.
- **2c — dependency test.** The 2,000-position ownership certification,
  interactive, on the laptop, before `k = 0` is called ownership.
- **2d — overnight training.** The queue on the mini, one job at a time,
  summaries rsynced each morning. Record wall time and peak RSS per run.
- **2e — k sweep.** Validation-only selection over `k ∈ {0, 6, 12, 30,
  unrestricted}` with the +0.002 tie rule; the user reads the table and
  confirms.
- **2f — statistics, gate and report.** Paired seed-mean intervals, Holm
  within each arm's family, the two mechanism contrasts, the
  non-inferiority gates, then the untouched-cohort scoring for any arm that
  clears them. Astra to sign-off, then commit, then the verdict.

---

## 9. Inherited open items and traps

- `models/embeddings/eval_kit/` is pinned to the **v3** split (186,912-row
  masks) and does not align with the i7 test split. Use `--no-kit`. If you
  need unseen-pair diagnostics, rebuilding the kit on the i7 split is a
  separate, declared piece of work.
- The global outcome prior in every schema-v4 stats cache is summed over
  the whole corpus with no as-of cutoff. Shared exposure across arms,
  differential effect unmeasured. Backlog item in `TODO.md`; restate it in
  the stage 2 report's limitations rather than discovering it again.
- Venue history is not recency-weighted; same status, same backlog.
- `sim_t1.py` honours `OMP_NUM_THREADS` for its torch thread count since
  stage 1. If you run anything in the simulator, set threads deliberately;
  ten one-thread processes gave 6.4× the throughput of one four-thread
  process on the laptop.
- The stage 1 `deviations`, `known_asymmetries` and `known_limitations`
  blocks in `experiments/configs/seq_stage1_sim_v1.yaml` are the template
  for how stage 2 should record its own. Read them before writing the
  report; never silently drop one.
- `research/sequence_track/queue.yaml` still contains the stage 1
  placeholder job. Replace it; do not leave it in a night's queue.
- The mini is 6 commits behind. Sync before the first night.

---

## 10. What needs the user's explicit go

1. Arm priority and how many arms in the first night (§ 5.1).
2. Whether to reuse the stage 1 five-seed MLP and vanilla T1 checkpoints
   (§ 5.2).
3. The untouched-cohort construction, including the id-only read of the
   forward holdout for exclusion (§ 5.3).
4. The result of the ownership dependency test, before `k = 0` is called
   ownership (§ 5.4).
5. The `k` chosen from the validation sweep (§ 2e).
6. Every advancement decision and every verdict.

The user reads results in plain language, point by point, at the end of a
stage: what was tested, what came out, what it means for which model is
best, and what to do next. Write that section into the stage 2 report as
§ 10 of the stage 1 report does, and say it in chat as well.
