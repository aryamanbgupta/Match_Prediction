# Sequence and embeddings track plan (2026-09-10)

Status: v5, APPROVED by the user 2026-09-10 after three Codex Astra
(gpt-6-astra, low) review rounds; round-1 findings are marked "[A<n>]",
round-2 residuals "[A<n>.2]", round-3 residuals "[A<n>.3]". Astra
sign-off: stage 0 yes; stage 1 execution conditional on stage 0 checks
passing; stage 2 cleared after [A25.3]. Implementation starts with a
fresh agent on stage 0 (see "Stage 0 kickoff brief"). Nothing has
run. This file is the to-do for the remaining research on branch
`embeddings-ladder`. It sits beside the remediation plan
(`docs/REMEDIATION_PLAN_2026-09-09.md`), which owns engineering items 1-9;
this file owns the modelling questions the 2026-08 program left open. A
pointer from `TODO.md` is added at stage 5.

Everything here obeys the branch's standing rules: identical information
across arms, five seeds, paired match-clustered (or seed-mean plus match)
intervals, registered configs before results, no golden or forward holdout
read, no post-hoc calibration, verdicts only through `claim_gate` and
`log_verdict`.

## Why this plan exists

The 2026-08 program closed with three findings that this plan re-opens on
purpose (`research/reports/embeddings/PROGRAM_STATUS_20260813.md`):

- Every embedding rung lost to a linear model on the same EB inputs. The
  closest, E4, was 0.010 test LL behind, single seed. The rungs were all
  "learn who the player is"; none tried interactions, frozen-base
  residuals, uncertainty, or non-outcome data. [A22] E1-E4 refute those
  tested implementations, not the stage 4 proposals, each of which states
  below exactly how it differs from the failed rung.
- The T1 transformer's gain over logistic was almost entirely captured by a
  token MLP with no sequence memory (MLP-logistic -0.0047 CI-clean; full
  T1-MLP -0.0004, interval crosses zero). Full T1 hurt death overs on 0/5
  seeds (+0.0053). Hypotheses for the harm: stale history is noise at the
  death; the history belongs to other batters and bowlers; one head serves
  a regime that changes at the death; teacher-forced evaluation hides
  rollout drift.
- T4 (T1 through the simulator) failed at 0.7156 winner LL, but on a
  broken engine with a random bowler selector, 50 sims, and an odds loader
  that later turned out to place zero bets. Under matched random selectors
  T1 (0.7176) and the production ball model (0.7199) were at parity on
  >=$50k. No fair post-repair run exists.

Dropped from consideration, with reason:

- "Borrow from other formats" for cold-start players: most T20 debutants
  have no other professional history, so the prior would be empty where it
  is needed (user decision 2026-09-10).

## Statistical rules for the whole track [A25, A26, A27, A15]

- Hypothesis families are registered per stage in a family map (one
  table per stage naming every confirmatory contrast, its reference arm,
  its slice, and the advancement decision it feeds). Stage 1's family has
  three contrasts (C-B, B-A, C-A); stage 2's has, per candidate arm, the
  primary contrast vs its matched control plus the two death-over and
  chase non-inferiority gates, since those gates decide advancement and
  so belong in the confirmatory family [A25.3]; everything else (phase, innings, thin-player
  buckets, prop families, secondary slices) is exploratory and labelled so
  in the report. Confirmatory claims within a family use Holm-adjusted
  paired intervals. Untouched-cohort confirmation is required for any
  confirmatory claim in stages 2-4, not only stage 2 [A25.2].
- Sign convention [A10.2]: every contrast is candidate minus reference,
  in log loss, so a negative delta is favourable. Stage 1 references: A
  for B-A and C-A, B for C-B.
- Seed rule: five training seeds; report seed spread and the 4/5-direction
  count alongside the seed-mean interval. Two distinct estimands are
  always reported separately [A26.2]: (i) the registered checkpoint's own
  performance, with uncertainty from paired tournament-block resampling
  of matches and independent simulation batches; (ii) across-training-seed
  robustness, from paired resampling over seeds and tournament blocks
  jointly. Five seeds speak to (ii), never to (i).
- The ablation test split has been read by the 2026-08 program and shaped
  the death-over hypothesis, so stage 2 test-split results are exploratory.
  Confirmatory sequence claims need an untouched temporal cohort: ball rows
  from matches dated after 2026-04-16, excluding golden and sealed forward
  fixtures, and additionally excluding any match previously used for
  training, tuning, or inspected evaluation by any model or pipeline on the
  branch; an eligibility audit script lists the surviving matches with the
  reason each excluded match was dropped, and the cohort is frozen with a
  hash before stage 2 starts. **User decision 2026-09-10:** the cohort
  is built from matches dated 2026-04-17 onward (the current ball-frame
  golden split, 471 matches to 2026-08-05), and the match-level golden
  window is reset to start after that data ends (fresh cricsheet and
  Polymarket captures, sealed under the daily protocol of remediation
  item 3). Open caveat for the user: the sealed forward fixtures
  (2026-06-02 to 2026-07-13, 137 matches) were scored at match level and
  the forward rule forbids using them for candidate selection; the plan's
  default excludes them from the ball cohort, leaving 2026-04-17 to
  2026-06-01 plus non-forward matches to 2026-08-05. Including them is a
  separate written exception. [A15, new-2]
- Ball LL is a screening metric. It has no fixed conversion to winner-market
  value; a market claim needs a paired winner-market evaluation through the
  claim gate with registered cost scenarios. [A27]

## Roles and execution model (user decision 2026-09-10)

- **Fable orchestrates.** Writes each stage's acceptance checks before
  work starts, assigns jobs, reads every diff and every result, decides
  what goes to the queue and what stays interactive, and is the only one
  who calls `log_verdict`.
- **Opus executes.** Implements code, configs, and runs under Fable's
  brief; never writes verdicts, never edits frozen evidence, program.md,
  or the research ledgers.
- **Astra reviews.** Codex `gpt-6-astra` (low effort, read-only sandbox)
  reviews at the end of every stage and before any commit that lands
  stage results, following the same round-until-sign-off protocol used on
  this plan. A stage is not closed without its Astra sign-off recorded in
  the stage report.
- **Overnight queue, not an agent loop.** Every arm is pre-registered, so
  unattended work runs through a deterministic queue runner
  (`research/sequence_track/run_queue.sh` + `queue.yaml`): jobs in order,
  each with config, command, output dir, expected hours, and machine tag;
  skip-if-complete by config hash so a killed night resumes; caffeinate,
  perl-alarm per-job timeout, STOP file, tee'd logs, retry once. Two
  guards the auto-research runner lacks: a free-memory check before each
  launch and a resident-memory watchdog that kills a job above a cap. No
  model in the loop, no verdicts, no git. Building the runner and its two
  guards is the first stage 0 task.
- **What stays interactive:** stage 0 pinning, the 1a smoke and 1b timing
  reads, choosing the simulation count from the convergence table, the
  ownership dependency test result, k selection, every advancement
  decision, every verdict, every Astra round.

Machine split (laptop 48 GB / 15 cores; Mac mini 16 GB unified / 10
cores / ~22 GB disk free):

| work | machine | reason |
|---|---|---|
| stage 0 checks, A50 retrain, config pinning | laptop, interactive | short, human reads output |
| stage 1 smoke, timing, shard checks | laptop, interactive | human chooses the sim count |
| stage 1 full run, all arms | laptop, 4-6 shards, may run overnight | one run stays on one machine so shard outputs are bit-identical |
| stage 2-4 multi-seed training | mini, one job at a time, overnight queue | serial, low memory, no interaction |
| stage 2 k-sweep reads, dependency test, bootstrap, gate | laptop | fast, human reads |
| any simulator pass for a survivor | laptop | as stage 1 |

Mini rules: one job at a time; never training and simulation together;
torch and BLAS threads capped at 4-6; raw checkpoints to a mini-local
scratch dir keeping only the best checkpoint per seed; summaries rsynced
to the laptop each morning (`--stats`, not `--info`; see memory notes).
Disk, not memory, is the mini's binding limit. The first mini job is one
seed of vanilla T1 timed end to end, which sets how many arms fit a night.

## Stage 0: readiness gate before any simulator run [A1, A2, A4]

All of the following are recorded in the stage 1 config with hashes before
1a runs:

- Queue runner `research/sequence_track/run_queue.sh`, `queue.yaml`
  schema, the memory pre-check and watchdog, and a dry-run test on a
  10-second dummy job (skip-if-complete, timeout kill, STOP file, resume).

- `uv run --no-sync python scripts/artifacts.py verify` passes; BR2 gates
  are green or carry a written exception (currently: prop A/B, G1, G3 pass;
  G5 by exception). T1 replay lifecycle, snapshot guard and parity tests
  pass on this checkout. `claim_gate` and `log_verdict` are exercised end
  to end on a synthetic paired pair and the HA0 swap-smoke evidence before
  1a, so a stage 1 verdict cannot be the first time they run [A1.2].
- Cross-arm audit [A3.2]: for every fixture, the prediction-time cutoff
  (state as-of date, tracker walk end) and eligibility (odds present,
  resolved, male T20) are identical across A, B and C; a script asserts
  the three arms' fixture sets and as-of stamps are equal before scoring.
- Exact artifacts per arm: model checkpoint directory and hash for B and C
  (the T1 runner otherwise defaults to `models/embeddings/t1`), stats
  version and cache hash, context corpus (`--t1-context-dir`), player
  metadata csv hash, bowler usage / B10 corpus / roster policy files,
  extras graft sidecar (B18 is active in `sim_t1.py` only through
  `T1_EXTRAS_GRAFT_PATH`; the same sidecar must be active for A), run-out
  sidecar, odds role `odds_iteration_v2`, prop fair-baseline corpus. All
  three arms must produce the same delivery composition, first-over
  routing, and fallback behaviour; a pre-run assertion checks selector and
  sidecar hashes match across arms.
- Complete command lines for each arm, committed.

## Stage 1: three ball models through the fixed simulator

**COMPLETE 2026-09-11. Outcome: the transformer wins at equal information
(beats the token MLP and the 50-feature XGBoost CI-clean in rollout) and
ties production without the 64 hand-built history features. Priority
follow-up: C114, the transformer on the full 114-feature production set.**
(Under the registered promote-to-production rule nothing was promoted;
that rule is about replacing production, not about whether the finding is real.) Commits `504c17f` (one i7 cache, B and C retrained),
`4109fc1` (1b/1c/1d, gate, report), `84cc1a8` (verdict). Ledger `SQ1
FAILED` = advancement not established, not inferiority. Primary slice
>=$50k, 18 blocks, Holm step-down: C-B -0.0257 [-0.0346, -0.0102]
favourable; B-A +0.0247 inconclusive; C-A -0.0009 [-0.0277, +0.0288]
inconclusive; A50-A +0.0276 [+0.0095, +0.0419] adverse (exploratory);
C-A50 -0.0286 [-0.0462, -0.0101] favourable (post-hoc, addendum). Every
arm is a single checkpoint, so every gate is provisional and no market
claim is made. Report `research/reports/embeddings/SEQ_STAGE1_REPORT.md`;
acceptance `docs/sequence_track/stage1_acceptance.md` (D1-D12); Astra
signed off in two end-of-stage rounds. The pre-registered expectation was
parity everywhere; C-B and C-A50 departed from it in the favourable
direction and survived the gate.

**User decision 2026-09-11, carried into stage 2:** the transformer is
treated as the better architecture at equal information, on the strength of
C-B and C-A50. The confirmation work (five-seed rollout repeat, matched
teacher-forced score of these checkpoints) is backlog in `TODO.md`, not a
precondition. Note that stage 2 re-measures the vanilla-T1-versus-token-MLP
question at ball level on five seeds by its own design, so it is itself a
partial confirmation.


Question, restated precisely [A3, A10]: (i) system comparison: does the
50-feature MLP or T1 stack, on the certified replay path, simulate the
winner market as well as the production i7 ball model? (ii) incremental
sequence evidence: does C improve over B in rollout? Only (ii) speaks to
sequence value; (i) compares systems with different inputs and paths.

Arms, all in the same engine, same fixtures, same registered per-fixture
seeds:

| arm | model | role |
|---|---|---|
| A | `ball_model_prod` (i7 no-weights XGBoost, 114 features, manifest role) | production system |
| A50 | XGBoost, production config, trained on T1's 50 features only | production family at equal information (user request 2026-09-10) |
| B | token MLP, 50 features, registered checkpoint (training seed chosen from the ablation by validation LL only) | nonlinear control |
| C | full T1, 50 features, registered checkpoint (same seed-selection rule) | sequence candidate |
| C114 | full T1 given the production 114 features (optional, after A50) | sequence beyond hand-built history |

Feature-set note (measured 2026-09-10): T1's 50 features are a strict
subset of production's 114. The 64 production-only columns are: raw
career and recent averages/strike rates and economy (14), ELO and team
strength (9), pace/spin and left/right splits (8), player metadata such
as hand, arm, age (6), matchups (3), **hand-built history: last 5/10/30
ball runs, balls since boundary, recent dots, partnership, recent dot and
boundary percentages, pressure index (9)**, head-to-head (2), and basic
state such as inning index, wickets, balls in over, toss and batting-first
flags, venue code (10), plus lead gap and venue average score. So
production already carries hand-crafted sequence features that T1 was
never given; C114 tests whether learned sequence adds anything beyond
them. A50 makes A-vs-B/C a clean system comparison at equal information
and is cheap (one XGBoost retrain).

Stage 1 is a **single-checkpoint screen** [A5]: one training seed per
neural arm. A survivor is confirmed later on five training seeds and two
independent simulation batches before any claim.

Settings: iteration set `data/polymarket_test_v2` (255), odds
`odds_iteration_v2`, `EmpiricalBowlerSelector` with the corrected
roster-aware policy, B18 extras graft on all arms, no calibration, T1 and
MLP through `scripts/sim_eval/run_sim_eval_t1.py` (certified replay path),
prefix cache OFF, thread and device settings frozen for every run
including the full run [A8].

Monte Carlo resolution [A9]: 100 sims per match gives probability SE 0.05
at p = 0.5 and roughly 0.005 plug-in LL bias, close to the 0.007 floor. The
config registers probability clipping to [0.01, 0.99] and a replicated
convergence check on the timing shard [A9.2]: at each candidate n_sims
(100, 200, 400, ...) run three independent simulation batches per arm and
compute the spread of the paired C-B and B-A winner LL across batches;
the full run uses the smallest n_sims at which that spread's 95% range is
below 0.002, with no fixed ceiling. The chosen n_sims and the spread
table are recorded in the config before 1d.

Seeds [A6]: a per-fixture seed derived from the cricsheet id and a
registered base seed, separate streams for outcome sampling, extras, and
selector draws; the evaluator's hard-coded `random_seed=42` is overridden
by config. Paired Monte Carlo variability is measured by re-running the
timing shard with a second base seed.

Decision rule, registered [A10]: primary slice >=$50k; primary contrasts
C-B (sequence), B-A and C-A (system), all candidate minus reference. Equivalence margin 0.007 LL: a
contrast whose 95% block interval lies inside [-0.007, +0.007] is "parity";
one whose interval excludes zero and whose point estimate clears 0.007 is
"favourable" or "adverse"; anything else is "inconclusive". Advancement is
symmetric: B advances only if B-A is "favourable", C only if C-A is
"favourable", and a market claim additionally needs the claim gate's market
comparison and cost scenarios.

Scored three ways, committed in the config:

1. Winner LL vs the registered market through `claim_gate`, tournament
   block bootstrap, slices all / >=$50k / >=$100k.
2. Realism (exploratory): innings score P10/P50/P90 coverage,
   first-innings bias, extras per innings, wickets, unique bowlers used,
   batting-first minus chasing flip on the PPC cohort.
3. Props (exploratory): Brier vs `prop_fair_baselines`, paired per family.

Shard protocol, on the laptop:

- 1a smoke: 3-5 matches, 10 sims, all three arms. Pass = every arm loads,
  writes the sliced JSON, and the realism and prop scorers run. Odds
  resolution is tested separately on a deterministic fixture known to
  produce a bet, since a legitimate model can place zero bets [A8].
  Numbers are not read.
- 1b timing: 10 matches chosen to include long innings, 100 sims, all
  three arms, run as concurrent shards to measure real memory pressure.
  Record seconds per match per arm; extrapolate to 255.
- 1c shard-consistency check [A7]: serial vs sharded outputs must be
  identical on several boundary cases, not one: a same-day pair split
  across shards, a fixture at a date boundary, a fixture without odds, a
  fixture whose same-day sibling is in the context corpus but not in the
  evaluated set, and the first fixture of a shard (process initialisation)
  [A7.2]. Each shard keeps
  the full replay context. After the full run, assert the union of shard
  fixtures equals the registered set, no duplicate fixtures, every odds
  record claimed by exactly one fixture, and coverage counts match [A7.2]. The T1 runner refuses `--parallel` by design (the replay orders
  every stats read), so sharding is the only parallelism.
- 1d full run: 255 fixtures at the convergence-selected n_sims from the
  replicated check above (no fixed ceiling), sharded per 1b, then gate
  [A9.3].

Expected outcome, written down so it cannot move: A-C parity, B-C parity,
realism close on all arms; no arm advances.

## Stage 2: sequence variants with forgetting and ownership

**CLOSED 2026-09-12 (Astra sign-off `f4c92b5`); ledger row SQ2 DESCRIPTIVE
2026-09-13.** Finding: forgetting helps (fixed decay, xLSTM,
production-residual control CI-clean on 5/5 seeds); death-over harm
reproduced; no mechanism isolated; cohort unopened (condition 0 closed by user attestation 2026-09-13, see `docs/sequence_track/stage2_cohort_consumer_audit.md`; it opens for the first candidate that clears a rollout contrast). Reports:
`research/reports/embeddings/SEQ_STAGE2_REPORT.md`,
`SEQ_STAGE2_FIVE_SEED_ADDENDUM.md`. Kickoff brief:
`docs/sequence_track/stage2_kickoff_brief.md`.

Question: does the death-over harm come from stale history, from history
that belongs to other players, or from neither?

Stage 1 sharpened why this matters: the transformer beats both the token
MLP and the 50-feature XGBoost in rollout, so the history it reads is
carrying something, and the open question is *which part* of that history
and whether the death-over harm survives on the i7 frame. Stage 2 runs on
the i7 frame (the retrain path landed in stage 1), five seeds per arm, on
the Mac mini through the queue runner built in stage 0 and never yet used.

Protocol: the ablation lineage (`experiments/configs/t1_ablation_v1_mps.yaml`),
5 seeds, seed-mean plus match bootstrap, primary contrast per arm vs the
token MLP on the full validation split; phase, innings, and thin-player
slices exploratory; death-over and chase non-inferiority gates with
registered margins (upper confidence bound of the harm below +0.002 on
each) [A14].

Input-access matrix, registered [A11]: every arm is described by four
columns, current 50 features / shifted outcome history / identity
relationships (which earlier tokens share batter or bowler) / frozen
production logits. Each addition has a matched control.

| arm | features | history | identity | prod logits | what it tests |
|---|---|---|---|---|---|
| token MLP | yes | no | no | no | control |
| vanilla T1 | yes | yes | no | no | reference |
| recency-only mask (last k rows, any player) | yes | yes | no | no | control for the mask's recency component [A13] |
| fixed-decay T1 (ALiBi-style, no learned gate) | yes | yes | no | no | control for FoX's learned gate; identical to FoX in every other setting, including no positional embedding, QK-norm, width, depth, dropout and optimiser [A13, A13.2] |
| FoX T1 | yes | yes | no | no | learned data-dependent forgetting (per-head sigmoid gate, cumulative log-gate bias added before softmax, no positional embedding) |
| participant-aligned history T1 (no mask) | yes | yes, participant-aligned | yes | no | control for the aligned-history input alone [A11.2] |
| same-entity mask T1 | yes | yes, participant-aligned | yes | no | ownership |
| LSTM | yes | yes | no | no | classic forget gate |
| xLSTM | yes | yes | no | no | exponential gating |
| residual T1 | yes | yes | no | yes | sequence over the production model |
| residual MLP | yes | no | no | yes | control for the residual arm [A17] |

Same-entity mask details [A12, A16]:
- Windows are defined in delivery rows (inclusive of extras), not overs;
  innings reach 138-144 rows. k = 30 rows is the registered default. The
  sweep is k in {0, 6, 12, 30, unrestricted}, where "unrestricted" is the
  true whole-innings endpoint.
- The previous-outcome input of a visible token can belong to another
  player. In the mask arm the history input is participant-aligned: the
  previous outcome of the same batter, and the same bowler, as separate
  embeddings, instead of the innings-previous outcome.
- With two layers, allowed tokens can relay excluded history. The mask is
  applied at every layer, and ownership isolation is certified by a
  computational dependency test [A12.2]: for a sample of 2,000 target
  positions, perturb the features of every token outside the ownership
  set and assert the target's output is unchanged to 1e-6; if any
  dependence remains, the arm is rebuilt so the ownership set is the only
  path (for example, a one-layer model, or per-layer masks that also
  restrict what the allowed tokens themselves may have seen). k = 0 is
  called "ownership" only after this test passes.
- Window selection is on validation only: choose the k with the best
  validation LL; if no k beats k = 30 by more than 0.002, keep 30. A
  ball-specific tolerance is used because the ablation's whole T1-MLP gain
  was 0.0004, so 0.002 is the smallest difference the five-seed protocol
  resolves; early stopping uses a validation date-prefix disjoint from the
  selection rows where feasible [A15]. Only the chosen k is scored on test,
  and test is exploratory per the track rules.

Residual arm definition [A17]: p = softmax(log p_base + r_theta), with
p_base from the production ball model floored at 1e-4, r_theta
zero-initialised with L2 shrinkage to zero, base logits produced
temporally out-of-fold (the production model is trained on rows before the
validation split, so validation and test logits are out-of-sample; train
logits are refit out-of-fold by date blocks). Controls: base-only, and the
residual MLP without history. Residual T1's confirmatory contrast is
residual T1 minus residual MLP; beating the plain token MLP does not
qualify it, because that gain could come from the production logits alone
[new-1].

Recurrent arms [A18]: block composition frozen in the config (LSTM: 2
layers, hidden matched to T1's 128-d; xLSTM: 2 blocks, sLSTM then mLSTM,
128-d), state reset at innings start, full-innings backpropagation, equal
tuning budget (learning rate and dropout grid of the same size as T1's),
parameters and compute reported.

Reading rule, fixed now, and stated as suggestive rather than causal
[A13]: the mechanism question is answered by direct contrasts, FoX minus
fixed-decay (learned forgetting), same-entity minus recency-only
(ownership beyond recency), each with its own paired interval. "Arm X is
significant and arm Y is not" is never treated as a difference. If no
sequence arm beats the MLP by a CI-clean margin, the conclusion is that
this model family, at this resolution, shows no sequence gain; it is not a
proof that the scoreboard summary is sufficient [A14]. Only an arm that
beats the MLP CI-clean, passes both non-inferiority gates, and survives the
untouched cohort goes to the simulator under the stage 1 protocol.

Not chosen, recorded: Gated DeltaNet, Mamba-2, Titans. Linear-time
recurrence buys nothing at 120-144 tokens per innings, and FoX keeps the
existing T1 code and ablation arms.

## Stage 3: Delphyne-style ideas

**Status 2026-09-13:** 3a run and closed (negative transfer not detected,
`SEQ_STAGE3A_NIGHT3_REPORT.md`; equal-epoch replication backlog); 3b not
started (TODO backlog, contract in
`docs/sequence_track/delphyne_mapping.md`); 3c CLOSED by
`STAGE3C_LEAKAGE_AUDIT.md`. Ledger row SQ3.

Source: Ding, Mittal, Gopal, "DELPHYNE: A Pre-Trained Model for General and
Financial Time Series", NeurIPS 2025, arXiv 2506.06288. It is a numeric
time-series model; our data is discrete events with per-event context, so
it does not drop in. Three ideas transfer. Precondition: read the full
paper and write a one-page mapping before code. Order and design [A19, A20]:

- 3a Negative transfer, first, and it does not wait for a stage 2 winner:
  it runs on the token MLP if stage 2 produces none. One experiment, not
  three: a partially pooled tier-conditioned model (tier embedding plus
  per-tier bias, shared weights) against the pooled model and against a
  matched-data control trained on the same number of rows from the target
  tier only, and a conditioning-matched pooled control (the same tier
  embedding and per-tier bias, trained on the full pool) [A19.2]. A
  negative-transfer claim requires the tier-restricted or partially pooled
  arm to beat the conditioning-matched pooled control, so conditioning
  and pooling are not changed together. Evaluated per tier and pooled.
- 3b Over-level patching, strictly causal: completed-over history as patch
  tokens plus a causal within-over decoder for the current over, so no
  token contains balls after the one being predicted. Compared to
  ball-level tokens at equal compute.
- 3c Masked pretraining, last and lowest priority: the same 1.9M balls are
  already fully supervised, and redundant scoreboard fields make
  reconstruction trivial. Deferred until a leakage-resistant objective and
  an equal-compute from-scratch baseline are written down.

## Stage 4: embeddings that are not "learn a player ID"

**Status 2026-09-13:** references built; 4d run (null); 4b re-registered
as `seq_stage3_batch2_4b_v2` after the v1 driver refusal; 4a audit only
(`STAGE4_PAIR_GRAPH_AUDIT.md`); 4c and 4e not started. C114 (stage 1
follow-up) run teacher-forced: token MLP beats the transformer on 114
features (`SEQ_STAGE3_BATCH2_REPORT.md`). Ledger row SQ4. Rollouts now go
through `docs/sequence_track/rollout_protocol.md`.

Order by expected value under current data and cost [A21]: 4d, 4b, 4a,
4c, 4e. All five are tried; the order sets what runs first. Each is a
registered experiment with its control and its stated difference from the
failed E-rung [A22]. Every stage 4 experiment, without exception, reports
against two fixed references in addition to its named control: the frozen
EB linear model and a raw-rate (unshrunk counts) linear model [A22.2].
Ball LL is the first readout except for 4e.

- 4d Uncertainty-aware ratings (first). Causal EB posterior spread, count,
  and recency as explicit features, then Glicko-style or learned-variance
  ratings, against a counts-only control. Difference from E1-E4: no
  identity vector is learned; the addition is the spread of the existing
  estimate.
- 4b Residual-only embedding. Freeze the EB linear model; the embedding
  predicts only its residual with strong shrinkage to zero. Control: the
  frozen EB model. Difference from E4: E4 let the network re-learn the
  base; here the base cannot move.
- 4a Interaction factorisation [A23]. Low-rank batter x bowler term on top
  of the frozen EB linear model. Before training: an exposure-graph audit
  (connectivity and overlap of batter-bowler pairs across leagues), since
  league segregation confounds the factors; marginal and intercept
  aliasing constrained by centering each factor against its EB marginal;
  rank and shrinkage registered; zero fallback for any player without a
  factor. Results reported separately for seen pairs, unseen pairs of
  known players, and unseen players (13% of validation and 23% of test
  balls involve a wholly unseen player, where free factors cannot help).
  Difference from E1: E1 was additive per player; this is the product term.
- 4c Style vectors from line, length, shot, and control labels
  (DeepCrease). The coverage audit starts now, independent of stage 2,
  because the idea's value depends on it. Player-grouped holdouts,
  context-residualised, provenance settled before anything leaves the
  repo. Control: EB outcome vectors.
- 4e Team-as-set readout [A24]. Not under the ball-LL gate. A separate
  match-level gate through `scripts/experiment_harness.py` and the claim
  gate against `match_model_prod`, with identical lineup information,
  simple pooled-EB controls, and role- and order-aware pooling variants
  alongside the permutation-invariant one, since batting order and
  bowling roles carry information a plain set discards.

## Stage 5: documentation alignment

Before any merge: bring CLAUDE.md, ARCHITECTURE, OPERATIONS, TODO.md,
IMPROVEMENTS.md, the embeddings report index, and the auto-research ledger
into agreement with every stage result; add the TODO.md pointer to this
file; restate any number that a stage moved; append errata rather than
editing frozen reports; regenerate the atlas snapshot if it is kept.

## Stage 6: merge

Remediation plan item 5 step 6 is the merge of `embeddings-ladder` to
`main`; this stage is that same step, not a second merge [A1.2]. It runs
last, after the engine gates are green and stage 5 is complete. Stages 1-4
do not depend on the merge.
Nothing from stages 1-4 is promoted by the merge itself; a promotion is a
separate manifest change with its own audit.

## Bookkeeping

- Configs under `experiments/configs/`, results under
  `research/reports/embeddings/`, raw runs under `models/embeddings/`
  (gitignored), verdicts through `research/log_verdict.py` with the gate
  JSON.
- Every stage records: config sha, seeds, engine and model md5s from
  `models/MANIFEST.yaml`, wall time, and the pre-registered pass condition.
- Compute: laptop (48 GB, 15 cores) for interactive shards and the stage 1
  full run; the Mac mini (16 GB, 10 cores) remains available for overnight
  runs, e.g. multi-seed stage 2 training, with capped threads and jobs run
  sequentially. The exact division is decided per stage when work starts
  (user, 2026-09-10).

## Stage 0 kickoff brief (for the fresh agent)

Read, in order: this file; `docs/REMEDIATION_PLAN_2026-09-09.md` sections
0, 5 and 11; `research/reports/embeddings/README.md`;
`research/reports/embeddings/T1_ABLATION_V1_MPS.md`;
`research/reports/embeddings/T4_rollout_eval.md`;
`research/reports/auto/BR2.md`; the header docstrings of
`scripts/transformer_t1.py`, `scripts/sim_t1.py`,
`scripts/sim_eval/run_sim_eval_t1.py`, `scripts/run_t1_ablation.py`,
`scripts/registered_experiment.py`, `scripts/sim_eval/claim_gate.py`,
`research/log_verdict.py`, `scripts/artifacts.py`, and
`research/night_v3.sh` (for the runner idioms to reuse).

Deliverables, each with an acceptance check written before it starts:

1. `docs/remediation/`-style acceptance file
   `docs/sequence_track/stage0_acceptance.md`.
2. Queue runner and guards, with the dummy-job dry run recorded.
3. Readiness record: `artifacts.py verify` output, BR2 status, replay and
   parity test run, claim gate exercised on the HA0 evidence.
4. A50: XGBoost on the 50 T1 features with the production config; parquet
   built from the i7 frame by column selection, hashes recorded.
5. Stage 1 config `experiments/configs/seq_stage1_sim_v1.yaml` pinning
   every artifact per arm (A, A50, B, C; C114 marked deferred), the
   per-fixture seed rule, the clipping rule, the convergence protocol,
   the decision rule, and complete command lines.
6. Cross-arm cutoff and eligibility audit script with a passing run on
   the 1a smoke fixtures.
7. Astra review of items 1-6 to sign-off; then commit (Fable commits;
   the Codex sandbox cannot write `.git`).

Do not run 1b or anything beyond the 1a smoke without Fable's go.

## References

- Forgetting Transformer: https://arxiv.org/abs/2503.02130 ;
  code https://github.com/zhixuan-lin/forgetting-transformer
- xLSTM: https://arxiv.org/abs/2405.04517
- DELPHYNE: https://arxiv.org/abs/2506.06288
- Gated DeltaNet (not chosen): https://arxiv.org/abs/2412.06464
- Titans (not chosen): https://arxiv.org/abs/2501.00663
