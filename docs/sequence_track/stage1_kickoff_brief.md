# Stage 1 kickoff brief (sequence and embeddings track)

Written 2026-09-11 by the stage 0 orchestrator, for the agent that picks up
stage 1. Stage 0 is complete, reviewed to sign-off and committed at
`2587701`. Nothing past the 1a smoke has run.

Read this file first, then `docs/SEQUENCE_TRACK_PLAN.md` (v5, approved by the
user 2026-09-10 — the plan of record, and the only place the stage 2-6 design
lives), then `docs/sequence_track/stage0_acceptance.md` (every stage 0 check
with its recorded result) and `docs/sequence_track/stage0_readiness.md`.

---

## 1. Your role and the rules

You orchestrate. Opus subagents implement code, configs and runs; Codex Astra
(`gpt-6-astra`, low effort, read-only sandbox) reviews at the end of the stage
and before any commit that lands results; you are the only one who commits and
the only one who calls `research/log_verdict.py`. The user decides every
advancement, every verdict, the simulation count, and the cache question in
§4. Do not fan out more than three subagents at once.

Rules that override convenience:

- Never read or evaluate against `data/golden/` or `data/forward_holdout/`.
  Hashing them through `scripts/artifacts.py verify` is integrity checking and
  is fine; opening their contents in a model, scorer or runner is not.
- Never overwrite a production artifact or anything a manifest role points at.
  Stage 1 writes only under `models/embeddings/seq_stage1/`.
- Frozen evidence and committed reports are never edited; errata are new files.
- Python only through `uv run --no-sync`. Never `pip`; `uv add` for deps.
- Report numbers verbatim from files. If a run did not finish, the number does
  not exist.
- Commit convention is in `CLAUDE.md` (summary plus body, no AI-attribution
  trailers). Read it before your first commit.
- Everything in stage 1 runs on the laptop (48 GB, 15 cores). The Mac mini is
  for stage 2 multi-seed training, not for this.
- Write acceptance checks before the work, in
  `docs/sequence_track/stage1_acceptance.md`, one table per deliverable with
  numbered check ids and pass conditions, and record results under each table.
  That is how stage 0 was run and it is what Astra reviews against.

---

## 2. What this workstream is for

The 2026-08 program closed with three findings that this track deliberately
re-opens (`research/reports/embeddings/PROGRAM_STATUS_20260813.md`):

1. Every embedding rung lost to a linear model on the same empirical-Bayes
   inputs. The rungs all asked "learn who the player is"; none tried
   interactions, frozen-base residuals, uncertainty, or non-outcome data.
2. The T1 transformer's gain over logistic was almost entirely a token MLP
   with no sequence memory. Full T1 minus the MLP was −0.0004 with an interval
   crossing zero, and full T1 **hurt the death overs on 0 of 5 seeds**
   (+0.0053). Whether within-innings sequence carries anything is unresolved,
   not settled.
3. T4 put T1 through the simulator and failed at 0.7156 winner log loss — on a
   broken engine, with a random bowler selector, 50 simulations, and an odds
   loader that placed zero bets. No fair post-repair run exists.

**Stage 1 is the fair post-repair run.** Its question, stated precisely:

- **(i) system comparison** — does the 50-feature MLP or T1 stack, on the
  certified replay path, simulate the winner market as well as the production
  i7 ball model?
- **(ii) incremental sequence evidence** — does full T1 improve on the token
  MLP in rollout? Only (ii) speaks to sequence value; (i) compares systems
  with different inputs.

Two framing rules the plan fixes and you must not relax:

- **The pre-registered expected outcome is parity everywhere and no arm
  advancing.** It is written down so it cannot move after the fact. A surprise
  has to survive the gate, not the narrative.
- **Log loss decides; returns are a safety check.** Confirming a true 10%
  return edge needs roughly 1,100-1,800 priced bets. At about 300 sharp-market
  fixtures a year, returns cannot decide anything on a project timescale.
  Ball log loss is a screening metric with no fixed conversion to market value.

Stage 1 is a **single-checkpoint screen**: one training seed per neural arm. A
survivor is confirmed later on five training seeds and two independent
simulation batches before any claim.

---

## 3. What stage 0 built and proved

Commit `2587701`, on top of `dfa45ad`. Astra reviewed in four rounds
(NO SIGN-OFF, NO SIGN-OFF, NO SIGN-OFF, SIGN-OFF); every finding and its
disposition is in the acceptance file's D8 table. Full suite at the end:
828 passed, 5 skipped, 0 failed.

| deliverable | what exists now |
|---|---|
| Queue runner | `research/sequence_track/run_queue.sh` + `queue.yaml` + `scripts/sequence_track/queue_lib.py`. Ordered jobs, skip-if-complete by config sha256, caffeinate, per-job timeout enforced by both a perl alarm and the watchdog, STOP file, tee'd logs, one retry, free-memory floor re-read before every attempt (fails closed on a bad reading), process-group RSS watchdog. No model, verdict or git call inside it. Unused so far — stage 1 runs interactively; it exists for stage 2 |
| Readiness | `artifacts.py verify` OK on 24 roles; BR2 gates green (G5 by written exception, bar now ≥89%); T1 replay lifecycle, snapshot guard, prefix cache and parity tests pass; `claim_gate` and `log_verdict` exercised end to end on the item 6 swap-smoke evidence in dry run, with the real ledgers untouched |
| A50 | `models/embeddings/seq_stage1/a50/`. Production XGBoost config on only T1's 50 features, column-selected from the i7 frame, no class weights, seed 29, early stop at `best_iteration` 24, booster md5 `7705b330…`. Ball log loss train 1.3449013907578484, validation 1.441408870047355, test 1.4332497087527303 |
| Stage 1 config | `experiments/configs/seq_stage1_sim_v1.yaml`, generated and verified by `scripts/sequence_track/pin_stage1.py`. Pins every artifact per arm, 34 source files, runtime versions, the seed rule and its joint overlap screen, the clip, the convergence protocol, the decision rule, and complete command lines. `--verify` must pass before anything runs |
| Arm runner | `scripts/sequence_track/run_arm.py`. One runner for all four arms through the same-day replay lifecycle. Writes `arm_provenance.json` (contract v2), `raw_sims.jsonl`, `eval.json`. `--config` verifies the pins and refuses any drift between the flags and the registered block |
| Cross-arm audit | `scripts/sequence_track/audit_cross_arm.py`. Typed schema, block binding, registration validation against the pinned config, per-fixture seed recomputation, and equality across arms of fixture sets, as-of stamps, eligibility, odds-row identity, selector and sidecar hashes, config/odds/context/engine/runner hashes |
| Scorers | `scripts/sequence_track/score_realism.py` and `score_props.py` over the runner's raw simulations. 15 prop families scored, 18 skipped with the missing field named per family |
| 1a smoke | Four fixtures (`1478908`, `1501896`, `1505127` sharing 2025-10-31, plus `1477609`), 10 simulations, all four arms, run from the config's own command strings. Every arm loads, writes its sliced JSON, both scorers run, the audit passes, and odds resolution places a bet on fixture `1512749`. **No log loss, Brier, ROI or edge from the smoke was read, and you must not read them either** |

Engineering facts stage 0 discovered that you inherit:

- The simulator seeds **one global RNG stream** per simulation (`random_seed + i`).
  Outcome sampling, extras, selector draws and run-outs all draw from it. The
  three named sub-seeds are derived and recorded but not separately consumed.
  Because the scheme is additive, two fixtures whose 31-bit seeds lie within
  `n_sims` of each other share draw schedules; a joint overlap screen across
  the three registered batch seeds caps any batch at **3,200** simulations
  (joint minimum gap 3,290).
- The evaluator has **always** clipped match probabilities to [0.05, 0.95].
  The registered stage 1 clip of [0.01, 0.99] therefore *widens* that, and is
  wider than every earlier run including the BR2 G1 gate. Confirmed by the
  user 2026-09-11. The seam defaults to legacy behaviour everywhere else.
- **BR2's G1 number is not a stage 1 reference.** It ran with the plain
  empirical selector, the snapshot provider and the old clip.
- `highest_individual_mae` cannot be scored: the prop fair-baseline corpus of
  record predates the match-level top-score column. An unregistered
  `models/prop_fair_baseline_corpus_v3.pkl` exists on disk; registering it is
  a decision, not a fix to apply silently.

---

## 4. The open decision: one stats cache for all four arms

**This is the user's decision and the first thing to settle.** They have said
they want a single cache for every arm and will not split. The material is
below so you can help them decide; do not decide it yourself.

### What the candidates are

| cache | built | venues | corpus ends | alias contract | same-day ordering |
|---|---|---|---|---|---|
| `models/player_stats_cache_i7.sqlite` | 2026-07-25 | 373 | 2026-04-16 | `venue_aliases_v1` | `date_then_match_id_lexicographic_v1` |
| `models/player_stats_cache_v3.sqlite` (legacy) | 2026-04-25 | 467 | 2026-04-16 | none | **absent** |
| `models/player_stats_cache_v3.sqlite.local_20260910` | 2026-08-08 | 378 | 2026-08-05 | `venue_aliases_v1` | present |

The third is a moved-aside file the user has ruled off limits for use. It
matters only because it is what the existing neural checkpoints were actually
trained on — the T1 ablation report's registered parquet hashes match
`data/xgb_data_v3_local_20260910`, not the current `data/xgb_data_v3`.

### What actually differs, measured

**Venue identity.** 94 hand-approved alias folds over 92 canonical grounds,
versioned and hashed (`config/identity/venue_aliases_v1.csv`, sha256
`853b32b0…`), exact-string matching only, raw Cricsheet never rewritten. The
fold is an exact count union; `k_venue = 200` is nearly inert at these sample
sizes, so the whole movement is the union re-weighting the estimate.

**Ordering.** The legacy cache predates the deterministic same-day ordering
contract and carries no version, so its within-date replay inherited
filesystem order. On Finals Day 2014 at Edgbaston it replays the final before
its own two semi-finals. Consequence, measured over the whole corpus:

| table | rows | differing between i7 and legacy |
|---|---|---|
| `batting_elo` | 147,419 | 127,586 (86.5%), max 2.10 ELO |
| `bowling_elo` | 111,465 | 96,195 (86.3%), max 2.13 ELO |
| `batting` recent-form columns | 150,006 | 585 |
| `bowling` recent-form columns | 114,850 | 457 |
| career counters, vs-type, vs-hand, h2h, non-alias venue rows | — | **0** |

**Feature-level effect on the evaluation window** (255 iteration fixtures,
max absolute difference across the six venue columns per fixture):

| comparison | median | p95 | max | fixtures differing |
|---|---|---|---|---|
| i7 vs the neural arms' training cache | 0.000007 | 0.000119 | 0.000515 | ball counts identical on all 255; residual is the global prior only |
| i7 vs legacy | 0.001082 | 0.020591 | 0.042700 | 132 of 255, 48 of them above 0.01 |

No fixture in the window has a raw venue string that *is* an alias — the
window opens after the feed cutover — but 132 are played at a canonical target
of a fold, and those are exactly where the legacy cache serves truncated
history. For the 50-feature arms the two caches differ in **only the six
venue columns**; every player column is bit-identical.

### The finding that complicates the choice

The alias split is a feed convention change, not stadium eras: 81 of 94 pairs
are temporally disjoint but 63 of those cutovers land in 2020-21, and the 13
overlapping pairs interleave by competition within the same seasons. However,
because the relabelling happened at a date boundary, each fold merges a
pre-2021 era with a post-2021 one, and scoring at the big grounds moved far
more across that boundary than the league did (Wankhede +0.0202 six rate,
Eden Gardens +0.0283, Chinnaswamy +0.0204, versus +0.0052 corpus-wide). Venue
history is cumulative and unweighted, so the fold drags Eden Gardens' six rate
from 0.0757 to 0.0560. Under the legacy identity the canonical label held only
post-2021 balls, so the broken identity was accidentally acting as a recency
filter. That is a plausible mechanism for the I7 rebuild measuring slightly
worse, and it is logged as its own backlog item in `TODO.md`
("Venue history is not recency-weighted"). It is **not** an argument for
un-folding, and it is not stage 1 work.

### What the identity change bought, measured

The one clean identity-only comparison
(`reports/i7_rebuild_checkpoint_20260725.md`, same config, same class-weight
setting, only the map changed):

| readout | legacy | i7 |
|---|---|---|
| ball test log loss | 1.629655 | 1.6316 |
| simulated winner log loss, ≥$50k | 0.6970 | 0.7176 |
| match model log loss, ≥$50k | 0.6299 | 0.6421 |

Every i7 point estimate is worse, no block interval excludes zero, the
venue-enriched slice did not improve, and neither i7 model was promoted on
these numbers. Do not confuse this with the large ball-model gain: 1.6316 →
1.4253 came from **removing balanced class weights** (D16, a paired twin on
one frame, so no identity confound), not from venue identity. The ROI figures
in that report predate the v2 odds correction and must not be quoted.

### The factors

Arguments for **i7**:

- The production ball model, production match model, live serving state and
  the Hundred path are all i7 already. Arm A cannot move without retraining
  production.
- The legacy identity **cannot be trained against**: `scripts/xgboost_v2.py`
  asserts the venue-alias contract on the frame's feature hash, the legacy
  frame carries no alias fields, and the run dies in one second (D6). That is
  a deliberate fail-closed guard encoding a human migration decision, not a
  bug to route around.
- The legacy cache violates CLAUDE.md invariant 5 (same-day order is a
  versioned data contract).
- The neural arms' own training frame is alias-canonical, so i7 is the closer
  match to them, not the further one.

Arguments for **legacy**, stated fairly:

- Every measured identity-only comparison favours it, even though none is
  CI-clean, and the recency mechanism above gives a reason why that might be
  real rather than noise.
- It is the frame the retired v7 ball line and its reports were built on.

Cost of each: moving everything to i7 means retraining two small neural arms,
about thirty minutes. Moving everything to legacy means rebuilding the cache
and frame without the alias map, reversing the guard, and retraining the
production ball model, the production match model, the Hundred model, the live
state and A50 — and accepting non-deterministic same-day ordering.

### The recommendation to put to the user

**i7 for all four arms, with B and C retrained on the i7 frame.** The retrain
is not needed for correctness — serving the existing checkpoints from i7 would
land within 0.0005 of their own training cache — but the user asked for it on
principle and it buys real comparability: A50 sits on a 186,667-row test split
and the existing neural checkpoints on a 186,912-row one (different Cricsheet
exports, plus two registry corrections and one extra in-window match), so
their ball log losses cannot be put in one table today. After the retrain they
can.

What would change the recommendation: a decision to treat the recency finding
as blocking, in which case the honest move is to fix the weighting first and
re-decide, not to adopt the legacy identity as a proxy for it.

### The retrain work list, if i7 is chosen

1. `scripts/transformer_t1.py` hardcodes `cricket_data_v3_{split}.parquet`;
   parameterise the stem so `--data-dir data/xgb_data_i7` works.
2. `scripts/sim_t1.py` refuses any checkpoint whose recorded `data_dir` does
   not contain `xgb_data_v3`. It is a substring test on a path, not a
   semantics test, and both frames declare `inclusive_total_runs_v1`. Replace
   it with a delivery-semantics check so an i7 checkpoint can be served.
3. Retrain arms `mlp` and `full`, five seeds each (`7, 13, 29, 42, 101`),
   select by validation log loss only, ties to the lowest seed — the rule is
   already in the config and `pin_stage1.py` implements it.
4. The eval kit (`models/embeddings/eval_kit/`) is pinned to the old split
   (186,912-row masks) and will not align with the i7 test split. Selection
   does not need it; any ball-level diagnostic or logistic baseline does.
   Rebuilding it and refitting the logistic control is a separate, declared
   piece of work — do not quote a ball log loss against the old ablation table
   across frames.
5. Re-pin (`pin_stage1.py --write` then `--verify`) and re-run the 1a smoke
   from the regenerated commands, then the audit. Recorded cost from the
   ablation's file timestamps: about one minute per MLP seed and three per T1
   seed on MPS.

---

## 5. How the four arms are tested

One runner, one chronology, one replay lifecycle, identical sidecars. The
audit is what makes "identical information" a checked fact rather than a
claim.

| arm | model | role |
|---|---|---|
| A | `ball_model_prod` (i7 no-weights XGBoost, 114 features) | production system |
| A50 | XGBoost, production config, T1's 50 features only | production family at equal information |
| B | token MLP, 50 features, registered checkpoint | nonlinear control |
| C | full T1, 50 features, registered checkpoint | sequence candidate |
| C114 | full T1 on the production 114 features | **deferred**, registered with no artifacts |

Fixed settings, all registered: iteration set `data/polymarket_test_v2` (255),
odds role `odds_iteration_v2`, roster-aware empirical bowler selector, the B18
extras sidecar (`ad6e863b…`) on every arm, run-out constant 0.075077, no
calibration anywhere, prefix cache off, cpu, four threads, probability clip
[0.01, 0.99], per-fixture seed = low 31 bits of
sha256(`"<cricsheet_id>:<base_seed>"`) with base seed 20260910.

**Scored three ways**, as registered:

1. Winner log loss through `claim_gate` against the registered market, with
   the I3 tournament-block bootstrap, on slices all / ≥$50k / ≥$100k.
2. Realism, exploratory: innings score P10/P50/P90 coverage, first-innings
   bias, extras per innings, wickets, unique bowlers, the batting-first minus
   chasing effect.
3. Props, exploratory: Brier against `prop_fair_baselines`, paired per family.

**Decision rule, registered, do not restate it after seeing results.** Primary
slice ≥$50k. Contrasts are candidate minus reference in log loss, so negative
is favourable: C−B (sequence), B−A and C−A (system), each Holm-adjusted within
the confirmatory family; A50−A is exploratory. Equivalence margin 0.007: an
interval inside [−0.007, +0.007] is parity; an interval excluding zero whose
point estimate clears 0.007 is favourable or adverse; anything else is
inconclusive. B advances only if B−A is favourable, C only if C−A is, and a
market claim additionally needs the gate's market comparison and cost
scenarios.

### The remaining steps

**1b — timing and convergence.** Ten fixtures chosen to include long innings
(the iteration set's median is 240 deliveries, p90 253, max 271 — the four
smoke fixtures averaged 215, so the shard will be slower per simulation than
the smoke was). Three independent simulation batches per arm at each candidate
count, run as concurrent shards so memory pressure is real. The user has fixed
the candidates: **50, 100, 200, 400, 800**. Record seconds per match per arm
and extrapolate to 255. Also re-run the shard with the second registered base
seed to measure paired Monte Carlo variability.

Two things to handle before it runs:

- **50 is not in the registered permitted list.** Add it to the `timing_1b`
  block's permitted `n_sims`, re-run the joint seed overlap screen to include
  it, and re-pin. At 50 simulations the plug-in log loss bias is about 0.01,
  above the 0.007 noise floor, so 50 is a point for fitting the noise curve,
  never a candidate for the full run. Say so in the report.
- **The stop rule needs an operational reading, and this is a user decision.**
  As written, the full run uses the smallest count at which the 95% range of
  the paired C−B and B−A spread across batches falls below 0.002, measured on
  the timing shard. Averaging over ten fixtures instead of 255 makes that
  spread roughly five times larger than the same quantity on the full run, so
  on ten fixtures it plausibly never reaches 0.002 at any affordable count.
  The two honest readings are to scale the shard spread by the square root of
  10/255 and apply the threshold to that full-set equivalent, or to run the
  convergence batches on a larger subsample. The stage 0 orchestrator leaned
  to the scaled reading; put it to the user with the measured numbers in hand,
  and record whichever they choose in the config before 1d.

Cost extrapolated from the smoke, all four arms, before the long-innings
adjustment: about 11 minutes per candidate at 100, scaling linearly, so the
five candidates together are roughly two hours. Full T1 is three times slower
than the other three and sets the wall clock everywhere.

**1c — shard consistency.** Serial and sharded outputs must be identical on
five boundary cases, not one: a same-day pair split across shards, a fixture
at a date boundary, a fixture without odds, a fixture whose same-day sibling
is in the context corpus but not in the evaluated set, and the first fixture
of a shard. Each shard keeps the full replay context. After the full run,
assert the union of shard fixtures equals the registered set, no duplicates,
every odds record claimed by exactly one fixture, and coverage counts match.
The T1 runner refuses `--parallel` by design, so sharding is the only
parallelism.

**1d — full run.** 255 fixtures at the chosen count, sharded per 1b, all four
arms, then the gate. Cost at 400 is roughly six hours across the arms and at
1600 roughly a day; bring the user the timing table and let them choose.

**Gate and verdict.** `claim_gate --kind match_model` against
`odds_iteration_v2`, then the realism and prop scorers, then the verdict
through `research/log_verdict.py` with the unchanged gate JSON. The user makes
the advancement call. Remember invariant 7: match-winner ROI uncertainty uses
I3 blocks, and fewer than ten blocks is descriptive only.

---

## 6. Inherited open items

- `timing_1b` has no fixture directory yet; its block is pinned as absent and
  cannot be claimed by a run until you build the shard and re-pin.
- The full-run command still carries a `<chosen_n_sims>` token by design; set
  `CHOSEN_N_SIMS` in `pin_stage1.py` and re-write once the user picks.
- The eval kit and logistic control need rebuilding on the i7 split before any
  ball log loss is comparable across frames.
- `models/prop_fair_baseline_corpus_v3.pkl` is unregistered; registering it
  would let `highest_individual_mae` be scored.
- Nine registered deviations and four known asymmetries are recorded in the
  config. Read them before you write any report, and never silently drop one.

---

## 7. Backlog to shape with the user

One finding came out of the cache work that is not stage 1 work and is
deliberately unspecified: **venue history is not recency-weighted**. The
evidence and the warning against un-folding the aliases are in `TODO.md`
("Venue history is not recency-weighted"); §4 above carries the measurements.

The user wants to design this with you rather than receive a plan. Open
questions worth putting to them, in roughly the order they matter:

- **Mechanism.** Exponential decay on the count vector before shrinkage, a
  hard window, or an era-aware prior that backs a thin ground off to the
  recent run-scoring environment rather than the all-time one? These behave
  differently for grounds with little recent history, which is exactly where
  it bites.
- **Scope.** Venue only, or the player EB vectors too? `batting_recent` and
  `bowling_recent` already cover part of the player side, so the venue case
  may be the only real gap — worth checking before building anything.
- **Parameterisation.** One global half-life, or per-venue by data volume? A
  swept half-life needs a validation-only selection rule and a registered
  candidate set, the way the track handles every other tuned quantity.
- **The gate.** This is testable at ball level before any simulator run, which
  makes it cheap to falsify. What clears it: a CI-clean ball log loss
  improvement, a venue-slice improvement, or something about calibration on
  the folded grounds specifically?
- **Where it sits.** It could run as its own experiment against the current
  ball model, or fold into stage 4d, which already proposes count and recency
  as explicit features. It is not a stage 1 dependency either way.

Do not start it without the user, and do not let it delay stage 1.

---

## 8. What needs the user's explicit go

1. The stats-cache decision in §4, and with it the retrain.
2. Permission to start 1b, after the re-pin and re-smoke.
3. The reading of the convergence stop rule.
4. The simulation count for 1d, from the 1b table.
5. Permission to start 1d.
6. Every advancement decision and every verdict.

The user has asked to review the 1b results and choose the 1d count with
timing estimates in hand. Bring them the convergence table, the per-arm
seconds per match, and the extrapolation to 255 — and no log loss from the
smoke or the timing shard.
