# Standing rollout protocol for ball models (draft v1, revision 3, 2026-09-13)

Status: **Astra SIGN-OFF, round 4** (`docs/sequence_track/astra/
rollout_review_round{1,2,3,4}.md`; rounds 1–3 AGREE WITH CHANGES, every item
folded in below and tagged `[R1-…]` / `[R2-…]` / `[R3-…]`). Ready for
registration: the protocol file in § 2.4 is generated, hashed and committed
by the implementation steps in `astra_handoff_2026-09-13.md` § 4. One
non-blocking item stays open from round 4: the mini's qualification bound
(§ 2.8) must name the exact `convergence_1b.json` field, contrast and
absolute bound before that machine runs a ledger rollout; laptop
registration does not wait on it.

## 1. The problem this solves

The track has never had a standing way to put a trained ball checkpoint
through the simulator. Each stage that needed a rollout built its own:

| stage | what was rolled out | why it could not be reused later |
|---|---|---|
| 2026-08 T4 | v3-frame T1 and MLP | pre-I7 frame; simulator engine later corrected (BR2) |
| Stage 1 (2026-09-11) | production A (114 features), A50, token MLP B and full T1 C (50 features), one seed each | launcher and adapter bound to the four Stage 1 arm blocks; `sim_t1.py` serves `mlp`, `full`, `no_history` on the 50-feature contract only |
| Stage 2 (16 arms, 5 seeds) | nothing | fixed-decay, FoX, aligned, masked, LSTM, xLSTM and residual arms cannot be served: no positional-embedding arms get `max_seq_len = 1` in the wrapper, recurrent arms have no step interface, residual arms need production log-probabilities on the arm's own state |
| Batch 2 C114 / 4d | nothing | 114-column contract and sidecar columns are refused by the serving-contract guard; the C114 checkpoints store vocabulary sizes, not encoders |

Three root causes `[R1-M7]`:

1. **The simulator's model adapter and launcher serve a fixed set of arms.**
   Any new arm type, feature contract or sidecar means adapter work before a
   rollout is possible, so rollouts happen only when someone budgets that
   work, and each time for one arm.
2. **There is no standing compatibility and reuse policy.** Stored per-fixture
   outputs from Stage 1 *are* comparable to a later run under the same
   simulation settings, but nothing says which settings must match, nothing
   verifies it, and nothing indexes the stored outputs, so the safe choice
   has always been to re-simulate the control.
3. **Teacher-forced validation log loss is the only number the training queue
   produces, and it does not answer the rollout question.** Stage 1 measured
   a rollout gap (C−B) that the teacher-forced ablation had not shown. Batch
   2's C114 ordering is teacher-forced only; its report says explicitly that
   no match was simulated and that it neither confirms nor contradicts Stage
   1. For every Stage 2 and batch 2 checkpoint the rollout question is
   **unmeasured**, not contradicted.

## 2. The design

One immutable simulation contract, separately versioned adapter
implementations, separately versioned analysis registrations, one ledger
`[R1-M4]`. A checkpoint is rolled out **once per compatible simulation
contract**; its per-fixture records and raw simulations are stored, and
later comparisons are computed from storage. Re-simulation happens only on
an explicit invalidation (a simulator defect, a contract version change, or
a ledger-integrity failure), never to make a comparison.

### 2.1 Serving manifest `[R1-M2]`

A checkpoint is served from a **versioned serving manifest**, not "from its
run directory alone". `serving_manifest.json`, written next to `model.pt` by
an export step and verified by content hash, resolves every dependency:

- `model.pt`, `metrics.json` (`training_contract`, `arm_params`, saved
  standardisation means and SDs), `run_record.json` (config identity,
  overrides, training signature), the `COMPLETE.json` integrity record;
- feature contract id and the **encoders**: for `v7_114` the four
  category-to-code mappings, reconstructed deterministically from the pinned
  training frame and exported without retraining, then verified by parity.
  Production encoders are never substituted silently;
- sidecar declarations and their builders (exposure sidecar, tier map);
- for residual arms: the production booster by manifest role and md5, the
  class permutation, floor, normalisation and log applied at training
  (`build_base_logits.py`), so the arm sees `log p_base` on **its own**
  simulated pre-ball state, never A's trajectory `[R1-Q6.3]`;
- for identity-residual arms: the frozen reference artifact and its hash;
- the stats cache role and md5, `delivery_semantics`, `venue_alias_version`,
  the ordered feature names.

Entries for A and A50 are typed as XGBoost artifacts (booster md5, sidecars),
not assumed to carry `model.pt`.

### 2.2 Adapter: a capability registry, not "any checkpoint" `[R1-S1]`

`scripts/sequence_track/serve_checkpoint.py` (new) dispatches on the
registered combination of *wiring × history input × feature contract ×
sidecars* and **refuses** any combination not in its registry. It reuses the
training model implementations (`transformer_t1.py`, `recurrent_arms.py`)
rather than a second attention stack, sets BLAS/OMP caps before heavy
imports as `run_arm.py` does, keeps prefix caches off, and keeps every
mutable feature buffer per arm (the XGBoost feature extractor reuses a
non-thread-safe buffer). Feature building from simulated state preserves the
existing rules: live same-day counts plus simulated player updates, venue
distributions pre-match, aligned and masked arms capture participants before
each delivery, relay-free arms carry previous rows' own outcomes, windows
count delivery rows including extras. Recurrent arms are served by
**full-prefix replay** first; persistent stepping is an optimisation that
must match full-prefix predictions before it is used.

Capability registry, with Astra's incremental engineering estimates `[R1-Q7]`:

| capability | source | hours | order |
|---|---|---:|---|
| serving manifest, launcher, admission harness, ledger + import checks | shared | 16–24 | first |
| `token_mlp`, `full`, `no_history` on 50 features | Stage 1, 2 | 4–6 | first |
| `v7_114` contract (`mlp_114`, `full_114`): encoder export, numeric parity, scaling | batch 2 | 20–36 | first |
| `fixed_decay` (no positional embedding: fix the `max_seq_len = 1` wrapper path) | Stage 2 | 4–8 | first batch |
| `fox`, full-prefix inference | Stage 2 | 4–8 | after fixed-decay |
| `residual_mlp`, `residual_t1` (production log-probabilities on own state) | Stage 2 | 12–20 | second batch |
| `lstm` with persistent stepping | Stage 2 | 6–10 | before xLSTM if needed |
| participant-aligned standard history | Stage 2 | 6–10 | later |
| `xlstm` with persistent stepping | Stage 2 | 12–20 | deferred |
| relay-free / aligned-rf / ownership windows | Stage 2 | 12–20 | deferred |
| exposure counts / spread / recency sidecar | batch 2 | 8–14 | deferred |
| identity residual + frozen-reference serving | 4b v2 | 12–20 | deferred until admissible artifacts exist |

Deferred capabilities are registered as deferred; the registry does not
change after a batch's results are read.

### 2.3 Admission: three requirements, not one `[R1-M3]`

A checkpoint enters the ledger only when all three hold, recorded in its entry:

1. **Artifact and serving-contract verification.** The Stage 1 D4 guard is
   kept and generalised: delivery semantics equality, venue alias version
   equality, ordered feature names against the checkpoint's own contract,
   and the launcher's actual cache-hash **equality** check (`run_arm.py`),
   not presence only.
2. **Per-checkpoint prediction parity.** Full validation split through the
   serving path in teacher-forced mode reproduces the checkpoint's
   `predictions_validation.npz` (`probs`, `y`, `innings_id`) at a stated
   absolute tolerance (registered per capability; 1e-5 for float32
   attention arms), with finite normalised six-class rows, class order
   checked, the validation file's identity hashed, and **within-innings row
   order** preserved rather than a join on innings id.
3. **Serving-path certification, once per capability, not per checkpoint.**
   Live feature parity at 1e-6 against the training feature builder (the
   `T1_SIM_PARITY_PPC_V1` precedent), plus a reusable lifecycle suite:
   opening extras, strike and bowler changes, wickets, innings transitions,
   new simulations, repeated and skipped calls, copied states, long innings,
   unknown identities, same-day cutoffs; extras composition certified
   separately from six-class parity; for recurrent or optimised attention
   paths, stepped versus full-prefix equality.

Requirement 3 is what replaces per-arm adapter review; requirements 1 and 2
are automated per checkpoint.

### 2.4 The simulation contract `[R1-M1]`

`experiments/configs/rollout_protocol_v1.yaml` carries **exactly Stage 1's
simulation settings**, so Stage 1's outputs import without re-running:

- fixture set `data/polymarket_test_v2` (role `iteration_set_v2`, 255
  fixtures, dir md5 pinned); cluster source the same directory
- odds `betting_odds_polymarket_v2.json` (role `odds_iteration_v2`)
- context `data/t20s_json`, same-day replay cutoffs,
  `date_then_match_id_lexicographic_v1`
- `n_sims` 1,600; `base_seed` 20260910 and the Stage 1 per-fixture seed
  schedule; ten one-thread CPU shards, `prefix_cache: off`, `device: cpu`,
  `threads: 1`; merge by `merge_shards.py`, cross-arm audit
- `RosterEmpiricalBowlerSelector`, `bowler_phase_usage.json` (md5), roster
  policy (md5), B18 extras graft (**sha256**, as Stage 1 pins it), the run-out
  channel as a source-covered constant, **clip [0.01, 0.99]** `[R2-S3]`
- production anchor `ball_model_prod` by manifest role and md5
- the stats cache, player metadata and the source closure hashes of the
  simulator engine

**Statistics are versioned separately** (`analysis_v1`): the claim gate's
`tournament_time_block_v1` with **10,000 seed-42** whole-event resamples and
explicit bet placement, primary slice ≥$50k, `<10 blocks = descriptive`,
Holm within a registered family, the Stage 1 practical threshold 0.007 for
favourable / adverse / parity readings. Changing an interval calculation
re-scores stored outputs; it never re-runs matches.

**Verification.** `pin_stage1.py --verify` checks a Stage 1 file against the
Stage 1 schema and is not a generic verifier. A new `rollout_protocol.py
--write/--verify` generates the protocol file from the Stage 1 pins, hashes
the source closure of the engine and adapter, and provides an explicit
**historical-import compatibility check** (§ 2.6). Adapter extensions are
versioned (`adapter_v1.1`, …) with a compatibility procedure: an extension
must leave every previously certified serving path byte-identical on the
lifecycle suite and on a fixed replay fixture, or the contract version bumps.

### 2.5 The ledger `[R1-M4]`

`models/embeddings/rollout_ledger/v1/<entry_id>/` holds the merged
per-fixture evaluator records, per-fixture provenance, **`raw_sims.jsonl`
retained** (realism and prop analyses need it), the parity and certification
results, and `entry.json`:

```
entry_id           <config_id>/seed_<s>@<artifact md5[:12]>+<serving_manifest sha[:8]>+<contract id>   (typed: torch checkpoint | xgboost booster) [R2-S3]
serving_manifest   path + sha256
source_config      experiments/configs/<...>.yaml + sha256; run_record training signature
arm, wiring, history_input, feature_contract, sidecars, capability_id, adapter_version
training_seed, seed_selection {roster, rule, chosen, validation_ll (full precision)}
protocol           rollout_protocol_v1 + sha256; analysis_v1
machine {name, chip, torch, numpy, xgboost versions, thread caps}, started, finished, wall_s, peak_rss_mb
shards {count, inventories sha256, union == registered set, merge audit pass}
result_hashes {evaluator, provenance, raw_sims}; completion atomic (written last)
admission {contract_verified, parity {tol, max_abs_diff, rows, pass}, certification_id}
limitations inherited (global prior not as-of-date; simulation-noise note)
```

Integrity rules: atomic completion, resumable failed shards, duplicate
detection by entry id and result hash, and **fail-closed pairing** `[R2-M3]`:
the registered eligibility inventory (`eligibility_v1.json`: the 252 scored
fixture ids of 255 on `all`, the 167 on the primary slice, and the documented
reason for every exclusion) is part of the protocol; every entry's scored ids
and exclusion reasons must **equal** that inventory or the entry is refused;
two entries are compared only when their fixture ids, labels and tournament
blocks are identical. An unexplained omission is an error, never intersected
away. `compare_ledger.py A B` pairs stored records by
fixture and bootstraps by tournament block under `analysis_v1`.
`ROLLOUT_LEDGER.md` is rendered from entries: one row per entry with winner
LL on all / ≥$50k / ≥$100k, realism and prop summaries, and every registered
contrast with its interval and its inherited limitations `[R1-N2]`.

### 2.6 Importing Stage 1 `[R1-M1, R1-Q3]`

Stage 1's A, A50, B and C are imported as the first four entries only after
the import check verifies: the exact boosters and dependencies for A/A50;
B/C checkpoint hashes and the seed-101 selection provenance; the original
config hash, source closure and runtime (historical source verification,
not today's files); fixture inventory and eligibility, odds identity, replay
cutoffs, cache, context, metadata, selector, usage and roster artifacts,
extras, run-out setting, clip, seeds and simulation count; ten disjoint shard
inventories whose union is the registered set; merge and audit pass;
unchanged tournament-block identities; per-fixture agreement between
evaluator records, provenance and raw simulations. A and A50 get a
historical admission record, not an invented neural parity result. Stage
1's registered outcomes and limitations travel with the entries. The 2026-08
T4 rows stay where they are, linked as historical evidence `[R1-Q6.5]`.

### 2.7 Seed policy `[R1-M5]`

- **Selection.** One checkpoint per configuration enters first, chosen from
  the configuration's fixed, complete seed roster by lowest full-precision
  validation log loss, ties to the lowest seed; the choice is recorded in
  `entry.json` before the rollout starts. This estimates the selected
  checkpoint's performance, not an average seed.
- **One-seed readings are directional** (provisional; at most PROMISING).
  Two selected checkpoints keep their **actual seed identities** in the
  ledger; the gate for a one-seed contrast is run with the seed key
  `selected` on both arms and the actual seeds recorded in the gate note and
  both entries, exactly as Stage 1 did with its two seed-101 checkpoints.
  Whether `claim_gate` accepts that key is verified at implementation; if it
  does not, the one-seed gate is a registered `analysis_v1` extension
  `[R2-M2]`.
- **Reading rule, one rule only** `[R2-M4]`: Stage 1's, verbatim from its
  config. *Favourable*: the 95% block interval excludes zero and the point
  is below −0.007. *Adverse*: excludes zero and the point is above +0.007.
  *Parity*: the interval lies entirely inside [−0.007, +0.007].
  *Inconclusive*: anything else. Precedence in that order. An interval that
  excludes zero while crossing −0.007 with a point below −0.007 is
  favourable. No "touching" rule; one-seed rows carry a simulation-noise
  note (§ 2.9), not a stricter threshold.
- **Research follow-up (five-seed expansion)** `[R3-M1]`: every family
  names exactly **one candidate**, and its primary is written candidate
  minus reference. Expansion of that candidate is triggered only by the
  primary reading **favourable** after Holm. An adverse or other reading
  expands nothing; no family expands a reference arm, and no member other
  than the primary can trigger expansion. Expansion rolls out
  the missing seeds of the candidate **and** of every learned comparator in
  the family, reusing completed seed rows; five-seed comparisons use paired
  seed-and-block resampling with seed spread and the 4/5 direction count.
- **Production comparison against A is cumulative, not alternative**: it
  never triggers expansion on its own, and promotion needs the family
  primary favourable **and** the candidate − A reading favourable **and** the
  operational guards (realism, prop no-regression as Stage 1 G1/G3/G5)
  **and** a `claim_gate --kind match_model` verdict, which also checks profit
  harm. A is one fixed artifact: a five-seed candidate versus A is a
  **typed fixed-reference contrast** in `analysis_v1` (seed-mean of the
  candidate's five entries against the single A entry, block bootstrap),
  registered as an extension because the present gate requires equal seed
  sets and rejects repeated baseline predictions. Until that extension is
  implemented and tested, candidate − A is reported per seed with the 4/5
  count and no LANDED claim is possible `[R2-M2]`. The iteration set remains
  selection evidence afterwards.

### 2.8 Machine qualification `[R1-S2, R2-S3]`

The laptop is qualified by Stage 1. The mini is qualified only after pinned
runtime versions (python, torch, numpy, xgboost) and numerical settings are
recorded and a cross-machine comparison passes on the **Stage 1 timing shard**
(the ten registered fixtures, `timing/SHARD_RULE.md`) at 50 simulations:
teacher-forced predictions equal within 1e-5, and the cross-machine
paired shard-mean **winner log-loss contrast** (laptop minus mini, same arm,
same seeds) within the Stage 1 two-batch range recorded for that count in
`timing/convergence_1b.json` — the quantity that range actually measures
`[R3-N7]`.
Qualification is invalidated by any runtime version change and must be
re-run. Until then the mini runs training, not ledger rollouts. Whole-arm
placement stays the operating rule.

### 2.9 Simulation noise `[R1-S3]`

1,600 simulations was Stage 1's operational compromise (extrapolated
two-batch SD ≈ 0.002), not a convergence guarantee. Ledger reports keep that
limitation and distinguish simulation noise from training-seed spread. The
reading rule is § 2.7's only; this section adds a note, not a threshold.

## 3. First ledger batch

Sequencing per Astra: Stage 1 imports, then C114, then fixed-decay; FoX and
the production residuals second; xLSTM, masked and sidecar arms deferred.
Wall estimates below are Stage 1 measurements for B/C-type arms; residual
and 114-feature serving costs are unmeasured and are timed on one shard first.

| entry | source | note |
|---|---|---|
| A, A50, B, C | Stage 1 import | § 2.6 |
| `mlp_114`, `full_114` | batch 2, selected seed by § 2.7 | ~45 min / ~2.5 h (B/C-type reference) |
| `full_50` | batch 2, selected seed (101, the same seed number Stage 1's C selected) | a fresh checkpoint under the batch-2 training run, admitted after its training signature and artifact identity are checked; **not** an independent seed replication `[R1-M6]` |
| `fixed_decay` | Stage 2, selected seed | ~2.5 h |
| second batch: `fox`, `residual_mlp`, `residual_t1` | Stage 2 | after the capabilities above are certified |

Registered families (`analysis_v1`, Holm within each, membership **frozen
now across batches**, primary slice ≥$50k, § 2.7 reading rule)
`[R1-M5, R1-M6, R1-Q5, R2-M1]`. Each family is one decision, stated:

- `family_rollout_c114` (first batch, m = 3). **Candidate `full_114`.**
  Primary `full_114 − mlp_114`. **Decision:** favourable → `full_114`
  expands to five seeds against `mlp_114`'s five; anything else → no
  expansion. Members `full_114 − full_50` (the production-contract
  question, informational) and `full_114 − A` (production reading,
  cumulative for any later promotion registration, never a trigger).
- `family_rollout_mlp114` (first batch, m = 2). **Candidate `mlp_114`.**
  Primary `mlp_114 − B` (the 114-feature token MLP against the 50-feature
  token-MLP anchor; a comparison of two specific systems, registered as
  this candidate's own primary). **Decision:** favourable → `mlp_114`
  expands against the Stage 1 retrain roster of B; anything else → no
  expansion. Member `mlp_114 − A` (production reading, cumulative).
- `family_rollout_fixed_decay` (first batch, m = 1). **Candidate
  `fixed_decay`.** Primary `fixed_decay − C`. Research-only: favourable →
  expands against C's five seeds; no production reading registered.
- `family_rollout_fox` (second batch, m = 1, registered now). **Candidate
  `fox`.** Primary `fox − fixed_decay`. Research-only, same rule. A
  mechanism claim additionally needs the matched-training control verified.
- `family_rollout_residual` (second batch, m = 3, registered now).
  **Candidate `residual_t1`.** Primary `residual_t1 − residual_mlp`.
  **Decision:** favourable → `residual_t1` expands against `residual_mlp`'s
  five; anything else → no expansion. Members `residual_mlp − A` and
  `residual_t1 − A` (production readings, cumulative, never triggers;
  `residual_mlp` is a production-prior control and has no expansion path).
- `family_rollout_recurrent` (deferred, m = 1, registered now). **Candidate
  `xlstm`.** Primary `xlstm − lstm`, research-only.

No family membership changes after any batch result is read; a deferred
family that is never run stays unfilled. **No selection across families**:
each decision is taken inside its family, and the ledger row for a batch
(SQ5 for the first batch) is gated on **`family_rollout_c114`'s primary**,
with the other members and families as supporting gates; the row's verdict
is driven by that primary alone. `family_rollout_mlp114` is reported in the
same row as a supporting gate; if it alone triggers expansion, that
expansion is logged under its own idea id.

Expected outcome, written before any run: the hypotheses are informed by the
teacher-forced results; the C114 teacher-forced ordering may or may not
survive rollout; Stage 1's C−B is re-measured on a fresh batch-2 checkpoint
of the same seed number; no arm advances on one seed.

## 4. What this does not change

- The cohort stays `DEFERRED_UNOPENED`. The ledger runs on the iteration set
  only; golden, forward and cohort windows are never read.
- Training stays as it is: fresh checkpoints per stage, five seeds, queue
  runner. The ledger changes only what happens to a checkpoint afterwards.
- The simulator engine is untouched; the adapter and launcher grow under
  their own version.

## 5. Astra round 1: answers adopted

1. D4 and the cache-hash equality check are kept (§ 2.3).
2. Full-prefix versus step is a debugging aid; full per-checkpoint parity
   stays mandatory (§ 2.3).
3. A loaded production scorer may be shared in-process; computed results are
   shared only for the identical pre-ball state; buffers isolated (§ 2.1).
4. The mini is allowed after qualification (§ 2.8).
5. T4 rows stay in place, linked as history (§ 2.6).
