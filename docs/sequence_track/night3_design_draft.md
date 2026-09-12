# Sequence track — night 3 design (v6, 2026-09-12, after Astra rounds 1–4 and user decisions)

Round 1 verdict was NO SIGN-OFF with ten MUST-FIX items; this revision
answers each (mapping at the end). Scope tonight, in queue order:
Block B (3a negative-transfer screen, two targets), Block E (Stage 4
references, sidecars, 4d, 4b-if-ready), C114 (readiness-gated). Block A
(aligned history plus decay) was REMOVED by the user on 2026-09-12 before
any run: the batter–bowler history line is not a priority; it goes to the
end of the backlog in TODO.md with its v3 design preserved below for
later pickup, and no run of it is registered tonight.
Deferred with written reasons: 3b (causal and compute contract incomplete),
3c (design and audit only, functional smoke without an inspected loss).

Shared protocol, unchanged from Stage 2: i7 frame (`ball_frame_i7`,
`stats_cache_i7`), 50-feature contract unless stated, d_model 128, 2 layers,
4 heads, batch 128, lr 3e-4, patience 3, seeds {7, 13, 29, 42, 101}, laptop
seeds 7/29/42 and mini seeds 13/101 (split by seed, never by arm; the
inherited seed–machine confounding is reported, not modelled), validation
only, cohort `DEFERRED_UNOPENED`, no advancement, no market claim. Every
family: Holm within family, strict upper bounds, paired tournament-block
bootstrap (2,000 reps, seed 29, ≥10 blocks), both estimands, five complete
paired seeds and ≥4/5 favourable primary directions; below that, no screen
passes. One reviewed commit frozen on both machines before launch; nothing
is appended to a running freeze; a missing registered member is
`NOT_EVALUABLE`, never a reduced seed count.

## Evidence status of the motivation (MUST-FIX 1)

Stage 2 five-seed facts as recorded, not paraphrased: on `all`,
`fixed_decay − mlp` −0.00405 [−0.00559, −0.00263] and `xlstm − mlp`
−0.00400 [−0.00534, −0.00280] are CI-clean favourable on 5/5 seeds; FoX's
registered contrast is `fox − fixed_decay` −0.00011 [−0.00024, +0.00004],
unresolved (no detected benefit over fixed decay). **None of the three cleared the full screen**: fixed decay's death
gate U95 was +0.00350 and xLSTM's +0.00203, both above the +0.002 margin;
only the history-free `residual_mlp` cleared, and nothing advanced. The
ownership readings are unresolved point estimates (`aligned_hist − full`
−0.00074, 3/5, the two-seed CI-clean claim withdrawn; `same_entity_k30 −
recency_k30` −0.00084, 4/5). Every design choice below that follows from
those numbers is an outcome-informed new screening hypothesis, frozen here
before any new performance read; it is not confirmation.

## Block A — aligned history plus decay — NOT RUN TONIGHT (user decision; backlog)

Retained as a design record only. Nothing in this section is registered
for tonight; no config, queue entry or family is created for it.

Configs, all trained fresh (D2.2): `aligned_hist_decay` (new),
`fixed_decay`, `aligned_hist`, `mlp`.

`aligned_hist_decay`: participant-aligned history input, standard causal
wiring (it does NOT exclude other participants; it is `aligned_hist` with
the attention bias of `fixed_decay`), ALiBi bias with the same per-head
slopes as `fixed_decay`, **no positional embedding** (matching
`fixed_decay`). Consequently `aligned_hist_decay − aligned_hist` changes
decay *and* positional encoding together and is labelled so.

Family `family_aligned_hist_decay` (five members, Holm within):
- primary: `aligned_hist_decay − fixed_decay` on `all` (does aligned
  history add to decay), t = 0;
- death contrast: `aligned_hist_decay − aligned_hist` on `death`, t = 0
  (does the combined decay-plus-positional-policy change improve death
  LL; it does not isolate decay) — registered inside the family, reported
  and labelled, optional for the screen;
- all-row condition (the inherited candidate − shared-control member):
  `aligned_hist_decay − mlp` on `all`, t = 0;
- gates: `aligned_hist_decay − mlp` on `death` and on `chase`, margin
  +0.002.
Five members, Holm within. SCREEN_PASS requires the primary, the all-row
member and both gates to reject favourably; the death contrast is reported
and labelled, not required. The 4/5 favourable-direction rule applies to
the primary. Mechanism labels: a favourable primary supports "aligned
history on top of decay"; a favourable death contrast supports "the
combined change improves death LL"; an interval crossing zero is
unresolved, not absence.

Implementation: arm tables in `transformer_t1.py` (`ARM_WIRING` standard,
`ARM_HISTORY` participant_aligned, `ARM_BIAS` alibi, no pos-emb); the
`arm == "same_entity"` mask literal replaced by a registered mask property
(`ARM_MASK_OWNERSHIP`) so a later same-entity decay arm cannot silently
lose its mask; dependency certificate on trained checkpoints at all five
seeds with target-label and future perturbations and the positive control.
The same-entity + decay combination is retained as a separately controlled
follow-on, not run tonight (relay-free ALiBi is unimplemented and the
contrast would change wiring, keys, alignment and window at once).

## Block B — 3a negative-transfer screen (30 runs: two registered targets)

Two target definitions are registered before any read, each with its own
four-arm family. The pooled arms `mlp_pool` and `mlp_pool_tiercond` are
shared; each target adds its target-only arm and its row-matched control,
so there are 6 configurations and 30 runs.

- Target P (premium leagues) = frame `competition_tier == 3`, the
  pre-match event-name rule in `parsing_v2.classify_match_context`
  (`PREMIUM_LEAGUES`: IPL, BBL, CPL, PSL, SA20, ILT20, MLC).
- Target E (elite) = target P ∪ every international (frame
  `is_international == 1`) where BOTH teams are ICC full members under the
  frozen name mapping `parsing_v2.ICC_FULL_MEMBERS` (team names as they
  appear in the cricsheet `info.teams`; the mapping and any alias are
  pinned in the freeze manifest). Event name plays no role: a qualifier
  or World Cup match between two full members is in E; any match with an
  associate side is not.
- Exploratory readout slice `big3` = E ∩ matches involving at least one of
  India, Australia or England (an elite subset; matches of those teams
  against associates are NOT in it). Descriptive only: fixed contrasts
  (the same six configs' deltas), ball-weighted, row and block counts
  reported, no family and no "tier above" inference.

Six configurations, three families:

| config id | training rows | conditioning | families |
|---|---|---|---|
| `mlp_pool` | all train rows | none | shared control in all three |
| `mlp_pool_tiercond` | all train rows | tier emb + bias | candidate in `family_3a_cond`; reference in `family_3a_P`, `family_3a_E` |
| `mlp_P_tiercond` | `target_P_matches.json` | tier emb + bias | candidate in `family_3a_P` |
| `mlp_rowmatch_P_tiercond` | `rowmatch_P_matches.json` | tier emb + bias | reference in `family_3a_P` |
| `mlp_E_tiercond` | `target_E_matches.json` | tier emb + bias | candidate in `family_3a_E` |
| `mlp_rowmatch_E_tiercond` | `rowmatch_E_matches.json` | tier emb + bias | reference in `family_3a_E` |

Shared-artifact contract: exactly one checkpoint and one row-aligned
validation prediction file per (config, seed) is trained and reused by
every family that names it; checkpoint selection is the common
all-validation rule below for every config, never re-selected per target.
Disclosure: all-row selection can select against target-only performance
for the target-only arms; it is registered anyway so the artifact is
shared and the count holds. Multiplicity: claims are family-local only.
`family_3a_P` and `family_3a_E` are correlated screens on overlapping
rows and shared controls, not independent replications, and no aggregate
"either target passes" claim is made. `family_3a_cond` is registered
once. The tier-conditioning input stays
  the frame's `competition_tier`; only the training-row filter and the
  validation slice differ between targets.
The exact match lists and their hashes are frozen in the config before
training (user decision 2026-09-12 evening, before any Block B run). No market
designation is attached to the tier. The ten-block requirement is checked
on tournament blocks of the `tier3` validation slice, not on match counts;
if fewer than ten, the slice is descriptive and the screen cannot pass.

Four token-MLP arms; the three conditioned arms share **identical**
conditioning (tier embedding dim 8 added to the token input, plus a
per-tier output bias, shared weights):
1. `mlp_pool` — pooled, unconditioned (shared control for the gates);
2. `mlp_pool_tiercond` — pooled, conditioned (the plan's partially pooled
   arm and its conditioning-matched pooled control collapse to this one
   arm under shared weights; recorded as such);
3. `mlp_<T>_tiercond` — conditioned, trained on the target's rows only;
4. `mlp_rowmatch_<T>_tiercond` — conditioned, trained on a pooled subsample of
   whole TRAINING matches: matches are shuffled with seed 42 and added in
   order until the cumulative row count first reaches or exceeds arm 3's
   row count (deterministic; the overshoot is at most one match and is
   recorded). Train/validation match disjointness is asserted by the
   freeze script. The match list and its hash are frozen in the config.
   Tier embeddings and biases for tiers the tier-3-only arm never sees
   stay at initialisation and are recorded as untrained.

Optimisation control: every arm trains for exactly S optimiser steps with
no early termination; checkpoints are evaluated every E steps and the
selected checkpoint is the best validation LL among those evaluations
(selection rule frozen here: validation log loss = ball-weighted mean
negative log-likelihood over ALL validation rows for every arm, including
the target-only arms; ties broken by the earliest step). One optimiser
step = one batch of 128 innings, as in the existing loop; the final
partial batch of an epoch is kept. Integer S = 30 × ceil(n_train_innings /
128) and E = ceil(n_train_innings / 128), where n_train_innings is the
UNFILTERED pooled training innings count (16,319 → E = 128, S = 3,840),
identical for all six arms, are computed by the freeze script from the
pinned train parquet and written into the config before any
performance read.
Smaller datasets therefore see more epochs. Reported per arm: total tokens seen, target-tier tokens
seen, steps to best. Amendment recorded: equal steps, equal tokens and
equal target-tier exposure cannot all hold at the pool's proportions; equal
steps is registered and the residual confounding (tokens and exposure
differ) is stated.

Families `family_3a_P` and `family_3a_E` (four members each, Holm
within), evaluated on the target's own validation rows (`target_P` =
`competition_tier == 3`; `target_E` = the frozen E match list restricted
to validation), negative favourable:
- superiority 1: `mlp_<T>_tiercond − mlp_pool_tiercond`, t = 0;
- superiority 2: `mlp_<T>_tiercond − mlp_rowmatch_<T>_tiercond`, t = 0;
- gates: `mlp_<T>_tiercond − mlp_pool` on `death` and `chase`
  restricted to target rows, margin +0.002;
for T ∈ {P, E}, evaluated on the target's own validation rows.
The inherited all-row condition (candidate − shared control on `all`) is
explicitly amended away for this family: a tier-3-only model is not
expected to serve all rows (they are outside its intended deployment
scope), so that condition is not part of its screen. Screen: PASS only if both superiority members
reject favourably with U95 < 0 and both gates hold; the 4/5
favourable-direction rule applies to BOTH superiority members; every
required slice of EACH target must contribute ≥10 tournament blocks. Registered
separately, decision-bearing: family `family_3a_cond` with three members —
`mlp_pool_tiercond − mlp_pool` on `all` (primary, t = 0) and
`mlp_pool_tiercond − mlp_pool` gates on `death` and `chase` (margin
+0.002); PASS requires all three. Exploratory: per-tier readouts for tiers 1, 2, 4 (tier 4 has 11
validation matches; descriptive).

Implementation: `competition_tier` loaded in `load_split()`; `--tier-embed`
on the mlp arm; `--train-tier` and `--train-match-list <frozen json>`
filters in the driver; `--max-steps` and `--eval-every`; new params
admitted to driver validation, the training signature (`arm_params` and
`training_block` components) and the pin.

## Family machinery (MUST-FIX 5)

`stage2_stats.py` is generalised, not patched: explicit `role: candidate |
control` per configuration; `statistics.families.shared_control` replaces
the `"mlp"` literal; family members carry their own contrast, reference,
slice and threshold; families may have 2–6 members (six is required by 4b; validated and
tested at six); the screen condition
is registered per family (which members must reject, with which bound).
The inherited `candidate − shared control @ all` condition is never
implicit: each family either registers it as a named member (Block A,
C114, 4d) or amends it away in writing (Block B).
Holm step-down, tie order, missing-member placeholder, strict bounds, both
estimands, the 4/5 direction rule and the ten-block rule are unchanged and
covered by the existing tests plus new ones for the role and member
schema. Mechanism contrasts are read from `statistics.contrasts` entries
with `role: mechanism`, not from a Python tuple. A `tier3` slice predicate
is added (`competition_tier == 3`), gated on the column being loaded.

## C114 (15 runs, readiness-gated)

Configs: `full_114`, `mlp_114`, `full_50` (fresh). Families: `full_114 −
mlp_114` on `all` (primary; this IS the candidate − shared-control all-row
member, shared control `mlp_114`) with gates vs `mlp_114`; `full_114 −
full_50` on `all` (primary) plus the all-row member `full_114 − mlp_114`
and gates vs `mlp_114` (four members). Every new option (feature contract,
tier options, step budget, sidecar columns, reference logits, identity
vocabulary, penalties) is admitted by driver validation, enters the
training signature and the pin, and is source-pinned by sha256. Feature contract `v7_114` pinned as an ordered
column list with transformations, categorical encoders fitted on train
only, dimensions and a sha256 in the config; the driver validates it and
it enters the signature's `frame` component. Cutoff: if the contract is
not smoke-clean and pinned by 22:30 IST it runs tomorrow; nothing else
waits for it.

## Block E — Stage 4 (20 runs if both land; 4d first)

References (fitted on i7 training rows, then frozen, deterministic, one
fit each, never presented as five seeds):
- `ref_eb_ctx`: 42 EB columns + the 4 match-state columns of the E15/E2
  fair control; stated explicitly to omit the four state features the
  50-feature arms see;
- `ref_lin_50`: multinomial logistic on the full 50-feature contract (the
  reference any embedding-specific claim must beat);
- `ref_raw_ctx`: identical context and preprocessing to `ref_eb_ctx`, with
  unshrunk as-of player rates `n_c/N`, prior fallback at N = 0.
Feature lists, regularisation (L2, C = 1.0, max_iter 500), class mapping
and row-aligned outputs are pinned with hashes.

Sidecar (`models/embeddings/stage4/exposure/{split}.parquet`, row-aligned,
keyed by `innings_id` and `ball_idx`, hashed): as-of `batter_N`,
`bowler_N`, per-class counts, Dirichlet spread at k = 30, and recency
counts under the cache's strict as-of rule.

4d (10 runs): `mlp_counts` = the 50-feature contract + `batter_N`,
`bowler_N` (as-of ball counts); `mlp_spread_recency` = the same + the
twelve spread fields `{batter,bowler}_sd_{0,1,2,4,6,w}` where
sd_c = sqrt(p_c(1−p_c)/(N+k+1)), p_c = (n_c + k·π_c)/(N+k), k = 30, π =
the cache's global prior (its provenance recorded in the sidecar manifest)
+ `batter_recent_N`, `bowler_recent_N` = the ball counts inside the cache's
recent window, which is the player's five most recent match rows dated
strictly before the match date (the same window that produces the frame's
`batsman_recent_*` / `bowler_recent_*` columns; players with fewer than
five prior matches use what exists; zero when none). π is the global prior
stored in the i7 cache `_meta`; it is a whole-corpus prior computed over
the cache's source corpus, which runs to 2026-04-16 while the training
split ends 2024-12-30 (stage 1 acceptance D-record), so it carries
post-training exposure that the production features already carry; its values and the cache md5 are
pinned in the sidecar manifest and its unknown differential effect across
arms is disclosed. The arm adds spread PLUS recency; it does not isolate
spread. Family: primary
`mlp_spread_recency − mlp_counts` on `all` (this is also the all-row
member; shared control `mlp_counts`), gates vs `mlp_counts` on `death` and
`chase`; three members, PASS requires all three; both arms reported
against the three references.

4b (10 runs, only if its code passes the gate by 23:00): identity-residual
model over the frozen `ref_eb_ctx` logits: batter and bowler embeddings dim
16, vocabulary = players seen in train, UNK index for unseen; UNK receives
gradient through id-dropout 0.05 (each id replaced by UNK with p = 0.05
during training, the E4 precedent); output = frozen base logits + W·[e_bat;
e_bowl] + b. Penalty = λ · mean over batch rows of ‖r‖² where r = W·[e_bat; e_bowl] + b
is the residual logit vector (six entries) added to the frozen base — the
penalty is on the residual function itself, so no rescaling of e against W
can escape it; no other normalisation; λ ∈ {1e-3, 1e-2}, frozen. One six-member Holm family:
`4b_1e-3 − ref_eb_ctx` on `all`, `4b_1e-2 − ref_eb_ctx` on `all` (both
superiority, t = 0), and each setting's `death` and `chase` gates vs
`ref_eb_ctx` (margin +0.002). Per-setting PASS = its primary and its two
gates reject; the family reports both settings and never selects between
them. The reference is deterministic, so estimand (i) pairs over blocks
only and (ii) over seed×block on the candidate side. Each setting's
primary is itself the candidate − control all-row member. Stated difference from E4: the
base cannot move.

4a: exposure-graph audit only (report). 4c: coverage and provenance audit
only. 4e: tomorrow.

## Deferred: 3b (over patching)

Not run tonight. The contract still lacks: wides and no-balls (six legal
balls ≠ six rows), incomplete final overs, innings boundaries, the pooling
choice (mean vs order-preserving projection), a prohibition on any future
or target information entering the pool, a whole-path multilayer
certificate, and a registered total-FLOP budget for the equal-compute
control (steps-matched `full` is supplementary only). These are written
tomorrow with a fresh `fixed_decay` family if that comparison is kept.

## Deferred: 3c (masked pretraining)

Design and audit only; no training tonight. The design written tomorrow
must choose the actual visibility scheme (forward-only visible context, or
removal of every downstream channel) and freeze the same-row total-compute
baseline before any run is registered. Tonight's audit (`stage3c_leakage_audit.py`) can
flag leaks but cannot certify absence; the written design will add
deterministic dependency analysis over the feature-construction code and
adversarial reconstruction checks, specify forward-only visible context or
removal of every downstream channel (later scoreboard fields, shifted and
aligned outcome inputs), and freeze the same-row pretrain-plus-finetune
compute baseline. A functional smoke with no inspected loss is permitted.

## Validation-only helpers (MUST-FIX 11)

`stage4_references.py`, `stage4_exposure_sidecar.py` and
`stage4_pair_graph_audit.py` run on train and validation only. Input
validation happens BEFORE any read: every parquet path is resolved
(symlinks followed) and must equal the pinned train or validation file by
sha256 of its first 64 KiB and size, the cache path must equal the
`stats_cache_i7` role md5, and any other path (including a train-named
symlink to a test parquet) is refused. The preflight records a positive
check (pinned inputs accepted) and negative checks (test parquet, a
symlink to it, a golden path, a forward-holdout path all refused). No test
log loss is computed or stored.

## Capacity (MUST-FIX 10)

Measured Stage 2 wall times include laptop seed-42 outliers (mlp 1,333 s,
full 3,633 s) and mini `full` 206–286 s, relay-free 350–478 s; the budget
is built from tonight's smoke timings per arm and machine with the
runner's 2× alarm, one retry, memory floor and certification and
reporting overhead, and a whole experiment is deferred on readiness or
time, never trimmed in seeds. The per-arm/per-machine budget table, the
maximum training duration, and the queue deadline are filled into the
night runbook from the smoke before launch and are part of the frozen
commit. Readiness cutoffs are cutoffs for inclusion in that single commit:
nothing is implemented after the freeze, and an experiment not ready at
the freeze is deferred whole. Queue order: B, E(4d), C114, E(4b). The populated per-arm/per-machine
table (smoke seconds, budgeted hours with the 2× alarm and one retry,
maximum training duration per run, preparation/certification/reporting
overhead, queue deadline) lives in `docs/sequence_track/night3_runbook.md`
and is referenced from the frozen commit; launch is refused if that table
is absent.

## MUST-FIX mapping

1 → "Evidence status"; 2 → Block A; 3 → Block A implementation; 4 → Block
B optimisation and freezing; 5 → Family machinery; 6 → C114; 7 → Deferred
3b; 8 → Deferred 3c; 9 → Block E; 10 → Capacity and the one-commit rule.
SHOULD items: `ref_lin_50` added; 4c audit started; replay compatibility
and seed–machine confounding retained in the report.
