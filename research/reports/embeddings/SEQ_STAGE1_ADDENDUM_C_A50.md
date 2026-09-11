# Stage 1 addendum — the transformer against the 50-feature XGBoost (post-hoc)

Written 2026-09-11, after `SEQ_STAGE1_REPORT.md` was rendered and stage 1 was
gated, committed (`4109fc1`) and given its verdict (`SQ1 FAILED`, meaning
advancement not established). This is an **addendum, not an edit**: the stage
report and the twelve registered gate JSONs are unchanged.

## Why this exists

The registered confirmatory family was C−B, B−A and C−A, with A50−A as the
single registered exploratory contrast. **C−A50 was never registered and
never run.** After reading the stage result the user observed that, at equal
information (the same 50 features), the transformer appears to beat both
other architectures, which is a statement about C−B *and* C−A50. The second
half of that was arithmetic on two other contrasts, not a measurement, so it
was measured.

## Status of this number

**Post-hoc exploratory.** It was chosen after seeing the stage result, it is
outside the registered family, it carries no multiplicity adjustment, and it
advances nothing. It does not change the stage 1 verdict, which rests on
B−A and C−A, both inconclusive.

## The contrast

`claim_gate --kind match_model`, candidate C, baseline A50, odds role
`odds_iteration_v2`, clusters `data/polymarket_test_v2`,
`tournament_time_block_v1` with 10,000 seed-42 whole-event resamples. Same
merged 1,600-simulation outputs the registered gates used. Δ is candidate
minus baseline in winner log loss, so negative favours the transformer.

| slice | n | blocks | ΔLL point | 95% block interval | gate verdict |
|---|---|---|---|---|---|
| ≥$50k (primary) | 167 | 18 | −0.02857 | [−0.04617, −0.01008] | PROMISING |
| ≥$100k | 110 | 11 | −0.02901 | [−0.06190, −0.00853] | PROMISING |
| all | 252 | 25 | −0.02037 | [−0.03550, −0.00300] | PROMISING |

Gate JSONs: `models/embeddings/seq_stage1/full/gate/posthoc_C-A50_{50000,100000,all}.json`
(gitignored with the rest of the run outputs). `PROMISING` is the gate's own
LANDED-rule readout for a single-checkpoint pair, not a stage-1 label.

## What it does and does not support

Supported, as an exploratory reading on one checkpoint per arm: **given the
same 50 features, the transformer scores better than both the token MLP
(C−B −0.0257 [−0.0331, −0.0135]) and the XGBoost of the production family
(C−A50 −0.0286 [−0.0462, −0.0101]), on every slice, with intervals excluding
zero and point estimates past the 0.007 margin.** The two comparisons are not
independent: they share arm C, and C−A50 shares the A50 arm with the
registered A50−A contrast.

Not supported by this number:

- **Nothing about sequence memory specifically.** C differs from B in
  outcome-history embedding, attention and parameter count together, and from
  A50 in model family entirely. The registered rule already forbids reading
  C−B as a measurement of sequence memory, and that applies here.
- **Nothing about production.** Production is A, with 114 features. The
  registered C−A contrast is inconclusive (−0.0009 [−0.0277, +0.0288]), so
  the transformer is not shown to beat the model actually in use.
- **No market claim.** Every gate in stage 1 is provisional (one training
  checkpoint per neural arm) and no cost-scenario analysis was run for this
  pair.

## How it is being used

The user's decision, 2026-09-11: treat the transformer as the better
architecture at equal information and proceed to stage 2 on that basis,
with the confirmation work (five seeds in rollout, a matched teacher-forced
score) moved to the backlog rather than run first. Stage 2's own protocol
re-tests the vanilla-T1-versus-token-MLP question at ball level on five
seeds, so it independently re-measures the C−B half of this reading.
Recorded in `TODO.md` and `docs/SEQUENCE_TRACK_PLAN.md`.
