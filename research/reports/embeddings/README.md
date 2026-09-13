# Embeddings / sequence / simulator / xR program — report index

Reading order for the 2026-08 embeddings-ladder program. Every verdict below
uses the calibrated claim-status language from the 2026-08-11 audit; no sealed
golden or forward holdout was read anywhere in this directory.

## Program-level documents (read these first)

| Doc | Owns |
|---|---|
| [`PROGRAM_GOAL_PRIORITY_18.md`](PROGRAM_GOAL_PRIORITY_18.md) | The standing execution prompt: objective, priority list, evidence rules, definition of done. |
| [`PROGRAM_STATUS_20260813.md`](PROGRAM_STATUS_20260813.md) | Closure summary: which priorities completed, headline numbers, what was deliberately not started. |
| [`CLAIM_STATUS_AUDIT_20260811.md`](CLAIM_STATUS_AUDIT_20260811.md) | What changed versus the originally committed verdicts — which claims were downgraded, withdrawn, or upheld, and why. |

When these disagree, the most specific experiment report wins; the program
docs summarize, they are not the evidence.

## Experiment reports, in ladder order

1. [`E1.md`](E1.md) — pure-ID player embeddings. Supported negative vs the EB
   baseline (single seed).
2. [`E15_E2.md`](E15_E2.md) — E1.5 learned backoff (landed) and the tabled E2
   context rung.
3. [`E4.md`](E4.md) — EB-anchored embeddings. Negative vs the linear control.
4. [`E3_and_phase1_exit.md`](E3_and_phase1_exit.md) — season vectors negative;
   Phase-1 exit summary for the whole embedding ladder.
5. [`T1_and_discriminability.md`](T1_and_discriminability.md) — T1 innings
   transformer + discriminability probes. Nonlinear current-token gain
   supported; the sequence claim is settled by the ablation below.
6. [`T1_ABLATION_V1_MPS.md`](T1_ABLATION_V1_MPS.md) — the registered
   five-arm exact-information ablation (final).
   [`T1_ABLATION_V1_INTERIM.md`](T1_ABLATION_V1_INTERIM.md) is the superseded
   partial run, kept as registration evidence only.
7. [`T15_multitask.md`](T15_multitask.md) — T1.5 multi-task aux heads
   collapse to base rates; negative.
8. [`XR1_two_stage.md`](XR1_two_stage.md) — the xR two-stage decomposition
   (mechanism supported as measurement).
9. [`T4_rollout_eval.md`](T4_rollout_eval.md) — Monte Carlo rollout
   evaluation: original run failed; documents the first-over selector defect,
   orientation bias, and the corrected-engine closure.
10. [`T1_SIM_PARITY_PPC_V1.md`](T1_SIM_PARITY_PPC_V1.md) — exact feature
    parity, B18 extras PPC, and roster-aware bowling PPC for the corrected
    simulator (pass).
11. [`XR_SAME_COHORT_V1.md`](XR_SAME_COHORT_V1.md) — staged same-cohort
    xR/xW decomposition; realized-information gates pass (measurement, not
    forecasting).
12. [`XR_FUTURE_PLAYER_VALIDATION_V1.md`](XR_FUTURE_PLAYER_VALIDATION_V1.md)
    — the future-player promotion gate. **Fails**: Delta/shot-xR does not
    beat EB at N=50/100/250, so the auction payload stays internal.

## Sequence and embeddings track (2026-09, branch `embeddings-ladder`)

The 2026-08 program above closed with three open findings; the track
re-opens them under `docs/SEQUENCE_TRACK_PLAN.md` (v5, plan of record).

13. [`SEQ_STAGE1_REPORT.md`](SEQ_STAGE1_REPORT.md) — stage 1: four ball
    models through the fixed simulator on 255 fixtures x 1,600 simulations.
    The transformer beats the token MLP CI-clean in rollout (C-B -0.0257)
    and ties production (C-A inconclusive) without the hand-built history
    features; the teacher-forced ablation had put the pair at -0.0004.
    Not promoted to production (ledger `SQ1 FAILED` = not promoted, not
    inferior). Priority follow-up: C114, transformer on all 114 features. Single checkpoint per arm; every
    gate provisional; no market claim.
14. [`SEQ_STAGE1_ADDENDUM_C_A50.md`](SEQ_STAGE1_ADDENDUM_C_A50.md) —
    post-hoc: the transformer also beats the 50-feature XGBoost
    (-0.0286 [-0.0462, -0.0101]). Outside the registered family, advances
    nothing, recorded because the equal-information reading rests on it.
15. `SEQ_STAGE2_REPORT.md` + `SEQ_STAGE2_FIVE_SEED_ADDENDUM.md` — stage 2:
    sixteen sequence variants, five seeds, teacher-forced validation only.
    Finding: forgetting helps — fixed decay (−0.0040), xLSTM (−0.0040) and
    the production-residual control (−0.0055) beat the memory-less token
    MLP CI-clean on 5/5 seeds; the death-over harm of keep-everything
    attention reproduces (+0.0046, 0/5); no mechanism is isolated and the
    two-seed aligned-history claim was withdrawn at five seeds. Ledger row
    SQ2 DESCRIPTIVE (validation-only). Cohort unopened.
16. `SEQ_STAGE3A_NIGHT3_REPORT.md` — stage 3a negative-transfer screen, two
    targets, five seeds. Finding: pooling all tiers does not hurt the
    premium or elite tiers; target-only arms are worse on their own rows
    (premium +0.0044 on 6 blocks, descriptive; elite +0.0019, unresolved);
    tier conditioning is null. Equal-epoch replication is backlog. Ledger
    row SQ3.
17. `STAGE3C_LEAKAGE_AUDIT.md` — stage 3c masked pretraining is CLOSED on
    the 50-feature contract: the masked outcome is recovered exactly from
    scoreboard and EB-tracker differences. Recorded under SQ3.
18. `SEQ_STAGE3_BATCH2_REPORT.md` — stage 4 rung 4d and block C114, five
    seeds. Finding: on the production 114-feature contract the token MLP
    (1.4277) is the best teacher-forced ball model of the track and the
    transformer (1.4306) is CI-clean worse (+0.0029, 0/5); the 64
    hand-built history features are worth 0.0068 to the transformer;
    exposure/spread/recency features add nothing; rung 4b produced no
    admissible runs (driver defect, re-registered as
    `seq_stage3_batch2_4b_v2`). Teacher-forced, not rollout; the
    rollout-vs-teacher-forcing gap goes to
    `docs/sequence_track/rollout_protocol.md`. Ledger row SQ4 DESCRIPTIVE.

## Estimator conventions

Two different seed-aware bootstrap estimators appear across these reports and
carry distinct labels (implementation:
`scripts/registered_experiment.py`):

- **seed-mean+match** (T1 ablation): CI of the across-seed mean.
- **seed-draw+match** (xR reports): one fitted seed per replicate; wider,
  includes single-seed variance.

## Configs and raw artifacts

Registered YAMLs live in `experiments/configs/` (`t1_ablation_v1*.yaml`,
`t1_sim_*.yaml`, `xr_*.yaml`). Registered configs already used by a run are
frozen — supersession is recorded here and inside successor configs
(`predecessor:` keys), never by editing a consumed file. Raw run outputs live
under `models/embeddings/` (local, gitignored); durable JSON summaries under
[`artifacts/`](artifacts/).
