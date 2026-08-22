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
