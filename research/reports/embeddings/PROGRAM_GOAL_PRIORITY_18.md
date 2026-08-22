# Goal prompt — defensible CricML process metrics and prediction program

Use this as the standing execution prompt for the embeddings, sequence,
simulator, and expected-runs work.

> Role: the standing prompt and priority list. Closure summary lives in
> `PROGRAM_STATUS_20260813.md`; verdict deltas in
> `CLAIM_STATUS_AUDIT_20260811.md`; reading order in `README.md`.

## Objective

Harden CricML's embeddings/sequence/xR work into a defensible, reproducible
process-metric and prediction program. Never read or use sealed golden or
forward holdouts. Preserve valid negative results, separate teacher-forced
measurement from free-running prediction, and retire hypotheses that fail fair
controls.

## Priority work, in order

1. Audit and rewrite the E1-E4, discriminability, T1/T1.5, xR1, and T4
   verdicts. Keep valid negatives; downgrade unsupported claims about sequence
   gain, exposure bias, first-innings bias, and xR.
2. Run an identical-information T1 ablation on the same 50 causal pre-ball
   features: converged logistic, per-token MLP, attention-disabled transformer,
   history-masked transformer, and full T1. Use at least five neural seeds,
   paired match-clustered uncertainty, calibration, and thin-player/phase/
   innings slices.
3. Audit simulator parity end to end: empirical bowling selection,
   production-aligned extras/delivery semantics, exact training-versus-live
   features, first-over selection, cache/reset behavior, state copying, team-
   label symmetry, and batting-order sensitivity. Use at least 200 simulations
   and posterior-predictive checks before any promotion.
4. Do not use post-hoc match calibration until structural parity and behavior
   pass. Whenever calibration is eventually tested, report raw and calibrated
   results side by side and never use calibration to conceal a mechanical
   defect.
5. Build same-cohort xR/xW with identical rows, heads, capacity, and training
   recipe for context-only, +realized line/length, +realized shot, and +clean-
   contact/control. Express outputs in expected team runs and wicket
   probability. Measure delivery value, shot selection, contact value,
   execution residual, calibration, uncertainty, and coverage. Keep
   DeepCrease-derived work internal unless provenance becomes releasable.
6. Make every experiment reproducible: configurable paths, focused tests,
   committed configs/reports, deterministic builders, checksums, seeds,
   environment metadata, durable raw outputs, and a clean `uv` dependency
   lock. Do not install Python packages with pip; use `uv add` when needed.

## Expansion backlog, ranked

7. Keep permanent team/innings storage-symmetry and counterfactual behavior
   tests in CI.
8. Lock one decision payload, preferably luck-adjusted player/auction
   valuation, before optimizing models against it.
9. Test a hybrid residual transformer over frozen production logits rather
   than replacing the production model wholesale.
10. Build a dedicated nonlinear chase-pressure model only if controlled
    batting-order analysis shows residual chase misfit.
11. Separate causal event channels for extras, wickets, batter runs, dismissal
    attribution, and legal-ball advancement.
12. Add multi-horizon ball/over/innings/chase objectives, with each horizon
    evaluated on its own units and cohort.
13. Add match/innings latent conditions only if PPCs show missing within-
    innings correlation or dispersion after mechanical fixes.
14. Learn the bowling-selection policy—active unit, over quotas, phase roles,
    and spell persistence—from strictly prior matches.
15. Test frozen sequence representations in direct match, form, props, batting
    order, and valuation models.
16. Learn context-residualized contrastive/hierarchical player-style
    representations with player, league, and season holdouts.
17. Model sparse-player uncertainty explicitly with adaptive EB shrinkage and
    propagate it into decision outputs.
18. Validate player xR at N={50,100,250,500,1000} for future prediction versus
    raw, EB, Delta, and RAA baselines, including residual persistence.

## Evidence and decision rules

- Use identical information and cohorts for attribution comparisons.
- Use repeated seeds plus paired match/player-clustered uncertainty.
- Treat realized delivery/shot/contact arms as post-delivery measurements, not
  forecasts.
- Express xR in runs and xW as wicket probability; do not relabel
  classification log loss as expected runs.
- A simulator must pass exact feature/state parity, label symmetry, realistic
  innings PPCs, and explainable batting-order behavior before calibration or
  promotion.
- Promote a process metric only after it improves future small-sample
  prediction or the locked decision payload.
- Register configs before inspecting results. Keep raw ignored artifacts
  durable locally, and commit human-readable summaries plus checksums and
  deterministic rebuild paths.
- Report failed gates plainly and stop investing in hypotheses whose controlled
  evidence is negative.

## Definition of done

The program is successful when fair controls and repeated-seed uncertainty are
complete; simulator parity/invariance and behavior are clean; xR/xW has a
same-cohort staged decomposition in explicit units; at least one process metric
improves future small-sample prediction or the locked valuation decision; all
evidence is reproducible; and failed ideas are honestly retired.

## 2026-08-13 execution status

Priorities 1-6 are complete. The fair T1 sequence gate failed while the
nonlinear current-token gain survived; exact simulator parity and the final
empirical-extras/roster-aware PPC pass; same-cohort xR/xW decomposition is
complete; and configs, checksums, deterministic builders, focused tests, and CI
contracts are present. No sealed golden or forward holdout was read.

Expansion 7 is implemented through permanent focused symmetry, first-over,
order-behavior, and causal-policy tests plus the embeddings CI workflow.
Expansion 14's causal roster-aware bowling policy is also implemented and
passes PPC.

Expansion 8 is locked as `luck_adjusted_auction_shortlist_v1`, but it is **not
promoted**: the first expansion-18 future-player validation fails. Shot-xR/
Delta is worse than EB at N=50 and has no clean advantage at N=100 or 250;
N=500 is underpowered and N=1000 unavailable. Keep the payload internal and do
not optimize decisions against it. The next defensible research move is a
broader/open annotation cohort or a materially revised metric that can beat EB,
not proceeding automatically to priorities 9-13.
