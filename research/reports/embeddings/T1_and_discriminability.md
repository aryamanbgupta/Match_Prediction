# T1 (innings transformer) + discriminability test — rung report

2026-08-08 · branch `embeddings-ladder` · scripts/transformer_t1.py,
scripts/discriminability_test.py

## T1 — claim status: NONLINEAR TOKEN GAIN SUPPORTED; SEQUENCE GATE FAILED

2-layer, 128-d causal transformer over innings sequences; tokens =
proj(EB batter 18 + EB bowler 18 + venue 6 + state 8) + embedding of
previous ball's outcome; 298,758 params, seed 42, early stop ~epoch 29.

The original seed-42 score was 1.4372 / 1.4288, but its quoted control omitted
four continuous state variables. The completed exact-information matrix uses
the same 50 pre-ball features for every arm and five MPS seeds:

| per-ball LL, mean ± seed SD | validation | test |
|---|---:|---:|
| converged logistic | 1.443294 | 1.434040 |
| **token MLP** | **1.438116 ± 0.000306** | **1.429320 ± 0.000288** |
| attention disabled | 1.440668 ± 0.000919 | 1.432332 ± 0.000586 |
| outcome history masked | 1.440416 ± 0.001501 | 1.431773 ± 0.001310 |
| **full T1** | **1.437331 ± 0.000648** | **1.428934 ± 0.000573** |
| production XGBoost (114 features; reference only) | — | 1.4253 |

The MLP beats logistic by 0.005177 / 0.004720 with paired two-level seed-and-
match bootstrap intervals below zero. Nonlinear current-state interactions
therefore explain most of the original headline.

Full T1 beats the MLP by only **0.000785 validation / 0.000387 test**. The
two-level intervals cross zero: [−0.001736, +0.000229] and
[−0.001215, +0.000489], although direction is favorable in 4/5 seeds. The
registered sequence gate fails on both splits.

The residual is heterogeneous. Full T1 consistently helps in middle overs
(−0.001580 validation / −0.002062 test) but consistently hurts in death overs
(+0.004615 / +0.005342, 0/5 seeds) and gives no second-innings benefit. This is
a useful mechanism diagnostic, not an overall promotion result. Full details:
`research/reports/embeddings/T1_ABLATION_V1_MPS.md`.

## Discriminability — preliminary outcome-signature separability

Same-player-vs-different-player classification from two random 600-ball
halves (signatures: phase-wise outcome dists + wide rate + boundary/dot):

| role | unmatched AUC | type-matched AUC |
|---|---|---|
| bowler (n=405) | 0.8155 | 0.7426 |
| batter (n=369) | 0.8282 | 0.8095 |

These AUCs use ordinary row-level five-fold CV over constructed pairs, not
player-grouped folds, and the signatures are not residualized for league, era,
venue, opposition, or role. They show separability of raw outcome environments;
they do not yet isolate intrinsic player style.

## Audited synthesis

1. In the tested E-ladder, EB-shrunk marginals capture the useful identity
   signal better than the learned point-vector variants; those variants add
   nothing at equal information.
2. Raw outcome signatures are separable at AUC 0.74–0.81 under the current
   pair-CV protocol. A grouped, context-residualized test is required before
   interpreting that separability as style.
3. **Nonlinear current-state modeling is supported; a broad sequence gain is
   not.** The token MLP captures most of T1's edge. Full T1's small residual
   fails the preregistered uncertainty gate and reverses sharply in death overs.

## Next

- Audit simulator parity and the first-innings rollout bias before any
  calibration or production promotion.
- After parity, test sequence modeling as a residual over the token MLP or
  production logits, with explicit death-over and chase harm gates.
- Keep the discriminability and DeepCrease work internal/preliminary until
  grouped residualized validation and releasable provenance are available.
