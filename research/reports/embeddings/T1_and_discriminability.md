# T1 (innings transformer) + discriminability test — rung report

2026-08-08 · branch `embeddings-ladder` · scripts/transformer_t1.py,
scripts/discriminability_test.py

## T1 — verdict: LANDED (the sequence hypothesis is real)

2-layer, 128-d causal transformer over innings sequences; tokens =
proj(EB batter 18 + EB bowler 18 + venue 6 + state 8) + embedding of
previous ball's outcome; 298,758 params, seed 42, early stop ~epoch 29.

| per-ball LL | val | test |
|---|---|---|
| **T1 transformer** | **1.4372** | **1.4288** |
| linear control (B0b+ctx) | 1.4508 | 1.4404 |
| production XGBoost (B0c, 114 feats) | — | 1.4253 |
| T1 unseen-pairs slice | 1.4362 | **1.4245** |

- **+0.0136 val / +0.0116 test over the linear control** — an order of
  magnitude larger than any E-ladder effect, and ~4x the seed-noise floor.
- **Within 0.0035 of production XGBoost** with no ELO, no form windows, no
  venue identity, no tuning, innings scope only, first attempt.
- **Wins every frequency bucket** vs the control, including unknowns
  (1.4342 vs 1.4474): sequence context substitutes for player history —
  watching an unknown player's innings so far tells you who they are.
- On unseen pairs T1 (1.4245) beats even XGBoost's overall number.

Val buckets: unknown 1.4342 · 1-199 1.4367 · 200-999 1.4344 ·
1000-4999 1.4436 · 5000+ 1.4051.

## Discriminability — players ARE identifiable from outcomes

Same-player-vs-different-player classification from two random 600-ball
halves (signatures: phase-wise outcome dists + wide rate + boundary/dot):

| role | unmatched AUC | type-matched AUC |
|---|---|---|
| bowler (n=405) | 0.8155 | 0.7426 |
| batter (n=369) | 0.8282 | 0.8095 |

## The synthesized story (paper backbone)

1. Identity information in per-ball outcomes is saturated by EB-shrunk
   marginals (E-ladder); learned point vectors add nothing.
2. Yet players are individually identifiable from outcome signatures
   (AUC 0.74–0.81 type-matched) — style exists but is nearly orthogonal
   to single-ball expected outcome. (Repo's own I8 gate failure is
   independent confirmation: phase-conditioned EB features didn't clear
   the noise gate either.)
3. **Within-innings sequence context is where the real information was:**
   worth ~0.012–0.014 LL, nearly the entire engineered-feature stack's
   edge, from 300k params. It especially rescues thin-data players —
   lateral generalization achieved through history rather than identity.

## Next

- T2: add ELO/form features to tokens (close the 0.0035 to B0c; then
  exceed it), venue identity; calibration + reliability curves.
- Multi-task with DeepCrease targets (shot/line/length/control, 680k
  joinable balls 2021–26, join = p_match=cricsheet id + (inns, over-1,
  ball)) — style vocabulary for the representation question.
- T4: PredictionModel wrapper -> simulator rollouts, prop pricing,
  Polymarket calibration.
- Caveats: single seed; innings-scope (chase context via features only);
  DeepCrease/odata provenance unlicensed — internal training/validation
  only, commentary-derived annotations remain the open-data path.
