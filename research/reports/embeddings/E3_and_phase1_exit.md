# E3 (season offsets) + Phase-1 exit review

2026-08-08 · branch `embeddings-ladder` · scripts/embeddings_e1.py
`--eb-anchor --context --venue-ctx --season-offsets` · probe:
scripts/embeddings_e3_probe.py

## E3 — verdict: FAILED

Per-ball LL (expected ~flat since eval-era seasons are pinned to zero):
val 1.4651 / test 1.4557 — actually slightly WORSE than E4+venue
(1.4584/1.4500); season offsets soak up train-era variance that
generalized better through the shared tables.

**Next-season probe** (1,895 batter-season pairs, 560 batters, 2007–2023,
player-grouped 5-fold CV, target = next-season runs/ball):

| predictor | R² | Spearman |
|---|---|---|
| career-to-date raw rpb (scalar!) | **+0.2102** | **0.4449** |
| raw this-season rpb (scalar) | +0.2003 | 0.4337 |
| E3 season vector + raw rates | +0.2025 | 0.4384 |
| E3 season vector | +0.0931 | 0.3259 |
| career vector (anchor+offset) | +0.0407 | 0.2439 |

A single scalar career average predicts next-season performance twice as
well as the full learned season vector. Vectors ADD nothing to raw rates
(0.2025 ≤ 0.2102). Season offsets do beat career-only vectors (0.09 vs
0.04) — the time signal exists — but the representation as a whole encodes
far less skill information than simple averages.

## Phase-1 exit: the complete ladder

| rung | one ingredient | val LL | verdict |
|---|---|---|---|
| B0a prior | — | 1.4920 | baseline |
| E1 pure IDs | embeddings | 1.4879 | FAILED |
| E1.5 | learned UNK + emb decay | 1.4838 | LANDED (fixed its target) |
| E2 | + match state | 1.4625 | TABLED (control deflated it) |
| B0b EB logistic | — | 1.4691 | baseline |
| **B0b+ctx linear control** | — | **1.4508** | **best identity-based model** |
| E4 EB-anchored | anchor + offsets | 1.4584 | FAILED vs control |
| E3 season offsets | + time | 1.4651 | FAILED |
| B0c production XGBoost | full features | (test 1.4253) | reference |

## Phase-1 findings (the paper's baseline section)

1. **T20's churn makes generalization the central problem**: 59%/70% of
   val/test balls are unseen batter–bowler pairs; 13%/23% involve a
   player absent from training entirely.
2. **EB-shrunk marginals + linear model saturate per-ball identity
   information.** No embedding variant — pure, regularized, context-
   deconfounded, EB-anchored, season-aware, wider — beats multinomial
   logistic at equal information. The outcome surface given shrunk
   marginals is essentially linear.
3. **Point-estimate player vectors learned from outcome prediction are
   information-poor**: probes at chance for handedness/type/arm; a scalar
   career average beats them 2:1 at next-season prediction.
4. Mechanistic lessons that survive: learned UNK backoff is necessary
   and works (E1.5); uniform weight decay ≠ adaptive shrinkage; state
   features are worth ~0.02 LL to any model.
5. Production XGBoost's remaining edge over the linear control
   (1.4253 vs 1.4404 test) bounds what ELO + form + venue + richer state
   add: ~0.015. The engineered stack is well into diminishing returns.

## Where the information frontier actually is

Everything unexplored points one direction: **within-innings sequence
context** — what no EB marginal, no static vector, and no state snapshot
captures. That is the T-ladder's hypothesis (design doc, Phase 2), now
with unusually strong baselines to beat and a clean negative to motivate
it. Downstream ΔLL test (design step 4) is expected null given probes;
deprioritized in favor of T1.

## Not done / honest gaps

- Downstream ΔLL (embeddings → XGBoost features): skipped, expected null.
- Bowler-side next-season probe: not run (batter-side only).
- Neighbor panel was never convincing at any rung — no similarity claim
  survives Phase 1.
