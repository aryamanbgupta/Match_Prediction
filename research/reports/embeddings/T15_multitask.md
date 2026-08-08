# T1.5 — multi-task with DeepCrease style targets: rung report

2026-08-08 · scripts/transformer_t1.py --aux (heads: shot 24-class, line 5,
length 6, control 3; weight 0.2) · labels: scripts/deepcrease_join.py
(394k/50k/74k train/val/test balls, 100% runs-agreement on join)

## Verdict: FAILED at this configuration (two informative negatives)

**1. Multi-task hurt the main objective**: val 1.4440 / test 1.4348 vs
T1's 1.4372 / 1.4288 (−0.007, above noise floor). At 299k params with
21% label coverage, style supervision competes with rather than aids
next-ball prediction.

**2. Aux heads collapsed to base rates** (val acc vs majority):
shot 0.149 vs 0.158 · line 0.473 vs 0.474 · length 0.402 vs 0.395 ·
control 0.715 vs 0.715. From PRE-BALL context + innings history, the
model predicts none of them beyond majority class.

## Why this is expected in hindsight (and the design fix)

We asked for shot/control WITHOUT conditioning on the delivery — but shot
selection is primarily a REACTION to line/length, which in our causal
token layout is revealed-at-delivery information. The Ben Stokes paper
(SSAC 2020 finalist) predicts shot type WITH ball line/length as input.
The right structure is the two-stage xR decomposition:
  stage 1 (pre-ball): predict line/length = bowler intent — our result
  says intent is ~unpredictable from public context (line 47% ≈ majority),
  itself a claim worth verifying with a dedicated (not 0.2-weight) model;
  stage 2 (post-delivery): predict shot/control GIVEN line/length.
Also untested here: higher aux weight, larger model, labeled-subset-only
training, aux-only fine-tune. This rung establishes the baseline, not the
ceiling.

## Session scoreboard (full arc, per-ball val LL)

| model | val LL |
|---|---|
| prior | 1.4920 |
| best E-ladder (E4 anchored) | 1.4584 |
| linear control (EB+state) | 1.4508 |
| **T1 transformer** | **1.4372** |
| T1.5 multi-task | 1.4440 |
| production XGBoost (test 1.4253) | — |

Paper arc: (1) identity saturated by shrunk marginals [E-ladder];
(2) style identifiable (AUC 0.74–0.81) but orthogonal to outcomes
[discriminability]; (3) innings history is the real frontier [T1];
(4) style targets need staged conditioning, motivating the annotated/
commentary data layer [T1.5].

## Next session queue

- T2: ELO + form features in tokens; seed repetition for T1's headline;
  calibration/reliability curves.
- Two-stage aux: line/length pre-ball head; shot/control head conditioned
  on realized line/length (teacher-forced) — the xR prototype.
- T4: PredictionModel wrapper #8 → sim rollouts, prop pricing, Polymarket.
- DeepCrease caveats: labels cover 2021+ major T20 only; provenance
  unlicensed (internal use; open-data path = commentary annotations).
