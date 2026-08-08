# xR1 — two-stage delivery decomposition: rung report

2026-08-08 · scripts/transformer_xr.py · labels via scripts/deepcrease_join.py
· all staged heads read DETACHED encoder states (measurement, not training
signal — T1.5's interference eliminated by construction).

## Verdict: LANDED — the xR decomposition is quantified

**Main objective preserved exactly** (the detachment worked):
val 1.4378 / test 1.4290 vs T1's 1.4372 / 1.4288 (within noise).

**The headline — what knowing the delivery is worth** (same labeled balls,
same encoder state; only difference = line+length conditioning):

| labeled-subset per-ball LL | val (n=50,383) | test (n=73,541) |
|---|---|---|
| main head (pre-ball info only) | 1.4481 | 1.4479 |
| xR head (+ realized line/length) | **1.4270** | **1.4296** |
| **delivery information value** | **0.0211** | **0.0183** |

Delivery placement is worth ~0.02 LL per ball — comparable to the ENTIRE
sequence-context gain (T1 vs linear control ~0.012-0.014) and to the state
-features gain (~0.02). This is the stage-1 quantity of the expected-runs
decomposition: xR(delivery) vs xR(context) separates bowler execution from
everything upstream.

**Reaction (given the delivery):** shot accuracy 0.325 vs majority 0.158 —
double the base rate once you know where the ball is (T1.5's pre-ball
version was AT the base rate: shot selection is a reaction, now shown
directly). Control stays at majority (0.715) even given the delivery —
mis-execution is unpredictable from public features, i.e. control ≈ the
luck term at this feature set.

**Intent (pre-ball):** line 0.474 / length 0.401 vs majorities 0.474/0.395
— with full gradient this time, bowler intent remains unpredictable from
public context. Either intent is genuinely high-entropy (plausible: mixing
is optimal), or it needs features we don't have (field settings, plans).

## The four-line xR story (for the paper)

  E[runs | context]              = 1.448   (all pre-ball public info)
  E[runs | context + delivery]   = 1.427   (bowler execution revealed)
  shot | delivery                : 2x majority (reaction is predictable)
  control | delivery             : majority  (execution noise irreducible)

## Caveats / next

- Teacher-forced line/length are REALIZED outcomes of bowler execution —
  stage 1 conflates intent and execution; a plan-vs-execution split needs
  intent priors (e.g. bowler's historical line/length mix as pre-ball
  features — cheap next experiment).
- Labeled subset only (2021+ major T20, ~40% of eval era); single seed.
- DeepCrease labels: internal use only (provenance unlicensed).
