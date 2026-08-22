# xR1 — delivery-information prototype: rung report

2026-08-08 · scripts/transformer_xr.py · labels via scripts/deepcrease_join.py
· all staged heads read DETACHED encoder states (measurement, not training
signal — T1.5's interference eliminated by construction).

## Claim status: original prototype superseded by a controlled same-cohort measurement

**Main objective preserved exactly** (the detachment worked):
val 1.4378 / test 1.4290 vs T1's 1.4372 / 1.4288 (within noise).

**The headline — what knowing the delivery is worth** (same labeled balls,
same encoder state; only difference = line+length conditioning):

| labeled-subset per-ball LL | val (n=50,383) | test (n=73,541) |
|---|---|---|
| main head (pre-ball info only) | 1.4481 | 1.4479 |
| xR head (+ realized line/length) | **1.4270** | **1.4296** |
| **delivery information value** | **0.0211** | **0.0183** |

Realized line/length reduces six-class log loss by ~0.02 relative to the main
head on the same evaluation rows. This establishes that the labels contain
outcome-relevant information. It does **not** yet isolate delivery information:
the main head is trained on all balls while the xR head is a different-capacity
head trained on the labeled modern subset. A same-cohort, same-capacity
context-only head is required.

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

## What can and cannot be reported

  log loss, main head on labeled rows        = 1.448
  log loss, delivery-conditioned head        = 1.427
  shot | delivery                : 2x majority (reaction is predictable)
  control | delivery             : majority  (execution noise irreducible)

The first two numbers are log loss, not `E[runs]`. They must not be labeled xR
until class probabilities are converted to explicit expected-run quantities
and compared with identically trained staged controls. The result is also
provisional because model selection followed main-task LL rather than a
separately selected staged-control objective.

## Caveats / next

- Teacher-forced line/length are REALIZED outcomes of bowler execution —
  stage 1 conflates intent and execution; a plan-vs-execution split needs
  intent priors (e.g. bowler's historical line/length mix as pre-ball
  features — cheap next experiment).
- Labeled subset only (2021+ major T20, ~40% of eval era); single seed.
- DeepCrease labels: internal use only (provenance unlicensed).

## 2026-08-12 same-cohort follow-up

`scripts/run_xr_same_cohort.py` removes the original capacity/cohort confound.
Four arms use the identical complete-label rows, fixed 88-column input width,
same residual MLP parameter count, training recipe, and five seeds. Unrevealed
realized fields are zeroed: context only; +realized line/length; +realized shot;
+DeepCrease control, which is the available clean-contact proxy.

Coverage is 279,779 train rows (14.9% of the long historical split), 49,709
validation rows (40.0%), and 72,801 test rows (38.9%). DeepCrease remains
internal/provenance-restricted, so these are internal process metrics.

| same-cohort result | validation | test |
|---|---:|---:|
| context LL | 1.4549 | 1.4541 |
| + line/length LL | 1.4277 | 1.4310 |
| ΔLL | **-0.0272** [-0.0305, -0.0240] | **-0.0231** [-0.0262, -0.0201] |
| + shot LL | 1.2793 | 1.2767 |
| + contact/control LL | 1.1434 | 1.1392 |
| context → delivery xR MAE change | -0.0259 | -0.0223 |
| delivery → shot xR MAE change | -0.0696 | -0.0561 |
| shot → contact xR MAE change | -0.1504 | -0.1470 |

Every staged LL improvement is favorable in 5/5 seeds with a paired
seed-and-match interval below zero. The controlled delivery-information result
is therefore supported, not merely a prototype artifact.

The heads are now expressed in run and wicket units. Six-class probabilities
are converted with a frozen train-cohort mapping for classes
`[0,1,2,4,6,wicket]` of `[0,1,2.0611,4.0108,6.0074,0.0275]` team runs; xW is
the wicket-class probability. Test xR calibration slopes are 0.942 context,
0.923 delivery, 0.947 shot, and 0.985 contact; the shot/contact arms have a
+0.062/+0.060 run mean bias on the later test era, so player aggregation needs
era-aware uncertainty rather than a silent recalibration.

The large shot/contact improvements are not pre-ball predictive gains: those
fields are realized after delivery. They support a staged process description.
At row level, persist `delivery_value = xR_delivery - xR_context`,
`shot_selection_value = xR_shot - xR_delivery`, `contact_value =
xR_contact - xR_shot`, and `execution_residual = actual runs - xR_shot`.
Cohort means can cancel even when row-level information is large, so information
value is judged by paired LL/MAE and process value by player/innings aggregates.

The first future-usefulness gate is complete and negative. Using the first N
eligible batter deliveries and the next 100 deliveries from strictly later
matches, shot-xR/Delta fails to beat validation-tuned EB at N=50, 100, or 250.
Delta-minus-EB future MAE is +0.0223 [+0.0035,+0.0419] at N=50, +0.0106
[-0.0082,+0.0298] at N=100, and +0.0181 [-0.0037,+0.0409] at N=250. Only 11
players qualify at N=500 and none at N=1000. Context-adjusted RAA is the best
point estimate at N=50/100/250 but its advantage over EB is uncertain.

Therefore retain xR/xW as an internal descriptive decomposition, do not use
the locked auction shortlist operationally, and require a broader/open cohort
or a materially revised metric to clear EB before promotion. See
`XR_FUTURE_PLAYER_VALIDATION_V1.md`.
