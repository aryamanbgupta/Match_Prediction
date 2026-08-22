# Embeddings / sequence / xR claim-status audit

> Role: what changed versus the originally committed verdicts. Closure
> summary lives in `PROGRAM_STATUS_20260813.md`; the standing prompt in
> `PROGRAM_GOAL_PRIORITY_18.md`; reading order in `README.md`. Numbers here
> restate the underlying experiment reports, which stay authoritative.

2026-08-11 to 2026-08-12 · branch `embeddings-ladder` · claim audit plus
registered T1 ablation and simulator diagnostics. No sealed golden or forward
holdout was read. The work used only the existing train/validation/test
parquets, ordinary test-match JSON, and corrected 255-match evaluation artifact.

## Authoritative status

| Work | Status after audit | What the evidence supports | Required next evidence |
|---|---|---|---|
| E1 pure IDs | supported negative | loses to EB logistic overall/unseen pairs | repeated seeds for uncertainty only |
| E1.5 backoff | mechanism supported | fixes E1 unknown-player pathology | not a win over EB |
| E2 context | negative vs fair control | coarse state helps; embeddings still lose | none for the tested rung |
| E4 EB anchor | supported negative | tested offsets/capacity lose to 46-feature linear control | downstream use remains untested |
| E3 season offsets | supported negative | vector loses to scalar career rate for next-season probe | bowler probe optional |
| discriminability | preliminary | raw outcome signatures are separable | player-grouped, context-residualized CV |
| T1 | sequence gate failed; nonlinear token gain supported | MLP robustly beats logistic; full-vs-MLP interval crosses zero | simulator parity; residual sequence model only after parity |
| T1.5 | supported negative at one config | auxiliaries hurt main LL; pre-ball heads near base rate | staged/same-cohort experiment |
| xR1 | same-cohort process measurement supported; future promotion failed v1 | realized line/length, shot and clean-contact proxy add stable information, but shot-xR/Delta does not beat EB in future-player MAE | broader/open annotations or a revised metric must beat EB before decision use |
| T4 | original run failed; corrected structural PPC passes | exact feature/label parity, empirical extras, roster-aware bowling, symmetry, order behavior, score and winner guards pass | downstream winner validation; monitor chase duration/death PPC |

## T1 control correction

The published 46-feature control (1.4508 validation / 1.4404 test) omits four
continuous state inputs used by T1. The registered 2026-08-11 scaled logistic
fit using the exact `transformer_t1.build_features` 50-column matrix converged
in 271 iterations and scored **1.443294 / 1.434040**. T1's apparent advantage
is therefore only **0.0061 / 0.0052**, not 0.0136 / 0.0116. The four neural
arms and repeated-seed attribution were initially pending.

The completed single-device matrix gives the current-token MLP
**1.438116 ± 0.000306 / 1.429320 ± 0.000288** and full T1
**1.437331 ± 0.000648 / 1.428934 ± 0.000573**. Full-minus-MLP is only
−0.000785 [−0.001736, +0.000229] validation and −0.000387
[−0.001215, +0.000489] test under a paired two-level seed-and-match bootstrap.
Direction is favorable in 4/5 seeds, but both intervals cross zero, so the
registered sequence gate fails.

Full T1 helps middle overs but hurts death overs by +0.004615 / +0.005342 and
provides no second-innings gain relative to the MLP. The supported finding is a
nonlinear current-state gain, not a broad sequence gain.

## T4 orientation correction

On the existing 255-match corrected-odds artifact:

- first listed team equals first-innings team: 255/255;
- mean predicted first-innings win probability: **0.5953**;
- actual first-innings win rate: **0.4549**;
- fraction with first-innings probability >0.5: **0.8627**.

Liquidity re-slicing gives LL/flat ROI: all **0.7156 / -15.62%**; >=$50k
(n=168) **0.7176 / -16.36%**; >=$100k (n=110) **0.6921 / -21.83%**. This
supports an unambiguous rollout failure and a structural innings-orientation
diagnostic. It does not identify exposure bias as the cause.

The 2026-08-12 to 2026-08-13 follow-up changes the diagnosis while retaining
the original T4 failure as historical evidence. Exact feature parity passes on
59,291 regulation rows (zero above `1e-6`) and final storage-label symmetry
passes 480/480. The audit discovered that every first over bypassed the
configured bowler selector; absolute PPC numbers from before that repair are
superseded.

Under the fixed engine, the clean B18 comparison passes all three gates:
extras move from 2.274 flat to 5.127 per innings versus 4.583 observed;
first-innings bias moves from -6.296 to -0.288; and MAE improves from 31.205
to 30.153. Winner LL changes by -0.0021 with interval [-0.0380,+0.0321], so
there is no winner-gain claim.

The causal roster-aware bowling arm passes all five final PPC gates. Unique
bowlers are 6.010/5.949 versus 6.125/6.000 observed, storage symmetry is exact,
first-innings MAE changes only +0.156, winner LL has no CI-clean regression,
and its -3.97-point batting-first effect is compatible with the ordinary
cohort reference. The defensible claim is now **the original T4 market rollout
failed, but its apparent storage/orientation signature was not stable and the
corrected simulator passes the registered structural PPC**.

## xR same-cohort correction

The controlled four-arm matrix uses identical complete-label rows, fixed input
width/capacity, and five seeds. Realized line/length beats context by ΔLL
-0.0272 [-0.0305, -0.0240] validation and -0.0231 [-0.0262, -0.0201] test,
favorable in 5/5 seeds. Expected-run MAE improves by 0.0259/0.0223 runs per
delivery. Realized shot and clean-contact proxy add much larger post-delivery
information blocks. These remain valid internal xR/xW process measurements in
explicit team-run/wicket units.

The registered future-player validation uses the first N eligible deliveries
and the next 100 deliveries from strictly later matches, tunes shrinkage only
on validation, and freezes the auction payload before test inspection. It
fails promotion. At N=50, luck-adjusted Delta is worse than EB by +0.0223 MAE
[+0.0035,+0.0419]; at N=100 and 250 it is also worse by +0.0106 and +0.0181,
with intervals crossing zero. N=500 has only 11 players and N=1000 has none.
Context-adjusted RAA is the best point estimate at N=50/100/250, but does not
beat EB with clean uncertainty. Do not optimize player/auction decisions
against xR/Delta from this cohort.

## Reporting rules after Priority 1-6 closure

1. Report T1 as a failed broad sequence gate with a supported nonlinear-token
   control; do not promote it from the small mean LL difference.
2. Do not express classification log loss as `E[runs]` or xR.
3. Do not attribute the rollout failure to exposure bias without an evolving-
   state posterior-predictive test.
4. Structural parity now passes, but any future calibration experiment must
   report raw and calibrated values side by side and may not redefine the PPC.
5. Treat local ignored model/result artifacts as evidence to be registered,
   checksummed, and summarized—not as reproducible deliverables by themselves.
6. Report xR/Delta as a descriptive internal process metric whose first
   future-player promotion gate failed; RAA is exploratory, not promoted.
