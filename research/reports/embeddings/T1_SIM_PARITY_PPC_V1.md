# T1 simulator parity and posterior-predictive audit

2026-08-12 to 2026-08-13 · configs `t1_sim_ppc_v1.yaml`,
`t1_sim_extras_graft_ppc_v2.yaml`, and
`t1_sim_roster_bowling_ppc_v1.yaml` · ordinary test cohort only · no golden or
forward holdout read · 200 uncalibrated draws per match/arm.

## Final verdict: structural simulator PPC passes

The repaired T1 simulator passes exact training/live feature parity, storage
symmetry, the corrected empirical-extras comparison, realistic active-bowling
unit size, score and winner guards, and the registered batting-order behavior
check. No post-hoc ball or match calibration was applied.

The original flat-extras and pre-first-over-fix PPC numbers are retained only
as superseded diagnostics. A defect discovered during the audit made the first
over bypass the configured bowling selector and use lineup index zero. The
engine now routes the first over through the same configured policy as every
later over, and a focused regression test enforces that contract.

## Exact feature and state parity

Across all 255 ordinary test matches, the audit replayed 59,291 regulation
deliveries and 183 earlier same-day context fixtures. Thirty super-over rows
were excluded because this simulator implements two regulation innings.

| check | result |
|---|---:|
| rows above registered `1e-6` tolerance | **0** |
| maximum absolute feature difference | **2.384e-7** |
| storage-swap comparisons in final PPC | **480** |
| storage-swap mismatches | **0** |

Corrections included causal per-delivery EB updates, same-day chronological
replay, the exact run-rate formula after an opening extra, cache/reset and
position-capacity checks, complete state copying, legal ball-119 extras, and
explicit `inclusive_total_runs_v1` delivery semantics. League and roster
fallbacks use only complete prior years.

## Corrected isolated extras comparison: PASS

The matched comparison uses the same 24 frozen matches, empirical B10 bowler
selector, fixed first-over engine, checkpoint, chronology, seed, and 200 draws.
Only extras event rates and event-run laws differ.

| diagnostic | corrected flat | empirical B18 | observed / paired result |
|---|---:|---:|---:|
| extras events per innings | 2.274 | **5.127** | 4.583; candidate error +0.544 |
| first-innings score bias | -6.296 | **-0.288** | shift +6.009 [+5.593, +6.429] |
| first-innings score MAE | 31.205 | **30.153** | improved |
| P10-P90 coverage | 0.667 | 0.667 | unchanged |
| winner log loss | 0.6763 | 0.6742 | -0.0021 [-0.0380, +0.0321] |
| winner Brier | 0.2416 | 0.2408 | contextual; no winner-gain claim |

All three registered extras gates pass: event frequency is within one event per
innings, absolute first-innings bias improves, and first-innings MAE improves.
The winner-metric difference is indistinguishable from zero. The sidecar uses
wide probability 0.037702 and no-ball probability 0.004409; its SHA-256 is
`ad6e863b1a2dfa3b47259d3d952157f392a0579af960cf597bcb90d2eb68bfa1`.

## Causal roster-aware bowling PPC: PASS

The active bowling unit is sampled from strictly prior complete years, players
are weighted using causal B10 usage evidence, and 20-over quotas are allocated
with schedule-feasibility constraints. The final 24-match actual-order plus
order-flipped run passes all five registered gates.

| diagnostic | B18+B10 baseline | roster candidate | observed / result |
|---|---:|---:|---:|
| unique bowlers, innings 1 | 7.370 | **6.010** | 6.125; error -0.115 |
| unique bowlers, innings 2 | 6.885 | **5.949** | 6.000; error -0.051 |
| first-innings score | 170.754 | 169.728 | 171.042; bias -1.313 |
| first-innings MAE | 30.153 | 30.310 | +0.156; guard ≤ +2 passes |
| winner log loss | 0.6742 | 0.6839 | +0.0098 [-0.0226, +0.0404] |
| storage swaps | — | 480/480 exact | pass |

The same-team batting-first-minus-chasing effect is -3.97 percentage points
[-5.40, -2.61]. The ordinary 252-match decided-test reference has a descriptive
chasing advantage of +7.94 points with interval [-4.76, +20.63]. Because
batting order is a behavior-changing intervention, the registered gate asks
whether the simulated effect is reference-compatible rather than forcing it to
zero; that gate passes. The reference remains descriptive, not a controlled
causal estimate.

Final innings-level means are close on totals: innings one 169.73 simulated
versus 171.04 observed and innings two 153.00 versus 152.92. Remaining
diagnostic discrepancies include innings-two wickets (6.29 versus 5.63),
innings-two duration (108.61 versus 103.42 legal balls), and late chase scoring
(22.91 versus 16.92 death-over runs), partly reflecting chase truncation.
These are monitoring targets, not failures of the registered structural PPC.

## Reproducibility and decision

Deterministic builders reconstruct the extras sidecar and bowling-roster
policy; configs freeze cohorts, seeds, gates, and checksums; ignored raw draws
remain durable locally; human-readable results are tracked here. Focused tests
cover feature construction, causal caches, first-over selection, extras
composition, state copying, storage symmetry, causal bowling priors, roster
feasibility, and order-behavior interpretation. These contracts also run in
`.github/workflows/embeddings-contracts.yml`.

The structural simulator is now eligible for downstream controlled use. This
does not resurrect the broad T1 sequence claim, establish market-level winner
superiority, or authorize calibration by itself. Raw and calibrated outputs
must remain separate if calibration is later evaluated.
