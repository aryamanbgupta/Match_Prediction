# B12 executor result — B10 selector re-gated on `bowler_wkts_1plus` primary at fresh seed 44

Idea id: **B12** (P2). Claim commit `c35cd8b`. Plan: `research/handoff/B12/plan.md`.
Executed 2026-07-31. All numbers below are copied verbatim from
`research/handoff/B12/raw/gate_output.txt` (and the two run logs).

**I do not issue the verdict.** The pre-committed mapping printed by the gate
script is recorded at the bottom; the orchestrator decides.

---

## GATE 1 (PRIMARY) — `bowler_wkts_1plus` paired dBrier (b10 − blind), seed 44

```
family                                 n  drop       blind         b10     delta   95% CI (b10-blind)   flag
bowler_wkts_1plus                   3106     0      0.2578      0.2532   -0.0046   [-0.0071,-0.0021]  DOWN(better)
  (positional cross-check)          2549                                 -0.0020   [-0.0060,+0.0020]  ~noise

  GATE 1: MET
```

Seed-43 (B10) reference for the same family:
`3107 rows, blind 0.2584 → b10 0.2535, -0.0049 [-0.0075,-0.0023] DOWN(better)`;
its positional cross-check was `-0.0041 [-0.0078,-0.0005] DOWN(better)`.

**Caveat to flag for the orchestrator:** the identity-keyed pairing (the
pre-committed statistic) reproduces the seed-43 effect almost exactly
(−0.0046 vs −0.0049), but the **positional cross-check is ~noise at seed 44**
(−0.0020 [−0.0060,+0.0020]) where it was CI-clean at seed 43
(−0.0041 [−0.0078,−0.0005]). The cross-check drops rows (2549 of 3106) because
the selector change permutes bowler first-appearance order; the pre-committed
gate statistic is the identity-keyed one.

## GATE 2 — guards: no CI-clean regression

```
family                                 n  drop       blind         b10     delta   95% CI (b10-blind)   flag
top_bowler                          5831     4      0.0773      0.0771   -0.0002   [-0.0008,+0.0005]  ~noise
  (positional cross-check)          5831                                 -0.0002   [-0.0008,+0.0005]  ~noise
bowler_wkts_2plus                   3106     0      0.2154      0.2145   -0.0009   [-0.0024,+0.0007]  ~noise
  (positional cross-check)          2625                                 -0.0001   [-0.0023,+0.0020]  ~noise
batter_runs_mae                     4253     1     14.3905     14.4030   +0.0125   [-0.0427,+0.0683]  ~noise
  (positional cross-check)          3748                                 +0.0130   [-0.0461,+0.0686]  ~noise
team_first_over_mae                  522     0      3.4103      3.4225   +0.0122   [-0.0006,+0.0246]  ~noise
  (positional cross-check)           522                                 +0.0122   [-0.0006,+0.0246]  ~noise
  top_bowler                   ok
  bowler_wkts_2plus            ok
  batter_runs_mae              ok
  team_first_over_mae          ok
  GATE 2: MET
```

---

## Context (non-gate)

### G5 bowler coverage, both arms

```
  G5 coverage [blind]: 3095/3108 = 0.9958
  G5 coverage [b10]: 3105/3108 = 0.9990
```

(Seed-43 reference was identical to 4 dp: blind 0.9958 / b10 0.9990.)

### Full 32-family scan — CI-excludes-0 families (either direction)

```
    bowler_wkts_1plus                    -0.0046  [-0.0071,-0.0021]  DOWN(better)
    highest_over_runs_ou_24_5            -0.0014  [-0.0030,-0.0000]  DOWN(better)
```

**Zero families CI-clean worse anywhere in the scan** — same qualitative picture
as seed 43. Note the differences vs seed 43: `highest_over_runs_ou_24_5`
reproduces at the CI boundary (−0.0014 [−0.0030,−0.0000], same point estimate),
while `first_wicket_runs_ou_30_5` — CI-clean at seed 43 (−0.0024 [−0.0046,−0.0000])
— is ~noise at seed 44: `-0.0022 [-0.0044,+0.0000]`.

Full 33-row scan table is in `research/handoff/B12/raw/gate_output.txt`.

### B9 sim−usage `top_bowler` margin, recomputed on both seed-44 details

```
  [blind] rows=5835  Brier_sim=0.0776  Brier_usage=0.0747  sim-usage=+0.0029 CI [+0.0018,+0.0041]  UP(worse)
  [blind] head-only (both p>=2%) rows=3545  sim-usage=+0.0035 CI [+0.0018,+0.0053]  UP(worse)
  [b10] rows=5835  Brier_sim=0.0773  Brier_usage=0.0747  sim-usage=+0.0027 CI [+0.0015,+0.0039]  UP(worse)
  [b10] head-only (both p>=2%) rows=3711  sim-usage=+0.0034 CI [+0.0016,+0.0051]  UP(worse)
```

Seed-43 reference: +0.0028 blind → +0.0026 b10. The residual sim-vs-usage gap
is essentially unchanged by the selector fix at seed 44 as well.

### B10 relaxation triggers (Arm-B log)

```
  'B10 relaxation triggered' lines: 8
  'B10 ... ACTIVE' startup lines:   1
```

(8 triggers — identical count to the seed-43 B10 run; benign. The eight lines
are committed verbatim in `research/handoff/B12/raw/b10_relaxation_lines.txt`:
one Sydney Sixers death-phase and seven Bangladesh powerplay cases, all
`eligible=4`.)

---

## Unit checks

`scripts/auto/b10_unit_check.py` run **twice**, both PASS end-to-end
(`research/handoff/B12/raw/unit_check_pre.txt`,
`research/handoff/B12/raw/unit_check_pre_armb.txt`):

- `[PASS] d15 unit check 30 PASS / 0 FAIL, exit=0` (subprocess)
- `[PASS] default payload leaves _b10 None` / `[PASS] default payload has no b10_asof_usage key`
- `[PASS] weight vectors float-exact vs independent recomputation (40 lineups x 8 configs)  (max |delta| = 0.000e+00)`
- `[PASS] live select_bowler == HEAD sampling on 320 same-seed draws  (mismatches = 0)`
- `[PASS] sidecar k_usage == b9_usage_baseline.K_USAGE  (5.0 vs 5.0)`
- `[PASS] prior_balls parity on 1420 (name,date) pairs  (max |delta| = 0.000e+00)`
- `[PASS] exp_balls parity on 1420 (name,date) pairs  (max |delta| = 0.000e+00)`
- `[PASS] n=0 rows return prior_balls exactly (20 such pairs)`
- `[PASS] sidecar payload activates _b10`
- `[PASS] true debutants get a LARGER share under B10 (plan: tiny -> ~9%)  (1.153% -> 8.733%)`
- `[PASS] known bowlers keep ~their share (only renormalization)  (12.018% -> 11.385%)`
- `[PASS] unknown players' normalized share == as-of expected-balls share  (max |delta| = 0.000e+00)`
- `[PASS] models/bowler_phase_usage.json md5 unchanged  (ea0c73d3ddb48f499b6273f9a397b0e3 == ea0c73d3ddb48f499b6273f9a397b0e3)`
- `B10 UNIT CHECK PASSED: all assertions hold.`
- Known/accepted mechanism finding re-reported verbatim by the check: veteran
  never-bowlers RISE 0.270% → 0.496% (ratio 1.83) — informational, not pass/fail.
- Relaxation triggers during the no-sim table pass: 0.

Production prior `models/bowler_phase_usage.json` md5 verified
`ea0c73d3ddb48f499b6273f9a397b0e3` before both runs and after; never written.
Nothing in `models/auto/b10/` was overwritten (its seed-43 details are untouched;
all B12 outputs went to `models/auto/b12/`).

## Startup-line confirmation, per arm

Arm A (`research/handoff/B12/raw/run_blind.log`, lines 1–12):

```
Ball calibrator: vector scaling (models/xgb_v3/vector_scaling_calibrator_v1.pkl)
Loaded XGBoost v2 model with 114 features with real player stats with player metadata; venue encoder ACTIVE (467 venues)
Bowler selector: empirical
Run-out dismissal channel ACTIVE (p_runout=0.0751, nonstriker_share=0.4685)
Running prop backtest on 261 matches × 100 sims
```
`grep -c "B10 usage-aligned" run_blind.log` = **0** (correct — blind arm).

Arm B (`research/handoff/B12/raw/run_b10.log`, lines 1–14): same five lines, plus

```
B10 usage-aligned bowler selector ACTIVE (k_u=5.0)
  as-of corpus: /Users/aryamangupta/CricML/Match_Prediction/models/auto/b10/usage_corpus.pkl (7433 players); min_eligible=5, min_share=0.01
```
`grep -c "B10 usage-aligned bowler selector ACTIVE" run_b10.log` = **1**.

Both arms processed 261/261 matches (`grep -c "^  \["` = 261 each), 0 Tracebacks.

## Eval wall times

- Arm A (blind, seed 44): `Done in 2150.3s` (35.8 min)
- Arm B (b10, seed 44): `Done in 2149.6s` (35.8 min)

Both well inside the 100-min kill budget (B10 seed-43 reference was 2857 s /
2841 s; this box ran faster). Runs were strictly sequential — never concurrent.

## Artifacts produced

- `models/auto/b12/detail_blind_s44_n261.json` (11,489,982 B)
- `models/auto/b12/report_blind_s44_n261.md`
- `models/auto/b12/detail_b10_s44_n261.json` (11,505,360 B)
- `models/auto/b12/report_b10_s44_n261.md`
- `research/handoff/B12/raw/{unit_check_pre.txt,unit_check_pre_armb.txt,run_blind.log,run_b10.log,gate_output.txt}`

(`models/auto/` is gitignored — expected. `*.log` is also gitignored, exactly as
in B10, so the two full run logs live on disk but are not tracked; the committed
excerpts `blind_start.txt` / `blind_end.txt` / `b10_start.txt` / `b10_end.txt` /
`b10_relaxation_lines.txt` carry the startup banners, the `Done in …s` lines and
the relaxation triggers, matching B10's convention.)

## Commits created (both BEFORE any eval output existed)

| SHA | Message |
|---|---|
| `7617ec3` | `Auto[B12]: re-apply B10 selector — revert of a8c061b (which reverted B10 implement ad144ea)` |
| `a50f905` | `Auto[B12]: implement — re-apply B10 selector + pre-committed b12 gate (bowler_wkts_1plus primary, seed 44)` |

Step-1 verification required by the plan:
`git diff ad144ea HEAD -- scripts/sim_v1_2.py scripts/auto/b10_build_usage_sidecar.py scripts/auto/b10_gate_analysis.py scripts/auto/b10_unit_check.py`
returned **empty** — the B10 code is restored byte-identical. `b10_gate_analysis.py`
was not edited; `b12_gate_analysis.py` is a separate new file whose GATE 1 is
`bowler_wkts_1plus` and which drops the obsolete D15 drift section.

`git diff --stat c35cd8b HEAD` (claim commit → now):

```
 research/handoff/B12/plan.md                | 207 +++++++++++++++
 research/handoff/B12/raw/unit_check_pre.txt |  65 +++++
 scripts/auto/b10_build_usage_sidecar.py     |  97 +++++++
 scripts/auto/b10_gate_analysis.py           | 359 +++++++++++++++++++++++++
 scripts/auto/b10_unit_check.py              | 394 ++++++++++++++++++++++++++++
 scripts/auto/b12_gate_analysis.py           | 352 +++++++++++++++++++++++++
 scripts/sim_v1_2.py                         | 216 ++++++++++++++-
 7 files changed, 1682 insertions(+), 8 deletions(-)
```

## Anomalies / things that ran long

- No crashes, no restarts, nothing exceeded budget.
- Operational note only: `prop_backtest.py` stdout is block-buffered through
  `tee`, so the startup banner did not appear in either log until ~70 matches in.
  Both banners were verified from the log once flushed (quoted above); no
  mitigation was needed and no command was changed.

---

## Pre-committed verdict MAPPING printed by the gate script

```
GATE 1: MET | GATE 2: MET
Pre-committed verdict MAPPING (orchestrator decides): LANDED
```
