# B8 result — hybrid calibrator: stale v1 global + B7 venue-ON over-0 vector

Executor report for PROTOCOL steps 3–4. **No verdict is issued here** — the
gate script's pre-committed mapping line is reported verbatim; the
orchestrator decides.

## Headline

The hybrid is a **null change** on the current engine. GATE 1a (the
`team_first_over_mae` improvement that is the entire point of the idea) did
NOT reproduce: dMAE **+0.004 [-0.014, +0.024] ~noise**, i.e. very slightly
worse and nowhere near CI-clean. GATE 2 guards held, and no family anywhere
in the run — gate, guard, or context (32 families) — has a CI excluding 0 in
either direction.

Gate script's printed mapping: **TABLED** (GATE 1 NOT MET, GATE 2 HELD).

## Commits created

- claim `7690909` (orchestrator), plan `90c3299` (orchestrator).
- implement **`ba024ce`** — `Auto[B8]: implement — hybrid over-0 calibrator
  compose + pre-committed gate`, committed **07:11 UTC**, i.e. BEFORE the
  eval started (07:12:55 UTC). Gate script pre-committed as required.

`git diff --stat 7690909..HEAD` (at implement commit, before this result):

```
 research/handoff/B8/plan.md                   | 203 ++++++++++++++++++++++++++
 research/handoff/B8/raw/compose_output.txt    |  34 +++++
 research/handoff/B8/raw/gate_wiring_check.txt |  14 ++
 research/handoff/B8/raw/parity_check.txt      |   4 +
 scripts/auto/b8_compose_hybrid.py             | 122 ++++++++++++++++
 scripts/auto/b8_gate_analysis.py              | 188 ++++++++++++++++++++++++
 6 files changed, 565 insertions(+)
```

`models/auto/b8/` is gitignored, so `hybrid_calibrator.pkl`, the detail JSON
and the report are not committed — `scripts/auto/b8_compose_hybrid.py`
reproduces the pkl deterministically from the two input artifacts.

## Engine-state parity check — PASS

`git diff 91be8d7 HEAD -- scripts/sim_v1_2.py scripts/calibration.py`
produced **0 bytes** (empty `--stat` too). Raw:
`research/handoff/B8/raw/parity_check.txt`. The B8 challenger therefore ran
on exactly the engine that produced the b10 blind baseline.

## Baseline provenance (orchestrator ruling 1)

Baseline = `models/auto/b10/detail_blind_s43_n261.json`. From
`research/handoff/B10/plan.md` (Arm A, blind twin), the exact launch command
recorded there:

```
uv run python scripts/sim_eval/prop_backtest.py \
  --test-dir data/polymarket_test --n-matches all --n-sims 100 --seed 43 \
  --ball-calibrator vector \
  --detail-out models/auto/b10/detail_blind_s43_n261.json \
  --report-out models/auto/b10/report_blind_s43_n261.md \
  2>&1 | tee research/handoff/B10/raw/run_blind.log
```

Verified from `research/handoff/B10/raw/run_blind.log` (startup): `Ball
calibrator: vector scaling (models/xgb_v3/vector_scaling_calibrator_v1.pkl)`
— i.e. `--ball-calibrator vector` with NO `--ball-calibrator-path` override
— plus `venue encoder ACTIVE (467 venues)`, `Bowler selector: empirical`,
`Run-out dismissal channel ACTIVE (p_runout=0.0751,
nonstriker_share=0.4685)`, `Running prop backtest on 261 matches × 100 sims`,
seed 43. `research/handoff/B10/raw/blind_start.txt` records the launch at
`2026-07-30T22:28:41`. Only delta vs B8: `--ball-calibrator-path`.

## Compose step — ALL ASSERTIONS PASSED

`scripts/auto/b8_compose_hybrid.py` (raw:
`research/handoff/B8/raw/compose_output.txt`). No fitting, no val sim — pure
composition of two existing artifacts.

```
v1  _v      = [0.252126 0.298308 0.131146 0.126464 0.109106 0.082851]
b7  _v[0]   = [0.231471 0.30464  0.113281 0.119189 0.140743 0.090676]
b7  _global = [0.270071 0.323834 0.108699 0.126141 0.097031 0.074224]   <-- REFIT global, DROPPED

  [PASS] sorted(b7._v.keys()) == [0]
  [PASS] max|hybrid._global - v1._v| == 0.0  (got 0.000e+00)
  [PASS] max|hybrid._v[0] - b7._v[0]| == 0.0  (got 0.000e+00)
  (dropped refit global vs v1: max abs diff 0.025526, max|ratio-1| 0.171162)
  [PASS] max|b7._global/v1._v - 1| > 0.05  (got 0.171162)
  [PASS] sorted(hybrid._v.keys()) == [0]

  [PASS] hybrid.calibrate_probs(p, over=1)  == v1.calibrate_probs(p)  (max abs diff 0.000e+00)
  [PASS] hybrid.calibrate_probs(p, over=7)  == v1.calibrate_probs(p)  (max abs diff 0.000e+00)
  [PASS] hybrid.calibrate_probs(p, over=19) == v1.calibrate_probs(p)  (max abs diff 0.000e+00)
  [PASS] hybrid.calibrate_probs(p, over=None)== v1.calibrate_probs(p) (max abs diff 0.000e+00)
  [PASS] hybrid.calibrate_probs(p, over=0)  == b7.calibrate_probs(p, over=0) (max abs diff 0.000e+00)
  [PASS] hybrid over-0 output differs from v1 output (max abs diff 8.358e-02 > 0)

Over-0 vs global ratio: dot 0.9181 one 1.0212 two 0.8638 four 0.9425
                        six 1.2900 wicket 1.0945
  max|_v[0]/_global - 1| = 0.2900        (plan expected >~0.2)
```

The poison `_global` was dropped as specified; the hybrid is bit-exactly v1
outside over 0.

### PLAN DEVIATION 1 (assertion form, no effect on the artifact)

The plan specified `max|b7._global - v1._v| > 0.05` while stating the
expected divergence is "~0.17". Those are inconsistent: both vectors are
sum-normalised to 1, so the **absolute** max diff is only **0.025526** and
the plan's literal threshold is unsatisfiable by construction, whereas the
**relative** divergence `max|ratio-1|` is **0.171162** — exactly the plan's
own stated "~0.17". I implemented the relative form (which is the check the
plan describes) and printed both numbers. The composed hybrid is unaffected;
this changed only the assertion expression. Flagged in the implement commit
message and in a code comment.

### PLAN DEVIATION 2 (CLI flag name)

The plan's eval recipe uses `--out <report>.md`. `prop_backtest.py` has no
such argument (its options are `--detail-out` and `--report-out`; `--out` is
not an unambiguous prefix of either, so argparse rejects it). I used
`--report-out` for the report path. Everything else verbatim.

## Smoke run — startup lines verified

```
Ball calibrator: vector scaling (models/auto/b8/hybrid_calibrator.pkl)
Loaded XGBoost v2 model with 114 features with real player stats with player metadata; venue encoder ACTIVE (467 venues)
Bowler selector: empirical
Run-out dismissal channel ACTIVE (p_runout=0.0751, nonstriker_share=0.4685)
Running prop backtest on 3 matches × 5 sims
Done in 1.3s
```

Both required lines present. Raw: `research/handoff/B8/raw/smoke.log`.

## Full eval — completed, no crash, under budget

Launched detached (nohup, background), one heavy process at a time, never
`--parallel`, nothing else heavy concurrent.

```
nohup uv run python scripts/sim_eval/prop_backtest.py \
  --test-dir data/polymarket_test --n-matches all --n-sims 100 --seed 43 \
  --ball-calibrator vector \
  --ball-calibrator-path models/auto/b8/hybrid_calibrator.pkl \
  --detail-out models/auto/b8/detail_b8_s43_n261.json \
  --report-out models/auto/b8/report_b8_s43_n261.md \
  > research/handoff/B8/raw/run_b8_s43.log 2>&1 &
```

- PID 22359, start **2026-07-31T07:12:55Z**, end **2026-07-31T07:49:14Z**.
- Script-reported **`Done in 2153.9s`** (35.9 min); session wall 2179 s
  (36.3 min). Well inside the 100 min kill threshold and faster than B10's
  twins (2857 s / 2841 s).
- Completion lines: `[261/261] 1529381 ...`, `Done in 2153.9s`, `Detail
  written to models/auto/b8/detail_b8_s43_n261.json`, `Report written to
  models/auto/b8/report_b8_s43_n261.md`.
- Startup confirmed identical to the baseline except the calibrator path:
  `Ball calibrator: vector scaling
  (models/auto/b8/hybrid_calibrator.pkl)` / `venue encoder ACTIVE (467
  venues)` / `Bowler selector: empirical` / `Run-out dismissal channel
  ACTIVE (p_runout=0.0751, nonstriker_share=0.4685)` / `Running prop
  backtest on 261 matches × 100 sims`.
- Nothing crashed; nothing ran long. No background processes left running.

Raw log: `research/handoff/B8/raw/run_b8_s43.log`.

## GATE RESULTS (verbatim from `research/handoff/B8/raw/gate_output.txt`)

`blind (bare v1, venue-ON s43): 261 matches | hybrid (B8, venue-ON s43): 261 matches`

### GATE 1a — improvement: `team_first_over_mae` dMAE < 0 with CI excluding 0

```
team_first_over_mae                 522  MAE_blind=3.411  MAE_hybrid=3.415  dMAE=+0.004  [-0.014,+0.024]  ~noise
```

**NOT MET.** Direction is even slightly the wrong way. For context the plan
predicted ≈ −0.02 CI-clean if the B7/A15 gain held (B7 refit-stack −0.024
[−0.040, −0.007]; A15 venue-blind −0.018 [−0.032, −0.003]).

### GATE 1b — no-regress (no CI-excludes-0 INCREASE on the three families)

tail-pool lines:

```
family                                n Brier_blindBrier_hybrid    dBrier   95% CI (hybrid-blind)   flag
pp_total_ou_45_5                    522      0.2479      0.2493   +0.0014   [-0.0009,+0.0035]  ~noise
pp_total_ou_50_5                    522      0.2543      0.2555   +0.0012   [-0.0009,+0.0034]  ~noise
pp_total_ou_55_5                    522      0.2190      0.2198   +0.0008   [-0.0012,+0.0028]  ~noise
first_wicket_runs_ou_30_5           522      0.2370      0.2368   -0.0002   [-0.0021,+0.0016]  ~noise
highest_over_runs_ou_18_5           261      0.2339      0.2337   -0.0001   [-0.0039,+0.0037]  ~noise
highest_over_runs_ou_24_5           261      0.0976      0.0968   -0.0008   [-0.0021,+0.0005]  ~noise

  >>> pooled tail dBrier over 6 lines (2610 obs) = +0.0005  95% CI [-0.0005, +0.0017]  ->  NOISE (CI straddles 0)

bowler_wkts_1plus                  2959      0.2572      0.2567   -0.0005   [-0.0026,+0.0015]  ~noise
batter_runs_mae                    4169  MAE_blind=14.444  MAE_hybrid=14.444  dMAE=+0.000  [-0.036,+0.038]  ~noise
```

**MET** — `[pooled tail worse: no; bowler_wkts_1plus worse: no;
batter_runs_mae worse: no]`.

Notable: the hybrid confirms the B8 hypothesis' *mechanism* even though the
payoff did not appear. B7's headline tail regression (pooled +0.0079
CI-clean worse) is fully gone at pooled **+0.0005 [-0.0005, +0.0017]**, and
`bowler_wkts_1plus` moved from B7's +0.0024 to **−0.0005 [-0.0026,+0.0015]**
— i.e. those regressions really did come from the refit global, which the
hybrid drops. What did not survive is the over-0 benefit.

### GATE 2 — guards (no CI-excludes-0 increase)

```
top_bowler                         5831      0.0773      0.0774   +0.0001   [-0.0002,+0.0005]  ~noise
team_total_fours_mae                522  MAE_blind=3.688  MAE_hybrid=3.681  dMAE=-0.007  [-0.026,+0.011]  ~noise
team_total_sixes_mae                522  MAE_blind=2.993  MAE_hybrid=3.000  dMAE=+0.007  [-0.010,+0.024]  ~noise
```

**HELD.**

### Verdict-mapping line printed by the pre-committed gate script

```
GATE 1a (team_first_over_mae CI-clean improvement): NOT MET
GATE 1b (no CI-clean regression on the 3 no-regress families): MET  [pooled tail worse: no; bowler_wkts_1plus worse: no; batter_runs_mae worse: no]
GATE 1 = 1a AND 1b: NOT MET
GATE 2 (guards): HELD
VERDICT per pre-committed mapping: TABLED
```

## CONTEXT scan (verbatim; cannot flip the verdict)

```
family                                n Brier_blindBrier_hybrid    dBrier   95% CI (hybrid-blind)   flag
bowler_wkts_2plus                  2975      0.2089      0.2097   +0.0008   [-0.0004,+0.0021]  ~noise
bowler_wkts_3plus                  3045      0.0868      0.0866   -0.0002   [-0.0007,+0.0002]  ~noise
batter_50plus                      4246      0.0795      0.0794   -0.0001   [-0.0005,+0.0003]  ~noise
batter_6plus_six                   4222      0.2242      0.2250   +0.0008   [-0.0002,+0.0018]  ~noise
batter_fours_1plus                 4222      0.2199      0.2207   +0.0008   [-0.0004,+0.0021]  ~noise
batter_fours_2plus                 4226      0.2035      0.2036   +0.0001   [-0.0009,+0.0010]  ~noise
batter_fours_3plus                 4230      0.1584      0.1583   -0.0001   [-0.0009,+0.0007]  ~noise
team_highest_individual_ou_29_5     522      0.0793      0.0793   +0.0000   [-0.0008,+0.0009]  ~noise
team_highest_individual_ou_34_5     522      0.1316      0.1312   -0.0003   [-0.0018,+0.0012]  ~noise
team_highest_individual_ou_39_5     522      0.1751      0.1736   -0.0015   [-0.0034,+0.0005]  ~noise
innings_runs_ou_160_5               522      0.2511      0.2498   -0.0013   [-0.0038,+0.0012]  ~noise
innings_runs_ou_170_5               522      0.2414      0.2415   +0.0001   [-0.0025,+0.0028]  ~noise
innings_runs_ou_180_5               522      0.1993      0.1991   -0.0001   [-0.0023,+0.0021]  ~noise
match_total_sixes_ou_15_5           261      0.2239      0.2245   +0.0005   [-0.0030,+0.0040]  ~noise
match_total_sixes_ou_20_5           261      0.1173      0.1173   -0.0000   [-0.0023,+0.0023]  ~noise
bowler_economy_ou_8_5              2981      0.2534      0.2541   +0.0007   [-0.0013,+0.0027]  ~noise
bowler_economy_ou_10_5             3007      0.1887      0.1880   -0.0007   [-0.0021,+0.0006]  ~noise
top_batter                         5835      0.0759      0.0762   +0.0002   [-0.0001,+0.0007]  ~noise
p_tie                               261      0.0002      0.0002   +0.0000   [-0.0000,+0.0000]  ~noise
highest_individual_mae              261  MAE_blind=16.791  MAE_hybrid=16.710  dMAE=-0.081  [-0.234,+0.056]  ~noise
batter_fours_mae                   4206  MAE_blind=1.398  MAE_hybrid=1.397  dMAE=-0.001  [-0.005,+0.003]  ~noise
```

**Zero** of the 32 scanned families is CI-clean in either direction.

## Executor notes for the orchestrator

- The wiring is not in question: the compose assertions prove the over-0
  vector is applied and materially different (max prob shift 8.4e-2 on
  over-0 balls), and the smoke/full startup lines confirm the pkl was loaded.
  The over-0 correction simply does not move any measured prop on the
  current engine.
- The B7 → B8 decomposition held on the *regression* side (tail +0.0079 →
  +0.0005; `bowler_wkts_1plus` +0.0024 → −0.0005) but not on the *gain* side.
  The `team_first_over_mae` baseline itself has drifted from the A15/A8-era
  3.526/3.535 to **3.411** on the b10 blind twin, i.e. D1/D15 (notably the
  run-out dismissal channel, ACTIVE at p_runout=0.0751) already absorbed
  roughly the improvement A15 was buying. That is the most plausible reading
  of why a stale-but-valid over-0 vector now nets to nothing.
- Nothing was reverted, `research/results.tsv` and `research/IDEAS.md` were
  not touched, no push, no second idea, no verdict issued.
- Raw-artifact note: `.gitignore:44` ignores `*.log` repo-wide, so
  `raw/run_b8_s43.log` and `raw/smoke.log` are present **on disk** but not
  committed — same as B10's and B7's run logs. The committed raw artifacts
  are `compose_output.txt`, `parity_check.txt`, `gate_wiring_check.txt`,
  `gate_output.txt`, `b8_start.txt`, `b8_end.txt`.
