# B18 — executor result (no verdict; orchestrator decides)

Idea: **B18 [P2]** — empirical extras rates on the promoted i7 quote path.
Plan: `research/handoff/B18/plan.md` (committed `c3e184f`).
Executed 2026-08-03.

**STATUS: COMPLETE — all three evals ran to completion and the
pre-committed gate was executed.** The run was initially BLOCKED at the
plan's Fit step by its own pre-committed sanity rule; the orchestrator
ruled on the ambiguity mid-session, the sidecar was then fitted, and the
plan's Steps 3/4/5 were executed. Both the original STOP finding and the
ruling are preserved verbatim below, followed by the full results.
**No verdict is stated here — the orchestrator applies the mapping.**

### Eval-relay note (why there are two s46 logs)

The recipe-B run (plan Step 3) was launched once earlier on 2026-08-03 and
**died at match 76/261** when the launching session closed and took the
child process with it. No output artifact was produced by that attempt (no
detail JSON, no report, no quote JSON, no gate output). Its truncated log
is kept verbatim as
`research/handoff/B18/raw/run_b18_s46_attempt1_truncated.{log,txt}`
(last line `[76/261] 1494266 …`). A follow-on session re-ran the **identical
invocation** to completion; every number below comes from that completed
run and from the two quote twins and the gate, all of which ran end-to-end
in one session with nothing left in the background.

Because `*.log` is gitignored (`.gitignore:44`), every raw log in
`research/handoff/B18/raw/` is mirrored byte-identically to a committed
`.txt` twin (`cmp -s` verified) — same convention the earlier B18 commits
used for `fit.txt` / `unit_check.txt` / `self_test.txt`.

---

## ORCHESTRATOR RULING (received mid-session, recorded verbatim)

> ORCHESTRATOR RULING on the B18 no-ball definition blocker — proceed as follows.
>
> Operative definition: no-ball event runs = the `noball_runs` channel ONLY
> (recomputed mean 1.071168). Wides = total extras runs on wide deliveries
> (1.204439), as you already have. My plan's prose gloss "penalty + byes =
> runs − batter_runs" was the error; the pre-committed anchor (≈1.071 in the
> IDEAS.md B18 method text, which is the B17-committed design arithmetic) is
> the operative quantity, and your recomputation matches it exactly. This
> ruling is also the one consistent with the plan's own residual accounting:
> byes/leg-byes (on legal AND no-ball deliveries, ≈ −0.0072) stay UNMODELED,
> exactly as the "Scope guard" and "Expected residual" sections state. The
> gate, tolerances, and verdict mapping are unchanged.
>
> Record this ruling verbatim in result.md (keep your original STOP finding
> in there too — it is correct and stays part of the record: the two
> definitions and their means 1.175182 vs 1.071168, and that the anchor
> matched the narrow one).
>
> Then resume the plan from where you stopped: [steps 1–6 …]
>
> All other constraints stand: no verdict, no revert, no results.tsv/IDEAS.md
> edits, no push, no golden, nothing shipped into production dirs, no
> background processes left running.

The original STOP record is retained in full in the
"THE BLOCKER" section below — the two candidate definitions and their means
(**1.175182** vs **1.071168**), and the fact that the plan's ≈1.071 anchor
matched the narrow one, remain part of the permanent record. The fit script
still measures and prints all three candidates on every run; only the
operative one is written to the sidecar.

---

## Original STOP finding (retained — correct, and still part of the record)

**At the time of the STOP: NO sim had run, NO eval number existed, NO
sidecar had been written.**

The plan's Fit section says, verbatim:

> Sanity: recomputed means ≈1.204 (wides) / ≈1.071 (no-balls). If either
> differs by >0.05 from those values, **STOP and record the discrepancy**
> (the attribution arithmetic would be wrong) — **do not proceed to evals.**

The wide mean reproduces (1.204439 vs 1.204). The no-ball mean, computed
under the plan's own operative definition, is **1.175182 — 0.104182 from
the 1.071 anchor**, i.e. more than 2× the stop threshold. Per that rule and
per the executor brief ("do not improvise around a pre-committed gate"), I
stopped before Step 3/4/5 and recorded. Everything below is arithmetic on
the val frame plus code validation; **no Monte Carlo was run at any point.**

---

## Commits created

| sha | message (subject) |
|---|---|
| `4f8050d` | `Auto[B18]: implement — opt-in empirical extras graft (engine, inert by default) + fit BLOCKED on the plan's pre-committed no-ball sanity anchor` |
| `1907903` | `Auto[B18]: pre-commit the gate script (mandatory self-test PASS, 6/6, while ZERO B18 eval output exists)` |
| `775267e` | `Auto[B18]: result.md — BLOCKED record (no eval ran)` |
| `cf2855d` | `Auto[B18]: fit UNBLOCKED under the orchestrator ruling — sidecar built, unit checks 2+3 PASS on the real artifact` |
| (final) | `Auto[B18]: eval evidence — recipe-B s46 + quote twins s49 + gate output` |

Ordering discipline: the gate script (`1907903`) and the sidecar fit
(`cf2855d`) both predate **every** B18 eval artifact — the first sim of any
kind started after `cf2855d` was already committed.

`git diff --stat 336524e` (claim commit → `cf2855d`, i.e. everything before
the evals; the final commit adds only this file, the raw logs and
`research/reports/auto/B18_props.md`):

```
 research/handoff/B18/plan.md                   | 235 ++++++++++++
 research/handoff/B18/raw/fit.txt               |  73 ++++
 research/handoff/B18/raw/self_test.txt         |  41 +++
 research/handoff/B18/raw/unit_check.txt        |  32 ++
 research/handoff/B18/raw/unit_check_fitted.txt |  33 ++
 research/handoff/B18/result.md                 | 393 +++++++++++++++++++++
 scripts/auto/b18_fit_extras_graft.py           | 471 +++++++++++++++++++++++++
 scripts/auto/b18_gate_analysis.py              | 395 +++++++++++++++++++++
 scripts/auto/b18_unit_check.py                 | 348 ++++++++++++++++++
 scripts/sim_v1_2.py                            | 157 ++++++++-
 10 files changed, 2176 insertions(+), 2 deletions(-)
```

---

## Step 0 — engine parity (before implementing): PASS

```
$ git diff ea4acdb HEAD -- scripts/sim_v1_2.py
(empty)
$ git status --porcelain scripts/sim_v1_2.py
(empty)
```

The engine was byte-identical to the D16 engine that produced
`models/auto/d16/detail_noweights_raw_s46_n261.json`, and the working tree
was clean, so a seed-46 pairing against that baseline would have been valid.

## Step 3 (model dir) — `models/auto/b18/`: PASS

Every file of `models/xgb_i7_noweights_production/` copied and md5-verified
file-by-file — **10/10 MATCH**:

```
batter_encoder_i7.pkl            a63af729f48492f701d1df5755493b01  MATCH
bowler_encoder_i7.pkl            96e76ad9bde782bc9a5df111f1b713fa  MATCH
feature_columns_i7.txt           3fd6629c6cf5be70873b92ab2e619119  MATCH
feature_importance.json          5b07cc465e59c17790d215ace1817808  MATCH
marginal_audit.json              615207d4bd68743e80f8782870b42b4b  MATCH
matchup_encoder_i7.pkl           264c11f6d8903347433017e8124a2ec1  MATCH
outcome_dist_config_i7.json      bec7d99140b4ae5e98977e18ca6ab670  MATCH
training_contract_i7.json        ff46131825d7669562dcfd1ee3191aac  MATCH
venue_encoder_i7.pkl             bcbb6e3fb60e08df50d752ea282c0009  MATCH
xgboost_model_i7.pkl             7ee1e1809917f45be7e726b3ea4a8a6c  MATCH
ALL_MATCH=1
```

Booster md5 `7ee1e1809917f45be7e726b3ea4a8a6c` — starts `7ee1e180` as the
plan requires. **No `extras_graft_v1.json` was added**, so `models/auto/b18/`
is currently behaviour-identical to production.

---

## THE BLOCKER — fit log verbatim (`research/handoff/B18/raw/fit.txt`)

### Population (exact B17 val frame; b17_runmass_audit loading code reused)

```
  model dir : .../models/xgb_i7_noweights_production
  parquet   : .../data/xgb_data_i7/cricket_data_i7_validation.parquet
  booster md5: 7ee1e1809917f45be7e726b3ea4a8a6c
  parquet md5: 326436317310adadabe0175825e57d1b
  rows 124,292  features 114  matches 545  innings 1088
  window 2024-12-31 .. 2025-06-29
  population hash (sorted match ids + row count): 4ac7accc089e4cc3d8df761a117cbb4c
```

### STEP 1 — D3 rate anchors: **REPRODUCE EXACTLY**

```
  deliveries 124,292   legal 119,058   wide rows 4,686   no-ball rows 548
  p_wide    = 0.037701542  (D3 anchor 0.037702)  rounded 0.037702
  p_no_ball = 0.004408972  (D3 anchor 0.004409)  rounded 0.004409
  anchors reproduce to 6 dp: True
  cricsheet cross-check: 124,292 deliveries, 119,058 legal, 169,099 runs   p_wide 0.037702  p_no_ball 0.004409
  parquet-vs-cricsheet delta: deliveries +0  legal +0  runs +0
```

### STEP 2 — per-event run laws: wide PASSES, no-ball TRIPS THE STOP

```
  WIDE (total runs on the delivery; all runs on a wide are extras)
    n_events 4,686   mean 1.204439   anchor 1.204 (tol 0.05)
    1:0.911225  2:0.042467  3:0.011097  4:0.001067  5:0.034144
    counts 1x4270  2x199  3x52  4x5  5x160
  NO_BALL (EXTRAS PORTION ONLY = runs - batter_runs)
    n_events 548   mean 1.175182   anchor 1.071 (tol 0.05)
    1:0.875912  2:0.105839  3:0.001825  5:0.016423
    counts 1x480  2x58  3x1  5x9
  context: mean OFF-BAT runs on no-balls = 1.465328 (deliberately NOT credited — the 6-class labels carry it)
  context: mean TOTAL runs on no-balls   = 2.640511

  candidate no-ball run definitions (discrepancy record):
    extras_portion_runs_minus_batter     mean 1.175182  |mean - 1.071| = 0.104182
    noball_runs_channel_only             mean 1.071168  |mean - 1.071| = 0.000168
    total_delivery_runs                  mean 2.640511  |mean - 1.071| = 1.569511

  |wide mean - anchor|    = 0.000439  -> OK
  |no-ball mean - anchor| = 0.104182  -> OUT OF TOLERANCE
```

### STEP 3 — analytic g recompute (B17 arithmetic re-derived from scratch)

```
    batter_encoded: 6932 classes, 0 rows unseen -> -1
    bowler_encoded: 5179 classes, 0 rows unseen -> -1
    matchup_type_encoded: 27 classes, 0 rows unseen -> -1
    venue_encoded: 373 classes, 0 rows unseen -> -1
  R_model (venue_on, all delivery rows) = 1.347522
  A (actual runs per legal ball)        = 1.420308   (B17 1.420308, delta -2.508e-07)
  actual extras channel per legal ball  = 0.059559   (B17 0.059559)

  OLD graft: carried extras 0.020000  M 1.367522  g -0.052785   (B17 logged -0.052785, delta -3.287e-07)
  NEW graft: carried extras = (p_w*r_w + p_nb*r_nb)/(1-p_w-p_nb)
             = (0.037702*1.204439 + 0.004409*1.175182) / 0.957889
             = 0.052815
             M 1.400337  g -0.019971
  predicted improvement in carried mass: +0.032815 runs per legal ball
  predicted residual channels (byes/leg-byes on legal balls, threes fold, 6-class head): -0.019971

  PRE-COMMITTED TOLERANCE: g_new in [-0.03, -0.012]
    g_new = -0.019971  ->  IN TOLERANCE
  expected cp6 quote-bias shrink ratio |g_new/g_old| = 0.3783  (plan projection: -4.78 -> -1.81 runs)

  sensitivity — analytic g under EVERY candidate no-ball run law:
    extras_portion_runs_minus_batter     r_nb 1.175182  carried 0.052815  g_new -0.019971  IN tolerance
    noball_runs_channel_only             r_nb 1.071168  carried 0.052336  g_new -0.020449  IN tolerance
    total_delivery_runs                  r_nb 2.640511  carried 0.059559  g_new -0.013226  IN tolerance
```

**B17 reproduces to 7 decimal places** (g_old −0.052785 vs B17's logged
−0.052785, delta −3.287e-07), so the audit this idea rests on is confirmed
independently.

### The STOP, verbatim

```
==============================================================================
STOP — PRE-COMMITTED SANITY CHECK TRIPPED
==============================================================================
  The plan's operative clause defines the no-ball event runs as
  'extras-portion only (exclude off-bat runs)' == runs - batter_runs,
  and its 'Easy to get wrong' note glosses that as 'penalty + byes'.
  That quantity measures 1.175182 on this population, not the
  plan's sanity anchor 1.071 (|delta| 0.104182 > tol 0.05).
  The anchor corresponds to the NARROWER `noball_runs` channel alone
  (no-ball penalty, excluding byes/leg-byes on the same delivery):
    extras_portion_runs_minus_batter     mean 1.175182
    noball_runs_channel_only             mean 1.071168
    total_delivery_runs                  mean 2.640511
  Per the plan: STOP and record; do NOT proceed to evals.
  NOTE (arithmetic, not a decision): the pre-committed ANALYTIC
  tolerance g_new in [-0.03, -0.012] is met under EVERY candidate
  definition above, so the B17 attribution arithmetic is not broken;
  the mismatch is between the plan's prose definition and its own
  sanity anchor. Disambiguation is the orchestrator's call.
  NO SIDECAR WRITTEN.
EXIT=3
```

### What the discrepancy actually is (for the orchestrator)

The plan specifies the no-ball run law three ways, and they do not agree:

| plan text | implies | measured |
|---|---|---|
| "extras-portion only (**exclude off-bat runs**)" (Fit, operative clause) | `runs - batter_runs` | **1.175182** |
| "EXTRAS-PORTION only (**penalty + byes**)" (Easy to get wrong #1) | `runs - batter_runs` | **1.175182** |
| sanity anchor **"≈1.071"** (Fit) | `noball_runs` channel alone (penalty only, byes/leg-byes on the same delivery excluded) | **1.071168** |
| "the SAME extras-channel accounting as `b17_runmass_audit.py`" | `extras_act` there is total runs on non-legal rows | **2.640511** |

The 1.071 anchor is exactly B17's `noball_runs 0.004930` per legal ball
divided by the no-ball event rate (548/119,058 = 0.004603) → 1.0710. So the
anchor was derived from the *narrower* channel while the prose specifies the
*wider* one. The gap, 0.104 runs per no-ball event, is ~57 runs of
byes/leg-byes recorded on no-ball deliveries across 548 events.

Decision-relevant: **the choice barely matters for the mechanism** — the two
candidates differ by 0.000478 runs per legal ball in carried mass
(0.052815 vs 0.052336) and both land the analytic `g_new` comfortably inside
the pre-committed `[-0.030, -0.012]` window. What the orchestrator needs to
do is pick one definition (and, if it wants, relax or restate the 1.071
anchor); the arithmetic then proceeds unchanged.

---

## OPERATIVE FIT under the ruling — verbatim (`raw/fit_ruling.txt`)

Re-run of `scripts/auto/b18_fit_extras_graft.py` with the ruling's operative
no-ball law (`noball_runs` channel only). Same population, same rate
anchors; only the no-ball run law changed. This is the run that WROTE
`models/auto/b18/extras_graft_v1.json`.

```
  rows 124,292  features 114  matches 545  innings 1088
  window 2024-12-31 .. 2025-06-29
  population hash (sorted match ids + row count): 4ac7accc089e4cc3d8df761a117cbb4c

  deliveries 124,292   legal 119,058   wide rows 4,686   no-ball rows 548
  p_wide    = 0.037701542  (D3 anchor 0.037702)  rounded 0.037702
  p_no_ball = 0.004408972  (D3 anchor 0.004409)  rounded 0.004409
  anchors reproduce to 6 dp: True

  WIDE (total runs on the delivery; all runs on a wide are extras)
    n_events 4,686   mean 1.204439   anchor 1.204 (tol 0.05)
    1:0.911225  2:0.042467  3:0.011097  4:0.001067  5:0.034144
  NO_BALL (OPERATIVE per the orchestrator ruling: `noball_runs` channel ONLY)
    n_events 548   mean 1.071168   anchor 1.071 (tol 0.05)
    1:0.928832  2:0.071168
  |wide mean - anchor|    = 0.000439  -> OK
  |no-ball mean - anchor| = 0.000168  -> OK

  R_model (venue_on, all delivery rows) = 1.347522
  A (actual runs per legal ball)        = 1.420308   (B17 1.420308, delta -2.508e-07)
  actual extras channel per legal ball  = 0.059559   (B17 0.059559)

  OLD graft: carried extras 0.020000  M 1.367522  g -0.052785   (B17 logged -0.052785, delta -3.287e-07)
  NEW graft: carried extras = (p_w*r_w + p_nb*r_nb)/(1-p_w-p_nb)
             = (0.037702*1.204439 + 0.004409*1.071168) / 0.957889
             = 0.052336
             M 1.399858  g -0.020449
  PRE-COMMITTED TOLERANCE: g_new in [-0.03, -0.012]
    g_new = -0.020449  ->  IN TOLERANCE
  expected cp6 quote-bias shrink ratio |g_new/g_old| = 0.3874  (plan projection: -4.78 -> -1.85 runs)

  wrote /Users/.../models/auto/b18/extras_graft_v1.json
  engine loader round-trip OK: B18 empirical extras graft ACTIVE (p_wide=0.037702, p_no_ball=0.004409, mean_runs w=1.2044/nb=1.0712)
```

The B17 attribution reproduces to 7 dp (`g_old` −0.052785), so the audit the
idea rests on is independently confirmed.

## Unit checks re-run on the REAL fitted artifact — ALL PASS (`raw/unit_check_fitted.txt`)

```
== Part 1: default-path inertness vs pre-B18 engine (ea4acdb) ==
  [PASS] 1a. predict_next_ball float-EXACT on 8 prob vectors (no sidecar)  (bit-identical on every key)
  [PASS] 1b. 400 same-seed simulate_ball draws identical (outcome, runs)  (6 wide/no-ball events in the sequence)
  [PASS] 1b. RNG state after the draws identical (ZERO extra draws consumed)

== Part 2: d15_unit_check.py unchanged (D2+D14+run-out, 30 assertions) ==
  [PASS] 2. d15_unit_check.py passes  (UNIT CHECK PASSED: all assertions hold.)

== Part 3: sidecar contract (REAL fitted B18 artifact) ==
  sidecar: models/auto/b18/extras_graft_v1.json
  B18 empirical extras graft ACTIVE (p_wide=0.037702, p_no_ball=0.004409, mean_runs w=1.2044/nb=1.0712)
  [PASS] 3a. wide mass == p_wide exactly
  [PASS] 3a. no_ball mass == p_no_ball exactly
  [PASS] 3a. sums to 1  (total=1.000000000000000)
  [PASS] 3a. 6-class relative marginals preserved
  [PASS] 3a. 6-class block mass == 1 - p_extras
  [PASS] 3a. non-unit block: extras mass still exact + marginals kept
  [PASS] 3a. degenerate block: uniform spread, sums to 1
  [PASS] 3a. LIVE XGBoostModelV2 path: extras mass exact
  [PASS] 3a. LIVE path: relative marginals preserved
  [PASS] 3a. LIVE path: sums to 1
  [PASS] 3b. simulated wide rate == sidecar p_wide (3 sigma)  (0.037463 vs 0.037702 (tol 0.001043))
  [PASS] 3b. simulated no-ball rate == sidecar p_no_ball (3 sigma)  (0.004620 vs 0.004409 (tol 0.000363))
  [PASS] 3b. sampled wide event-run mean == sidecar mean (3 sigma)  (1.2086 vs 1.2044 (tol 0.0219, n=11239))
  [PASS] 3b. sampled wide runs stay inside the sidecar support  (observed [1, 2, 3, 4, 5])
  [PASS] 3b. sampled no-ball event-run mean == sidecar mean (3 sigma)  (1.0620 vs 1.0712 (tol 0.0207, n=1386))
  [PASS] 3b. sampled no-ball runs stay inside the sidecar support  (observed [1, 2])
  [PASS] 3c. graft draws reproducible under a fixed seed

ALL PASS
```

---

## Implemented and validated anyway (written at the time of the STOP)

Because the engine edit is committed and stays in the tree, I proved it is
inert rather than asserting it. The block below is the pre-ruling record;
the only line since superseded is the "no sidecar exists" note (the sidecar
was written by the operative fit above, under `models/auto/b18/` only —
never in any production dir).

### Engine (`scripts/sim_v1_2.py`, +157/−2)

`ExtrasGraftConfig` + `graft_extras()` (D3's marginal-preserving
composition, recovered from `git show 8dfda3a`), sidecar auto-detect in
`XGBoostModelV2.__init__` on `extras_graft_v1.json` (same pattern as the
venue encoder, `sim_v1_2.py:1145–1152`), and per-event integer run crediting
in `T20Rules.simulate_ball` via `process_ball(..., team_runs=...)`.

Scope kept exactly as the plan specifies: no byes/leg-byes on legal
deliveries, no threes fold, no `run_rate`/selector/calibrator changes. Only
the XGBv2 wrapper got sidecar detection; every other wrapper keeps its
literal flat graft. Batter/bowler/**strike-rotation** attribution on an
extras event is left identical to the legacy flat path (the plan authorises
run crediting only) — flagging it because a multi-run wide realistically
changes strike, and that is a deliberate non-change, not an oversight.

**No `extras_graft_v1.json` exists anywhere in the repo**
(`find . -name extras_graft_v1.json` → empty), so production i7, legacy v3
replay, I5 and every other path are currently untouched by this commit.

### Unit check — ALL PASS (`research/handoff/B18/raw/unit_check.txt`)

```
== Part 1: default-path inertness vs pre-B18 engine (ea4acdb) ==
  [PASS] 1a. predict_next_ball float-EXACT on 8 prob vectors (no sidecar)  (bit-identical on every key)
  [PASS] 1b. 400 same-seed simulate_ball draws identical (outcome, runs)  (6 wide/no-ball events in the sequence)
  [PASS] 1b. RNG state after the draws identical (ZERO extra draws consumed)

== Part 2: d15_unit_check.py unchanged (D2+D14+run-out, 30 assertions) ==
  [PASS] 2. d15_unit_check.py passes  (UNIT CHECK PASSED: all assertions hold.)

== Part 3: sidecar contract (SYNTHETIC fixture, NOT the B18 fit) ==
  B18 empirical extras graft ACTIVE (p_wide=0.040000, p_no_ball=0.006000, mean_runs w=1.2200/nb=1.1500)
  [PASS] 3a. wide mass == p_wide exactly
  [PASS] 3a. no_ball mass == p_no_ball exactly
  [PASS] 3a. sums to 1  (total=1.000000000000000)
  [PASS] 3a. 6-class relative marginals preserved
  [PASS] 3a. 6-class block mass == 1 - p_extras
  [PASS] 3a. non-unit block: extras mass still exact + marginals kept
  [PASS] 3a. degenerate block: uniform spread, sums to 1
  [PASS] 3a. LIVE XGBoostModelV2 path: extras mass exact
  [PASS] 3a. LIVE path: relative marginals preserved
  [PASS] 3a. LIVE path: sums to 1
  [PASS] 3b. simulated wide rate == sidecar p_wide (3 sigma)  (0.039953 vs 0.040000 (tol 0.001073))
  [PASS] 3b. simulated no-ball rate == sidecar p_no_ball (3 sigma)  (0.006180 vs 0.006000 (tol 0.000423))
  [PASS] 3b. sampled wide event-run mean == sidecar mean (3 sigma)  (1.2228 vs 1.2200 (tol 0.0221, n=11986))
  [PASS] 3b. sampled wide runs stay inside the sidecar support  (observed [1, 2, 5])
  [PASS] 3b. sampled no-ball event-run mean == sidecar mean (3 sigma)  (1.1359 vs 1.1500 (tol 0.0249, n=1854))
  [PASS] 3b. sampled no-ball runs stay inside the sidecar support  (observed [1, 2])
  [PASS] 3c. graft draws reproducible under a fixed seed

ALL PASS
```

Part 1 is a genuine A/B: the pre-B18 engine is materialised from
`git show ea4acdb:scripts/sim_v1_2.py` into a temp module and executed
alongside HEAD. Part 3 uses a **synthetic fixture** (p_wide 0.04,
p_no_ball 0.006, means 1.22/1.15) precisely so a passing unit check can
never be mistaken for a fitted B18 artifact.

### Gate script pre-committed, self-test PASS 6/6 (`raw/self_test.txt`)

`scripts/auto/b18_gate_analysis.py` encodes the plan's gate verbatim
(P-A, P-B, G-1, G-2, G-3, 33-family scan), importing quote machinery from
`b15_gate_analysis` and prop pairing from `b12_gate_analysis`. It was
committed while **zero** B18 eval output existed, and its mandatory
self-test reproduces every B16 number the plan named:

```
            raw_bias  expected -4.781/-3.026/-1.946          got -4.781/-3.026/-1.946          PASS
        raw_coverage  expected 0.787/0.798/0.684             got 0.787/0.798/0.684             PASS
        b15_coverage  expected 0.822/0.838/0.792             got 0.822/0.838/0.792             PASS
     pooled_raw_dmae  expected -3.417 [-4.878, -2.066]       got -3.417 [-4.878, -2.066]       PASS
             raw_mae  expected 20.678/16.579/11.986          got 20.678/16.579/11.986          PASS
           naive_mae  expected 25.897/20.000/13.575          got 25.897/20.000/13.575          PASS

  SELF-TEST: PASS
```

B15 scales asserted at 1.19/1.09/1.26, shift 0; never refit.

---

## EVAL RESULTS — the numbers (all verbatim from tool output)

### Runs executed, wall times, banner blocks

| step | run | wall time | outcome |
|---|---|---|---|
| 3 | recipe-B, B18 arm, **seed 46**, n=261 × 100 sims | **1335.9 s** (22.3 min) | 261/261 matches, detail + report written |
| 4a | quote twin **RAW** (production stack), **seed 49** | **916.0 s** (15.3 min) | 756 rows / 253 matches / 8 skips |
| 4b | quote twin **B18** (graft sidecar), **seed 49** | **959.6 s** (16.0 min) | 756 rows / 253 matches / 8 skips |
| 5 | gate `b18_gate_analysis.py` (all defaults) | ~2 min | ran clean, output below |

Nothing was killed, nothing exceeded its budget (thresholds were 50 min for
recipe-B and 35 min per quote run), nothing crashed, and **no background
process is running.** Strictly one heavy process at a time; `--parallel`
never used.

**Recipe-B startup banner block (`raw/run_b18_s46.txt`, verbatim):**

```
Loading stats provider + player metadata + model ...
StatsProvider: using SQLite backend player_stats_cache_i7.sqlite (56.7 MB)
Loading player metadata from data/all_players_enriched.csv...
  Loaded 11,256 players
  Batting: 8178 right, 1814 left, 1264 unknown
  Bowling: 6256 right, 1381 left, 3619 unknown
Loaded XGBoost v2 model with 114 features with real player stats with player metadata; venue encoder ACTIVE (373 venues)
B18 empirical extras graft ACTIVE (p_wide=0.037702, p_no_ball=0.004409, mean_runs w=1.2044/nb=1.0712)
  sidecar: models/auto/b18/extras_graft_v1.json
Bowler selector: empirical
Run-out dismissal channel ACTIVE (p_runout=0.0751, nonstriker_share=0.4685)
Running prop backtest on 261 matches × 100 sims
B10 usage-aligned bowler selector ACTIVE (k_u=5.0)
  as-of corpus: .../models/b10_usage_corpus.pkl (7433 players); min_eligible=5, min_share=0.01
```

Every item the plan required is present: i7 sqlite, venue encoder ACTIVE
(373 venues), B18 graft ACTIVE at the fitted constants, B10 k_u=5.0
selector, run-out channel ACTIVE. `grep -in "calibrat" run_b18_s46.txt` →
**(none)**, i.e. no calibrator line, as required on the promoted no-weights
stack. Tail: `Done in 1335.9s`.

**RAW quote twin banner (`raw/quote_raw_s49.txt`) — graft must be ABSENT:**

```
Ball calibrator: NONE (raw probabilities)
Model: models/xgb_i7_noweights_production/xgboost_model_i7.pkl  (stats version i7)
Loaded XGBoost v2 model with 114 features with real player stats with player metadata; venue encoder ACTIVE (373 venues)
Run-out dismissal channel ACTIVE (p_runout=0.0751, nonstriker_share=0.4685)
B5 in-play quotes: 261 matches x 100 sims x checkpoints [6, 10, 15], seed 49
B10 usage-aligned bowler selector ACTIVE (k_u=5.0)
```
`grep -in "extras graft\|sidecar" quote_raw_s49.txt` → **(none)**. Correct.

**B18 quote twin banner (`raw/quote_b18_s49.txt`) — graft must be PRESENT:**

```
Ball calibrator: NONE (raw probabilities)
Model: models/auto/b18/xgboost_model_i7.pkl  (stats version i7)
Loaded XGBoost v2 model with 114 features with real player stats with player metadata; venue encoder ACTIVE (373 venues)
B18 empirical extras graft ACTIVE (p_wide=0.037702, p_no_ball=0.004409, mean_runs w=1.2044/nb=1.0712)
  sidecar: models/auto/b18/extras_graft_v1.json
Run-out dismissal channel ACTIVE (p_runout=0.0751, nonstriker_share=0.4685)
B5 in-play quotes: 261 matches x 100 sims x checkpoints [6, 10, 15], seed 49
B10 usage-aligned bowler selector ACTIVE (k_u=5.0)
```

Twin configs differ **only** in `model` path (both seed 49, both
`stats_version i7`, both `ball_calibrator null`, both
`usage_json models/bowler_phase_usage.json`, both `quote_center sim_p50`).

### Skip-list check — PASS, identical to B16

Both s49 twins skipped exactly the same 8 matches, and that set is
**identical to B16's s48 skip list** (the skip rule is deterministic on the
actual innings, not on sim draws):

```
raw_s49 skips == b16_s48 skips : True
b18_s49 skips == b16_s48 skips : True
raw_s49 skips == b18_s49 skips : True

  1493238.json innings 1 curtailed (66 legal balls, 5 dismissals)
  1493268.json innings 1 curtailed (30 legal balls, 0 dismissals)
  1493279.json innings 1 curtailed (60 legal balls, 5 dismissals)
  1494256.json innings 1 curtailed (24 legal balls, 0 dismissals)
  1494267.json innings 1 curtailed (72 legal balls, 7 dismissals)
  1514482.json innings 1 curtailed (72 legal balls, 6 dismissals)
  1527562.json innings 1 curtailed (78 legal balls, 5 dismissals)
  1527686.json innings 1 curtailed (66 legal balls, 3 dismissals)
```
No difference to record.

---

### PRIMARY — quote layer, same-seed (s49) twins

Gate header (verbatim):

```
  same-seed check: seed 49 on both twins — OK
  row/match parity: raw 756/253  b18 756/253  OK
```

**ARM: RAW twin (no sidecar), uncorrected**

```
checkpoint  6 (n=253):  applied shift +0.0000 scale 1.00
  MAE  corr(P50)  20.709  naive  25.897  dMAE  -5.188 [-7.807, -2.685]  CORRECTED BETTER
  P10-P90 coverage corr  0.787 [0.735, 0.834]  target [0.7, 0.9]  IN BAND
  context: bias corr P50 -4.872  bias raw P50 -4.872  band width corr 59.0  raw 59.0  actual sd 29.9
checkpoint 10 (n=253):  applied shift +0.0000 scale 1.00
  MAE  corr(P50)  16.573  naive  20.000  dMAE  -3.427 [-4.988, -1.962]  CORRECTED BETTER
  P10-P90 coverage corr  0.802 [0.755, 0.850]  target [0.7, 0.9]  IN BAND
  context: bias corr P50 -3.134  bias raw P50 -3.134  band width corr 51.9  raw 51.9  actual sd 25.3
checkpoint 15 (n=250):  applied shift +0.0000 scale 1.00
  MAE  corr(P50)  11.976  naive  13.575  dMAE  -1.599 [-2.535, -0.670]  CORRECTED BETTER
  P10-P90 coverage corr  0.696 [0.640, 0.752]  target [0.7, 0.9]  OUT OF BAND
  context: bias corr P50 -1.976  bias raw P50 -1.976  band width corr 30.8  raw 30.8  actual sd 16.9

pooled paired dMAE (corrected - naive, 756 rows, cluster-boot by match): -3.412 [-4.866, -2.070]
```

**ARM: B18 twin (graft), uncorrected**

```
checkpoint  6 (n=253):  applied shift +0.0000 scale 1.00
  MAE  corr(P50)  20.144  naive  25.897  dMAE  -5.753 [-8.496, -3.192]  CORRECTED BETTER
  P10-P90 coverage corr  0.802 [0.755, 0.854]  target [0.7, 0.9]  IN BAND
  context: bias corr P50 -0.121  bias raw P50 -0.121  band width corr 61.1  raw 61.1  actual sd 29.9
checkpoint 10 (n=253):  applied shift +0.0000 scale 1.00
  MAE  corr(P50)  16.243  naive  20.000  dMAE  -3.757 [-5.461, -2.077]  CORRECTED BETTER
  P10-P90 coverage corr  0.779 [0.727, 0.834]  target [0.7, 0.9]  IN BAND
  context: bias corr P50 +0.093  bias raw P50 +0.093  band width corr 46.5  raw 46.5  actual sd 25.3
checkpoint 15 (n=250):  applied shift +0.0000 scale 1.00
  MAE  corr(P50)  12.014  naive  13.575  dMAE  -1.561 [-2.580, -0.597]  CORRECTED BETTER
  P10-P90 coverage corr  0.708 [0.648, 0.768]  target [0.7, 0.9]  IN BAND
  context: bias corr P50 -1.054  bias raw P50 -1.054  band width corr 31.0  raw 31.0  actual sd 16.9

pooled paired dMAE (corrected - naive, 756 rows, cluster-boot by match): -3.699 [-5.233, -2.262]
```

**P-A — |P50 bias| shrinks at ALL THREE checkpoints (point test, same-seed twins)**

```
  cp 6: bias raw  -4.872  b18  -0.121   |raw|  4.872 -> |b18|  0.121   shrink  +4.751   SHRANK
  cp10: bias raw  -3.134  b18  +0.093   |raw|  3.134 -> |b18|  0.093   shrink  +3.042   SHRANK
  cp15: bias raw  -1.976  b18  -1.054   |raw|  1.976 -> |b18|  1.054   shrink  +0.922   SHRANK
  P-A: MET
```

**P-B — pooled paired dMAE (B18 P50 − naive), cluster-boot by match, 2000 reps seed 29, CI hi < 0**

```
  cp 6 (n=253): b18 MAE  20.144  naive  25.897  dMAE  -5.753 [-8.496, -3.192]
  cp10 (n=253): b18 MAE  16.243  naive  20.000  dMAE  -3.757 [-5.461, -2.077]
  cp15 (n=250): b18 MAE  12.014  naive  13.575  dMAE  -1.561 [-2.580, -0.597]
  pooled paired dMAE (756 rows): -3.699 [-5.233, -2.262]
  P-B: MET
  (context — RAW twin pooled dMAE: -3.412 [-4.866, -2.070])

  PRIMARY (P-A AND P-B): MET   [P-A=True P-B=True]
```

Cross-check against the plan's own projection: the analytic pre-check
predicted the cp6 bias would shrink by |g_new/g_old| = 0.3874, i.e.
−4.78 → ≈ −1.85. It actually went to **−0.121** — the mechanism
over-delivered at cp6/cp10 relative to the arithmetic, and under-delivered
at cp15 (−1.976 → −1.054). Recording that as an observation, not a claim.

### G-3 — B15-scaled coverage on the B18 arm, all three cps in [0.70, 0.90]

B15 scales asserted at 1.19 / 1.09 / 1.26, shift 0; **never refit**.

```
  cp 6: scale 1.19  coverage 0.866 [0.822, 0.909]  IN BAND   (raw context 0.802)
  cp10: scale 1.09  coverage 0.810 [0.767, 0.862]  IN BAND   (raw context 0.779)
  cp15: scale 1.26  coverage 0.788 [0.736, 0.840]  IN BAND   (raw context 0.708)
  B18+B15 coverage: 0.866/0.810/0.788   RAW-twin context: 0.787/0.802/0.696
  G-3: MET
```

Coverage per arm, both raw and B15-scaled, in one place:

| cp | RAW twin raw | RAW twin B15-scaled (context) | B18 twin raw | B18 twin B15-scaled (G-3) |
|---|---|---|---|---|
| 6 | 0.787 [0.735, 0.834] IN BAND | — | 0.802 [0.755, 0.854] IN BAND | **0.866 [0.822, 0.909] IN BAND** |
| 10 | 0.802 [0.755, 0.850] IN BAND | — | 0.779 [0.727, 0.834] IN BAND | **0.810 [0.767, 0.862] IN BAND** |
| 15 | 0.696 [0.640, 0.752] **OUT OF BAND** | — | 0.708 [0.648, 0.768] IN BAND | **0.788 [0.736, 0.840] IN BAND** |

(The gate reports B15 scaling only on the certified B18 arm, per the plan;
the RAW-twin B15 column is therefore not produced.)

**ARM-3 conditional was NOT needed and was NOT run** — G-3 came out in band
on the first pass at all three checkpoints, so no fourth eval, no scale
refit, no seed-50 val quote run. Total evals this session: 3.

### G-1 — recipe-B paired vs the canonical D16 baseline

Pairing (verbatim): `baseline (d16_raw): 261 matches
(detail_noweights_raw_s46_n261.json)` vs `b18 (b18_graft): 261 matches
(detail_b18_s46_n261.json)`; `cluster bootstrap by match, n_boot=2000,
seed=29; delta = b18_graft - d16_raw (positive = B18 worse)`.

```
family                                 n  drop     d16_raw   b18_graft     delta   95% CI (b18_graft-d16_raw)   flag
innings_runs_ou_160_5                522     0      0.2381      0.2227   -0.0154   [-0.0278,-0.0035]  DOWN(better)
  (positional cross-check)           522                                 -0.0154   [-0.0278,-0.0035]  DOWN(better)
innings_runs_ou_170_5                522     0      0.2297      0.2183   -0.0115   [-0.0223,-0.0012]  DOWN(better)
  (positional cross-check)           522                                 -0.0115   [-0.0223,-0.0012]  DOWN(better)
innings_runs_ou_180_5                522     0      0.1984      0.1831   -0.0153   [-0.0246,-0.0064]  DOWN(better)
  (positional cross-check)           522                                 -0.0153   [-0.0246,-0.0064]  DOWN(better)
pp_total_ou_45_5                     522     0      0.2367      0.2241   -0.0126   [-0.0213,-0.0040]  DOWN(better)
  (positional cross-check)           522                                 -0.0126   [-0.0213,-0.0040]  DOWN(better)
pp_total_ou_50_5                     522     0      0.2355      0.2249   -0.0106   [-0.0196,-0.0012]  DOWN(better)
  (positional cross-check)           522                                 -0.0106   [-0.0196,-0.0012]  DOWN(better)
pp_total_ou_55_5                     522     0      0.2005      0.1958   -0.0047   [-0.0121,+0.0027]  ~noise
  (positional cross-check)           522                                 -0.0047   [-0.0121,+0.0027]  ~noise
  innings_runs_ou_160_5        ok
  innings_runs_ou_170_5        ok
  innings_runs_ou_180_5        ok
  pp_total_ou_45_5             ok
  pp_total_ou_50_5             ok
  pp_total_ou_55_5             ok
  G-1: MET
```

Note for the orchestrator: this is the exact failure mode D3 died on for
legacy (innings totals + PP regressing). Here five of the six guard lines
move **CI-clean better** and the sixth is noise-indistinguishable — the
opposite sign of D3's legacy result.

### G-2 — `batter_runs_mae` must NOT regress CI-clean

```
family                                 n  drop     d16_raw   b18_graft     delta   95% CI (b18_graft-d16_raw)   flag
batter_runs_mae                     4254     0     13.8864     13.9552   +0.0688   [-0.0145,+0.1453]  ~noise
  (positional cross-check)          2913                                 +0.1174   [+0.0279,+0.2083]  UP(worse)
  G-2: MET

  GUARDS (G-1 AND G-2 AND G-3): MET   [G-1=True G-2=True G-3=True]
```

**FLAGGED HONESTLY:** G-2 is scored MET on the pre-committed statistic (the
full n=4254 family delta, CI lo −0.0145 ≤ 0), but the gate's **positional
cross-check on the same family (n=2913) IS a CI-clean regression**:
`+0.1174 [+0.0279, +0.2083] UP(worse)`. This is the only CI-clean adverse
movement anywhere in the B18 output. Magnitude context: batter_runs_mae
moves 13.8864 → 13.9552, i.e. +0.069 runs (~0.5%) on the headline family.
The plan's instruction was to "flag ANY CI-clean regression anywhere", so it
is flagged here rather than buried in the scan; whether it should change the
verdict is the orchestrator's call, not mine.

### Full 33-family scan (context; cannot flip the gate)

```
family                                 n  drop     d16_raw   b18_graft     delta   95% CI (b18_graft-d16_raw)   flag
batter_50plus                       4254     0      0.0786      0.0779   -0.0007   [-0.0015,+0.0000]  ~noise
batter_6plus_six                    4254     0      0.2195      0.2188   -0.0007   [-0.0027,+0.0013]  ~noise
batter_fours_1plus                  4254     0      0.2183      0.2180   -0.0003   [-0.0026,+0.0019]  ~noise
batter_fours_2plus                  4254     0      0.2011      0.2004   -0.0007   [-0.0028,+0.0014]  ~noise
batter_fours_3plus                  4254     0      0.1554      0.1553   -0.0001   [-0.0017,+0.0014]  ~noise
batter_fours_mae                    4254     0      1.3585      1.3666   +0.0080   [-0.0005,+0.0163]  ~noise
batter_runs_mae                     4254     0     13.8864     13.9552   +0.0688   [-0.0145,+0.1453]  ~noise
bowler_economy_ou_10_5              3107     0      0.1912      0.1911   -0.0001   [-0.0024,+0.0020]  ~noise
bowler_economy_ou_8_5               3107     0      0.2501      0.2467   -0.0034   [-0.0061,-0.0007]  DOWN(better)
bowler_wkts_1plus                   3107     0      0.2437      0.2452   +0.0015   [-0.0007,+0.0038]  ~noise
bowler_wkts_2plus                   3107     0      0.2103      0.2101   -0.0002   [-0.0020,+0.0016]  ~noise
bowler_wkts_3plus                   3107     0      0.0946      0.0941   -0.0005   [-0.0012,+0.0003]  ~noise
first_wicket_runs_ou_30_5            522     0      0.2395      0.2322   -0.0073   [-0.0171,+0.0021]  ~noise
highest_individual_mae               261     0     16.1951     16.1637   -0.0315   [-0.3608,+0.2821]  ~noise
highest_over_runs_ou_18_5            261     0      0.2445      0.2282   -0.0163   [-0.0258,-0.0065]  DOWN(better)
highest_over_runs_ou_24_5            261     0      0.0971      0.0963   -0.0008   [-0.0034,+0.0015]  ~noise
innings_runs_ou_160_5                522     0      0.2381      0.2227   -0.0154   [-0.0278,-0.0035]  DOWN(better)
innings_runs_ou_170_5                522     0      0.2297      0.2183   -0.0115   [-0.0223,-0.0012]  DOWN(better)
innings_runs_ou_180_5                522     0      0.1984      0.1831   -0.0153   [-0.0246,-0.0064]  DOWN(better)
match_total_sixes_ou_15_5            261     0      0.1980      0.1976   -0.0004   [-0.0061,+0.0057]  ~noise
match_total_sixes_ou_20_5            261     0      0.1006      0.0996   -0.0010   [-0.0037,+0.0017]  ~noise
p_tie                                261     0      0.0003      0.0003   +0.0001   [-0.0000,+0.0001]  ~noise
pp_total_ou_45_5                     522     0      0.2367      0.2241   -0.0126   [-0.0213,-0.0040]  DOWN(better)
pp_total_ou_50_5                     522     0      0.2355      0.2249   -0.0106   [-0.0196,-0.0012]  DOWN(better)
pp_total_ou_55_5                     522     0      0.2005      0.1958   -0.0047   [-0.0121,+0.0027]  ~noise
team_first_over_mae                  522     0      3.3785      3.3846   +0.0061   [-0.0371,+0.0480]  ~noise
team_highest_individual_ou_29_5      522     0      0.0757      0.0757   -0.0000   [-0.0018,+0.0019]  ~noise
team_highest_individual_ou_34_5      522     0      0.1293      0.1280   -0.0012   [-0.0044,+0.0020]  ~noise
team_highest_individual_ou_39_5      522     0      0.1737      0.1707   -0.0030   [-0.0073,+0.0011]  ~noise
team_total_fours_mae                 522     0      3.4987      3.5036   +0.0050   [-0.0372,+0.0465]  ~noise
team_total_sixes_mae                 522     0      2.6457      2.6403   -0.0054   [-0.0334,+0.0216]  ~noise
top_batter                          5835     0      0.0756      0.0751   -0.0005   [-0.0014,+0.0002]  ~noise
top_bowler                          5827     8      0.0772      0.0769   -0.0003   [-0.0009,+0.0004]  ~noise

  CI-excludes-0 families: 7 (7 favorable, 0 regressions)
    highest_over_runs_ou_18_5            -0.0163  [-0.0258,-0.0065]  DOWN(better)
    innings_runs_ou_160_5                -0.0154  [-0.0278,-0.0035]  DOWN(better)
    innings_runs_ou_180_5                -0.0153  [-0.0246,-0.0064]  DOWN(better)
    pp_total_ou_45_5                     -0.0126  [-0.0213,-0.0040]  DOWN(better)
    innings_runs_ou_170_5                -0.0115  [-0.0223,-0.0012]  DOWN(better)
    pp_total_ou_50_5                     -0.0106  [-0.0196,-0.0012]  DOWN(better)
    bowler_economy_ou_8_5                -0.0034  [-0.0061,-0.0007]  DOWN(better)
```

**Scan summary:** 7 of 33 families move CI-clean, **all 7 favorable, 0
CI-clean regressions at family level.** The single CI-clean adverse result
anywhere in the run is the `batter_runs_mae` **positional cross-check**
(+0.1174 [+0.0279, +0.2083]) reported under G-2 above.

### Gate bottom line (verbatim, final two lines of `raw/gate_output.txt`)

```
PRIMARY: MET (P-A=True, P-B=True) | GUARDS: MET (G-1=True, G-2=True, G-3=True)
verdict mapping (orchestrator applies it, not this script): BOTH -> LANDED, exactly one -> TABLED, neither -> FAILED
```

The verdict is deliberately **not stated in this file.**

---

## Read-only / protected-surface integrity (re-verified AFTER all evals)

`raw/readonly_mtimes_before.txt` (captured before any eval) vs
`raw/readonly_mtimes_after.txt` (recomputed 2026-08-03T13:00:07Z, after the
gate): `diff` of the md5 + mtime lines → **IDENTICAL**.

```
MD5 (models/xgb_i7_noweights_production/xgboost_model_i7.pkl) = 7ee1e1809917f45be7e726b3ea4a8a6c
MD5 (models/bowler_phase_usage.json) = 2e650423f0c949631fca1f15dd1c8a56
MD5 (models/auto/d16/detail_noweights_raw_s46_n261.json) = d816ebcd5cc9190bc4c4ca578dd6bbf1
MD5 (models/auto/b15/quotes_s45_n261.json) = 1e05da01640c994e26d5ce97dac47c9f
MD5 (models/auto/b15/quote_calibrator_scale_only.json) = 12f2bd1a7549dad5bbad138f2d1b56ba
MD5 (models/auto/b16/quotes_i7_s48_n261.json) = 0a92a2e33808339805fa9533bc769c05
1785718941 models/xgb_i7_noweights_production/xgboost_model_i7.pkl
1785520845 models/bowler_phase_usage.json
1785545951 models/auto/d16/detail_noweights_raw_s46_n261.json
1785739132 models/auto/b16/quotes_i7_s48_n261.json
```

So the production booster, the shared usage prior, the D16 pairing
baseline, the B15 calibrator and the B16 quote reference are all
byte-unchanged **and** mtime-unchanged — nothing was even rewritten
in place.

```
$ git diff --name-only 336524e -- scripts/sim_eval/   -> (empty)
$ git status --porcelain data/golden/                 -> (empty)
$ ls -la models/xgb_i7_noweights_production/          -> all 10 files still Aug 2 21:02
```

`scripts/sim_eval/`, `scripts/parsing_v2.py`, `scripts/stats_provider.py`,
`scripts/stats_sqlite_backend.py`, `betting_odds_polymarket.json` and
`data/polymarket_test/` were never modified. `data/golden/` was never read
or evaluated against. No production artifact was written — every B18
artifact lives under `models/auto/b18/` and the sidecar stays opt-in there.
`research/results.tsv` and `research/IDEAS.md` were not touched. Nothing was
pushed. Nothing was reverted. No verdict was issued. No second idea was
started.

## Artifacts produced by this session

| path | note |
|---|---|
| `models/auto/b18/detail_b18_s46_n261.json` | recipe-B detail, 261 matches (gitignored) |
| `models/auto/b18/quotes_raw_s49_n261.json` | RAW twin, 756 rows / 253 matches (gitignored) |
| `models/auto/b18/quotes_b18_s49_n261.json` | B18 twin, 756 rows / 253 matches (gitignored) |
| `research/reports/auto/B18_props.md` | prop-calibration report for the B18 arm (committed) |
| `research/handoff/B18/raw/run_b18_s46.{log,txt}` | completed recipe-B log |
| `research/handoff/B18/raw/run_b18_s46_attempt1_truncated.{log,txt}` | attempt-1 log, died at 76/261 |
| `research/handoff/B18/raw/quote_raw_s49.{log,txt}` | RAW twin log |
| `research/handoff/B18/raw/quote_b18_s49.{log,txt}` | B18 twin log |
| `research/handoff/B18/raw/gate_output.txt` | full gate output |
| `research/handoff/B18/raw/readonly_mtimes_{before,after}.txt` | integrity snapshots |
