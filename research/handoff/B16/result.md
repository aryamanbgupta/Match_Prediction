# B16 — executor result (no verdict; orchestrator decides)

Idea: **B16** — quote-layer coverage re-check on the promoted i7 stack.
Plan: `research/handoff/B16/plan.md` (committed `7f38bdf`).
Executed 2026-08-03. **Both pre-committed gates MET; certified arm = B15
scales (arm 2), RAW (arm 1) FAILS GATE 2 at cp15.** Verdict is not issued
here.

## Commits created

| sha | message |
|---|---|
| `653069a` | `Auto[B16]: implement — i7-stack args on quote harness + pre-committed gate (self-test PASS)` |
| (final) | `Auto[B16]: eval + gate output + result.md (no verdict — orchestrator decides)` |

`git diff --stat 445a1f3` (claim commit → after the implement commit; the
eval commit adds only `research/handoff/B16/raw/*.txt` + this file):

```
 research/handoff/B16/plan.md                       | 182 ++++++++++++
 research/handoff/B16/raw/readonly_mtimes_before.txt |   7 +
 research/handoff/B16/raw/self_test.txt             |  37 +++
 scripts/auto/b16_gate_analysis.py                  | 313 +++++++++++++++++++++
 scripts/auto/b5_inplay_quotes.py                   |  70 ++++-
 5 files changed, 597 insertions(+), 12 deletions(-)
```

`git diff --name-only 445a1f3 -- scripts/sim_eval/` → **0 files** (eval
framework untouched). `data/golden/` never read or touched.

## Step 1 — mandatory self-test: PASS

`uv run python scripts/auto/b16_gate_analysis.py --self-test`
(`research/handoff/B16/raw/self_test.txt`), run on the frozen
`models/auto/b15/quotes_s45_n261.json` with
`models/auto/b15/quote_calibrator_scale_only.json`:

```
   scale_only_coverage  expected 0.818/0.834/0.768   got 0.818/0.834/0.768   PASS
          raw_coverage  expected 0.755/0.791/0.660   got 0.755/0.791/0.660   PASS
       pooled_raw_dmae  expected -3.131 [-4.909, -1.356]  got -3.131 [-4.909, -1.356]  PASS
        scale_only_mae  expected 20.773/16.990/12.338  got 20.773/16.990/12.338  PASS
             naive_mae  expected 25.897/20.000/13.575  got 25.897/20.000/13.575  PASS
   RAW arm == raw rows  expected 0.755/0.791/0.660   got 0.755/0.791/0.660   PASS

  SELF-TEST: PASS
```

Assertion precision: the full-precision B15 expectations are not stored in
`research/handoff/B15/raw/`, so the self-test compares the **3-dp formatted
strings**, i.e. exact equality at the precision B15's result was recorded at
(`research/reports/auto/B15.md` and `research/handoff/B15/raw/gate_output.txt`).
The self-test's printed arm block is line-for-line identical to B15's logged
gate output, including every per-cp CI, bias, band width and actual-sd figure.
The self-test was committed (`653069a`) **before** the i7 quote run produced
any output.

## Step 2 — i7 quote run (one fresh run, seed 48)

`models/auto/b16/quotes_i7_s48_n261.json`, log
`research/handoff/B16/raw/quote_run_i7_s48.txt`.

Startup banner check — **all correct, nothing killed/restarted**:

```
StatsProvider: using SQLite backend player_stats_cache_i7.sqlite (56.7 MB)
Ball calibrator: NONE (raw probabilities)
Model: models/xgb_i7_noweights_production/xgboost_model_i7.pkl  (stats version i7)
Loaded XGBoost v2 model with 114 features with real player stats with player metadata; venue encoder ACTIVE (373 venues)
Run-out dismissal channel ACTIVE (p_runout=0.0751, nonstriker_share=0.4685)
B5 in-play quotes: 261 matches x 100 sims x checkpoints [6, 10, 15], seed 48
B10 usage-aligned bowler selector ACTIVE (k_u=5.0)
  as-of corpus: .../models/b10_usage_corpus.pkl (7433 players); min_eligible=5, min_share=0.01
```

No `Ball calibrator: vector` line anywhere. Usage json =
`models/bowler_phase_usage.json` (production, md5
`2e650423f0c949631fca1f15dd1c8a56`, B10 key present).

Structure — **exactly as predicted**:

```
Done in 898.8s — 756 quote rows from 253 matches (8 matches skipped)
```

756 rows / 253 matches / **8 skips**, and the 8 skipped files are the *same
8 files with the same reasons* as B15's legacy run (`diff` of the skip lists:
identical). Wall time **898.8 s** (14.98 min) vs B15 legacy 1487.2 s — well
inside budget, no kill needed.

## Step 3 — gate analysis (verbatim from `research/handoff/B16/raw/gate_output.txt`)

Config echoed by the gate script:
`model=models/xgb_i7_noweights_production/xgboost_model_i7.pkl
stats_version=i7 ball_calibrator=None n_sims=100 seed=48
quote_center=sim_p50 usage_json=models/bowler_phase_usage.json
elapsed=898.8s`; B15 calibrator assertion PASS (scales 1.19/1.09/1.26,
shifts 0; no refit anywhere).

### ARM 1 — RAW (no quote calibrator)

```
checkpoint  6 (n=253):  applied shift +0.0000 scale 1.00
  MAE  corr(P50)  20.678  naive  25.897  dMAE  -5.219 [-7.803, -2.723]  CORRECTED BETTER
  P10-P90 coverage corr  0.787 [0.735, 0.834]  target [0.7, 0.9]  IN BAND
  context: bias corr P50 -4.781  bias raw P50 -4.781  band width corr 58.4  raw 58.4  actual sd 29.9
checkpoint 10 (n=253):  applied shift +0.0000 scale 1.00
  MAE  corr(P50)  16.579  naive  20.000  dMAE  -3.421 [-4.986, -1.942]  CORRECTED BETTER
  P10-P90 coverage corr  0.798 [0.751, 0.846]  target [0.7, 0.9]  IN BAND
  context: bias corr P50 -3.026  bias raw P50 -3.026  band width corr 51.7  raw 51.7  actual sd 25.3
checkpoint 15 (n=250):  applied shift +0.0000 scale 1.00
  MAE  corr(P50)  11.986  naive  13.575  dMAE  -1.589 [-2.525, -0.658]  CORRECTED BETTER
  P10-P90 coverage corr  0.684 [0.624, 0.740]  target [0.7, 0.9]  OUT OF BAND
  context: bias corr P50 -1.946  bias raw P50 -1.946  band width corr 30.6  raw 30.6  actual sd 16.9

pooled paired dMAE (corrected - naive, 756 rows, cluster-boot by match): -3.417 [-4.878, -2.066]
```

### ARM 2 — B15 scales (1.19/1.09/1.26, shift 0)

```
checkpoint  6 (n=253):  applied shift +0.0000 scale 1.19
  MAE  corr(P50)  20.678  naive  25.897  dMAE  -5.219 [-7.803, -2.723]  CORRECTED BETTER
  P10-P90 coverage corr  0.822 [0.775, 0.866]  target [0.7, 0.9]  IN BAND
  P10-P90 coverage raw   0.787 [0.735, 0.834]   (uncorrected context)
  context: bias corr P50 -4.781  bias raw P50 -4.781  band width corr 69.6  raw 58.4  actual sd 29.9
checkpoint 10 (n=253):  applied shift +0.0000 scale 1.09
  MAE  corr(P50)  16.579  naive  20.000  dMAE  -3.421 [-4.986, -1.942]  CORRECTED BETTER
  P10-P90 coverage corr  0.838 [0.791, 0.881]  target [0.7, 0.9]  IN BAND
  P10-P90 coverage raw   0.798 [0.751, 0.846]   (uncorrected context)
  context: bias corr P50 -3.026  bias raw P50 -3.026  band width corr 56.3  raw 51.7  actual sd 25.3
checkpoint 15 (n=250):  applied shift +0.0000 scale 1.26
  MAE  corr(P50)  11.986  naive  13.575  dMAE  -1.589 [-2.525, -0.658]  CORRECTED BETTER
  P10-P90 coverage corr  0.792 [0.740, 0.840]  target [0.7, 0.9]  IN BAND
  P10-P90 coverage raw   0.684 [0.624, 0.740]   (uncorrected context)
  context: bias corr P50 -1.946  bias raw P50 -1.946  band width corr 38.5  raw 30.6  actual sd 16.9

pooled paired dMAE (corrected - naive, 756 rows, cluster-boot by match): -3.417 [-4.878, -2.066]
pooled paired dMAE (RAW - naive, uncorrected context): -3.417 [-4.878, -2.066]
```

Scale-only leaves P50 untouched, so ARM 2's MAE/dMAE/bias rows are
identically ARM 1's — only the bands differ.

### ARM 3 — refit scale-only: **NOT RUN**

Per the plan, the conditional i7 VAL run + refit fires *only if both cheaper
arms fail GATE 2*. ARM 2 is in band at all three checkpoints, so the val run
was not executed (no second heavy job, no extra draws).

### GATE 1 — raw sim P50 vs naive

```
  cp 6 (n=253): raw MAE  20.678  naive  25.897  dMAE  -5.219 [-7.803, -2.723]
  cp10 (n=253): raw MAE  16.579  naive  20.000  dMAE  -3.421 [-4.986, -1.942]
  cp15 (n=250): raw MAE  11.986  naive  13.575  dMAE  -1.589 [-2.525, -0.658]
  pooled paired dMAE (raw - naive, 756 rows): -3.417 [-4.878, -2.066]
  GATE 1 (CI hi < 0): MET
```

### GATE 2 — coverage in [0.70, 0.90] at all three cps

```
    RAW: coverage 0.787/0.798/0.684  in-band [True, True, False]  -> FAILS
    B15: coverage 0.822/0.838/0.792  in-band [True, True, True]  -> IN BAND (all 3)
  pre-committed preference order: RAW > B15
  GATE 2: MET — certified arm = B15 (2. B15 scales (1.19/1.09/1.26, shift 0))
```

### Pre-committed gate pair (script's closing block, verbatim)

```
GATE 1 (pooled raw dMAE CI hi < 0): MET (pooled -3.417 hi -2.066)
GATE 2 (an arm in band at all 3 cps): MET (arm B15, coverage 0.822/0.838/0.792)
verdict mapping (orchestrator applies it, not this script): BOTH -> LANDED, exactly one -> TABLED, neither -> FAILED
```

## Arm-by-gate summary (facts, not a verdict)

| arm | GATE 1 (pooled raw dMAE CI hi < 0) | GATE 2 (coverage in band at cps 6/10/15) |
|---|---|---|
| 1. RAW (no quote calibrator) | MET (−3.417 [−4.878, −2.066]; the raw dMAE is arm-independent) | **FAILS** — 0.787 / 0.798 / **0.684** (cp15 below the 0.70 floor) |
| 2. B15 scales (record, shift 0) | MET (identical, P50 untouched) | **MET** — 0.822 / 0.838 / 0.792 |
| 3. refit scale-only | n/a — not run (arm 2 already in band) | n/a |

Certified arm under the pre-committed preference order: **B15 scales**
(RAW is preferred *only if* it is in band, and it is not).

## Context vs the B15 legacy-stack baseline (different engine + seed; not paired)

| quantity | B15 legacy (s45) | B16 i7 (s48) |
|---|---|---|
| raw coverage cp6/cp10/cp15 | 0.755 / 0.791 / 0.660 | 0.787 / 0.798 / **0.684** |
| B15-scaled coverage | 0.818 / 0.834 / 0.768 | 0.822 / 0.838 / 0.792 |
| raw P50 MAE cp6/cp10/cp15 | 20.773 / 16.990 / 12.338 | 20.678 / 16.579 / 11.986 |
| pooled raw dMAE | −3.131 [−4.909, −1.356] | −3.417 [−4.878, −2.066] |
| raw band width cp6/cp10/cp15 | 58.3 / 48.2 / 29.4 | 58.4 / 51.7 / 30.6 |
| raw P50 bias cp6/cp10/cp15 | +4.259 / +2.777 / +0.410 | **−4.781 / −3.026 / −1.946** |
| rows / matches / skips | 756 / 253 / 8 | 756 / 253 / 8 (same 8 files) |
| wall time | 1487.2 s | 898.8 s |

## Things worth flagging to the orchestrator

1. **The B16 hypothesis's premise did not reproduce at the quote layer.**
   A13's STEP 0 found the promoted stack disperses materially wider at the
   *prop* layer, which motivated the "B15 scales are now stale / RAW may
   already be in band" expectation. At the in-play quote layer the raw bands
   widened only modestly (cp6 58.3→58.4, cp10 48.2→51.7, cp15 29.4→30.6) and
   raw cp15 coverage is still **out of band** (0.684, CI [0.624, 0.740]) —
   the same defect B5 and B15 found on the legacy engine. RAW therefore fails
   GATE 2 and the B15 scales are *not* stale for this path.
2. **The raw P50 bias flipped sign.** Legacy over-predicted remaining runs
   (+4.259/+2.777/+0.410); the promoted no-weights stack under-predicts
   (−4.781/−3.026/−1.946) at every checkpoint, and the magnitude at cp15
   grew from +0.410 to −1.946. This is a genuinely new fact about the
   promoted engine's in-play behaviour, and it re-confirms B15's lesson from
   the other direction: a location/shift term fit on any earlier stack or era
   would now be pointed the wrong way. Both arms here use shift ≡ 0, so it
   does not affect the gate.
3. **Point-estimate MAE improved at every checkpoint** on the i7 path
   (20.678/16.579/11.986 vs 20.773/16.990/12.338) with a more negative pooled
   dMAE and a tighter CI. Different engine *and* different seed, so this is
   context only — no paired claim.
4. Nothing crashed, nothing was killed, nothing ran long (898.8 s against a
   25–45 min expectation and a 90 min kill threshold). No background
   processes are left running.

## Read-only / production integrity

md5 of the production ball model, **before and after** the run — unchanged:

```
before: MD5 (models/xgb_i7_noweights_production/xgboost_model_i7.pkl) = 7ee1e1809917f45be7e726b3ea4a8a6c
after : MD5 (models/xgb_i7_noweights_production/xgboost_model_i7.pkl) = 7ee1e1809917f45be7e726b3ea4a8a6c
```

Also unchanged (mtime + md5, `raw/readonly_mtimes_{before,after}.txt`):
`models/auto/b15/quotes_s45_n261.json` (`1e05da01…`, mtime 1785527967),
`models/auto/b15/quote_calibrator_scale_only.json` (`12f2bd1a…`, mtime
1785528220), `models/auto/b14/quote_calibrator.json` (mtime 1785525614),
`models/bowler_phase_usage.json` (`2e650423…`).

Harness default check: `git diff scripts/auto/b5_inplay_quotes.py` shows every
new argparse default is the legacy v3 path
(`models/xgb_v3/xgboost_model_v3.pkl`, `--stats-version v3`,
`--ball-calibrator vector` with the v3 calibrator), so a no-args invocation
is behaviour-identical to the B5/B14/B15 runs; only the hard-coded strings in
the model construction and the output config block were replaced by the
corresponding args.

## Artifacts

- `scripts/auto/b16_gate_analysis.py` (new, pre-committed `653069a`)
- `scripts/auto/b5_inplay_quotes.py` (opt-in stack args; legacy defaults)
- `models/auto/b16/quotes_i7_s48_n261.json` (+ `.partial.jsonl`) — gitignored
- `research/handoff/B16/raw/self_test.txt`, `quote_run_i7_s48.txt`,
  `gate_output.txt`, `readonly_mtimes_{before,after}.txt`
