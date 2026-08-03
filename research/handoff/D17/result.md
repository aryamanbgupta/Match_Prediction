# D17 — executor result

Claim `2a9389b` · plan `8aa1629` · implement `538ca24`.
Executed 2026-08-01 (UTC) per `research/handoff/D17/plan.md`, verbatim.

**Question:** does a `VectorScalingCalibrator` fit on the D16 no-weights arm's
OWN i7 validation predictions buy anything on top of running that arm RAW?

**Headline (executor states facts, orchestrator issues the verdict):**
GATE 1 **NOT MET** (both conjuncts fail), GATE 2 **MET**.
The gate script's pre-committed mapping prints **FAILED**.
Per the idea text this negative is decision-grade: the marginal-calibration
chain (E5 → A8 → A14/A15 → A16 → B7 → B8) is CLOSED for the structural arm,
and RAW is certified as the final i7 ball stack for the I17 bundle.

Nothing shipped. No production, legacy, `models/auto/d16/`, `data/`, or
`scripts/sim_eval/` path was written.

---

## 1. Calibrator fit (`research/handoff/D17/raw/fit_calibrator.log.txt`)

Fit with the EXISTING `scripts/auto/d16_fit_vector_calibrator.py` (not edited),
pointed at `--model-dir models/auto/d16/noweights`. Output verbatim:

```
training contract: delivery_semantics='inclusive_total_runs_v1'  data_version='i7'  venue_identity='venue_aliases_v1'
validation parquet: data/xgb_data_i7/cricket_data_i7_validation.parquet  rows=124,292  features=114
encoding path: explicit training-time encoders from models/auto/d16/noweights (batter/bowler/matchup/venue)
  venue encoder ACTIVE (373 venues); distinct venue codes in val: 101
  cross-check _apply_encoders_to_df[batter_encoded]: identical
  cross-check _apply_encoders_to_df[bowler_encoded]: identical
  cross-check _apply_encoders_to_df[venue_encoded]: identical
  cross-check _apply_encoders_to_df[matchup_type_encoded]: identical

validation balls: 124,292
raw log loss:        1.433437
calibrated log loss: 1.433238
actual marginals:    [0.297284 0.40367  0.071598 0.11259  0.059143 0.055716]
raw marginals:       [0.294303 0.401781 0.074153 0.112108 0.058167 0.059488]
calibrated marginal: [0.297284 0.40367  0.071598 0.11259  0.059143 0.055716]
weights (fitted 6-vector): [0.17029491 0.16923329 0.16252808 0.16918399 0.17117034 0.15758939]
```

Encoder cross-check passed on all four columns (no exception fired).
Val LL improvement: **1.433437 → 1.433238** (0.000199).
Fit residual `max |calibrated marginal − actual|` = **2.810e-11**.

Fit-residual table (verbatim):

| class | actual | raw | calibrated | resid |
|---|---:|---:|---:|---:|
| dot | 0.297284 | 0.294303 | 0.297284 | -2.81e-11 |
| one | 0.403670 | 0.401781 | 0.403670 | +1.39e-11 |
| two | 0.071598 | 0.074153 | 0.071598 | +3.66e-12 |
| four | 0.112590 | 0.112108 | 0.112590 | +1.24e-12 |
| six | 0.059143 | 0.058167 | 0.059143 | +6.39e-12 |
| wicket | 0.055716 | 0.059488 | 0.055716 | +2.91e-12 |

## 2. Pre-run expectation check (REQUIRED; written BEFORE the eval)

`research/handoff/D17/raw/expectation_check.txt`, committed in `538ca24`
before the sim was launched.

**Scale caveat, recorded not worked around:** `VectorScalingCalibrator.fit`
ends every iteration with `v = v / v.sum()` (`scripts/calibration.py:440`), so
the fitted vector sums to 1 and the IDENTITY vector is 1/6 = 0.1666667 per
element, **not 1**. The plan's literal `max |v−1|` is therefore degenerate — it
is ≈0.83+ for every possible vector including exact identity. Both readings
were recorded; both give the same outcome, so nothing hinges on the choice.

- literal `max |v − 1|` = **0.8424106112460958** → ≥ 0.05 → live test
- identity-relative `max |6v − 1|` = **0.05446366747657527** → ≥ 0.05 → live test

**Recorded outcome: "live test"** — but flagged in the file as a MARGINAL one
on the meaningful scale: only the wicket class exceeds the ~0.05 A8/A12/B7
washout threshold, and it clears by 0.0045; the other five classes are all
below it.

Per-class identity-relative deviation (6v − 1):

| class | 6v | dev |
|---|---:|---:|
| dot | 1.021769 | +0.021769 |
| one | 1.015400 | +0.015400 |
| two | 0.975168 | -0.024832 |
| four | 1.015104 | +0.015104 |
| six | 1.027022 | +0.027022 |
| wicket | 0.945536 | **-0.054464** |

For scale, the same script's D16 CONTROL (balanced-weights) fit:
`6v = [1.6145, 1.950837, 0.655079, 0.759885, 0.572557, 0.447141]`,
`max |6v−1| = 0.950837072326487`. The control arm needed a ~2×/0.45×
correction; the no-weights arm needs at most 5.4%. That gap is the D16
structural result restated in calibrator parameters, and it is why D17 was
expected to be a small effect in either direction.

## 3. Eval run

One run, recipe B, Arm N settings + the d17 calibrator, exactly as planned.
Launched detached (nohup + tee) and waited synchronously in-session.

- start `2026-08-01T01:15:01Z`, exit by `2026-08-01T01:36:47Z`
- **`Done in 1309.8s`** (21.83 min) — inside the 22–40 min estimate, far
  under the 100 min kill threshold. D16 Arm N raw was 1297.1 s.
- **`[261/261]` matches, 0 skips.** Grep for
  `skip|error|traceback|exception|fail` over the full log: **count 0**.
- `Detail written to models/auto/d17/detail_noweights_vec_s46_n261.json`
- `Report written to models/auto/d17/report_noweights_vec_s46_n261.md`

**All four required banners verified verbatim** (`run_noweights_vec.log`
lines 1, 3, 8, 12):

```
1:Ball calibrator: vector scaling (models/auto/d17/vector_scaling_calibrator_d17.pkl)
3:StatsProvider: using SQLite backend player_stats_cache_i7.sqlite (56.7 MB)
8:Loaded XGBoost v2 model with 114 features with real player stats with player metadata; venue encoder ACTIVE (373 venues)
12:B10 usage-aligned bowler selector ACTIVE (k_u=5.0)
```

**373 venues, not 467** — correct model. Supporting banners identical to D16
Arm N: `Bowler selector: empirical`,
`Run-out dismissal channel ACTIVE (p_runout=0.0751, nonstriker_share=0.4685)`,
`as-of corpus: .../models/b10_usage_corpus.pkl (7433 players); min_eligible=5, min_share=0.01`.
`--bowler-usage-path` was NOT passed (default B12-shipped B10 sidecar).

Operational note: the first ~5 min of the run showed only 3 log lines. This was
Python's block-buffered stdout through the `tee` pipe, not a hang — `ps` showed
the child (PID 6851) at 135.6% CPU, 8:16 CPU-time, 469 MB RSS, state `RN`. The
buffer flushed normally and the run completed with full per-match output. No
intervention was taken.

## 4. Gate results (`research/handoff/D17/raw/gate_output.txt`)

`uv run python scripts/auto/d17_gate_analysis.py` — all defaults.
Baseline 261 matches, d17 261 matches. Cluster bootstrap by match,
n_boot=2000, seed=29. delta = `noweights_vec − noweights_raw`
(negative = calibrated better).

### GATE 1(i) — pooled tail dBrier must be CI-clean negative

| stat | n | drop | noweights_raw | noweights_vec | delta | 95% CI | flag |
|---|---:|---:|---:|---:|---:|---|---|
| pooled tail | 4673 | 0 | 0.2372 | 0.2383 | **+0.0011** | **[-0.0001,+0.0022]** | ~noise |

Context, equal weight per family: **-0.0010**.
**GATE 1(i): FAIL** (not CI-clean negative; point estimate is the wrong sign).

Per-family breakdown (context; the pool is the gate):

| family | n | drop | raw | vec | delta | 95% CI | flag |
|---|---:|---:|---:|---:|---:|---|---|
| pp_total_ou_45_5 | 522 | 0 | 0.2367 | 0.2340 | -0.0027 | [-0.0053,+0.0001] | ~noise |
| pp_total_ou_50_5 | 522 | 0 | 0.2355 | 0.2329 | -0.0026 | [-0.0053,+0.0001] | ~noise |
| pp_total_ou_55_5 | 522 | 0 | 0.2005 | 0.1990 | -0.0015 | [-0.0038,+0.0007] | ~noise |
| bowler_wkts_1plus | 3107 | 0 | 0.2437 | 0.2465 | +0.0028 | [+0.0012,+0.0044] | UP(worse) |

Positional cross-checks agreed in sign and flag on all four
(pp45 -0.0027, pp50 -0.0026, pp55 -0.0015, bowler_wkts_1plus +0.0042
[+0.0019,+0.0064] on n=2653).

**Structural note (worth the orchestrator's attention, does not change the
gate):** the row pool is dominated by `bowler_wkts_1plus` (3107 of 4673 rows =
66.5%), which is the ONE family that regresses CI-clean. All three PP-total
families move favorably. Hence the sign split between the row-pool statistic
(+0.0011, the gate) and the equal-weight-per-family context (-0.0010). The
row-pool weighting was pre-committed in the plan and was NOT revisited after
seeing the numbers.

### GATE 1(ii) — batter_runs_mae must NOT regress CI-clean

| family | n | drop | raw | vec | delta | 95% CI | flag |
|---|---:|---:|---:|---:|---:|---|---|
| batter_runs_mae | 4254 | 0 | 13.8864 | 14.0303 | **+0.1439** | **[+0.0942,+0.1949]** | UP(worse) |

Positional cross-check: +0.1514 [+0.0955,+0.2092] on n=3623, same flag.
**GATE 1(ii): FAIL** — CI-clean regression. This is exactly the trade the
E5-era calibrator historically lost, reproduced here on the structural arm.

**GATE 1 (i AND ii): NOT MET  [i=False ii=False]**

### GATE 2 — guards

| family | n | drop | raw | vec | delta | 95% CI | flag | verdict |
|---|---:|---:|---:|---:|---:|---|---|---|
| top_bowler | 5833 | 2 | 0.0775 | 0.0776 | +0.0001 | [-0.0003,+0.0005] | ~noise | ok |
| team_first_over_mae | 522 | 0 | 3.3785 | 3.3951 | +0.0166 | [-0.0002,+0.0337] | ~noise | ok |

**GATE 2: MET.** (`team_first_over_mae` is borderline — CI lo -0.0002 — but
does not regress CI-clean.)

### CI-clean movers, full 33-family context scan

10 families exclude 0; **6 favorable, 4 unfavorable**:

| family | delta | 95% CI | flag |
|---|---:|---|---|
| innings_runs_ou_170_5 | -0.0094 | [-0.0151,-0.0041] | DOWN(better) |
| innings_runs_ou_160_5 | -0.0089 | [-0.0154,-0.0026] | DOWN(better) |
| innings_runs_ou_180_5 | -0.0078 | [-0.0118,-0.0040] | DOWN(better) |
| highest_over_runs_ou_18_5 | -0.0067 | [-0.0130,-0.0003] | DOWN(better) |
| first_wicket_runs_ou_30_5 | -0.0040 | [-0.0068,-0.0013] | DOWN(better) |
| bowler_economy_ou_8_5 | -0.0023 | [-0.0043,-0.0004] | DOWN(better) |
| bowler_wkts_2plus | +0.0017 | [+0.0006,+0.0027] | UP(worse) |
| bowler_wkts_1plus | +0.0028 | [+0.0012,+0.0044] | UP(worse) |
| batter_fours_mae | +0.0117 | [+0.0067,+0.0167] | UP(worse) |
| batter_runs_mae | +0.1439 | [+0.0942,+0.1949] | UP(worse) |

Readable pattern: the calibrator helps TEAM/INNINGS-level totals and hurts
PER-PLAYER bowler-wicket and batter-runs resolution. It shaves the wicket
class by 5.4%, which flatters aggregate run totals and costs per-bowler wicket
and per-batter run accuracy. This is a redistribution, not a gain.

### Pre-committed mapping printed by the script

```
GATE 1: NOT MET | GATE 2: MET
Pre-committed verdict MAPPING (orchestrator decides): FAILED
```

## 5. Baseline-consistency cross-check vs the D16 gate

The D16 anchors the plan told me not to re-derive, reproduced exactly where
the row set is identical:

- pooled tail, noweights_raw arm: D16 printed **0.2372**, D17 baseline column
  prints **0.2372** — identical, same n=4673, 0 dropped.
- `team_first_over_mae`, noweights_raw arm: D16 **3.3785**, D17 baseline
  **3.3785** — identical, n=522.
- `batter_runs_mae`: D16 **13.8905** on n=4252 vs D17 baseline **13.8864** on
  n=4254. Expected and benign — the paired row set is the INTERSECTION with
  the partner detail, and D16's partner (control_vec) matched 2 fewer rows
  than D17's. Same underlying arm, different denominator.
- `top_bowler`: D16 **0.0772** (n=5827, 8 dropped) vs D17 baseline **0.0775**
  (n=5833, 2 dropped) — same cause.

## 6. Integrity checks

md5 before (`raw/md5_before.txt`, 01:14:13Z) vs after (`raw/md5_after.txt`,
01:37:56Z) — `diff` of the two is **empty; ALL MD5s UNCHANGED**:

| file | md5 before | md5 after |
|---|---|---|
| `models/auto/d16/detail_noweights_raw_s46_n261.json` | d816ebcd5cc9190bc4c4ca578dd6bbf1 | d816ebcd5cc9190bc4c4ca578dd6bbf1 |
| `models/auto/d16/noweights/xgboost_model_i7.pkl` | 7ee1e1809917f45be7e726b3ea4a8a6c | 7ee1e1809917f45be7e726b3ea4a8a6c |
| `models/xgb_i7/xgboost_model_i7.pkl` | 32b81888407dcd5cd652eb72b4ad7725 | 32b81888407dcd5cd652eb72b4ad7725 |
| `models/xgb_v3/xgboost_model_v3.pkl` | d448822e9bcd8cb93c126f8558b8ab46 | d448822e9bcd8cb93c126f8558b8ab46 |
| `models/auto/d17/vector_scaling_calibrator_d17.pkl` | c3e915ab4cc1f90a7214dec15d7b89d6 | c3e915ab4cc1f90a7214dec15d7b89d6 |

The baseline detail and both protected production boosters stayed byte-frozen
across the whole session.

`git diff --stat 2a9389b` at the implement commit (`538ca24`, before any eval
output existed):

```
 research/handoff/D17/plan.md                    | 182 +++++++++++++++++++
 research/handoff/D17/raw/expectation_check.txt  |  49 +++++
 research/handoff/D17/raw/fit_calibrator.log.txt |  27 +++
 research/handoff/D17/raw/md5_before.txt         |   7 +
 scripts/auto/d17_gate_analysis.py               | 231 ++++++++++++++++++++++++
 5 files changed, 496 insertions(+)
```

Zero source files modified — D17 is purely additive (one new gate script under
`scripts/auto/`, plus handoff evidence). `scripts/sim_eval/`,
`scripts/auto/d16_*.py`, `scripts/calibration.py`, all production and legacy
paths, and `data/golden/` were never touched or read.

## 7. Artifacts

Gitignored (`models/`), so they do not survive in git:
`models/auto/d17/vector_scaling_calibrator_d17.pkl`,
`models/auto/d17/detail_noweights_vec_s46_n261.json`,
`models/auto/d17/report_noweights_vec_s46_n261.md`.

Durable evidence under `research/handoff/D17/raw/` (`*.log` is gitignored, so
every log has a `.txt` twin): `expectation_check.txt`,
`fit_calibrator.log.txt`, `run_noweights_vec.log.txt`,
`gate_output.txt`, `report_noweights_vec_s46_n261.md.txt`,
`md5_before.txt`, `md5_after.txt`.

## 8. Anomalies

None material. Two things logged for the record: (a) the `max |v−1|`
parameterisation mismatch in the plan, handled by recording both readings —
they agree, so the expectation check is unambiguous; (b) the ~5-minute
block-buffered log silence at startup, diagnosed as buffering via `ps` and not
a hang. Nothing crashed, nothing ran long, no skips, no encoder mismatch.
