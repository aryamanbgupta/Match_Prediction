# B15 executor result — Scale-only quote calibrator re-gated on fresh test draws

Idea id: **B15** (P3, B14 follow-up). Claim commit: `71691d0`.
Executor: implementation + evaluation only. **The verdict is the orchestrator's.**

Both pre-committed primaries are **MET** on the scale-only arm (facts below).

---

## 1. Pre-run integrity checks

### Usage pin (md5 pre-check)

Printed by the eval command before the run (`research/handoff/B15/raw/quotes_s45_run.log`
is the tee'd run log; the md5/date lines went to the wrapping shell):

```
ea0c73d3ddb48f499b6273f9a397b0e3
Fri Jul 31 19:34:38 UTC 2026
```

Matches the required `ea0c73d3ddb48f499b6273f9a397b0e3` exactly.

Independent confirmation that the pin actually matters:

```
b10_asof_usage in pinned json: False
b10_asof_usage in default shipped json: True
```

### Banner check (must be absent)

```
=== banner check (must be 0) ===
0
```

`grep -c "B10 usage-aligned bowler selector ACTIVE"` over the full run log = **0**.
The run is NOT confounded by the post-B12 B10-active default prior.

### Venue encoder

```
Loaded XGBoost v2 model with 114 features with real player stats with player metadata; venue encoder ACTIVE (467 venues)
```

ACTIVE with 467 venues, as expected (default sidecar).

### Row / match / skip counts + runtime

```
Done in 1487.2s — 756 quote rows from 253 matches (8 matches skipped) -> models/auto/b15/quotes_s45_n261.json
uv run python scripts/auto/b5_inplay_quotes.py --test-dir data/polymarket_tes  1422.61s user 427.78s system 124% cpu 24:49.23 total
```

```
rows 756 matches 253 skips 8
config {'test_dir': 'data/polymarket_test', 'n_sims': 100, 'seed': 45, 'checkpoints': [6, 10, 15], 'model': 'models/xgb_v3/xgboost_model_v3.pkl', 'ball_calibrator': 'models/xgb_v3/vector_scaling_calibrator_v1.pkl', 'bowler_selector': 'empirical', 'usage_json': 'models/auto/b12/bowler_phase_usage_pre_b12.json', 'quote_center': 'sim_p50', 'elapsed_s': 1487.2287542819977}
```

756 rows / 253 matches / 8 skips — **exactly** B5's counts (the skips are the same
structural set: rain-curtailed + fifty-over). Well inside the STOP thresholds
(matches ≥ 240, skips ≤ 15). Runtime 1487.2 s vs B5's 1495.8 s; wall clock
24:49, far under the 2× (≈50 min) kill budget. Seed 45, ∉ {42, 43, 44, 47}.

### Read-only inputs unmodified

mtimes/sizes before and after the whole session are byte-identical:

```
models/auto/b5/quotes_s43_n261.json 1785486405 266869
models/auto/b14/quote_calibrator.json 1785525614 1084
models/auto/b14/quotes_val_s47_n545.json 1785525585 541365
```

No refit occurred anywhere; the gate script asserts the six val-fit values and
prints `calibrator assertion: PASS`.

---

## 2. Self-test — B14-full arm on the FROZEN s43 quotes (verbatim)

Run **before** the fresh quotes existed and committed in `7dfcb4a`
(full output: `research/handoff/B15/raw/selftest_frozen_s43.txt`).

```
=== ARM: B14-full (val-fit shift + scale) ===
checkpoint  6 (n=253):  applied shift -1.4482 scale 1.19
  MAE  corr(P50)  21.139  naive  25.897  dMAE  -4.759 [-7.790, -1.791]  CORRECTED BETTER
  P10-P90 coverage corr  0.802 [0.755, 0.850]  target [0.7, 0.9]  IN BAND
checkpoint 10 (n=253):  applied shift -1.7812 scale 1.09
  MAE  corr(P50)  17.435  naive  20.000  dMAE  -2.565 [-4.679, -0.392]  CORRECTED BETTER
  P10-P90 coverage corr  0.802 [0.755, 0.854]  target [0.7, 0.9]  IN BAND
checkpoint 15 (n=250):  applied shift -2.9715 scale 1.26
  MAE  corr(P50)  12.597  naive  13.575  dMAE  -0.977 [-2.565, +0.461]  CORRECTED BETTER
  P10-P90 coverage corr  0.756 [0.704, 0.808]  target [0.7, 0.9]  IN BAND

pooled paired dMAE (corrected - naive, 756 rows, cluster-boot by match): -2.774 [-4.631, -0.864]
```

```
=== SELF-TEST vs B14's logged frozen-quote numbers ===
  B14-full pooled dMAE  expected -2.774 [-4.631, -0.864]  got -2.774 [-4.631, -0.864]  PASS
  B14-full coverage     expected 0.802/0.802/0.756  got 0.802/0.802/0.756  PASS
  SELF-TEST: PASS
```

Bonus reproduction (not required, but confirms the raw path too): on the same
frozen quotes the script's uncorrected context reproduces B5's logged
`pooled paired dMAE (RAW - naive) -3.086 [-4.869, -1.289]` and raw coverage
`0.755 / 0.794 / 0.664`. The correction transform, error/coverage definitions,
and cluster-boot machinery are therefore identical to `b14_gate_analysis.py`.

---

## 3. Fresh-draw numbers (seed 45, `models/auto/b15/quotes_s45_n261.json`)

Full output: `research/handoff/B15/raw/gate_output.txt`. All figures below are
copied verbatim.

### Header

```
  calibrator assertion: PASS — all 6 val-fit values match B14 exactly (NO refit anywhere in this script)
  config: n_sims=100 seed=45 quote_center=sim_p50 usage_json=models/auto/b12/bowler_phase_usage_pre_b12.json elapsed=1487.2s
  rows: 756 from 253 matches (8 matches skipped)
  val-fit correction: cp6: shift -1.4482 scale 1.19  cp10: shift -1.7812 scale 1.09  cp15: shift -2.9715 scale 1.26
  bootstrap: 2000 reps, seed 29, cluster by match
```

### ARM: scale-only (shift := 0) — the gated arm

```
=== ARM: scale-only (shift := 0) ===
checkpoint  6 (n=253):  applied shift +0.0000 scale 1.19
  MAE  corr(P50)  20.773  naive  25.897  dMAE  -5.125 [-8.053, -2.176]  CORRECTED BETTER
  MAE  raw (P50)  20.773  naive  25.897  dMAE  -5.125 [-8.053, -2.176]   (uncorrected context)
  P10-P90 coverage corr  0.818 [0.771, 0.866]  target [0.7, 0.9]  IN BAND
  P10-P90 coverage raw   0.755 [0.704, 0.810]   (uncorrected context)
  context: bias corr P50 +4.259  bias raw P50 +4.259  band width corr 69.4  raw 58.3  actual sd 29.9
checkpoint 10 (n=253):  applied shift +0.0000 scale 1.09
  MAE  corr(P50)  16.990  naive  20.000  dMAE  -3.010 [-4.955, -1.009]  CORRECTED BETTER
  MAE  raw (P50)  16.990  naive  20.000  dMAE  -3.010 [-4.955, -1.009]   (uncorrected context)
  P10-P90 coverage corr  0.834 [0.787, 0.881]  target [0.7, 0.9]  IN BAND
  P10-P90 coverage raw   0.791 [0.739, 0.842]   (uncorrected context)
  context: bias corr P50 +2.777  bias raw P50 +2.777  band width corr 52.5  raw 48.2  actual sd 25.3
checkpoint 15 (n=250):  applied shift +0.0000 scale 1.26
  MAE  corr(P50)  12.338  naive  13.575  dMAE  -1.237 [-2.581, +0.016]  CORRECTED BETTER
  MAE  raw (P50)  12.338  naive  13.575  dMAE  -1.237 [-2.581, +0.016]   (uncorrected context)
  P10-P90 coverage corr  0.768 [0.716, 0.824]  target [0.7, 0.9]  IN BAND
  P10-P90 coverage raw   0.660 [0.604, 0.720]   (uncorrected context)
  context: bias corr P50 +0.410  bias raw P50 +0.410  band width corr 37.1  raw 29.4  actual sd 16.9

pooled paired dMAE (corrected - naive, 756 rows, cluster-boot by match): -3.131 [-4.909, -1.356]
pooled paired dMAE (RAW - naive, uncorrected context): -3.131 [-4.909, -1.356]
```

Note (by construction): the scale-only arm leaves P50 untouched, so its MAE /
dMAE / bias rows are identical to the raw uncorrected rows. Only the band moves
— e.g. cp15 mean band width 29.4 → 37.1, lifting cp15 coverage 0.660 → 0.768.

### ARM: B14-full (val-fit shift + scale) — context only

```
=== ARM: B14-full (val-fit shift + scale) ===
checkpoint  6 (n=253):  applied shift -1.4482 scale 1.19
  MAE  corr(P50)  21.032  naive  25.897  dMAE  -4.865 [-7.872, -1.916]  CORRECTED BETTER
  P10-P90 coverage corr  0.810 [0.763, 0.858]  target [0.7, 0.9]  IN BAND
  context: bias corr P50 +5.707  bias raw P50 +4.259  band width corr 69.4  raw 58.3  actual sd 29.9
checkpoint 10 (n=253):  applied shift -1.7812 scale 1.09
  MAE  corr(P50)  17.328  naive  20.000  dMAE  -2.672 [-4.758, -0.518]  CORRECTED BETTER
  P10-P90 coverage corr  0.802 [0.755, 0.854]  target [0.7, 0.9]  IN BAND
  context: bias corr P50 +4.558  bias raw P50 +2.777  band width corr 52.5  raw 48.2  actual sd 25.3
checkpoint 15 (n=250):  applied shift -2.9715 scale 1.26
  MAE  corr(P50)  12.617  naive  13.575  dMAE  -0.957 [-2.531, +0.471]  CORRECTED BETTER
  P10-P90 coverage corr  0.748 [0.696, 0.800]  target [0.7, 0.9]  IN BAND
  context: bias corr P50 +3.381  bias raw P50 +0.410  band width corr 37.1  raw 29.4  actual sd 16.9

pooled paired dMAE (corrected - naive, 756 rows, cluster-boot by match): -2.839 [-4.677, -0.943]
```

### Side-by-side MAE table (all copied from the two blocks above)

| cp | naive MAE | raw / scale-only corr MAE | B14-full corr MAE | scale-only dMAE (CI) | B14-full dMAE (CI) |
|---|---|---|---|---|---|
| 6  | 25.897 | 20.773 | 21.032 | -5.125 [-8.053, -2.176] | -4.865 [-7.872, -1.916] |
| 10 | 20.000 | 16.990 | 17.328 | -3.010 [-4.955, -1.009] | -2.672 [-4.758, -0.518] |
| 15 | 13.575 | 12.338 | 12.617 | -1.237 [-2.581, +0.016] | -0.957 [-2.531, +0.471] |
| pooled | — | — | — | **-3.131 [-4.909, -1.356]** | **-2.839 [-4.677, -0.943]** |

### Coverage table

| cp | raw coverage (CI) | scale-only corr coverage (CI) | B14-full corr coverage (CI) |
|---|---|---|---|
| 6  | 0.755 [0.704, 0.810] | 0.818 [0.771, 0.866] | 0.810 [0.763, 0.858] |
| 10 | 0.791 [0.739, 0.842] | 0.834 [0.787, 0.881] | 0.802 [0.755, 0.854] |
| 15 | 0.660 [0.604, 0.720] | 0.768 [0.716, 0.824] | 0.748 [0.696, 0.800] |

Raw cp15 coverage on fresh draws is 0.660 — OUT of [0.70, 0.90], reproducing
B5's failure mode (0.664 frozen). Both correction arms fix it.

### DIAGNOSTIC bias table (no gate weight)

```
 cp  val shift (fit)  this-quotes raw bias   B5 frozen raw bias
---------------------------------------------------------------
  6          -1.4482                +4.259               +4.670
 10          -1.7812                +2.777               +3.204
 15          -2.9715                +0.410               +0.514
```

The val→test sign mismatch that B14 flagged **replicates on independent fresh
draws**: val shifts are negative at all three checkpoints, fresh-test raw bias
is positive at all three. Fresh-draw biases are slightly smaller in magnitude
than the frozen s43 values but the same sign and ordering.

### SECONDARY (recommendation only, no gate weight)

```
  pooled dMAE scale-only -3.131 [-4.909, -1.356]
  pooled dMAE B14-full   -2.839 [-4.677, -0.943]
  scale-only - B14-full = -0.292  (more negative dMAE is better => scale-only better on point estimate)
```

### PRE-COMMITTED GATE (scale-only arm)

```
=== PRE-COMMITTED B15 GATE (scale-only arm) ===
PRIMARY-A (corrected coverage in [0.7, 0.9] at all 3 cps): MET (per-cp coverage [0.818, 0.834, 0.768], in-band [True, True, True])
PRIMARY-B (corrected MAE beats naive at all 3 cps + pooled cluster-boot CI hi < 0): MET (per-cp [True, True, True], pooled hi -1.356)
```

**PRIMARY-A: MET. PRIMARY-B: MET.** (Verdict mapping is the orchestrator's.)

---

## 4. Commits created

| SHA | message |
|---|---|
| `7dfcb4a` | `Auto[B15]: implement — pre-committed scale-only gate + frozen-quotes self-test` |
| (this commit) | `Auto[B15]: executor result — ...` |

`git diff --stat 71691d0` immediately before writing this file:

```
 research/handoff/B15/plan.md                       | 174 ++++++++++
 .../handoff/B15/raw/readonly_mtimes_before.txt     |   3 +
 research/handoff/B15/raw/selftest_frozen_s43.txt   |  72 +++++
 scripts/auto/b15_gate_analysis.py                  | 355 +++++++++++++++++++++
 4 files changed, 604 insertions(+)
```

ONE new file was added under `scripts/`: `scripts/auto/b15_gate_analysis.py`.
Nothing else in `scripts/` changed — `b5_inplay_quotes.py`, `b5_gate_analysis.py`,
`b14_gate_analysis.py`, `b14_fit_quote_calibrator.py`, `scripts/sim_eval/`, and
`sim_v1_2.py` are byte-untouched. `models/` is gitignored, so the fresh quote
artifacts stay on disk only. `research/results.tsv` and `research/IDEAS.md` were
not touched.

---

## 5. Crashes / hangs / surprises

- **Nothing crashed or hung.** The quote run exited 0 in 1487.2 s.
- Harness note (not a run problem): the foreground Bash tool caps at 600 s, so
  the 25-minute quote run was moved to a background task by the harness after
  10 minutes. It was NOT abandoned — the executor blocked on it until the output
  json existed and then verified exit code 0 and the `Done in 1487.2s` line. No
  background processes remain (`pgrep -fl b5_inplay_quotes` → none running).
- The tee'd run log looked frozen at 3 lines for ~15 minutes because Python
  block-buffers stdout through a pipe; the `.partial.jsonl` row count was used
  to confirm live progress (337 rows at 11 min, 613 at ~19 min). The full 274-line
  log flushed at exit and is intact.
- A pre-existing warning appears twice in the run log and is unrelated to B15:
  `WARNING: SQLite same-day ordering mismatch: models/player_stats_cache_v3.sqlite
  has None, code expects 'date_then_match_id_lexicographic_v1'.` The same warning
  is present in B5's and B14's runs, so it does not differentiate the arms.
- Mild surprise worth flagging to the orchestrator: because the scale-only arm
  does not move P50, its pooled dMAE **is** the raw B5-style statistic. On these
  fresh draws that number is -3.131 [-4.909, -1.356] versus B5's frozen
  -3.086 [-4.869, -1.289] — i.e. the seed-to-seed movement in the headline
  statistic (0.045) is an order of magnitude smaller than the shift term's cost
  (0.292), so the scale-only > B14-full ordering is not plausibly seed noise.
  Framed differently: on fresh draws the B15 arm is "keep B5's P50 exactly,
  widen the band," and it clears both bars where B5's raw bands did not.
