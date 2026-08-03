# B14 — executor result

Per-checkpoint quote-layer recalibration for in-play bands (B5 follow-up).
Executed 2026-07-31. Verdict is **NOT** decided here — the orchestrator applies
the mapping (both gates → LANDED; exactly one → TABLED; none → FAILED).

**Headline: GATE 1' MET, GATE 2' MET.**

## Relay note (process, not results)

This iteration was executed by **two** sessions. The original executor did plan
steps 1–4 (implement + smoke + launch the full val quote run) and its session
ended while the long eval was still in flight as a detached process (PID 60229,
child of uv PID 60228). A **replacement executor** picked the iteration up at
15:08 local, waited on the surviving process to completion (it was never
restarted, never killed, and no code was re-implemented or re-committed), then
ran plan steps 5–7: the val calibrator fit, the pre-committed gate analysis,
this report, and the result commit. The commits `4a57f0a` (claim), `66b6cc4`
(plan) and `5e9c963` (implement) are the original executor's and were left
untouched.

Nothing crashed. Nothing was retried. The val run took 3215.2 s (~53.6 min)
wall clock against an estimate of ~55 min, well inside the 2.5 h abort ceiling.

## Pre-flight re-verification (Step B — both passed)

- `md5 -q models/auto/b12/bowler_phase_usage_pre_b12.json` →
  `ea0c73d3ddb48f499b6273f9a397b0e3` (matches the expected hash exactly).
  A `json.load` of that file has top-level keys
  `['built_at', 'by_player', 'by_year_league', 'gender', 'global_league',
  'n_deliveries', 'n_matches', 'n_unresolved_names', 'schema_version',
  'source_dir']` and **no `b10_asof_usage` key** — usage is pinned to the
  pre-B12 legacy prior, as the plan requires.
- `models/auto/b5/quotes_s43_n261.json` (frozen TEST quotes, never touched or
  regenerated) still reads:
  `config = {"ball_calibrator": "models/xgb_v3/vector_scaling_calibrator_v1.pkl", "bowler_selector": "empirical", "checkpoints": [6, 10, 15], "elapsed_s": 1495.8358759880066, "model": "models/xgb_v3/xgboost_model_v3.pkl", "n_sims": 100, "quote_center": "sim_p50", "seed": 43, "test_dir": "data/polymarket_test"}`,
  **756 rows / 253 matches / 8 skips**.

## Val quote run (Step A)

Command (exactly as launched by the original executor):

```
uv run python scripts/auto/b5_inplay_quotes.py \
    --test-dir data/auto/b3/val_matches --n-matches all --n-sims 100 --seed 47 \
    --usage-json models/auto/b12/bowler_phase_usage_pre_b12.json \
    --out models/auto/b14/quotes_val_s47_n545.json
```

Final line, verbatim:

```
Done in 3215.2s — 1532 quote rows from 512 matches (33 matches skipped) -> models/auto/b14/quotes_val_s47_n545.json
```

- **Rows: 1532. Matches: 512. Skips: 33 of 545 = 6.06%**, under the plan's 10%
  abort condition (test-set precedent was 8/261 ≈ 3.07%). All 33 skips are the
  harness's standard structural exclusions — 30 `innings 1 curtailed (...)`,
  3 `not a 20-over match (overs=50)`. None are model or usage-prior failures.
- Startup banner confirmed **`venue encoder ACTIVE (467 venues)`** and
  **no `B10 ... ACTIVE` line** (a grep for `b10|usage` over the full 562-line
  log returns nothing), i.e. the pinned legacy usage prior was in force for
  every one of the 545 attempted matches, not just the smoke subset.
- The written JSON records `"usage_json": "models/auto/b12/bowler_phase_usage_pre_b12.json"`,
  `"seed": 47`, `"n_sims": 100`, `"test_dir": "data/auto/b3/val_matches"`.

## Step C — val fit table (verbatim from `research/handoff/B14/raw/fit.log`)

```
B14 quote-calibrator fit on models/auto/b14/quotes_val_s47_n545.json
  config: n_sims=100 seed=47 quote_center=sim_p50 usage_json=models/auto/b12/bowler_phase_usage_pre_b12.json elapsed=3215.2s
  rows: 1532 from 512 matches (33 matches skipped)

 cp     n    shift  scale  cov_raw  cov_corr  mae_raw  mae_corr
---------------------------------------------------------------
  6   512  -1.4482   1.19   0.7344    0.7988  20.2490   20.1953
 10   512  -1.7812   1.09   0.7754    0.8027  16.0430   16.0785
 15   508  -2.9715   1.26   0.7106    0.7992  11.1604   11.0864

Wrote models/auto/b14/quote_calibrator.json
```

Full-precision parameters as persisted in `models/auto/b14/quote_calibrator.json`:
cp6 `shift` −1.4482 / `scale` 1.19; cp10 `shift` −1.78125 / `scale` 1.09;
cp15 `shift` −2.9714566929133857 / `scale` 1.26. Corrected val coverages land at
0.7988 / 0.8027 (`0.802734375`) / 0.7992 (`0.7992125984251969`) against the
pre-committed 0.80 val fitting target — coverage objectives 0.0027343749999999556
(cp10) and 0.0007874015748031704 (cp15), i.e. the scale grid resolved the target
essentially exactly.

**Note the sign.** On val the fitted shifts are all *negative*
(`shift = mean(sim_p50 − actual)`), so the sim under-predicts remaining runs on
the val pool, while on the frozen test pool the raw P50 bias is *positive*
(+4.670 / +3.204 / +0.514). The correction therefore pushes the test P50 further
in the direction of its existing bias (corrected test bias +6.118 / +4.985 /
+3.485). This is a genuine val/test bias-sign mismatch and it is why every
corrected test MAE is slightly *worse* than its uncorrected counterpart. It did
not break GATE 1' — the correction is small relative to the sim-vs-naive
margin — but it is the honest weak point of this result and is called out again
under "Anomalies".

## Step C — gate analysis (verbatim from `research/handoff/B14/raw/gate_output.txt`)

```
B14 gate analysis
  calibrator: models/auto/b14/quote_calibrator.json (fit on models/auto/b14/quotes_val_s47_n545.json, val target coverage 0.8)
  TEST quotes: models/auto/b5/quotes_s43_n261.json
  config: n_sims=100 seed=43 quote_center=sim_p50 elapsed=1496s
  rows: 756 from 253 matches (8 matches skipped)
  correction applied (val-fit): cp6: shift -1.4482 scale 1.19  cp10: shift -1.7812 scale 1.09  cp15: shift -2.9715 scale 1.26

checkpoint  6 (n=253):
  MAE  corr(P50)  21.139  naive  25.897  dMAE  -4.759 [-7.790, -1.791]  CORRECTED BETTER
  MAE  raw (P50)  20.860  naive  25.897  dMAE  -5.038 [-7.970, -2.082]   (uncorrected context)
  P10-P90 coverage corr  0.802 [0.755, 0.850]  target [0.7, 0.9]  IN BAND
  P10-P90 coverage raw   0.755 [0.704, 0.810]   (uncorrected context)
  context: bias corr P50 +6.118  bias raw P50 +4.670  band width corr 68.8  raw 57.8  actual sd 29.9
checkpoint 10 (n=253):
  MAE  corr(P50)  17.435  naive  20.000  dMAE  -2.565 [-4.679, -0.392]  CORRECTED BETTER
  MAE  raw (P50)  17.061  naive  20.000  dMAE  -2.939 [-4.947, -0.909]   (uncorrected context)
  P10-P90 coverage corr  0.802 [0.755, 0.854]  target [0.7, 0.9]  IN BAND
  P10-P90 coverage raw   0.794 [0.743, 0.846]   (uncorrected context)
  context: bias corr P50 +4.985  bias raw P50 +3.204  band width corr 52.3  raw 48.0  actual sd 25.3
checkpoint 15 (n=250):
  MAE  corr(P50)  12.597  naive  13.575  dMAE  -0.977 [-2.565, +0.461]  CORRECTED BETTER
  MAE  raw (P50)  12.314  naive  13.575  dMAE  -1.261 [-2.613, -0.004]   (uncorrected context)
  P10-P90 coverage corr  0.756 [0.704, 0.808]  target [0.7, 0.9]  IN BAND
  P10-P90 coverage raw   0.664 [0.608, 0.724]   (uncorrected context)
  context: bias corr P50 +3.485  bias raw P50 +0.514  band width corr 37.2  raw 29.5  actual sd 16.9

pooled paired dMAE (corrected - naive, 756 rows, cluster-boot by match): -2.774 [-4.631, -0.864]

GATE 1' (corrected MAE beats naive at all 3 cps + pooled cluster-boot CI hi < 0): MET (per-cp [True, True, True], pooled hi -0.864)
GATE 2' (corrected coverage in [0.7,0.9] at all 3 cps): MET (per-cp [True, True, True])

(verdict mapping is applied by the orchestrator, not here)
```

### Corrected TEST MAE vs naive (GATE 1' stat)

| cp | n | corrected MAE | naive MAE | paired dMAE | cluster-boot CI (2000 reps, seed 29, by match) | raw dMAE (context) |
|---|---|---|---|---|---|---|
| 6 | 253 | 21.139 | 25.897 | −4.759 | [−7.790, −1.791] | −5.038 [−7.970, −2.082] |
| 10 | 253 | 17.435 | 20.000 | −2.565 | [−4.679, −0.392] | −2.939 [−4.947, −0.909] |
| 15 | 250 | 12.597 | 13.575 | −0.977 | [−2.565, +0.461] | −1.261 [−2.613, −0.004] |
| **pooled** | **756 rows** | — | — | **−2.774** | **[−4.631, −0.864]** | −3.086 [−4.869, −1.289] |

GATE 1' requires the *point estimate* to beat naive at all three checkpoints
(it does: −4.759 / −2.565 / −0.977) and the **pooled** CI hi < 0 (it is:
−0.864). The cp15 per-checkpoint CI straddles zero ([−2.565, +0.461]); the gate
as pre-committed does not require per-cp CI cleanliness, only per-cp point
estimates plus the pooled CI, so this is inside the bar as written — but it is
weaker than the uncorrected cp15 CI ([−2.613, −0.004]), which just cleared zero.

### Corrected TEST P10–P90 coverage (GATE 2' stat)

| cp | corrected coverage | corrected CI | uncorrected coverage | uncorrected CI | band [0.70, 0.90] |
|---|---|---|---|---|---|
| 6 | 0.802 | [0.755, 0.850] | 0.755 | [0.704, 0.810] | IN BAND |
| 10 | 0.802 | [0.755, 0.854] | 0.794 | [0.743, 0.846] | IN BAND |
| 15 | 0.756 | [0.704, 0.808] | 0.664 | [0.608, 0.724] | IN BAND |

This is the intended effect: cp15 was B5's sole failure (0.664, below the 0.70
floor) and the val-fit widening lifts it to 0.756 with a CI whose lower bound
(0.704) now also sits above the floor, without pushing cp6 or cp10 through the
0.90 ceiling.

## Comparison to the B5 baseline (results.tsv row 2026-07-31, idea B5)

| metric | B5 uncorrected | B14 corrected | direction |
|---|---|---|---|
| cp6 MAE vs naive | 20.860 vs 25.897 (−5.038 [−7.970, −2.082]) | 21.139 vs 25.897 (−4.759 [−7.790, −1.791]) | slightly worse margin, still CI-clean |
| cp10 MAE vs naive | 17.061 vs 20.000 (−2.939 [−4.947, −0.909]) | 17.435 vs 20.000 (−2.565 [−4.679, −0.392]) | slightly worse margin, still CI-clean |
| cp15 MAE vs naive | 12.314 vs 13.575 (−1.261 [−2.613, −0.004]) | 12.597 vs 13.575 (−0.977 [−2.565, +0.461]) | worse; per-cp CI no longer excludes 0 |
| pooled dMAE | −3.086 [−4.869, −1.289] | −2.774 [−4.631, −0.864] | ~0.31 smaller margin, still CI-clean |
| cp6 coverage | 0.755 IN | 0.802 IN | closer to 0.80 |
| cp10 coverage | 0.794 IN | 0.802 IN | ~flat, closer to 0.80 |
| cp15 coverage | **0.664 OUT** | **0.756 IN** | **fixed — the whole point of B14** |

The trade is explicit and modest: B14 buys the cp15 coverage failure at the
cost of ~0.31 of pooled dMAE margin. No engine change, no sim-engine slot
consumed, default sim path untouched.

## Anomalies / caveats (recorded honestly, for the orchestrator)

1. **Val/test bias sign mismatch (the main one).** Fitted shifts are negative
   on val; raw test bias is positive at all three checkpoints. The de-biasing
   term therefore moves the test P50 the *wrong way* (corrected test bias
   +6.118 / +4.985 / +3.485 vs raw +4.670 / +3.204 / +0.514), and every
   corrected test MAE is worse than its uncorrected counterpart. GATE 1' still
   passes on the pre-committed construction, but essentially all of B14's value
   is coming from the *scale* (band-widening) term, not the shift. A shift-free
   variant (scale only) is the obvious follow-up and was not run — running it
   now would be post-hoc selection against the frozen test set, which the plan
   forbids.
2. **cp15 per-checkpoint MAE CI now straddles zero** ([−2.565, +0.461]) where
   uncorrected it did not ([−2.613, −0.004], itself barely clean). The gate is
   met via the pooled CI as pre-committed; nobody should quote cp15 alone as a
   CI-clean win after correction.
3. **Skip rate is 2× the test precedent** (6.06% vs 3.07%) but well under the
   10% abort bar, and every skip is structural (curtailed first innings, or
   50-over fixtures in the val pool).
4. The val run emits the pre-existing
   `WARNING: SQLite same-day ordering mismatch: models/player_stats_cache_v3.sqlite has None, code expects 'date_then_match_id_lexicographic_v1'`
   banner. This is the known repo-wide cache-metadata warning, identical to the
   one present in the frozen B5 test run, so it does not differentiate the two
   pools.
5. The calibrator is fit on val only (seed 47, `data/auto/b3/val_matches`) and
   applied unchanged to the frozen test quotes (seed 43). The test quotes file
   was read only; its mtime and 756/253/8 shape are unchanged.

## Commits

| role | sha |
|---|---|
| claim | `4a57f0a` |
| plan | `66b6cc4` |
| implement | `5e9c963` |
| result (this report) | the commit carrying this file and `raw_excerpt.md`, message `Auto[B14]: executor result — GATE 1' MET (pooled dMAE -2.774 [-4.631, -0.864]), GATE 2' MET (corrected coverage 0.802/0.802/0.756 all in band)`. A commit cannot embed its own hash; its SHA is `git log -1 --format=%h` at the tip of `auto-20260731` and is quoted verbatim in the executor's closing message. |

`git diff --stat 4a57f0a` (implementation surface, before this report commit):

```
 research/handoff/B14/plan.md             | 178 +++++++++++++++++++++++++++++++
 scripts/auto/b14_fit_quote_calibrator.py | 132 +++++++++++++++++++++++
 scripts/auto/b14_gate_analysis.py        | 178 +++++++++++++++++++++++++++++++
 scripts/auto/b5_inplay_quotes.py         |   6 +-
 4 files changed, 493 insertions(+), 1 deletion(-)
```

The only production-path file touched is `scripts/auto/b5_inplay_quotes.py`
(+6/−1), the additive `--usage-json` flag; with the flag absent the script's
behaviour is unchanged.

## Artifacts

- `models/auto/b14/quotes_val_s47_n545.json` — val quotes, 1532 rows / 512
  matches / 33 skips, seed 47.
- `models/auto/b14/quote_calibrator.json` — fitted per-checkpoint shift/scale.
- `research/handoff/B14/raw/val_run.log`, `fit.log`, `gate_output.txt`,
  `smoke.log` — full logs, left untracked on disk per the plan.
- `research/handoff/B14/raw_excerpt.md` — committed small excerpt.
