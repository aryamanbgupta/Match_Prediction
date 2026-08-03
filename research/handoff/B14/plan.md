# B14 — Per-checkpoint quote-layer recalibration for in-play bands (B5 follow-up)

Orchestrator plan, 2026-07-31. Claim commit: `4a57f0a`.

## Idea, hypothesis, gate pair

**Idea id:** B14 (research/IDEAS.md, P2, claimed RUNNING 2026-07-31T18:10Z).

**Hypothesis:** B5 (TABLED earlier today) proved the sim's in-play
remaining-runs P50 beats naive run-rate extrapolation CI-clean at every
checkpoint (pooled dMAE −3.086 [−4.869, −1.289]) and failed ONLY the cp-15
coverage bar: P10–P90 coverage 0.664 [0.608, 0.724] vs the 0.70 floor
(late-innings band too narrow — width 29.5 vs actual sd 16.9). A post-hoc
per-checkpoint correction — bias shift on P50 plus multiplicative widening of
the P10/P90 band about the corrected P50, both fit on VAL quotes only —
should lift cp15 into the band without disturbing the GATE-1 MAE win and
without any engine change. Non-engine (B3/B5 precedent — no sim-engine slot
consumed; default sim path untouched; B12 consumed today's engine slot).

**Gate pair (sim/prop gate — pre-committed in the IDEAS.md entry):**
- **GATE 1′ (no-regress):** corrected P50 still beats naive on remaining-runs
  MAE at ALL THREE checkpoints (point estimate) AND the pooled paired per-row
  delta |corr_err| − |naive_err| has a cluster-bootstrap CI (by match, 2000
  reps, seed 29 — the exact `b5_gate_analysis.py` construction) with hi < 0.
- **GATE 2′ (calibration):** corrected inclusive P10–P90 coverage lies in
  [0.70, 0.90] at ALL THREE checkpoints (point estimates; CIs as context).

Verdict mapping (orchestrator applies it, NOT you): both → LANDED; exactly
one → TABLED; none → FAILED.

## Baseline to compare against (results.tsv row dated 2026-07-31, idea B5)

Uncorrected quotes (`models/auto/b5/quotes_s43_n261.json`, 756 rows / 253
matches / 8 skips, seed 43, 100 sims):
- MAE sim vs naive: cp6 20.860 vs 25.897 (dMAE −5.038 [−7.970, −2.082]);
  cp10 17.061 vs 20.000 (−2.939 [−4.947, −0.909]); cp15 12.314 vs 13.575
  (−1.261 [−2.613, −0.004]); pooled −3.086 [−4.869, −1.289].
- Coverage: cp6 0.755 [0.704, 0.810] IN; cp10 0.794 [0.743, 0.846] IN;
  cp15 0.664 [0.608, 0.724] OUT (<0.70).
- P50 bias (mean sim_p50 − actual): +4.670 / +3.204 / +0.514 at cps 6/10/15.

B14 must report the corrected-test numbers against these.

## Exact changes to make

### 1. `scripts/auto/b5_inplay_quotes.py` — additive flag only
Add `--usage-json` (default `"models/bowler_phase_usage.json"` = current
behavior) and change the selector construction at ~line 294 to
`T20Rules(EmpiricalBowlerSelector(usage_path=args.usage_json))`. Record
`"usage_json": args.usage_json` in the payload `config` dict. NOTHING else
changes in this file.

Why: B12 shipped the `b10_asof_usage` key into
`models/bowler_phase_usage.json` at 14:03 today, so the default path is now
B10-ACTIVE — but the frozen test quotes were generated at 04:26 today,
pre-B12 (legacy selector). The val fit must see the SAME quote-generating
engine as the test quotes, so the val run pins
`--usage-json models/auto/b12/bowler_phase_usage_pre_b12.json` (pre-ship
backup; orchestrator verified md5 `ea0c73d3ddb48f499b6273f9a397b0e3` and no
`b10_asof_usage` key — with the key absent the selector code path is
float-exact legacy, proven by the B10 AND B12 unit checks).

### 2. NEW `scripts/auto/b14_fit_quote_calibrator.py`
Reads a val quotes JSON (`--quotes`), writes
`models/auto/b14/quote_calibrator.json`. Per checkpoint cp ∈ {6, 10, 15}:
- `shift[cp]` = mean(sim_p50 − actual_remaining) over val rows at cp.
- Corrected quantiles for a row:
  `corrP50 = sim_p50 − shift[cp]`,
  `corrP10 = corrP50 − s·(sim_p50 − sim_p10)`,
  `corrP90 = corrP50 + s·(sim_p90 − sim_p50)`.
- `scale[cp]` = the s in grid 0.50..3.00 step 0.01 minimizing
  |val inclusive coverage(corrP10 ≤ actual ≤ corrP90) − 0.80|; ties → smaller
  s. (Pre-committed here; do not tune differently after seeing numbers.)
- Output JSON: per cp {shift, scale, n_rows, val_coverage_raw,
  val_coverage_corrected, val_mae_raw, val_mae_corrected}. Print the table.

### 3. NEW `scripts/auto/b14_gate_analysis.py`
Do NOT edit `scripts/auto/b5_gate_analysis.py` (it is the B5 record). New
file; copy its `cluster_boot_ci` verbatim (2000 reps, seed 29, percentile
95%, cluster = match_id). Inputs: `--calibrator
models/auto/b14/quote_calibrator.json`, `--quotes
models/auto/b5/quotes_s43_n261.json` (the FROZEN test quotes — never
regenerate them). Applies the correction to every test row, then prints:
- Per cp: corrected MAE vs naive MAE, paired dMAE + cluster-boot CI,
  corrected coverage + cluster-boot CI, corrected mean bias; and the
  uncorrected numbers alongside for context.
- Pooled paired |corr_err| − |naive_err| dMAE + CI (GATE 1′ stat).
- Explicit `GATE 1' MET/NOT MET` and `GATE 2' MET/NOT MET` lines with the
  exact conditions restated.

## Order of operations (STRICT)

1. Read `program.md` (repo root) and obey DO NOT CHEAT in full.
2. Implement the three changes above. Run a 5-match smoke of the val run:
   `uv run python scripts/auto/b5_inplay_quotes.py --test-dir
   data/auto/b3/val_matches --n-matches 5 --n-sims 100 --seed 47
   --usage-json models/auto/b12/bowler_phase_usage_pre_b12.json
   --out models/auto/b14/quotes_val_smoke.json`
   Verify: it completes, produces rows, and the startup output does NOT
   contain any "B10" ACTIVE banner (pinned legacy usage). Also verify the
   default-path banner behavior is untouched by your flag (do not run a
   default-path eval — just confirm the argparse default is the production
   path string).
3. **Commit BEFORE the full eval** (this is the gate pre-commit — the gate
   script must be in git before any corrected TEST number exists):
   `Auto[B14]: implement — usage-json flag + val-fit quote calibrator + pre-committed b14 gate (corrected-coverage primary, val seed 47)`
4. Full val quote run (ONE heavy process at a time, nothing else running):
   `uv run python scripts/auto/b5_inplay_quotes.py --test-dir
   data/auto/b3/val_matches --n-matches all --n-sims 100 --seed 47
   --usage-json models/auto/b12/bowler_phase_usage_pre_b12.json
   --out models/auto/b14/quotes_val_s47_n545.json 2>&1 | tee
   research/handoff/B14/raw/val_run.log`
   Expected ~55–75 min (test ran 253 matches in 1495.8 s ≈ 5.9 s/match; val
   dir has 545 files). If it exceeds 2.5 h, kill it and report the failure —
   do not restart it more than once.
5. Fit: `uv run python scripts/auto/b14_fit_quote_calibrator.py --quotes
   models/auto/b14/quotes_val_s47_n545.json 2>&1 | tee
   research/handoff/B14/raw/fit.log`
6. Gate: `uv run python scripts/auto/b14_gate_analysis.py 2>&1 | tee
   research/handoff/B14/raw/gate_output.txt`
7. Write `research/handoff/B14/result.md` (see below), commit it plus a
   small raw-excerpt file (startup banner lines, done-line, the gate
   verdict block) as `Auto[B14]: executor result — <one line>`. Full *.log
   files stay on disk (gitignored is fine — do not force-add them).

## Pre-flight verifications (do these before step 4; abort and report if any fails)

- `md5 -q models/auto/b12/bowler_phase_usage_pre_b12.json` ==
  `ea0c73d3ddb48f499b6273f9a397b0e3`, and a `json.load` of it has NO
  `b10_asof_usage` key.
- `models/auto/b5/quotes_s43_n261.json` config block reads: n_sims 100,
  seed 43, quote_center sim_p50, elapsed ≈1495.8 s; 756 rows, 253 matches,
  8 skips. Do not touch this file.
- `data/auto/b3/val_matches/` contains 545 .json files.

## Easy-to-get-wrong list

- **Do not regenerate or overwrite the test quotes** — GATE numbers come
  from applying the correction to the existing
  `models/auto/b5/quotes_s43_n261.json` rows. No new test sim.
- **Never touch**: `models/bowler_phase_usage.json` (B12 just shipped it),
  `sim_v1_2.py`, anything in `scripts/sim_eval/`, `b5_gate_analysis.py`,
  `b5_unit_check.py`, `data/golden/`, production model dirs.
- The 0.80 coverage target is the VAL fitting target; the TEST bar is
  [0.70, 0.90] at all three checkpoints. Don't conflate them.
- Band arithmetic: lower half-width is (p50 − p10), upper is (p90 − p50) —
  they are asymmetric; widen each about the CORRECTED P50.
- Coverage is inclusive (≤ / ≥), matching `b5_gate_analysis.py`.
- The naive baseline is NOT corrected — it is the fixed comparator.
- Val skips: the harness hard-asserts replay parity and raises ReplayError
  on rain-curtailed innings; ~3% skips expected (test had 8/261). If more
  than ~10% of val matches skip, stop and report instead of proceeding.
- Seed 47 for the val run is fine (no pairing against anything; B14 entry
  says a seed distinct from 43 is fine). Do not reuse 42/43/44.
- The harness writes a `.partial.jsonl` sidecar next to `--out` — expected.
- `models/auto/b14/` and the raw logs are gitignored — that is fine; commit
  code, result.md, and the raw excerpt only.

## result.md must contain (verbatim from tool output — no rounding)

- The val fit table: per cp shift, scale, n_rows, raw & corrected val
  coverage.
- Per-cp corrected test MAE vs naive + paired dMAE CIs; pooled dMAE + CI.
- Per-cp corrected test coverage + CIs (and the uncorrected values
  alongside).
- The GATE 1′ / GATE 2′ MET/NOT-MET lines exactly as the gate script printed
  them.
- Row/match/skip counts for the val run; wall-clock seconds for the val run.
- Commit SHAs you created; `git diff --stat` against claim commit `4a57f0a`.
- Anything that crashed, was retried, or ran long.

## What you must NOT do

- Do not decide the verdict (LANDED/TABLED/FAILED) — the orchestrator does.
- Do not revert anything. Do not touch `research/results.tsv` or
  `research/IDEAS.md`. Do not `git push`. Do not start any other idea.
- Do not run any other eval (no prop_backtest, no recipe A/B runs).
- Do not leave background processes running when you finish.
