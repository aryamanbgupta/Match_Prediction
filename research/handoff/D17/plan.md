# D17 — Val-fit vector calibrator on the no-weights arm (orchestrator plan)

Claim commit: `2a9389b` (`Auto[D17]: claim`, status RUNNING 2026-08-01T01:08:57Z).

## Idea

**D17 [P2]** (D16 follow-up). D16 LANDED: the no-class-weights i7 ball retrain
(`models/auto/d16/noweights`) passes the teacher-forced marginal audit RAW
(ΔP(wkt) +0.00319 tol 0.005, Δruns/ball +0.01093 tol 0.05, 186,667 test balls)
and beats the deployed-stack twin (control + fresh vector) CI-clean on the
pooled tail and batter_runs_mae. But the raw residuals are nonzero. D17 asks
the one remaining cheap marginal question: does a `VectorScalingCalibrator`
fit on the no-weights arm's OWN val predictions buy anything further on top of
raw? A null here is decision-grade: it closes the marginal-calibration chain
(E5 → A8 → A14/A15 → A16 → B7 → B8) for the structural arm and certifies RAW
as the final i7 ball stack for the I17 bundle.

**Gate metric pair (sim gate, stated per-idea):**

- **GATE 1 (improvement):** pooled tail dBrier over the row pool
  {`pp_total_ou_45_5`, `pp_total_ou_50_5`, `pp_total_ou_55_5`,
  `bowler_wkts_1plus`} — delta = (noweights+d17vec) − (noweights raw) — is
  **CI-clean negative**, AND `batter_runs_mae` delta is **not CI-clean
  positive** (no regression; this is the exact trade the E5-era calibrator
  historically lost).
- **GATE 2 (guards):** `top_bowler` dBrier and `team_first_over_mae` dMAE —
  no CI-clean positive delta.

Pre-committed verdict mapping (orchestrator applies it, you don't):
GATE1 fully met + GATE2 held → LANDED. GATE1's pooled-tail conjunct met but
batter_runs_mae or a guard regresses CI-clean → TABLED. Pooled tail NOT
CI-clean negative → FAILED regardless of guards (null = chain closed —
per the idea text this negative is decision-grade, not a disappointment).

## Baseline for comparison

`models/auto/d16/detail_noweights_raw_s46_n261.json` — the EXISTING D16 Arm N
detail (seed 46, n=261×100, 261/261 matches, 0 skips; verified present,
11,519,598 bytes, mtime Jul 31 20:59). Same seed + same engine + only delta =
the calibrator → clean pairing (B7 precedent). Cluster bootstrap by match,
n_boot=2000, seed=29, exactly as `d16_gate_analysis.py` does.

Context anchors from the D16 gate output (do not re-derive): noweights-raw
pooled tail Brier 0.2372 (vs control_vec 0.2488, dBrier −0.0116
[−0.0159,−0.0073]); batter_runs_mae 13.8905 (vs 14.4354, −0.5449
[−0.6714,−0.4142]).

## What to implement (exact)

Nothing in the sim engine. Two artifacts + one new script:

1. **Fit the calibrator** with the EXISTING script (do not edit it):

```
mkdir -p models/auto/d17 research/handoff/D17/raw
uv run python scripts/auto/d16_fit_vector_calibrator.py \
    --version i7 --data-dir data/xgb_data_i7 \
    --model-dir models/auto/d16/noweights \
    --out models/auto/d17/vector_scaling_calibrator_d17.pkl \
    2>&1 | tee research/handoff/D17/raw/fit_calibrator.log
```

The script is arm-agnostic (all paths derive from `--model-dir`; its docstring
says CONTROL only because that's what D16 used it for). It prints the fitted
6-vector (`weights (fitted 6-vector)`), val raw/calibrated LL, and the
residual table. **Do NOT reuse `models/auto/d16/vector_scaling_calibrator_d16.pkl`**
— that one is fit on the CONTROL arm and is the wrong object for this idea.

2. **Pre-run expectation check (REQUIRED, before the eval):** from the fit
log, compute and record max |v−1| over the fitted 6-vector in
`research/handoff/D17/raw/expectation_check.txt`. If it is below the ~0.05
washout threshold (A8/A12/B7 precedent), write "EXPECT NULL — divergence
below washout threshold" and run the eval anyway (the null still closes the
chain). If ≥0.05, write "live test". This must exist before the sim starts.

3. **`scripts/auto/d17_gate_analysis.py`** — copy-adapt
`scripts/auto/d16_gate_analysis.py`. Changes:
   - baseline default = `models/auto/d16/detail_noweights_raw_s46_n261.json`
     (label `noweights_raw`), challenger default =
     `models/auto/d17/detail_noweights_vec_s46_n261.json` (label
     `noweights_vec`); delta = vec − raw, negative = vec better.
   - DROP the D16 GATE 1(a) marginal-audit-file section (that audit is D16's;
     D17's pre-run check is item 2 above).
   - GATE 1 = (i) pooled tail over the SAME row pool
     {pp_total_ou_45_5, pp_total_ou_50_5, pp_total_ou_55_5, bowler_wkts_1plus}
     CI-clean negative, and (ii) batter_runs_mae not CI-clean positive.
   - GATE 2 = guards top_bowler + team_first_over_mae not CI-clean positive.
   - Keep the identical bootstrap (cluster by match, n_boot=2000, seed=29),
     the identical row-pairing machinery, and the full 33-family context scan.
   - Print the verdict-mapping block from this plan verbatim at the end.

**Commit the gate script + this plan's artifacts BEFORE launching the eval**
(`Auto[D17]: implement — <what>`). The gate script must be committed before
any eval output exists (loop discipline; D16/B12 precedent).

## Eval recipe (ONE run — recipe B, Arm N settings + the d17 calibrator)

```
uv run python scripts/sim_eval/prop_backtest.py \
  --test-dir data/polymarket_test --n-matches all --n-sims 100 --seed 46 \
  --stats-version i7 \
  --model-path models/auto/d16/noweights/xgboost_model_i7.pkl \
  --batter-encoder models/auto/d16/noweights/batter_encoder_i7.pkl \
  --bowler-encoder models/auto/d16/noweights/bowler_encoder_i7.pkl \
  --feature-columns models/auto/d16/noweights/feature_columns_i7.txt \
  --ball-calibrator vector \
  --ball-calibrator-path models/auto/d17/vector_scaling_calibrator_d17.pkl \
  --detail-out models/auto/d17/detail_noweights_vec_s46_n261.json \
  --report-out models/auto/d17/report_noweights_vec_s46_n261.md \
  2>&1 | tee research/handoff/D17/raw/run_noweights_vec.log
```

- Do NOT pass `--bowler-usage-path` (default = the B12-shipped B10-active
  sidecar; identical to both D16 arms).
- Startup banners MUST show:
  `Ball calibrator: vector scaling (models/auto/d17/vector_scaling_calibrator_d17.pkl)`,
  `StatsProvider: using SQLite backend player_stats_cache_i7.sqlite`,
  `venue encoder ACTIVE (373 venues)` (467 = WRONG model, kill immediately),
  `B10 usage-aligned bowler selector ACTIVE (k_u=5.0)`. If any banner is
  wrong, kill before burning the run.
- Expected duration ≈ 22–40 min (D16 Arm N raw took 1297.1 s; the calibrator
  adds per-ball overhead — control+vec took 2251.8 s on the bigger booster).
  Launch detached (`nohup ... &` with tee) and WAIT SYNCHRONOUSLY in-session,
  polling the log + PID until the process exits — children are reaped at
  session close; this has killed 7+ prior launches. Kill at 100 min → record
  CRASH facts and stop.
- Expect `[261/261]` and 0 skips (D16 had zero on this exact path); any skip
  is a red flag — count it and report prominently.

Then:

```
uv run python scripts/auto/d17_gate_analysis.py \
    2>&1 | tee research/handoff/D17/raw/gate_output.txt
```

## Integrity checks (do these, record in result.md)

- BEFORE the eval: `md5 models/auto/d16/detail_noweights_raw_s46_n261.json`
  and md5 of `models/auto/d16/noweights/xgboost_model_i7.pkl`; AFTER the gate:
  re-run both — they must be unchanged (the baseline must stay byte-frozen).
- md5 of protected `models/xgb_i7/xgboost_model_i7.pkl` and
  `models/xgb_v3/xgboost_model_v3.pkl` before/after (D16 precedent — prove
  nothing production was touched).
- `git diff --stat` vs `2a9389b` in result.md.

## Easy to get wrong

- **Seed 46 is mandatory** — the pairing against the existing detail is the
  design. A different seed voids the comparison.
- Everything under `models/auto/` and `*.json` is **gitignored** — evidence
  that must survive goes under `research/handoff/D17/` as logs / `.txt`
  copies (B12/D6 precedent). Copy the fit log, expectation check, gate
  output, and a `.txt` copy of the calibrator's printed vector + marginal
  table.
- The fit script raises on encoder cross-check mismatch — that's a real
  failure, not something to patch around. Record it verbatim if it fires.
- Never write into `models/auto/d16/` (baseline lineage), `models/xgb_i7/`,
  `models/xgb_v3/`, `data/xgb_data_i7/`, or any production/legacy path.
  Never touch `data/golden/`. Never edit `scripts/sim_eval/` or
  `scripts/auto/d16_*.py`.
- Ships NOTHING regardless of outcome: artifacts stay in `models/auto/d17/`;
  the default (legacy v3) sim path and the canonical
  `models/auto/b12/detail_b10_s44_n261.json` baseline are untouched. i7
  serving remains gated on the human I17 promotion bundle.

## Executor deliverables

1. Read `program.md` first; obey DO NOT CHEAT in full.
2. Implement as above; commit BEFORE the eval (`Auto[D17]: implement — ...`).
3. Run the eval to completion; tee everything to `research/handoff/D17/raw/`.
4. Write `research/handoff/D17/result.md`: fitted 6-vector + max|v−1| +
   expectation-check outcome, val raw/calibrated LL, every gate stat with CI
   verbatim, the CI-clean movers from the context scan, eval duration + skip
   count, all banner lines, md5 before/after table, commit SHAs,
   `git diff --stat` vs `2a9389b`, anything that crashed or ran long. Commit
   (`Auto[D17]: executor result — <one line>`).
5. Final message = short summary.

You do NOT: decide the verdict, revert anything, touch
`research/results.tsv` or `research/IDEAS.md`, `git push`, `git reset`, or
start a second idea.
