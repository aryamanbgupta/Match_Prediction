# B16 — Quote-layer coverage re-check on the promoted i7 stack (orchestrator plan)

Idea id: **B16** (P3, claimed 2026-08-03T06:18Z, claim commit `445a1f3`).
Full idea text: `research/IDEAS.md` § B16 (line ~1210).

## Hypothesis

The B15 scale-only quote calibrator (record
`models/auto/b15/quote_calibrator_scale_only.json`, scales cp6/cp10/cp15 =
1.19/1.09/1.26, shifts 0) was fit against the LEGACY balanced-weights ball
engine's under-dispersion. A13's STEP 0 (2026-08-03) showed the promoted i7
no-weights stack disperses materially wider at the prop layer (first_over
P10–P90 coverage 64.6→77.2%, batter_runs 73.8→82.5%). If the in-play quote
path migrates to the promoted stack, the B15 scales would over-widen the
bands (plausibly through the 0.90 ceiling), and the RAW promoted-stack
quotes may already be in band with NO quote calibrator — mirroring D17's
null one layer up.

## Gate metric pair (sim-gate; pre-commit the script BEFORE any i7 quote number exists)

Computed on ONE fresh i7 test quote run (see recipe), all arms being
deterministic transforms of the same run's rows:

- **GATE 1 (skill retained on the i7 path):** pooled paired dMAE
  (|raw sim P50 err| − |naive err|), cluster-bootstrapped by match
  (2000 reps, seed 29 — the exact B14/B15 contract), CI hi < 0.
  This is B5's GATE 1 re-tested on the migrated stack.
- **GATE 2 (coverage in band):** at least one pre-declared arm keeps
  inclusive P10–P90 coverage (c10 ≤ actual ≤ c90) in **[0.70, 0.90] point
  estimate at ALL THREE checkpoints** (overs 6/10/15). Arm preference is
  PRE-COMMITTED (parsimony order, D17 mirror):
  1. **RAW** (no quote calibrator) — if in band, the calibrator is retired
     for the i7 quote path; do NOT prefer a scaled arm even if also in band.
  2. **B15 scales** (existing record applied unchanged, shift 0).
  3. **Refit scale-only** (fit on an i7 VAL quote run; only run this arm if
     both 1 and 2 fail GATE 2 — see conditional step 5).
- Verdict: BOTH gates met → **LANDED** (record the winning arm; if RAW wins,
  the outcome is "calibrator retired for the i7 quote path"; B15 scales
  remain the record for the legacy path regardless). Exactly one → TABLED.
  Neither → FAILED. Report per-cp coverage cluster-boot CIs as context
  (gate is on point estimates, per the B14/B15 precedent).

## Baseline rows to compare against (context, not pairing)

From `research/results.tsv` row B15 (2026-07-31) and
`research/reports/auto/B15.md` — legacy stack, fresh s45 draws, 756 rows /
253 matches / 8 skips:

- RAW legacy coverage cp6/cp10/cp15: **0.755 / 0.791 / 0.660** (cp15 OUT
  of band, CI [0.604, 0.720]).
- B15 scale-only corrected coverage: **0.818 / 0.834 / 0.768** (all IN).
- Pooled raw dMAE vs naive: **−3.131 [−4.909, −1.356]** (per-cp point
  −5.125 / −3.010 / −1.237).

There is no paired legacy-vs-i7 statistic in the gate — the engines differ
by design; the gate is self-contained on the i7 run.

## Implementation (exact changes, by file)

1. **`scripts/auto/b5_inplay_quotes.py`** (research harness — editable;
   `scripts/sim_eval/` is NOT):
   - Add opt-in stack args mirroring `scripts/sim_eval/prop_backtest.py`
     (lines ~856–932): `--model-path`, `--batter-encoder`,
     `--bowler-encoder`, `--feature-columns`, `--stats-version`,
     `--ball-calibrator {vector,none}` (+ `--ball-calibrator-path`).
   - **Defaults must reproduce current legacy behavior exactly**
     (models/xgb_v3 paths, stats version v3, vector calibrator at
     `models/xgb_v3/vector_scaling_calibrator_v1.pkl`) so a no-args
     invocation is behavior-identical to the B5/B15 runs. Do not flip any
     default.
   - `--ball-calibrator none` must pass `ball_calibrator=None` to
     `XGBoostModelV2` and record `"ball_calibrator": null` in the output
     config block; the config block must also record the actual model path
     and stats version (replace the hard-coded strings at lines ~365–366).
   - StatsProvider becomes `StatsProvider("models", version=args.stats_version)`.
   - Venue/matchup encoders are auto-discovered from the model dir by the
     wrapper (same construction as prop_backtest; D16's run printed
     "venue encoder ACTIVE (373)"). Do not add explicit encoder plumbing.
2. **`scripts/auto/b16_gate_analysis.py`** (new; adapt from
   `scripts/auto/b15_gate_analysis.py`, reusing its transform/coverage/
   bootstrap functions):
   - Inputs: `--quotes` (i7 test quotes JSON), `--b15-calibrator`
     (default `models/auto/b15/quote_calibrator_scale_only.json`),
     optional `--refit-calibrator`.
   - Emits per-cp: raw coverage (+cluster-boot CI), B15-scaled coverage
     (+CI), refit-scaled coverage if provided; raw P50 MAE vs naive MAE,
     pooled paired dMAE with 2000-rep seed-29 cluster bootstrap; and the
     pre-committed arm-preference + verdict mapping printed verbatim.
   - **MANDATORY `--self-test` mode** (B15 precedent): run the transform on
     the frozen `models/auto/b15/quotes_s45_n261.json` and hard-assert it
     reproduces the B15 LANDED numbers EXACTLY as logged in
     `research/reports/auto/B15.md`: scale-only coverage 0.818/0.834/0.768,
     raw coverage 0.755/0.791/0.660, pooled raw dMAE −3.131 [−4.909, −1.356]
     (pull the full-precision expectations from
     `research/handoff/B15/raw/` gate output if available). Self-test must
     PASS and its output be saved to `research/handoff/B16/raw/` BEFORE the
     i7 quote run starts.
3. If (and only if) step 5 fires: **`scripts/auto/b16_fit_scale_only.py`**
   (adapt `b14_fit_quote_calibrator.py`): B15 fit rule verbatim — shift := 0
   (never fit a shift; B15 diagnostic: val→test bias sign mismatch
   replicates), scale grid 0.50..3.00 step 0.01 targeting val coverage
   0.80, ties → smaller scale. Output
   `models/auto/b16/quote_calibrator_i7_scale_only.json`.

## Eval recipe (run AFTER committing implementation + gate script + self-test log)

```bash
# 1. Self-test the gate tooling on frozen B15 quotes (must PASS; ~seconds)
uv run python scripts/auto/b16_gate_analysis.py --self-test \
    2>&1 | tee research/handoff/B16/raw/self_test.log

# 2. ONE fresh i7 test quote run — promoted stack, RAW (no ball calibrator,
#    no quote calibrator), fresh seed 48, production usage json (B10 key
#    present — do NOT pin the pre-B12 json; the promoted-stack eval config
#    is B10-ACTIVE, matching the D16/D17/B13 banners):
uv run python scripts/auto/b5_inplay_quotes.py \
    --test-dir data/polymarket_test --n-matches all --n-sims 100 --seed 48 \
    --model-path models/xgb_i7_noweights_production/xgboost_model_i7.pkl \
    --batter-encoder models/xgb_i7_noweights_production/batter_encoder_i7.pkl \
    --bowler-encoder models/xgb_i7_noweights_production/bowler_encoder_i7.pkl \
    --feature-columns models/xgb_i7_noweights_production/feature_columns_i7.txt \
    --stats-version i7 --ball-calibrator none \
    --usage-json models/bowler_phase_usage.json \
    --out models/auto/b16/quotes_i7_s48_n261.json \
    2>&1 | tee research/handoff/B16/raw/quote_run_i7_s48.log

# 3. Gate analysis (raw + B15-scaled arms)
uv run python scripts/auto/b16_gate_analysis.py \
    --quotes models/auto/b16/quotes_i7_s48_n261.json \
    2>&1 | tee research/handoff/B16/raw/gate_output.txt
```

**Step 4 (conditional — ONLY if raw AND B15-scaled both fail GATE 2):**
one i7 VAL quote run on the 545-file pool `data/auto/b3/val_matches`
(seed 49, same i7 args, `--out models/auto/b16/quotes_i7_val_s49_n545.json`),
fit the refit arm with `b16_fit_scale_only.py`, re-run the gate with
`--refit-calibrator`, tee everything to `research/handoff/B16/raw/`.
Do NOT run this if either cheaper arm is already in band.

Expected timings: test quote run ~25–45 min (legacy took 1487s; i7 ball
evals ran ~0.6× legacy time); val run if needed ~35–60 min. Kill + record
CRASH at 2× (i.e. >90 min for the test run).

## Sanity checks the executor MUST perform and log

- Startup banners of the quote run: `player_stats_cache_i7.sqlite` (i7
  stats), "venue encoder ACTIVE (373 venues)", B10 selector banner
  (k_u=5.0), NO "Ball calibrator: vector" line. If any banner is wrong,
  STOP — do not burn the eval.
- Row/skip structure: expect ~756 rows / 253 matches / 8 skips (replay is
  engine-independent; B5/B15 both saw exactly 8 structural
  rain-curtailment skips). If skips differ from 8 or rows deviate >5%,
  report it prominently in result.md — do not silently proceed.
- `git diff` on `scripts/auto/b5_inplay_quotes.py` shows argparse-default
  behavior unchanged (legacy defaults intact); no file under
  `scripts/sim_eval/` touched; `models/xgb_i7_noweights_production/` and
  `models/xgb_v3/` are read-only (verify md5 of
  `models/xgb_i7_noweights_production/xgboost_model_i7.pkl` before and
  after and record both).

## Easy to get wrong (read twice)

1. **No calibrators anywhere on the i7 arms**: no `--ball-calibrator
   vector` (D17 closed the chain — the promoted stack runs RAW), and the
   raw arm has no quote-layer transform either.
2. **Usage json**: production `models/bowler_phase_usage.json` (WITH the
   b10_asof_usage key, md5 2e650423f0c949631fca1f15dd1c8a56). The B5/B15
   legacy runs pinned the PRE-B12 json — that pin was for pairing with
   frozen legacy quotes and must NOT be copied here.
3. **Arm preference is not a choice**: if RAW is in band at all three cps,
   RAW is the certified arm even if a scaled arm looks "more centered".
   Preferring a scaled arm post hoc is slice-shopping.
4. Coverage is **inclusive** (c10 ≤ actual ≤ c90) and the scaled transform
   is c10 = c50 − scale·(p50 − p10), c90 = c50 + scale·(p90 − p50) with
   shift 0 (scale-only leaves P50, hence dMAE, untouched).
5. The gate script must be committed (with the passing self-test) BEFORE
   step 2 produces any output. Commit ordering is part of the evidence.
6. Never touch `data/golden/`, never edit `scripts/sim_eval/`, never
   modify `models/auto/b15/` or `models/auto/b14/` inputs (mtime-verify
   `models/auto/b15/quotes_s45_n261.json` unchanged after the self-test).
7. One heavy process at a time; do not launch the val run concurrently
   with anything.
