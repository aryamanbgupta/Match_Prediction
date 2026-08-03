# D6 — Retrain the ball model WITHOUT balanced class weights (structural E5 alternative)

Claim commit: `a2188b0`. Orchestrator plan; executor implements, does not decide.

## Hypothesis

E5's root cause was `balanced` class weights sampled raw (P(wicket) 2.1× actual,
runs/ball +0.38 teacher-forced). Every fix so far is post-hoc marginal patching
(E5 global vector → A8 → A14/A15 → A16 → B7 → B8 — that chain is now CLOSED),
which is phase-uniform and costs `batter_runs_mae` by breaking an accidental
boundary/wicket cancellation. The never-tried structural alternative: retrain
with sample weights OFF so the booster estimates P(outcome|state) directly.
Balanced weights only help ranking, which a generative sim doesn't need.

## Gate pair (sim gate — pre-committed here, before any result)

- **GATE 1 (primary), all three conditions:**
  - (a) teacher-forced marginal audit PASSES tolerance (below), recorded BEFORE
    the sim eval;
  - (b) paired vs the canonical baseline, pooled tail dBrier over the row-pool
    {`pp_total_ou_45_5`, `pp_total_ou_50_5`, `pp_total_ou_55_5`,
    `bowler_wkts_1plus`} improves CI-clean (CI hi < 0);
  - (c) `batter_runs_mae` does NOT regress CI-clean (paired dMAE CI lo ≤ 0).
- **GATE 2 (guards):** `top_bowler` dBrier AND `team_first_over_mae` dMAE — no
  CI-clean regression on either (CI lo ≤ 0 each).
- Verdict mapping (orchestrator applies it, not you): both gates → LANDED;
  exactly one → TABLED; neither → FAILED. LANDED ships NOTHING: the model stays
  in `models/auto/d6/`, default sim path untouched, NO re-baseline; promotion is
  a human follow-up.

**Marginal-audit tolerance (from `reports/e5_class_weight_fix.md`):**
teacher-forced on the v3 test parquet, PASS iff
`|P̂(wicket) − actual|` ≤ 0.005 AND `|runs/ball − actual|` ≤ 0.05.
(Raw-balanced production is +0.064 / +0.383; the deployed vector calibrator
achieves −0.002 / +0.024 — the bar is "matches marginals about as well as the
explicit marginal-matching calibrator".) Record the full per-class table and
test multiclass LL (production context: raw 1.608, calibrated 1.520).

## Baseline (comparison target)

The claim under test is "structural retrain beats the DEPLOYED calibrated
stack". Deployed stack = production `models/xgb_v3` model + stale v1 vector
calibrator on today's default sim path (venue-ON + D1 run_rate + D15
attribution + B12-shipped B10 usage). Its canonical measurement is
**`models/auto/b12/detail_b10_s44_n261.json`** (seed 44, n=261×100, fresh
2026-07-31; report `models/auto/b12/report_b10_s44_n261.md`). Orchestrator
ruling: the idea text's "canonical venue-ON baseline" predates B10/B12 — the
B12 detail is the baseline; do not use any older detail JSON. Reference numbers
from its report: bowler_wkts_1plus Brier 0.2532, pp_total 45/50/55 =
0.2490/0.2526/0.2180, top_bowler 0.0773.

So the verdict comparison is d6-RAW (no ball calibrator) vs
baseline-WITH-v1-vector, same seed 44, same engine — the calibrator asymmetry
is deliberate and is the point of the idea.

## Step 0 — read `program.md` first, obey DO NOT CHEAT in full

Extra hard constraints for this idea:
- **`scripts/xgboost_v2.py` defaults `model_dir` to `models/xgb_v3/` (line
  ~34) — the PRODUCTION ball model dir. You MUST pass the new `--model-dir`
  override on every training invocation. Never write into `models/xgb_v3/`.**
- Never touch `data/golden/`, `scripts/sim_eval/`, `scripts/parsing_v2.py`,
  `scripts/stats_provider.py`, `scripts/stats_sqlite_backend.py`.
- One heavy process at a time (16 GB box): training and eval strictly
  sequential, never concurrent.
- All python via `uv run`.

## Step 1 — implement (commit BEFORE any eval)

1. **`scripts/xgboost_v2.py`** (trainer is editable): add two opt-in flags,
   defaults preserving current behavior byte-exactly:
   - `--model-dir <path>`: when set, `model_dir = Path(args.model_dir)`
     (else the existing `models/xgb_{data_version}`). `artifact_suffix` stays
     `v3`, so artifacts are named `*_v3.pkl` inside the override dir.
   - `--no-class-weights`: skip passing `sample_weight` at BOTH fit sites —
     final fit (~lines 395–399) and the optuna objective (~324–329; tune is
     off but gate it anyway). Leave the weight computation (~286–291) in place;
     print a loud banner `[D6] class weights DISABLED (uniform sample weights)`
     when active, and the existing behavior message when not.
2. **`scripts/auto/d6_marginal_audit.py`**: copy-adapt
   `scripts/e5_teacher_forced_bias.py` (it hardcodes `MODEL = models/xgb_v3`
   at line 40 — do NOT edit the original). Add `--model-dir`; emit a JSON
   (`models/auto/d6/marginal_audit.json`) with per-class predicted vs actual
   marginals on the v3 test parquet, Δrun/ball, ΔP(wicket), test multiclass
   LL, and an explicit `"pass": true/false` against the tolerances above.
3. **`scripts/auto/d6_gate_analysis.py`**: model on
   `scripts/auto/b12_gate_analysis.py` (same paired cluster-boot-by-match
   machinery, same resample count/seed as that tooling). Inputs: baseline
   detail (B12 path above) + d6 detail. Outputs: the GATE 1 pooled-tail stat,
   batter_runs_mae, the GATE 2 guards, the full all-family context scan, and
   the printed both/one/none mapping. It reads GATE 1(a) from
   `marginal_audit.json`. It must NOT decide the verdict.

Commit: `Auto[D6]: implement — no-class-weights trainer flag + marginal audit
+ pre-committed d6 gate (tail-pool primary, seed 44)`. This commit must exist
BEFORE any sim-eval output.

## Step 2 — train d6

Build the config-json EXACTLY as `run_experiment.py` would for
`experiments/configs/xgb_v6_hierarchical_shrink.yaml`: import
`build_training_cmd` from `scripts/run_experiment.py`, call it on the loaded
yaml config (read its signature first), and extract the string after
`--config-json` from the returned cmd list — zero hand-transcription. Then:

```
uv run python scripts/xgboost_v2.py \
    --config-json '<extracted json>' \
    --no-class-weights \
    --model-dir models/auto/d6 \
    2>&1 | tee research/handoff/D6/raw/train_d6.log
```

Expected ≤ ~60 min; kill at 120 min → CRASH facts. Then verify, and record
md5s in the log:
- `models/auto/d6/` sidecars are BYTE-IDENTICAL to `models/xgb_v3/`
  counterparts: `batter_encoder_v3.pkl`, `bowler_encoder_v3.pkl`,
  `venue_encoder_v3.pkl`, `matchup_encoder_v3.pkl`, `feature_columns_v3.txt`,
  `outcome_dist_config_v3.json` (B1 precedent says refits byte-match). If ANY
  differs → STOP, do not eval, report it as the finding (env/parquet drift).
- `models/xgb_v3/` is untouched (md5 of `xgboost_model_v3.pkl` before/after).

## Step 3 — marginal audit (BEFORE the sim eval; its PASS/FAIL is frozen here)

```
uv run python scripts/auto/d6_marginal_audit.py --model-dir models/auto/d6 \
    2>&1 | tee research/handoff/D6/raw/marginal_audit.log
```

## Step 3b — attribution control (conditional, attribution-only, NOT a gate)

Only if the d6 training took ≤ 45 min: retrain weights-ON
(same config-json, NO `--no-class-weights`) to `--model-dir
models/auto/d6/control`, tee to `raw/train_control.log`; then compare
teacher-forced `predict_proba` on the v3 test parquet vs
`models/xgb_v3/xgboost_model_v3.pkl` and record max|Δp|. Parity → the d6
delta is attributable to weights alone; non-parity or skipped → say so in
result.md (the verdict comparison survives either way under the
deployed-stack framing). Never eval the control in the sim.

## Step 4 — sim eval (verdict arm; run it regardless of audit outcome)

```
uv run python scripts/sim_eval/prop_backtest.py \
  --test-dir data/polymarket_test --n-matches all --n-sims 100 --seed 44 \
  --model-path models/auto/d6/xgboost_model_v3.pkl \
  --batter-encoder models/auto/d6/batter_encoder_v3.pkl \
  --bowler-encoder models/auto/d6/bowler_encoder_v3.pkl \
  --feature-columns models/auto/d6/feature_columns_v3.txt \
  --ball-calibrator none \
  --detail-out models/auto/d6/detail_d6raw_s44_n261.json \
  --report-out models/auto/d6/report_d6raw_s44_n261.md \
  2>&1 | tee research/handoff/D6/raw/run_d6raw.log
```

- Do NOT pass `--bowler-usage-path` — the default
  `models/bowler_phase_usage.json` is the B12-shipped B10-active one.
- Startup banner MUST show: `venue encoder ACTIVE (467 venues)`, empirical
  bowler selector with `B10 usage-aligned bowler selector ACTIVE (k_u=5.0)`,
  run-out dismissal channel active, and NO ball calibrator. If wrong, kill
  immediately and fix before burning 40+ minutes.
- ≈ 36–50 min (B12 twins: 2150 s). Launch detached (nohup + tee) then WAIT
  synchronously in-session until completion — children are reaped at session
  close; this killed 7 prior launches. Kill at 100 min → CRASH facts.
- Optional informational arm (NEVER the verdict arm): only if the audit
  FAILED tolerance AND total elapsed < 3 h — fit a fresh
  `VectorScalingCalibrator` on d6's teacher-forced val-parquet predictions
  (save `models/auto/d6/vector_scaling_calibrator_d6.pkl`), rerun the command
  above with `--ball-calibrator vector --ball-calibrator-path <that pkl>` and
  `d6raw→d6vec` in the out paths/log. Label it informational in result.md.

## Step 5 — gate analysis

```
uv run python scripts/auto/d6_gate_analysis.py \
    2>&1 | tee research/handoff/D6/raw/gate_output.txt
```

## Step 6 — result.md

Write `research/handoff/D6/result.md`: numbers VERBATIM from tool output —
marginal-audit table + deltas + pass flag, every gate stat with its CI, the
context-scan CI-clean movers, eval durations, banner lines, sidecar md5
results, control-parity max|Δp| (or why skipped); commit SHAs you created;
`git diff --stat` vs `a2188b0`; anything that crashed or ran long. Commit
everything (result commit message `Auto[D6]: executor result — <one line>`).
Return a short summary as your final message.

## What you must NOT do

No verdict. No revert. No edits to `research/results.tsv` or
`research/IDEAS.md`. No `git push`, no `git reset`. Do not start a second
idea or a second engine change. Do not modify `scripts/sim_eval/`,
`scripts/e5_teacher_forced_bias.py`, or anything under `models/xgb_v3/` or
`data/golden/`. Do not summarize raw eval output away — tee everything under
`research/handoff/D6/raw/`. Do not leave background processes running.

## Easy to get wrong

- The trainer writes to `models/xgb_v3/` unless `--model-dir` is passed.
- Line numbers in the idea text (:235/:273-278/:343-347) are STALE — current
  sites are ~286–291 (compute), ~324–329 (optuna fit), ~395–399 (final fit).
- Recipe B is `prop_backtest.py`, NOT `run_sim_eval.py`.
- Seed 44 (pairs with the canonical B12 detail), `--n-matches all`, 100 sims.
- d6 arm runs with `--ball-calibrator none`; the baseline ran with the v1
  vector — deliberate asymmetry (structural fix vs deployed patched stack).
- The audit's PASS/FAIL is decided and logged BEFORE the sim eval starts.
- Early stopping will pick a different tree count with weights off — that is
  part of the structural change, not a bug.
