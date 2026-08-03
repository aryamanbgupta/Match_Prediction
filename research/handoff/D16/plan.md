# D16 — No-class-weights ball retrain on the i7 frame, paired twin design (D6 redesign)

Claim commit: `ba9cd3f`. Orchestrator plan; executor implements, does not decide.

## Hypothesis

D6's structural question is still untested: `balanced` class weights corrupt the
sim's conditional distributions (E5: P(wicket) ~2× actual teacher-forced,
runs/ball +0.38); every fix so far is post-hoc marginal patching (E5 vector →
A8 → A14/A15 → A16 → B7 → B8 — chain CLOSED), which is phase-uniform and costs
`batter_runs_mae`. Retraining with sample weights OFF lets the booster estimate
P(outcome|state) directly — balanced weights only help ranking, which a
generative sim doesn't need. D6 CRASHED because the legacy `data/xgb_data_v3`
frame fail-closes under the I7 venue-alias contract. The i7 ball frame is
trainable, so the test runs there as a PAIRED TWIN with zero frame confound:
control (weights ON) + fresh val-fit vector calibrator vs no-weights RAW — the
deployed-stack design transplanted to i7; the calibrator asymmetry is the point.

## Feasibility (ALREADY VERIFIED by orchestrator — do not re-derive, do not skip the banner checks below)

- `data/xgb_data_i7/.feature_hash` carries `venue_alias_version: venue_aliases_v1`
  → the trainer's `assert_venue_alias_contract` PASSES on this frame (the D6
  fail-close does not apply).
- `models/xgb_i7/xgboost_model_i7.pkl` loads through `XGBoostModelV2` with
  banner `venue encoder ACTIVE (373 venues)`; all sidecars auto-detect via the
  `i7` suffix convention (`_artifact_suffix = stem.removeprefix('xgboost_model_')`).
- `prop_backtest.py --stats-version i7` → `StatsProvider("models", version="i7")`
  opens `models/player_stats_cache_i7.sqlite` (verified).
- `TestMatchLoader` canonicalizes venue at state build (`sim_eval/loaders.py:100`)
  → sim-time venue strings match the i7 encoder classes and i7 cache keys.
- Recipe B needs no tracker snapshot; the i7 as-of cache covers the test window
  (it materialized `cricket_data_i7_test.parquet`, 186,667 rows).
- `experiments/configs/xgb_i7_venue_identity.yaml` is the exact recipe that
  built `models/xgb_i7` (data version i7, 114 features, n_estimators 444,
  max_depth 10, lr 0.2404, subsample 0.8777, colsample 0.7424, k_player 30,
  k_venue 200).

## Gate pair (sim gate — pre-committed here, before any result)

- **GATE 1 (primary), all three conditions:**
  - (a) the NO-WEIGHTS arm's teacher-forced marginal audit PASSES tolerance on
    its own served input distribution (the i7 test parquet with training-time
    encoder codes): `|P̂(wicket) − actual| ≤ 0.005` AND
    `|runs/ball − actual| ≤ 0.05`. Recorded and frozen BEFORE any sim eval.
  - (b) paired vs the CONTROL+VECTOR twin (same seed, same engine, only delta =
    weights∘calibrator stack), pooled tail dBrier over the row-pool
    {`pp_total_ou_45_5`, `pp_total_ou_50_5`, `pp_total_ou_55_5`,
    `bowler_wkts_1plus`} improves CI-clean (CI hi < 0);
  - (c) `batter_runs_mae` does NOT regress CI-clean (paired dMAE CI lo ≤ 0).
- **GATE 2 (guards):** `top_bowler` dBrier AND `team_first_over_mae` dMAE — no
  CI-clean regression on either (CI lo ≤ 0 each).
- Verdict mapping (orchestrator applies it, not you): both gates → LANDED;
  exactly one → TABLED; neither → FAILED. LANDED ships NOTHING: artifacts stay
  in `models/auto/d16/`, default sim path untouched, NO re-baseline of the
  legacy canonical detail (`models/auto/b12/detail_b10_s44_n261.json` remains
  canonical for legacy-path ideas); i7 serving is gated on the human I17
  promotion bundle.

## Baseline (comparison target)

The verdict comparator is the SAME-SESSION control+vector twin — there is no
pre-existing i7 sim detail anywhere, and cross-frame comparison against the
legacy B12 detail is confounded (different model, encoder, stats identity) —
use it as context only, never for the verdict. Legacy context numbers (B12
report, seed 44): bowler_wkts_1plus Brier 0.2532, pp_total 45/50/55 =
0.2490/0.2526/0.2180, top_bowler 0.0773. E5 legacy marginal context: raw
+0.0647 ΔP(wkt) / +0.3829 Δruns-per-ball; v1-calibrated −0.0016/+0.0237 on its
venue_zero fit path (D6 side finding: +0.00576/+0.0814 on the served venue-ON
path).

## Step 0 — read `program.md` first, obey DO NOT CHEAT in full

Extra hard constraints for this idea:
- **The trainer defaults `model_dir` to `models/xgb_<data_version>` — for
  data_version i7 that is `models/xgb_i7/`, an ARCHIVED artifact this idea's
  determinism control compares against. You MUST pass `--model-dir` on EVERY
  training invocation. Never write into `models/xgb_i7/` or `models/xgb_v3/`.**
  Record md5 of `models/xgb_i7/xgboost_model_i7.pkl` and
  `models/xgb_v3/xgboost_model_v3.pkl` BEFORE step 2 and again at the end
  (`research/handoff/D16/raw/md5_protected_{before,after}.txt`).
- Never touch `data/golden/`, `data/forward_holdout/`, `scripts/sim_eval/`,
  `scripts/parsing_v2.py`, `scripts/stats_provider.py`,
  `scripts/stats_sqlite_backend.py`, `data/xgb_data_i7/` (read-only input).
- Never bypass, weaken, or edit any identity fail-close
  (`assert_venue_alias_contract` etc.). If one fires, STOP and report — that is
  a result, not an obstacle.
- One heavy process at a time (16 GB box): trainings and evals strictly
  sequential, never concurrent.
- All python via `uv run`.

## Step 1 — implement (single commit BEFORE any training/eval output)

1. **Recover the D6 trainer flags verbatim**:
   `git checkout 5fb16bb -- scripts/xgboost_v2.py`. Nothing else has touched
   this file since the D6 revert, so afterwards
   `git diff HEAD~0 --stat` must show ONLY the two-flag delta
   (`--model-dir`, `--no-class-weights` + the two `_weight_kwargs` fit-site
   changes and banner prints). Verify with
   `git diff 2a815e2..WORKTREE -- scripts/xgboost_v2.py` that the delta equals
   the 5fb16bb patch exactly; abort if anything else appears.
2. **`scripts/auto/d16_marginal_audit.py`**: copy-adapt from
   `git show 5fb16bb:scripts/auto/d6_marginal_audit.py`. Changes: `BALL_DIR` →
   `data/xgb_data_i7`; `--suffix` default `i7`; the reference/context arm
   (`PROD_DIR`) → `models/xgb_i7` (its raw marginals = the balanced-weights
   context on i7); keep the JSON output shape (per-class predicted vs actual
   marginals, Δruns/ball, ΔP(wicket), test multiclass LL, explicit
   `"pass": true/false` against the tolerances). Output paths under
   `models/auto/d16/`.
3. **`scripts/auto/d16_fit_vector_calibrator.py`** (or use
   `scripts/fit_i5_ball_calibrator.py` UNMODIFIED if it fits i7 cleanly —
   read it first; it already takes `--version i7 --data-dir data/xgb_data_i7
   --model-dir <arm> --out <pkl>`): fit `VectorScalingCalibrator` on the
   CONTROL arm's teacher-forced val-parquet predictions
   (`cricket_data_i7_validation.parquet`, 124,292 rows). The fit must use
   training-time encoder codes INCLUDING the i7 venue encoder (the served
   distribution — check `_apply_encoders_to_df` handles the venue column; if
   it does not, copy-adapt into `scripts/auto/` — do NOT edit
   `scripts/fit_i5_ball_calibrator.py` or `scripts/calibration.py`). Print and
   log the fitted 6-vector and which encoding path was used. Save to
   `models/auto/d16/vector_scaling_calibrator_d16.pkl`.
4. **`scripts/auto/d16_gate_analysis.py`**: copy-adapt from
   `git show 5fb16bb:scripts/auto/d6_gate_analysis.py` (same paired
   cluster-boot-by-match machinery, same resample count/seed as the b12
   tooling). Inputs: baseline = `models/auto/d16/detail_control_vec_s46_n261.json`,
   candidate = `models/auto/d16/detail_noweights_raw_s46_n261.json`. Outputs:
   GATE 1(b) pooled-tail stat, GATE 1(c) batter_runs_mae, GATE 2 guards, full
   all-family context scan, printed both/one/none mapping; reads GATE 1(a)
   from the no-weights `marginal_audit` JSON. It must NOT decide the verdict.

Commit everything as one commit: `Auto[D16]: implement — recover D6 trainer
flags + i7 marginal audit + control-val vector calibrator fit + pre-committed
d16 gate (twin design, seed 46)`. This commit must exist BEFORE any training
or eval output.

## Step 2 — twin trainings (sequential; control first)

Build the config-json EXACTLY as `run_experiment.py` would for
`experiments/configs/xgb_i7_venue_identity.yaml`: import `build_training_cmd`
from `scripts/run_experiment.py`, call it on the loaded yaml config (read its
signature first), and extract the string after `--config-json` from the
returned cmd list — zero hand-transcription. Then (`mkdir -p` the arm dirs
first):

```
uv run python scripts/xgboost_v2.py \
    --config-json '<extracted json>' \
    --model-dir models/auto/d16/control \
    2>&1 | tee research/handoff/D16/raw/train_control.log

uv run python scripts/xgboost_v2.py \
    --config-json '<extracted json>' \
    --no-class-weights \
    --model-dir models/auto/d16/noweights \
    2>&1 | tee research/handoff/D16/raw/train_noweights.log
```

Expected ≤ ~60 min each; kill at 120 min → CRASH facts. The no-weights run
must print `[D6] class weights DISABLED (uniform sample weights)`; the control
must print the ACTIVE message. After both, record in
`research/handoff/D16/raw/artifact_checks.txt`:
- **HARD CHECK (STOP on failure):** the two arms' sidecars are byte-identical
  to EACH OTHER (`batter_encoder_i7.pkl`, `bowler_encoder_i7.pkl`,
  `venue_encoder_i7.pkl`, `matchup_encoder_i7.pkl`, `feature_columns_i7.txt`,
  `outcome_dist_config_i7.json`) — same parquet in, same encoders out; a
  mismatch means the only-delta-is-weights contract is broken (env drift) →
  STOP, do not eval, report as the finding.
- **Determinism control (informational, NEVER a stop):** md5 of
  `models/auto/d16/control/xgboost_model_i7.pkl` and every control sidecar vs
  `models/xgb_i7/` counterparts. Byte-match → free determinism control (I19
  precedent); mismatch → record max teacher-forced |Δp| on 5k test-parquet
  rows vs `models/xgb_i7` and note it in result.md. Either way the twin
  pairing stands (both arms same-session).
- `models/xgb_i7/` and `models/xgb_v3/` untouched (md5 before/after).

## Step 3 — fit the d16 vector calibrator on the CONTROL arm

```
<the step-1.3 script> --version i7 --data-dir data/xgb_data_i7 \
    --model-dir models/auto/d16/control \
    --out models/auto/d16/vector_scaling_calibrator_d16.pkl \
    2>&1 | tee research/handoff/D16/raw/fit_calibrator.log
```

Log the fitted 6-vector. Sanity: applied to the control's val predictions the
corrected marginals must match val class frequencies (that is what the fit
does — print the residuals).

## Step 4 — marginal audits (BEFORE the sim evals; PASS/FAIL frozen here)

```
uv run python scripts/auto/d16_marginal_audit.py \
    --model-dir models/auto/d16/noweights --suffix i7 \
    2>&1 | tee research/handoff/D16/raw/marginal_audit_noweights.log
uv run python scripts/auto/d16_marginal_audit.py \
    --model-dir models/auto/d16/control --suffix i7 \
    2>&1 | tee research/handoff/D16/raw/marginal_audit_control.log
```

The no-weights audit is GATE 1(a) — its PASS/FAIL is decided and logged NOW.
The control audit is context (the E5 balanced-weights table reproduced on i7);
if the audit script supports a calibrator arm, also record control+d16-vector
marginals (context for how much of the tilt the fresh vector removes).

## Step 5 — twin sim evals (sequential, seed 46, identical settings, only the model/calibrator stack differs)

Arm C (control + fresh vector — the deployed-stack design on i7):

```
uv run python scripts/sim_eval/prop_backtest.py \
  --test-dir data/polymarket_test --n-matches all --n-sims 100 --seed 46 \
  --stats-version i7 \
  --model-path models/auto/d16/control/xgboost_model_i7.pkl \
  --batter-encoder models/auto/d16/control/batter_encoder_i7.pkl \
  --bowler-encoder models/auto/d16/control/bowler_encoder_i7.pkl \
  --feature-columns models/auto/d16/control/feature_columns_i7.txt \
  --ball-calibrator vector \
  --ball-calibrator-path models/auto/d16/vector_scaling_calibrator_d16.pkl \
  --detail-out models/auto/d16/detail_control_vec_s46_n261.json \
  --report-out models/auto/d16/report_control_vec_s46_n261.md \
  2>&1 | tee research/handoff/D16/raw/run_control_vec.log
```

Arm N (no-weights RAW — the structural alternative):

```
uv run python scripts/sim_eval/prop_backtest.py \
  --test-dir data/polymarket_test --n-matches all --n-sims 100 --seed 46 \
  --stats-version i7 \
  --model-path models/auto/d16/noweights/xgboost_model_i7.pkl \
  --batter-encoder models/auto/d16/noweights/batter_encoder_i7.pkl \
  --bowler-encoder models/auto/d16/noweights/bowler_encoder_i7.pkl \
  --feature-columns models/auto/d16/noweights/feature_columns_i7.txt \
  --ball-calibrator none \
  --detail-out models/auto/d16/detail_noweights_raw_s46_n261.json \
  --report-out models/auto/d16/report_noweights_raw_s46_n261.md \
  2>&1 | tee research/handoff/D16/raw/run_noweights_raw.log
```

- Do NOT pass `--bowler-usage-path` — the default json is the B12-shipped
  B10-active one and is identity-agnostic; BOTH arms must show the same
  selector banner.
- Startup banners MUST show, on BOTH arms:
  `StatsProvider: using SQLite backend player_stats_cache_i7.sqlite`,
  `venue encoder ACTIVE (373 venues)` (467 means you loaded the WRONG model —
  kill immediately), `B10 usage-aligned bowler selector ACTIVE (k_u=5.0)`;
  Arm C additionally `Ball calibrator: vector scaling (models/auto/d16/...)`,
  Arm N no calibrator line. If any banner is wrong, kill before burning 40 min.
- ≈ 36–50 min each (B12 twins: ~2150 s; i7 cache may differ slightly). Launch
  detached (`nohup ... &` with tee) then WAIT synchronously in-session (poll
  the log + PID in a loop) until completion — children are reaped at session
  close; this killed 7 prior launches. Kill at 100 min each → CRASH facts.
- 261/261 matches must appear in each detail JSON (a handful of load-skips
  would print SKIP — if any skip fires, record which and how many; >0 skips is
  a red flag on the i7 path, include it prominently in result.md).

## Step 6 — gate analysis

```
uv run python scripts/auto/d16_gate_analysis.py \
    2>&1 | tee research/handoff/D16/raw/gate_output.txt
```

## Step 7 — result.md

Write `research/handoff/D16/result.md`: numbers VERBATIM from tool output —
both marginal-audit tables + deltas + the frozen no-weights pass flag, the
fitted calibrator 6-vector, every gate stat with its CI, the context-scan
CI-clean movers, eval durations + skip counts, all banner lines, the
sidecar/determinism check results, protected-artifact md5 before/after;
commit SHAs you created; `git diff --stat` vs `ba9cd3f`; anything that crashed
or ran long. Commit everything (message
`Auto[D16]: executor result — <one line>`). Note that `models/auto/` and
`*.json` are gitignored — commit logs/text evidence under
`research/handoff/D16/` (use `.txt` copies for any JSON you must preserve, the
B12/D6 precedent). Return a short summary as your final message.

## What you must NOT do

No verdict. No revert. No edits to `research/results.tsv` or
`research/IDEAS.md`. No `git push`, no `git reset`. Do not start a second idea
or touch the legacy (`v3`) sim path, `models/xgb_v3/`, `models/xgb_i7/`,
`models/bowler_phase_usage.json`, or anything under `data/golden/`. Do not
edit `scripts/sim_eval/`, `scripts/calibration.py`,
`scripts/fit_i5_ball_calibrator.py` (copy-adapt into `scripts/auto/` instead),
or `scripts/e5_teacher_forced_bias.py`. Do not summarize raw eval output away —
tee everything under `research/handoff/D16/raw/`. Do not leave background
processes running when you exit.

## Easy to get wrong

- **Artifact suffix on i7 is `i7`** — every file is `*_i7.pkl` /
  `feature_columns_i7.txt`, not `*_v3.*`.
- **`--model-dir` is MANDATORY on both trainings** — the default for
  data_version i7 is `models/xgb_i7/` (a protected archived artifact).
- **`--stats-version i7` is MANDATORY on both evals** — the default is `v3`
  (wrong cache = silent identity mismatch; check the StatsProvider banner).
- The config source is `experiments/configs/xgb_i7_venue_identity.yaml`, NOT
  `xgb_v6_hierarchical_shrink.yaml` (data_version v3 → the D6 fail-close).
- Seed 46 on BOTH evals (fresh — 42/43/44/45/47 are taken by prior
  runs; no pre-existing detail pairs with these arms, pairing is twin-internal).
- Arm N runs `--ball-calibrator none`; Arm C runs the FRESH d16 vector (never
  the legacy `models/xgb_v3/vector_scaling_calibrator_v1.pkl` — that was fit
  on the legacy model's venue_zero distribution and is meaningless here).
- The no-weights audit PASS/FAIL is frozen BEFORE any sim eval starts.
- Early stopping will pick a different effective tree count with weights off —
  that is part of the structural change, not a bug.
- D6-recovered scripts hardcode legacy paths (`BALL_DIR = data/xgb_data_v3`,
  `PROD_DIR = models/xgb_v3`) — every one must be re-pointed at i7 in the
  copy-adapt; grep for `_v3` before running.
- Trainings may not `mkdir` the override dir — `mkdir -p` both arm dirs first.
