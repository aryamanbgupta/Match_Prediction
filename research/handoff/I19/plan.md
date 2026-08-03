# I19 — Coherent-contract i7 frame re-key (executor plan)

Idea: **I19 [P2]** from `research/IDEAS.md` (line ~2030). Claim commit: `fc831c9`.

## Hypothesis

The successor line's training frame `data/xgb_match_data_i7` predates I15/I16
(its `match_id` is the legacy display string; no `display_match_id` /
`match_identity_version` / `elo_update_version` columns). I18 proved the 46
features and all shared metadata are bit-identical across that schema drift, so
re-keying the frame to the current cricsheet-primary contract is pure identity
hygiene. Producing a coherent-contract twin (`data/xgb_match_data_i7_v2`) and
proving retrains on it reproduce the I17 arms exactly removes the
mixed-contract hazard from the human promotion bundle.

## Gate (correctness/instrumentation — no LL/ROI eval, no market slicing)

LANDED iff ALL THREE hold; fail closed otherwise:
1. `scripts/auto/i18_frame_parity.py` **relaxed** gate passes for the v2 frame
   vs the frozen reference `data/xgb_match_data_i7` (strict pass 1 is
   informational and EXPECTED TO FAIL — that's the whole schema-drift point).
2. Both retrained seed-29 arms reproduce the archived I17 predictions at
   **max |Δp| = 0** (exactly zero, not "small").
3. The v2 frame carries the full identity contract (checks specified below).

You do NOT issue the verdict — you report numbers; the orchestrator judges.

## Decision already made — do not revisit

- **Reuse path, not regeneration.** `data/auto/i18/frame/` is still on disk
  and is I18's parity-verified 57-column production-subset frame (all four
  splits + the three identity sidecars). The v2 frame is created by **copying
  files** from it. Do NOT run `materialize_match_features.py` and do NOT run
  `build_i7_match_frame.py` — regeneration would read `data/golden/t20s_json`,
  which is forbidden to this loop (program.md rule 2; I18 had a one-off
  supervisor exception, you do not).
- **CRITICAL:** `scripts/build_i7_match_frame.py` has
  `--out-dir data/xgb_match_data_i7` (the frozen frame of record) as its
  DEFAULT. This is one reason you must not run it at all.

## Steps

### 0. Setup
Read `program.md` at the repo root first; obey the DO NOT CHEAT section in
full. Confirm `git status` is clean apart from your own work. All raw tool
output gets teed to `research/handoff/I19/raw/` (dir exists).

### 1. Build the v2 frame (file copy only)
```bash
mkdir -p data/xgb_match_data_i7_v2
cp data/auto/i18/frame/train.parquet \
   data/auto/i18/frame/validation.parquet \
   data/auto/i18/frame/test.parquet \
   data/auto/i18/frame/golden_test.parquet \
   data/auto/i18/frame/venue_identity.json \
   data/auto/i18/frame/match_identity.json \
   data/auto/i18/frame/elo_update.json \
   data/xgb_match_data_i7_v2/
```
Byte-copy only (`cp`). Never round-trip a parquet through pandas to "copy" it
— column order and encodings are load-bearing (colsample is order-sensitive).
Do NOT delete, rename, or write anything inside `data/xgb_match_data_i7/` or
`data/auto/i18/`.

### 2. Write the check harness, then COMMIT before running anything
Create `scripts/auto/i19_contract_check.py` and
`scripts/auto/i19_repro_check.py` (specs below), then commit them together
with any scratch notes as:
`Auto[I19]: implement — coherent-contract i7_v2 frame (I18-frame reuse) + check harness`
(The frame itself is gitignored data; the commit is the harness.)

**`i19_contract_check.py`** — for each of the four v2 split parquets assert
and PRINT (verbatim counts):
- `match_id == cricsheet_id` on 100% of rows (cricsheet-primary re-key);
- `display_match_id` present, non-null, and unique per row;
- `match_identity_version` column constant `cricsheet_primary_v1`;
- `elo_update_version` column constant `fixed_competition_k_v1`;
- row counts: train 7972 / validation 528 / test 798 / golden_test 227;
- `match_id` unique per row in every split.
Then the sidecars:
- `venue_identity.json` byte-equal to `data/xgb_match_data_i7/venue_identity.json`;
- `match_identity.json` == `scripts/match_identity.identity_contract()`
  (import it; `build_i7_match_frame.py` shows the import pattern);
- `elo_update.json` == `{"elo_update_version": "fixed_competition_k_v1"}`.

**`i19_repro_check.py`** — for each arm in {base, swap}: load
`models/auto/i19/<arm>_seed29/test_predictions.json` and
`models/auto/i17/<arm>_seed29/test_predictions.json` (both are dicts keyed by
cricsheet-primary match_id, 798 entries). Assert identical key SETS, then
print `n_keys` and `max |Δ p_team1|` in `%.3e` format — the gate needs exactly
`0.000e+00`. Also assert byte-equality of the two `feature_columns.txt` files
(48 lines) and print the val/test log-loss pairs from both
`train_metrics.json` files side by side.

### 3. Run parity + contract checks
```bash
uv run python scripts/auto/i18_frame_parity.py \
    --candidate-dir data/xgb_match_data_i7_v2 \
    --reference-dir data/xgb_match_data_i7 \
    2>&1 | tee research/handoff/I19/raw/parity.log

uv run python scripts/auto/i19_contract_check.py \
    2>&1 | tee research/handoff/I19/raw/contract.log
```
Expected: PASS 1 (strict) FAILS on the identity columns — that is the
documented expectation, not a problem; PASS 2 line
`I18 RELAXED PARITY GATE: PASS` is what the gate needs. If relaxed parity
FAILS, stop after recording the output — do not "fix" the frame.

### 4. Retrain both seed-29 arms on the v2 frame
```bash
uv run python scripts/xgboost_match_v1.py --cmd both \
    --data-dir data/xgb_match_data_i7_v2 \
    --model-dir models/auto/i19/base_seed29 \
    --monotone --seed 29 \
    2>&1 | tee research/handoff/I19/raw/train_base_seed29.log

uv run python scripts/xgboost_match_v1.py --cmd both \
    --data-dir data/xgb_match_data_i7_v2 \
    --model-dir models/auto/i19/swap_seed29 \
    --monotone --swap-augment --seed 29 \
    2>&1 | tee research/handoff/I19/raw/train_swap_seed29.log
```
This is the exact I17 command (docs/I17_I7_SWAP_SUCCESSOR.md:38-40) with only
the data-dir and model-dir changed. Trainer defaults == M7 config; pass no
other flags. Expect in the logs: 48 features; n_train 7972 (base) / 15944
after swap-doubling (swap); n_val 528; n_test 798. Runs take a few minutes
each; run them sequentially, never concurrently.

### 5. Reproduction check
```bash
uv run python scripts/auto/i19_repro_check.py \
    2>&1 | tee research/handoff/I19/raw/repro.log
```
Reference targets (from the archived I17 arms; your retrains must reproduce
the predictions exactly, which implies these too):
- base_seed29: val LL 0.6531566579763646, test LL 0.5973871207926102
- swap_seed29: val LL 0.6503653371453989, test LL 0.5859296475141611

**If max |Δp| ≠ 0 on either arm: STOP.** Do not iterate, tweak, or retrain
again. Record everything in result.md — the idea text pre-declares that a
nonzero delta means something about column order or encoders is load-bearing,
and diagnosing it is the orchestrator's problem.

### 6. Write `research/handoff/I19/result.md`
Numbers copied VERBATIM from the tee'd logs:
- the `I18 RELAXED PARITY GATE:` line and the per-split PASS/FAIL list;
- every contract-check line (row counts, key equality, sidecar results);
- per-arm `max |Δ p_team1|` in scientific notation, n_keys,
  feature_columns.txt equality, and the four log-loss values vs references;
- commit SHAs you created, `git diff --stat fc831c9..HEAD`, wall-clock time
  of each step, anything that crashed or surprised you.
Commit the result + raw logs as `Auto[I19]: executor result`.
Then return a short (≤15 line) final summary message.

## What you must NOT do

- No verdict, no `git revert`, no touching `research/results.tsv` or
  `research/IDEAS.md`, no `git push`, no second idea.
- Never read/write/list anything under `data/golden/` (the frame's
  `golden_test.parquet` copy under `data/auto/i18/frame/` is sanctioned; the
  `data/golden/` tree itself is not).
- Never write into `data/xgb_match_data_i7/`, `models/auto/i17/`,
  `models/auto/i18/`, `models/xgb_match_i7/`, or any production artifact
  (program.md rule 3 list). New artifacts go ONLY to
  `data/xgb_match_data_i7_v2/` and `models/auto/i19/`.
- Do not run `materialize_match_features.py`, `build_i7_match_frame.py`,
  `predict_golden.py`, any sim/prop eval, or anything touching
  `scripts/sim_eval/`.
- No new dependencies; always `uv run` for Python.

## Baseline / bookkeeping context

This is an instrumentation-gate idea (results.tsv row will use
`(instrumentation)` placeholders like D10/B4) — there is no ≥$50k LL/ROI
comparison and you must not produce one. The comparison of record is
prediction-level identity vs `models/auto/i17/{base,swap}_seed29`.
