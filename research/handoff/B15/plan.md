# B15 executor plan — Scale-only quote calibrator re-gated on fresh test draws

Idea id: **B15** (P3, B14 follow-up). Claim commit: `71691d0`.

## Hypothesis

B14 LANDED a per-checkpoint quote calibrator (shift + band scale, val-fit) but
with a val/test bias SIGN MISMATCH: fitted P50 shifts are all negative (sim
under-predicts the 2024-12→2025-06 val pool) while frozen-test raw bias is
positive (+4.670/+3.204/+0.514 at cps 6/10/15) — so the shift term moved test
P50 the WRONG way and cost MAE (pooled −3.086 → −2.774; cp15 per-cp CI
reopened across 0). Essentially all of B14's win is the band-widening scale
term. A **scale-only** calibrator (shift ≡ 0, same scales, NO refit) should
keep the cp15 coverage fix at ~zero MAE cost. Testing it on the SAME frozen
s43 quotes after seeing B14's decomposition would be post-hoc selection, so it
needs **fresh test draws** (B1→B6 precedent).

## Gate (pre-committed; sim-gate idea, dual-condition)

Both computed on the FRESH quote draws, **scale-only arm**:

- **PRIMARY-A (coverage)**: corrected P10–P90 coverage point estimate in
  **[0.70, 0.90] at ALL THREE checkpoints** (6, 10, 15).
- **PRIMARY-B (MAE)**: pooled paired dMAE (|corrected P50 err| − |naive err|),
  cluster-boot by match (2000 reps, seed 29 — same constants as
  `b14_gate_analysis.py`), **CI hi < 0** AND corrected P50 MAE beats naive MAE
  at **each** checkpoint in point estimate.

Both → LANDED; exactly one → TABLED; none → FAILED. The verdict is the
ORCHESTRATOR's to issue, not yours.

- **SECONDARY (recommendation only, no gate weight)**: scale-only vs B14-full
  pooled dMAE point estimate on the same fresh draws (decides which calibrator
  is "of record" if LANDED).
- **DIAGNOSTIC (no gate weight)**: per-checkpoint bias table — val shifts
  (from the calibrator json) vs raw fresh-test bias mean(P50_raw − actual) per
  cp, alongside B5's frozen-test raw bias +4.670/+3.204/+0.514.

## What to implement (exact)

ONE new file: `scripts/auto/b15_gate_analysis.py`. Nothing else changes. Do
NOT edit `b5_inplay_quotes.py`, `b5_gate_analysis.py`,
`b14_gate_analysis.py`, `b14_fit_quote_calibrator.py`, anything in
`scripts/sim_eval/`, or `sim_v1_2.py`. The default sim path stays
byte-untouched (this is a NON-engine idea; B12 consumed tonight's engine
slot).

`b15_gate_analysis.py` must:

1. Load `models/auto/b14/quote_calibrator.json` and **assert** the six values
   match exactly: cp6 shift −1.4482421875 / scale 1.19; cp10 shift −1.78125 /
   scale 1.09; cp15 shift −2.9714566929133857 / scale 1.26. NO refit anywhere.
2. Build two correction arms: **B14-full** (shift + scale) and **scale-only**
   (same scales, shift := 0). READ `scripts/auto/b14_gate_analysis.py` FIRST
   and copy its correction transform, error/coverage definitions, and
   cluster-boot machinery (BOOT_REPS=2000, BOOT_SEED=29) verbatim — do NOT
   re-derive the shift sign convention or the band-widening formula from
   prose; the code is the spec.
3. Support `--quotes <path>` so the same script runs on frozen and fresh
   quotes.
4. **Self-test (mandatory, BEFORE the fresh run is used for anything)**: run
   the B14-full arm against the FROZEN `models/auto/b5/quotes_s43_n261.json`
   and confirm it reproduces B14's logged numbers: pooled corrected dMAE
   **−2.774 [−4.631, −0.864]** and corrected coverage **0.802/0.802/0.756**.
   If these do not reproduce to printed precision, your transform is wrong —
   fix the script; do not proceed to fresh-draw numbers.
5. Print, per arm (scale-only, B14-full) and for raw-uncorrected context:
   per-cp corrected MAE vs naive MAE, pooled paired dMAE + CI, per-cp
   P10–P90 coverage + cluster-boot CI, the diagnostic bias table, and an
   explicit PRIMARY-A / PRIMARY-B MET / NOT MET line for the scale-only arm.

## Order of operations (pre-commit discipline)

1. Read `program.md` (repo root) in full; obey DO NOT CHEAT.
2. Write `scripts/auto/b15_gate_analysis.py`; run the SELF-TEST against the
   frozen s43 quotes (tee output to
   `research/handoff/B15/raw/selftest_frozen_s43.txt`).
3. **Commit** the gate script + self-test output BEFORE launching the fresh
   quote run: `Auto[B15]: implement — pre-committed scale-only gate + frozen-quotes self-test`.
4. Run the eval recipe below.
5. Run the gate script on the fresh quotes; tee to
   `research/handoff/B15/raw/gate_output.txt`.
6. Write `research/handoff/B15/result.md`; commit
   `Auto[B15]: executor result — <one line>`.

## Eval recipe (exact)

```bash
mkdir -p models/auto/b15 research/handoff/B15/raw

# Usage pin pre-check — MUST print ea0c73d3ddb48f499b6273f9a397b0e3
md5 -q models/auto/b12/bowler_phase_usage_pre_b12.json

# Fresh test quote run, seed 45 (∉ {42,43,44,47}), ~25 min (B5's was 1495.8 s)
uv run python scripts/auto/b5_inplay_quotes.py \
    --test-dir data/polymarket_test \
    --n-matches all \
    --n-sims 100 \
    --seed 45 \
    --usage-json models/auto/b12/bowler_phase_usage_pre_b12.json \
    --out models/auto/b15/quotes_s45_n261.json \
  2>&1 | tee research/handoff/B15/raw/quotes_s45_run.log

# Gate
uv run python scripts/auto/b15_gate_analysis.py \
    --quotes models/auto/b15/quotes_s45_n261.json \
  2>&1 | tee research/handoff/B15/raw/gate_output.txt
```

Post-run log checks (record all three in result.md):
- The log must NOT contain the `B10 usage-aligned bowler selector ACTIVE`
  banner (the pinned legacy json has no `b10_asof_usage` key). If the banner
  appears the run is confounded — kill/discard and re-check the pin.
- Venue encoder ACTIVE (467 venues) expected (default sidecar).
- Row/match/skip counts ≈ B5's 756 rows / 253 matches / 8 skips (skips are
  structural: rain-curtailed + fifty-over). If matches < 240 or skips > 15,
  STOP and report — do not "fix" anything.

## Baseline numbers (context; the gate itself is absolute bars on fresh draws)

| quantity | frozen s43 value |
|---|---|
| B5 raw pooled dMAE (sim − naive) | −3.086 [−4.869, −1.289] |
| B5 raw coverage cp6/10/15 | 0.755 / 0.794 / 0.664 (cp15 OUT) |
| B5 raw bias cp6/10/15 | +4.670 / +3.204 / +0.514 |
| B14-full pooled dMAE | −2.774 [−4.631, −0.864] |
| B14-full coverage cp6/10/15 | 0.802 / 0.802 / 0.756 (all IN) |
| B14 val fit (shift/scale) | cp6 −1.4482/1.19, cp10 −1.7812/1.09, cp15 −2.9715/1.26 |

`results.tsv` baseline rows: B14 row (2026-07-31, LANDED) and B5 row
(2026-07-31, TABLED) — both `(sim-gate)` placeholder rows; numbers above are
copied from them.

## Easy to get wrong

1. **Usage pin.** The shipped default `models/bowler_phase_usage.json` is
   B10-ACTIVE post-B12 and would confound (calibrator + all B5/B14 history
   were built on legacy usage). Pin `--usage-json
   models/auto/b12/bowler_phase_usage_pre_b12.json`, verify md5, verify no
   banner.
2. **Shift sign / widening formula.** Do not re-derive; copy from
   `b14_gate_analysis.py` and prove correctness via the frozen-quotes
   self-test reproducing −2.774 [−4.631, −0.864] and 0.802/0.802/0.756.
3. **No refit.** Scales come from the existing calibrator json only. If
   anything tempts you to refit on the fresh quotes, that is selection on the
   test set — forbidden.
4. **Read-only inputs**: `models/auto/b5/quotes_s43_n261.json` and everything
   under `models/auto/b14/` must not be modified (check mtimes after).
5. **Seed 45**, not 42/43/44/47.
6. `models/` is gitignored — artifacts stay on disk; commit only scripts,
   handoff raw text, and result.md.
7. One heavy process at a time; if the quote run exceeds ~60 min (2× B5's
   1495.8 s), kill it and report CRASH facts in result.md.
8. Numbers in result.md must be copied VERBATIM from tool output.

## What you must NOT do

- Do not decide the verdict (LANDED/TABLED/FAILED) — report facts only.
- Do not revert anything.
- Do not touch `research/results.tsv` or `research/IDEAS.md`.
- Do not `git push`, `git reset`, or `git checkout -- .`.
- Do not start a second idea, a second sim run beyond the recipe, or any
  retraining.
- Never touch, read, or evaluate against `data/golden/`.

## result.md must contain

- md5 pre-check output; banner check; row/match/skip counts; runtime.
- Self-test block (frozen s43 reproduction) verbatim.
- Fresh-draw numbers verbatim for BOTH arms + raw: per-cp corrected vs naive
  MAE, pooled dMAE + CI, per-cp coverage + CI, diagnostic bias table,
  PRIMARY-A/PRIMARY-B MET/NOT-MET lines.
- Commit SHAs created; `git diff --stat` vs claim commit `71691d0`.
- Anything that crashed, hung, or surprised you.
