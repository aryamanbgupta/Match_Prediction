# B17 orchestrator plan — i7 in-play P50 bias attribution (diagnostic-only iteration)

Claim commit: `7f56a9a` (2026-08-03 06:52 UTC). Orchestrator: Fable. Executor: one Opus subagent.

## Idea

**B17 [P3]** (IDEAS.md:1256): B16 found the raw i7 quote-path P50 bias
sign-flipped vs legacy — legacy +4.259/+2.777/+0.410 remaining runs at
checkpoints (end of overs) 6/10/15 → promoted i7 no-weights RAW stack
−4.781/−3.026/−1.946 — while still beating naive on MAE (pooled dMAE −3.417
[−4.878, −2.066] CI-clean). Prime suspect: the flat 1%+1% extras graft
(true val wides 0.037702, no-balls 0.004409 per D3) composes differently
with the no-weights class distribution. D3 showed the legacy
labels-fold-extras channel over-carried runs; the retrained conditionals may
now under-carry, leaving continuation totals ~1.9–4.8 runs short.

## SCOPE DECISION (orchestrator, made at claim, before any result)

**This iteration runs B17's diagnostic steps (1)+(2) ONLY. The step-(3)
engine arm is out of scope tonight**: B13 already consumed tonight's
one-sim-engine-idea-per-night slot (engine change evaluated 01:36–02:07 EDT,
reverted at TABLED). B17's method text attaches "Respect
one-sim-engine-idea-per-night" to step (3) — so no `sim_v1_2.py` change, no
recipe-B run, no quote run this iteration. The diagnostics are pure analysis
+ teacher-forced booster scoring (minutes, not hours).

## PRE-COMMITTED VERDICT MAPPING (orchestrator applies it; executor does NOT decide)

Definitions (all in **runs per legal ball**, same code, same val matches,
both stacks):

- `M_stack` = sim-composed expected runs carried per legal ball under the
  stack's serving config (6-class expectation composed with the flat graft,
  crediting runs exactly as the engine does — see Task 2).
- `A` = actual runs per legal ball on the same matches (ALL runs: off-bat +
  wides + no-balls + byes + leg-byes + penalty).
- `g_stack = M_stack − A` (negative = under-carry).

Observed quote biases per legal ball at cp6 (14 remaining overs = 84 legal
balls; B16 gate output): i7 −4.781/84 = **−0.0569**; legacy +4.259/84 =
**+0.0507**; sign-flip delta = **−0.1076**.

- **Attribution CLEAN** iff BOTH:
  (a) `g_i7 ≤ −0.0285` (the carry deficit explains ≥ half of i7's own cp6
      bias), AND
  (b) `g_i7 − g_legacy ≤ −0.0538` (the paired stack contrast explains ≥ half
      of the sign-flip delta; robust to channels shared by both stacks —
      threes-folding, unmodeled byes).
  → Verdict **TABLED** (attribution done, engine arm split out as a new
  PENDING idea needing its own engine-slot night).
- **Otherwise → FAILED** per B17's own pre-committed mapping
  ("diagnostic-only outcome, attribution inconclusive"), with the full
  decomposition logged — the decomposition then names the true channel
  (e.g. class-marginal or wicket-rate, not extras composition).

## TASK 1 — decompose the continuation bias on the EXISTING B16 quotes (free)

Input: `models/auto/b16/quotes_i7_s48_n261.json` (756 rows / 253 matches /
8 skips, seed 48). Schema: read `scripts/auto/b5_inplay_quotes.py` (the
harness that wrote it). If the legacy twin exists (B15's s45 quote run,
look under `models/auto/b15/`), run the identical decomposition on it for
contrast; if absent, note that and proceed i7-only (B16/B5 logged means are
the legacy reference).

1. **Reproduce the headline first**: per-checkpoint mean P50 bias
   (P50 − actual remaining runs) must match B16's logged
   −4.781/−3.026/−1.946 at cp6/10/15. If it does not, STOP and report the
   mismatch — do not proceed.
2. Bias per remaining over per checkpoint (÷14/÷10/÷5).
3. **Paired segment rates** on matches where both checkpoint rows exist:
   segment 6→10 = (cp6 bias − cp10 bias)/4 overs, segment 10→15 =
   (cp10 − cp15)/5, segment 15→20 = cp15/5. This is the over/phase profile
   of the deficit: flat ≈ per-ball run-mass channel; concentrated at the
   death ≈ different mechanism.
4. Bias by wickets-fallen-at-checkpoint band (0–2 / 3–5 / 6+) and by
   score-at-checkpoint tercile, per checkpoint.
5. Write `research/handoff/B17/raw/decomposition.txt` (human-readable) +
   `models/auto/b17/decomposition.json`.

## TASK 2 — teacher-forced run-mass audit, BOTH stacks

Goal: measure exactly how much run mass per legal ball each serving stack
carries vs actual, and decompose the gap by channel.

**Stacks (serving configs — mirror them exactly):**
- **i7 (promoted)**: `models/xgb_i7_noweights_production/xgboost_model_i7.pkl`
  + its encoders + `feature_columns_i7.txt`, scored RAW (no calibrator —
  D17), on `data/xgb_data_i7/cricket_data_i7_validation.parquet`.
- **legacy**: `models/xgb_v3/xgboost_model_v3.pkl` + encoders +
  `feature_columns_v3.txt`, with `models/xgb_v3/vector_scaling_calibrator_v1.pkl`
  applied (the legacy quote path serves calibrated), on
  `data/xgb_data_v3/validation.parquet`.

First read `models/xgb_i7_noweights_production/marginal_audit.json` (D16's
audit sidecar) — reuse/cross-check whatever per-class marginals it already
contains; report any disagreement with your recompute.

**Steps:**
1. Verify both val parquets cover the same match set/window (expected: post-B2
   stems, 2024-12-31 ≤ date < 2025-06-30, 545 matches — the D3/B3 val
   convention). Log match and row counts per frame. Determine — do not
   assume — whether parquet rows are all deliveries (extras included) or
   legal deliveries only: compare per-match row counts vs cricsheet legal
   and total delivery counts on ≥5 matches, per frame. Document the answer;
   the composition formula below depends on it.
2. Batch-score each val parquet with its booster (+ calibrator for legacy) →
   mean 6-class probability vector, overall and per phase (overs 0–5 /
   6–14 / 15–19). Also actual class frequencies per frame → per-class
   marginal deltas (predicted − actual) for both stacks.
3. **Compose the engine's carried run mass.** Read the wrapper graft in
   `scripts/sim_v1_2.py` (flat sites ≈ :1638, :1714) and
   `T20Rules.simulate_ball` to confirm exact crediting: graft adds
   wide 0.01 + no_ball 0.01 then renormalizes (effective 0.009804 each,
   = 0.01/1.02); confirm what runs a wide/no-ball credits in the engine
   (expected: exactly 1 + re-delivery; verify, and verify whether a no-ball
   also produces off-bat runs). Mirror the engine analytically — document
   the formula you derive in result.md. Reference shape (VERIFY against the
   code; if the engine differs, follow the engine):
   `E_delivery = (Σ_c p_c·runs_c)/1.02 + (0.01/1.02)·1 + (0.01/1.02)·1`
   with `runs_c = {dot:0, one:1, two:2, four:4, six:6, wicket:0}`, and
   `M = E_delivery / (1 − 0.02/1.02)` per legal ball. Use STRICT class run
   values (the sim credits class two as exactly 2 — the threes fold is
   measured separately below).
4. **Actuals** on the same matches from `data/t20s_json/` (male, val window;
   reuse the split convention in `scripts/auto/d3_build_extras_rates.py`):
   `A` = total runs (all channels) / legal deliveries. Also break out:
   actual off-bat runs per legal ball, actual wide+no-ball run mass per
   legal ball, actual bye+leg-bye+penalty mass per legal ball, actual
   frequency-weighted mean runs on class-2-labeled balls (the 2-vs-3 fold
   shortfall).
5. Report per stack: `M`, `A`, `g = M − A`, and the channel decomposition —
   explicit-extras channel (graft mass 0.0196·(…) vs actual extras mass),
   off-bat channel (6-class expectation vs actual off-bat mass), shared
   unmodeled channels (byes, threes fold), per-class marginal deltas, and
   per-phase `g`. Then the paired contrast `g_i7 − g_legacy` and which
   channels carry it.
6. Evaluate the two pre-committed conditions (a) and (b) NUMERICALLY and
   print them labeled in the raw output — but do NOT write a verdict.

Known anchors to sanity-check against (from D3, val split): p_wide
0.037702, p_no_ball 0.004409, old graft 0.009804/0.009804; D16
`marginal_audit.json` for i7 marginals.

## Outputs & commits (executor)

- Scripts: `scripts/auto/b17_decompose_quotes.py`,
  `scripts/auto/b17_runmass_audit.py`. COMMIT THEM BEFORE running
  (`Auto[B17]: implement — quote-bias decomposition + teacher-forced
  run-mass audit (diagnostic-only)`).
- Raw output teed to `research/handoff/B17/raw/` (decomposition.txt,
  runmass_audit.txt); JSON artifacts to `models/auto/b17/` (gitignored is
  fine).
- `research/handoff/B17/result.md`: verbatim numbers (per-checkpoint biases,
  segment rates, M/A/g per stack, channel table, conditions (a)/(b)
  arithmetic), commit SHAs, `git diff --stat` vs claim `7f56a9a`, anything
  that crashed. Commit
  (`Auto[B17]: eval + result.md (no verdict — orchestrator decides)`).

## Baseline rows for comparison

- B16 row (results.tsv 2026-08-03): i7 raw P50 bias −4.781/−3.026/−1.946,
  pooled dMAE −3.417 [−4.878, −2.066], 756 rows/253 matches.
- B5 row (2026-07-31): legacy quote path, bias +4.670/+3.204/+0.514 (B5's
  own s43 run); B16's legacy comparator run (s45, B15) logged
  +4.259/+2.777/+0.410. Use the B16-era numbers (+4.259 at cp6) for the
  pre-committed thresholds, as derived above.

## Easy to get wrong (read twice)

1. **Row ≠ delivery ≠ legal ball.** Labels fold extras (D3: a wide-1 is
   labeled `one`), so parquet rows may include non-legal deliveries.
   Establish the row semantics empirically (step 1) before dividing by
   anything. All final quantities are per LEGAL ball.
2. **Legacy must be scored calibrated** (vector v1 applied to the booster
   probs BEFORE the graft composition — the calibrator runs pre-graft in the
   engine; confirm at the graft site). i7 must be RAW.
3. **Strict class run values** in M (sim credits two=2, never 3). The
   threes fold is a separate reported channel, NOT folded into M.
4. Both stacks' `A` come from the SAME match set; if the two frames' val
   match sets differ, intersect them and log the drop.
5. This is NOT an engine change and NOT a sim run. Do not edit
   `sim_v1_2.py`, do not launch `prop_backtest.py` / `run_sim_eval.py` /
   `b5_inplay_quotes.py`. Reading them is fine.
6. Never touch `data/golden/`, `scripts/sim_eval/`, production model dirs,
   `research/results.tsv`, `research/IDEAS.md`. Never push/reset/revert.
