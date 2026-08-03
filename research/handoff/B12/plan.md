# B12 executor plan — B10 selector re-gated on bowler_wkts_1plus primary at fresh seed 44

Idea id: **B12** (P2, claimed 2026-07-31T16:31:00Z, claim commit `c35cd8b`).
Full idea text: `research/IDEAS.md` § B12. Read `program.md` FIRST and obey
its DO NOT CHEAT section in full.

Orchestrator pick note: D6 (P1) was skipped, not claimed — its own start
condition (">6 h of night remain", ~5 h budget) is unmet in this window and
the D-series ordering runs D6 last; B12 is the front of the P2 queue and the
direct continuation of last night's B10 verdict.

## Hypothesis

B10 (TABLED 2026-07-31, seed 43) share-matched the usage-absent branch of
`EmpiricalBowlerSelector` to B9's as-of expected-balls share. Its
pre-committed `top_bowler` primary was flat (−0.0002 [−0.0008, +0.0005]),
but the guard scan found a CI-clean improvement exactly where D15 left its
residual defect: **`bowler_wkts_1plus` −0.0049 [−0.0075, −0.0023]**, plus
`first_wicket_runs_ou_30_5` −0.0024 [−0.0046, −0.0000] and
`highest_over_runs_ou_24_5` −0.0014 [−0.0030, −0.0000] at the CI boundary,
with ZERO families CI-clean worse anywhere in the 32-family scan. Mechanism
is coherent (debutant bowlers now actually bowl: 1.15%→8.73% XI share vs ≈9%
actual — exactly the tail feeding early/first wickets). But the signal was
identified POST HOC via a guard on seed-43 draws, so it must be confirmed on
fresh Monte Carlo draws with a pre-committed primary before shipping
(B1→B6 precedent: that confirmation reproduced −0.175 → −0.162).

## Gate (sim pair — decided by the ORCHESTRATOR, not you)

- **GATE 1 (primary):** `bowler_wkts_1plus` paired dBrier (b10 − blind)
  improves CI-clean (95% CI entirely < 0) at **seed 44**.
- **GATE 2 (guards):** no CI-clean regression (CI entirely > 0 on a Brier
  family / MAE family) on `top_bowler`, `bowler_wkts_2plus`,
  `batter_runs_mae`, `team_first_over_mae`.
- Both → LANDED (ship decision is the orchestrator's); exactly one →
  TABLED; none → FAILED (B10's guard signal was seed-43 selection noise).
  You do NOT issue the verdict.

## Baseline / comparison numbers (results.tsv row B10, 2026-07-31, seed 43)

These are CONTEXT ONLY — the B12 verdict stands entirely on the fresh
seed-44 paired CIs. Seed-43 reference (from
`research/handoff/B10/raw/gate_output.txt`):

- bowler_wkts_1plus: blind 0.2584 → b10 0.2535, **−0.0049 [−0.0075, −0.0023]**
- top_bowler: −0.0002 [−0.0008, +0.0005] (~noise); G5 coverage blind 0.9958 / b10 0.9990
- bowler_wkts_2plus: −0.0010 [−0.0026, +0.0005]; batter_runs_mae +0.0145
  [−0.0400, +0.0701]; team_first_over_mae +0.0120 [−0.0008, +0.0242]
- 8 relaxation triggers (benign); run times 2857 s (blind) / 2841 s (b10)

## Implementation (exact — you are implementing a decision, not making one)

Repo: `/Users/aryamangupta/CricML/Match_Prediction`. All Python via
`uv run`. New artifacts to `models/auto/b12/`; raw logs to
`research/handoff/B12/raw/`. NEVER edit `scripts/sim_eval/`,
`scripts/parsing_v2.py`, stats backends, `models/bowler_phase_usage.json`,
`models/xgb_v3/`, `data/golden/`, production caches.

### 1. Re-apply B10 verbatim

```
git revert --no-edit a8c061b
```

(a8c061b is the revert of B10's implement commit ad144ea; reverting it
restores `scripts/auto/b10_build_usage_sidecar.py`,
`scripts/auto/b10_gate_analysis.py`, `scripts/auto/b10_unit_check.py`, and
the `sim_v1_2.py` selector branch.) Verified by the orchestrator pre-claim:
nothing under `scripts/sim_v1_2.py` or `scripts/auto/b10_*` changed after
a8c061b, so this applies cleanly. THEN VERIFY:
`git diff ad144ea HEAD -- scripts/sim_v1_2.py scripts/auto/b10_build_usage_sidecar.py scripts/auto/b10_gate_analysis.py scripts/auto/b10_unit_check.py`
must be EMPTY. If not empty, STOP and report.

Do NOT re-implement anything by hand; the B10 code (memoization, name-vs-id
keying, ≥5-eligible relaxation, 1e-9 weight floor, date().isoformat()
as-of semantics) is already correct in the reverted commit.

### 2. Verify artifacts (no rebuild expected)

- `models/auto/b10/bowler_phase_usage_b10.json` and
  `models/auto/b10/usage_corpus.pkl` exist (they survived the revert;
  gitignored). If either is missing, rebuild via
  `uv run python scripts/auto/b10_build_usage_sidecar.py`.
- `md5 -q models/bowler_phase_usage.json` MUST equal
  `ea0c73d3ddb48f499b6273f9a397b0e3` (production usage prior untouched).
- Do NOT touch `models/auto/b10/detail_*_s43_n261.json` — the blind one is
  the canonical seed-43 baseline for the whole loop.

### 3. Pre-run unit check (ALL must pass)

```
uv run python scripts/auto/b10_unit_check.py 2>&1 | tee research/handoff/B12/raw/unit_check_pre.txt
```

Expected: d15_unit_check 30/30 (subprocess), legacy-parity float-exact with
`_b10 is None` on the default payload, exp_balls parity vs
`b9_usage_baseline.AsOfUsage` exact to 1e-12, the weight-mechanism table
(debutants ≈9%, veteran never-bowlers reported — note B10 finding: they
RISE 0.270%→0.496%, that is a known, accepted property of this
implementation, not a failure), production md5 unchanged. If the d15 check
fails: STOP everything and report — blocking engine-state finding.

### 4. Gate script — `scripts/auto/b12_gate_analysis.py` (NEW FILE, COMMIT BEFORE ANY EVAL RESULT)

Clone `scripts/auto/b10_gate_analysis.py` (restored by step 1) and retool —
do NOT edit b10_gate_analysis.py itself:

- Inputs: `models/auto/b12/detail_blind_s44_n261.json` and
  `models/auto/b12/detail_b10_s44_n261.json`. Identical pairing machinery:
  paired per-match deltas, cluster-boot 2000 draws, boot seed 29.
- **GATE 1 block: PRIMARY = `bowler_wkts_1plus`** paired dBrier
  (b10 − blind) with CI; state MET iff CI hi < 0.
- **GATE 2 block: guards = `top_bowler`, `bowler_wkts_2plus`,
  `batter_runs_mae`, `team_first_over_mae`**; each REGRESSED iff CI lo > 0.
- Context (non-gate, report verbatim): G5 coverage both arms (b10 ≥ 0.90
  expected — report it; it is context for B12, not a gate), full
  32-family scan flagging every CI-excludes-0 delta either direction,
  recomputed sim−usage top_bowler margin on the b10 detail via the B9
  pairing (seed-43 reference: +0.0028 blind → +0.0026 b10), relaxation
  trigger count from the Arm-B log. DROP B10's D15-drift-check section
  (obsolete — both arms here are fresh twins).
- Print the both/one/none verdict mapping but do NOT decide.

### 5. Commit BEFORE running any eval

Two commits by this point (order matters; both before any eval output
exists): the step-1 revert commit, then
`Auto[B12]: implement — re-apply B10 selector + pre-committed b12 gate (bowler_wkts_1plus primary, seed 44)`
covering `b12_gate_analysis.py`. (`models/auto/` is gitignored — expected.)

### 6. Evals — sequential, synchronous, never concurrent (16 GB box)

Arm A (blind, seed 44):
```
uv run python scripts/sim_eval/prop_backtest.py \
  --test-dir data/polymarket_test --n-matches all --n-sims 100 --seed 44 \
  --ball-calibrator vector \
  --detail-out models/auto/b12/detail_blind_s44_n261.json \
  --report-out models/auto/b12/report_blind_s44_n261.md \
  2>&1 | tee research/handoff/B12/raw/run_blind.log
```

Arm B (b10 selector, seed 44): same command plus
`--bowler-usage-path models/auto/b10/bowler_phase_usage_b10.json`, with
`blind→b10` in both out paths and the log name.

- Startup lines MUST show, BOTH arms: `venue encoder ACTIVE (467 venues)`,
  empirical bowler selector, run-out dismissal channel ACTIVE, vector
  calibrator. Arm B MUST additionally show
  `B10 usage-aligned bowler selector ACTIVE (k_u=5.0)`; Arm A MUST NOT.
  If wrong, kill immediately and fix before burning 48 minutes.
- Re-run `b10_unit_check.py` once more right before Arm B
  (tee to `research/handoff/B12/raw/unit_check_pre_armb.txt`).
- Each run ≈ 48 min (B10 measured 2857/2841 s). Launch detached (nohup)
  with the tee, then poll the log's progress inside this session until
  completion — do NOT end your run with an eval still going (children are
  reaped at session close; this killed 7 prior launches). Kill at 100 min
  (~2× budget) and report CRASH facts.

### 7. After both evals

```
uv run python scripts/auto/b12_gate_analysis.py 2>&1 | tee research/handoff/B12/raw/gate_output.txt
```

## result.md (write to `research/handoff/B12/result.md`)

Numbers copied VERBATIM from tool output — never rounded or paraphrased:
- GATE 1: bowler_wkts_1plus blind/b10 Brier, dBrier + CI.
- GATE 2: all four guard deltas + CIs.
- Context: G5 coverage both arms; full family-scan CI-clean list (both
  directions); recomputed sim−usage margin; relaxation trigger count.
- Unit-check outcomes (both runs), startup-line confirmations per arm.
- Commit SHAs you created; `git diff --stat` vs claim commit `c35cd8b`;
  eval wall times; anything that crashed or ran long.
Return a SHORT summary as your final message (the orchestrator reads
result.md, not your transcript).

## What you must NOT do

- Do not decide the verdict; do not revert anything; do not touch
  `research/results.tsv` or `research/IDEAS.md`; do not `git push`; never
  `git reset`; do not start a second idea.
- Never edit `scripts/sim_eval/`, `parsing_v2.py`, `stats_provider.py`,
  `stats_sqlite_backend.py`; never touch `data/golden/`, `models/xgb_v3/`,
  `models/xgb_match_v3_m7*`, production caches, or
  `models/bowler_phase_usage.json` itself.
- Do not modify the restored B10 selector code or the default (non-b10)
  path — when the b10 key is absent the code path must remain behaviorally
  identical to HEAD-at-ad144ea.
- Do not overwrite anything in `models/auto/b10/` (seed-43 details are the
  canonical baseline lineage). All B12 outputs go to `models/auto/b12/`.
- No new dependencies; no `--parallel`; one heavy process at a time.

## Easy to get wrong

- **Seed 44 everywhere**: both arms, and `s44` in all four out-file names.
  A seed-43 rerun would silently invalidate the fresh-draws confirmation.
- The verdict primary is `bowler_wkts_1plus`, NOT `top_bowler` — do not
  read the verdict off the restored `b10_gate_analysis.py` (its GATE 1 is
  B10's old primary; it must not be the committed gate for B12).
- b10_gate_analysis.py may hardcode the seed-43 paths — when cloning,
  point b12_gate_analysis.py at the b12 seed-44 details.
- The B9 corpus keys on cricsheet NAME, phase-usage on player_id (already
  handled in the reverted code — just don't "fix" it).
- Gate script committed BEFORE any eval result exists; evidence order is
  checked at verdict time.
