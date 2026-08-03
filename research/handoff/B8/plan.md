# B8 — Hybrid calibrator: stale v1 global + venue-ON over-0 vector (B7 decomposition)

Idea id: **B8** (P2, claimed RUNNING 2026-07-31T07:06Z, claim commit 7690909).
Gate type: **sim pair** (per-idea gate below). Match-level LL/ROI are NOT
involved; results.tsv row will use `(sim-gate)` placeholders.

## Hypothesis

B7 (TABLED 2026-07-17) decomposed cleanly: refitting the *global* vector
calibrator on venue-ON val predictions actively hurts (pooled 6-line tail
dBrier +0.0079 CI-clean worse → the E5 v1 global is NOT stale under
venue-ON), while the *over-0* vector delivers the A14/A15 first-over gain
under venue-ON (`team_first_over_mae` −0.024 [−0.040, −0.007] CI-clean,
separable because over-0 balls never see the global vector). The current
venue-ON default runs bare v1 with NO first-over correction, so A15's
validated gain is not collected. A hybrid
`OverVectorScalingCalibrator{_global = stale v1 vector, _v[0] = B7's
venue-ON refit over-0 vector}` should retain the first-over win with none
of B7's tail regressions (those came from the refit global, which the
hybrid drops).

## Two orchestrator rulings (pre-committed — do not re-derive)

1. **Baseline = `models/auto/b10/detail_blind_s43_n261.json`**, NOT the
   `models/auto/b6/detail_venue_s43_n261.json` named in the IDEAS.md entry.
   The idea text itself says the b6 baseline is valid "only while no
   sim-engine idea has landed since B6" — D1 and D15 landed after B6, and
   B10 (2026-07-31) additionally measured CI-clean drift on the same-seed
   D15 detail across the I5/I9 refactors and declared the b10 blind twin
   the canonical seed-43 baseline. The b10 blind detail is the ONLY detail
   JSON produced under the current engine: seed 43, venue-ON default path,
   `--ball-calibrator vector` (v1, no path override). Same seed 43 → clean
   pairing.
2. **Over-0 source = the venue-ON refit** at
   `models/auto/b7/over0_calibrator_venueon.pkl` (the idea's stated
   default; A15's venue-blind `models/auto/a15/over0_calibrator.pkl` is
   NOT used). CAUTION: that pkl's `_global` is B7's harmful REFIT global —
   it must NOT leak into the hybrid. Only its `_v[0]` is used.

## Implementation (exact, by file)

### 1. `scripts/auto/b8_compose_hybrid.py` (new)

Composes the hybrid from existing artifacts. NO fitting, NO val sim runs
anywhere in this idea. Steps:

- Load `models/xgb_v3/vector_scaling_calibrator_v1.pkl`
  (`VectorScalingCalibrator`, its 6-vector is attribute `._v`) and
  `models/auto/b7/over0_calibrator_venueon.pkl`
  (`OverVectorScalingCalibrator`, class in `scripts/calibration.py:477`;
  vectors in `._v` dict + `._global`). Use `joblib` and import the classes
  from `calibration` (see `scripts/auto/a15_fit_over0_calibrator.py` for
  the sys.path pattern).
- Build `hybrid = OverVectorScalingCalibrator(weights={0: b7._v[0]},
  global_weights=v1._v)` and save with joblib to
  `models/auto/b8/hybrid_calibrator.pkl`.
- Assertions (all must pass; print each):
  - `sorted(b7._v.keys()) == [0]`
  - `max|hybrid._global − v1._v| == 0.0` exactly
  - `max|hybrid._v[0] − b7._v[0]| == 0.0` exactly
  - `max|b7._global − v1._v| > 0.05` (confirms the refit global we are
    deliberately DROPPING diverges ~0.17 — i.e. the hybrid is materially
    different from B7's evaluated stack)
  - functional: for a random (seed-0) batch of prob rows,
    `hybrid.calibrate_probs(p, over=k) == v1.calibrate_probs(p)`
    bit-exact for k in {1, 7, 19} and for `over=None`;
    `hybrid.calibrate_probs(p, over=0) ==
    b7.calibrate_probs(p, over=0)` bit-exact.
- Print the over-0 divergence `max|_v[0]/_global − 1|` (staleness/washout
  context; expect ≳0.2).

### 2. `scripts/auto/b8_gate_analysis.py` (new, pre-committed BEFORE the eval)

Copy `scripts/auto/b7_gate_analysis.py` and change only: the two detail
paths (baseline = `models/auto/b10/detail_blind_s43_n261.json`, challenger
= `models/auto/b8/detail_b8_s43_n261.json`), the docstring, and the gate
logic to B8's pre-committed gate:

- **GATE 1 (PRIMARY, decisive)** — BOTH sub-conditions required:
  - (a) improvement: `team_first_over_mae` dMAE (b8 − blind) < 0 with 95%
    CI excluding 0;
  - (b) no-regress: NO CI-excludes-0 *increase* on ANY of {pooled tail
    dBrier over the same 6-line `TAIL_POOL` as b7_gate_analysis.py,
    `bowler_wkts_1plus` dBrier, `batter_runs_mae` dMAE}.
- **GATE 2 (guards)** — no CI-excludes-0 increase on `top_bowler` dBrier,
  `team_total_fours_mae`, `team_total_sixes_mae`.
- Verdict mapping printed by the script: GATE1 met + GATE2 held → LANDED;
  exactly one → TABLED; neither → FAILED.
- Keep the b7 script's context families (reported, cannot flip verdict).
- Reuse `a8_gate_analysis` helpers (paired rows, cluster bootstrap by
  match) exactly as b7_gate_analysis.py does. Do not change resample
  count/seed from that tooling's defaults.

### 3. Engine-state parity check (before the eval)

`git diff 91be8d7 HEAD -- scripts/sim_v1_2.py scripts/calibration.py`
must be EMPTY (91be8d7 = B10 claim, pre-B10-implement; B10 was reverted
and I19 touched no sim files — so current HEAD's default sim path is
byte-identical to what produced the blind baseline, whose b10-sidecar
branch was proven inert when the key is absent). If the diff is
non-empty, STOP — do not run the eval; report the diff in result.md.

Also confirm from `research/handoff/B10/raw/` (blind-arm startup log)
that the baseline ran: venue encoder ACTIVE (467 venues), empirical
bowler selector, `--ball-calibrator vector` with NO `--ball-calibrator-path`,
seed 43, 261 matches × 100 sims. Quote the exact launch command you find
there in result.md.

**Commit everything above (`Auto[B8]: implement — hybrid over-0 calibrator
compose + pre-committed gate`) BEFORE launching the eval.**

## Eval recipe (one heavy run)

Smoke first (~1 min, validates wiring):

```bash
uv run python scripts/sim_eval/prop_backtest.py \
  --test-dir data/polymarket_test --n-matches 3 --n-sims 5 --seed 43 \
  --ball-calibrator vector \
  --ball-calibrator-path models/auto/b8/hybrid_calibrator.pkl \
  --detail-out models/auto/b8/smoke_detail.json \
  --out models/auto/b8/smoke_report.md
```

Startup MUST print `Ball calibrator: vector scaling
(models/auto/b8/hybrid_calibrator.pkl)` and `venue encoder ACTIVE (467
venues)`. If either line is absent, stop and diagnose before the full run.

Full run (expected ~48 min; B10's twin runs took 2857 s / 2841 s):

```bash
mkdir -p research/handoff/B8/raw
nohup uv run python scripts/sim_eval/prop_backtest.py \
  --test-dir data/polymarket_test --n-matches all --n-sims 100 --seed 43 \
  --ball-calibrator vector \
  --ball-calibrator-path models/auto/b8/hybrid_calibrator.pkl \
  --detail-out models/auto/b8/detail_b8_s43_n261.json \
  --out models/auto/b8/report_b8_s43_n261.md \
  > research/handoff/B8/raw/run_b8_s43.log 2>&1 &
```

Launch detached exactly as above (nohup + background, `start_new_session`
semantics — the D2 lesson: evals die if the session closes while they
run), then WAIT IN-SESSION, polling the log every few minutes until
`261/261` completes and the process exits. Kill + report CRASH if it
exceeds 100 min (2× budget). One heavy process at a time — never
`--parallel`, nothing else heavy concurrently.

Then:

```bash
uv run python scripts/auto/b8_gate_analysis.py \
  | tee research/handoff/B8/raw/gate_output.txt
```

## Baseline numbers for context

- Blind (v1) team_first_over MAE will be read from the b10 blind detail by
  the gate script; historical anchors: a8 vec era 3.535, D15 era 3.526 →
  3.415. Expected b8 first-over dMAE ≈ −0.02 CI-clean if the B7/A15 gain
  holds (B7 refit-stack −0.024 [−0.040,−0.007]; A15 venue-blind −0.018
  [−0.032,−0.003]).
- B7's failure mode to watch: pooled 6-line tail +0.0079 CI-clean worse,
  bowler_wkts_1plus +0.0024 — those came from the refit GLOBAL. The hybrid
  contains v1's global bit-exactly, so any tail movement can only come from
  over-0 balls' contribution to those lines (pp_total lines include over 0;
  A15 saw pp_total_ou_45_5 IMPROVE −0.0037). A CI-clean tail regression
  would mean the b7 over-0 vector is stale on the current engine.

## Easy to get wrong

1. **No refitting.** If you find yourself running a val sim or calling
   `fit`/`set_global` on real data, stop — the idea is composition of
   existing artifacts only.
2. **The b7 pkl's `_global` is poison** (B7's refit global). The hybrid's
   `_global` must be v1's vector, asserted bit-exact.
3. **Baseline is the b10 blind detail** — not b6, not d15, not b7's
   detail. Do not "correct" this after seeing numbers.
4. Only delta vs the baseline run = `--ball-calibrator-path`. Same seed
   43, same flags otherwise, current HEAD engine.
5. `models/` is gitignored — the pkl will not be committed; the compose
   script (committed) makes it reproducible.
6. Commit implement BEFORE the eval; tee raw output under
   `research/handoff/B8/raw/`; write `result.md` IMMEDIATELY once the
   gate script finishes (a recent executor died before writing result.md
   — do not batch this to the end of a long turn).
7. Do not touch `research/results.tsv`, `research/IDEAS.md`,
   `scripts/sim_eval/`, `data/golden/`, production artifacts. No
   `git push`, no reverts, no second idea, no verdict — the orchestrator
   decides the verdict.

## result.md must contain (verbatim from tool output)

- Compose-script assertion lines + the over-0 divergence number.
- Engine-parity diff result + the b10 blind launch command found in raw/.
- Smoke startup lines (calibrator path + venue encoder).
- Full-run wall time and `261/261` completion line.
- The full GATE 1 / GATE 2 numbers with CIs from b8_gate_analysis.py
  (team_first_over_mae dMAE + CI; pooled tail dBrier + CI;
  bowler_wkts_1plus; batter_runs_mae; top_bowler; fours/sixes MAE), plus
  the context-family scan.
- Commit SHAs created; `git diff --stat 7690909..HEAD`.
- Anything that crashed, hung, or ran long.
