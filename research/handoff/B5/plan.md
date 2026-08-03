# B5 executor plan — in-play over/under quote prototype (fresh claim after CRASH)

Claim commit: 404905f (`Auto[B5]: claim`, 2026-07-31T07:57Z).

## Idea

**B5 [P2]** — in-play over/under quote prototype (analytics-engine seed).

**Hypothesis:** the calibrated sim can produce live over/under quotes for
remaining-innings-1 runs from actual mid-innings states (end of overs
6/10/15). Feasibility prototype: quote quality vs realized outcomes and vs a
naive run-rate-extrapolation baseline.

**History you must respect:** B5 was claimed 2026-07-23 and implemented at
`26a7fd9`, but the eval died at session limits and the claim was closed as a
supervised CRASH row (2026-07-30). The harness was then I15-compat-patched
(`cc0b9cb`, `cricsheet_id=` kwarg into `loader._create_match_state`) AFTER
implementation, which voided the old claim. This fresh claim is valid only if
you (a) re-run `b5_unit_check.py` successfully on CURRENT code before any
eval, and (b) re-verify the pre-committed gate script is intact before any
eval result exists.

## Gate metric pair (sim/prop idea — pre-committed in `scripts/auto/b5_gate_analysis.py`)

- **GATE 1 (MAE vs naive):** sim P50 remaining-runs MAE < naive
  run-rate-extrapolation MAE at ALL THREE checkpoints (point estimates), AND
  the pooled paired per-row `|sim_err| − |naive_err|` cluster-bootstrap CI
  (by match, 2000 reps, seed 29) has hi < 0.
- **GATE 2 (calibration band):** empirical P10–P90 coverage (inclusive) in
  [0.70, 0.90] at ALL THREE checkpoints.

Both → LANDED; exactly one → TABLED; none → FAILED. The gate script prints
the verdict but **you do not issue the verdict** — the orchestrator does.

## Baseline

There is NO results.tsv baseline row for B5 — the comparison baseline (naive
run-rate extrapolation, `(runs_at_cp / cp) * (20 − cp)`) is computed inside
the harness per row. No pairing against any sim detail JSON. Nothing depends
on `models/auto/b10/detail_blind_s43_n261.json`.

## Code changes

**None expected.** The harness is already committed and current:

- `scripts/auto/b5_inplay_quotes.py` — teacher-forced replay + continuations
- `scripts/auto/b5_gate_analysis.py` — pre-committed gate (DO NOT EDIT)
- `scripts/auto/b5_unit_check.py` — replay-parity + live-path smoke

If (and only if) the unit check fails on current code due to drift in the
b5 harness's *own* assumptions, you may make a minimal compat fix to
`b5_inplay_quotes.py` / `b5_unit_check.py` ONLY, commit it
(`Auto[B5]: <what>`) and re-run the unit check from scratch.
`b5_gate_analysis.py` must not change under any circumstances. If the unit
check cannot be made to pass with a minimal harness-side fix, STOP and
report the failure verbatim in result.md — do not touch anything else.

## Steps (run in order, all from repo root)

```bash
mkdir -p research/handoff/B5/raw

# 1. Preflight: record gate-script integrity BEFORE any eval
git log --oneline -2 -- scripts/auto/b5_gate_analysis.py \
    | tee research/handoff/B5/raw/gate_provenance.log
git status --porcelain | tee -a research/handoff/B5/raw/gate_provenance.log
# expected: last commit touching the gate script is 26a7fd9; tree clean

# 2. Unit check (MUST pass before the eval; ~5 min)
uv run python scripts/auto/b5_unit_check.py \
    2>&1 | tee research/handoff/B5/raw/unit_check.log
# expected shape (from the 2026-07-23 run, may differ slightly post-I15):
#   Part 1: replay parity PASS on ~253/261 (curtailed skips only)
#   Part 2: crease-pair match >= 95% (was 756/756)
#   Part 3: live-path smoke deterministic, venue encoder ACTIVE

# 3. Commit the preflight evidence BEFORE the eval
git add research/handoff/B5/
git commit -m "Auto[B5]: preflight — unit check re-run PASS on current code, gate script verified intact at 26a7fd9"

# 4. The eval (~30–60 min; run SYNCHRONOUSLY, do not detach, do not
#    background — block until it completes; D2 lesson: detached children
#    die at session close)
uv run python scripts/auto/b5_inplay_quotes.py \
    --test-dir data/polymarket_test --n-sims 100 --seed 43 \
    --out models/auto/b5/quotes_s43_n261.json \
    2>&1 | tee research/handoff/B5/raw/eval.log

# 5. Gate analysis (fast)
uv run python scripts/auto/b5_gate_analysis.py \
    --quotes models/auto/b5/quotes_s43_n261.json \
    2>&1 | tee research/handoff/B5/raw/gate.log
```

## result.md (write to `research/handoff/B5/result.md`)

Copy VERBATIM from tool output (no rounding, no paraphrase):

- Per-checkpoint blocks from gate.log: n, MAE sim(P50), MAE naive, dMAE with
  CI, coverage with CI and IN/OUT OF BAND, the context line (bias, band
  width, actual sd).
- The pooled paired dMAE line with CI.
- The GATE 1 / GATE 2 lines and the printed VERDICT line (as the script's
  output, not your judgment).
- Eval wall time (`Done in ...s` line), row/match/skip counts from eval.log,
  and unit-check summary lines (Part 1/2/3).
- Commit SHAs you created; `git diff --stat` vs the claim commit 404905f.
- Anything that crashed, was retried, or ran long.

Then commit: `git add research/handoff/B5/ && git commit -m "Auto[B5]: executor result"`.
(`models/auto/b5/` is gitignored — do not force-add it.)

## Hard constraints (program.md DO NOT CHEAT — read it first, obey in full)

- Never modify `scripts/sim_eval/`, `sim_v1_2.py`, `parsing_v2.py`,
  `stats_provider.py`, `stats_sqlite_backend.py`, `betting_odds_polymarket.json`,
  `data/polymarket_test/`.
- Never touch or read `data/golden/`.
- Never overwrite production artifacts; all outputs under `models/auto/b5/`
  and `research/handoff/B5/`.
- `uv run` for every Python invocation.
- ONE heavy process at a time; nothing else may run concurrently with the
  eval.
- If the eval exceeds ~2 h wall time with no per-match progress lines, kill
  it and record the fact in result.md — do not restart it more than once.
- You do NOT: decide the verdict, revert anything, touch
  `research/results.tsv` or `research/IDEAS.md`, `git push`, or start any
  other idea.

## Easy-to-get-wrong list

1. The unit check and the eval must run on the CURRENT default sim path —
   Part 3 hard-asserts the venue encoder autoloads. If that assert fires,
   something is wrong with the environment, not the test; stop and report.
2. The eval prints one line per match — it processing ~253 in-scope matches
   (8-ish curtailed skips is normal). ~7–12 s per match is the expected pace.
3. Seed is 43 and the out path is `models/auto/b5/quotes_s43_n261.json` —
   keep both exactly (the gate script defaults to that path).
4. The `.partial.jsonl` sidecar is a progress artifact, not the result; the
   gate reads only the final JSON.
5. Do not "fix" anything based on the gate numbers — the gate script's
   printed VERDICT is advisory output for the orchestrator; your job ends at
   result.md.
