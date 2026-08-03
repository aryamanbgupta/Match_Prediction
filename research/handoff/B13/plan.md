# B13 — Never-bowler damping in the usage-absent branch (B10 defect-(b) fix)

Orchestrator plan, 2026-08-03. Claim commit: `55677fb` (`Auto[B13]: claim`).
Idea entry: `research/IDEAS.md` § B13 (P3, RUNNING). Read `program.md` first
and obey DO NOT CHEAT in full.

## Hypothesis

B10/B12 (LANDED, shipped) aligned the `EmpiricalBowlerSelector` usage-absent
branch to B9's as-of expected-balls — fixing defect (a) (true debutants now
bowl ~9% as they should) but mechanically WORSENING defect (b): a veteran
never-bowler (n≥20 XI appearances, 0 corpus balls bowled) gets
`exp_balls = k_u·prior/(k_u+n)` ≈ 1–2 balls at k_u=5, ABOVE the legacy α
floor, so that cohort's bowling share ROSE 0.270% → 0.496% (actual ≈0). A
zero-usage-aware damping — P(bowls at all | n appearances, 0 balls) that → 0
as n grows — keeps the debutant fix (n=0 path untouched) while sending
high-n zero-ball veterans (keepers/pure batters: PD Salt 285 apps/0 balls,
DP Conway 194/0, SD Hope 203/0 — all in the corpus) toward their true ≈0
share. Attacks the residual sim−usage top_bowler margin (+0.0026 after B10 on
the legacy stack) and B4's 477-row p_sim≥2% non-bowler tail.

## Gate metric pair (sim pair — verbatim from the IDEAS entry, pre-committed)

- **GATE 1 (PRIMARY, both conjuncts required):**
  1a. `top_bowler` Brier improves CI-clean paired (b13 − blind, CI < 0).
  1b. The recomputed sim−usage margin SHRINKS vs the blind arm
      (margin = sim top_bowler Brier − B9 usage-baseline top_bowler Brier,
      both computed on the promoted stack; point comparison, report the
      paired CI as context).
- **GATE 2 (guards):** `bowler_wkts_1plus`, `bowler_wkts_2plus`,
  `batter_runs_mae`, `team_first_over_mae` — no CI-clean regression.
- Verdict (orchestrator's job, NOT yours): both gates → LANDED; exactly one
  → TABLED; none → FAILED.

## Baseline (blind arm) — DO NOT re-run it

`models/auto/d16/detail_noweights_raw_s46_n261.json` — the canonical
production-stack baseline per program.md recipe B (D16 no-weights arm,
**seed 46**, n=261×100, promoted i7 stack, RAW/no calibrator). Verified by
the orchestrator tonight:

- Its raw log (`research/handoff/D16/raw/run_noweights_raw.log`) shows the
  B10 selector ACTIVE (`B10 usage-aligned bowler selector ACTIVE (k_u=5.0)`),
  venue encoder ACTIVE (373 venues), run-out channel ACTIVE, no calibrator →
  it IS the current default path, so one fresh b13 run at seed 46 pairs
  cleanly against it (delta = damping alone).
- `git diff ea4acdb..HEAD -- scripts/sim_v1_2.py` is EMPTY (engine unchanged
  since that run); the promotion commit `73c6dea` only re-pointed
  `prop_backtest.py` defaults at byte-identical artifacts (booster md5
  `7ee1e1809917f45be7e726b3ea4a8a6c`).
- `models/bowler_phase_usage.json` md5 is still the B12-shipped
  `2e650423f0c949631fca1f15dd1c8a56`.

Re-verify all three facts yourself before the eval; abort and report if any
mismatch.

## Implementation (be exact; you are implementing decisions, not making them)

### 1. `scripts/auto/b13_build_damping_sidecar.py` (NEW file)

Fits the damping constants from `models/b10_usage_corpus.pkl` (structure:
`corpus["player"][name] = [(iso_date, balls), ...]`; 7,433 players, 208,753
appearance rows, 45.8% of rows are 0-ball appearances) and writes the opt-in
sidecar. Mirror `scripts/auto/b10_build_usage_sidecar.py`'s style (read it).

- **Event set:** for every player, sort rows by date (same ordering as
  `_B10AsOfExpBalls`). For each row i (0-indexed) with prior appearance
  count n = i ≥ 1, prior sum of balls == 0, and row date < `2025-07-01`
  (training-window discipline, same cutoff as D15's run-out rates): emit
  event `(n, bowled = balls_i > 0, balls_i)`.
- **k_damp:** MLE of `P(bowls | n) = k/(k+n)` over the events — maximize
  `Σ log(p) if bowled else log(1−p)` on a deterministic 1-D grid
  `k ∈ np.geomspace(0.01, 1000, 2001)`. No new dependencies.
- **mu_active:** pooled `mean(balls_i | bowled)` over the same events.
- **Diagnostics (print + save to the handoff):** event count, bowled count;
  empirical P(bowls) binned by n ∈ {1, 2, 3–5, 6–10, 11–20, 21–50, 51+} vs
  the fitted curve; example damped expected balls
  `(k_damp/(k_damp+n))·mu_active` at n ∈ {1, 5, 20, 100, 285} vs the
  undamped B9 `exp_balls` at the current global prior.
- **Sidecar:** `models/auto/b13/bowler_phase_usage_b13.json` = deep copy of
  `models/bowler_phase_usage.json` with ONE addition nested INSIDE the
  existing `b10_asof_usage` object:
  `"b13_never_bowler_damping": {"k_damp": <float>, "mu_active": <float>}`.
  Keep `corpus_path` as-is (`models/b10_usage_corpus.pkl` — no copy needed).
  Assert the production json md5 `2e650423f0c949631fca1f15dd1c8a56` before
  AND after (it is READ-ONLY; never modify it).

### 2. `scripts/sim_v1_2.py` edit (sim engine is editable; `scripts/sim_eval/` is NOT)

Two sites, both gated on the cfg key so the default path is byte-identical
when the key is absent (B10 opt-in precedent):

- `_ensure_b10` (~line 600–625): after `self._b10_cfg = cfg`, read
  `d = cfg.get("b13_never_bowler_damping")`; if present, store
  `self._b13_k_damp = float(d["k_damp"])`,
  `self._b13_mu_active = float(d["mu_active"])` (else attribute stays
  `None`) and print a banner:
  `B13 never-bowler damping ACTIVE (k_damp=..., mu_active=...)`.
- `_b10_share_weights` unknown branch (~line 700–704, the
  `shares.append(self._b10.exp_balls(...) / _B10_INNINGS_BALLS)` line): when
  the b13 cfg is present, compute
  `n, b = self._b10.player_sums(player.name, date)`; if `n >= 1 and b == 0`
  use `share = (k_damp / (k_damp + n)) * mu_active / _B10_INNINGS_BALLS`,
  else fall through to the existing `exp_balls` path. Do NOT touch
  `_B10AsOfExpBalls.exp_balls` or its cache. n=0 debutants and
  usage-present players must be byte-untouched.

Nothing else changes: not the relaxation logic, not the legacy branch, not
`select_bowler`. Note the per-(date,phase,lineup) weight cache already keys
the output — damping flows through it naturally.

### 3. `scripts/auto/b13_unit_check.py` (NEW file — do NOT edit `b10_unit_check.py`; its md5 pin is pre-ship-stale by design)

Read `scripts/auto/b10_unit_check.py` first and mirror its lineup-battery /
weight-table construction over the 261 test matches. Compute XI-share tables
for three arms: legacy (`force_legacy=True`), b10 (production json), b13
(sidecar json). PASS requires ALL of:

1. Cohort (b) — ≥20 prior apps, 0 prior balls: b13 share **< 0.10%**
   (legacy α gave 0.270%, b10 gave 0.496%).
2. Cohort (a) — true debutants (0 prior apps): b13 share within ±0.5pp of
   b10's (≈8.7%; the n=0 path is untouched and the renormalization keeps
   unknown players' target shares ≈ their s_i by construction).
3. Usage-PRESENT players' weights float-equal b10 vs b13 in every battery
   lineup (their `legacy[i]` weights never pass through the scale).
4. Inertness without the key: on the production json, the damped branch is
   never taken (instrument a counter or assert weights identical to b10's
   table) — the shipped default path must be behavior-identical until a
   human ships the cfg.
5. Report (context, not pass/fail): relaxation-trigger count b10 vs b13 on
   the battery — damping can push a never-bowler's share below
   `min_share=0.01` and reduce the eligible count; if triggers explode
   (≫ B12's 8), say so loudly in result.md.

### 4. `scripts/auto/b13_gate_analysis.py` (NEW file — COMMIT BEFORE THE EVAL STARTS)

Template: `scripts/auto/b12_gate_analysis.py` (read it; reuse its
identity-keyed pairing + paired cluster-bootstrap by match, 2000 reps, fixed
seed). Inputs: blind = `models/auto/d16/detail_noweights_raw_s46_n261.json`,
b13 = `models/auto/b13/detail_b13_s46_n261.json`. Emit:

- GATE 1a: `top_bowler` dBrier (b13 − blind) with CI.
- GATE 1b: sim−usage margin on BOTH arms — reuse
  `scripts/auto/b9_usage_baseline.py` machinery read-only (B10/B12 did
  exactly this recompute; follow their call pattern) on the same markets;
  margin_b13 < margin_blind → shrunk.
- GATE 2: the four guards with CIs.
- Full family scan table (all ~33 families, B12 format) for context.
- Print a pre-committed PASS/FAIL line per gate condition. No verdict.

### 5. Eval (ONE run; the blind arm already exists)

```bash
mkdir -p research/handoff/B13/raw models/auto/b13
uv run python scripts/sim_eval/prop_backtest.py \
    --test-dir data/polymarket_test --n-matches all --n-sims 100 --seed 46 \
    --bowler-usage-path models/auto/b13/bowler_phase_usage_b13.json \
    --detail-out models/auto/b13/detail_b13_s46_n261.json \
    --report-out research/reports/auto/B13_props.md \
    2>&1 | tee research/handoff/B13/raw/run_b13_s46.log
```

Defaults already load the promoted stack (i7 sqlite, no calibrator). Expect
~35–50 min serial; kill + report CRASH if past ~100 min. One heavy process
at a time; NEVER `--parallel`. Before launching, confirm the three baseline
facts above; after launch, confirm the startup banners: i7 sqlite, `venue
encoder ACTIVE (373 venues)`, `Run-out dismissal channel ACTIVE`, `B10
usage-aligned bowler selector ACTIVE (k_u=5.0)`, your new B13 banner, and NO
`Ball calibrator:` line. Then run the gate script, tee its output to
`research/handoff/B13/raw/gate_output.log`.

## Commit discipline

- Commit sidecar builder + engine edit + unit check + gate script (message
  prefix `Auto[B13]:`) BEFORE running the eval. Multiple commits fine.
- After the eval + gate: commit the report/handoff artifacts. `models/` and
  large JSONs are gitignored — that is expected; do not force-add them.

## `research/handoff/B13/result.md` must contain

- Fitted k_damp, mu_active, event counts, the binned empirical-vs-fitted
  table, and the damped-vs-undamped expected-balls examples.
- Unit-check table: cohort (a)/(b) shares under legacy / b10 / b13 + the
  four PASS lines + relaxation-trigger counts.
- Gate numbers VERBATIM from gate_output.log: GATE 1a dBrier + CI; GATE 1b
  margin_blind vs margin_b13 (and the underlying four Brier numbers); each
  guard's delta + CI; count of CI-clean better/worse families in the scan.
- Eval runtime, matches completed/skipped, relaxation triggers during the
  real eval (grep the run log), commit SHAs, `git diff --stat 55677fb..HEAD`.
- Anything that crashed, surprised you, or ran long.

## Easy to get wrong

1. **Do not re-run the blind arm.** The d16 s46 detail is the pre-verified
   pair. If (and only if) one of the three baseline facts fails your
   re-check, STOP and report — do not improvise a twin run.
2. **Seed 46**, not 42/43/44 — pairing is same-seed vs the d16 detail.
3. The damping must key on **prior appearances ≥1 AND prior balls == 0**
   as-of the sim date (via `player_sums`) — NOT on career totals over the
   whole corpus (temporal integrity).
4. `models/bowler_phase_usage.json`, `models/xgb_i7_noweights_production/`,
   `scripts/sim_eval/*`, `data/golden/` are all untouchable. The sidecar
   json under `models/auto/b13/` is the ONLY usage payload you create.
5. The fit cutoff `2025-07-01` applies to the CONSTANTS (k_damp, mu_active);
   the per-player (n, b) lookup at sim time is naturally as-of via bisect.
6. `--n-matches all` — the default is 30 and would silently give you a
   useless n=30 run.
7. Gate script committed before any eval output exists (loop discipline;
   verifiable from git history).
8. You do NOT: decide the verdict, revert anything, touch
   `research/results.tsv` or `research/IDEAS.md`, `git push`, or start any
   other idea.
