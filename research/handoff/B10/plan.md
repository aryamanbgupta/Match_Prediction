# B10 executor plan — sim who-bowls usage alignment (debutants + non-bowlers)

Idea id: **B10** (P2, claimed 2026-07-31T02:13:24Z, claim commit `91be8d7`).
Full idea text: `research/IDEAS.md` § B10. Read `program.md` first and obey
its DO NOT CHEAT section in full.

## Hypothesis

B9 proved (CI-clean) that the calibrated sim's `top_bowler` probabilities
lose to an as-of usage-share fair baseline: sim − usage dBrier
**+0.0038 [+0.0026, +0.0051]**, head-only +0.0049. Both exposed distortions
are WHO-BOWLS errors in `EmpiricalBowlerSelector`, concentrated in one code
branch — players **absent** from `models/bowler_phase_usage.json` (usage is
None) all get the same flat floor weight α = k·league_share:

- (a) true debutants (0 prior XI appearances) are under-bowled ~5× (sim
  prices them 1.63% vs 9.06% lineup-uniform / 8.47% actual);
- (b) veteran never-bowlers (keepers/pure batters, many appearances, 0
  career balls) are over-bowled (sim 1.29% vs 0.62% actual).

Fix: for that branch only, replace the flat α with a weight matched to
B9's as-of expected-balls share (EB-shrunk toward the lineup-uniform
prior). Debutants rise toward ~9% share; veteran never-bowlers fall
toward ~0. Players present in phase-usage are untouched.

## Gate (sim pair — decided by the ORCHESTRATOR, not you)

- **GATE 1 (primary):** `top_bowler` Brier improves CI-clean paired
  (b10 − blind, cluster-boot by match, CI < 0) AND G5 bowler coverage in
  the b10 arm ≥ 90% (defined below).
- **GATE 2 (guards):** no CI-clean regression on `bowler_wkts_1plus`,
  `bowler_wkts_2plus`, `batter_runs_mae`, `team_first_over_mae`.
- Both → LANDED; exactly one → TABLED; none → FAILED. You do NOT issue
  the verdict.

Baseline row context (results.tsv): B10 is a sim-gate idea — the paired
baseline is the fresh blind twin you will produce in-session (see
"Orchestrator adaptations"), not an LL/ROI row. The D15 row (2026-07-21)
documents the engine-canonical lineage.

## Orchestrator adaptations (binding)

1. **Twin fresh runs, NOT the D15 detail.** `sim_v1_2.py` was refactored
   after D15 by interactive commits `7f159a5`, `f846484` (I5 legal/off-bat
   stack: `update()`/`process_ball` signature rework, gated on
   `delivery_semantics == 'legal_off_bat_v1'`) and `f766476` (I9 ELO
   version plumbing). Intended-inert on the legacy path but never verified
   at n=261, so `models/auto/d15/detail_d15_s43_n261.json` is not a
   trustworthy paired baseline. Run BOTH arms fresh at seed 43 (B6
   precedent): blind (current default path) and b10 (opt-in sidecar).
   Only delta between arms = the selector weighting for usage-absent
   players.
2. **Scope: only the `usage is None` branch changes.** Both B9-quantified
   defects live there (phase-usage only contains players who have bowled).
   Do NOT touch the weighting of players present in phase-usage — the
   idea's "or thin" extension is explicitly scoped OUT (no quantified
   target; keep the diff minimal).

## Implementation (exact)

Repo: `/Users/aryamangupta/CricML/Match_Prediction`. All Python via
`uv run`. Artifacts to `models/auto/b10/`; raw logs to
`research/handoff/B10/raw/`. NEVER edit `scripts/sim_eval/`,
`scripts/parsing_v2.py`, stats backends, `models/bowler_phase_usage.json`,
`models/xgb_v3/`, `data/golden/`, production caches.

### 1. Sidecar usage artifact — `scripts/auto/b10_build_usage_sidecar.py`

- Copy `models/auto/b9/usage_corpus.pkl` → `models/auto/b10/usage_corpus.pkl`
  (provenance: rebuildable via `scripts/auto/b9_usage_baseline.py`).
- Write `models/auto/b10/bowler_phase_usage_b10.json` = deep copy of
  `models/bowler_phase_usage.json` plus one NEW top-level key:
  ```json
  "b10_asof_usage": {"corpus_path": "models/auto/b10/usage_corpus.pkl",
                     "k_usage": 5.0, "min_eligible": 5, "min_share": 0.01}
  ```
  `k_usage` MUST equal B9's `K_USAGE` (5.0 — verify by reading the
  constant from `b9_usage_baseline.py`, don't hardcode blindly).
- md5 `models/bowler_phase_usage.json` before and after; assert unchanged.

### 2. Selector change — `scripts/sim_v1_2.py` (`EmpiricalBowlerSelector`)

Activation: in `_load()`, if the payload contains `b10_asof_usage`, load
the corpus pkl once (module-level cache alongside `_BOWLER_USAGE_CACHE`)
and build an as-of accessor; else `self._b10 = None` and every code path
below MUST be behaviorally identical to current HEAD (the blind arm and
production both run with the default `models/bowler_phase_usage.json`,
which has no such key).

As-of accessor: implement a compact `_B10AsOfExpBalls` class inside
`sim_v1_2.py` replicating B9's `AsOfUsage.global_stats` + `player_sums` +
the exp_balls formula EXACTLY:
```
exp_balls = (k_u * prior_balls + sum_balls) / (k_u + n)  if n else prior_balls
prior_balls = total_corpus_balls_before_date / n_rows_before_date
```
(strictly-before via `bisect_left` on ISO date strings; cold-start
fallback 120/11). The B9 corpus is keyed by cricsheet **NAME** — use
`player.name`. The phase-usage payload is keyed by **player_id** — keep
using `player.player_id` there. Do not mix the keys.

`select_bowler` when b10 active:
- Compute legacy weights for all `available` exactly as now.
- Partition available: known (usage non-None) keep legacy weight
  `phase_balls + alpha`; unknown (usage is None) get share-matched weight:
  `s_i = exp_balls(name, match_date.isoformat()) / 120.0`,
  `w_i = s_i * W_known / max(1 - S_unknown, 0.05)` where `W_known` = sum
  of known legacy weights and `S_unknown` = Σ s_i over unknown available.
  If `W_known == 0` (all-unknown XI): `w_i = s_i` directly. Floor every
  weight at 1e-9.
- If `state.match_date` is None: fall back to legacy weights for that
  call (can't do as-of).
- **≥5-eligible relaxation (pre-committed):** per (match_date, lineup)
  compute full-XI shares under the new weighting once; if fewer than 5 XI
  members have share > `min_share` (0.01), restore the legacy α floor for
  ALL unknowns of that lineup and log
  `B10 relaxation triggered for <team>`. Count and report triggers
  (expect ~0).
- **Memoize** the per-(match_date, lineup player-ids, phase) weight
  vectors so runtime stays ≈ baseline (~8.6 s/match; select_bowler runs
  ~1M times per eval).
- Startup print once when active:
  `B10 usage-aligned bowler selector ACTIVE (k_u=5.0)` — the run logs are
  the proof of which arm ran which selector.

### 3. Pre-run unit check — `scripts/auto/b10_unit_check.py` (ALL must pass)

1. `d15_unit_check.py` still passes 30/30 on HEAD (subprocess). If it
   FAILS: STOP everything and report — that is a blocking engine-state
   finding (I5 refactor broke the legacy path), not a B10 failure.
2. Legacy parity: with the default usage path, the selector object has
   `_b10 is None` and produces weights EXACTLY equal (float-exact) to the
   formula `phase_balls + k*league_share` recomputed independently from
   the payload, on ≥20 real lineups from `data/polymarket_test`.
3. exp_balls parity: `_B10AsOfExpBalls` matches
   `b9_usage_baseline.AsOfUsage` (import it) on ≥1,000 sampled
   (name, date) pairs — exact to 1e-12 (both exp_balls and prior_balls).
4. Weight-mechanism table (deterministic, no sim) over the 261 test
   lineups: mean full-XI share old vs new for (a) true debutants
   (0 prior corpus appearances) — expect tiny → ≈9%; (b) veteran
   never-bowlers (≥20 appearances, 0 career balls) — expect α-share →
   ≪1%; (c) known bowlers — expect unchanged shares up to
   renormalization. Print the table verbatim (this is the idea's
   "before/after balls-bowled share" pre-run check at the weights level).
5. md5 of `models/bowler_phase_usage.json` unchanged.

### 4. Gate script — `scripts/auto/b10_gate_analysis.py` (COMMIT BEFORE ANY EVAL RESULT)

Clone the pairing machinery from `scripts/auto/d15_gate_analysis.py`
(paired per-match deltas, cluster-boot 2000 draws, boot seed 29 —
identical construction). Inputs: the blind + b10 detail JSONs.

- GATE 1: `top_bowler` paired dBrier (b10 − blind) with CI; G5 coverage =
  fraction of real-match bowlers (players with ≥1 delivery in the actual
  `data/polymarket_test` match JSON) whose `bowler_wkts_1plus` p_sim > 0
  in the b10 detail — report for BOTH arms, gate on b10 ≥ 0.90.
- GATE 2: paired deltas + CIs for `bowler_wkts_1plus`, `bowler_wkts_2plus`,
  `batter_runs_mae`, `team_first_over_mae`.
- Print the verdict mapping (both/one/none) but do NOT decide.
- Context (non-gate, report verbatim): full family scan (all families,
  flag every CI-excludes-0 delta either direction); recomputed sim−usage
  top_bowler margin on the b10 detail via the B9 pairing (how much of
  +0.0038 closed); relaxation trigger count.
- Drift check (non-gate, descriptive): paired deltas blind-arm vs
  `models/auto/d15/detail_d15_s43_n261.json` (same seed 43, pre/post the
  I5 refactor) for the 6 gate/guard families — tells the orchestrator
  whether the refactor moved the legacy path.

### 5. Commit BEFORE running the eval

`Auto[B10]: implement — <one line>` covering: sidecar builder, selector
change, unit check, gate script. (`models/auto/` is gitignored — expected.)

### 6. Evals — sequential, synchronous, never concurrent (16 GB box)

Arm A (blind twin):
```
uv run python scripts/sim_eval/prop_backtest.py \
  --test-dir data/polymarket_test --n-matches all --n-sims 100 --seed 43 \
  --ball-calibrator vector \
  --detail-out models/auto/b10/detail_blind_s43_n261.json \
  --report-out models/auto/b10/report_blind_s43_n261.md \
  2>&1 | tee research/handoff/B10/raw/run_blind.log
```
Arm B (b10): same command with additionally
`--bowler-usage-path models/auto/b10/bowler_phase_usage_b10.json` and
`blind→b10` in both out paths and the log name.

- Startup lines MUST show, both arms: `venue encoder ACTIVE (467 venues)`,
  `Bowler selector: empirical`, `Run-out dismissal channel ACTIVE`,
  vector calibrator. Arm B additionally MUST show the B10-ACTIVE line;
  Arm A MUST NOT. If wrong, kill and fix before burning 40 minutes.
- Each run ≈ 35–45 min (~8.6 s/match × 261). Launch detached (nohup) with
  the tee to the raw log, then poll the log's `[k/261]` progress inside
  this session until completion — do NOT end your run with an eval still
  going (D2 lesson: children are reaped at session close). Kill at 90 min
  (2× budget) and report CRASH facts.
- Run `b10_unit_check.py` once more right before Arm B (re-verify at eval
  time, D3 discipline).

### 7. After both evals

Run `b10_gate_analysis.py`, tee to
`research/handoff/B10/raw/gate_output.txt`.

## result.md (write to `research/handoff/B10/result.md`)

Numbers copied VERBATIM from tool output — never rounded or paraphrased:
- GATE 1: top_bowler dBrier + CI; G5 coverage both arms.
- GATE 2: the four guard deltas + CIs.
- Context: family scan CI-clean list; sim−usage margin recomputation;
  relaxation count; drift check vs D15 detail.
- Unit-check outputs (d15 30/30, parity checks, the weight table).
- Commit SHAs you created; `git diff --stat` vs claim commit `91be8d7`;
  eval wall times; anything that crashed or ran long.

## What you must NOT do

- Do not decide the verdict; do not revert anything; do not touch
  `research/results.tsv` or `research/IDEAS.md`; do not `git push`; never
  `git reset`; do not start a second idea.
- Never edit `scripts/sim_eval/`, `parsing_v2.py`, `stats_provider.py`,
  `stats_sqlite_backend.py`; never touch `data/golden/`,
  `models/xgb_v3/`, `models/xgb_match_v3_m7*`, production caches, or
  `models/bowler_phase_usage.json` itself.
- Do not modify the default (non-b10) selector behavior — when the b10
  key is absent the code path must be behaviorally identical to HEAD.
- No new dependencies; no `--parallel`; one heavy process at a time.

## Easy to get wrong

- The B9 corpus keys on cricsheet NAME; phase-usage keys on player_id.
- ISO date strings compare lexicographically — use
  `match_date.isoformat()`; strictly-before = `bisect_left`.
- `exp_balls` when n=0 must be exactly `prior_balls` (the `if n else`
  branch), not the shrunk formula with n=0.
- `prior_balls` = total corpus balls / number of appearance ROWS (not
  distinct players) strictly before the date — replicate B9, don't
  re-derive.
- Memoize or the eval will crawl; `player_sums` does a bisect + list-sum
  per call and select_bowler runs ~1M times.
- The share-matching denominator `max(1 - S_unknown, 0.05)` guards
  degenerate all-debutant XIs; keep it.
- Weights must stay strictly positive (1e-9 floor) or the sampler's
  defensive `random.choice` branch can trigger.
