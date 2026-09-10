# Stage 0 readiness record

Recorded 2026-09-10 on the laptop, branch `embeddings-ladder` at commit
`cd62991` plus the uncommitted stage 0 files. Checks are numbered per
`docs/sequence_track/stage0_acceptance.md` § D3. Outputs are pasted verbatim
from the captured files; nothing here is a model result.

Qualification of the "no sealed read" rule: `artifacts.py verify` hashes every
manifest role, including `odds_golden_v2`, `golden_set_v2`, `forward_holdout`
and `forward_state`. That is byte-level integrity hashing of files on disk, not
an evaluation, a fit, or a selection input; no stage 0 model, scorer, or
runner opened a sealed path (the runners and scorers refuse them via
`reject_sealed`). Smoke evaluator logs, which print per-match metrics, were
redirected to `run.log` files under the gitignored smoke dirs and were only
grepped for banners, exit stamps and timing lines; no metric line was read.

## 3.1 `uv run --no-sync python scripts/artifacts.py verify`

```
Thu Sep 10 19:19:55 IST 2026
ball_model_prod: OK
match_model_prod: OK
match_model_i7_hundred: OK
ball_encoders_i7_archive: OK
stats_cache_i7: OK
stats_cache_v3_legacy: OK
live_state_i7: OK
ball_frame_i7: OK
ball_frame_v3_legacy: OK
match_frame_i7_v2: OK
match_frame_i7: OK
bowler_phase_usage: OK
bowler_roster_policy: OK
bowler_usage_corpus_b10: OK
prop_fair_baseline_corpus_v2: OK
prop_fair_baseline_corpus_legacy: OK
match_winner_map: OK
odds_iteration_v2: OK
iteration_set_v2: OK
iteration_set_legacy: OK
odds_golden_v2: OK
golden_set_v2: OK
forward_holdout: OK
forward_state: OK
```

## 3.2 BR2 gate status (quoted from `research/reports/auto/BR2.md`)

| Gate | Required | Result |
|---|---|---|
| prop A/B | no CI-clean regression in any family; movements only where SIM1/SIM2/PROP3/engine changes fire | **PASS** (20 families CI-clean better, 8 tied, 0 regressions) |
| G1 | winner-market LL parity, empirical selector not worse than random by more than +0.002 | **PASS** |
| G3 | `top_batter` paired delta interval includes zero or favourable | **PASS** |
| G5 | bowler coverage ≥90% on the declared population | **PASS BY WRITTEN EXCEPTION (user, 2026-09-10)**; bar is ≥89% going forward |
| E2 | `highest_individual_mae` re-derived; restated either way | **RESTATED: parity** |

BR2's closing line: "All four mandatory gates pass (G5 by written exception)."

## 3.3 T1 replay lifecycle, snapshot guard, prefix cache, parity tests

```
uv run --no-sync pytest -q tests/test_run_sim_eval_t1_lifecycle.py tests/test_sim_t1_snapshot_guard.py tests/test_sim_t1_prefix_cache.py tests/test_sim_t1_parity.py
tests/test_sim_t1_prefix_cache.py::test_prefix_cache_desync_fails_closed
tests/test_sim_t1_prefix_cache.py::test_new_innings_cache_resets_positions
-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
23 passed, 4 warnings in 1.12s
```

## 3.4 `claim_gate` exercised on the HA0 (item 6 swap-smoke) evidence

Command (candidate and baseline = the five `sliced_50000.json` files per arm
under `experiments/results/item6_swap_smoke/`, seeds 7 13 29 42 101, odds role
`odds_iteration_v2`, cluster source `data/polymarket_test_v2`, output to the
session scratchpad):

```
Thu Sep 10 19:20:33 IST 2026
LANDED 23f840a5510eaeed3ff055f9de1981ab9b68455bb3d7b587a893309d2545f23f
exit=0
```

Regenerated verdict LANDED; committed `experiments/results/item6_swap_smoke/gate.json` verdict LANDED; the two payloads' `inputs` blocks compare equal. The scratch gate JSON was not copied into the repo.

## 3.5 `log_verdict` exercised end to end (scratch ledgers, dry run)

A synthetic `## HA0 [P1] [PENDING] …` entry was appended to a scratch copy of
`research/IDEAS.md`; `research/results.tsv` was copied to scratch. `claim`
wrote only the scratch copy; `verdict … --dry-run` wrote nothing.

```
--- verdict --dry-run (rerun with --result-text-file)
Thu Sep 10 19:21:04 IST 2026
--- HA0: [RUNNING 2026-09-10T13:50:56Z] -> [LANDED]
--- a/IDEAS_scratch.md
+++ b/IDEAS_scratch.md
@@ -2857,3 +2857,3 @@
 
-## HA0 [P1] [RUNNING 2026-09-10T13:50:56Z] Stage 0 gate-plumbing exercise on the item 6 swap-smoke evidence
+## HA0 [P1] [LANDED] Stage 0 gate-plumbing exercise on the item 6 swap-smoke evidence
 **Hypothesis:** synthetic entry used only to exercise claim_gate + log_verdict end to end in a scratch copy (stage 0 readiness check 3.5); never written to the real ledgers.
@@ -2861,2 +2861,2 @@
 **Gate:** the committed gate rule.
-**Result:** —
+**Result:** LANDED 2026-09-10 (stage 0 readiness check 3.5; scratch-only exercise of the verdict path on the item 6 swap-smoke gate JSON).

--- appending 1 row to <scratch>/results_scratch.tsv:
2026-09-10	HA0	cd62991	(sim-gate)	(sim-gate)	(sim-gate)	(sim-gate)	(sim-gate)	LANDED	stage0 readiness 3.5 dry run; scratch copies only gate_sha256=23f840a5510eaeed3ff055f9de1981ab9b68455bb3d7b587a893309d2545f23f

DRY RUN — nothing written.
exit=0
```

`git diff --stat -- research/results.tsv research/IDEAS.md` afterwards: empty.
Note for operators: `verdict` requires `--result-text-file` (the module
docstring example omits it).

## 3.6 Full suite

Run 2026-09-11 00:28–00:30 IST after every stage 0 file landed (post Astra round 3 fixes):

```
uv run --no-sync pytest -q
828 passed, 5 skipped, 9 warnings in 85.80s (0:01:25)

uv run --no-sync pytest -q --strict-markers -m "not needs_artifacts"
801 passed, 32 deselected, 9 warnings in 65.45s (0:01:05)
```

Baseline before stage 0 (item 5 acceptance, same day): artifact-free collection
414 ids; stage 0 adds `test_run_queue.py` (23), `test_train_a50.py` (12),
`test_run_arm.py`, `test_audit_cross_arm.py`, `test_pin_stage1.py` and
`test_manifest_defaults.py` (212 together), `test_score_realism.py` and `test_score_props.py` (35 together).

## 3.7 Machine

```
Thu Sep 10 19:20:44 IST 2026
hw.memsize=51539607552
hw.ncpu=15
/dev/disk3s5   926Gi   353Gi   533Gi    40%    2.4M  5.6G    0%   /System/Volumes/Data
Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                8646.
Pages active:                            995466.
Pages inactive:                          993582.
Pages speculative:                         2226.
Pages throttled:                              0.
available_gb(free+inactive+speculative)=15.31
```
