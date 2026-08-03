# D6 — executor result

**Status: BLOCKED at Step 2. No sim eval was run; no gate was evaluated; no verdict is
issued here.** The pre-committed gate machinery is implemented and committed, but the
idea's core operation — retraining the production ball model with `balanced` class
weights off — is **not executable on the production ball frame**, because that frame is
the pre-I7 legacy venue identity and `scripts/xgboost_v2.py` fail-closes on it.

Claim commit `a2188b0`; plan commit `7171c8f`.
Commits created by this executor:

| SHA | message |
|---|---|
| `5fb16bb` | `Auto[D6]: implement — no-class-weights trainer flag + marginal audit + pre-committed d6 gate (tail-pool primary, seed 44)` |
| (this doc) | `Auto[D6]: executor result — ...` |

`git diff --stat a2188b0 5fb16bb`:

```
 research/handoff/D6/plan.md                   | 209 ++++++++++++++++++++++
 research/handoff/D6/raw/md5_xgb_v3_before.txt |   8 +
 scripts/auto/d6_gate_analysis.py              | 226 ++++++++++++++++++++++++
 scripts/auto/d6_marginal_audit.py             | 240 ++++++++++++++++++++++++++
 scripts/xgboost_v2.py                         |  40 ++++-
 5 files changed, 717 insertions(+), 6 deletions(-)
```

---

## Step 1 — implement (DONE, committed before any eval)

1. `scripts/xgboost_v2.py` — two opt-in flags, defaults byte-preserving:
   - `--model-dir <path>`: overrides the output dir; `artifact_suffix` still `v3`.
   - `--no-class-weights`: omits the `sample_weight` kwarg entirely at BOTH fit sites
     (final fit and the optuna objective); the weight computation is untouched; banner
     `[D6] class weights DISABLED (uniform sample weights)` when active, and
     `Class weights ACTIVE (balanced sample weights passed to fit)` when not.
2. `scripts/auto/d6_marginal_audit.py` — copy-adapted from
   `scripts/e5_teacher_forced_bias.py` (which was **not** edited). Takes `--model-dir`,
   scores the v3 test parquet teacher-forced under two input distributions
   (`venue_on` = real training-time venue codes, today's default sim path after B1;
   `venue_zero` = the pre-B1 sim input the deployed v1 calibrator was fit under), emits
   per-class predicted-vs-actual marginals, Δruns/ball, ΔP(wicket), test multiclass LL
   and an explicit `"pass"` against |ΔP(wkt)| ≤ 0.005 and |Δruns/ball| ≤ 0.05.
3. `scripts/auto/d6_gate_analysis.py` — imports the pairing machinery verbatim from
   `b12_gate_analysis` (paired per-row delta, cluster bootstrap **by match**,
   `N_BOOT=2000`, `seed=29`, rows matched on `(team, name)` with a positional
   cross-check). Implements GATE 1(a) (reads `marginal_audit.json`), GATE 1(b) (pooled
   tail row-pool dBrier over `pp_total_ou_45_5/50_5/55_5` + `bowler_wkts_1plus`),
   GATE 1(c) (`batter_runs_mae` no CI-clean regression), GATE 2 guards (`top_bowler`,
   `team_first_over_mae`), the full family scan, and prints the both/one/none mapping.
   It decides nothing.

All three compile; `d6_gate_analysis` imports clean. **It was never run** — its d6 input
JSON does not exist.

## Step 2 — train d6 (FAILED, fail-closed in 1 second)

Config-json extracted programmatically from `run_experiment.build_training_cmd` on
`experiments/configs/xgb_v6_hierarchical_shrink.yaml` (zero hand-transcription; saved to
`research/handoff/D6/raw/config_json.txt`, 114 features resolved, matching v7).

Command run (`research/handoff/D6/raw/train_d6.log`), start `2026-07-31T20:18:19Z`,
exit `rc=1` at `2026-07-31T20:18:20Z` — **1 second, before any data was loaded into the
model**:

```
[D6] train start 2026-07-31T20:18:19Z
Loading split datasets (v3)...
Traceback (most recent call last):
  File "/Users/aryamangupta/CricML/Match_Prediction/scripts/xgboost_v2.py", line 70, in <module>
    assert_venue_alias_contract(
  File "/Users/aryamangupta/CricML/Match_Prediction/scripts/identity_maps.py", line 159, in assert_venue_alias_contract
    raise RuntimeError(
RuntimeError: ball training parquet venue-alias contract mismatch: {'venue_alias_version': (None, 'venue_aliases_v1'), 'venue_alias_sha256': (None, '853b32b0ce3098dd8c0f33ba1437846f5505d50d9a425fbd37bff9c9f76745d8'), 'venue_alias_active_count': (None, 94)}. Rebuild the artifact from the active identity map.
[D6] train exit rc=1 2026-07-31T20:18:20Z
```

`models/auto/d6/` contains no model artifacts. The sidecar byte-identity check and the
`models/xgb_v3` before/after md5 check both reduce to "nothing was written":

- `research/handoff/D6/raw/md5_xgb_v3_before.txt` vs `md5_xgb_v3_after.txt`: **identical**
  (`diff` clean). `xgboost_model_v3.pkl` = `d448822e9bcd8cb93c126f8558b8ab46` before and
  after. Production untouched.

### Root cause (verified, not inferred) — `research/handoff/D6/raw/frame_diagnosis.log`

`data/xgb_data_v3/.feature_hash` predates the I7 venue identity contract
(commits `11a5304` / `5175f6f`, 2026-07-25). Its keys are
`['gender_filter','hash','k_player','k_venue','n_features','splits','version']` — no
`venue_alias_*` fields at all. The trainer's two data asserts split:

- `assert_elo_update_version` **passes** (missing metadata resolves to the frozen
  baseline `fixed_competition_k_v1`, which is what the config expects).
- `assert_venue_alias_contract` **fails closed** — and it is *correct* to fail. The v3
  ball parquet is genuinely on the legacy frame:

| | value |
|---|---|
| distinct venue strings in the v3 ball parquet | **467** |
| renamed by the ACTIVE alias map | **95** |
| distinct venues AFTER canonicalization | **373** |
| fold groups (≥2 raw strings collapsing to one canonical) | **92** |
| `models/xgb_v3/venue_encoder_v3.pkl` classes | **467** (legacy) |
| `models/xgb_i7/venue_encoder_i7.pkl` classes | **373** (I7) |
| `models/xgb_i8/venue_encoder_i8.pkl` classes | **373** (I7) |
| `models/xgb_v3/training_contract_v3.json` | **absent** |
| `models/xgb_i7/training_contract_i7.json` | present, `venue_alias_active_count: 94` |

Three-way folds included (`M Chinnaswamy Stadium` / `M Chinnaswamy Stadium, Bengaluru` /
`M.Chinnaswamy Stadium` → one canonical; likewise the two `Dr. Y.S. Rajasekhara Reddy
ACA(-)VDCA` spellings plus the `, Visakhapatnam` form).

**So the ball stack is now in exactly the position CLAUDE.md already documents for the
pre-D12 match model: the deployed artifact lives on a legacy identity contract, and new
training on that frame is deliberately impossible.** `models/xgb_v3` — the model behind
the B12 baseline detail, the `venue encoder ACTIVE (467 venues)` banner the plan
pre-commits to, and every current prop/sim number — can no longer be retrained under any
configuration change, D6's or otherwise.

### Why I did not substitute a different frame

`data/xgb_data_i7/` exists (same feature hash `c520a3ba08ae`, same 114 features, same
splits, WITH the contract), and `models/xgb_i7/` is its balanced-weights twin. Training
D6 there was rejected as out of plan scope, because it would have changed four things at
once relative to the pre-committed B12 baseline instead of one:

1. class weights (the idea), 2. ball calibrator (planned asymmetry), 3. the training
venue-identity frame (467 → 373), and 4. the stats provider, since the arm would need
`--stats-version i7` against `models/player_stats_cache_i7.sqlite`.

It would also have violated the plan's own pre-committed startup check
(`venue encoder ACTIVE (467 venues)`; an i7 model prints 373). The resulting pooled-tail
delta would not have been a class-weights measurement. Re-materializing
`data/xgb_data_v3/` in place was rejected outright: it overwrites a production input that
`models/xgb_v3` is already trained against, and would desynchronize the deployed model
from its own data. Both are orchestrator calls, not executor calls.

## Step 3 — marginal audit

**GATE 1(a) has no recorded value.** `models/auto/d6/marginal_audit.json` does not exist
and was never written, because there is no D6 model to audit.

The audit tooling *was* exercised against the production model as a read-only reference
(`--model-dir models/xgb_v3 --out models/auto/d6/marginal_audit_prod_reference.json`,
log `research/handoff/D6/raw/marginal_audit_prod_reference.log`, ~2 min, n = 186,667
balls). This is context for the orchestrator, **not** a gate value.

**It reproduces `reports/e5_class_weight_fix.md` exactly under the input distribution
that report used (`venue_zero`)**, which validates the tooling:

| arm (venue_zero) | ΔP(wicket) | Δruns/ball | test LL | E5 published |
|---|---:|---:|---:|---|
| `xgb_v3` raw | **+0.06470** | **+0.3829** | **1.6076** | +0.065 / +0.383 / 1.608 |
| `xgb_v3` + v1 vector | **−0.00160** | **+0.0237** | **1.5197** | −0.002 / +0.024 / 1.520 |

`venue_zero` + v1 vector is the only arm that **PASSES** the D6 tolerance
(|ΔP(wkt)| ≤ 0.005, |Δruns/ball| ≤ 0.05).

**New, previously unrecorded finding — the deployed calibrator is out of its fit
distribution.** B1 landed the venue encoder, so the default sim path now feeds real venue
codes (`venue_on`), while `vector_scaling_calibrator_v1.pkl` was fit under
`venue_zero`. Scored on `venue_on`:

| arm (venue_on = today's default sim path) | ΔP(wicket) | Δruns/ball | test LL | tolerance |
|---|---:|---:|---:|---|
| `xgb_v3` raw | +0.07700 | +0.4378 | 1.6297 | FAIL (as expected) |
| `xgb_v3` + v1 vector | **+0.00576** | **+0.0814** | 1.5090 | **FAIL on both** |

Per-class, venue_on + v1 vector: dot −0.00925, one −0.02256, two +0.01177, four +0.00267,
six +0.01162, wicket +0.00576. The residual runs/ball tilt is **+0.0814**, i.e. ~19% of
the raw overshoot (+0.4378) survives, versus ~6% (+0.0237 of +0.3829) in the distribution
the calibrator was fit on — a ~3.4× larger residual, and enough to miss the tolerance the
calibrator itself set. Note the raw booster is also *worse* on `venue_on` than
`venue_zero` (+0.4378 vs +0.3829 runs/ball; LL 1.6297 vs 1.6076), so this is a
model-plus-calibrator interaction with the venue input, not calibrator arithmetic alone.

Full per-class tables for all four arms: `models/auto/d6/marginal_audit_prod_reference.json`.

## Steps 3b / 4 / 5 — not reached

No attribution control, no sim eval, no gate analysis. `models/auto/d6/` holds only
`marginal_audit_prod_reference.json`. No detail/report JSONs were produced, so no number
in this document is a D6 gate value.

## Housekeeping

- Nothing under `models/xgb_v3/`, `data/golden/`, `scripts/sim_eval/`,
  `scripts/parsing_v2.py`, `scripts/stats_provider.py`,
  `scripts/stats_sqlite_backend.py`, or `scripts/e5_teacher_forced_bias.py` was modified.
- No background processes remain. Nothing crashed except the fail-closed trainer above;
  nothing ran long (total wall clock ≈ 4 min of compute).
- Raw evidence: `research/handoff/D6/raw/{train_d6.log, config_json.txt,
  frame_diagnosis.log, marginal_audit_prod_reference.log, md5_xgb_v3_before.txt,
  md5_xgb_v3_after.txt}`.

## For the orchestrator

Two decisions are now unblocked, neither of which the executor may take:

1. **The ball-model retrain path is dead on the v3 frame.** Any future idea that retrains
   the ball model (D6 and every successor) needs a ruling: re-baseline the whole sim/prop
   stack onto the I7 frame (`data/xgb_data_i7` + `models/xgb_i7` +
   `--stats-version i7`, which means a fresh baseline detail JSON at the chosen seed
   before any arm is gated), or keep serving `models/xgb_v3` as a frozen legacy-contract
   artifact and stop queueing ball-retrain ideas. D6's hypothesis is untested either way
   and should stay in the queue.
2. **The deployed v1 vector calibrator is measurably stale** on the venue-ON path it is
   actually deployed on (+0.0814 runs/ball, +0.00576 P(wicket) — both outside D6's own
   tolerance, which the same calibrator meets on `venue_zero`). Refitting it on
   `venue_on` teacher-forced validation predictions is an eval-only, no-retrain change
   that is executable today on the legacy frame, and is a strictly cheaper probe of the
   same "post-hoc patching is fragile" question D6 was asking.
