# D16 — executor result

Executor for `research/handoff/D16/plan.md`. **No verdict is issued here** —
the gate script's mapping line is reproduced verbatim below and the
orchestrator decides.

Claim commit `ba9cd3f`, plan commit `6b58ca7`.
Commits created by this executor:

| SHA | message |
|---|---|
| `ea4acdb` | `Auto[D16]: implement — recover D6 trainer flags + i7 marginal audit + control-val vector calibrator fit + pre-committed d16 gate (twin design, seed 46)` |
| (this commit) | `Auto[D16]: executor result — ...` |

`git diff --stat ba9cd3f` (working tree, before the result commit):

```
 research/handoff/D16/plan.md              | 310 ++++++++++++++++++++++++++++++
 scripts/auto/d16_fit_vector_calibrator.py | 162 ++++++++++++++++
 scripts/auto/d16_gate_analysis.py         | 245 +++++++++++++++++++++++
 scripts/auto/d16_marginal_audit.py        | 270 ++++++++++++++++++++++++++
 scripts/xgboost_v2.py                     |  40 +++-
 5 files changed, 1021 insertions(+), 6 deletions(-)
```

Nothing crashed. Nothing ran long (all four heavy steps came in well under
budget). No background process is left running.

---

## Headline

| gate | statistic | result |
|---|---|---|
| GATE 1(a) | no-weights teacher-forced marginal audit, frozen before any sim eval | **PASS** (dP(wicket) **+0.00319**, tol ±0.005; d runs/ball **+0.01093**, tol ±0.05) |
| GATE 1(b) | pooled tail dBrier, 4673 rows | **−0.0116  [−0.0159, −0.0073]  DOWN(better)** → PASS |
| GATE 1(c) | `batter_runs_mae` dMAE | **−0.5449  [−0.6714, −0.4142]  DOWN(better)** → PASS |
| GATE 2 | `top_bowler` dBrier | **−0.0004  [−0.0014, +0.0004]  ~noise** → ok |
| GATE 2 | `team_first_over_mae` dMAE | **−0.0898  [−0.2008, +0.0173]  ~noise** → ok |

Gate script's final lines, verbatim:

```
GATE 1: MET | GATE 2: MET
Pre-committed verdict MAPPING (orchestrator decides): LANDED
```

---

## Step 1 — implementation (commit `ea4acdb`, before any training/eval output)

1. `git checkout 5fb16bb -- scripts/xgboost_v2.py`. Verified the recovered
   delta equals the 5fb16bb patch exactly:
   `diff <(git diff 2a815e2 -- scripts/xgboost_v2.py) <(git diff 2a815e2 5fb16bb -- scripts/xgboost_v2.py)`
   → empty, printed `DELTA IDENTICAL TO 5fb16bb PATCH`. `git diff HEAD --stat`
   showed only `scripts/xgboost_v2.py | 40 ++++++++++++++++++++++++++++++------`
   (the `--model-dir` / `--no-class-weights` flags, the two `_weight_kwargs`
   fit sites, and the banner prints). Nothing else appeared.
2. `scripts/auto/d16_marginal_audit.py` — copy-adapted from
   `git show 5fb16bb:scripts/auto/d6_marginal_audit.py`. Re-pointed
   `BALL_DIR data/xgb_data_v3 → data/xgb_data_i7`,
   `PROD_DIR models/xgb_v3 → models/xgb_i7` (read-only context arm),
   `--suffix` default `v3 → i7`; the reference arm is now scored with **its
   own** encoders rather than the audited arm's; added an optional
   `--calibrator` context arm. The legacy `PROD_CALIBRATOR`
   (`vector_scaling_calibrator_v1.pkl`) is gone — it does not exist on the i7
   frame. `grep -n '_v3'` over all three new scripts returns matches **only
   inside docstrings** (documenting the re-pointing), never in a path
   constant.
3. `scripts/auto/d16_fit_vector_calibrator.py` — copy-adapted from
   `scripts/fit_i5_ball_calibrator.py`, which could **not** be used unmodified:
   it hard-refuses any `training_contract` whose `delivery_semantics` is not
   `legal_off_bat_v1`, and i7 is `inclusive_total_runs_v1`
   (`raise RuntimeError("refusing to fit I5 calibrator for non-I5 training
   contract")`). `scripts/fit_i5_ball_calibrator.py` and
   `scripts/calibration.py` were NOT edited.
   `calibration._apply_encoders_to_df` **does** handle the venue column
   (`'venue_encoded': ('venue', 'venue_encoder')`), and the new script
   cross-checks its own explicit encoding against it on all four encoded
   columns (see Step 3 log — all four `identical`).
4. `scripts/auto/d16_gate_analysis.py` — copy-adapted from
   `git show 5fb16bb:scripts/auto/d6_gate_analysis.py`; same pairing machinery
   imported verbatim from `b12_gate_analysis` / `a8_gate_analysis`
   (`N_BOOT=2000`, `BOOT_SEED=29`, cluster bootstrap by match), same gate
   definitions, new default input paths. It prints the mapping and does not
   decide.

---

## Step 2 — twin trainings

Config JSON extracted programmatically (zero hand-transcription) by importing
`build_training_cmd` from `scripts/run_experiment.py`, calling it on the loaded
`experiments/configs/xgb_i7_venue_identity.yaml` (with
`resolve_feature_list` → 114 features) and taking the element after
`--config-json`. `build_training_cmd` returned no extra args. Saved verbatim to
`research/handoff/D16/raw/config_json.txt`:

```
{"features": {"groups": ["basic", "player_stats", "h2h", "momentum", "pressure", "chase", "medium", "player_metadata", "matchup", "type_based", "team_strength", "batter_outcome_dist", "bowler_outcome_dist", "batter_vs_type_dist", "bowler_vs_hand_dist", "venue_outcome_dist"], "exclude": [], "include_extra": []}, "model": {"hyperparameters": {"n_estimators": 444, "max_depth": 10, "learning_rate": 0.24036372383981375, "subsample": 0.8776663421127178, "colsample_bytree": 0.7424085095268674, "reg_alpha": 0.8503122682099661, "reg_lambda": 0.18186045420525845}}, "outcome_dist": {"k_player": 30.0, "k_venue": 200.0}, "data": {"version": "i7", "cache_schema_version": 4, "delivery_semantics": "inclusive_total_runs_v1", "elo_update_version": "fixed_competition_k_v1", "source_dir": "data/t20s_json", "gender_filter": "male", "splits": {"train_end": "2024-12-31", "val_end": "2025-06-30", "test_end": "2026-04-16", "golden_start": "2026-04-17"}}}
```

`--model-dir` was passed on BOTH invocations. Both arms loaded
`Train: 1876971 balls / Validation: 124292 balls / Test: 186667 balls` and
`[config-json] Using 114 features from experiment config`. The
`assert_venue_alias_contract` fail-close that killed D6 did **not** fire on
this frame.

Banner lines (verbatim):

```
control    Class weights: {0: '0.55', 1: '0.40', 2: '2.19', 3: '1.55', 4: '3.67', 5: '3.08'}
control    Class weights ACTIVE (balanced sample weights passed to fit)
noweights  Class weights: {0: '0.55', 1: '0.40', 2: '2.19', 3: '1.55', 4: '3.67', 5: '3.08'}
noweights  [D6] class weights DISABLED (uniform sample weights)
```

| arm | wall clock | last verbose row | best_iteration | trees in booster | model bytes | final val LL | final test LL |
|---|---|---|---|---|---|---|---|
| control | ~7 min | `[443] validation_0-mlogloss:0.93923  validation_1-mlogloss:1.63759` | 443 | 444 | 134,512,481 | 1.6376 | 1.6316 |
| noweights | ~2 min | `[123] validation_0-mlogloss:1.14821  validation_1-mlogloss:1.44766` | 24 | 125 | 37,845,224 | 1.4334 | 1.4253 |

The tree-count divergence is the anticipated consequence of early stopping
under a different loss surface (plan: "Early stopping will pick a different
effective tree count with weights off — that is part of the structural change,
not a bug"). Both the sim wrapper (`sim_v1_2.py:1609`) and the audit script
call `model.predict_proba` on the loaded sklearn `XGBClassifier`, so
xgboost 2.1.4 applies `best_iteration` identically in both places.

### HARD CHECK — the two arms' sidecars are byte-identical to each other

```
batter_encoder_i7.pkl            control=a63af729f48492f701d1df5755493b01 noweights=a63af729f48492f701d1df5755493b01  IDENTICAL
bowler_encoder_i7.pkl            control=96e76ad9bde782bc9a5df111f1b713fa noweights=96e76ad9bde782bc9a5df111f1b713fa  IDENTICAL
venue_encoder_i7.pkl             control=bcbb6e3fb60e08df50d752ea282c0009 noweights=bcbb6e3fb60e08df50d752ea282c0009  IDENTICAL
matchup_encoder_i7.pkl           control=264c11f6d8903347433017e8124a2ec1 noweights=264c11f6d8903347433017e8124a2ec1  IDENTICAL
feature_columns_i7.txt           control=3fd6629c6cf5be70873b92ab2e619119 noweights=3fd6629c6cf5be70873b92ab2e619119  IDENTICAL
outcome_dist_config_i7.json      control=bec7d99140b4ae5e98977e18ca6ab670 noweights=bec7d99140b4ae5e98977e18ca6ab670  IDENTICAL
HARD CHECK RESULT: PASS
```

The only-delta-is-weights contract holds. The two `training_contract_i7.json`
files are also identical to each other (same `data_version i7`,
`inclusive_total_runs_v1`, `fixed_competition_k_v1`, same row counts, same
`venue_aliases_v1` / sha256 `853b32b0…` / 94 active aliases).

### Determinism control (informational, never a stop)

```
xgboost_model_i7.pkl             control=32b81888407dcd5cd652eb72b4ad7725 xgb_i7=32b81888407dcd5cd652eb72b4ad7725  MATCH
batter_encoder_i7.pkl            control=a63af729f48492f701d1df5755493b01 xgb_i7=a63af729f48492f701d1df5755493b01  MATCH
bowler_encoder_i7.pkl            control=96e76ad9bde782bc9a5df111f1b713fa xgb_i7=96e76ad9bde782bc9a5df111f1b713fa  MATCH
venue_encoder_i7.pkl             control=bcbb6e3fb60e08df50d752ea282c0009 xgb_i7=bcbb6e3fb60e08df50d752ea282c0009  MATCH
matchup_encoder_i7.pkl           control=264c11f6d8903347433017e8124a2ec1 xgb_i7=264c11f6d8903347433017e8124a2ec1  MATCH
feature_columns_i7.txt           control=3fd6629c6cf5be70873b92ab2e619119 xgb_i7=3fd6629c6cf5be70873b92ab2e619119  MATCH
outcome_dist_config_i7.json      control=bec7d99140b4ae5e98977e18ca6ab670 xgb_i7=91b0b145cb4a0d1dc2361bb42f0b7af0  DIFFER
```

**The control booster reproduces the archived `models/xgb_i7` artifact
byte-for-byte** — a free, exact determinism control (I19 precedent), so no
teacher-forced |Δp| probe was needed. The single `DIFFER` is the k-sidecar,
and it is a pure schema addition, not a value change:

```
control/noweights: {"k_player":30.0,"k_venue":200.0,"k_phase":30.0,"k_h2h":60.0}
archived xgb_i7:   {"k_player":30.0,"k_venue":200.0}
```

The archived file predates the `k_phase`/`k_h2h` keys added to the sidecar
writer; `k_player=30.0` / `k_venue=200.0` — the values the sim wrappers read —
agree. Independent confirmation of the byte-match at the prediction level: the
`control raw / venue_on` and `xgb_i7 raw / venue_on` marginal-audit tables in
Step 4 are numerically identical to every printed digit.

### Protected artifacts untouched

`research/handoff/D16/raw/md5_protected_{before,after}.txt` are identical
(`diff` empty, printed `PROTECTED ARTIFACTS UNCHANGED`):

```
MD5 (models/xgb_i7/xgboost_model_i7.pkl) = 32b81888407dcd5cd652eb72b4ad7725
MD5 (models/xgb_v3/xgboost_model_v3.pkl) = d448822e9bcd8cb93c126f8558b8ab46
```

---

## Step 3 — fresh vector calibrator, fit on the CONTROL arm's validation predictions

`research/handoff/D16/raw/fit_calibrator.log(.txt)`, verbatim:

```
training contract: delivery_semantics='inclusive_total_runs_v1'  data_version='i7'  venue_identity='venue_aliases_v1'
validation parquet: data/xgb_data_i7/cricket_data_i7_validation.parquet  rows=124,292  features=114
encoding path: explicit training-time encoders from models/auto/d16/control (batter/bowler/matchup/venue)
  venue encoder ACTIVE (373 venues); distinct venue codes in val: 101
  cross-check _apply_encoders_to_df[batter_encoded]: identical
  cross-check _apply_encoders_to_df[bowler_encoded]: identical
  cross-check _apply_encoders_to_df[venue_encoded]: identical
  cross-check _apply_encoders_to_df[matchup_type_encoded]: identical

validation balls: 124,292
raw log loss:        1.637592
calibrated log loss: 1.511479
actual marginals:    [0.297284 0.40367  0.071598 0.11259  0.059143 0.055716]
raw marginals:       [0.225698 0.249412 0.121795 0.16356  0.10701  0.132525]
calibrated marginal: [0.297284 0.40367  0.071598 0.11259  0.059143 0.055716]
weights (fitted 6-vector): [0.26908337 0.32513951 0.10917985 0.12664755 0.09542619 0.07452353]

fit residuals (calibrated marginal - actual val frequency):
| class   |     actual |        raw |  calibrated |       resid |
| dot     |   0.297284 |   0.225698 |    0.297284 |   -1.40e-10 |
| one     |   0.403670 |   0.249412 |    0.403670 |   +4.78e-11 |
| two     |   0.071598 |   0.121795 |    0.071598 |   +1.66e-11 |
| four    |   0.112590 |   0.163560 |    0.112590 |   +1.69e-11 |
| six     |   0.059143 |   0.107010 |    0.059143 |   +3.77e-11 |
| wicket  |   0.055716 |   0.132525 |    0.055716 |   +2.09e-11 |
max |residual| = 1.400e-10
saved: models/auto/d16/vector_scaling_calibrator_d16.pkl
```

Fitted 6-vector (dot, one, two, four, six, wicket):
`[0.26908337, 0.32513951, 0.10917985, 0.12664755, 0.09542619, 0.07452353]`.
The served distribution is the venue-ON one (venue encoder ACTIVE, 373 venues,
101 distinct venue codes present in validation), matching the sim path. The
sanity check the plan asked for is the residual column: the corrected marginals
match the validation class frequencies to ≤1.4e-10.

---

## Step 4 — teacher-forced marginal audits (frozen BEFORE the sim evals)

Test parquet `data/xgb_data_i7/cricket_data_i7_test.parquet`, 186,667 balls,
114 features, each arm encoded with its own training-time encoders. Tolerance
`|dP(wicket)| ≤ 0.005` and `|d runs/ball| ≤ 0.05`.

### GATE 1(a) — NO-WEIGHTS arm, `venue_on` (PRIMARY) — **PASS**

```
--- noweights raw / venue_on (PRIMARY)  (n = 186,667 balls) ---
| class   |      pred |    actual |     delta |
|---------|-----------|-----------|-----------|
| dot     |   0.31099 |   0.31267 |  -0.00168 |
| one     |   0.39424 |   0.39715 |  -0.00291 |
| two     |   0.07431 |   0.07567 |  -0.00136 |
| four    |   0.10579 |   0.10579 |  -0.00000 |
| six     |   0.05476 |   0.05200 |  +0.00276 |
| wicket  |   0.05991 |   0.05672 |  +0.00319 |
  runs/ball   pred 1.2946  actual 1.2837  delta +0.0109  (tol 0.05) -> ok
  P(wicket)   pred 0.05991  actual 0.05672  delta +0.00319  (tol 0.005) -> ok
  test multiclass LL 1.4253
  ARM VERDICT: PASS
```

`venue_zero` context arm also passes (dP(wkt) +0.00392, d runs/ball +0.0136,
LL 1.4260). Written to `models/auto/d16/noweights/marginal_audit.json`
(preserved as `raw/marginal_audit_noweights.json.txt`); script closed with
`GATE 1(a) marginal audit (primary = venue_on): PASS`.

### CONTROL arm (context) — raw FAILS, + d16 vector PASSES

```
--- control raw / venue_on (PRIMARY)  (n = 186,667 balls) ---
| dot     |   0.23936 |   0.31267 |  -0.07332 |
| one     |   0.24691 |   0.39715 |  -0.15024 |
| two     |   0.12334 |   0.07567 |  +0.04767 |
| four    |   0.15459 |   0.10579 |  +0.04880 |
| six     |   0.10184 |   0.05200 |  +0.04984 |
| wicket  |   0.13396 |   0.05672 |  +0.07724 |
  runs/ball   pred 1.7230  actual 1.2837  delta +0.4394  (tol 0.05) -> FAIL
  P(wicket)   pred 0.13396  actual 0.05672  delta +0.07724  (tol 0.005) -> FAIL
  test multiclass LL 1.6316
  ARM VERDICT: FAIL

--- control + vector_scaling_calibrator_d16.pkl / venue_on (CONTEXT) ---
| dot     |   0.31314 |   0.31267 |  +0.00047 |
| one     |   0.39739 |   0.39715 |  +0.00024 |
| two     |   0.07216 |   0.07567 |  -0.00351 |
| four    |   0.10558 |   0.10579 |  -0.00022 |
| six     |   0.05597 |   0.05200 |  +0.00396 |
| wicket  |   0.05577 |   0.05672 |  -0.00095 |
  runs/ball   pred 1.2998  actual 1.2837  delta +0.0161  (tol 0.05) -> ok
  P(wicket)   pred 0.05577  actual 0.05672  delta -0.00095  (tol 0.005) -> ok
  test multiclass LL 1.5072
  ARM VERDICT: PASS
```

Control raw `venue_zero`: +0.06362 / +0.3717, LL 1.6072 (FAIL).
Control + d16 vector `venue_zero`: −0.00832 / −0.0465, LL 1.5265 (FAIL on the
wicket tolerance — this is the off-path distribution the vector was NOT fit on,
included only as context). The archived `models/xgb_i7` reference arms printed
in both audit logs are numerically identical to the control raw arms
(+0.07724 / +0.4394 venue_on; +0.06362 / +0.3717 venue_zero).

Reproduced on the i7 frame, the E5 balanced-weights signature is
**+0.0772 ΔP(wicket) / +0.4394 Δruns-per-ball** on the served venue-ON path —
larger than the legacy-frame context numbers (+0.0647 / +0.3829). Test
multiclass LL: control raw 1.6316 → control+d16-vector 1.5072 → **noweights raw
1.4253**, i.e. the structural arm beats the calibrated control by 0.0819 LL on
the ball task.

---

## Step 5 — twin sim evals (seed 46, n=261 × 100 sims, `--stats-version i7`)

Startup banners (verbatim, both arms):

```
Arm C  Ball calibrator: vector scaling (models/auto/d16/vector_scaling_calibrator_d16.pkl)
Arm C  StatsProvider: using SQLite backend player_stats_cache_i7.sqlite (56.7 MB)
Arm C  Loaded XGBoost v2 model with 114 features with real player stats with player metadata; venue encoder ACTIVE (373 venues)
Arm C  Bowler selector: empirical
Arm C  Run-out dismissal channel ACTIVE (p_runout=0.0751, nonstriker_share=0.4685)
Arm C  Running prop backtest on 261 matches × 100 sims
Arm C  B10 usage-aligned bowler selector ACTIVE (k_u=5.0)
Arm C    as-of corpus: models/b10_usage_corpus.pkl (7433 players); min_eligible=5, min_share=0.01

Arm N  (no "Ball calibrator" line — --ball-calibrator none)
Arm N  StatsProvider: using SQLite backend player_stats_cache_i7.sqlite (56.7 MB)
Arm N  Loaded XGBoost v2 model with 114 features with real player stats with player metadata; venue encoder ACTIVE (373 venues)
Arm N  Bowler selector: empirical
Arm N  Run-out dismissal channel ACTIVE (p_runout=0.0751, nonstriker_share=0.4685)
Arm N  Running prop backtest on 261 matches × 100 sims
Arm N  B10 usage-aligned bowler selector ACTIVE (k_u=5.0)
Arm N    as-of corpus: models/b10_usage_corpus.pkl (7433 players); min_eligible=5, min_share=0.01
```

`373 venues` (not 467) on both arms confirms the i7 model/encoder pair.
`--bowler-usage-path` was not passed, so both arms used the same default
B12-shipped B10-active sidecar and printed the same selector banner.

| arm | duration | matches | skips |
|---|---|---|---|
| C — control + d16 vector | **`Done in 2251.8s`** (37.5 min) | `[261/261]`, detail JSON has **261** matches | **0** (`grep -ic skip` = 0) |
| N — no-weights raw | **`Done in 1297.1s`** (21.6 min) | `[261/261]`, detail JSON has **261** matches | **0** (`grep -ic skip` = 0) |

Zero load-skips on the i7 path in either arm. Both launched detached with
`nohup` and waited on synchronously in-session; both exited on their own.
(One polling shell was cut short by the harness's 10-minute per-call cap — the
detached eval was unaffected and polling simply resumed. This is a harness
limit, not a run event.)

---

## Step 6 — gate analysis (`research/handoff/D16/raw/gate_output.txt`, verbatim)

```
baseline: 261 matches (detail_control_vec_s46_n261.json)
d16:      261 matches (detail_noweights_raw_s46_n261.json)
pairing: cluster bootstrap by match, n_boot=2000, seed=29; delta = d16 - baseline (negative = d16 better)

GATE 1(a) — teacher-forced marginal audit of the NO-WEIGHTS arm (frozen BEFORE the sim evals)
  audit file: models/auto/d16/noweights/marginal_audit.json
  primary arm: noweights raw / venue_on (PRIMARY)  (n = 186,667)
  dP(wicket)   +0.00319   (tol +/-0.005)
  d runs/ball  +0.01093   (tol +/-0.05)
  test multiclass LL 1.4253
  GATE 1(a): PASS

GATE 1(b) — POOLED TAIL dBrier over the row-pool pp_total_ou_45_5, pp_total_ou_50_5, pp_total_ou_55_5, bowler_wkts_1plus must be CI-clean negative
pooled tail                         4673     0      0.2488      0.2372   -0.0116   [-0.0159,-0.0073]  DOWN(better)
  (context, equal weight per family: -0.0089)

  per-family breakdown (context; the pool is the gate):
family                                 n  drop control_vec   noweights     delta   95% CI (noweights-control_vec)   flag
pp_total_ou_45_5                     522     0      0.2414      0.2367   -0.0047   [-0.0174,+0.0082]  ~noise
  (positional cross-check)           522                                 -0.0047   [-0.0174,+0.0082]  ~noise
pp_total_ou_50_5                     522     0      0.2434      0.2355   -0.0079   [-0.0213,+0.0052]  ~noise
  (positional cross-check)           522                                 -0.0079   [-0.0213,+0.0052]  ~noise
pp_total_ou_55_5                     522     0      0.2096      0.2005   -0.0091   [-0.0193,+0.0010]  ~noise
  (positional cross-check)           522                                 -0.0091   [-0.0193,+0.0010]  ~noise
bowler_wkts_1plus                   3107     0      0.2575      0.2437   -0.0138   [-0.0178,-0.0100]  DOWN(better)
  (positional cross-check)          2235                                 -0.0138   [-0.0190,-0.0088]  DOWN(better)

  GATE 1(b): PASS

GATE 1(c) — batter_runs_mae must NOT regress CI-clean (CI lo <= 0)
family                                 n  drop control_vec   noweights     delta   95% CI (noweights-control_vec)   flag
batter_runs_mae                     4252     0     14.4354     13.8905   -0.5449   [-0.6714,-0.4142]  DOWN(better)
  (positional cross-check)          2912                                 -0.6405   [-0.8067,-0.4689]  DOWN(better)
  GATE 1(c): PASS

  GATE 1 (a AND b AND c): MET   [a=True b=True c=True]

GATE 2 — guards: no CI-clean regression on top_bowler, team_first_over_mae
family                                 n  drop control_vec   noweights     delta   95% CI (noweights-control_vec)   flag
top_bowler                          5827     8      0.0777      0.0772   -0.0004   [-0.0014,+0.0004]  ~noise
  (positional cross-check)          5827                                 -0.0004   [-0.0014,+0.0004]  ~noise
team_first_over_mae                  522     0      3.4683      3.3785   -0.0898   [-0.2008,+0.0173]  ~noise
  (positional cross-check)           522                                 -0.0898   [-0.2008,+0.0173]  ~noise
  top_bowler                   ok
  team_first_over_mae          ok
  GATE 2: MET
```

### CONTEXT — full family scan (cannot flip the verdict)

```
family                                 n  drop control_vec   noweights     delta   95% CI (noweights-control_vec)   flag
batter_50plus                       4252     0      0.0802      0.0786   -0.0016   [-0.0030,-0.0002]  DOWN(better)
batter_6plus_six                    4252     0      0.2262      0.2196   -0.0066   [-0.0097,-0.0033]  DOWN(better)
batter_fours_1plus                  4252     0      0.2209      0.2184   -0.0025   [-0.0057,+0.0005]  ~noise
batter_fours_2plus                  4252     0      0.2035      0.2012   -0.0023   [-0.0052,+0.0005]  ~noise
batter_fours_3plus                  4252     0      0.1588      0.1554   -0.0034   [-0.0058,-0.0010]  DOWN(better)
batter_fours_mae                    4252     0      1.4004      1.3589   -0.0415   [-0.0562,-0.0272]  DOWN(better)
batter_runs_mae                     4252     0     14.4354     13.8905   -0.5449   [-0.6714,-0.4142]  DOWN(better)
bowler_economy_ou_10_5              3107     0      0.1921      0.1912   -0.0008   [-0.0044,+0.0028]  ~noise
bowler_economy_ou_8_5               3107     0      0.2509      0.2501   -0.0008   [-0.0068,+0.0052]  ~noise
bowler_wkts_1plus                   3107     0      0.2575      0.2437   -0.0138   [-0.0178,-0.0100]  DOWN(better)
bowler_wkts_2plus                   3107     0      0.2179      0.2103   -0.0076   [-0.0104,-0.0050]  DOWN(better)
bowler_wkts_3plus                   3107     0      0.0953      0.0946   -0.0007   [-0.0018,+0.0004]  ~noise
first_wicket_runs_ou_30_5            522     0      0.2386      0.2395   +0.0009   [-0.0078,+0.0099]  ~noise
highest_individual_mae               261     0     16.9043     16.1951   -0.7091   [-1.5647,+0.1995]  ~noise
highest_over_runs_ou_18_5            261     0      0.2375      0.2445   +0.0070   [-0.0077,+0.0217]  ~noise
highest_over_runs_ou_24_5            261     0      0.1000      0.0971   -0.0030   [-0.0061,+0.0000]  ~noise
innings_runs_ou_160_5                522     0      0.2509      0.2381   -0.0128   [-0.0344,+0.0077]  ~noise
innings_runs_ou_170_5                522     0      0.2418      0.2297   -0.0120   [-0.0345,+0.0091]  ~noise
innings_runs_ou_180_5                522     0      0.1956      0.1984   +0.0028   [-0.0163,+0.0215]  ~noise
match_total_sixes_ou_15_5            261     0      0.2086      0.1980   -0.0106   [-0.0365,+0.0156]  ~noise
match_total_sixes_ou_20_5            261     0      0.1143      0.1006   -0.0137   [-0.0284,+0.0016]  ~noise
p_tie                                261     0      0.0004      0.0003   -0.0001   [-0.0002,-0.0000]  DOWN(better)
pp_total_ou_45_5                     522     0      0.2414      0.2367   -0.0047   [-0.0174,+0.0082]  ~noise
pp_total_ou_50_5                     522     0      0.2434      0.2355   -0.0079   [-0.0213,+0.0052]  ~noise
pp_total_ou_55_5                     522     0      0.2096      0.2005   -0.0091   [-0.0193,+0.0010]  ~noise
team_first_over_mae                  522     0      3.4683      3.3785   -0.0898   [-0.2008,+0.0173]  ~noise
team_highest_individual_ou_29_5      522     0      0.0783      0.0757   -0.0026   [-0.0054,-0.0001]  DOWN(better)
team_highest_individual_ou_34_5      522     0      0.1349      0.1293   -0.0056   [-0.0109,-0.0008]  DOWN(better)
team_highest_individual_ou_39_5      522     0      0.1795      0.1737   -0.0058   [-0.0133,+0.0011]  ~noise
team_total_fours_mae                 522     0      3.6059      3.4987   -0.1072   [-0.2306,+0.0162]  ~noise
team_total_sixes_mae                 522     0      2.9035      2.6457   -0.2579   [-0.3953,-0.1208]  DOWN(better)
top_batter                          5835     0      0.0766      0.0756   -0.0010   [-0.0021,+0.0000]  ~noise
top_bowler                          5827     8      0.0777      0.0772   -0.0004   [-0.0014,+0.0004]  ~noise

  CI-excludes-0 families (either direction):
    batter_runs_mae                      -0.5449  [-0.6714,-0.4142]  DOWN(better)
    team_total_sixes_mae                 -0.2579  [-0.3953,-0.1208]  DOWN(better)
    batter_fours_mae                     -0.0415  [-0.0562,-0.0272]  DOWN(better)
    bowler_wkts_1plus                    -0.0138  [-0.0178,-0.0100]  DOWN(better)
    bowler_wkts_2plus                    -0.0076  [-0.0104,-0.0050]  DOWN(better)
    batter_6plus_six                     -0.0066  [-0.0097,-0.0033]  DOWN(better)
    team_highest_individual_ou_34_5      -0.0056  [-0.0109,-0.0008]  DOWN(better)
    batter_fours_3plus                   -0.0034  [-0.0058,-0.0010]  DOWN(better)
    team_highest_individual_ou_29_5      -0.0026  [-0.0054,-0.0001]  DOWN(better)
    batter_50plus                        -0.0016  [-0.0030,-0.0002]  DOWN(better)
    p_tie                                -0.0001  [-0.0002,-0.0000]  DOWN(better)
```

**Eleven CI-clean movers, all in the no-weights arm's favour; zero CI-clean
regressions anywhere in the 33-family scan.** `top_bowler` shows 8 dropped rows
in both directions of the keyed match (unchanged between arms, and its
positional cross-check agrees to 4 decimals).

---

## Evidence preserved (`models/auto/` and `*.json` are gitignored)

Committed under `research/handoff/D16/raw/` (`.gitignore` excludes `*.log` and
`*.json`, so those are committed as `.txt` copies — B12/D6 precedent):

| file | what |
|---|---|
| `config_json.txt` | the extracted training config JSON, verbatim |
| `train_control.log.txt`, `train_noweights.log.txt` | full training stdout |
| `artifact_checks.txt` | HARD CHECK, determinism control, contracts, tree counts |
| `md5_protected_before.txt`, `md5_protected_after.txt` | protected-artifact md5s |
| `fit_calibrator.log.txt` | calibrator fit incl. the 6-vector and residuals |
| `marginal_audit_noweights.log.txt`, `marginal_audit_control.log.txt` | audit stdout |
| `marginal_audit_noweights.json.txt`, `marginal_audit_control.json.txt` | audit JSONs (`.txt` copies, B12/D6 precedent) |
| `run_control_vec.log.txt`, `run_noweights_raw.log.txt` | full 261-match eval stdout |
| `report_control_vec_s46_n261.md.txt`, `report_noweights_raw_s46_n261.md.txt` | prop_backtest reports |
| `gate_output.txt` | full gate analysis |

Not committed (gitignored, regenerable): the two 11 MB detail JSONs, the two
arm model dirs, `models/auto/d16/vector_scaling_calibrator_d16.pkl`.

## Compliance notes

- No verdict issued, nothing reverted, `research/results.tsv` and
  `research/IDEAS.md` untouched, no `git push`, no `git reset`.
- Nothing under `data/golden/`, `data/forward_holdout/`, `scripts/sim_eval/`,
  `scripts/calibration.py`, `scripts/fit_i5_ball_calibrator.py`,
  `scripts/e5_teacher_forced_bias.py`, `models/xgb_v3/`, `models/xgb_i7/`, or
  `models/bowler_phase_usage.json` was modified. `data/xgb_data_i7/` was read
  only.
- No identity fail-close was bypassed, weakened or edited; the trainer's
  `assert_venue_alias_contract` passed on this frame on both arms.
- One heavy process at a time throughout; all Python via `uv run`; no new
  dependencies; no network access.
- Per the plan, LANDED would ship nothing: the artifacts stay in
  `models/auto/d16/`, the default sim path is untouched, and
  `models/auto/b12/detail_b10_s44_n261.json` remains the canonical baseline for
  legacy-path ideas. The legacy B12 numbers were used nowhere in this gate —
  the comparator is the same-session control+vector twin.
