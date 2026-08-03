# I19 — Coherent-contract i7 frame re-key — EXECUTOR RESULT

Executor run 2026-07-31 (local). Claim commit `fc831c9`; plan commit `2239bc0`.
**No verdict is issued here — the orchestrator judges.**

All numbers below are copied verbatim from the tee'd logs in
`research/handoff/I19/raw/`.

---

## Headline

| Gate | Result |
|---|---|
| 1. `i18_frame_parity.py` **relaxed** gate on the v2 frame | **PASS** (exit 0) |
| 2. Both seed-29 arms reproduce archived I17 predictions at max \|Δp\| = 0 | **PASS** (exit 0) |
| 3. `i19_contract_check.py` full identity contract | **FAIL** (exit 1) — one sub-assertion only: `display_match_id` uniqueness |

Gate 3 failed on the plan's `display_match_id` **unique per row** sub-assertion.
That sub-assertion is contradicted by the identity contract's own design (see
"Surprises", below): the legacy display string is `date_team1_team2_venue` and is
documented in `scripts/match_identity.py` as "not unique for same-day
doubleheaders and must never be used as a new primary key." Every other sub-check
in gate 3 — including `match_id` uniqueness, which is the property that actually
matters for a primary key — passed on all four splits.

I did **not** weaken the committed gate to make it pass. The failing assertion is
reported as-is, plus a separate read-only diagnostic
(`raw/display_id_collision_diagnostic.log`) so the orchestrator can decide
whether the assertion or the data is at fault.

---

## Step 0 — setup

`git status` clean at start. HEAD = `2239bc0` (`Auto[I19]: executor plan`) on
branch `auto-20260730`. `program.md` read in full before any other action.

## Step 1 — build the v2 frame (file copy only)

`data/xgb_match_data_i7_v2/` created by `cp` of the seven files from
`data/auto/i18/frame/`. **No** regeneration: `materialize_match_features.py` and
`build_i7_match_frame.py` were never invoked. Nothing under
`data/xgb_match_data_i7/`, `data/auto/i18/`, or `data/golden/` was written, and
`data/golden/` was never read or listed.

md5 verification (`raw/copy.log`) — source and copy, same order:

```
=== md5 compare ===          --- v2 ---
633925ce7c2e62269b2bda9556d00ba7   633925ce7c2e62269b2bda9556d00ba7
2fa1f3ff70c520830441a67ff6713c45   2fa1f3ff70c520830441a67ff6713c45
0ea8cf28441dd424cfbb47bc4efe04ac   0ea8cf28441dd424cfbb47bc4efe04ac
46eb6a3f66daf6df7c8f16bc2e2220cd   46eb6a3f66daf6df7c8f16bc2e2220cd
80f3ac4fcd8a2347af651e046098eada   80f3ac4fcd8a2347af651e046098eada
f2c622bb5a15caa0e35c7f247ad655e6   f2c622bb5a15caa0e35c7f247ad655e6
72f62135985943c7a7fa412aef78c646   72f62135985943c7a7fa412aef78c646
```

All seven byte-identical. Both `data/xgb_match_data_i7_v2/` and
`models/auto/i19/` are gitignored (`.gitignore:8:data/`, `.gitignore:16:models/`).

Wall clock: < 1 s.

## Step 2 — harness commit (before any run)

Committed `24d0cc6` — `Auto[I19]: implement — coherent-contract i7_v2 frame
(I18-frame reuse) + check harness`, adding `scripts/auto/i19_contract_check.py`
(226 lines) and `scripts/auto/i19_repro_check.py` (172 lines). Both fail closed.

## Step 3a — relaxed parity (`raw/parity.log`, exit 0, 1 s)

PASS 1 (STRICT, informational) FAILED exactly as the plan predicted — shape
`(N, 57)` vs `(N, 54)`, candidate-only columns
`['display_match_id', 'match_identity_version', 'elo_update_version']`,
reference-only `[]`, and `match_id` differing on 100% of rows in every split
(e.g. test row 0: cand `'1482013'` vs ref
`'2025-07-01_San_Francisco_Unicorns_Seattle_Orcas_Central_Broward_Regional_Park_Stadium_Turf_Ground,_Lauderhill'`).

```
STRICT PARITY: FAIL
```

PASS 2 (RELAXED — the binding gate), per split:

| split | added cols exactly identity set | ref.match_id == cand.display_match_id | cand.match_id == cand.cricsheet_id | cricsheet_id set/order equal | shared cols (excl. match_id) |
|---|---|---|---|---|---|
| train | OK | 7972/7972 | 7972/7972 | True / True | 53, all bit-identical |
| validation | OK | 528/528 | 528/528 | True / True | 53, all bit-identical |
| test | OK | 798/798 | 798/798 | True / True | 53, all bit-identical |

`venue_identity.json` OK:
`{"venue_alias_active_count": 94, "venue_alias_sha256": "853b32b0ce3098dd8c0f33ba1437846f5505d50d9a425fbd37bff9c9f76745d8", "venue_alias_version": "venue_aliases_v1"}`

`golden_test.parquet`: `rows=227  columns=57`,
`match_identity_version values: ['cricsheet_primary_v1']`,
`cricsheet_id: 227 unique, 0 null`,
`match_date range: 2026-04-17 -> 2026-07-12`, OK.

Per-check list, verbatim:

```
  train: PASS
  validation: PASS
  test: PASS
  venue_identity: PASS
  golden_test: PASS

I18 RELAXED PARITY GATE: PASS
```

## Step 3b — contract check (`raw/contract.log`, exit 1, < 1 s)

Per-split, verbatim:

| split | rows (expected) | cols | match_id == cricsheet_id | display_match_id | match_identity_version | elo_update_version | match_id | result |
|---|---|---|---|---|---|---|---|---|
| train | 7972 (7972) | 57 | 7972/7972 | **7943 unique**, 0 null, 0 blank | `['cricsheet_primary_v1']` | `['fixed_competition_k_v1']` | 7972 unique, 0 null | **FAIL** |
| validation | 528 (528) | 57 | 528/528 | **522 unique**, 0 null, 0 blank | `['cricsheet_primary_v1']` | `['fixed_competition_k_v1']` | 528 unique, 0 null | **FAIL** |
| test | 798 (798) | 57 | 798/798 | **788 unique**, 0 null, 0 blank | `['cricsheet_primary_v1']` | `['fixed_competition_k_v1']` | 798 unique, 0 null | **FAIL** |
| golden_test | 227 (227) | 57 | 227/227 | **222 unique**, 0 null, 0 blank | `['cricsheet_primary_v1']` | `['fixed_competition_k_v1']` | 227 unique, 0 null | **FAIL** |

The only failing line on each split is
`FAIL: display_match_id is not unique per row`. Row counts, cricsheet-primary
re-key, both version constants, and `match_id` uniqueness all passed everywhere.

Sidecars, verbatim:

```
  [venue_identity.json]
    candidate bytes=174  reference bytes=174
    byte-equal to data/xgb_match_data_i7/venue_identity.json: True
    content: {"venue_alias_active_count": 94, "venue_alias_sha256": "853b32b0ce3098dd8c0f33ba1437846f5505d50d9a425fbd37bff9c9f76745d8", "venue_alias_version": "venue_aliases_v1"}
    PASS
  [match_identity.json]
    observed: {"display_key": "display_match_id", "match_identity_version": "cricsheet_primary_v1", "primary_key": "cricsheet_id"}
    expected: {"display_key": "display_match_id", "match_identity_version": "cricsheet_primary_v1", "primary_key": "cricsheet_id"}
    PASS
  [elo_update.json]
    observed: {"elo_update_version": "fixed_competition_k_v1"}
    expected: {"elo_update_version": "fixed_competition_k_v1"}
    PASS
```

Summary block, verbatim:

```
  train: FAIL
  validation: FAIL
  test: FAIL
  golden_test: FAIL
  venue_identity: PASS
  match_identity: PASS
  elo_update: PASS

I19 CONTRACT GATE: FAIL
```

## Step 3c — display-id collision diagnostic (`raw/display_id_collision_diagnostic.log`)

Read-only, not a gate; run to characterise the gate-3 failure.

```
[train] rows=7972 display_match_id unique=7943 collision_rows=58 lost_keys=29
[validation] rows=528 display_match_id unique=522 collision_rows=12 lost_keys=6
[test] rows=798 display_match_id unique=788 collision_rows=20 lost_keys=10
[golden_test] rows=227 display_match_id unique=222 collision_rows=10 lost_keys=5
```

Every collision is a same-day doubleheader between the same two teams at the same
venue — e.g.
`match_id=1207081` and `match_id=1207082` both map to
`2020-03-08_Spain_Germany_Desert_Springs_Cricket_Ground,_Almeria`;
`match_id=1494774` / `1494775` both map to
`2025-08-03_Switzerland_Estonia_Estonian_National_Cricket_and_Rugby_Field`;
`match_id=1533397` / `1533398` both map to
`2026-05-08_Cyprus_Finland_Happy_Valley_Ground`.

And the load-bearing consequence — the frozen reference frame keys `match_id` by
exactly this string, so **its primary key is non-unique**:

```
[train] rows=7972  legacy match_id unique=7943  cricsheet_id unique=7972
[validation] rows=528  legacy match_id unique=522  cricsheet_id unique=528
[test] rows=798  legacy match_id unique=788  cricsheet_id unique=798
```

45 rows across train/val/test in `data/xgb_match_data_i7/` share a `match_id`
with another row. The v2 re-key removes that.

## Step 4 — retrains on the v2 frame

Exact I17 commands, only `--data-dir` / `--model-dir` changed. Run sequentially,
never concurrently. Both exited 0.

**base_seed29** (`raw/train_base_seed29.log`, wall clock ~1 s):

```
  train: 7,972   val: 528   test: 798
  features (48): [...]
  monotone constraints: 10/48 features constrained (8 +1, 2 -1)
[0]	validation_0-logloss:0.69004
[50]	validation_0-logloss:0.65471
[85]	validation_0-logloss:0.65625

  val LL  = 0.6532   val Brier  = 0.2313
  test LL = 0.5974   test Brier = 0.2056
  standalone test (798 matches): LL=0.5974  Brier=0.2056
```

**swap_seed29** (`raw/train_swap_seed29.log`, wall clock ~1 s):

```
  train: 7,972   val: 528   test: 798
  swap-augment: train doubled → 15,944 rows (base rate 0.5000); val/test untouched
  features (48): [...]
  monotone constraints: 10/48 features constrained (8 +1, 2 -1)
[0]	validation_0-logloss:0.69045
[50]	validation_0-logloss:0.65483
[100]	validation_0-logloss:0.65069
[141]	validation_0-logloss:0.65371

  val LL  = 0.6504   val Brier  = 0.2306
  test LL = 0.5859   test Brier = 0.2012
  standalone test (798 matches): LL=0.5859  Brier=0.2012
```

All the expected shapes appeared: 48 features, n_train 7972 base / 15944 swap,
n_val 528, n_test 798.

## Step 5 — reproduction check (`raw/repro.log`, exit 0, < 1 s)

```
  [base_seed29]
    n_keys candidate=798  reference=798
    OK: key sets identical
    n_keys=798
    max |delta p_team1| = 0.000e+00
    rows with nonzero delta: 0
    OK: predictions reproduce exactly (max |delta p| = 0)
    feature_columns.txt: candidate 48 lines, reference 48 lines, byte-equal=True
    train_metrics.json (candidate vs reference):
      val_log_loss: 0.6531566579763646  |  0.6531566579763646
      test_log_loss: 0.5973871207926102  |  0.5973871207926102
      val_brier: 0.23130043775124437  |  0.23130043775124437
      test_brier: 0.20560113187271795  |  0.20560113187271795
      n_train: 7972  |  7972
      n_val: 528  |  528
      n_test: 798  |  798
      seed: 29  |  29
    base_seed29: PASS
  [swap_seed29]
    n_keys candidate=798  reference=798
    OK: key sets identical
    n_keys=798
    max |delta p_team1| = 0.000e+00
    rows with nonzero delta: 0
    OK: predictions reproduce exactly (max |delta p| = 0)
    feature_columns.txt: candidate 48 lines, reference 48 lines, byte-equal=True
    train_metrics.json (candidate vs reference):
      val_log_loss: 0.6503653371453989  |  0.6503653371453989
      test_log_loss: 0.5859296475141611  |  0.5859296475141611
      val_brier: 0.23060272830523185  |  0.23060272830523185
      test_brier: 0.2011695634868112  |  0.2011695634868112
      n_train: 15944  |  15944
      n_val: 528  |  528
      n_test: 798  |  798
      seed: 29  |  29
    swap_seed29: PASS

  base_seed29: PASS
  swap_seed29: PASS

I19 REPRODUCTION GATE: PASS
```

Both arms match the plan's reference targets to the last digit:
base_seed29 val 0.6531566579763646 / test 0.5973871207926102;
swap_seed29 val 0.6503653371453989 / test 0.5859296475141611.

### Stronger-than-required extra check (informational)

The trained artifacts themselves are byte-identical, not merely the predictions:

```
02d4bbbe08b08bdff9fffd6874b0936d   models/auto/i19/base_seed29/model.pkl
02d4bbbe08b08bdff9fffd6874b0936d   models/auto/i17/base_seed29/model.pkl
54faf58638a799468d551beb3493b22d   models/auto/i19/swap_seed29/model.pkl
54faf58638a799468d551beb3493b22d   models/auto/i17/swap_seed29/model.pkl
```

### The re-key actually changed something (the point of I19)

The i17 arms were trained off a frame with no `match_identity.json`, so
`xgboost_match_v1.py` stamped them legacy. The i19 arms carry the current
contract:

```
models/auto/i19/base_seed29 -> {"display_key": "display_match_id", "match_identity_version": "cricsheet_primary_v1", "primary_key": "cricsheet_id"}
models/auto/i17/base_seed29 -> {"display_key": "match_id", "match_identity_version": "synthetic_fixture_v1", "primary_key": "match_id"}
models/auto/i19/swap_seed29 -> {"display_key": "display_match_id", "match_identity_version": "cricsheet_primary_v1", "primary_key": "cricsheet_id"}
models/auto/i17/swap_seed29 -> {"display_key": "match_id", "match_identity_version": "synthetic_fixture_v1", "primary_key": "match_id"}
```

Same weights, same predictions, honest provenance stamp instead of a legacy one.
That is the whole deliverable.

---

## Exit codes (`raw/exit_codes.log`)

```
parity_exit=0
contract_exit=1
repro_exit=0
```

## Commits created

| SHA | message |
|---|---|
| `24d0cc6` | `Auto[I19]: implement — coherent-contract i7_v2 frame (I18-frame reuse) + check harness` |
| (this file) | `Auto[I19]: executor result` |

`git diff --stat fc831c9..HEAD` at the time of the implement commit:

```
 research/handoff/I19/plan.md       | 177 +++++++++++++++++++++++++++++
 scripts/auto/i19_contract_check.py | 226 +++++++++++++++++++++++++++++++++++++
 scripts/auto/i19_repro_check.py    | 172 ++++++++++++++++++++++++++++
 3 files changed, 575 insertions(+)
```

(The frame and the model artifacts are gitignored, so they do not appear.)

## Wall clock

Every step was seconds, not minutes — the plan's "runs take a few minutes each"
estimate was pessimistic for a 7,972-row / 48-feature frame with early stopping
at 85 (base) and 141 (swap) rounds.

| step | wall clock |
|---|---|
| 1. frame copy + md5 | < 1 s |
| 3a. relaxed parity | 1 s |
| 3b. contract check | < 1 s |
| 3c. collision diagnostic | < 1 s |
| 4. base_seed29 retrain | ~1 s |
| 4. swap_seed29 retrain | ~1 s |
| 5. reproduction check | < 1 s |

Total executor compute well under a minute; the session was dominated by
reading and writing, not running.

---

## Crashes and surprises

**Nothing crashed.** Two things surprised me, one of them load-bearing.

1. **Gate 3's `display_match_id` uniqueness assertion is mis-specified, and the
   frozen frame is worse off than the plan assumed.** The plan asked me to assert
   that `display_match_id` is unique per row. It is not, and it is not supposed
   to be: `scripts/match_identity.py`'s module docstring states that the
   `date_team1_team2_venue` string "is not unique for same-day doubleheaders and
   must never be used as a new primary key." 45 rows across train/val/test (plus
   5 in golden_test) collide, all genuine same-day doubleheaders between the same
   two teams at the same venue in associate-nation cricket.

   The sharp consequence: because the frozen `data/xgb_match_data_i7/` keys
   `match_id` by exactly that string, **the frame of record for the successor
   line has a non-unique primary key on 45 rows**. This is not a defect I
   introduced; it is a pre-existing hazard, and it strengthens rather than
   weakens I19's motivation. It also means a downstream join keyed on the legacy
   `match_id` against that frame would silently fan out or silently pick a
   winner on those 45 rows.

   I left the gate script exactly as specified and let it fail, rather than
   relaxing the assertion to `match_id`-uniqueness to manufacture a pass. If the
   orchestrator agrees the assertion should have targeted `match_id` uniqueness
   (which passed 4/4 splits), gate 3 passes on every remaining sub-check and the
   fix is a one-line change to `i19_contract_check.py`. That is the
   orchestrator's call, not mine.

2. **Reproduction is stronger than the gate required.** The gate asked for
   max |Δp| = 0; I also got byte-identical `model.pkl` for both arms. So the
   I15/I16 schema drift is provably inert with respect to training: the three
   added columns are all strings and all in `METADATA_COLS`, the identity
   columns are inserted after `team1_wins` and therefore never perturb the
   numeric feature order that `colsample_bytree` is sensitive to, and
   `_auto_numeric_features` filters them out cleanly. Nothing about column order
   or encoders turned out to be load-bearing here.

One smaller note: my first attempts to capture exit codes through
`tee` + `${PIPESTATUS[0]}` printed empty (zsh scoping in the harness's
non-interactive shell), so I re-ran all three checks with output to
`/dev/null` purely to read `$?`. The checks are deterministic and read-only,
so the re-runs changed nothing; the tee'd logs are from the original runs.

## Compliance

- No verdict issued; `research/results.tsv` and `research/IDEAS.md` untouched.
- No `git push`, no `git reset`, no `git revert`, no second idea.
- `data/golden/` never read, written, or listed.
- Nothing written to `data/xgb_match_data_i7/`, `models/auto/i17/`,
  `models/auto/i18/`, `models/xgb_match_i7/`, or any production artifact.
  New artifacts only in `data/xgb_match_data_i7_v2/` and `models/auto/i19/`.
- `materialize_match_features.py`, `build_i7_match_frame.py`,
  `predict_golden.py`, and everything under `scripts/sim_eval/` were not run.
- No new dependencies; all Python via `uv run`.
- No background processes left running.
