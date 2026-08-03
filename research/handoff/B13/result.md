# B13 — result (executor). Never-bowler damping in the usage-absent branch

Executor run 2026-08-03, branch `auto-20260803`. Plan: `research/handoff/B13/plan.md`.
**No verdict is issued here** — the gate script's pre-committed mapping is
reported verbatim and the orchestrator decides.

**Headline:** GATE 1a NOT MET, GATE 1b NOT MET, GATE 2 MET →
pre-committed mapping **TABLED**. The damping does exactly what the mechanism
audit predicted (veteran never-bowler XI share 0.496% → 0.020%, debutants
untouched), but neither the `top_bowler` Brier nor the B9 sim−usage margin
moves: both are flat to marginally worse. Defect (b) is now *mechanically*
fixed and it buys nothing measurable at the market layer.

---

## 0. Baseline re-verification (all three facts re-checked before the eval)

| fact | expected | observed | ok |
|---|---|---|---|
| D16 blind log shows the current default path | B10 selector ACTIVE (k_u=5.0), venue encoder ACTIVE (373 venues), run-out channel ACTIVE, no calibrator | `B10 usage-aligned bowler selector ACTIVE (k_u=5.0)` / `venue encoder ACTIVE (373 venues)` / `Run-out dismissal channel ACTIVE (p_runout=0.0751, nonstriker_share=0.4685)` / `grep -ci calibrat` = **0** | ✅ |
| `git diff ea4acdb..HEAD -- scripts/sim_v1_2.py` | empty | empty (checked at claim state, before my edit) | ✅ |
| `models/bowler_phase_usage.json` md5 | `2e650423f0c949631fca1f15dd1c8a56` | `2e650423f0c949631fca1f15dd1c8a56` (before build, after build, after unit check, after eval) | ✅ |

Extra provenance checks:
`md5 models/xgb_i7_noweights_production/xgboost_model_i7.pkl` =
`7ee1e1809917f45be7e726b3ea4a8a6c` = `md5 models/auto/d16/noweights/xgboost_model_i7.pkl`
(the promotion re-pointed `prop_backtest.py` defaults at byte-identical
artifacts, so defaults reproduce the D16 Arm N stack).

---

## 1. Damping fit (`scripts/auto/b13_build_damping_sidecar.py`)

Raw log: `research/handoff/B13/raw/build_sidecar.log(.txt)`.
JSON: `models/auto/b13/damping_fit.json`.

```
corpus: models/b10_usage_corpus.pkl (md5 42b313b138fefdd293c280a9d4be2e2b) — 7433 players, 208753 appearance rows

event set (n>=1, prior balls==0, date < 2025-07-01): 49485 events, 1005 bowled (2.031%)
k_damp (MLE on geomspace(0.01, 1000.0, 2001)): 0.153993  (logL -4070.266)
mu_active = mean(balls | bowled) = 12.163184  (n=1005)
  grid neighbours: k=0.153109 logL=-4070.268 | k=0.154882 logL=-4070.294
```

Fitted vs empirical `P(bowls | n prior appearances, 0 prior balls)`:

```
  n bin         events   bowled   empirical P    fitted P
  1               2862      334       11.670%     13.344%
  2               2272      167        7.350%      7.149%
  3-5             5007      229        4.574%      3.947%
  6-10            5631      123        2.184%      1.987%
  11-20           7220       86        1.191%      1.046%
  21-50          10742       45        0.419%      0.491%
  51+            15751       21        0.133%      0.160%
```

The one-parameter `k/(k+n)` curve tracks the empirical decay across three
orders of magnitude in n; it over-predicts slightly at n=1 (13.3% vs 11.7%)
and at n≥21 (0.49% vs 0.42%, 0.16% vs 0.13%), i.e. it is mildly conservative
exactly where the cohort lives.

Damped vs undamped expected balls (B9 global prior 10.477071 @ 2026-04-17,
k_usage 5.0):

```
       n    damped exp_balls     undamped B9     ratio   damped share   undamped share
       1            1.623096        8.730892    0.1859        1.3526%          7.2757%
       5            0.363416        5.238535    0.0694        0.3028%          4.3654%
      20            0.092936        2.095414    0.0444        0.0774%          1.7462%
     100            0.018702        0.498908    0.0375        0.0156%          0.4158%
     285            0.006569        0.180639    0.0364        0.0055%          0.1505%
```

(B9 prior_balls @ 2025-07-01 was 10.486322.) The n=285 row is PD Salt's exact
history: B9 gives him 0.18 balls/innings, the damping gives him 0.0066.

Sidecar: `models/auto/b13/bowler_phase_usage_b13.json` (md5
`9b59fe086d03ac0c685beb702c1bec0d`), containing

```
"b10_asof_usage": {"corpus_path": "models/b10_usage_corpus.pkl", "k_usage": 5.0,
                   "min_eligible": 5, "min_share": 0.01,
                   "b13_never_bowler_damping": {"k_damp": 0.1539926526059492,
                                                "mu_active": 12.16318407960199}}
```

---

## 2. Unit check (`scripts/auto/b13_unit_check.py`)

Raw log: `research/handoff/B13/raw/unit_check.log(.txt)`. **PASSED** (plus
`d15_unit_check.py` 30 PASS / 0 FAIL on the edited engine).

Full-XI mean selection share, 522 real lineups × 3 phases, blended
(pp 0.30 / mid 0.50 / death 0.20). `b13 eff` applies the ≥5-eligible
relaxation fallback; the pass condition uses the mechanism column, matching
the B10 table it is compared against.

| phase | group | rows | legacy | b10 | **b13** | b13 eff | b13/b10 |
|---|---|---:|---:|---:|---:|---:|---:|
| pp | true_debutant | 58 | 1.113% | 8.733% | 8.733% | 8.587% | 1.000 |
| pp | veteran_never_bowler | 1059 | 0.269% | 0.496% | **0.020%** | 0.033% | 0.039 |
| pp | known_bowler | 4282 | 12.023% | 11.385% | 11.712% | 11.711% | 1.029 |
| pp | other_unknown | 436 | 0.845% | 5.548% | 3.487% | 3.488% | 0.629 |
| mid | true_debutant | 58 | 1.150% | 8.733% | 8.733% | 8.733% | 1.000 |
| mid | veteran_never_bowler | 1059 | 0.272% | 0.496% | **0.020%** | 0.020% | 0.039 |
| mid | known_bowler | 4282 | 12.016% | 11.385% | 11.712% | 11.712% | 1.029 |
| mid | other_unknown | 436 | 0.900% | 5.548% | 3.487% | 3.487% | 0.629 |
| death | true_debutant | 58 | 1.220% | 8.733% | 8.733% | 8.586% | 1.000 |
| death | veteran_never_bowler | 1059 | 0.266% | 0.496% | **0.020%** | 0.024% | 0.039 |
| death | known_bowler | 4282 | 12.015% | 11.385% | 11.712% | 11.713% | 1.029 |
| death | other_unknown | 436 | 0.919% | 5.548% | 3.487% | 3.485% | 0.629 |
| **blend** | true_debutant | 58 | 1.153% | 8.733% | **8.733%** | 8.660% | 1.000 |
| **blend** | veteran_never_bowler | 1059 | 0.270% | 0.496% | **0.020%** | 0.025% | 0.039 |
| **blend** | known_bowler | 4282 | 12.018% | 11.385% | 11.712% | 11.712% | 1.029 |
| **blend** | other_unknown | 436 | 0.887% | 5.548% | 3.487% | 3.487% | 0.629 |

PASS lines, verbatim:

```
  [PASS] 1. cohort (b) veteran never-bowlers: b13 blend share < 0.10%  (legacy 0.270% / b10 0.496% -> b13 0.020%)
  [PASS] 2. cohort (a) true debutants: b13 within +-0.5pp of b10  (b10 8.733% vs b13 8.733% (delta 0.0000pp))
  [PASS] 3. usage-PRESENT players' weights float-equal b10 vs b13  (max |delta| = 0.000e+00)
  [PASS] 4a. production json: damped branch never taken (b13_damped_rows on the b10 arm)  (0 rows)
  [PASS] 4b. production-json weights float-exact vs an INDEPENDENT pre-B13 B10 re-derivation  (max |delta| = 0.000e+00)
  [INFO] b13 arm damped rows during the table pass: 3891
  [PASS] 4c. live select_bowler == recomputed B10 sampling on 320 same-seed draws (production json)  (mismatches = 0)
  [PASS] 4d. production arm still shows 0 damped rows after live draws  (0 rows)
```

**Relaxation triggers (item 5, context).** Over the 1,566 (lineup, phase)
cells of the battery: **b10 = 8, b13 = 32 (delta +24)**. In the real n=261 ×
100 eval, the same counts hold: `grep -c "B10 relaxation triggered"` gives
**8** on the D16 blind log and **32** on the B13 log. This is a 4× rise, not
an explosion (32 of 1,566 cells = 2.0%), and it is the expected side effect:
damping pushes never-bowlers below `min_share=0.01`, so lineups that only
just cleared 5 eligible bowlers now fall back to the flat-α legacy weights
(which is why the `b13 eff` column sits slightly above the mechanism column,
0.025% vs 0.020%). Affected sides are the thin-squad ones — Sydney Sixers,
Sydney Thunder, Melbourne Renegades, Paarl Royals, Lahore Qalandars, Quetta
Gladiators, Rawalpindiz, Bangladesh, Sri Lanka, UAE — all with `eligible=4`.

Side note worth flagging: the damping also drags the `other_unknown` cohort
(usage-absent, 1–19 prior appearances, or with some prior balls) from 5.548%
to 3.487%. That group is not the target and is not pinned by any pass
condition; part of it is genuine (n small, 0 balls → still damped), part is
renormalization.

---

## 3. Eval

One run; the blind arm was NOT re-run (D16 seed-46 detail reused per plan).

```
uv run python scripts/sim_eval/prop_backtest.py \
    --test-dir data/polymarket_test --n-matches all --n-sims 100 --seed 46 \
    --bowler-usage-path models/auto/b13/bowler_phase_usage_b13.json \
    --detail-out models/auto/b13/detail_b13_s46_n261.json \
    --report-out research/reports/auto/B13_props.md
```

Startup banners (verbatim, `research/handoff/B13/raw/run_b13_s46.log`):

```
StatsProvider: using SQLite backend player_stats_cache_i7.sqlite (56.7 MB)
Loaded XGBoost v2 model with 114 features with real player stats with player metadata; venue encoder ACTIVE (373 venues)
Bowler selector: empirical
Run-out dismissal channel ACTIVE (p_runout=0.0751, nonstriker_share=0.4685)
Running prop backtest on 261 matches × 100 sims
B10 usage-aligned bowler selector ACTIVE (k_u=5.0)
  as-of corpus: /Users/aryamangupta/CricML/Match_Prediction/models/b10_usage_corpus.pkl (7433 players); min_eligible=5, min_share=0.01
B13 never-bowler damping ACTIVE (k_damp=0.1539926526059492, mu_active=12.16318407960199)
```

No `Ball calibrator:` line (`grep -c "Ball calibrator"` = 0). 373 venues, not
467, confirms the i7 pair.

| | |
|---|---|
| runtime | `Done in 1285.5s` (21.4 min; D16 blind arm was 1297.1s) |
| matches | `[261/261]`, detail JSON has **261** matches |
| skips | **0** (`grep -ic skip` = 0) |
| relaxation triggers during the eval | **32** (blind: 8) |
| detail md5 | `38223fd85e6359078c18b233988474f3` |
| blind detail md5 | `d816ebcd5cc9190bc4c4ca578dd6bbf1` |

Well inside budget; nothing crashed, nothing was killed.

---

## 4. Gate numbers (VERBATIM from `research/handoff/B13/raw/gate_output.log`)

### GATE 1a (PRIMARY) — `top_bowler` paired Brier, b13 − blind, must be CI-clean negative

```
family                                 n  drop       blind         b13     delta   95% CI (b13-blind)   flag
top_bowler                          5831     4      0.0774      0.0775   +0.0001   [-0.0004,+0.0005]  ~noise
  (positional cross-check)          5831                                 +0.0001   [-0.0004,+0.0005]  ~noise

  GATE 1a: NOT MET
```

### GATE 1b — recomputed B9 sim−usage margin must SHRINK

```
  [blind] rows=5835  Brier_sim=0.077639  Brier_usage=0.074665  sim-usage=+0.002973 CI [+0.0021,+0.0038]  UP(worse)
  [blind] head-only (both p>=2%) rows=3897  sim-usage=+0.003704 CI [+0.0025,+0.0049]  UP(worse)
  [b13] rows=5835  Brier_sim=0.077704  Brier_usage=0.074666  sim-usage=+0.003038 CI [+0.0022,+0.0039]  UP(worse)
  [b13] head-only (both p>=2%) rows=3761  sim-usage=+0.004016 CI [+0.0027,+0.0053]  UP(worse)

  margin_blind = +0.002973  (Brier_sim 0.077639 - Brier_usage 0.074665)
  margin_b13   = +0.003038  (Brier_sim 0.077704 - Brier_usage 0.074666)
  change       = +0.000065

  GATE 1b: NOT MET (point comparison, pre-committed)

  GATE 1 (1a AND 1b): NOT MET
```

The four underlying Brier numbers: sim 0.077639 → 0.077704 (worse by
0.000065); usage baseline 0.074665 → 0.074666 (moves by 1e-6 — it is the same
baseline on the same rows, so this is just the identity-matching noise floor).
The head-only slice loses 136 rows under b13 (3897 → 3761) because damped
never-bowlers drop below the 2% inclusion cut on the `p_sim` side; on the
remaining head the margin widens from +0.003704 to +0.004016.

### GATE 2 — guards, no CI-clean regression

```
family                                 n  drop       blind         b13     delta   95% CI (b13-blind)   flag
bowler_wkts_1plus                   3107     0      0.2437      0.2431   -0.0006   [-0.0022,+0.0011]  ~noise
  (positional cross-check)          2615                                 -0.0010   [-0.0034,+0.0013]  ~noise
bowler_wkts_2plus                   3107     0      0.2103      0.2094   -0.0008   [-0.0019,+0.0002]  ~noise
  (positional cross-check)          2691                                 -0.0013   [-0.0029,+0.0003]  ~noise
batter_runs_mae                     4254     0     13.8864     13.8609   -0.0255   [-0.0564,+0.0067]  ~noise
  (positional cross-check)          3861                                 -0.0292   [-0.0633,+0.0049]  ~noise
team_first_over_mae                  522     0      3.3785      3.3868   +0.0084   [-0.0014,+0.0179]  ~noise
  (positional cross-check)           522                                 +0.0084   [-0.0014,+0.0179]  ~noise
  bowler_wkts_1plus            ok
  bowler_wkts_2plus            ok
  batter_runs_mae              ok
  team_first_over_mae          ok
  GATE 2: MET
```

### Full family scan — CI-clean movers

```
  families scanned: 33
  CI-excludes-0 families: 1 (1 better, 0 worse)
    batter_6plus_six                     -0.0013  [-0.0023,-0.0003]  DOWN(better)
```

**1 CI-clean better, 0 CI-clean worse, out of 33 families.** The single mover
is `batter_6plus_six` (0.2195 → 0.2182), which is a batting-side family — the
plausible channel is that damped never-bowlers stop bowling, so the real
bowlers face more of the innings and six-rate distributions shift. It was not
pre-committed and cannot flip anything.

### Context — G5 bowler coverage

```
  G5 coverage [blind]: 3106/3108 = 0.9994
  G5 coverage [b13]: 3103/3108 = 0.9984
```

Coverage falls by 3 real bowlers (0.9994 → 0.9984), still far above B10's 0.90
floor. That is the expected cost of pricing never-bowlers at ~0: a handful of
genuine first-time bowlers now get p_sim = 0.

### Final mapping line

```
GATE 1a: NOT MET | GATE 1b: NOT MET | GATE 1: NOT MET | GATE 2: MET
Pre-committed verdict MAPPING (orchestrator decides): TABLED
```

---

## 5. Commits and diff

| SHA | message |
|---|---|
| `55677fb` | `Auto[B13]: claim` (orchestrator, pre-existing) |
| `5e1d6fd` | `Auto[B13]: orchestrator plan (executor handoff)` (orchestrator, pre-existing) |
| `e2d8378` | `Auto[B13]: implement — never-bowler damping sidecar builder, opt-in sim_v1_2 branch, unit check + pre-committed gate` |
| `2d985c1` | `Auto[B13]: sidecar fit + unit-check raw logs (pre-eval)` |

`b13_gate_analysis.py` is in `e2d8378`, committed **before** the eval was
launched — verifiable from `git log --diff-filter=A -- scripts/auto/b13_gate_analysis.py`
against the eval log's timestamps.

`git diff --stat 55677fb..HEAD` (as of the pre-eval commit `2d985c1`; the
report/handoff commit follows this file):

```
 research/handoff/B13/plan.md                   | 212 +++++++++++++
 research/handoff/B13/raw/build_sidecar.log.txt |  32 ++
 research/handoff/B13/raw/unit_check.log.txt    |  64 ++++
 scripts/auto/b13_build_damping_sidecar.py      | 233 ++++++++++++++
 scripts/auto/b13_gate_analysis.py              | 369 ++++++++++++++++++++++
 scripts/auto/b13_unit_check.py                 | 421 +++++++++++++++++++++++++
 scripts/sim_v1_2.py                            |  30 ++
 7 files changed, 1361 insertions(+)
```

Final `git diff --stat 55677fb..HEAD` after the report/handoff commit
`212ab72` (`Auto[B13]: eval + gate output + result.md`):

```
 research/handoff/B13/plan.md                       | 212 +++++++++++
 research/handoff/B13/raw/build_sidecar.log.txt     |  32 ++
 research/handoff/B13/raw/gate_output.log.txt       | 105 +++++
 .../handoff/B13/raw/report_b13_s46_n261.md.txt     |  93 +++++
 research/handoff/B13/raw/run_b13_s46.log.txt       | 310 +++++++++++++++
 research/handoff/B13/raw/unit_check.log.txt        |  64 ++++
 research/handoff/B13/result.md                     | 353 +++++++++++++++++
 research/reports/auto/B13_props.md                 |  93 +++++
 scripts/auto/b13_build_damping_sidecar.py          | 233 ++++++++++++
 scripts/auto/b13_gate_analysis.py                  | 369 ++++++++++++++++++
 scripts/auto/b13_unit_check.py                     | 421 +++++++++++++++++++++
 scripts/sim_v1_2.py                                |  30 ++
 12 files changed, 2315 insertions(+)
```

(`models/auto/b13/*` — the sidecar, the fit JSON and the 11 MB detail — are
gitignored, as expected; not force-added. `*.log` is gitignored too, so each
raw log is also committed as `.log.txt` per the D16 convention.)

`scripts/sim_v1_2.py` is +30/−0 across three hunks: two attribute/counter
initialisations in `__init__`, the cfg read + banner in `_ensure_b10`, and the
damped branch (+ docstring) in `_b10_share_weights`. Nothing else in the
engine changed; the relaxation logic, the legacy branch and `select_bowler`
are untouched.

---

## 6. Anything that crashed, surprised, or ran long

- **Nothing crashed.** The eval ran 21.4 min (budget ~35–50 min, kill at
  ~100 min) and exited on its own. It was launched detached with `nohup` and
  waited on synchronously in-session with an `until ! kill -0 PID` loop,
  because the harness caps a single foreground call at 10 minutes. One heavy
  process at a time; `--parallel` was never used; no background process was
  left running (`ps -p 86971` is empty at exit).
- **Surprise 1 — the mechanism works perfectly and buys nothing.** Cohort (b)
  went 0.496% → 0.020% (a 25× reduction, and 13× below even the legacy α
  floor) and `top_bowler` Brier moved by +0.0001, well inside noise. The
  cohort is simply too small to matter at the market layer: 1,059 of 5,835
  player-lineup rows, each already priced near zero by both the sim and the
  B9 usage baseline. Removing a ~0.5% share from ~18% of rows redistributes
  ~0.09 percentage points of innings share, and the baseline moves with it,
  so the *margin* is unchanged.
- **Surprise 2 — GATE 1b moved the wrong way by 6.5e-5.** `margin_b13 >
  margin_blind`, i.e. the damping made the sim marginally *worse* relative to
  the usage baseline. The paired CIs on both arms are ~±0.0009 wide, so this
  is a coin-flip-scale difference and the honest reading is "no movement",
  but the pre-committed test is a point comparison and it fails.
- **Surprise 3 — the head-only row count dropped 3897 → 3761.** Damping moves
  136 rows below the 2% `p_sim` cut. That is a composition change in a
  context-only diagnostic, not a gate input, but it means the two head-only
  numbers are not on identical row sets and should not be compared as a
  paired delta.
- **Relaxation triggers 8 → 32** as flagged in the plan's item 5. Not an
  explosion (2.0% of cells), but worth knowing: 24 extra (lineup, phase)
  cells now fall all the way back to flat-α legacy weights, which partially
  undoes both the B10 debutant fix and the B13 damping on exactly the
  thin-squad sides (BBL/PSL) where usage data is scarcest. If B13 is ever
  revisited, `min_share` should be re-derived jointly with the damping rather
  than inherited from B10.
- **`other_unknown` collateral (5.548% → 3.487%)** is not covered by any pass
  condition and was not predicted in the plan. It is a real behavioural
  change to a 436-row cohort.
