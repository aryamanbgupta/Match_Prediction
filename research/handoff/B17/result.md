# B17 executor result — i7 in-play P50 bias attribution (diagnostic-only)

**No verdict here.** The orchestrator applies the pre-committed mapping in
`research/handoff/B17/plan.md`. This file reports arithmetic only.

Claim commit `7f56a9a`. Plan commit `bdebd43`. Branch `auto-20260803`.

| commit | message |
|---|---|
| `6302b9d` | `Auto[B17]: implement — quote-bias decomposition + teacher-forced run-mass audit (diagnostic-only)` |
| (this commit) | `Auto[B17]: eval + result.md (no verdict — orchestrator decides)` |

Scope executed: plan steps TASK 1 + TASK 2 only. No `sim_v1_2.py` edit, no
sim/prop/quote run, nothing under `scripts/sim_eval/`, `data/golden/`,
`research/results.tsv` or `research/IDEAS.md` touched. Nothing crashed.
Task 1 ran in ~1 s; Task 2 in ~2 min (two 124,292 × 114 batch scorings ×
two venue arms, plus a 545-match cricsheet pass).

Raw output (every number below is copy-pasteable from these files):

- `research/handoff/B17/raw/decomposition.txt`
- `research/handoff/B17/raw/runmass_audit.txt`
- `models/auto/b17/decomposition.json`, `models/auto/b17/runmass_audit.json`
  (gitignored via `.gitignore:16 models/`, regenerable)

---

## TASK 1 — quote-bias decomposition (existing B16 / B15 quote JSONs)

Inputs: `models/auto/b16/quotes_i7_s48_n261.json` (i7 no-weights RAW, seed 48)
and `models/auto/b15/quotes_s45_n261.json` (legacy v3 + vector calibrator,
seed 45). The legacy twin **does** exist, so the paired contrast was run.
Both files: 756 rows / 253 matches / 8 skips.

### 1. Headline reproduction — PASS on both stacks (tol 0.001)

i7:

```
|  cp |    n |  mean bias |       sd |   logged |   |diff| | repro |
|   6 |  253 |    -4.7806 |   26.484 |   -4.781 |  0.00037 |    OK |
|  10 |  253 |    -3.0257 |   20.791 |   -3.026 |  0.00031 |    OK |
|  15 |  250 |    -1.9460 |   14.958 |   -1.946 |  0.00000 |    OK |
```

legacy:

```
|   6 |  253 |    +4.2589 |   26.933 |   +4.259 |  0.00011 |    OK |
|  10 |  253 |    +2.7767 |   21.270 |   +2.777 |  0.00032 |    OK |
|  15 |  250 |    +0.4100 |   15.585 |   +0.410 |  0.00000 |    OK |
```

### 2. Bias per remaining over / per remaining legal ball

```
i7      |  cp | rem overs | rem legal |       bias |   per over | per legal ball |
        |   6 |        14 |        84 |    -4.7806 |   -0.34147 |      -0.056912 |
        |  10 |        10 |        60 |    -3.0257 |   -0.30257 |      -0.050428 |
        |  15 |         5 |        30 |    -1.9460 |   -0.38920 |      -0.064867 |

legacy  |   6 |        14 |        84 |    +4.2589 |   +0.30421 |      +0.050701 |
        |  10 |        10 |        60 |    +2.7767 |   +0.27767 |      +0.046278 |
        |  15 |         5 |        30 |    +0.4100 |   +0.08200 |      +0.013667 |
```

### 3. Paired segment rates (bias accrued inside each segment)

```
i7      |  segment | n paired | mean seg bias |   per over | per legal ball |
        |    6->10 |      253 |       -1.7549 |   -0.43874 |      -0.073123 |
        |   10->15 |      250 |       -1.4560 |   -0.29120 |      -0.048533 |
        |   15->20 |      250 |       -1.9460 |   -0.38920 |      -0.064867 |

legacy  |    6->10 |      253 |       +1.4822 |   +0.37055 |      +0.061759 |
        |   10->15 |      250 |       +2.0140 |   +0.40280 |      +0.067133 |
        |   15->20 |      250 |       +0.4100 |   +0.08200 |      +0.013667 |
```

i7's deficit is present in every segment (−0.049 to −0.073 runs per legal
ball, no monotone death concentration). Per the plan's own reading rubric
("flat ≈ per-ball run-mass channel; concentrated at the death ≈ different
mechanism"), this is the flat profile.

### 4a. Bias by wickets-at-checkpoint band

```
i7      |  cp |  band |    n |  mean bias |   per over | mean runs@cp |
        |   6 |   0-2 |  204 |    -5.4167 |   -0.38690 |        50.98 |
        |   6 |   3-5 |   49 |    -2.1327 |   -0.15233 |        38.16 |
        |   6 |    6+ |    0 |       +nan |       +nan |          nan |
        |  10 |   0-2 |  124 |    -6.3669 |   -0.63669 |        86.85 |
        |  10 |   3-5 |  122 |    -0.0041 |   -0.00041 |        71.81 |
        |  10 |    6+ |    7 |    +3.5000 |   +0.35000 |        61.43 |
        |  15 |   0-2 |   29 |    -0.5172 |   -0.10345 |       138.93 |
        |  15 |   3-5 |  174 |    -2.4109 |   -0.48218 |       124.76 |
        |  15 |    6+ |   47 |    -1.1064 |   -0.22128 |        95.51 |

legacy  |   6 |   0-2 |  204 |    +4.9534 |   +0.35382 |        50.98 |
        |   6 |   3-5 |   49 |    +1.3673 |   +0.09767 |        38.16 |
        |   6 |    6+ |    0 |       +nan |       +nan |          nan |
        |  10 |   0-2 |  124 |    +1.2258 |   +0.12258 |        86.85 |
        |  10 |   3-5 |  122 |    +4.4590 |   +0.44590 |        71.81 |
        |  10 |    6+ |    7 |    +0.9286 |   +0.09286 |        61.43 |
        |  15 |   0-2 |   29 |    +2.4310 |   +0.48621 |       138.93 |
        |  15 |   3-5 |  174 |    +0.4138 |   +0.08276 |       124.76 |
        |  15 |    6+ |   47 |    -0.8511 |   -0.17021 |        95.51 |
```

### 4b. Bias by score-at-checkpoint tercile

```
i7      |  cp | tercile |   runs@cp range |    n |  mean bias |   per over |
        |   6 |      T1 |            9-43 |   90 |    +0.6111 |   +0.04365 |
        |   6 |      T2 |           44-53 |   81 |    -4.2716 |   -0.30511 |
        |   6 |      T3 |          54-105 |   82 |   -11.2012 |   -0.80009 |
        |  10 |      T1 |           24-71 |   87 |    +3.2126 |   +0.32126 |
        |  10 |      T2 |           72-86 |   86 |    -6.3256 |   -0.63256 |
        |  10 |      T3 |          87-132 |   80 |    -6.2625 |   -0.62625 |
        |  15 |      T1 |          48-110 |   85 |    -0.1059 |   -0.02118 |
        |  15 |      T2 |         111-129 |   83 |    -1.5181 |   -0.30361 |
        |  15 |      T3 |         130-188 |   82 |    -4.2866 |   -0.85732 |

legacy  |   6 |      T1 |            9-43 |   90 |    +8.2889 |   +0.59206 |
        |   6 |      T2 |           44-53 |   81 |    +5.2222 |   +0.37302 |
        |   6 |      T3 |          54-105 |   82 |    -1.1159 |   -0.07970 |
        |  10 |      T1 |           24-71 |   87 |    +7.7126 |   +0.77126 |
        |  10 |      T2 |           72-86 |   86 |    +0.1977 |   +0.01977 |
        |  10 |      T3 |          87-132 |   80 |    +0.1812 |   +0.01812 |
        |  15 |      T1 |          48-110 |   85 |    +2.5529 |   +0.51059 |
        |  15 |      T2 |         111-129 |   83 |    +0.7771 |   +0.15542 |
        |  15 |      T3 |         130-188 |   82 |    -2.1829 |   -0.43659 |
```

Both stacks show the same shape in score tercile (bias falls as the
checkpoint score rises) — i.e. both under-respond to the innings' own
in-progress scoring rate; they differ by a near-constant offset.

### Paired i7 − legacy contrast (identical rows, 756/756 matched)

```
|  cp |    n |   i7 bias | legacy bias |     delta | delta/over | delta/legal ball |
|   6 |  253 |   -4.7806 |     +4.2589 |   -9.0395 |   -0.64568 |        -0.107613 |
|  10 |  253 |   -3.0257 |     +2.7767 |   -5.8024 |   -0.58024 |        -0.096706 |
|  15 |  250 |   -1.9460 |     +0.4100 |   -2.3560 |   -0.47120 |        -0.078533 |
```

(The plan's derived cp6 sign-flip delta of −0.1076 per legal ball is
confirmed exactly at −0.107613 from the row-paired computation.)

---

## TASK 2 — teacher-forced run-mass audit, both stacks

### Engine composition (derived from `scripts/sim_v1_2.py`, verified — not assumed)

- `XGBoostModelV2.predict_next_ball` (`sim_v1_2.py:1637-1644`) applies the
  ball calibrator **first** (`:1615-1621`), then sets
  `outcome_probs['wide'] = 0.01`, `outcome_probs['no_ball'] = 0.01` and
  renormalises over all 8 keys. So per delivery
  `p'_c = p_c/1.02` and `p'_wide = p'_nb = 0.01/1.02 = 0.009804`.
  This confirms the plan's note 2: the legacy calibrator runs **pre-graft**.
- `T20Rules.process_ball` (`:860-862`) credits a legacy-path WIDE or NO_BALL
  **exactly 1 run** (`team_runs is None` branch), and `MatchState.update`
  (`:345`) sets `is_legal = outcome not in [WIDE, NO_BALL]`, so the ball
  counter does not advance → re-delivery. A legacy-path no-ball produces
  **no** off-bat runs; the off-bat-on-no-ball composition at `:934` is
  reached only under the I5 `legal_off_bat_v1` branch.
- Renewal identity: per legal ball there is exactly one terminating 6-class
  draw, plus an expected `(0.02/1.02)/(1/1.02) = 0.02` wide/no-ball events at
  1 run each. Hence

  ```
  E_delivery = (Σ_c p_c·runs_c)/1.02 + (0.01/1.02)·1 + (0.01/1.02)·1
  M          = E_delivery / (1 − 0.02/1.02) = 1.02 · E_delivery
             = R_model + 0.02          [runs per LEGAL ball]
  ```

  with strict `runs_c = {dot:0, one:1, two:2, four:4, six:6, wicket:0}`.
  This matches the plan's reference shape exactly; the engine does not differ.
- Second-order engine details deliberately **not** modelled analytically
  (they apply equally to both stacks): `is_legal_outcome` converts a drawn
  wide/no-ball to DOT on ball 119, and a drawn wicket to DOT at 10 down.

### Step 1 — frames, match sets, and ROW SEMANTICS (established empirically)

```
  i7 matches 545   legacy matches 545   intersection 545   i7-only 0   legacy-only 0
  i7 rows 124,292   legacy rows 124,292
  window i7 2024-12-31..2025-06-29   legacy 2024-12-31..2025-06-29
```

Both val frames are the same 545 matches / 1,088 innings / 124,292 rows —
**no intersection drop**. This is exactly the D3 val bucket
(545 matches, 124,292 deliveries, 4,686 wides, 548 no-balls).

Per-match row counts vs cricsheet, first 10 matches, **per frame**
(identical tables for i7 and legacy):

```
  |   match_id |  pq rows | cs deliv | pq legal | cs legal | rows==deliv | legal match |
  |    1439899 |      203 |      203 |      197 |      197 |        True |        True |
  |    1439900 |      243 |      243 |      236 |      236 |        True |        True |
  |    1439901 |      251 |      251 |      240 |      240 |        True |        True |
  |    1439902 |      246 |      246 |      238 |      238 |        True |        True |
  |    1439903 |      188 |      188 |      183 |      183 |        True |        True |
  |    1443073 |      217 |      217 |      207 |      207 |        True |        True |
  |    1443074 |      239 |      239 |      237 |      237 |        True |        True |
  |    1443075 |      240 |      240 |      229 |      229 |        True |        True |
  |    1443076 |      229 |      229 |      223 |      223 |        True |        True |
  |    1443078 |      252 |      252 |      240 |      240 |        True |        True |
  sample: rows==cricsheet deliveries on 10/10; parquet legal==cricsheet legal on 10/10
```

**ROW SEMANTICS VERDICT: parquet rows are ALL DELIVERIES** (wides and
no-balls are rows, with their runs folded into the 6-class label);
legal balls = rows with `is_wide==0 and is_noball==0`
(119,058 of 124,292 = 1.043962 deliveries per legal ball, vs the engine's
assumed 1.02). All quantities below are per **legal** ball.

Full cricsheet aggregate over the same 545 matches, and the parquet
cross-check (zero disagreement):

```
  cricsheet: 124,292 deliveries, 119,058 legal, 169,099 runs
  cricsheet A (all runs / legal balls) = 1.420308
  cricsheet channels per legal ball: off-bat 1.342505  wides 0.047405  no-balls 0.004930
                                     byes 0.006862  leg-byes 0.018604  penalty 0.000000
  cricsheet p_wide 0.037702 (D3 anchor 0.037702)   p_no_ball 0.004409 (D3 anchor 0.004409)
  parquet  : 124,292 deliveries, 119,058 legal, 169,099 runs   A = 1.420308
  parquet-vs-cricsheet delta: deliveries +0  legal +0  runs +0  A +0.000000
```

Both D3 anchors reproduce to 6 dp.

### D16 `marginal_audit.json` cross-check

The D16 sidecar audits the **test** split
(`data/xgb_data_i7/cricket_data_i7_test.parquet`, 186,667 balls), not
validation, so it is a reference not a replication target. Its venue_on
numbers, verbatim:

```
    dot     pred 0.310994  actual 0.312674  delta -0.001680
    one     pred 0.394239  actual 0.397146  delta -0.002907
    two     pred 0.074312  actual 0.075670  delta -0.001357
    four    pred 0.105789  actual 0.105793  delta -0.000004
    six     pred 0.054762  actual 0.052002  delta +0.002760
    wicket  pred 0.059907  actual 0.056716  delta +0.003191
    pred_runs_per_ball 1.294591  actual 1.283666  delta +0.010926
```

**No disagreement in sign or magnitude** with the B17 val recompute below
(same small positive wicket/six tilt, same small negative dot/one tilt,
`delta_runs_per_ball` +0.0109 on test vs the val equivalent
R_model − label mass all rows = 1.347522 − 1.352082 = −0.004560).
The runs delta flips sign between splits, which is a split difference
(test runs/ball 1.2837 vs val 1.3521), not a scoring disagreement.

### Step 2 — per-class marginals, venue_on serving arm, all 124,292 delivery rows

```
| class  |   i7 pred | legacy pred |    actual |      d i7 |  d legacy |
| dot    |   0.29430 |     0.28611 |   0.29728 |  -0.00298 |  -0.01118 |
| one    |   0.40178 |     0.38184 |   0.40367 |  -0.00189 |  -0.02183 |
| two    |   0.07415 |     0.08745 |   0.07160 |  +0.00256 |  +0.01585 |
| four   |   0.11211 |     0.11488 |   0.11259 |  -0.00048 |  +0.00229 |
| six    |   0.05817 |     0.06687 |   0.05914 |  -0.00098 |  +0.00773 |
| wicket |   0.05949 |     0.06286 |   0.05572 |  +0.00377 |  +0.00714 |
```

Per-phase expected 6-class runs per delivery (strict values):

```
  | phase       |  n deliv |        i7 |    legacy |  i7-legacy |
  | pp_0_5      |   40,595 |   1.29540 |   1.37612 |   -0.08072 |
  | mid_6_14    |   57,507 |   1.29568 |   1.36926 |   -0.07358 |
  | death_15_19 |   26,190 |   1.54215 |   1.58743 |   -0.04528 |
```

Context arm: `venue_zero` R_model (all rows) — i7 **1.349357**,
legacy **1.352082**.

### Steps 3–5 — M, A, g, and the channel decomposition

Actual side, exact identity on the scored rows
(`A − (L_legal + residual + extras) = 0.000e+00`):

```
  scored population: 124,292 deliveries, 119,058 legal, 5,234 non-legal;
                     deliveries per legal ball 1.043962 (engine assumes 1.02)
  A                                                        = 1.420308
    L_legal   (6-class label mass on legal balls)          = 1.355449
    residual  (threes/fives folded down, label rounding)   = 0.005300
    extras_act(runs on non-legal deliveries)               = 0.059559   vs graft 0.0200
  parquet actual channel detail (runs per legal ball):
    batter_runs 1.342505  wide_runs 0.047405  noball_runs 0.004930
    bye_runs 0.006862  legbye_runs 0.018604  penalty_runs 0.000000
  class-2 legal balls: n 8,459   mean actual runs 2.049060 vs strict 2.0 (+0.049060 per class-2 ball)
  class-4 legal balls: n 13,749  mean actual runs 4.001527 vs strict 4.0 (+0.001527 per class-4 ball)
```

Composition, **primary** (R_model over all delivery rows — the sim calls
`predict_next_ball` once per delivery):

```
  | stack   |   R_model |         M |         A |   g = M-A |   C_class |  C_extras |    C_fold |
  | i7      |  1.347522 |  1.367522 |  1.420308 | -0.052785 | -0.007926 | -0.039559 | -0.005300 |
  | legacy  |  1.417474 |  1.437474 |  1.420308 | +0.017166 | +0.062025 | -0.039559 | -0.005300 |
  paired: g_i7 - g_legacy = -0.069951
          (== R_model_i7 - R_model_legacy = -0.069951; C_extras and C_fold are
           model-independent and cancel, residual 0.000e+00)
```

Sensitivity (R_model over legal rows only):

```
  | i7      |  1.345577 |  1.365577 |  1.420308 | -0.054731 | -0.009872 | -0.039559 | -0.005300 |
  | legacy  |  1.416086 |  1.436086 |  1.420308 | +0.015779 | +0.060638 | -0.039559 | -0.005300 |
  paired: g_i7 - g_legacy = -0.070510
```

Per-phase g:

```
  | phase       |  n legal |     i7 M |  legacy M |        A |      g i7 |  g legacy |
  | pp_0_5      |   38,688 |  1.31540 |   1.39612 |  1.39240 |  -0.07700 |  +0.00372 |
  | mid_6_14    |   55,552 |  1.31568 |   1.38926 |  1.34413 |  -0.02845 |  +0.04514 |
  | death_15_19 |   24,818 |  1.56215 |   1.60743 |  1.63434 |  -0.07219 |  -0.02691 |
```

**Which channels carry the contrast.** `C_extras` (−0.039559) and `C_fold`
(−0.005300) are algebraically identical for both stacks — they depend only
on the graft constant and on the actuals, not on the model. So the entire
paired contrast is the 6-class channel:
`g_i7 − g_legacy = C_class,i7 − C_class,legacy = R_model_i7 − R_model_legacy
= −0.069951` exactly (checked numerically, residual 0.000e+00).

Two observations that fall out of the table (reported, not adjudicated):

1. The flat graft under-carries explicit extras by **−0.039559** runs per
   legal ball for **both** stacks (0.0200 grafted vs 0.059559 actual — real
   wides alone are 0.047405). The i7 stack's 6-class channel is nearly
   marginally neutral (C_class = −0.0079), so that graft deficit is exposed
   almost in full. The legacy stack's 6-class channel **over**-carries by
   +0.062025, which more than offsets the graft deficit and the fold
   shortfall, netting +0.017166.
2. The legacy over-carry is a serving-path artifact, not a property of the
   calibrator's fit. Under `venue_zero` — the input distribution the E5
   vector calibrator was fit on (`scripts/calibration.py:410-422`) — the
   calibrated legacy stack reproduces the val marginals **exactly**
   (all six per-class deltas 0.00000, R_model 1.352082193917202 vs actual
   label mass per delivery 1.3520821935442346). Under the deployed
   `venue_on` arm (B1's venue-encoder sidecar, autoloaded by
   `XGBoostModelV2.__init__:1148-1157`), the same calibrated stack carries
   1.417474 — **+0.065392** more. i7 shows no comparable venue sensitivity
   (venue_zero 1.349357 vs venue_on 1.347522, −0.001835).

### Step 6 — pre-committed conditions, evaluated numerically

```
  observed cp6 quote bias per legal ball: i7 -4.781/84 = -0.056917
                                          legacy +4.259/84 = +0.050702
                                          sign-flip delta = -0.107619

  CONDITION (a): g_i7 <= -0.0285
      g_i7            = -0.052785
      threshold       = -0.028500
      margin (g - th) = -0.024285
      (a) MET: True

  CONDITION (b): g_i7 - g_legacy <= -0.0538
      g_i7            = -0.052785
      g_legacy        = +0.017166
      g_i7 - g_legacy = -0.069951
      threshold       = -0.053800
      margin          = -0.016151
      (b) MET: True

  BOTH (a) AND (b) MET: True
```

Sensitivities (same conditions, alternate measurement choices):

```
  R_model on legal rows only:
      g_i7 = -0.054731  -> (a) MET: True
      g_i7 - g_legacy = -0.070510  -> (b) MET: True
  A taken from cricsheet instead of the parquet:
      i7:     M 1.367522 - cricsheet A 1.420308 = g -0.052785
      legacy: M 1.437474 - cricsheet A 1.420308 = g +0.017166
```

(The cricsheet-A and parquet-A arms are identical to 6 dp because the two
populations are byte-for-byte the same 545 matches.)

For reference against the quote-side numbers: `g_i7` (−0.052785) is 92.7%
of i7's own observed cp6 bias per legal ball (−0.056917), and
`g_i7 − g_legacy` (−0.069951) is 65.0% of the observed sign-flip delta
(−0.107619).

---

## `git diff --stat 7f56a9a`

```
 research/handoff/B17/plan.md         | 185 +++++++++
 scripts/auto/b17_decompose_quotes.py | 279 ++++++++++++++
 scripts/auto/b17_runmass_audit.py    | 713 +++++++++++++++++++++++++++++++++++
 3 files changed, 1177 insertions(+)
```

(taken before this result commit; the raw/ directory and result.md are added
by the commit that carries this file. `models/auto/b17/` is gitignored.)

## Anomalies / caveats

- None crashed; no step was impossible; no improvisation around the plan was
  needed. The legacy quote twin was present, so the i7-only fallback in the
  plan was not used.
- The parquet val frame contains 46 super-over rows (`inning_idx` 3–8) out of
  124,292. They are inside both the scored population and the cricsheet
  aggregate, so the two sides remain exactly matched; no adjustment made.
- `wkts_at_cp` band "6+" at cp6 is empty (n=0) in both stacks — reported as
  `nan`, not an error.
- Teacher-forcing approximation, stated for the record: `R_model` is the mean
  model expectation over the *validation* delivery population, whereas the
  quote bias is measured on continuation states in the *iteration* test
  window (`data/polymarket_test`, 253 matches). The audit measures carried
  run mass on the split the serving configs were fit/validated against; it
  does not re-score the continuation states themselves.
