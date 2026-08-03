# B18 — orchestrator plan (executor handoff)

Idea: **B18 [P2] Empirical extras rates on the promoted i7 quote path**
(B17 attribution; D3 re-applied where it can work). Claimed
`RUNNING 2026-08-03T07:13:57Z` @ `336524e` (a prior session died right after
the claim; this session resumes the same idea — standard relay).

## Hypothesis (from IDEAS.md, verbatim in substance)

B17 attributed the i7 quote-path P50 under-prediction (−4.78/−3.03/−1.95
remaining runs at checkpoints 6/10/15) to carried run mass:
g_i7 = **−0.052785** runs/legal ball (92.7% of the cp6 bias), of which
**−0.039559** is the flat 1%+1% extras graft under-carrying explicit extras
(0.0200 grafted vs 0.059559 actual), −0.0053 the threes fold, and only
−0.0079 the 6-class head. D3's empirical-rates graft FAILED its guard gate
on legacy because the legacy serving path over-carries +0.0620 through the
6-class channel. The promoted no-weights i7 head is marginally neutral, so
the same fix should now close most of the gap WITHOUT the innings-total
regression — that is the falsifiable claim.

## Engine-slot note (orchestrator ruling, not yours to revisit)

B13 ran an engine arm earlier tonight but was **reverted**
(`sim_v1_2.py` byte-identical to pre-B13); B16/B17 were non-engine. The
one-engine-idea-per-night rule protects paired-comparison validity: the
canonical baseline `models/auto/d16/detail_noweights_raw_s46_n261.json` was
produced by an engine byte-identical to current HEAD, so pairing B18 vs
d16 at seed 46 is clean. You MUST verify this before implementing (step 0).

## Design decisions (already made — implement, don't re-decide)

1. **Two components, one opt-in unit.**
   (a) **Rates**: recover D3's `graft_extras()` marginal-preserving
   composition — scale the calibrated 6-class block by (1−p_wide−p_nb),
   set wide/no-ball mass exactly; p_wide **0.037702**, p_no_ball
   **0.004409** (D3 val-split anchors, 2024-12-31..2025-06-30, 545
   matches / 124,292 deliveries). D3 implement commit `8dfda3a`, reverted
   at `87f4fe6` — recover the diff via `git show 8dfda3a`.
   (b) **Event-run crediting**: wide/no-ball events currently credit
   exactly 1 run + re-delivery (B17 verified). Under the graft config,
   credit an INTEGER sampled from the empirical val distribution of
   per-event extras runs (wides mean ≈1.204, no-balls extras-portion mean
   ≈1.071 — recompute exactly; see Fit below). Use the SAME seeded RNG
   stream that draws outcomes, so `--seed` reproducibility holds.

2. **Activation = model-dir sidecar, auto-detected.** Mirror the venue
   encoder resolution (`sim_v1_2.py:1145–1152`): at `XGBoostModelV2` init,
   if `<model_dir>/extras_graft_v1.json` exists, load it and print a banner
   `B18 empirical extras graft ACTIVE (p_wide=…, p_no_ball=…, mean_runs
   w=…/nb=…)`. When absent → flat 1%+1% credit-1 behavior, byte-identical,
   ZERO extra RNG draws consumed. Only the XGBv2 wrapper needs sidecar
   detection (it is the only wrapper on the i7 path); the other wrappers
   keep legacy behavior via default parameters. Legacy replay
   (`models/xgb_v3/…`) has no sidecar → untouched.

3. **B18 model dir**: copy EVERY file of `models/xgb_i7_noweights_production/`
   into `models/auto/b18/` and verify md5 equality file-by-file (booster md5
   must be `7ee1e180…` — record full hash). Then add `extras_graft_v1.json`.
   Never write into the production dir.

4. **Scope guard**: do NOT graft byes/leg-byes on legal deliveries, do NOT
   touch the threes fold, do NOT touch `run_rate`/selector/calibrator paths.
   Expected residual after the fix: g_i7 ≈ −0.020 (byes/legbyes ≈ −0.0072 +
   threes −0.0053 + head −0.0079 remain). The cp6 bias should roughly
   shrink by the ratio 0.0205/0.0528, i.e. −4.78 → ≈ −1.9.

## Fit (`scripts/auto/b18_fit_extras_graft.py`)

- Population: the exact B17 val frame (`data/xgb_data_i7/…validation.parquet`
  joined to cricsheet deliveries as `scripts/auto/b17_runmass_audit.py` does
  — reuse its loading/verification code; rows == ALL deliveries).
- Reproduce the D3 rate anchors to 6 dp: p_wide 0.037702 / p_no_ball
  0.004409. If they don't reproduce, STOP and record.
- Per-event run distributions, using the SAME extras-channel accounting as
  `b17_runmass_audit.py` (so the fix targets the measured deficit):
  wides → distribution of total extras runs on wide deliveries; no-balls →
  extras-portion only (exclude off-bat runs — those are folded into the
  6-class labels; that is why the idea text says nb mean ≈1.071).
  Sanity: recomputed means ≈1.204 (wides) / ≈1.071 (no-balls). If either
  differs by >0.05 from those values, STOP and record the discrepancy
  (the attribution arithmetic would be wrong) — do not proceed to evals.
- Write `models/auto/b18/extras_graft_v1.json`: rates + integer run
  distributions (support + probabilities) + means + fit provenance
  (population hash/counts, date range).
- Analytic pre-check (before any sim): recompute the B17 teacher-forced
  run-mass audit arithmetic under the new graft: extras carried per legal
  ball = (p_w·r_w + p_nb·r_nb)/(1−p_w−p_nb) ≈ 0.0523 and predicted
  g_i7 ≈ −0.0205. Pre-committed tolerance: predicted g_i7 must land in
  [−0.030, −0.012]; outside → STOP and record.

## Unit checks (`scripts/auto/b18_unit_check.py`; all must pass BEFORE evals)

1. **Default-path inertness**: with no sidecar, the wrapper's output
   distribution is float-EXACT vs pre-B18 behavior on a grid of probability
   vectors, and N≥300 same-seed `simulate_ball` draws produce the identical
   outcome sequence (B13 precedent). `d15_unit_check.py` 30/30 unchanged.
2. **Composition exactness** (D3's `d3_unit_check.py` pattern, adapted):
   with the sidecar, extras mass exact (p_wide+p_nb) and 6-class RELATIVE
   marginals preserved exactly on the live XGBv2 path.
3. **Live draw rates**: ~300k live `simulate_ball` draws on the b18 path:
   wide/no-ball frequencies within 3σ of 0.037702/0.004409; sampled
   event-run means within 3σ of the sidecar means.

## Eval recipe (pre-commit the gate script FIRST, then run; ONE heavy
process at a time; tee ALL raw output to `research/handoff/B18/raw/`)

Step 0 (before implementing): engine parity —
`git diff ea4acdb HEAD -- scripts/sim_v1_2.py` must be EMPTY (ea4acdb =
D16). Also confirm no un-reverted working-tree changes to `sim_v1_2.py`.

Step 1: implement + fit + unit checks. Commit
(`Auto[B18]: implement — …`) BEFORE running any eval.

Step 2: gate script `scripts/auto/b18_gate_analysis.py`, committed BEFORE
any eval output exists, with a MANDATORY self-test (B16 precedent): from
the frozen `models/auto/b16/quotes_i7_s48_n261.json` it must reproduce
B16's logged numbers — raw bias −4.781/−3.026/−1.946, pooled dMAE-vs-naive
−3.417 [−4.878, −2.066], B15-scaled coverage 0.822/0.838/0.792 (B15 scales
1.19/1.09/1.26, shift 0, from
`models/auto/b15/quote_calibrator_scale_only.json`) — before any fresh
number exists. Reuse `b16_gate_analysis.py` machinery.

Step 3: recipe-B paired run (i7 stack, graft arm), seed 46 to pair with
the canonical d16 detail:

```
uv run python scripts/sim_eval/prop_backtest.py \
    --test-dir data/polymarket_test --n-matches all --n-sims 100 --seed 46 \
    --model-path models/auto/b18/xgboost_model_i7.pkl \
    --batter-encoder models/auto/b18/batter_encoder_i7.pkl \
    --bowler-encoder models/auto/b18/bowler_encoder_i7.pkl \
    --feature-columns models/auto/b18/feature_columns_i7.txt \
    --stats-version i7 \
    --detail-out models/auto/b18/detail_b18_s46_n261.json \
    --report-out research/reports/auto/B18_props.md
```

(Arg names per B13's logged invocation; if prop_backtest's argparse
differs for explicit i7 model paths, copy the exact working invocation
from the D16/D17 handoff raw logs. NO `--ball-calibrator`.) Verify startup
banners: i7 sqlite, venue encoder ACTIVE (373), B10 k_u=5.0, run-out
channel ACTIVE, **B18 graft ACTIVE**, no calibrator line. Expect ~22 min
(D17: 1310 s); kill + record CRASH at >50 min.

Step 4: twin quote runs at FRESH seed 49 (B5/B15/val/B16 used 43/45/47/48),
`scripts/auto/b5_inplay_quotes.py` with B16's i7 stack args:
- RAW twin: `--model-path models/xgb_i7_noweights_production/xgboost_model_i7.pkl`
  (+ matching encoder/feature-columns/stats-version i7, `--ball-calibrator
  none`, `--seed 49`, `--out models/auto/b18/quotes_raw_s49_n261.json`).
- B18 twin: same but all model paths → `models/auto/b18/…`,
  `--out models/auto/b18/quotes_b18_s49_n261.json`. Banner must show the
  graft ACTIVE (and NOT on the raw twin).
Expect ~15 min each (B16: 898.8 s); kill at >35 min. Expect 756 rows /
253 matches / 8 skips each; verify the skip lists match B16's.

Step 5: run the gate script on all outputs.

## Gate (pre-committed; = B17's original gate, mapped)

PRIMARY (quote layer, s49 twins):
- **P-A**: |P50 bias| shrinks at ALL THREE checkpoints — point test,
  |bias_b18| < |bias_raw| per cp on the same-seed twins.
- **P-B**: pooled paired dMAE (b18 P50 vs naive run-rate extrapolation,
  cluster-boot by match, 2000 resamples, boot seed 29) CI hi < 0.
PRIMARY MET = P-A AND P-B.

GUARDS:
- **G-1** (recipe-B, paired cluster-boot 2000/s29 vs
  `models/auto/d16/detail_noweights_raw_s46_n261.json`): dBrier on
  `innings_runs_ou_{160_5,170_5,180_5}` and `pp_total_ou_{45_5,50_5,55_5}`
  — NO line CI-clean worse (D3's exact legacy failure mode).
- **G-2**: `batter_runs_mae` no CI-clean regression (same pairing).
- **G-3**: coverage on the certified arm — apply B15 scales (1.19/1.09/1.26,
  shift 0) to the b18 s49 quotes: P10–P90 coverage at cps 6/10/15 ALL in
  [0.70, 0.90]. If out of band, you MAY run the B16 ARM-3 conditional
  (ONE val quote run under the graft, seed 50, refit scales per the B14
  pre-committed rule, re-apply) ONLY if wall clock allows; otherwise G-3
  is honestly NOT met.
GUARDS MET = G-1 AND G-2 AND G-3.

Verdict mapping (orchestrator applies it, not you): both PRIMARY and
GUARDS → LANDED; exactly one → TABLED; neither → FAILED. Report the full
33-family scan as context and flag ANY CI-clean regression anywhere.

If LANDED, ship NOTHING into production dirs (D16 precedent): the sidecar
stays opt-in under `models/auto/b18/`; production adoption is a human
promotion decision.

## Baselines to compare against (verbatim from the log)

- Canonical i7 recipe-B baseline: `models/auto/d16/detail_noweights_raw_s46_n261.json`
  (seed 46; batter_runs_mae 13.8905 context from D16).
- Quote layer (B16, s48, context only — the s49 raw twin is the paired
  reference): raw bias −4.781/−3.026/−1.946; pooled dMAE −3.417
  [−4.878, −2.066]; raw coverage 0.787/0.798/0.684; B15-scaled
  0.822/0.838/0.792.
- B17 attribution constants: g_i7 −0.052785; extras channel −0.039559
  (0.0200 vs 0.059559); threes −0.005300; head −0.007926; A = 1.420308.

## Easy to get wrong

1. **No-ball event runs must be EXTRAS-PORTION only** (penalty + byes),
   NOT total delivery runs — off-bat runs on no-balls are folded into the
   6-class labels and already carried by the head. Crediting totals would
   double-count exactly the way D3 died on legacy. Wides: all runs on the
   delivery are extras (batter cannot score off the bat on a wide).
   Derive both from `b17_runmass_audit.py`'s own extras-channel definition.
2. **RNG discipline**: sidecar-absent path must consume ZERO additional
   random draws (default stream byte-identical); sidecar-present sampling
   must use the engine's seeded RNG (no `np.random` global).
3. **Marginal preservation**: scale the 6-class block AFTER any calibrator
   (there is none on i7 — assert no calibrator loaded) and preserve
   relative class probabilities exactly (D3's composition).
4. **Pairing validity**: recipe-B arm must differ from the d16 baseline
   ONLY by the graft — same seed 46, same stats version, same usage json
   (production `models/bowler_phase_usage.json`, md5 `2e650423…`), same
   selector. Verify banners line-by-line against B13's logged banner block.
5. `models/` and `data/auto/` artifacts are gitignored — that's expected;
   commit code + evidence (logs are teed into `research/handoff/B18/raw/`).
6. `prop_backtest.py` lives in `scripts/sim_eval/` — you may RUN it, never
   edit it. Same for every DO-NOT-CHEAT surface in `program.md`.

## Deliverables

- Commits: implement (+fit+unit output), gate-script pre-commit, eval
  evidence — all prefixed `Auto[B18]:`.
- `research/handoff/B18/raw/`: fit log, unit-check log, both quote-run
  logs, recipe-B log, gate output — raw, not summarized.
- `research/handoff/B18/result.md`: numbers VERBATIM from tool output
  (per-cp biases both twins, pooled dMAE + CI, coverage per arm, every
  guard-line dBrier + CI, batter_runs_mae delta + CI, 33-family scan
  summary, analytic g recompute, banners, wall times, commit SHAs,
  `git diff --stat` vs claim `336524e`, anything that crashed/ran long).
- NO verdict, NO revert, NO `results.tsv`/`IDEAS.md` edits, NO push, NO
  golden, NO second idea, NO background processes left running.
