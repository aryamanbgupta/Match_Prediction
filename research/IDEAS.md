# Research Queue

Statuses: `PENDING` → `RUNNING <ts>` → `LANDED` / `TABLED` / `FAILED` / `CRASH`.
Rules: pull the highest-priority PENDING; append-only edits to history; new
ideas go at the bottom of their priority tier with a one-line rationale; max 2
new ideas appended per iteration; never delete an entry. TABLED ideas are raw
material for combination ideas (`C<n>`) once PENDING runs dry. Verdict logic
lives in `program.md` — both gate metrics up = LANDED, one = TABLED, none =
FAILED.

---

## A1 [P0] [LANDED] Fresh baseline + seed-variance floor
**Hypothesis:** the M7 headline (LL 0.6299 / ROI +21.90%) carries seed luck;
every later verdict needs an honest noise floor.
**Method:** retrain the M7 config (trainer defaults) 5× with different seeds
(add a `--seed` flag to `scripts/xgboost_match_v1.py` if not exposed — trainer
is editable, eval is not) → eval each via recipe A → record mean ± spread of
LL and ROI on ≥$50k. Append one `baseline` row to results.tsv with the mean,
and the spread in notes.
**Gate:** none — this is instrumentation; verdict is `LANDED` if 5 evals
complete. **Budget:** ~45 min.
**Result:** LANDED 2026-07-11. Fresh baseline mean (5 seeds, ≥$50k): LL
0.6318, ROI +20.56%. Noise floor: std 0.0068 LL / 2.3pp ROI. Models kept at
`models/auto/a1_seed*` for A2. See `research/reports/auto/A1.md`.

## A2 [P1] [FAILED] Seed bagging (5-seed probability ensemble)
**Hypothesis:** averaging predicted probabilities across the 5 seeds from A1
removes seed variance and improves both LL and ROI (variance reduction, no new
information — historically the safest kind of win).
**Method:** average `p_team1` across A1's five `test_predictions.json` files
into one prediction set → recipe A steps 2–3.
**Gate:** LL + ROI vs A1's fresh baseline mean. **Budget:** ~20 min (reuses A1
models; if A1 hasn't run, do its training step first).
**Result:** FAILED 2026-07-11. Bag ≥$50k: LL 0.6308 (vs baseline mean 0.6318 =
+0.0010, inside the 0.007 seed-std floor → not improved), ROI +18.23% (vs
+20.56% = −2.33pp → declined). Neither gate metric improved. Bagging five
*correlated* seeds gives only the tiny Jensen LL gain and no ROI help (ROI is a
threshold function, not a smooth mean). Code reverted (d994bbd); models kept at
`models/auto/a2/`. See `research/reports/auto/A2.md`.

## A3 [P1] [FAILED] Direct + v7 sim blend, fine w-sweep
**Hypothesis:** a small sim weight (w ∈ 0.05–0.3) adds ball-level information
the direct model lacks, helping the close-match slice without hurting the
headline.
**Method:** `blend_eval_json.py` with `--w 0.05 0.1 0.2 0.3` using the
existing envelope `eval_out_phase5_hier/hier_all_20260425_165622.json` +
production `test_predictions.json` → reslice each → compare ≥$50k and
close-match slice. Pick best w on ≥$50k only (avoid slice-shopping).
**Gate:** LL + ROI at the chosen w vs w=0. Pure eval composition — no
training. **Budget:** ~30 min.
**Result:** FAILED 2026-07-11. ≥$50k LL degrades MONOTONICALLY with w
(0.6299 → 0.6317 → 0.6339 → 0.6396 → 0.6470 at w=0/0.05/0.1/0.2/0.3); ROI
noisy, never clears the 2pp floor (best w=0.20 +0.31pp, with LL +0.0097
worse). Close-match slice worse at every w. v7 sim carries no winner-market
info the direct model lacks (loses that race by ~0.07 LL) — blending only
injects noise. Neither gate metric improved. Eval-only, nothing to revert.
See `research/reports/auto/A3.md`.

## A4 [P2] [TABLED] Alternative architecture: regularized logistic stack
**Hypothesis:** a heavily regularized logistic regression on the same 49
features is a different bias class; even if it loses standalone it may
ensemble well (if standalone fails but blend helps, TABLE with that note).
**Method:** train sklearn LogisticRegression (existing dep) on the same
parquet splits → recipe A. Optionally eval a 50/50 logit-average with M7.
**Gate:** LL + ROI vs fresh baseline. **Budget:** ~40 min.
**Result:** TABLED 2026-07-11. Standalone logistic (43 signed-diff features,
StandardScaler, val-selected C=0.003) FAILS: ≥$50k LL 0.6390 (+0.0072 worse
than A1 fresh mean 0.6318, beyond floor) / ROI +21.43% (+0.87pp, sub-floor).
The 50/50 logit-avg with M7 moves **ROI +23.62%** (+3.06pp vs fresh mean,
clears 2.3pp floor) but **NOT LL** (0.6298 = −0.0020, inside 0.007 floor) →
exactly one gate metric → TABLED. Blend ROI beats both parents (real ensemble
effect) but vs M7-*alone* the logistic adds only +1.72pp ROI / −0.0001 LL
(both sub-floor) — part of the win is M7 seed luck. Code reverted; `models/
auto/a4/` JSONs kept on disk. **Combine candidate** (C-series once PENDING dry):
3-way logit-avg {logistic, A1 seed-bag mean, M7} to keep ROI decorrelation
while averaging out the seed-luck confound. See `research/reports/auto/A4.md`.

## A5 [P2] [FAILED] Feature interactions: ELO × venue, toss × venue
**Hypothesis:** matchup strength conditioned on venue character (chase bias,
scoring dist) carries signal the additive features miss.
**Method:** BEFORE training, run the correlation check (per repo discipline:
|r| > 0.5 vs an existing M1/M2 feature requires clearly higher target
correlation — see `feedback_correlation_check_before_features`). If it
passes: re-materialize to `data/auto/a5/`, train, recipe A.
**Gate:** LL + ROI vs fresh baseline. **Budget:** ~2 h (materialization).
**Result:** FAILED 2026-07-11. Interactions are products of existing columns →
no re-materialization needed (built 6 mean-centered ELO/toss × venue terms,
train-only centering). Correlation check: all 6 PASS redundancy (max |r_existing|
0.06–0.12) but FAIL the M6 target-corr floor — |r_target| 0.005–0.032, only
`ix_elo_top6_x_avgscore` (0.032) clears 0.03, every one weaker than the weakest
meaningful existing feature. Trained anyway (materialization free) at A1's same
5 seeds for a paired test: mean LL 0.6322 vs A1 base 0.6318 (ΔLL +0.0004,
sub-floor → not improved); mean ROI +18.08% vs +20.56% (ΔROI −2.48pp, beyond
floor, down in 4/5 seeds → declined). Neither gate metric improved. Depth-4 XGB
already captures interactions natively; explicit products only add variance.
5th match-level feature direction to die at the corr-check (M3–M6, A5) —
additive frontier exhausted. Eval-only (gitignored scratch), nothing to revert.
See `research/reports/auto/A5.md`.

## A6 [P2] [RUNNING 2026-07-11T06:36:38Z] New data: historical weather (dew proxy)
**Hypothesis:** evening humidity/dew at the venue affects chase advantage
(wet ball), which the model can't currently see; strongest for night games.
**Method:** GET-only pull from the open-meteo historical archive for venue
lat/lon + match date (save raw under `data/external/weather/`); join
evening-hours humidity/temp as 2–3 match features; re-materialize to
`data/auto/a6/`; train; recipe A. Respect the correlation-check discipline.
**Gate:** LL + ROI vs fresh baseline. **Budget:** ~3 h — start only if >4 h
of night remain.
**Result:** —

## A7 [P2] [LANDED] Betting layer: slice-conditional edge threshold (M8 follow-up)
**Hypothesis:** requiring ~10% edge on mismatch fixtures (top6 ELO diff > 5)
while betting flat elsewhere improves ROI without touching predictions (M8
documented this but never forward-tested it as a rule).
**Method:** re-score production `test_predictions.json` bets under the
conditional threshold using the reslice outputs; compare ROI/CI.
**Gate:** betting-layer rule — ROI improves, predictions untouched → LANDED.
**Budget:** ~30 min.
**Result:** LANDED 2026-07-11. Pure betting-layer filter on frozen production
pnls (`scripts/auto/a7_conditional_threshold.py`; LL unchanged = 0.6299).
≥$50k: baseline flat-thr-0 +21.90%/168 bets → conditional (close |elo_diff|≤5
bet flat, mismatch >5 require edge>10%) **+36.93%/109 bets** — ΔROI **+15.03pp**
(>> 2pp floor), and NOTHING degrades: win 51.8→56.0%, max drawdown 12.52→5.91,
total profit 36.79→40.25u, ROI CI lo +2.28→+12.06. The 59 dropped mismatch/
low-edge bets are a net-losing subset (−5.86% ROI), reproducing M8's
over-confidence-on-lopsided finding. 100k directionally consistent
(+26.39→+35.86%, CI lo −0.99 at n=72). Recommended production sizing becomes
slice-conditional (wiring into a live betting harness is human follow-up;
`predict_fixture.py` only emits probabilities). Kept commit. C-series candidate:
stack on A4's logit-avg (orthogonal — probability ensemble × bet filter). See
`research/reports/auto/A7.md`.

## A8 [P3] [PENDING] Sim: phase-conditional vector scaling (E5 follow-up)
**Hypothesis:** one global vector calibrator over/under-corrects by phase;
per-phase (PP/mid/death) scaling removes the residual bowler-wicket overshoot
without breaking what E5 fixed.
**Method:** fit 3 phase-bucket vector scalers on the same val ball data as
`vector_scaling_calibrator_v1.pkl`; add as `--ball-calibrator vector-phase`;
recipe B on n=261.
**Gate (sim pair):** wicket/PP overshoot vs single-vector run AND top_bowler
margin over fair baseline must not regress. **Budget:** ~2.5 h.
**Result:** —

## A9 [P3] [PENDING] E4 quantile pooling forward test
**Hypothesis:** pooling across the quantile ensemble (filed at E4 as a
forward-test hypothesis) improves tail calibration of match predictions.
**Method:** per the E4 report's stated recipe (`reports/` E-series); eval via
recipe A.
**Gate:** LL + ROI vs fresh baseline. **Budget:** ~1 h.
**Result:** —

## A10 [P3] [PENDING] Sim signal as a direct-model feature (A3 follow-up)
**Hypothesis:** post-hoc probability blending fails (A3: any sim weight strictly
worsens LL), but a sim-derived scalar fed as a *feature* into the direct model
lets the tree learn where/whether to trust ball-level info instead of averaging
it in blindly.
**Method:** extract one or two scalars per match from the v7 sim envelope
(simulated P(team1 win); optionally projected margin spread), join as new
column(s) into the match parquet. BEFORE training, run the correlation check vs
the existing 49 M1/M2 features (repo discipline: |r|>0.5 vs an existing feature
needs clearly higher target correlation). If it passes: re-materialize to
`data/auto/a10/`, train, recipe A.
**Gate:** LL + ROI vs fresh baseline. **Budget:** ~1 h (envelope already
exists; join is cheap, retrain is fast — no full re-materialization of the
career trackers needed if only appending sim columns).
**Result:** —

## A11 [P3] [PENDING] A7 boundary sweep — is |elo_diff|=5 the right mismatch cut?
**Hypothesis:** A7 landed the slice-conditional threshold with the mismatch/close
boundary fixed at |top6_batting_elo_diff|=5 (inherited from the M8 write-up) and
edge>10%. The edge threshold is robust (A7's 0.05/0.10/0.15 sweep all improved),
but the *boundary* was never varied — a different cut may separate the
over-confident lopsided bets more cleanly.
**Method:** rerun `scripts/auto/a7_conditional_threshold.py` at boundaries
{3, 5, 8, 12} with edge>10% fixed. To avoid slice-shopping, DO NOT pick the
max-ROI boundary post hoc: pre-commit that A7's landed boundary=5 stays
production unless another boundary beats it on BOTH ≥$50k and ≥$100k ROI by
more than the 2pp floor AND keeps ≥$50k CI excluding 0. Report the full curve.
**Gate:** betting-layer rule — a challenger boundary must clear the pre-committed
dual-slice bar to replace 5; else A7's rule stands (idea = FAILED, no change).
**Budget:** ~20 min (reuses A7 tooling + existing eval JSONs, no retraining).
**Result:** —

---

## Combination ideas (C-series)
Created only when no PENDING ideas remain, from TABLED entries. Follow
PROTOCOL step 1.
