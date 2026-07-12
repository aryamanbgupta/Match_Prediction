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
existing envelope `eval_out/phase5_hier/hier_all_20260425_165622.json` +
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

## A6 [P2] [FAILED] New data: historical weather (dew proxy)
**Hypothesis:** evening humidity/dew at the venue affects chase advantage
(wet ball), which the model can't currently see; strongest for night games.
**Method:** GET-only pull from the open-meteo historical archive for venue
lat/lon + match date (save raw under `data/external/weather/`); join
evening-hours humidity/temp as 2–3 match features; re-materialize to
`data/auto/a6/`; train; recipe A. Respect the correlation-check discipline.
**Gate:** LL + ROI vs fresh baseline. **Budget:** ~3 h — start only if >4 h
of night remain.
**Result:** FAILED 2026-07-11. Built the full open-meteo pipeline (geocode 340/464
venues = 73%, hourly archive, evening 18–22h RH/dewpoint; 89.9% test coverage;
joined onto existing parquet — no re-materialization). 3 features incl. an
*oriented* `wx_dew_adv_team1=(RH−trainmean)×(1−2·bat_first)`. Correlation check =
M6/A5 pattern: PASS redundancy (max|r_existing| 0.13–0.14) but FAIL target-floor
(|r_tgt| 0.004–0.008; oriented feat 0.0049 ≪ 0.03). Paired 5-seed (A1 seeds):
mean LL 0.6320 vs base 0.6318 (ΔLL +0.0002, sub-floor → not improved), ROI
+19.04% vs +20.56% (ΔROI −1.52pp, down 3/5, within noise → not improved). Neither
gate up → FAILED. Dew is real physics but weak / already priced by
`venue_chase_win_pct`×`team1_batting_first` at match-level. Code reverted; ~450 MB
weather cache kept at `data/external/weather/` for reuse (see A12). See
`research/reports/auto/A6.md`.

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

## A8 [P3] [FAILED] Sim: phase-conditional vector scaling (E5 follow-up)
**Hypothesis:** one global vector calibrator over/under-corrects by phase;
per-phase (PP/mid/death) scaling removes the residual bowler-wicket overshoot
without breaking what E5 fixed.
**Method:** fit 3 phase-bucket vector scalers on the same val ball data as
`vector_scaling_calibrator_v1.pkl`; add as `--ball-calibrator vector-phase`;
recipe B on n=261.
**Gate (sim pair):** wicket/PP overshoot vs single-vector run AND top_bowler
margin over fair baseline must not regress. **Budget:** ~2.5 h.
**Result:** FAILED 2026-07-11. Null-effect refinement. Implemented cleanly
(global fallback == v1 exactly, max abs diff 0; per-phase vectors diverge 3–7%
from global; selected via `--ball-calibrator-path` with NO eval-framework edit).
Paired Brier_sim (phase − vec), n=261×100 seed 42, cluster-boot by match: GATE 1
tail overshoot (pp_total + bowler_wkts, want DOWN) = pooled dBrier **−0.0001, CI
[−0.0008, +0.0006]** → indistinguishable from 0, mixed per-family signs → NOT
improved. GATE 2 top_bowler (must not regress) = **+0.0002, CI [−0.0003, +0.0007]**
→ flat (no regress, no improve). E5 regressors unchanged (team_first_over_mae
+0.003, batter_runs_mae +0.003, both noise). **Neither gate up → FAILED.** The
global vector already captures the marginal correction; a 3-bucket linear scaler
lacks resolution for the residual (which is plausibly per-over or a dispersion
problem — see report). n=30 smoke looked all-worse but was pure sampling noise.
Code reverted (f532396); scratch kept at `models/auto/a8/` (gitignored). See
`research/reports/auto/A8.md`.

## A9 [P3] [FAILED] E4 quantile pooling forward test
**Hypothesis:** pooling across the quantile ensemble (filed at E4 as a
forward-test hypothesis) improves tail calibration of match predictions.
**Method:** per the E4 report's stated recipe (`reports/` E-series); eval via
recipe A.
**Gate:** LL + ROI vs fresh baseline. **Budget:** ~1 h.
**Result:** FAILED 2026-07-11. Forward-tested E4's `all8` (+8 pooled-ELO
survivors) vs `e4-base` as a **paired 5-seed** test {29,7,13,42,101} on the E4
unfrozen parquet, in E4's exact column order (colsample makes order load-bearing;
seed 29 byte-reproduces E4: base 0.6312/+15.38, all8 0.6288/+24.35). Paired MEAN
≥$50k: **ΔLL +0.0019** (all8 WORSE, sub-floor 0.007; only seed 29 favors all8,
4/5 flat-to-worse) / **ΔROI −0.76pp** (all8 WORSE, sub-floor 2.3; 3/5 seeds
worse). **Neither gate up → FAILED.** E4's promising single-seed readout was
**seed-29 luck** — seed 29 is the sole seed where all8 wins both metrics (ΔROI
+8.97pp). Resolves the E4 open hypothesis negative for the winner market;
vindicates E4's val rule (DISCARD) and the A1/E3 multi-seed discipline. No
production code changed (quantile cols already in the materializer, excluded
since E4); harness `scripts/auto/a9_run.py` kept. See
`research/reports/auto/A9.md`.

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

## A11 [P3] [FAILED] A7 boundary sweep — is |elo_diff|=5 the right mismatch cut?
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
**Result:** FAILED 2026-07-11. Swept boundary {3,5,8,12} @edge>10% fixed on BOTH
slices (`scripts/auto/a11_boundary_sweep.py`, betting-layer, LL unchanged 0.6299;
incumbent b5 reproduces A7 exactly: ≥$50k 109 bets +36.93 [+12.06,+68.89], ≥$100k
72 bets +35.86 [-0.99,+80.58]). The two slices point OPPOSITE: ≥$50k ROI is
monotone-DECREASING in boundary (b3 +40.96 → b12 +27.73), but ≥$100k ROI PEAKS at
b5 (+35.86; b3 +32.56 is −3.30pp WORSE and its CI reopens across 0). Only b3 beats
b5 on ≥$50k (+4.03pp) — by over-filtering (n 168→90) — and loses ≥$100k, so NO
challenger clears the pre-committed dual-slice bar → boundary 5 stands (A7 rule
unchanged). Boundary 5 is CONFIRMED robust: it is the ≥$100k optimum, and win rate
is flat ~54–56% across all boundaries (ROI deltas are pnl-mix, not sharper
discrimination). No production code changed; analysis harness kept (cf. A9). See
`research/reports/auto/A11.md`.

## A12 [P3] [FAILED] Dew as a ball-level second-innings covariate (A6 follow-up)
**Hypothesis:** A6 showed dew carries no *match-level* winner signal, but the
physics (wet ball → harder to grip/field in the chase) is fundamentally a
**ball-level, second-innings** effect that match aggregation washes out. A dew
covariate feeding second-innings scoring/wicket rates in the v7 sim is a cleaner
test of the same physics — and it lives in the sim/prop gate, not the winner gate.
**Method:** reuse the already-built weather cache (`data/external/weather/`,
evening RH per venue/date — no new pulls needed). Join evening RH as a per-ball
feature active only in innings 2 (or as a sim-time multiplier on 2nd-innings
bowling economy / wicket prob). Recipe B on n=261 with `--ball-calibrator vector`.
**Gate (sim pair):** second-innings scoring/wicket calibration vs the current
single-vector run AND no regression of an established sim skill (team-fours or
`top_bowler` margin vs fair baseline). **Budget:** ~2.5 h (data free; sim eval
is the cost).
**Result:** FAILED 2026-07-11 (resumed a prior iteration that fit the calibrator
+ ran the n=261 dew sim but stopped before the verdict). Centered dew-tilt
`sqrt(v_high/v_low)` on the E5 v1 global vector, fit on val innings-2 balls split
at median evening RH 68%; innings-1/no-coverage path == v1 exactly (96.6% inn-2
coverage). Paired Brier/MAE (dew − vec) cluster-boot by match, n=261×100 seed 42
vs `models/auto/a8/detail_vec_n261.json`; pairing sanity holds (n_chg ≈ ½ n_all
on every inn-2 family — only inn-2 obs differ). **GATE 1** (2nd-inns
scoring+wicket, want dBrier DOWN, CI < 0): pooled **−0.0003 CI [−0.0013, +0.0006]
INCLUDES 0 → NOT improved** (scoring +0.0003 [−0.0015,+0.0021]; wicket/econ
−0.0004 [−0.0014,+0.0005], both directionless). **GATE 2** guard: `top_bowler`
+0.0003 CI [+0.0001,+0.0005] **excludes 0** = tiny regression; `team_total_fours_mae`
+0.000 [−0.010,+0.010] held. GATE 1 fails → **FAILED**. Dew signal genuinely weak
(max |tilt−1| = 0.066 on wicket, all others <2%; applied per-ball tilt ≤3.3%,
mostly <1%) — below the A8 washout threshold (A8's 3–7% tilts already netted 0 on
aggregated props). Third dew negative (A6 match-level + A12 ball-level); dew lever
exhausted at both resolutions. Reverted `eabd701`; default sim path byte-unchanged;
harness `scripts/auto/a12_*.py` + `models/auto/a12/` scratch kept. See
`research/reports/auto/A12.md`.

## A13 [P3] [PENDING] Sim dispersion calibration on sampled score totals (A8 follow-up)
*(claim reset by supervisor: the 2026-07-11 iteration was cut mid-eval by the
wall clock — no verdict was reached, safe to re-run)*
**Hypothesis:** A8 showed vector scaling (a *marginal-rate* correction) cannot move
the tail-overshoot props — but the vec baseline report shows the sim **under-disperses**
score totals (`team_first_over` P10–P90 coverage 53% vs ideal 80%; `batter_runs`
72%). Under-dispersion is a *variance* defect, orthogonal to marginal calibration,
and it's exactly what inflates over/under Brier on `pp_total`/`first_wicket`/`highest_over`.
E2's standing result says the sim's only real skill is continuous score forecasts, so
widening the predictive spread to nominal coverage is the validated-ceiling lever.
**Method:** fit a per-family dispersion/temperature scalar on the *validation* sim
(inflate the sampled total's spread around its mean to hit ~80% P10–P90 coverage —
e.g. a multiplicative fan-out on centered per-sim totals, or resampling the per-ball
outcome with a small negative-binomial dispersion). Apply at eval only; recipe B on
n=261 with `--ball-calibrator vector`. Pair vec-vs-dispersion via
`scripts/auto/a8_gate_analysis.py` (reuse the paired-Brier tooling). NO eval-framework
edit — implement inside `sim_v1_2.py` aggregation or a post-hoc transform.
**Gate (sim pair):** P10–P90 coverage moves toward 80% (calibration) AND over/under
tail Brier (`pp_total` + `first_wicket_runs` + `highest_over_runs`) improves paired
vs the single-vector run, with no regression of `batter_runs_mae`. **Budget:** ~2 h
(two n=261 sim runs; the vec baseline `detail_vec_n261.json` may be reused if unchanged).
**Result:** —

## A14 [P3] [LANDED] Per-over (not per-3-phase) ball calibrator for single-over props (A8 follow-up)
**Hypothesis:** A8's 3-bucket phase calibrator netted to null on multi-phase aggregate
props because a bowler's/team's balls span phases and the small per-ball corrections
wash out. But **single-over** props (`team_first_over_mae`, `highest_over_runs_ou_*`)
live inside one over — and the first over is lumped with overs 2–6 in A8's PP bucket,
so a per-over (or smooth spline on `balls_bowled`) scaler has resolution A8's flat
buckets lacked *precisely where it can't wash out*. Lower conviction than A13 (still
linear marginal scaling, which A8 showed is weak), so scoped tightly to the props
where finer indexing is mechanically distinct.
**Method:** fit a scaling vector indexed by over (0–19), or a monotone spline on
`balls_bowled`, on the same val balls as v1; global fallback == v1. Dispatch off the
feature buffer like A8 did (no eval-framework edit). Recipe B on n=261; pair vs the
single-vector run with `a8_gate_analysis.py`.
**Gate (sim pair):** `team_first_over_mae` / `highest_over_runs` Brier|MAE improves
paired vs single-vector AND `top_bowler` + `bowler_wkts` do not regress. If it only
matches A8 (null on single-over props too), FAILED — linear ball-prob scaling is
exhausted, pivot to A13's dispersion lever. **Budget:** ~2 h.
**Result:** LANDED 2026-07-11. Per-over (20×6) vector calibrator; sim gate pair,
paired Brier/MAE (over − vec) cluster-boot by match, n=261×100 seed 42 vs
`models/auto/a8/detail_vec_n261.json`. **GATE 1** (single-over prop improvement):
`team_first_over_mae` dMAE **−0.022 CI [−0.038, −0.006]** — CI cleanly EXCLUDES 0
(vec MAE 3.535 → over 3.513), a real above-noise gain → IMPROVED; `highest_over_runs`
18.5/24.5 both flat (−0.0000/−0.0001). **GATE 2** (guard, no-regress): `top_bowler`
+0.0003 [−0.0003,+0.0008], `bowler_wkts` 1/2/3plus +0.0001/+0.0009/+0.0003, all CIs
include 0 → HELD. Both gate conditions met → LANDED. Vindicates the hypothesis:
A8's 3-phase bucket was null on `team_first_over_mae` (+0.003) because over 0 was
lumped with overs 1–5; per-over resolution recovers the first-over correction
(over-0 vector six/wicket ×1.23). Pooled tail dBrier +0.0002 [−0.0005,+0.0009] noise
(A8's multi-over-aggregate washout reproduces). Global fallback == v1 exactly; sim
engine edit (not eval framework); default v1 path byte-unchanged. Kept `0e1bbb0`;
artifact `models/auto/a14/over_vector_calibrator.pkl` (gitignored, reproducible via
`scripts/auto/a14_fit_over_calibrator.py`). See `research/reports/auto/A14.md`.

## A15 [P3] [LANDED] Minimal over-0-only ball calibrator (A14 parsimony follow-up)
**Hypothesis:** A14 LANDED a 20-vector per-over calibrator, but the entire
`team_first_over_mae` win is concentrated in **over 0** (its vector diverges most
from global, six/wicket ×1.23); the other 19 per-over vectors buy nothing
observable and are pure overfitting surface. A **2-vector** calibrator — one
first-over vector + the global vector for overs 1–19 — should capture the full
first-over gain with 10× fewer parameters, which is more robust and cleanly
isolates *where* the sim's first-over error lives.
**Method:** reuse `scripts/auto/a14_fit_over_calibrator.py` but fit only over 0
(all other overs fall back to global); save `models/auto/a15/over0_calibrator.pkl`
(the `OverVectorScalingCalibrator` already supports a sparse `_v` with global
fallback — no new class needed). Recipe B on n=261 with the over0 calibrator;
pair vs the SAME single-vector baseline (`models/auto/a8/detail_vec_n261.json`)
with `scripts/auto/a8_gate_analysis.py`.
**Gate (sim pair):** `team_first_over_mae` improves paired vs single-vector (must
retain A14's ~−0.022 dMAE, CI excluding 0) AND `top_bowler` + `bowler_wkts` do not
regress. If over0-only matches A14's gain → prefer it (parsimony, LANDED replaces
the 20-vector artifact as the recommended first-over calibrator); if it loses the
gain, the win genuinely needs the full per-over grid (A14 stands). **Budget:**
~1 h (one n=261 sim run; baseline reused).
**Result:** LANDED 2026-07-11. Over-0-only (2 effective vectors: over-0 + global)
vs A14's 20-vector. Refit from the same val balls, keeping only over 0 in `_v`;
overs 1–19 fall back to global. **Byte-identity verified** at fit time (max abs
diff 0.00e+00 on all three: `_global`==v1, A15 over-0 vec==A14 over-0 vec, A15
`_global`==A14 `_global`) → A15 differs from A14 *only* by reverting overs 1–19 to
global, isolating the parsimony question exactly. Sim gate pair, paired Brier/MAE
(over0 − vec) cluster-boot by match, n=261×100 seed 42, baseline
`models/auto/a8/detail_vec_n261.json`. **GATE 1** `team_first_over_mae` dMAE
**−0.018 CI [−0.032, −0.003]** — CI **excludes 0** (vec 3.535 → over0 3.517);
statistically indistinguishable from A14's −0.022 [−0.038, −0.006] (CIs overlap)
→ first-over gain **RETAINED by over 0 alone** → IMPROVED. **GATE 2** guards
`top_bowler` +0.0000 [−0.0003,+0.0003], `bowler_wkts` 1/2/3plus
−0.0002/−0.0003/−0.0002 all CIs include 0 → HELD (no regression). Both conditions
met → LANDED. Direct A15-vs-A14 cross-check: `team_first_over_mae` dMAE +0.004
[−0.004,+0.011] = noise (first over IDENTICAL), and A15 is marginally *cleaner* on
the pooled tail (−0.0007 [−0.0014,−0.0000]) → A14's 19 extra per-over vectors were
slight overfitting. Parsimony confirmed: 10× fewer params, full gain retained, no
guard regress. A15 = recommended minimal first-over calibrator (supersedes A14's
20-vector for that purpose); A14 stays LANDED (established the win). Secondary:
`pp_total_ou_45_5` −0.0037 [−0.0063,−0.0010] also improves from fixing over 0.
Kept `d6202e0`; artifact `models/auto/a15/over0_calibrator.pkl` (gitignored,
reproducible via `scripts/auto/a15_fit_over0_calibrator.py`); default `vector` path
and A14 sim wiring byte-unchanged. See `research/reports/auto/A15.md`.

## A16 [P3] [FAILED] Sparse regime-change-over calibrator (A15 extension)
**Hypothesis:** A15 showed the sim's single largest per-over calibration defect is
**over 0** (fixing it alone earns the full `team_first_over_mae` win AND
`pp_total_ou_45_5` −0.0037, CI excludes 0), while A14's blanket 19 other per-over
vectors are inert/overfit. But over 0 is a "regime-change" over — the first ball a
fresh batter and a new bowler face. The other regime-change overs — **over 6**
(start of the middle overs, field restriction lifts, bowling change) and **over 15**
(start of the death, specialist death bowlers enter) — plausibly carry a similar,
mechanically-distinct marginal miscalibration that a *sparse* calibrator could fix,
whereas the ~17 non-boundary overs stay on global (A15 discipline: add a vector only
where it's mechanically distinct and observable, not a blanket grid).
**Method:** reuse `scripts/auto/a15_fit_over0_calibrator.py` but fit vectors for
overs {0, 6, 15} only (all others fall back to global == v1); save
`models/auto/a16/regime_calibrator.pkl`. Recipe B on n=261 with the calibrator;
pair vs the SAME single-vector baseline (`models/auto/a8/detail_vec_n261.json`) with
`scripts/auto/a8_gate_analysis.py`. Pre-commit: report the per-over vector
divergences BEFORE the sim run; if over 6 / over 15 vectors barely diverge from
global (max|ratio−1| ≲ the ~0.05 threshold below which A8/A12 washed out), expect
null and say so.
**Gate (sim pair):** a phase-boundary prop that lives in the added overs must
improve paired vs single-vector (candidates: `pp_total_ou_*` for over-6-region,
`highest_over_runs_ou_*` for the death) beyond the A15/over-0-only result, AND
`top_bowler` + `bowler_wkts` + `team_first_over_mae` do not regress. If {0,6,15}
only reproduces A15's over-0 gain with no incremental lift → FAILED (over 0 is the
sole miscalibrated regime-change over; A15 stands as the minimal calibrator).
**Budget:** ~1.5 h (one n=261 sim run; baseline reused).
**Result:** FAILED 2026-07-11 (resumed a prior iteration that fit the {0,6,15}
calibrator, logged pre-run divergences, and launched the n=261 sim; this
iteration waited for it to finish and ran the verdict). Fit byte-identical to
A15/A14 (global==v1, over0+global==A15, overs{0,6,15}==A14; max diff 0) → A16
differs from A15 ONLY by carrying over-6 + over-15 vectors. Pre-run divergences
vs global: over 0 **0.230** (wkt ×1.23), over 6 **0.138** (wkt ×1.14), over 15
**0.078** (four ×1.08) — all above the ~0.05 A8/A12 washout threshold, so the
incremental test was worth running. Paired Brier/MAE cluster-boot by match,
n=261×100 seed 42, vs single-vector (`models/auto/a8/detail_vec_n261.json`) AND
A15 over-0-only (`models/auto/a15/detail_over0_n261.json`). **GATE 1**
(INCREMENTAL A16-vs-A15: do overs 6 & 15 add a real gain BEYOND over 0?): of 10
death/middle/pp families only 2 cross significance — `highest_over_runs_ou_24_5`
−0.0008 [−0.0018,**−0.0000**] and `pp_total_ou_55_5` −0.0009 [−0.0017,**−0.0000**]
— **both with CI upper bound AT the noise boundary (−0.0000)**, consistent with
multiple-comparisons chance (10 families @95%). Decisively, `pp_total_ou_55_5` is
a **powerplay prop (overs 0–5)** that overs 6 & 15 cannot mechanically move → its
marginal significance is downstream MC noise, confirming the crossings are noise,
not signal. No clean above-noise incremental gain like A15's over-0 win
(`team_first_over_mae` −0.018, ~20–25× larger, CI well clear of 0) → GATE 1 **NOT
met**. **GATE 2** guards HELD (vs vec: `top_bowler` −0.0001 [−0.0004,+0.0003],
`bowler_wkts` 1/2/3plus CIs include 0, `team_first_over_mae` −0.018 retained;
incremental vs A15: all guards ~noise, first-over dMAE −0.000 = IDENTICAL). Zero
improvements → **FAILED** (honors the pre-committed rule). A16 is strictly
dominated by the simpler A15 (= A15 + 2 inert vectors). Reproduces the A8/A12
aggregation-washout: overs 6 & 15 exceed the ~0.05 marginal threshold but wash
out on **multi-over** props; A15's over-0 fix is the exception only because
`team_first_over_mae` is a **single-over** prop (no aggregation). No fixed-over
prop isolates over 6 or 15, so the per-over linear-scaling lever is exhausted at
over 0. Reverted `27e99c6` (implement `ac3cec9` = fit harness only; sim engine +
default `vector` path byte-unchanged — calibrator loads via runtime
`--ball-calibrator-path`). Scratch `models/auto/a16/` + `scripts/auto/a16_gate_analysis.py`
kept. Pivot to A13 dispersion lever (orthogonal, still PENDING). See
`research/reports/auto/A16.md`.

---

# B-series: ball-level / prop focus (seeded 2026-07-11 by supervisor)

Strategic shift: the match-level model is essentially at market (LL 0.6299 vs
0.6267); remaining match-level alpha is betting rules, not predictions. The
sim is the compounding asset (props, in-play, analytics). B-series priorities
outrank remaining A/C ideas.

## B1 [P0] [TABLED] Fix sim venue blindness: `venue_encoded=0` on every sim ball
**Hypothesis:** `sim_v1_2.py` `XGBoostModelV2._feat_buf` defaults missing keys
to 0, so every simulated ball scores as venue code 0 while training saw real
codes (TODO.md:337, filed at E5). The sim literally cannot see the venue — an
out-of-distribution input on every ball. Fixing it should improve venue-
sensitive props (totals, boundaries) across the board.
**Method:** save a venue encoder at training time; wire `venue_encoder_path`
into the XGB wrapper (LSTM/Transformer wrappers already do this — copy their
pattern). Re-run recipe B paired vs the current vec baseline
(`a8_gate_analysis.py` pairing).
**Gate (sim pair):** pooled prop Brier/MAE improves paired (CI excludes 0) on
venue-sensitive families (team totals, PP totals, fours/sixes) AND `top_bowler`
+ `bowler_wkts` guards hold. **Budget:** ~2.5 h. **Note:** this changes the
sim's input distribution — re-baseline before comparing anything else to
history; do NOT run the same night as other sim-engine ideas.
**Result:** TABLED 2026-07-12 (implemented + gate pre-committed 2026-07-11;
that iteration was wall-clock-killed at eval startup — this one re-ran the
n=261×100 seed-42 eval, 2198.6 s, startup log confirms "venue encoder ACTIVE
(467 venues)"). Encoder rebuild proven training-exact (byte-match of refit
batter/bowler encoders vs saved artifacts); 261/261 test venues covered.
Paired vs `models/auto/a8/detail_vec_n261.json` via pre-committed
`scripts/auto/b1_gate_analysis.py` (230427a). **GATE 1** (PRIMARY = pooled
dBrier over 8 venue-sensitive binary lines): **−0.0025 CI [−0.0074, +0.0024]
STRADDLES 0 → NOT met** (direction favorable, 6/8 lines down;
`pp_total_ou_50_5` alone −0.0084 [−0.0153,−0.0017]; venue-MAE non-regress
held). **GATE 2** guards **HELD** — none regress; two IMPROVE with CI
excluding 0: `bowler_wkts_1plus` −0.0044 [−0.0071,−0.0016] and
`batter_runs_mae` **−0.175 [−0.298,−0.048]**. Exactly one gate satisfied →
TABLED per the pre-committed mapping. Off-gate scan is ONE-SIDED positive
(0 families significantly worse anywhere, ~9 better), concentrated at the
**batter level** (batter_runs_mae = E2's validated continuous skill;
batter_50plus, batter_fours_1/3plus, batter_fours_mae, highest_over_18_5,
team_highest_individual 29_5/34_5) — the gate pooled the wrong families:
venue codes sharpen per-batter forecasts; team-total lines aggregate the
per-ball correction away (A8/A12/A16 washout pattern). Reverted `c37f1b0`
(default sim path was byte-unchanged anyway — encoder loads only via the
`models/auto/b1/` sidecar); gate script + scratch kept. Follow-up: B6
(re-gate on batter_runs_mae primary @ fresh seed). See
`research/reports/auto/B1.md`.

## B2 [P1] [PENDING] Fix `innings_id` unstable hash (blocks parquet↔cricsheet joins)
**Hypothesis:** `parsing_v2.py:1255` sets `innings_id = hash(json) % 100000` —
salted per process (irreproducible across runs), ~450 expected collisions at
corpus size (TODO.md:331, filed at E5/E6). E6 lost ~5% of matches to a
workaround join. Correctness fix that unblocks every future ball↔match join.
**Method:** emit the cricsheet filename stem instead; rebuild feature parquet
(cheap, no schema bump per TODO). Verify: E6's join recovers the ~5% drop;
row counts and a sample of per-ball features byte-match otherwise.
**Gate:** correctness idea — LANDED if joins are lossless and no eval metric
moves beyond eval-only noise (it should be a no-op on metrics). **Budget:** ~1.5 h.
**Result:** —

## B3 [P1] [PENDING] Continuous-forecast shrinkage blend (productize E2's finding)
**Hypothesis:** E2: the sim's only validated skill is continuous score
forecasts (batter-runs MAE −0.71 vs career baseline). A val-fit shrinkage
blend `α·sim + (1−α)·fair_baseline` per family should beat both parents —
the best score forecast the system can produce.
**Method:** fit α per family (batter_runs, highest_individual, team totals) on
val matches; evaluate on recipe B test with `--ball-calibrator vector`;
compare MAE vs sim-alone and baseline-alone (`prop_fair_baselines.py`).
**Gate (sim pair):** blend MAE beats BOTH parents (paired, CI excludes 0) on
≥1 family AND no family regresses vs its best parent. **Budget:** ~2 h.
**Result:** —

## B4 [P2] [PENDING] top_bowler pricing margin (post-calibration edge quantification)
**Hypothesis:** top_bowler is the only binary family beating its fair baseline
(E5). Quantify the margin as implied odds: at what price does the sim's
top_bowler probability have positive EV vs the fair baseline as synthetic
market? Output a pricing table, not a model change.
**Method:** per-match top_bowler probs (calibrated sim, n=261) vs fair
baseline; compute edge distribution, EV curves at hypothetical vig levels
(2/5/10%), Kelly fractions. Pure analysis on existing detail JSONs if
sufficient, else one recipe-B run.
**Gate:** instrumentation (like A1) — LANDED if the pricing table + report
are produced with real numbers. **Budget:** ~1.5 h.
**Result:** —

## B5 [P2] [PENDING] In-play over/under quote prototype (analytics-engine seed)
**Hypothesis:** `models/inplay_winprob_v1` (P(win|state)) + the calibrated sim
(score distributions from any mid-innings state) can produce live over/under
quotes for remaining-innings runs. Feasibility prototype: quote quality
measured against realized outcomes at standard checkpoints (end of overs
6/10/15).
**Method:** for each test match, roll the sim forward from the actual state at
overs 6/10/15 (100 sims, vector calibrator); emit P10/P50/P90 remaining-runs
quotes; score coverage + MAE vs actuals; compare vs a naive
run-rate-extrapolation baseline.
**Gate:** quotes beat the naive baseline on MAE AND P10–P90 coverage within
[70%, 90%] at all three checkpoints. **Budget:** ~3 h — needs sim-from-state
plumbing; if that plumbing doesn't exist, scope THIS iteration to building +
unit-testing it and mark the idea `TABLED (plumbing done, eval next)`.
**Result:** —

## B6 [P1] [LANDED] Venue-encoder fix re-gated on batter-level continuous primary at a fresh seed (B1 follow-up)
**Hypothesis:** B1 (TABLED) fixed the sim's venue blindness but its pre-committed
venue-binary pooled primary straddled 0 (−0.0025 [−0.0074,+0.0024]); the fix's
CI-clean effect concentrated instead in **batter-level** families —
`batter_runs_mae` −0.175 [−0.298,−0.048], E2's single validated sim skill, plus
batter_50plus / batter_fours binaries and MAE — with ZERO families significantly
worse anywhere in the scan. B1's gate targeted the wrong families: venue codes
sharpen per-batter forecasts; team-total lines aggregate the per-ball correction
away. Because the batter-level primary was identified *post hoc* on B1's seed-42
run, it must be confirmed on fresh Monte Carlo draws before it can land.
**Method:** re-apply B1's implementation (`git revert c37f1b0` or cherry-pick
`112c59e`; encoder artifact already at `models/auto/b1/venue_encoder_v3.pkl`,
rebuildable via `b1_build_venue_encoder.py`). TWO recipe-B runs at a **fresh
seed (43)**: venue-on and venue-blind (both new, so B1's seed-42 selection
cannot leak into the comparison); pair with the a8/b1 gate tooling. Write the
gate script BEFORE the runs (A-series discipline).
**Gate (sim pair):** PRE-COMMITTED PRIMARY = `batter_runs_mae` improves paired
(CI excludes 0) at seed 43. Guards = `top_bowler`, `bowler_wkts_{1,2,3}plus`,
`team_first_over_mae`, `team_total_{fours,sixes}_mae`: no CI-excludes-0
regression. Both → LANDED (ship the sidecar into `models/xgb_v3/` as the
default sim path and RE-BASELINE all future sim comparisons — B1's warning
applies); exactly one → TABLED; none → FAILED (B1's batter-level signal was
seed-42 selection noise). **Budget:** ~2 h (two n=261×100 runs, ~37 min each).
**Result:** LANDED 2026-07-12 (prior iteration's seed-43 runs were wall-clock-
killed at startup; this iteration re-ran both from scratch). Two fresh n=261×100
seed-43 runs (blind 2226.3 s / venue 2215.9 s; startup logs confirm "venue
encoder absent" vs "venue encoder ACTIVE (467 venues)"), paired via pre-committed
`scripts/auto/b6_gate_analysis.py` (422c266, written before any seed-43 result).
**GATE 1 PRIMARY** `batter_runs_mae` (3176 obs): dMAE **−0.162 CI
[−0.287, −0.036] EXCLUDES 0 → IMPROVED** — B1's seed-42 −0.175 [−0.298, −0.048]
reproduces on fresh draws; the batter-level signal was real, not selection
noise. **GATE 2 guards ALL HELD**: top_bowler +0.0004 [−0.0002,+0.0011],
bowler_wkts_1plus IMPROVED −0.0037 [−0.0064,−0.0008], 2/3plus noise,
team_first_over_mae +0.012 [−0.031,+0.055], fours/sixes MAE noise. Both →
LANDED. Context scan one-sided again: 0 families significantly worse; CI-clean
better incl. batter_50plus −0.0020, batter_fours_3plus −0.0030, batter_fours_mae
−0.016, team_highest_individual 29_5/34_5 −0.0026/−0.0053, highest_over_18_5
−0.0150, pp_total_50_5 −0.0081. **SHIPPED**: sidecar copied to
`models/xgb_v3/venue_encoder_v3.pkl` (new file; smoke confirms default sim path
now venue-ON). **RE-BASELINE IN FORCE**: all pre-B6 sim detail JSONs (a8 vec,
A14/A15/A16) are venue-blind; canonical venue-ON baseline =
`models/auto/b6/detail_venue_s43_n261.json` (seed 43; blind twin kept). Kept
11a19ea + 422c266. Follow-up B7 appended (stale ball calibrators). See
`research/reports/auto/B6.md`.

## B7 [P1] [PENDING] Refit ball calibrators on the venue-ON sim (B6 re-baseline follow-up)
**Hypothesis:** B6 shipped the venue encoder into the default sim path, changing
the model's input distribution — but both production ball calibrators were fit
on *venue-blind* val predictions: `vector_scaling_calibrator_v1.pkl` (E5) and
the A15 over-0 calibrator (`models/auto/a15/over0_calibrator.pkl`). Their
corrections may now be stale (over- or under-correcting the venue-aware
probabilities); refitting on venue-on val balls should recover any headroom and
re-validates that A15's first-over gain survives the re-baseline.
**Method:** refit the global vector calibrator and the over-0 vector on the same
val ball construction as E5/A15 but with the venue encoder ACTIVE (loaders in
`scripts/auto/a14_fit_over_calibrator.py` / `a15_fit_over0_calibrator.py`; save
to `models/auto/b7/`, do NOT overwrite `models/xgb_v3/vector_scaling_calibrator_v1.pkl`
until LANDED). Recipe B n=261 seed 43 with the refit calibrators, paired vs the
new canonical venue-ON baseline `models/auto/b6/detail_venue_s43_n261.json`
(same seed → clean pairing) via the a8/b6 gate tooling. Pre-commit the gate
script before the run.
**Gate (sim pair):** PRIMARY = `batter_runs_mae` AND `team_first_over_mae` do
not regress vs the venue-ON baseline while ≥1 of {pooled tail Brier
(pp_total/first_wicket/highest_over), bowler_wkts_1plus} improves CI-clean;
guards = top_bowler + team_total_{fours,sixes}_mae no CI-clean regression. If
the refit calibrators are byte-close to the stale ones (max ratio diff < the
~0.05 washout threshold), expect null and say so pre-run. **Budget:** ~1.5 h
(one n=261 run; baseline reused).
**Result:** —

---

## Combination ideas (C-series)
Created only when no PENDING ideas remain, from TABLED entries. Follow
PROTOCOL step 1.
