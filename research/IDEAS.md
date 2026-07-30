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
(+26.39→+35.86%, CI lo −0.99 at n=72). The slice-conditional rule is now wired
into `predict_fixture.py` by I10 as a ≥$50k shadow-only policy; it does not
authorize execution. Kept commit. C-series candidate: stack on A4's logit-avg
(orthogonal — probability ensemble × bet filter). See
`research/reports/auto/A7.md`.

**I3 revision (2026-07-23):** A7 remains the fixed forward betting policy, but
its historical economic claim is no longer confirmatory. Tournament-block
resampling gives ≥$50k ROI +36.93% **[-1.52%, +59.81%]** across 17 blocks and
≥$100k +35.86% **[-36.70%, +58.16%]** across 10 blocks. The point improvement
and drawdown reduction remain; the earlier positive i.i.d. lower bound is
superseded. See `reports/i3_eval_statistics_hardening.md`.

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

## B2 [P1] [LANDED] Fix `innings_id` unstable hash (blocks parquet↔cricsheet joins)
*(done INTERACTIVELY by supervisor 2026-07-16 — `parsing_v2.py` is
loop-forbidden (program.md rule 1), so this could never have been claimed by
the loop; do not re-claim)*
**Hypothesis:** `parsing_v2.py:1255` sets `innings_id = hash(json) % 100000` —
salted per process (irreproducible across runs), ~450 expected collisions at
corpus size (TODO.md:331, filed at E5/E6). E6 lost ~5% of matches to a
workaround join. Correctness fix that unblocks every future ball↔match join.
**Method:** emit the cricsheet filename stem instead; rebuild feature parquet
(cheap, no schema bump per TODO). Verify: E6's join recovers the ~5% drop;
row counts and a sample of per-ball features byte-match otherwise.
**Gate:** correctness idea — LANDED if joins are lossless and no eval metric
moves beyond eval-only noise (it should be a no-op on metrics). **Budget:** ~1.5 h.
**Result:** LANDED 2026-07-16 (interactive). `parse_match_data_v2` gains
`match_ref=` (cricsheet filename stem; legacy hash fallback for callers that
don't emit joinable rows); `materialize_features.py` threads the loader's
match_id. Re-materialized `data/xgb_data_v3/` via the v6 config (feature hash
c520a3ba08ae, 114 features, unchanged). Parity: all 140 non-id columns
byte-equal on train/validation/test → provably a no-op on metrics (no eval
needed). Collisions removed: 360/1/2 (train/val/test) — old train had 7,792
distinct suffixes for 8,152 matches, and 716 innings groups were silently
merged for the sequence models. Join losslessness: 100% of new suffixes
resolve to `data/t20s_json/<stem>.json`; 9,519 distinct matches, zero
cross-split overlap. Old parquets at `archive/xgb_data_v3_pre_b2/`. E6's
(date,venue) workaround kept for pre-B2 parquets; comment updated.

## B3 [P1] [LANDED] Continuous-forecast shrinkage blend (productize E2's finding)
**Hypothesis:** E2: the sim's only validated skill is continuous score
forecasts (batter-runs MAE −0.71 vs career baseline). A val-fit shrinkage
blend `α·sim + (1−α)·fair_baseline` per family should beat both parents —
the best score forecast the system can produce.
**Method:** fit α per family (batter_runs, highest_individual, team totals) on
val matches; evaluate on recipe B test with `--ball-calibrator vector`;
compare MAE vs sim-alone and baseline-alone (`prop_fair_baselines.py`).
**Gate (sim pair):** blend MAE beats BOTH parents (paired, CI excludes 0) on
≥1 family AND no family regresses vs its best parent. **Budget:** ~2 h.
**Result:** LANDED 2026-07-23. No engine change, no re-baseline (post-hoc
blend; `prop_fair_baselines.py` read-only; non-engine pick — D3 consumed
the night's sim-engine slot). Alphas fit VAL-ONLY on the exact ball-model
val split (545 matches 2024-12-31..2025-06-29 from the post-B2 parquet
stems; ONE prop_backtest run at the canonical D15 engine config, s42,
100 sims, 4642 s, 0 skips; grid 0..1 step .01 min row-MAE, pre-committed
`ce6be5d` before any result): batter_runs **0.93**, highest_indiv
**0.83**, team_fours **0.64**, team_sixes **0.79**, first_over **0.15**
(sim-heavy exactly where E2 said the skill lives; baseline-heavy on
first-over). Test = EXISTING canonical D15 s43 detail, paired |err|
deltas cluster-boot 2000/s29. **GATE 1 MET** on TWO families CI-clean vs
BOTH parents: `batter_runs_mae` (vs sim **−0.0155 [−0.0279,−0.0034]**,
vs base −0.3443 [−0.4952,−0.1845]) and `team_total_sixes_mae` (vs sim
**−0.1277 [−0.1797,−0.0781]**, vs base −0.2337 [−0.4017,−0.0693]).
**GATE 2 MET**: zero CI-clean regressions vs best parent — blend
point-beats the best parent on ALL FIVE families (highest_indiv 16.312
vs sim 16.492; fours 3.637 vs base 3.654; first_over 3.374 vs base
3.383). Both → LANDED; commits kept, nothing reverted. The blend is the
system's best continuous score forecast: never worse than either parent,
CI-clean better where the sim has skill. Context (post-verdict): val
alphas lean sim vs test-optimal (.93→.78, .83→.68, .64→.35, .79→.59;
first_over .15→.28) — the pre-stated E5-calibrator-in-sample caveat
confirmed in direction; the land survives it. Production adoption (live
prop quoting reads alpha.json) = human follow-up. Artifacts
`models/auto/b3/` (alpha.json, detail_val_s42_n545.json,
gate_numbers.json; gitignored), harness
`scripts/auto/b3_shrinkage_blend.py`. B11 appended (bias-free alpha
refit). See `research/reports/auto/B3.md`.

## B4 [P2] [LANDED] top_bowler pricing margin (post-calibration edge quantification)
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
**Result:** LANDED 2026-07-21. Pure analysis on the canonical D15 detail
(`models/auto/d15/detail_d15_s43_n261.json`, calibrated sim, 522
team-markets / 5,835 rows) vs the exact E2 as-of fair baseline
(`prop_fair_baselines.py` imported read-only); synthetic market
q=p_base×(1+vig), YES-only flat 1u, cluster-boot CIs by match. **Margin
re-verified on the current engine**: ΔBrier **−0.0017 [−0.0030, −0.0003]**
CI-clean (D1 same-seed context −0.0023 [−0.0035,−0.0010]; E5's standing
claim survives the B6/D1/D15 engine chain). **Pricing grid**: +EV at EVERY
vig 0–10% × thr 0–20% (flat ROI +138…+166%, all 16 cells CI-clean;
in-sample break-even vig >50%; Kelly mean 0.047/median 0.031). **Band
decomposition flips the reading**: 83% of PnL from p_base<2% longshots;
the head is flat/negative (10–20% band +0.8 [−18.0,+19.5]; ≥20% band
−20.8 [−47.3,+9.0]) — the honest bettable slice is the **5–10% band
(odds ~10–21): ROI +54.1 [+23.4, +84.6]**. Tail profit = two entangled
distortions, audited row-by-row: (1) the career-wickets-share baseline
catastrophically underprices debutants (zero-career players top wickets at
0.90% vs 0.26% priced; all 15 winners genuine debutants, no name-mismatch
artifact); (2) the sim overprices non-bowlers — **D5 quantified in betting
units**: 477 rows give zero-career-wicket players p_sim≥2% (mean 3.84%),
topped by keepers/batters (PD Salt 6%, S Konstas 9%, Sahibzada Farhan 10%,
DP Conway 5–6%, SD Hope 8%). Harness `scripts/auto/b4_pricing_margin.py`;
table `research/reports/auto/B4_pricing.md`; numbers
`models/auto/b4/pricing_numbers.json` (gitignored). E2's fair-baseline bar
shown weak in the tail → B9 appended (usage-share baseline re-test). Kept
`74856a6`+`b387cbb`. See `research/reports/auto/B4.md`.

## B5 [P2] [PENDING] In-play over/under quote prototype (analytics-engine seed)
**Resolution (2026-07-30):** the stale 2026-07-23 RUNNING claim was closed
with a supervised CRASH row in `results.tsv` (eval died at session limits;
no result ever produced). The additive harness
(`scripts/auto/b5_{inplay_quotes,gate_analysis,unit_check}.py`) is kept,
but it was I15-compat-patched after implementation, so the old claim is
void: a fresh claim must re-run `b5_unit_check` and re-verify the
pre-committed gate against current code before any eval.
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

## B7 [P1] [TABLED] Refit ball calibrators on the venue-ON sim (B6 re-baseline follow-up)
*(claim reset by supervisor 2026-07-12: iteration was cut by end-of-night quota
exhaustion after implement + pre-commit gate; calibrator artifacts already built
at `models/auto/b7/` — the re-run only needs the paired recipe-B eval + verdict)*
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
**Result:** TABLED 2026-07-17 (three-iteration relay: 2026-07-12 refit +
pre-commit gate `1a43a34`, quota-cut; 2026-07-17 morning re-claim `05d917c` +
eval launch, cut after the 2231.9 s run finished; this iteration = gate +
verdict). Pre-run staleness was REAL, not a washout-null: global
max|refit/v1−1| = **0.1712**, over-0 max|refit/A15−1| = **0.2155**, both ≫
0.05. Paired (refit − stale) cluster-boot by match, n=261×100 **seed 43**
(verified from launch cmd), vs `models/auto/b6/detail_venue_s43_n261.json`,
only delta = calibrator. **GATE 1a** no-regress MET — batter_runs_mae +0.017
[−0.068,+0.100] noise; `team_first_over_mae` **−0.024 [−0.040,−0.007]**
CI-clean BETTER. **GATE 1b** NOT MET — both improvement arms CI-clean WORSE:
pooled 6-line tail dBrier **+0.0079 [+0.0048,+0.0109]** (pp_total 45/50/55
+0.0066/+0.0104/+0.0066 all CI-clean worse; highest_over_18_5 +0.0302),
bowler_wkts_1plus **+0.0024 [+0.0004,+0.0043]**. **GATE 2** guards HELD
(top_bowler −0.0002, fours/sixes MAE noise). Exactly one gate → TABLED per
the pre-committed mapping. Context scan one-sided NEGATIVE (mirror of B1/B6's
positive scans): CI-clean worse batter_fours_1plus/2plus, team_highest_indiv
34_5/39_5, bowler_economy_10_5, batter_fours_mae; better only
match_total_sixes_20_5 −0.0049. **Reading:** the E5 v1 global correction is
NOT stale under venue-ON — the 17% refit divergence is val-composition noise
that doesn't transfer (refitting actively degrades tails); keep stale v1 as
the default global. But **A15's first-over gain survives the re-baseline**:
the refit over-0 vector delivers −0.024 under venue-ON (A15 venue-blind
−0.018 / A14 −0.022), and the components are separable (over-0 balls never
see the global vector). Nothing to revert (`1a43a34` = fit+gate harness only;
default sim path byte-unchanged; artifacts gitignored `models/auto/b7/`).
Follow-up B8 appended (hybrid v1-global + over-0 default). See
`research/reports/auto/B7.md`.

## B8 [P2] [PENDING] Hybrid calibrator: stale v1 global + over-0 vector as the venue-ON default (B7 decomposition)
**Hypothesis:** B7 decomposed cleanly: the refit *global* vector actively hurts
(pooled tail +0.0079 CI-clean worse — v1 is NOT stale under venue-ON) while the
*over-0* vector delivers the A14/A15 first-over gain under venue-ON
(`team_first_over_mae` −0.024 [−0.040,−0.007], separable because over-0 balls
never see the global vector). The current venue-ON default runs bare v1 with NO
first-over correction, so A15's validated gain is not collected in production.
A hybrid `OverVectorScalingCalibrator{_global = stale v1, _v[0] = over-0}`
should retain the first-over win with none of B7's regressions.
**Method:** compose the hybrid from existing artifacts (v1 global + the over-0
vector; both the venue-ON refit over-0 from `models/auto/b7/` and A15's
venue-blind over-0 are defensible — pick ONE pre-run, default = the venue-ON
refit, and say so before evaluating; save to `models/auto/b8/`). One recipe-B
run n=261 **seed 43** with `--ball-calibrator-path`; pair vs the SAME
`models/auto/b6/detail_venue_s43_n261.json` via the a8/b7 gate tooling.
Pre-commit the gate script. NOTE: baseline valid only while no sim-engine idea
has landed since B6 (D-series rule) — else re-run the baseline first.
**Gate (sim pair):** PRIMARY = `team_first_over_mae` improves CI-clean
(retaining ~−0.02) AND no CI-clean regression on ANY of {pooled 6-line tail
Brier, bowler_wkts_1plus, batter_runs_mae}. Guards = top_bowler,
team_total_{fours,sixes}_mae no CI-clean regression. Both → LANDED (ship the
hybrid as the default venue-ON calibrator, superseding bare v1; re-baseline
warning applies); exactly one → TABLED; none → FAILED. **Budget:** ~1 h (one
n=261 run; baseline + tooling reused).
**Result:** —

## B9 [P2] [LANDED] top_bowler margin vs a usage-share fair baseline (B4 follow-up)
**Hypothesis:** B4 showed the E2 career-wickets-share baseline is weak
exactly where the sim's top_bowler profit concentrates (p_base<2%
longshots): it prices debutants at 0.26% when they actually top the
wickets 0.90% of the time, because career-wickets share cannot see *who
actually bowls*. A stronger fair baseline — as-of expected-overs share
(bowler_phase_usage-style history, EB-shrunk, uniform-ish prior for
debutant XI members) × career wicket rate, normalized within XI — is the
"competent bettor" bar. If the sim's CI-clean ΔBrier margin (B4: −0.0017
on the D15 engine) survives THAT baseline, the E2/E5 skill claim is
robust; if it flips, "first binary family with real skill" needs a
caveat and the E2 bar should be flagged for interactive revision (the
eval framework itself is loop-forbidden — build the stronger baseline in
`scripts/auto/`, do NOT edit `prop_fair_baselines.py`).
**Method:** build the usage baseline as-of from cricsheet (per-player
balls-bowled share within team-match, EB-shrunk toward a lineup-uniform
prior; reuse the E2 corpus-log pattern in a NEW script under
`scripts/auto/`); verify it beats the career-share baseline on Brier
(it must, else it's not a stronger bar); rerun the B4 margin + band
analysis with it on the SAME canonical D15 detail JSON. Pure analysis,
no sim run.
**Gate:** instrumentation+verdict hybrid: report the paired ΔBrier
(sim − usage-baseline) with CI verbatim either way; LANDED if the
analysis completes with real numbers AND the conclusion (margin survives
or flips) is stated with its CI. **Budget:** ~1.5 h.
**Result:** LANDED 2026-07-21. **The margin FLIPS.** Usage baseline
(as-of EB-shrunk expected-deliveries × per-ball wicket rate,
lineup-uniform debutant prior, K=5 appearances / 120 balls, pre-committed
`4ebad9b`) on the canonical D15 detail; B4's career pairing reproduced
exact (|Δ|=0.00e+00). Stronger-bar check CI-clean: usage − career ΔBrier
**−0.0055 [−0.0070, −0.0040]** (standalone Brier 0.0747 vs career 0.0802
vs sim 0.0785). The margin: sim − usage **+0.0038 [+0.0026, +0.0051]
CI-clean WORSE** — identical at grid (2,60)/(10,240); head-only (both
p≥2%) +0.0049 [+0.0032,+0.0067] and 4/5 p_usage bands CI-clean worse →
NOT a tail artifact. **E5's "first binary family to beat a fair
baseline" is falsified at the competent-bettor bar**; B4's pricing grid
inverts (flat ROI −29…−45% at every vig×thr vs usage prices; <2% band
779 bets, 0 wins). Mechanism split — both are USAGE errors: true
debutants (0 appearances) top the wickets 8.47% actual; the
lineup-uniform prior prices them 9.06%, sim 1.63% (sim under-bowls
debutant bowlers); seen-never-took-wicket players 0.62% actual vs sim
1.29% (D5's over-bowled non-bowlers). E2 bar flagged for interactive
revision (I13); sim who-bowls alignment appended as B10. Harness
`scripts/auto/b9_usage_baseline.py`; artifacts `models/auto/b9/`
(gitignored). See `research/reports/auto/B9.md`.

## B10 [P2] [PENDING] Sim who-bowls usage alignment: debutants + non-bowlers (B9 follow-up, D5 superset)
**Hypothesis:** B9's usage baseline beats the calibrated sim on top_bowler
CI-clean in every band that matters (pooled +0.0038, head-only +0.0049),
and both exposed distortions are WHO-BOWLS errors, not per-ball-rate
errors: (a) true debutants (0 prior XI appearances) actually top the
team's wickets 8.47% of the time — the lineup-uniform expected-balls
prior alone prices them at 9.06% while the sim gives 1.63%, i.e. the
`EmpiricalBowlerSelector`'s unknown-bowler league-share floor α
under-bowls genuine debutant bowlers ~5×; (b) experienced
never-took-a-wicket players get p_sim 1.29% vs 0.62% actual (D5's
over-bowled keepers/batters, 477 rows ≥2% per B4). Aligning the
selector's shares to B9's as-of usage expectations — unknown/debutant
players toward the lineup-uniform prior, long-history zero-usage players
toward their ~0 historical share — attacks both tails of the same defect
mechanistically, where D5's eligibility threshold only cuts tail (b).
**Method:** sim-engine change in `sim_v1_2.py`'s bowler-selection layer
only (ball model untouched): blend `EmpiricalBowlerSelector`'s EB-shrunk
phase-usage share with the B9 as-of expected-balls share
(`models/auto/b9/usage_corpus.pkl` / rebuildable via the B9 harness) for
players absent from or thin in `models/bowler_phase_usage.json`; keep
the ≥5-eligible-bowlers relaxation. Report before/after simulated
balls-bowled share for true debutants and zero-usage veterans vs actual
(unit-level check, pre-run). Recipe B n=261×100 seed 43, venue-ON
default path, stale v1 vector calibrator both sides, paired vs the
canonical D15 baseline `models/auto/d15/detail_d15_s43_n261.json`
(re-verify no sim-engine idea landed since D15 first). Pre-commit the
gate script; wait on the eval synchronously in-session (D2/D14 lesson).
**Gate (sim pair):** PRIMARY = `top_bowler` Brier improves CI-clean
paired vs the D15 run (target: close part of the +0.0038 gap to the B9
usage baseline — recompute sim−usage on the new detail as context) AND
G5 bowler coverage stays ≥90%. Guards = `bowler_wkts_{1,2}plus`,
`batter_runs_mae`, `team_first_over_mae`: no CI-clean regression. Both →
LANDED (re-baseline warning applies; subsumes D5's mechanism — its
queue status becomes a supervisor call); exactly one → TABLED; none →
FAILED. **Budget:** ~2.5 h. One sim idea per night.
**Result:** —

## B11 [P3] [PENDING] Bias-free alpha refit for the B3 blend (B3 follow-up)
**Hypothesis:** B3's val-fit alphas systematically lean toward the sim
relative to the (diagnostic-only) test optimum on every sim-heavy family
(.93→.78, .83→.68, .64→.35, .79→.59) — the direction B3's pre-run caveat
predicted: the v1 vector calibrator was E5-fit on those same val balls, so
val sim forecasts are in-sample-flattered. A val construction free of that
bias (e.g. split val in half: refit a vector calibrator on half A only,
run the val sim + alpha fit on half B with the half-A calibrator; or
K-fold the same) should yield smaller, better-transferring alphas and a
CI-clean improvement of the blend over B3's blend on ≥1 family. Cheap
guard: B3's landed alphas stay in force unless the refit blend beats
B3's blend paired CI-clean (not just the parents).
**Method:** reuse `scripts/auto/b3_shrinkage_blend.py` + the existing val
detail where possible (the half-B sim run needs a NEW calibrator, so one
~40 min half-val sim run); gate on the canonical D15 s43 detail with the
B3 gate tooling, adding a paired blend-vs-B3-blend delta. Pre-commit the
gate before any run. NOTE: fitting a new vector calibrator must NOT touch
`models/xgb_v3/vector_scaling_calibrator_v1.pkl` (write to
`models/auto/b11/`); the TEST-side sim stays the canonical D15 detail
(stale v1 calibrator) — only the VAL fitting construction changes.
**Gate (sim pair):** PRIMARY = refit blend beats B3's blend paired
CI-clean on ≥1 family AND no family regresses CI-clean vs B3's blend.
Guards = B3's GATE 2 (no family CI-clean worse than its best parent).
Both → LANDED (alpha.json superseded); exactly one → TABLED; none →
FAILED (B3 alphas stand). **Budget:** ~1.5 h.
**Result:** —

---

# D-series: 2026-07-16 full-review seeds (supervisor)

Source: 2026-07-16 four-agent project review (match-level code, sim
internals, research history, data/eval integrity), user-approved. D1–D3 are
mandated sim correctness fixes; D4–D10 are approved improvements. Ordering:
- Every sim-engine change re-baselines. Pair each verdict against a fresh
  same-seed baseline run under the CURRENT default sim path, identical
  calibrator settings on both sides. `models/auto/b6/detail_venue_s43_n261.json`
  (seed 43) is a valid baseline ONLY while no sim-engine idea has landed
  after B6 — otherwise re-run the baseline first.
- Preferred sim order: B7 (already staged) → D1 → D2 → D3 → D4/D5 → D6 last
  (D6 retrains the ball model and, if it lands, obsoletes the calibrator
  chain). One sim idea per night.
- D7/D8 (match-level, cheap) can interleave on any night; D9/D10 after.
- Review items touching loop-forbidden surfaces (`parsing_v2.py`, stats
  backends, `scripts/sim_eval/`, the odds builder, production caches) are
  NOT queued as PENDING — they live in the Interactive backlog at the
  bottom of this file. Never claim those.

## D1 [P1] [LANDED] Fix sim-side `run_rate` scale skew (runs-per-over vs runs-per-ball)
**Hypothesis:** training computes `run_rate` as runs-per-OVER
(`parsing_v2.py:938`: `score / max(overs, 0.1)`) but the sim wrappers compute
runs-per-BALL (`sim_v1_2.py:761`: `runs/(balls+1)`; second site at `:1190`) —
a ~6× scale skew on every simulated ball, the same OOD-input bug class as
B1/B6 venue blindness. `run_rate` ranks ~104/114 by gain so expect a small
effect — a correctness fix; B6 showed this bug class can still carry a
CI-clean batter-level win.
**Method:** align ALL sim-side feature-assembly sites to the training formula
(runs per over, same 0.1 zero-guard) — sim_v1_2.py has duplicated
feature-build blocks (~601/1153/1596/2017 per TODO); grep every `run_rate`
assignment. Do NOT touch the `run_rate_required` chase features (:805 —
already correct). Teacher-forced spot check: rebuild features for a few
recorded states and match `run_rate` against the training parquet. Recipe B
paired vs fresh same-seed venue-ON baseline.
**Gate (sim pair):** correctness — LANDED if the teacher-forced check matches
training AND no guard regresses CI-clean paired (batter_runs_mae,
team_first_over_mae, top_bowler, bowler_wkts_1plus,
team_total_{fours,sixes}_mae); CI-clean improvements are a bonus, not
required. **Budget:** ~2 h.
**Result:** LANDED 2026-07-17 (relay: prior iteration claimed `b5173e9`,
implemented + pre-committed gate `8fd5a8a`, launched the n=261 eval
(2213.4 s) and was cut; this iteration verified config + ran gate +
verdict). All 6 sim-side `run_rate` sites aligned to the training formula
`score/max(balls/6, 0.1)` — live v7 path :761 and legacy :1190 were
runs-per-ball (~6.24× OOD skew); the 4 LSTM/MLP/MLPv2/Transformer blocks
only needed zero-guard alignment; `run_rate_required` untouched. **GATE 1**
teacher-forced parity MET pre-run: max|parquet−formula| = 0.0 on 310,959
test+val rows; live `extract_features` spot check exact incl. the
balls=0/score>0 guard case. Recipe B n=261×100 **seed 43**, venue-ON
default path + stale v1 vector calibrator (identical to the B6 baseline
run; ONLY delta = run_rate formula), paired cluster-boot by match vs
`models/auto/b6/detail_venue_s43_n261.json` via pre-committed
`d1_gate_analysis.py`. **GATE 2** guards ALL HELD (no CI-excludes-0
increase): batter_runs_mae −0.044 [−0.171,+0.088], team_first_over_mae
−0.019 [−0.038,+0.002], fours/sixes MAE +0.061/−0.020 noise, top_bowler
−0.0005, bowler_wkts_1plus −0.0005. Both → **LANDED**. Context scan
ONE-SIDED positive (0 CI-clean worse anywhere, 6 better):
highest_over_runs_18_5 **−0.0272 [−0.0426,−0.0120]**, bowler_economy_10_5
−0.0035, highest_over_24_5 −0.0021, batter_50plus −0.0020, top_batter
−0.0012, p_tie −0.0001; all pp_total lines favorable in point estimate but
straddle 0. Gains sit exactly where a correctly-scaled pace signal should:
which-over-explodes, economy, batter-level (B1/B6 pattern; team aggregates
wash out per A8/A12/A16). **RE-BASELINE IN FORCE**: b6 detail JSON is now
stale (pre-D1 run_rate); canonical venue-ON baseline =
`models/auto/d1/detail_d1_s43_n261.json` (seed 43, run_rate-aligned,
current default path). Kept `b5173e9`+`8fd5a8a`; nothing reverted. See
`research/reports/auto/D1.md`.

## D2 [P1] [TABLED] Fix strike rotation + balls-faced on extras in the sim
**Hypothesis:** `update()` rotates strike on any odd `runs`
(`sim_v1_2.py:331-332`) and WIDE/NO_BALL both carry runs=1 (`process_ball`
:560-563), so ~2% of deliveries wrongly swap the striker; and balls-faced
excludes only WIDE (:315-318), so NO_BALL increments the striker's ball
count. Misattributes batter exposure; biases batter-level props at the
margin.
**Method:** in `sim_v1_2.py` only: rotate strike on odd off-the-bat runs
(never on wide/no-ball), and exclude NO_BALL from balls-faced alongside
WIDE. Unit-check on a scripted over. Recipe B paired vs fresh same-seed
baseline.
**Gate (sim pair):** correctness — LANDED if guards hold (no CI-clean
regression on batter_runs_mae, top_bowler, bowler_wkts_1plus,
team_first_over_mae); batter-family improvements a bonus. **Budget:** ~2 h.
**Result:** TABLED 2026-07-20 (relay: implemented + gate pre-committed
`46f26dc` 2026-07-18; seven eval launches died at startup because each
iteration ended its turn after launching — children reaped at session
close; the eighth launch, blocked-on synchronously, completed in 2230.4 s).
**GATE 1** correctness MET (unit check 26/26, re-verified at eval time).
Recipe B n=261×100 **seed 43**, venue-ON + run_rate-aligned default path,
stale v1 vector calibrator both sides, ONLY delta = `update()` semantics;
paired cluster-boot by match vs `models/auto/d1/detail_d1_s43_n261.json`
via pre-committed `d2_gate_analysis.py`. **GATE 2** guards: batter_runs_mae
−0.014 [−0.062,+0.036], top_bowler +0.0001, bowler_wkts_1plus +0.0001 all
noise — but `team_first_over_mae` **+0.013 [+0.004, +0.023] CI-excludes-0
WORSE** → REGRESSED. Exactly one gate → TABLED per the pre-committed
mapping; reverted `bc44473` (harness kept; **D1 baseline remains
canonical** — no re-baseline occurred). Context scan near-null (only
batter_6plus_six +0.0018 [+0.0000,+0.0036] at the boundary; pp_total /
first_wicket / economy favorable but straddling). READING: the unit check
surfaced a pre-existing, entangled bug — `_simulate_innings` cards each
ball on `striker_idx` AFTER `update()` has rotated/replaced
(`sim_v1_2.py:3558/:3567`), so batter-prop extraction reads a mis-keyed
card; fixing rotation alone re-scrambles rather than removes the
attribution error. Fix + re-apply as a unit → **D14**. See
`research/reports/auto/D2.md`.

## D3 [P1] [TABLED] Fix extras graft: empirical rates + pre-calibration ordering (sim-side half)
**Hypothesis:** `predict_next_ball` grafts a flat 1% wide + 1% no-ball AFTER
the calibrator runs (`sim_v1_2.py`: calibrator ~:1121-1126, graft
:1141-1142, then renormalize) — de-tuning the calibrated 6-class marginals —
and the rates are wrong (real T20 wides ≈3.5–4%, no-balls ≈0.5%: the flat 2%
under-produces wides and over-produces no-balls ~2×). Extras are also baked
into training labels (a wide-1 is labeled `one`), so the graft
double-counts; the label-side rework needs `parsing_v2.py` (loop-forbidden —
see Interactive backlog I5). This idea is the sim-side half only.
**Method:** compute empirical wide/no-ball per-delivery rates from val-split
cricsheet JSONs (as-of; small script, artifacts to `models/auto/d3/`);
replace the hardcoded 0.01/0.01; restructure so the calibrated 6-class
RELATIVE marginals are preserved when extras mass is added (scale the six
classes by (1 − p_extras) instead of renormalizing after the graft). Recipe
B paired.
**Gate (sim pair):** PRIMARY = simulated wide/no-ball rates match the
empirical val rates (report before/after) AND no CI-clean guard regression
(batter_runs_mae, pp_total/team-total families, top_bowler, bowler_wkts).
Both → LANDED; total-line improvements a bonus. **Budget:** ~2 h.
**Result:** TABLED 2026-07-22 (relay: same-night iteration claimed
`a64baf7` + implemented `8dfda3a` — shared `graft_extras()` at all six
wrapper sites, val-split rates 0.037702/0.004409 replacing flat
0.009804/0.009804, unit check 13/13 + gate pre-committed — and launched
the eval twice; both died at session close. This iteration re-verified
the unit check at eval time, blocked-on the eval synchronously (2284.0 s,
261/261), gate + verdict.) **GATE 1 MET**: live-path draws over ~300k
deliveries 0.037750/0.004350 vs val 0.037702/0.004409 (3σ); old graft
wides ~3.8× under, no-balls ~2.2× over — hypothesis confirmed at the
data level. **GATE 2 NOT MET**: paired vs the canonical D15 s43 baseline
(only delta = graft), all three `innings_runs` lines CI-clean WORSE —
160_5 **+0.0087 [+0.0010,+0.0166]**, 170_5 **+0.0142 [+0.0058,+0.0230]**,
180_5 **+0.0128 [+0.0049,+0.0204]**; pp_total lean worse straddling;
top_bowler/bowler_wkts/batter_runs_mae noise; context
`team_total_sixes_mae` +0.055 [+0.019,+0.088] the only other CI-clean
move. Exactly one gate → TABLED. MECHANISM (called in the pre-run note):
training labels fold wide runs into the 6 classes (wide-1 → `one`, I5),
so the per-legal-ball model output already carries extras runs — grafting
the TRUE empirical mass on top double-counts (+2.9 extras runs/innings,
2.4 → 5.3), inflating simulated totals; the old flat 1%+1% graft was an
accidental partial compensation, not a defensible rate. Rate realism
alone bought no observable prop win. Reverted `87f4fe6` (sim
byte-identical to pre-D3 head; **D15 baseline remains canonical**);
harness + `models/auto/d3/` kept. **Combine candidate:** re-apply this
exact graft as the sim-side half of I5's label rework (post-I5 the
6-class block carries zero extras mass, so the composition becomes
correct by construction; `d3_unit_check.py` + `d3_gate_analysis.py` are
the ready-made acceptance check). No new ideas appended. See
`research/reports/auto/D3.md`.

## D4 [P2] [PENDING] Wicket-type modeling: run-outs in the sim (attribution fix)
**Hypothesis:** the sim attributes 100% of dismissals to bowler-vs-striker
(`sim_v1_2.py:321-324`; bowling card ~:3606-3614). Real T20 has ~5–8%
run-outs (often the non-striker, never credited to the bowler). This is
exactly the residual `bowler_wkts_1plus` overshoot (+0.03 vs fair baseline
even after calibration, `reports/e5_class_weight_fix.md:87`) — attribution,
not rates.
**Method:** on a sampled WICKET, draw dismissal type from empirical rates
(run-out fraction + non-striker share, computed as-of from cricsheet
`wickets[].kind`; script + artifact under `models/auto/d4/`); run-outs
dismiss striker/non-striker per the empirical share with NO bowler credit.
Total wicket rate unchanged — only attribution moves. Recipe B paired.
**Gate (sim pair):** `bowler_wkts_{1,2}plus` Brier margin vs fair baseline
improves CI-clean AND guards hold: top_bowler margin vs fair baseline no
CI-clean regression (it redistributes credit — watch it), batter_runs_mae,
team_first_over_mae. **Budget:** ~2.5 h.
**Result:** —

## D5 [P2] [PENDING] Bowler eligibility filter (stop keepers/pure batters bowling)
**Hypothesis:** `get_available_bowlers` (`sim_v1_2.py:263-279`) iterates all
11 players; `EmpiricalBowlerSelector` gives unknown bowlers a league-share
floor α, so non-bowlers bowl at a small nonzero rate, diluting bowler-family
props.
**Method:** gate eligibility on as-of career balls-bowled (e.g. ≥60) or real
support in `models/bowler_phase_usage.json`, relaxing the threshold whenever
an XI would otherwise have <5 eligible bowlers. Report before/after share of
overs bowled by zero-history players. Recipe B paired.
**Gate (sim pair):** PRIMARY = overs bowled by zero-career-ball players drop
to ≈0 AND bowler coverage stays ≥90% (the G5 bar) AND no CI-clean guard
regression (top_bowler, bowler_wkts, batter_runs_mae). **Budget:** ~1.5 h.
**Result:** —

## D6 [P1] [PENDING] Retrain the ball model WITHOUT balanced class weights (structural E5 alternative)
**Hypothesis:** E5's root cause was `balanced` class weights sampled raw; the
fix so far is post-hoc marginal patching (E5 global vector → A8 → A14/A15 →
A16 → B7), which "can't be right everywhere" (the tilt is phase-dependent)
and is what costs batter_runs_mae under calibration (it breaks an accidental
boundary/wicket cancellation — e5 report). The never-tried structural
alternative: retrain with sample weights OFF so the model estimates
P(outcome|state) with correct conditional structure directly — balanced
weights only help ranking, which a generative sim doesn't need. E5 compared
only raw-balanced / theoretical prior-division / vector-scaling.
**Method:** retrain the v6-hierarchical config with weights off
(`scripts/xgboost_v2.py` — trainer is editable; weight sites :235, :273-278,
:343-347; add a config/CLI flag, don't delete the path) to `models/auto/d6/`
(NEVER overwrite `models/xgb_v3/`). Run the E5 marginal audit
(teacher-forced per-class marginals vs actual). Then recipe B paired vs the
canonical venue-ON baseline: d6-raw (no ball calibrator) and, if marginals
are still visibly off, d6 + fresh val-fit vector as a second arm. Heavy —
start only if >6 h of night remain.
**Gate (sim pair):** PRIMARY = teacher-forced marginals ≈ actual (E5-report
tolerance) AND, paired vs the CALIBRATED venue-ON baseline, tail families
(`pp_total_*`, `bowler_wkts_1plus`) improve CI-clean while `batter_runs_mae`
does NOT regress CI-clean (the exact trade the calibrator couldn't deliver).
Guards: top_bowler, team_first_over_mae. Both → LANDED (promotion to default
sim model is a human follow-up decision). **Budget:** ~5 h.
**Result:** —

## D7 [P1] [LANDED] Team-swap symmetry augmentation (match model)
**Hypothesis:** the match model consumes absolute team1_/team2_ features and
team assignment is arbitrary cricsheet order, so it is not antisymmetric —
and it shows: train/val/test base rate drifts 0.488/0.476/0.472 and
importances are asymmetric (team2_batting_sr #2 vs team1_batting_sr #3;
is_team2_home weight 15 vs is_team1_home 5). Augmenting each train row with
its swapped copy (exchange team columns, negate signed diffs, flip label)
doubles effective n to ~15.8k and enforces P(t1|A,B) = 1 − P(t1|B,A).
Training-procedure change, no new features — the correlation-check
discipline and the M3–M6 failure modes don't apply. Appears nowhere in the
M/E/A history.
**Method:** in `scripts/xgboost_match_v1.py` (trainer editable), build the
swapped train copy: exchange every team1_*/team2_* pair, swap
is_team1_home/is_team2_home, negate *_diff columns, map
h2h_team1_win_rate_shrunk → 1−x, flip toss_winner_is_team1 and
team1_batting_first, flip the label. Derive the mapping from
feature_columns.txt explicitly; unit-check swap(swap(row)) == row and that
all 48 columns are covered. Augment TRAIN only (val stays raw for early
stopping). Paired 5-seed (A1 seeds) vs fresh baseline; recipe A.
**Gate:** LL + ROI vs fresh baseline mean, retraining noise floors (>1
seed-std). If LANDED, flag inference-time symmetrization (average p(A,B) and
1 − p(B,A)) as the follow-up idea. **Budget:** ~1.5 h.
**Result:** LANDED 2026-07-17. `--swap-augment` (opt-in) in
`xgboost_match_v1.py`: TRAIN-only mirrored copies — 13 team1/team2 pairs
exchanged, 7 signed diffs negated, h2h/toss/bat-first/label 1−x, 12
invariants; coverage + swap(swap)==id hard-fail guards. Pre-run checks all
EXACT: 7/7 diff identities dev 0.0, h2h Beta(1,1) prior 0.5 at n=0 (1,728
rows), ties dropped at materialization → label flip valid; train 7,912 →
15,824 rows at base rate exactly 0.5. Paired 5-seed (A1 seeds, v2_clean,
trainer defaults); same-session base reproduces A1's logged numbers EXACTLY
on all 5 seeds. ≥$50k paired MEAN: **ΔLL −0.0121** (0.6318 → 0.6196, floor
0.007, better 5/5 seeds) / **ΔROI +3.01pp** (+20.56 → +23.57, floor 2.3, up
4/5). **Both → LANDED.** Swap mean LL **0.6196 beats market 0.6267** (every
seed does; max 0.6244) — first retrain idea in the loop below market LL.
Seed LL std collapses 0.0068 → 0.0027 (symmetry removes most seed luck —
re-measure the floor before gating on a swap-augmented baseline); ROI CI lo
> 0 on 5/5 swap seeds (base 2/5). ≥$100k consistent: ΔLL −0.0233, ROI
+26.14 → +30.02%, CI lo > 0 on 4/5. Kept `9a3eb17`; default trainer path +
recipe-A baseline semantics unchanged (flag is opt-in). Production adoption
deliberately not done (different parquet/config) → D12. Follow-ups appended:
D11 (inference-time symmetrization), D12 (production-config transfer). See
`research/reports/auto/D7.md`.

## D8 [P1] [TABLED] Recency-weighted training (match model)
**Hypothesis:** train spans 2005–2024 equally weighted (`model.fit` passes no
sample_weight — `xgboost_match_v1.py:197`); T20 is non-stationary (base-rate
drift, scoring-era shift). Exponential time-decay concentrates capacity on
the modern game. Orthogonal to DROPPED M4 (form *features*): this weights
the loss and adds no features, so the correlation discipline doesn't apply.
**Method:** add `--decay-half-life-years`; sample_weight =
0.5^(age_years_at_train_end / HL). Sweep HL ∈ {3, 6, 10, ∞} selecting on VAL
LL ONLY (pre-commit: no iteration-set shopping); paired 5-seed at the chosen
HL vs fresh baseline; recipe A.
**Gate:** LL + ROI vs fresh baseline mean, standard floors. **Budget:** ~2 h.
**Result:** TABLED 2026-07-17. Mean-1-normalized decay weights on the TRAIN
loss only (val raw for early stopping). Pre-committed val-LL-only sweep
(HL {3,6,10,∞} × A1 seeds on v2_clean; ∞ arm = D7 base models): clean
**interior optimum HL\*=6** (mean val LL 0.6433 vs ∞ 0.6466; every decayed
arm beats uniform). Paired 5-seed ≥$50k vs D7's logged base: **ΔLL −0.0093**
(0.6318 → 0.6225, floor 0.007, better 5/5 seeds; mean **beats market
0.6267**, 3/5 seeds individually — second retrain idea below market after
D7; LL seed-std halves 0.0061 → 0.0032) / **ΔROI −1.46pp** (+20.56 →
+19.11, down 3/5, win rate flat) → exactly one gate → TABLED. ≥$100k: ΔLL
−0.0167 (5/5 better), ROI dead flat. A4 pattern: sharper probabilities,
threshold-function ROI unmoved. Trainer flag reverted (`9702450`; re-revert
to reuse); harness `scripts/auto/d8_run.py` + `models/auto/d8/` scratch
kept. **Combine candidate**: with D7 swap-augment (supplies the ROI arm) —
appended as D13. See `research/reports/auto/D8.md`.

## D9 [P2] [FAILED] Decayed margin-aware team-results ELO (replacement test for win_rate features)
**Hypothesis:** `win_rate_diff` (crude last-10) is the single highest-gain
feature — the direct team-result family is the model's best signal but its
poorest-estimated one. A time-decayed, margin-aware team ELO updated on
match results is a strictly richer estimator of the same construct, and it
is match-level BY NATURE — immune to the lineup-aggregation collapse that
killed M3–M5. It will likely FAIL the |r|>0.5 redundancy check vs
win_rate_diff, so frame as REPLACEMENT, not addition (per discipline: must
show clearly higher target correlation).
**Method:** new TeamEloTracker in `materialize_match_features.py`
(chronological; K and margin-scaling chosen on val, small grid);
re-materialize to `data/auto/d9/` (~30–60 min). Pre-training dual
correlation check FIRST (cheap exit à la M5 if it fails both framings).
Then paired 5-seed, recipe A, two arms: (a) ADD team-ELO features,
(b) REPLACE win_rate_last_10/win_rate_diff with them.
**Gate:** corr-check pass (replacement framing allowed), then LL + ROI vs
fresh baseline mean on the better arm, standard floors. **Budget:** ~3 h.
**Result:** FAILED 2026-07-20 (relay: prior iteration claimed `3a80a41` +
implemented `793eeed` with the full decision pipeline pre-committed in
`d9_run.py`, cut mid-materialization; this iteration re-ran everything).
Stage 0 unit check ALL PASS. Stage 1 parity vs v2_clean **FAILED** (post-B2
match_id semantics + same-day row order shift whole rows) → pre-committed
fallback: fresh same-session 5-seed base control on the 51-col subset of the
d9 frame = **0.6336/+19.93** (≥$50k mean; A1's logged 0.6318/+20.56 is stale
for re-materialized frames — future paired ideas take note). Stage 2 dual
corr check **PASSES — first feature idea to clear the target-corr bar since
the discipline** (T=|corr(elo_diff,y)| 0.178–0.206 vs T_wr 0.1552, all 6
variants clear the REPL bar 0.1707, margin-aware strongest; R_max 0.64–0.80
vs win_rate_diff as expected). Val-LL-only selection (seed 29): add→k32,
repl→k16md365. Paired 5-seed ≥$50k: **add_k32 ΔLL −0.0019** (sub-floor
0.007; 2/5 paired better) / **ΔROI −0.42pp** (1/5) → FAILED; **repl_k16md365
ΔLL +0.0017 / ΔROI −1.41pp** → FAILED. ≥$100k consistent (both arms
flat-to-worse vs base 0.5926/+25.52). Neither arm clears either floor →
**FAILED**. Reading: the premise was CONFIRMED (results-ELO is the better
univariate estimator) yet buys nothing conditioned on the full 48-feature
set — player-ELO lineup aggregates already carry team strength; and REPLACE
is strictly worse ⇒ crude last-10 win_rate carries incremental
bounded-window recency signal ELO lacks. 6th match-level feature direction
dead, first to die AFTER passing the corr check: univariate target-corr
premium ≠ conditional value. Reverted materializer at `4625c0d`
(byte-identical to pre-D9); harness kept (re-revert to reuse); scratch
`models/auto/d9/` + `data/auto/d9/` kept (gitignored). No new ideas
appended. See `research/reports/auto/D9.md`.

## D10 [P2] [LANDED] Characterization tests for the eval math (instrumentation)
**Hypothesis:** every gate verdict rests on untested statistics code —
nothing tests `match_evaluator._bootstrap_ci`, `_calculate_kelly_fraction`,
`_calculate_realized_pnl`, the reslice/blend paths, or min-volume
filtering; `realized_pnl != 0` conflates a zero-return win with "no bet"
(`match_evaluator.py:933,977`; `reslice_eval_json.py:200-201`). Refactors
here are unguarded.
**Method:** NEW test files under `scripts/tests/` only (`test_eval_math.py`,
…) with synthetic fixtures — do NOT modify `scripts/sim_eval/` or the odds
builder (loop-forbidden). Characterize current behavior (bootstrap
reproducibility at seed 42, Kelly caps, PnL arithmetic, edge/min-volume
boundaries). If a test exposes a real bug: document it in the report and
mark the test xfail with a comment — do NOT fix (forbidden files); flag it
for the Interactive backlog.
**Gate:** instrumentation (A1-style) — LANDED if the suite passes (xfails
allowed and reported) and covers the named functions. **Budget:** ~2 h.
**Result:** LANDED 2026-07-20. `scripts/tests/test_eval_math.py` (new file
only; no `sim_eval/` edits): **29 tests = 27 PASS + 2 XFAIL, 0 fail**;
standalone runner and pytest agree (0.34 s); hermetic synthetic fixtures.
All named functions covered. Key pins: evaluator and reslice `_bootstrap_ci`
copies are **bit-identical** (unstratified + stratified; the loop reads CIs
from both interchangeably), seed-42 reproducibility exact, numeric pins on
venv numpy 1.24.3 detect environment drift; Kelly confirmed **uncapped by
design** (kelly(0.99,100)=98/99); realized-pnl threshold strictly `>0`,
max-edge team choice, win/loss arithmetic exact; flat-ROI CI == per-bet
PnL-mean CI ×100; reslice min-volume boundary **inclusive**, absent-from-odds
matches kept only at `min_volume=None`; blend `_recompute_realized_pnl` ==
evaluator on a 9-case branch grid + Kelly/Brier parity + threshold constants
in lockstep. The known `realized_pnl != 0` bet-placed sentinel bug
(`match_evaluator.py:933/977`, `reslice_eval_json.py:200-201` — I3 backlog)
is documented as **2 strict xfail tests** asserting the CORRECT behavior:
they flip to a loud XPASS error when I3 is fixed interactively. Minor
divergence noted in-comment: reslice `_bootstrap_ci` lacks the evaluator's
`n_resamples<=0` guard (unreachable at defaults). **No new defect found**
beyond tracked I3 — the eval math is internally consistent, retroactively
de-risking historical CI/ROI rows. Existing suites unaffected. No new queue
ideas appended (nothing untracked surfaced). Context: picked because D2
consumed the night's one-sim-idea slot and D9 (~3 h) didn't fit the
remaining ~2 h 45 m. Kept `8c3aad7`. See `research/reports/auto/D10.md`.

## D11 [P2] [FAILED] Inference-time symmetrization on the swap-augmented model (D7 follow-up)
**Hypothesis:** D7's augmentation makes the model *approximately*
antisymmetric but not exactly (trees on augmented data still fit residual
orientation noise). Averaging the two orientations at predict time —
p_sym = (p(A,B) + 1 − p(B,A)) / 2 via the D7 `_swap_frame` mapping applied
to the TEST parquet — enforces exact antisymmetry and is a free eval-only
variance cut on top of D7 (flagged by D7's own gate note). Cheap: reuses the
5 trained swap models at `models/auto/d7/swap_seed*`; no retraining.
**Method:** script under `scripts/auto/`: load each D7 swap model, score
test.parquet both raw and swapped (re-encode venue/tier for the swapped
frame with the saved encoders — team columns don't touch the encoded
features, so encoding is unchanged; verify), write symmetrized
test_predictions.json per seed → recipe A steps 2–3 per seed. Paired
symmetrized-vs-raw over the same 5 seeds. Also report the same transform on
the base (non-augmented) models `models/auto/d7/base_seed*` as a second arm
(does symmetrization alone recover part of D7's gain?).
**Gate:** eval-only floors (ΔLL 0.002 / ΔROI 2pp) vs the D7 swap per-seed
results (primary arm). Both → LANDED; one → TABLED; none → FAILED.
**Budget:** ~45 min (no training).
**Result:** FAILED 2026-07-20. Eval-only, 3 arms × 5 seeds paired sym-vs-raw
(`scripts/auto/d11_symmetrize.py`, gate pre-committed @ 3912938); raw
re-scoring byte-reproduces the saved D7/D12 predictions on all 15 models
(max|Δp| = 0.0 on the 782 unique-id rows; the 791-row test parquet has 9
duplicate match_ids and the prediction JSON is last-write-wins — control
compares last occurrences). **PRIMARY d7_swap ≥$50k mean: ΔLL +0.0013
(WORSE, 4/5 seeds; floor ≤ −0.002) / ΔROI −0.83pp (floor ≥ +2) → neither →
FAILED.** Context: **d7_base** ΔLL −0.0011 (sub-floor) / **ΔROI +3.03pp**
(clears floor) — symmetrization alone recovers essentially ALL of D7's ROI
gain (+3.03 vs +3.01pp) but only ~10–15% of its LL gain; base+sym (0.6307)
stays strictly dominated by swap-trained (0.6196) → mechanism knowledge
(D7's ROI arm = antisymmetry enforcement, LL arm = training-time doubling),
no production lever. **d12_swap** (production frame): −0.0004 / +0.62pp,
both sub-floor. mean|asym| 0.052 base → 0.023 swap → 0.018 production:
augmentation halves residual orientation noise; averaging away the rest
collects nothing — D7's gate-note hypothesis resolved negative. Nothing to
revert (no production path touched); harness kept (A9/A11 precedent). No
new queue ideas. See `research/reports/auto/D11.md`.

## D12 [P1] [LANDED] Swap augmentation on the production config (D7 transfer confirmation)
**Hypothesis:** D7 landed on the loop's recipe-A baseline (v2_clean frozen
parquet, 45 features), but production
(`models/xgb_match_v3_m7_production`) trains on an unfrozen parquet with
the 48-feature M2-venue set. If the augmentation gain transfers there, the
production-adoption case is complete (adoption itself stays a human
decision). The swap mapping extends cleanly: venue_p4/p6/pw are
swap-INVARIANT; the M2 expected/diff columns pair/negate exactly like the
M1 families (`team{1,2}_top6_p*_expected`, `team{1,2}_bowlers_p*_expected`
pairs; `p*_batting_diff`/`p*_bowling_diff` negations) — the hard-fail
coverage guard forces the extension to be explicit and total.
**Method:** extend the `_SWAP_*` mapping in `xgboost_match_v1.py` for the
M2 columns (guard makes omissions impossible); subset
`data/xgb_match_data_v3_m3_unfrozen` (90 cols, superset) to metadata + the
production 48-feature list from
`models/xgb_match_v3_m7_production/feature_columns.txt` in that exact
column order (A9 pattern — colsample makes order load-bearing) →
`data/auto/d12/`; re-run the D7 pre-run verification (diff identities on
the NEW parquet, involution, coverage); paired 5-seed base-vs-swap
(`scripts/auto/d7_run.py` generalized), recipe A, ≥$50k gate.
**Gate:** LL + ROI paired vs the same-parquet base mean, retraining floors
(>1 seed-std: 0.007 LL / 2.3pp ROI). Both → LANDED (production retrain
recommendation goes in the report; not executed by the loop); one → TABLED;
none → FAILED. **Budget:** ~1.5 h (no re-materialization — parquet subset
only).
**Result:** LANDED 2026-07-17. Transfer CONFIRMED on the exact production
frame: `data/auto/d12` = m3_unfrozen subset to the production 48-feature
set in `feature_columns.txt` order; both arms `--monotone` + trainer
defaults (verified against production `model.pkl`: 12 constraints, lr 0.05,
cs 0.9, seed 29). Control airtight — base seed29 **byte-reproduces**
production `test_predictions.json` (782/782, max|Δp|=0.000e+00; its ≥$50k
row == the M7 headline 0.6299/+21.90 exactly). Swap mapping extended: 6 M2
expected pairs + 6 M2 diffs (verified exact t1−t2 mirrors on m3_unfrozen;
no-ops on the venue-only production frame) + venue_p4/p6/pw invariant.
Paired 5-seed (A1 seeds) ≥$50k MEAN: **ΔLL −0.0092** (0.6378 → 0.6286,
floor 0.007, better 5/5 seeds) / **ΔROI +3.39pp** (+19.67 → +23.06, floor
2.3, up 5/5). **Both → LANDED.** ROI seed-std halves 3.04 → 1.73; swap ROI
CI lo > 0 on 4/5 seeds (base 2/5); win 50.4 → 52.5%. ≥$100k consistent:
ΔLL −0.0123, ROI +23.85 → +25.39. Secondary findings: (a) M7 headline
0.6299/+21.90 confirmed best-of-5 **seed-29 luck** — production-config base
mean is 0.6378/+19.67 (vindicates E3 tempering); (b) D7's LL-std collapse
does NOT transfer (swap std 0.0060 vs base 0.0048; ROI-std halving does) —
re-measure the seed floor per-config before gating on swap baselines;
(c) swap beats market LL 0.6267 on 2/5 seeds here (v2_clean D7: 5/5); the
loop frame + swap (0.6196) outscores the production frame + swap (0.6286)
on the iteration set — frozen-vs-unfrozen flag for interactive review, not
relitigated by the loop. Production retrain recommendation (human decision)
in the report. Kept `998cad7`+`8a03cd9`; `models/auto/d12/` +
`data/auto/d12/` scratch kept (gitignored). D11 gains a free second arm
(production-config swap models at `models/auto/d12/swap_seed*`). See
`research/reports/auto/D12.md`.

## D13 [P1] [FAILED] Swap augmentation + recency decay combined (D7 × D8)
**Hypothesis:** D7 (LANDED, ΔLL −0.0121 / ΔROI +3.01pp) and D8 (TABLED, ΔLL
−0.0093 / ΔROI −1.46pp) are training-procedure changes with disjoint
mechanisms — data augmentation (antisymmetry) vs loss weighting (recency) —
and both independently push LL below market on v2_clean (0.6196 / 0.6225).
If the LL gains even partially stack while swap's ROI arm survives, the
combination beats the best single component and becomes the strongest
training recipe in the loop. Decay weights extend to mirrored rows
trivially (same match_date → same weight; mean-1 normalization unchanged
since the mirror doubles every weight exactly once).
**Method:** re-revert D8's revert (`git revert 9702450`) to restore
`--decay-half-life-years`; run both flags together (`--swap-augment
--decay-half-life-years 6` — HL fixed at 6 from D8's val sweep, no new
sweep = no extra selection) at the A1 seeds on v2_clean, trainer defaults.
Weight computation must run on the AUGMENTED frame (order the code so
weights are computed after `_swap_augment_train`; verify mirrored rows get
identical weights). Paired 5-seed recipe A vs the **D7 swap arm** per-seed
results (`models/auto/d7/d7_results.json` "swap" — the stronger parent is
the control, not the plain base).
**Gate:** LL + ROI vs the D7 swap-arm mean (0.6196 / +23.57), retrain
floors (0.007 LL / 2.3pp ROI — conservative: D7 measured swap-arm LL
seed-std at 0.0027, so an LL gain may be real below the generic floor;
report per-seed direction counts either way). Both → LANDED; one → TABLED;
none → FAILED. **Budget:** ~45 min (10 trainings already exist for one arm;
5 new trainings + 5 evals).
**Result:** FAILED 2026-07-17. Combo run exactly as specced (weights
computed on the augmented frame; Stage-0 hard-fail verification: mirror
rows share weights exactly, mean-1 exact, `w[:n]` == base weights; harness
`d13_run.py` pre-committed at `9338989` before any training). Paired
5-seed ≥$50k vs the D7 SWAP arm: **ΔLL −0.0026** (0.6196 → 0.6171,
sub-floor 0.007; better 4/5 seeds) / **ΔROI −3.67pp** (+23.57 → +19.90,
down 4/5 seeds) → neither gate → FAILED. The combo IS the loop's sharpest
model (mean 0.6171; ALL 5 seeds beat market 0.6267, min 0.6154) but the
decay LL gain overlaps ~70% with swap's (−0.0093 alone → −0.0026
incremental) and decay's ROI drag amplifies on the swap arm (−1.46pp alone
→ −3.67pp). ≥$100k same shape (ΔLL −0.0063, ROI +30.02 → +27.70). Third
confirmation (A4, D8, D13) that LL sharpening decouples from
threshold-function ROI; **decay lever closed for winner-market ROI**.
Reverted `c821172` (trainer back to swap-only); harness + `models/auto/d13/`
scratch kept (gitignored). Log commit landed the following iteration (this
one was wall-clock-cut after the revert). See `research/reports/auto/D13.md`.

## D14 [P1] [TABLED] Batting-card attribution fix + D2 extras semantics, gated as a unit (D2 follow-up)
**Hypothesis:** D2 TABLED with a lone CI-clean guard regression
(`team_first_over_mae` +0.013 [+0.004,+0.023]) despite provably-correct
`update()` semantics (26/26 scripted-over assertions). The D2 unit check
surfaced the likely culprit, a pre-existing independent bug:
`_simulate_innings` builds its per-ball batting card keyed on
`state.striker_idx` AFTER `update()` has already rotated strike / replaced a
dismissed batter (`sim_v1_2.py:3558` BallResult, `:3567` card) — a ONE
scored by batter 0 is carded to batter 1, and a WICKET ball cards a
ball-faced to the INCOMING batter. Internal `batsman_stats` (crease
features) credit correctly; the **card** is what batter-level prop
extraction reads. Under the old rotation the two attribution errors
partially overlapped; D2 alone re-scrambled the card mismatch instead of
removing it. The bugs are entangled — fix the card and re-apply D2 together,
gated as one unit.
**Method:** `sim_v1_2.py` only: in `_simulate_innings`, capture the striker
(and team) index BEFORE `state.update()` and key BallResult + the batting
card on the captured index; re-apply D2 (`git revert bc44473`). Extend
`scripts/auto/d2_unit_check.py` with card-vs-stats equality assertions (per
batter on the scripted over, the card must match internal `batsman_stats`).
Recipe B n=261×100 **seed 43**, venue-ON default path, stale v1 vector
calibrator, paired vs `models/auto/d1/detail_d1_s43_n261.json` (still the
canonical venue-ON baseline — D2 was reverted, so no re-baseline is in
force; re-verify no sim-engine idea landed since D1 before trusting it).
Pre-commit the gate script before the run. Eval must be waited on
synchronously in-session (see D2 operational notes — background launches
die at turn end).
**Gate (sim pair):** correctness — extended unit check passes (incl.
card-vs-stats) AND guards hold (no CI-excludes-0 regression on
batter_runs_mae, top_bowler, bowler_wkts_1plus, team_first_over_mae — i.e.
D2's first-over regression must disappear); batter-family improvements a
bonus. Both → LANDED (both fixes ship as one unit; re-baseline warning
applies); exactly one → TABLED; none → FAILED. **Budget:** ~1.5 h (one
n=261 run; baseline + tooling reused).
**Result:** TABLED 2026-07-20. D2 re-applied (`7079c96`) + pre-ball
attribution snapshot in `_simulate_innings` (`3cb2f7e`); scope declared
pre-run in the gate script: batting card keyed on the FACING striker,
bowling card on the DELIVERING bowler (`simulate_ball` reassigns
`bowler_idx` to the next over's bowler before the card read — every
over-final ball was mis-credited, ~1/6 of deliveries), BallResult
over/ball labels pre-increment (old code rolled each over's 6th legal
delivery into the next over, so the `b.over==0` first-over extraction was
systematically missing its final delivery); team_runs/wickets stay
POST-ball. **GATE 1** extended unit check MET pre-run (d2 26/26 +
scripted deterministic innings through the real `_simulate_innings`:
card-vs-stats equality, over-final bowler attribution, over labels,
conservation). Recipe B n=261×100 **seed 43**, venue-ON +
run_rate-aligned default path, stale v1 vector calibrator both sides,
paired cluster-boot by match vs `models/auto/d1/detail_d1_s43_n261.json`
via pre-committed `d14_gate_analysis.py` (@3cb2f7e). **GATE 2**:
batter_runs_mae +0.000 / top_bowler +0.0001 noise;
`team_first_over_mae` **−0.132 [−0.222, −0.043] CI-clean BETTER** (D2's
+0.013 regression gone and inverted, 3.526 → 3.395 — ~7× A15's
calibrator gain; extraction-window fix, not calibration); but
`bowler_wkts_1plus` **+0.0027 [+0.0007, +0.0050] CI-excludes-0 WORSE**
(2plus +0.0022 echoes) → REGRESSED. Exactly one gate → TABLED. READING:
correct over-final bowler keying un-smears per-bowler wickets and
UNMASKS the E5-residual bowler-wicket overshoot whose root cause D4
names (100% bowler credit, no run-out channel) — the old mis-keying was
accidentally diluting a known bias; D2→D14→D4 entanglement one layer
down. Batter-card mis-keying itself nets ~0 on batter families (washes
out across sims). Reverted `87d9133` (sim byte-identical to pre-D14
head; **D1 baseline remains canonical**); harness + `models/auto/d14/`
scratch kept. Fix + gate as a unit with D4 → **D15**. See
`research/reports/auto/D14.md`.

## D15 [P1] [LANDED] Full attribution unit: D2 + D14 snapshot + D4 run-out dismissals (D14 follow-up)
**Hypothesis:** D14 TABLED with a lone CI-clean regression on
`bowler_wkts_1plus` (+0.0027 [+0.0007,+0.0050]) while `team_first_over_mae`
improved CI-clean (−0.132) and correctness was proven. Mechanism: correct
over-final-ball bowler keying un-smears wickets across bowlers, unmasking
the bowler-wicket overshoot whose root cause **D4 already names** — the sim
credits 100% of dismissals to the bowler while real T20 has ~5–8% run-outs
(often the non-striker, never bowler-credited). The old mis-keying and the
missing run-out channel partially cancelled; each fix alone re-exposes the
other's bias (the D2→D14 pattern one layer down). Applying D2 semantics +
D14 attribution snapshot + D4's run-out dismissal channel as ONE unit
should keep D14's first-over and card-correctness wins while the run-out
channel removes the unmasked wicket overshoot (right sign, right rough
magnitude: ~5–8% fewer bowler-credited wickets).
**Method:** re-apply the D14 unit (`git revert 87d9133`); implement D4 per
its entry — empirical run-out fraction + non-striker share computed as-of
from cricsheet `wickets[].kind` (script + artifact under `models/auto/d15/`);
on a sampled WICKET draw dismissal type, run-outs dismiss striker/non-striker
per the empirical share with NO bowler credit; total wicket rate unchanged.
Extend `d14_unit_check.py` with run-out attribution assertions. Recipe B
n=261×100 **seed 43**, venue-ON default path, stale v1 vector calibrator,
paired vs `models/auto/d1/detail_d1_s43_n261.json` (re-verify canonical
first). Pre-commit the gate script; wait on the eval synchronously
in-session (D2/D14 operational lesson). Subsumes D4's mechanism — if D15
lands, running D4 alone is moot (supervisor call on its queue status; do
not edit its entry from the loop).
**Gate (sim pair):** PRIMARY = `bowler_wkts_1plus` does NOT regress
CI-clean vs the D1 baseline AND `team_first_over_mae` retains a CI-clean
improvement. Guards = `top_bowler` (D4's warning: credit redistribution),
`batter_runs_mae`, `bowler_wkts_2plus`: no CI-excludes-0 regression. Both →
LANDED (D2+D14+D4 ship as the attribution unit; re-baseline warning
applies); exactly one → TABLED; none → FAILED. **Budget:** ~2.5 h (one
n=261 run + empirical-rates build; D14 tooling reused).
**Result:** LANDED 2026-07-21 (relay: 2026-07-21 06:08 UTC iteration
claimed `0503849`, re-applied D2+D14 via `4c9778a`, implemented the
run-out channel + unit check + pre-committed gate `17fd2d1`, launched the
eval and died at session close; this iteration re-verified the unit check
30/30 at eval time, re-ran the eval blocked-on synchronously — 2218.6 s,
261/261 — and ran gate + verdict). Empirical as-of rates (< 2025-07-01,
8,697 matches / 108,356 dismissals): p_runout **0.075077**,
nonstriker_share **0.468470**, baked as constants into `sim_v1_2.py`
(no runtime file dep); sampled WICKETs draw dismissal type in
`simulate_ball`, run-outs dismiss striker/non-striker with NO
bowling-card credit, total wicket rate untouched; legacy dismissal=None
path == pre-D15. Recipe B n=261×100 **seed 43**, venue-ON +
run_rate-aligned default path, stale v1 vector calibrator both sides,
ONLY delta = the D2+D14+D15 unit; paired cluster-boot by match vs
`models/auto/d1/detail_d1_s43_n261.json` via pre-committed
`d15_gate_analysis.py`. **P1** `bowler_wkts_1plus` +0.0047
[−0.0001,+0.0095] CI includes 0 → no CI-clean regression (D14's +0.0027
[+0.0007,+0.0050] gone) → MET. **P2** `team_first_over_mae` **−0.111
[−0.185,−0.038]** CI-clean better (3.526 → 3.415; D14's win retained) →
MET. **GUARDS HELD**: batter_runs_mae +0.014, top_bowler +0.0007 noise;
BONUS `bowler_wkts_2plus` **−0.0054 [−0.0085,−0.0024]** CI-clean BETTER
(D14's regression inverted), context `bowler_wkts_3plus` −0.0013
[−0.0021,−0.0004] better — the run-out channel bites hardest on the
multi-wicket tail, as D4 predicted. Both → **LANDED**; commits
`4c9778a`+`17fd2d1` kept, nothing reverted. Caveats: 1plus point
estimate (+0.0047) not better than D14's, CI at the boundary — residual
1plus overshoot remains (D5/D6 are the queued levers); context
`team_total_fours_mae` +0.043 [+0.009,+0.075] CI-clean worse (~1.2%
rel), the scan's only CI-clean negative. **RE-BASELINE IN FORCE**: all
pre-D15 sim detail JSONs (incl. D1's) are stale; canonical venue-ON
baseline = `models/auto/d15/detail_d15_s43_n261.json` (seed 43,
attribution unit). D4's mechanism is subsumed — its queue status is a
supervisor call (entry not edited by the loop). No new ideas appended
(D5/D6 already cover the residual). See `research/reports/auto/D15.md`.

---

## Combination ideas (C-series)
Created only when no PENDING ideas remain, from TABLED entries. Follow
PROTOCOL step 1.

---

# Interactive backlog (supervisor + user sessions only — NOT for the loop)

These touch loop-forbidden surfaces (`parsing_v2.py`, stats backends,
`scripts/sim_eval/`, the odds builder, production caches) or have no loop
gate (live-inference behavior). Loop: never claim these — statuses here are
intentionally not PENDING.

## I1 [DONE 2026-07-16] Toss both-branch averaging in predict_fixture
Materializer defaults unknown toss to team1-bats-first
(`materialize_match_features.py:770`); `predict_fixture.py:220` defaults to
False — systematic train/serve skew on every pre-toss live prediction.
Approved design: predict both bat-first branches and average.
**Done:** unknown-toss fixtures now predict both `team1_batting_first`
branches and average; branch probs surfaced in diagnostics
(`toss_known`/`toss_branch_probs`). Smoke-tested on 2026-05-10 CSK-LSG
(branches 56.6%/57.5% → 57.1%).

## I2 [DONE 2026-07-16] `_split_elo` bowling fallback
`materialize_match_features.py::_split_elo`: lineups ≤6 make
`bottom5_bowling_elo_avg` (top-4 importance) silently equal the top-6
BATTING elos (`bot_bow_elos = ... if bottom else top_bat_elos`). Replace
with a neutral/guarded path + a predict-time lineup-length assertion;
verify zero training rows change (full-XI lineups never hit the branch).
**Done:** short lineups now fall back to bowling elos of the full lineup;
predict-time guard raises <7 / warns <11. Corpus scan: 0 of 9,519 male
matches have a ≤6 lineup → zero training rows change, no re-materialization
needed.

## I3 [DONE 2026-07-23] Eval statistics hardening (`scripts/sim_eval/`)
Centralized the eval math under `eval_statistics.py`. Bet placement now uses
explicit `bet_placed`/`bet_team` fields (legacy rows reconstruct from
edge+odds), so zero-return wins remain in bet count/ROI denominator/win rate.
Headline LL/ROI CIs use 10,000 seed-42 whole Cricsheet-event time-block
resamples (`tournament_time_block_v1`), with team-pair/season fallback,
metadata coverage, effective block count, and a `<10 blocks = descriptive`
guard. Historical point ROIs are unchanged but prior CI-clean claims do not
survive: M7 ≥$50k +21.90% **[-10.79,+50.18]** (19 blocks); A7 +36.93%
**[-1.52,+59.81]** (17); ball-v7 +6.11% **[-7.99,+25.70]** (19).
The frozen forward evaluation completed on 2026-07-23. Its ≥$50k M7 A7 result
was +96.72% **[-3.29,+623.85]** across only five betting blocks, so economic
confirmation failed; M7 probability confirmation passed (LL 0.6823 vs market
0.7445 and ball-v7 0.7015). See
`reports/i3_eval_statistics_hardening.md` and
`reports/forward_evaluation_2026-06-01_2026-07-13.md`.

## I4 [DEFERRED — LEGACY REBASELINE ONLY] Odds-build integrity (`build_polymarket_odds.py` + eval odds)
True pre-match check (price_timestamp < scheduled start; 259/261 stamps are
same-day as the match); log residual top_p>0.92 entries (9 remain, max
0.9995); remove the outcome-conditioned dedup tiebreak (criterion #2 uses
the realized Cricsheet winner). Regenerating `betting_odds_polymarket.json`
re-baselines ALL historical LL/ROI numbers — do deliberately, once.
**Forward-set progress (2026-07-23):** per user decision, the legacy extractor
and the historical 261-match artifacts remain unchanged. A separate strict
extractor now enforces explicit scheduled start, last CLOB tick strictly
before start, exact-title H2H, male scope, resolved two-team outcomes, and
full provenance. The outcome-blind forward builder sealed 137 new matches
(61 ≥$50k / 30 ≥$100k) with zero timing/result/overlap violations; see
`docs/FORWARD_HOLDOUT.md`. The set was subsequently evaluated under its
frozen protocol; the locked report is
`reports/forward_evaluation_2026-06-01_2026-07-13.md`. This completes the
guardrails for future data but does **not** rehabilitate or re-baseline the
legacy 261-match odds. The remaining work is therefore not on the current
forward path: it is a deliberate legacy rebaseline only if historical
headline metrics are ever regenerated. Do not mix it into new model work.

## I5 [DONE 2026-07-24 — NOT PROMOTED] Extras/threes label rework (`parsing_v2.py`, full ball retrain)
Implemented as the isolated `legal_off_bat_v1` / `i5` stack without modifying
the legacy default or production v3/v7 artifacts. Only legal deliveries enter
the six-class off-bat target; wides/no-balls/byes/leg-byes are composed through
a validation-only empirical extras model; scorecard attribution now handles
no-ball balls faced, byes/leg-byes, and non-striker run-outs correctly. Raw
threes are preserved but remain in the combined 2/3 class: they are only
10,463 / 2,477,116 = 0.4224% of local legal balls and have strong venue
dependence (MCG 239/22,574 = 1.0587%; SCG 190/19,987 = 0.9506%; Mirpur
170/68,991 = 0.2464%). A standalone draw is deliberately **not prioritized**:
it adds a seventh sparse probability, calibration burden, odd-run strike
rotation, and another per-delivery branch. Profiling subsequently reduced a
fixed calibrated I5 prop benchmark from 31.1 to 9.2 seconds, but that is a
reason to protect the recovered throughput, not spend it on a 0.42% event.
Preserve the raw signal now; revisit the sub-draw only after I14 supplies
time-indexed ground dimensions and an ablation demonstrates value beyond the
combined 2/3 class.

The full cache/parquet/model build passed legal-ball, tracker-log, and outcome
conservation checks. Raw I5 ball test LL improves 1.629655 → 1.626158, but
each model's validation-only vector calibration reverses the comparison
(v3 1.508998 vs I5 1.514956). On the used 261-match iteration set, raw I5
improves LL overall (0.7158 → 0.7097) and at ≥$50k (0.7402 → 0.7075), but
flat ROI is +0.37% overall and becomes -7.11% after removing one 19-unit
long-shot win. The full paired n=261, seed-43, 100-sim prop gate is TABLED:
batter-runs MAE is noise (+0.136 [-0.125,+0.386]) and bowler 2+/3+ wicket
Brier improves, but PP >55.5 and all innings total lines (>160.5/>170.5/
>180.5) are CI-clean worse. Profiling reduced the full prop runtime to 56.2
minutes via a byte-identical date-normalization hot-path fix. Verdict:
implementation DONE; promotion TABLED pending component ablations and a new
untouched post-2026-07-13 forward window. See
`reports/i5_legal_off_bat_evaluation_20260724.md`.

## I6 [DONE 2026-07-23] Same-day match ordering determinism
All chronological loaders now use the versioned
`(match_date, Cricsheet match_id)` order
`date_then_match_id_lexicographic_v1`; multi-directory walks share that
implementation and reject duplicate IDs. Cache acceptance now requires exact
source membership/count plus the ordering version. The forward sidecar
combines 9,519 historical and 401 context matches without modifying production
artifacts, freezes global/phase priors from the pre-holdout cache, and verifies
feature rows for all 137 sealed fixtures. The refreshed audit found 1,616
historical and 67 context same-day groups whose old filesystem order differed.
On 791 already-consumed test rows, M7 LL changed 0.592629 → 0.592474 and mean
absolute prediction movement was 0.079 percentage points. No forward fixture
was scored. See `docs/I6_SAME_DAY_ORDERING_AUDIT.md`.

## I7 [DONE 2026-07-25 — IDENTITY LANDED, MODELS NOT PROMOTED] Venue canonicalization + duplicate player-ID merge
~149 venue substring-collision pairs ("Bay Oval" vs "Bay Oval, Mount
Maunganui") fragment venue history; 94 player names map to >1 registry ID
(split ELO/stats histories, double cold-start). Alias/merge maps at
cache-build; full rebuild + retrain both models; sharpens LANDED venue
features.

**Audit checkpoint (2026-07-24):** the refreshed male-T20 corpus contains
9,519 matches, 467 venue strings, 187 normalized equality/substring pairs,
and the same 94 repeated player display names. The player premise is
falsified: zero of the 94 groups shares a non-empty Cricinfo ID or full-name
and DOB signature; inspected examples are distinct homonyms, so no player-ID
merge is currently justified. Venue candidates also contain dangerous generic
labels (`County Ground`, `National Stadium`) and real subvenues (`Eden Park`
vs `Eden Park Outer Oval`). `scripts/audit_identity_collisions.py` therefore
keeps all substring-only, multi-city, and possible-subvenue pairs review-only.
Proceed only with a reviewed, versioned venue map; do not create a player
merge map from display-name equality. See
`reports/i7_identity_collision_audit.md`.

**Activation checkpoint (2026-07-25):** all 94 reviewed aliases are active.
`scripts/identity_maps.py` is the single exact-match implementation and rejects
conflicts, chains, self-aliases, invalid statuses, and wrong versions. The map
version/SHA/count now travels with SQLite, ball/match parquet, trained-model,
live-snapshot, and new forward-holdout artifacts; smart caches and live
inference fail closed on old or missing contracts. Cache building, ball and
match features, live fixtures, same-day replay, Polymarket construction, and
evaluation IDs now share the same canonicalization. Raw JSON and frozen
historical odds/evaluation artifacts remain unchanged. Full cache/parquet
rebuilds and both model retrains were completed in isolated I7 namespaces;
neither model replaced production. See
`docs/I7_VENUE_IDENTITY_CONTRACT.md`.

**Rebuild checkpoint (2026-07-25):** the isolated cache preserved 9,519
matches and reduced 467 raw venue strings to 373 canonical identities. Raw
ball LL is mixed/neutral (validation -0.00129 better, test +0.00195 worse).
The direct M7 retrain is directionally worse on all paired iteration slices:
≥$50k LL 0.6299 → 0.6421 and ROI +21.90% → +17.49%, although
competition-block Δ intervals cross zero. The intended venue-enriched slice
does not improve.

The final ball-simulation gate used a fresh frozen-v3 control under the exact
same current D15 simulator. I7 is directionally worse overall (LL 0.6845 →
0.7042, Brier 0.2433 → 0.2530, ROI +9.07% → +0.46%) and at ≥$50k
(LL 0.6970 → 0.7176, ROI +9.79% → +1.38%); block-bootstrap delta intervals
cross zero. The venue-enriched slice also worsens directionally (LL 0.6959 →
0.7160; ROI +15.01% → -0.30%). Do not promote either I7 model. Retain the
reviewed identity layer, provenance contracts, fail-closed compatibility
checks, and isolated artifacts; do not attach I7 metadata to legacy models.
The consumed forward holdout remained closed. See
`reports/i7_rebuild_checkpoint_20260725.md`.

**Live compatibility checkpoint (2026-07-30):** `predict_fixture.py` now
requires an explicit venue-identity family. Its live default `legacy` mode
preserves raw venue labels and permits the frozen pre-I7 model/cache/snapshot;
opt-in `i7` canonicalizes the fixture and requires matching map provenance
across all three artifacts. Declared modes cannot be mixed, and diagnostics
record raw/effective venue. This is a serving bridge only: I7 remains mandatory
for I8 and all new training. See `docs/I7_LIVE_COMPATIBILITY.md`.

## I8 [DONE 2026-07-30] Per-player phase dists + batter-vs-bowler cell
The flagged highest-value untried ball features need new SQLite
getters/tables (`stats_sqlite_backend.py` — forbidden to the loop). Do the
schema-v5 plumbing interactively, then queue the training experiment as a
future D idea.

**Contract frozen 2026-07-30:** I8 adds exactly 18 features to the 114-feature
I7 recipe: current-phase batter and bowler six-class distributions plus the
exact batter–bowler six-class distribution. Player-phase cells shrink to the
player's already-shrunk overall profile with `k_phase=30`; H2H shrinks with
`k_h2h=60` to the arithmetic mean of the batter/bowler profiles. The active
I7 venue identity and inclusive-total-run I7 delivery semantics remain fixed.
Artifacts are isolated under the `i8` cache/data/model namespace and require
schema v5. See `docs/I8_FEATURE_CONTRACT.md`.

**Result:** schema-v5 storage/readers, 132-feature training, exact contract
sidecars, and an isolated fail-closed simulator landed. Against I7, I8
improved test ball LL 1.631620 → 1.628690 and Brier 0.781231 → 0.779532; the
paired test Brier delta was confidence-clean, while LL narrowly was not. On
the consumed 255-match diagnostic, LL improved 0.7042 → 0.6825 but flat ROI
fell +0.46% → -1.49%; every competition-block delta interval crossed zero.
Removing the largest 5% upset win leaves I7/I8 ROI at -7.02%/-8.98%. Keep I8
as an isolated candidate, do not tune on consumed sets, and require a new
post-2026-07-30 terminal window before promotion. See
`reports/i8_phase_matchup_checkpoint_20260730.md`.

## I9 [FAILED 2026-07-30] ELO cold-start / provisional K (`parsing_v2.py`)
Debutants start at exactly 1500 with K as low as 1.0 (domestic); add a
provisional high-K warm-up (or uncertainty-scaled K) so new signings don't
sit at the mean for a season. Full cache rebuild + retrain.

**Precommitted design (2026-07-30):** use independent batting/bowling rated-
delivery counts and
`K_role = K_base * (1 + 3 * max(0, 1 - n_role/120))`. The multiplier is 4×
at debut and reaches the existing K exactly at 120 prior role deliveries.
Do not sweep the multiplier or threshold in the first run. Build against the
I7 identity stack with I8 features disabled, version every state/artifact,
gate first on validation provisional-event LL with global/established-player
guards, and require a new untouched forward window for promotion. See
`docs/I9_PROVISIONAL_ELO_EXPERIMENT.md`.

**Result:** implementation, isolated caches/parquets/models, exposure
rehydration, same-day replay, and fail-closed provenance all completed. The
fixed-K control exactly reproduced I7 ball LL (validation 1.6376, test
1.6316). The candidate improved overall validation LL to 1.6358 and the
30,241-ball provisional slice by 0.00110, but the paired match-block interval
was [-0.00759,+0.00541], so the primary gate did not clear. Overall LL/Brier
and established-player guardrails all passed. The five-seed direct model was
worse at every seed; mean validation LL regressed 0.65276 → 0.65513
(+0.00237). I9 is rejected without parameter retuning; serving artifacts are
unchanged. See `reports/i9_provisional_elo_checkpoint_20260730.md`.

## I10 [DONE 2026-07-24] Live-fixture operational hardening (`predict_fixture.py`)
Live prediction now inspects SQLite/tracker coverage before provider/model
loading, requires matching source counts, and fails when either state
component is more than 14 days behind the fixture. A diagnostic stale override
is explicit and suppresses betting. Multiple tracker source pools merge under
the I6 order, so the consumed forward sidecar (9,920 matches through
2026-07-13) and future live-only caches have a supported non-destructive path.
Unknown players remain a soft warning.

A7 now uses normalized two-team market probabilities, the exact frozen
`|elo_diff| <= 5` positive-edge / `>5` edge>10pp rule, and requires ≥$50k
volume. It is deliberately shadow-only (`execution_authorized=false`,
`bet_team=null`) because the forward economic decision did not confirm;
stale state, missing liquidity, and boundary failures suppress the shadow
candidate. Ten focused tests cover freshness, source mismatch, odds
normalization, strict boundaries, liquidity, and stale-state suppression.

## I11 [DONE 2026-07-16] Sim micro-hygiene (`sim_v1_2.py`, two-liners)
`if config.random_seed:` / `if seed:` treat seed 0 as unseeded (:3657,
:3686) — compare against None instead; remove the shipped TypeError
debugging tripwire in `simulate_match` (:3522-3544).
**Done:** all three seed checks now `is not None`; tripwire removed.
Byte-identical numerics for any seed ≠ 0 (guard-only changes);
`test_lineup_extraction.py` 10/10 pass.

## I12 [INTERACTIVE] Women's-corpus model (new track)
1,745 women's matches already on disk are filtered out of everything; a
separate women's model is zero-download new coverage. No odds/eval gate
exists yet — needs its own eval design first.

## I13 [DONE 2026-07-24] Upgrade the E2 fair-baseline bar
B9 built a usage-share top_bowler baseline (as-of EB-shrunk
expected-deliveries × per-ball wicket rate, lineup-uniform debutant
prior) that beats BOTH the E2 career-share baseline (ΔBrier −0.0055
[−0.0070, −0.0040]) and the calibrated sim itself (sim − usage +0.0038
[+0.0026, +0.0051] CI-clean, robust to shrinkage) — falsifying E5's
"first binary family beats a fair baseline" at the competent-bettor bar.
`prop_fair_baselines.py` is loop-forbidden, so the bar upgrade is
interactive: adopt the usage-share construction for `top_bowler`
(reference implementation `scripts/auto/b9_usage_baseline.py`), and
consider the analogous expected-balls × rate count-model upgrade for
`bowler_wkts_{1,2,3}plus`. Re-state the standing prop-skill claims
(E2/E5 reports, CLAUDE.md prop-framework paragraph) after the bar moves;
sim-gate guard families that reference "margin vs fair baseline" keep
meaning only relative to the bar version in force.

**Done:** `prop_fair_baselines.py` now emits the versioned
`e2-v2-usage-top-bowler` bar from a v2 corpus that includes zero-ball XI
appearances. On the canonical D15 n=261 detail, the usage baseline
reproduces B9 exactly: Brier 0.0747 vs sim 0.0785; sim − baseline
**+0.0038 [+0.0026, +0.0051]**, so the baseline wins CI-clean. The
analogous Poisson expected-balls × rate candidates for
`bowler_wkts_{1,2,3}plus` were all CI-clean weaker than the existing
EB-shrunk threshold-rate baselines (candidate − retained +0.0067
[+0.0031,+0.0106], +0.0043 [+0.0022,+0.0064], +0.0011
[+0.0004,+0.0020]), so they were explicitly rejected. E2, E5, and the
standing prop guidance now state that no binary prop family clears the
fair-baseline bar.

## I14 [INTERACTIVE] Physical venue context: dimensions, altitude, and seasonal weather
The learned venue ID embedding has to infer physical ground characteristics
from match outcomes alone and cannot generalize those characteristics to a
new or renamed venue. Build a canonical venue registry with latitude,
longitude, altitude, and sourced boundary geometry (straight and square
distances, ideally ranges plus observation date). Boundary ropes move between
matches, so dimensions must carry provenance and uncertainty rather than be
treated as timeless exact constants. Add monthly climate normals for
temperature, humidity/dew point, wind, and precipitation; keep actual
match-day forecasts/observations as a separate time-indexed feature to avoid
mixing climatology with information that was unavailable at prediction time.

At modeling time, concatenate normalized physical/context features with the
learned venue embedding instead of attempting to encode continuous values as
venue IDs. Test whether this improves unseen/low-history venue performance,
six/four rates, innings totals, and chase calibration. Prerequisites: I7 venue
canonicalization, source/licensing audit, missing-value indicators, and an
as-of join test. Gate on grouped-by-venue holdout performance so repeated
matches at major grounds cannot hide regressions at sparse venues.

**Temporal-venue hypothesis (2026-07-30):** canonical identity should remain
stable, but the learned venue state need not be timeless. Renovations, square
rotation, pitch preparation, rope placement, and climate regimes can make one
ground play differently across eras. Do not represent that by retaining two
names for the same venue. Instead, test recency-weighted or era-indexed venue
outcome profiles/embeddings, with change points tied to dated physical-registry
evidence where available. Compare against the static I7 identity on
forward-chained and grouped-by-venue gates so temporal flexibility cannot
become leakage or a high-volume-ground memorizer.

**Three-run motivation (2026-07-24):** threes are rare overall
(10,463/2,477,116 legal balls, 0.4224%) but differ sharply at high-volume
grounds: MCG 1.0587%, SCG 0.9506%, Mirpur 0.2464%. This is one reason to retain
raw batter threes for future physical-venue modeling. It is not a reason to add
a three-run simulator sub-draw before the geometry exists: doing so now adds
complexity and runtime without a demonstrated downstream gain.

## I15 [DONE 2026-07-30] Stable match identity for same-day doubleheaders
The synthetic `{date}_{team1}_{team2}_{venue}` key is not unique. The I7 test
parquet has 798 rows but only 788 keys across ten same-day same-team/same-venue
doubleheaders; prediction JSON serialization is silently last-write-wins.
The frozen production test already had nine collisions. No collision overlaps
the current 261-match Polymarket set, so published iteration metrics are not
affected. Replace internal primary keys with Cricsheet ID, retain the
synthetic key only as display/join metadata, and make odds manifests carry the
Cricsheet ID resolved during construction. Fail closed on any remaining
one-to-many join. Do this before expanding evaluation to markets that can
contain same-day doubleheaders.

**Done:** future materialization, odds, prediction, simulation/evaluation, and
forward-holdout artifacts now use Cricsheet file stem as primary `match_id`;
the synthetic value is explicit `display_match_id`. Frozen artifacts remain
readable through a legacy alias path, while duplicate primaries and ambiguous
aliases fail closed. The audited I7 test parquets preserve all 798 Cricsheet
IDs versus 788 legacy keys. See `docs/I15_MATCH_IDENTITY_CONTRACT.md` and
`reports/i15_match_identity_checkpoint_20260730.md`.

## I16 [DONE 2026-07-30 — ADOPTED] Adopt D12 swap augmentation in the production match model
**Decided 2026-07-30 (human-approved, option a):** the archived D12 swap
arm was promoted verbatim as `models/xgb_match_v3_m7_swap_production` and
`predict_fixture.py` now defaults to it. Fresh retraining on the pre-I7
legacy frame is deliberately impossible (I7 trainer contract), so
promotion of the 2026-07-17 artifact — whose base sibling reproduces
frozen M7 at max |Δp| = 0 — is the faithful adoption. Confirmation:
I3-block iteration readout (≥$50k LL 0.6215 vs base 0.6299, ROI +24.53%
[-1.98, +46.37] over 19 blocks) plus golden audit (LL 0.6576/0.7009 vs
base 0.6680/0.7078; beats matched market LL on both slices where base
does not; descriptive at 5–6 blocks). Block ROI CIs still straddle zero —
no CI-clean betting-edge claim; A7 stays shadow-only. During
confirmation, a blend cluster-stamping bug that degraded I3 blocks
(134 vs 19) was found and fixed. See
`reports/d12_swap_promotion_20260730.md`.
The night loop validated team-swap symmetry augmentation on the exact
production configuration (D7 on v2_clean, then D12 on the production
48-feature frame; `research/reports/auto/D12.md`, LANDED 2026-07-17). The
D12 control byte-reproduces `models/xgb_match_v3_m7_production/`
test predictions (782/782, max |Δp| = 0); the swap arm improves the paired
5-seed ≥$50k mean by ΔLL −0.0092 (better 5/5 seeds, floor 0.007) and ΔROI
+3.39pp (up 5/5, floor 2.3), halves the ROI seed-std, and is consistent at
≥$100k (ΔLL −0.0123). The loop's recommendation — retrain production with
`--swap-augment`, otherwise identical config — was explicitly left as a
human decision and sat untracked from 2026-07-17 until this entry
(2026-07-30 review).

## I17 [DONE 2026-07-30 — ADOPT-CANDIDATE] Swap augmentation on the I7 identity frame
**Interactive, precommitted** (`docs/I17_I7_SWAP_SUCCESSOR.md` frozen
before any swap-arm training). Motivation: the legacy production line's
state cache ends 2026-04-16 and cannot be regenerated; only the I7 stack
has a fresh-state build path, and D12 swap was validated on the legacy
frame only. Paired A1 seeds {7,13,29,42,101}, M7 config + `--monotone` on
`data/xgb_match_data_i7`, base vs `--swap-augment`; eval via the
cricsheet-id-stamped iteration envelope (new
`scripts/patch_envelope_cricsheet_ids.py`; 261/261 joined, determinism
gate max |Δp| = 0 vs archived `models/xgb_match_i7`). **Transfer
confirmed: swap better LL on 5/5 seeds at every slice; ≥$50k mean ΔLL
−0.0144 (floor 0.007, legacy-frame effect was −0.0092), ΔROI +5.18pp.**
Swap-i7 beats the slice-matched market LL (0.6482) on 5/5 seeds; trails
legacy production by ~0.005–0.009 LL (inside the precommitted 0.02
threshold; legacy was selected on this set). Seed-29 I3 readout: ROI
+20.54% [−5.56, +43.47], 19 blocks — straddles zero, no edge claim.
Swap+M7-on-i7 is the designated production-successor config; promotion is
a separate decision gated on an i7 golden-frame audit and the fresh-state
serving cutover plan. `reports/i17_i7_swap_eval_20260730.md`.

## I18 [P1] [PENDING] i7-identity golden frame + swap-i7 golden audit (I17 promotion gate)
**Night-executable.** Build the golden (124-fixture) match frame under the
I7 identity contract and score the fixed I17 successor candidate on it.
This is a *gate readout for a human promotion decision* — golden is
audit-only, the candidate is pre-specified, and NOTHING about the result
feeds model selection or production. Steps:
1. Materialize an i7-identity golden frame (state through 2026-06-17
   fixtures; start from `scripts/build_i7_match_frame.py` and the I7/I15
   contracts — `docs/I15_MATCH_IDENTITY_CONTRACT.md`,
   `docs/OPERATIONS.md` § Operation 7). Frame must carry
   `venue_identity.json` (venue_aliases_v1), `cricsheet_id`, and
   `match_identity_version`; fail closed otherwise.
2. Gate A (integrity): all 124 golden odds rows join by `cricsheet_id`;
   `verify_forward_holdout` still reports zero overlap / unchanged
   fingerprint; no write touches production caches, `data/golden/`
   inputs, or `data/forward_holdout/`.
3. Gate B (readout): score `models/auto/i17/swap_seed29` (candidate) and
   `models/auto/i17/base_seed29` (control) — regenerate via the exact
   commands in `reports/i17_i7_swap_eval_20260730.md` if absent
   (deterministic, max |Δp|=0) — then blend→reslice vs
   `data/golden/betting_odds_golden.json` at all/≥50k/≥100k with I3
   blocks. Report LL/ROI vs slice-matched market. Verdict is DESCRIPTIVE
   regardless of numbers (golden slices are ≤11 blocks): log the readout,
   do not adopt, promote, or revert anything.
Deliverables: `reports/auto/I18.md`, frame under `data/xgb_match_data_i7/`
(golden split only; do not touch train/val/test), results.tsv row marked
descriptive. Human follow-up (NOT night work): fresh-state serving
cutover plan, then the promotion decision.

The decision now includes a frame choice: (a) retrain the frozen-M7 line
`xgb_match_v3_m7_production` with `--swap-augment` (direct adoption of the
D12 result), or (b) fold `--swap-augment` into the I7-identity line
(`models/xgb_match_i7`) and let the I8-style post-2026-07-30 terminal
window arbitrate promotion. Either path requires golden-eval confirmation
on the production-launch checklist before `predict_fixture.py` switches,
and must not touch the consumed forward set. D12's frozen-vs-unfrozen
frame observation (report caveat 3) folds into the same decision.
