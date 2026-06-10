# E-Series Improvement Experiments — Session Summary (2026-06-09/10)

**Branch**: `improvement-experiments` (12 commits; `main` untouched)
**Detailed per-experiment reports**: `reports/e1_temperature_sharpening.md`,
`reports/e2_prop_fair_baselines.md`, `reports/e3_seed_ensemble.md`,
`reports/e4_quantile_pooling.md`, `reports/e5_class_weight_fix.md`,
`reports/e6_inplay_winprob.md`

---

## The protocol

Six experiments (E1–E6), run autoresearch-style: one change per experiment, a
pre-registered keep/discard rule stated *before* seeing results, all fitting
and selection on the validation set only, the iteration/test sets used
strictly as readouts, and every outcome — including failures — committed with
a full report.

## The six experiments at a glance

| # | Experiment | Verdict | One-line result |
|---|---|---|---|
| E1 | Temperature sharpening (match model) | ❌ discarded | First-ever transform to beat market LL on iteration ≥$50k (0.6246 < 0.6267), but flat ROI fell +21.9% → +12.3% |
| E2 | Fair baselines for prop families | ✅ **landed** (eval harness) | No binary prop family beats a fair baseline; sim's real skill is continuous score forecasts |
| E3 | 10-seed ensemble (match model) | ❌ discarded | Ensemble *worse* than production — exposed seed luck in the M7 headline |
| E4 | Quantile lineup pooling (match model) | ❌ discarded (val rule) | Bowling quantiles carry real signal but val LL regressed; favorable iteration readout recorded as a forward-test hypothesis only |
| E5 | Ball-model bias root cause | ✅ **landed** (vector calibrator) | v7 was sampling class-weight-tilted probabilities raw; a 6-parameter val-fit correction fixes the tail-event overshoot |
| E6 | Direct in-play win-prob model | ✅ **landed** (`models/inplay_winprob_v1`) | In-play win probability = chase math × team strength; crease/momentum state adds nothing; new model supersedes the sim in-play |

Pipeline bugs found and filed along the way (TODO.md § "Pipeline bugs"):
the sim feeds `venue_encoded=0` on every ball, `innings_id` is an unstable
hash (`parsing_v2.py:1255`), and the class-weight tilt itself (fixed by E5).

---

## The three keepers, in depth

### E5 — The vector-scaling ball calibrator (the most consequential fix)

**The problem.** `xgboost_v2.py` trains the ball model with sklearn's
`balanced` class weights. During training, every wicket ball counts ~6× as
much as a dot ball in the loss (weights inversely proportional to class
frequency). That is a common choice for imbalanced classification when you
care about *ranking* rare events — but it has a precise, predictable side
effect on *probabilities*: the model no longer estimates P(outcome |
situation). It estimates what the probabilities *would be* in a fictional
world where all six outcomes are equally common. Mathematically, weighting by
w_c is equivalent to changing the class prior, so the model's output is
tilted: p_weighted(c|x) ∝ w_c · p_true(c|x).

The sim then sampled these tilted probabilities **raw** — `ball_calibrator`
has defaulted to `None` since v4. Teacher-forced on 186,667 real deliveries:
the model says P(wicket)=0.121 per ball when reality is 0.057; boundaries are
over-stated by ~+0.05 absolute each; dots and singles under-stated.

**Why nobody caught it for 14 months.** The two inflations partially cancel
at match level — simulated innings score *faster* but end *earlier*, so
innings totals and winner probabilities come out roughly sane. Every eval the
project ran (winner LL, ROI) aggregates to match level and blends the
offsetting errors away. But prop families that count tail events — "this
bowler takes 2+ wickets", "powerplay over 55.5" — amplify exactly the
inflated channels. That is precisely the list of families the prop framework
flagged as "systematically over-stated." The biases weren't a mystery about
cricket; they were an artifact of the loss function.

**Why the fix is shaped the way it is.** The textbook correction is to divide
by the class weights and renormalize. That *over*-corrects (runs/ball 1.04 vs
actual 1.28), because an early-stopped, regularized booster never reaches the
full theoretical tilt; it sits somewhere between the weighted and unweighted
posterior. So instead of assuming the tilt, it is *measured*: find six
multipliers such that, averaged over the validation split, the corrected
probabilities match the validation class frequencies (iterative proportional
fitting). Six parameters, fit on val only, fit under the sim's actual input
distribution. A per-class multiplicative correction preserves everything the
model knows — the *ordering* of situations by riskiness, the matchup effects,
the venue effects — and only fixes the overall exchange rate between classes.

**Validation.** Held-out test, teacher-forced: runs/ball error +0.383 →
+0.024, P(wicket) error +0.065 → −0.002, multiclass LL 1.608 → 1.520. Then
the full n=261 Monte-Carlo rerun: bowler-wickets overshoot halved-to-fixed
(+0.036→+0.016, +0.014→+0.003), PP-total 55.5 halved, several families moved
to parity, and **top_bowler became the first prop family ever to beat a fair
baseline with a CI excluding zero** (Δ −0.0023 [−0.0038, −0.0008]). Honest
costs: per-batter-runs forecasts lost their edge (the raw sim's two biases
accidentally cancelled for that family), and two boundary-*clustering*
families regressed (first-over runs, highest-over) — the tilt isn't uniform
across match phases, and a single global multiplier can't be right
everywhere. Clean follow-up: phase-conditional scaling (3×6 parameters);
filed, not yet built.

**How to use it:** pass `--ball-calibrator vector` to `prop_backtest.py` for
anything involving wicket counts, innings/PP totals, top-bowler, or
fours/sixes counts; keep the raw sim for per-batter runs until the
wicket-attribution model (G2) improves.

### E6 — The direct in-play win-probability model

**The idea.** The project already learned this lesson once at the pre-match
level: a model trained *directly* on "who won" beat the simulator at
predicting winners. E6 applies the identical logic in-play. Instead of
answering "who wins from this position?" by rolling a generative ball model
forward 150 balls — where every per-ball bias compounds — treat each
historical delivery as a labeled training example: (match state, did the
batting side go on to win?). The corpus gives 1.81 million such examples for
free. A discriminative model trained on them optimizes the target quantity
directly, with no compounding.

**The ladder design.** Four nested models with identical hyperparameters, so
any difference between them is purely the value of the added information:
(1) pre-match team strength only, (2) chase math only
(score/wickets/balls/target), (3) both, (4) both plus everything the sim
"knows" — who's at the crease, their form, partnership, last-30-balls
momentum, pressure index.

| model | test LL | test AUC |
|---|---:|---:|
| prior_only | 0.6703 | 0.604 |
| resource_only | 0.5520 | 0.792 |
| **fair_blend (landed)** | **0.5418** | **0.801** |
| full (+crease/momentum) | 0.5423 | 0.801 |

**The result, and why it matters.** Model 4 beats model 3 by nothing: Δ test
LL +0.0005, CI [−0.003, +0.004] across 780 held-out matches. This replicates
the MLC 2025 finding at 20× the sample, and it is a real piece of cricket
knowledge: *in-play win probability is resource arithmetic scaled by
pre-match team strength*. Who specifically is batting, and how the last few
overs went, carries no measurable additional signal about the final result —
it is already priced into the score and wickets. The simulator read ~0.60 LL
on the chase checkpoints that were its home turf; fair_blend reads 0.40–0.49
there.

**How to use it:** `models/inplay_winprob_v1` is now the source for win-prob
worms, "decision review" deltas (P(win) before/after an over), and in-play
scenario tables — instant inference, no Monte Carlo, validated out-of-sample.
It is also the natural starting model if in-play odds are ever captured (no
ROI claim exists until then).

### E2 — The fair-baseline harness (changes what counts as evidence)

**The idea.** "Skill" is only meaningful relative to what a competent bettor
already knows for free. The old prop framework measured Brier-skill against
the *base rate* — but nobody prices "will Charles hit 2+ fours" at the
league-average rate; they look up Charles's career rate. So for every prop
family, build the baseline a competent human would use: EB-shrunk career
rates for batter/bowler props, venue-shrunk historical rates for totals, a
positional prior for top-scorer — all computed strictly as-of each match date
(no future data, same first-write-wins semantics as the SQLite cache), with
significance from a bootstrap clustered by match (rows within a match are
correlated; row-level bootstraps overstate certainty).

**The result.** Against fair baselines, **zero** binary prop families showed
sim skill — the May 2026 "ship as-is" list was a base-rate artifact. The
sim's genuine, CI-backed edge is in *continuous score distributions*
(per-batter expected runs MAE 14.02 vs 14.73 career baseline; match highest
score 16.56 vs 18.45), which makes sense: that is where composing venue +
lineup + matchup interactions beats any single lookup rate. It also killed
the "inverse play" thesis cleanly: fading the sim is only profitable against
a counterparty that prices like the sim, and the fair baseline beats both the
sim and its inverse.

**How to use it:** `scripts/sim_eval/prop_fair_baselines.py` is now the
permanent bar — any future prop-bet claim must beat it, not the base rate.
After E5's calibrator, top_bowler is the first family to clear it.

---

## The discards — and what each one bought

**E1 (temperature sharpening).** The reliability diagnostic was right: the
model is under-confident (slope ~1.2 on val) with market-level resolution,
and a val-fit expansive transform delivered the long-sought LL win over the
market (iteration ≥$50k 0.6246 < market 0.6267). But flat ROI collapsed
(+21.9% → +12.3%, CI re-straddling zero), and the mechanism is now
understood: betting picks the side where model > market. A monotone
sharpening flips the bet side on exactly the matches where the market price
sits *between* the raw and sharpened probability — i.e., the matches where
the raw model quietly disagreed with the market toward 50%. Those timid
counter-market calls were the profit pocket. Durable rule: **the model's side
relative to the market is the alpha; the probability's magnitude is
mis-scaled for LL but correctly scaled for side-selection.** The fitted
temperature survives as a sizing/display layer (honest probabilities for
Kelly), never for picking sides.

**E3 (seed ensemble).** Averaging 10 seeds of the exact production config was
*worse* on val and iteration — because seed 29 (production) turns out to be
the best of all 10 on val, and M7's config sweep was run with that seed
fixed. The production headline therefore contains favorable seed variance.
Practical consequence: judge the forward test against the ensemble-tempered
expectation (**~0.64 LL / +16% ROI** on ≥$50k), not the headline
(0.6299 / +21.9%). This prevents wrongly concluding the model "broke" when
live results come in below the headline.

**E4 (quantile pooling).** The one feature idea the M3–M5 failures didn't
foreclose: pool lineup player ratings by max/spread/best-k instead of means
(means collapse to team aggregates; quantiles preserve "one elite bowler ≠
flat attack"). The correlation discipline validated the cricket logic —
best-2-bowling-ELO diff carries *more* target signal (r=0.158) than the
bottom-5 mean it replaces (r=0.143). But val LL regressed, so it was
discarded under the pre-registered rule — even though the iteration readout
was favorable (≥$50k ROI CI [+3.6, +47.0]). Landing it on that readout would
have been selecting on the test set. It is the top candidate to re-test the
day forward-captured data exists. Features remain in the materializer
(`_quantile_elo_features`), excluded from production via `--drop-features`.

---

## The big picture

The ball-level stack got materially better: its probabilities are now honest
(E5), its evaluation is now honest (E2), and its in-play job was handed to a
model that is actually validated (E6). The match-level model resisted three
disciplined attempts — which is itself informative: at n=261 readout
resolution, M7 appears to be at its local optimum, and every remaining
match-level question (E4's features, E1's sizing layer, E3's tempered
expectation) now has the same answer: **start the forward Polymarket capture
(C2)**. That has been the highest-value unstarted item in the TODO since M1,
and this session turned it from "nice to have" into the explicit gate for
every open hypothesis.

### Recommended next steps, in order

1. **Forward capture (C2)** — gates every open match-level hypothesis and is
   the only path to CLV, the durable measure of edge.
2. **G2 wicket attribution** — now cleanly isolated as the next ball-level
   target (no longer confounded with the marginal tilt).
3. **Phase-conditional vector scaling** — fixes the two boundary-clustering
   regressions E5's global correction introduced.
4. Re-render the prop per-match drilldowns from the calibrated n=261 detail
   (`reports/prop_calibration_detail_vec_n261.json`).
