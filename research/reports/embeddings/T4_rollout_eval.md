# T4 — first rollout evaluation: results, diagnosis, verdict

2026-08-08 · scripts/sim_t1.py + scripts/sim_eval/run_sim_eval_t1.py ·
255 matches × 50 sims, random bowler selector (empirical selector's
corpus artifact `models/b10_usage_corpus.pkl` is missing on this
machine — see Caveats) · odds: `betting_odds_polymarket_v2.json` (the
corrected, non-defective file) · full log:
`/Users/aryamangupta/.claude/jobs/e4159044/tmp/t4_eval.log` · raw results:
`match_evaluation_results_transformer_all_20260808_033548.json`

## Headline result: FAILED at the match/rollout level

| metric | T1 transformer (rollout) | market | production XGBoost match model |
|---|---|---|---|
| Log loss | **0.7156** [0.6701, 0.7585] | ~0.5940 (≥$50k corrected ref) | 0.6249 |
| Brier | 0.2611 | — | — |
| corr(prediction, actual result) | **0.149** | 0.420 | — |
| Flat ROI | **−15.6%** [−34.2%, +10.3%] | — | +3.38% [−14.6, +37.1] |
| Full Kelly ROI | −4.4% | — | — |
| 25% Kelly ROI | −1.1% | — | — |
| Win rate (bets placed) | 32.9% | — | — |
| Matches with disagreement ≥ threshold | 255/255 | — | — |

This is a clean, unambiguous regression relative to both the market and
the existing production stack — not a marginal or noise-level result (CI
for log loss does not overlap the market or production XGBoost numbers).
**This does not erase the per-ball diagnostics** (T1: 1.4372 val / 1.4288
test; delivery-information prototype: about +0.02 LL under its original,
non-comparable head setup). Those were computed on real, teacher-forced
sequences, so they answer a different question and retain their audited status:
T1 is promising but uncontrolled, and xR1 is only a proof of concept. What
failed here is the *rollout* — using T1 as a Monte Carlo simulator to produce
match win probabilities.

## Calibration and orientation: systematic first-innings bias

Binning by the transformer's first-team win probability was originally shown
only for probabilities >=0.5. Those rows total 223, not 255, so the table is a
partial diagnostic rather than a complete reliability curve:

| predicted range | n | mean predicted | actual win rate |
|---|---|---|---|
| [0.5, 0.6) | 83 | 0.550 | 0.434 |
| [0.6, 0.7) | 114 | 0.640 | **0.482** |
| [0.7, 1.0) | 26 | 0.740 | 0.577 |

Every shown bucket above 50% overstates the true win rate by 10–16 points. For
contrast, the market's calibration on the same matches is close to
diagonal (e.g. predicted 0.843 / actual 0.861 in the top bucket). The
summary statistic agrees: **average signed edge −7.9%**.

The stronger post-audit signature is orientation bias. Cricsheet's first listed
team is the first-innings team in all 255 evaluation fixtures. T1 assigns that
team a mean win probability of **0.5953** although it wins only **0.4549**, and
favors it above 0.5 in **86.3%** of fixtures. This 14.0-point bias must be
diagnosed through innings-level posterior-predictive checks before treating the
failure as generic sharpness or applying post-hoc calibration.

## History-sensitivity diagnostic: exposure bias remains a hypothesis

**Hypothesis.** T1's per-ball metrics were computed under *teacher
forcing* — `transformer_t1.collate` always feeds the model the REAL
previous-ball outcome as history (`py[b,1:n] = y[ix][:-1]`). But
`sim_t1.py`, used for Monte Carlo rollout, has no ground truth to feed —
it must condition on the model's OWN sampled outcomes at every step. This
is the standard exposure-bias failure mode for autoregressive sequence
models: strong under teacher forcing, degraded when forced to consume its
own (imperfect) samples, because self-generated trajectories can drift
into histories the model never saw in training.

**Test** (`scripts/t1_exposure_bias_check.py`, 300 held-out validation
innings, 34,134 balls, single seed): decode each innings two ways, same
balls, same true per-ball state/EB features (isolating the outcome-token
history path specifically, not the full rules engine):
- teacher-forced (real previous outcome fed each step)
- self-conditioned (model's own sampled previous outcome fed each step,
  same `random.choices`-style sampling the simulator uses)

| | mean per-ball LL |
|---|---|
| teacher-forced | 1.4387 (matches T1's reported 1.4372 — sanity check passes) |
| self-conditioned | **1.4654** |
| **gap** | **+0.0267** |

Gap by position in the innings (bucketed):

| balls | teacher-forced | self-conditioned | gap |
|---|---|---|---|
| 0–10 | 1.3819 | 1.3905 | +0.0086 |
| 10–30 | 1.4634 | 1.4857 | +0.0224 |
| 30–60 | 1.3764 | 1.4011 | +0.0247 |
| 60–90 | 1.4174 | 1.4523 | +0.0349 |
| 90–120 | 1.5118 | 1.5444 | +0.0326 |

The gap grows with position, demonstrating that the model is sensitive to a
sampled/corrupted outcome history. It does **not** provide a valid free-running
likelihood: after the first sampled outcome differs from reality, the script
scores the next real outcome while retaining the original real state/EB path,
creating a hybrid counterfactual trajectory. The +0.027 gap therefore cannot
be attributed quantitatively to exposure bias or used to explain T4 without a
full evolving-state posterior-predictive experiment.

## Confounds NOT yet isolated — do not over-attribute to the model itself

1. **Random bowler selector, not empirical.** `models/b10_usage_corpus.pkl`
   is missing on this machine (`build_bowler_phase_usage.py` refused to
   write a payload that would revert the B12-shipped selector). This run
   used `--bowler-selector random`, a documented departure from the
   production `EmpiricalBowlerSelector` default — who bowls when directly
   affects which EB features apply and is known to matter for realism.
   Production models are also evaluated with the empirical selector, so
   this is a real, uncontrolled confound in the comparison above, not
   just an equalizer.
2. **50 sims, not production's 100–1000.** Quantizes probabilities at 2%
   steps and adds Monte Carlo noise to the win-probability estimate
   itself — plausible contributor to variance, less plausible as a
   source of the *systematic* overconfidence pattern.
3. **Wrapper cache correctness**: reasoned through but not exhaustively
   proven bug-free. `sim_t1.py`'s per-innings token cache resets on every
   innings-number transition (verified: `MatchState.copy()` deep-copies
   `history`/`history_idx` per simulation, and innings alternates
   1→2→[new sim]→1 so mismatches are always caught) and the sampling path
   is confirmed genuine (`random.choices` on raw softmax weights, not
   greedy decoding) — but the desync/filler branch for out-of-order calls
   was never actually exercised/logged in this run, so it's unverified in
   practice, only by inspection.

## Verdict and next steps

**T4 rollout claim: FAILED as run.** Do not report match-level LL/ROI
numbers for T1 as representative of the transformer's quality — they are
confounded by the random bowler selector, low simulation count, unverified
wrapper paths, and a newly measured first-innings bias. Exposure bias is a
candidate mechanism, not a confirmed cause.

Before re-attempting a clean verdict:
1. Pull `models/b10_usage_corpus.pkl` (or rebuild via
   `scripts/auto/b9_usage_baseline.py --rebuild-corpus`) from the other
   workstation and re-run with `--bowler-selector empirical` — the
   apples-to-apples comparison to production.
2. Increase `--n-sims` (100+) to match production protocol.
3. Run innings-level posterior-predictive checks and counterfactual batting-
   order flips to locate the first-innings bias before changing training.
4. Re-run any history-sensitivity test with state/EB features evolving under
   the simulated trajectory and evaluate distributional sequence statistics.
5. Consider scheduled sampling only if those checks identify a genuine
   history-distribution failure. Do not use match-level recalibration to hide
   a persistent structural orientation defect.

## Honest scoreboard update

| layer | status |
|---|---|
| E-ladder (embeddings) | unaffected — FAILED as reported |
| Discriminability | PRELIMINARY — grouped/context-residualized test required |
| T1 per-ball | PROMISING PILOT — same-feature repeated ablation required |
| xR two-stage | SUPPORTED INTERNAL PROCESS MEASUREMENT — first future-player promotion gate FAILED |
| **T4 rollout (win probability, ROI)** | **FAILED original market run; corrected structural simulator PPC passes** |

## 2026-08-12 clean follow-up — superseded intermediate PPC

The registered parity/PPC follow-up resolves several T4 confounds without
overwriting the original result. The simulator now has exact training/live
feature parity (59,291 rows, zero failures at `1e-6`), empirical B10 bowling,
200 simulations per arm, explicit inclusive-delivery semantics, hardened
cache/reset behavior, and exact counterfactual tests.

On the frozen 24-match PPC cohort, the old artifact's mean first-batting-team
probability falls from 0.5714 to 0.4573; log loss improves from 0.7452 to
0.6848 and Brier from 0.2747 to 0.2454. Storage-label symmetry passes exactly
(480/480 comparisons). The original first-listed-team signature was therefore
not a stable model conclusion.

The rollout still fails its PPC. In paired batting-order flips, the same team
is 8.85 percentage points less likely to win when batting first than when
chasing (95% match-bootstrap interval -10.07 to -7.69 points), negative in all
24 matches. That is not an invariance failure: batting order is a real
intervention. It is compatible with the 7.94-point descriptive chase advantage
among 252 decided ordinary evaluation fixtures, though a strength/venue-
controlled estimate remains necessary. The actionable failures are elsewhere:
extras are under-produced by 2.32 events per innings and the empirical selector
uses roughly 7.5-7.9 bowlers rather than the observed 6.0-6.1. Calibration
remains ineligible. See `T1_SIM_PARITY_PPC_V1.md` for the registered design and
full diagnostics.

The isolated B18 empirical-extras arm subsequently fixes the extras frequency
(5.109 versus 4.583 observed), moves first-innings bias from -9.43 to +0.76
runs, lowers MAE from 31.62 to 30.17, and raises P10-P90 coverage from 0.625 to
0.708. Its winner LL is 0.7113 versus 0.6848 for the flat arm, an adverse paired
change whose 95% interval crosses zero. Adopt B18 for generative realism, not
as evidence of better winner prediction. Bowling-role behavior remains the
next structural blocker.

## 2026-08-13 corrected-engine closure

The preceding absolute PPC values are superseded by a defect found during the
bowling audit: the first over bypassed the configured selector. The fixed
engine routes the first over through the same policy as every later over and is
protected by a focused regression test.

On a matched corrected-engine control, B18 passes all three extras gates. It
moves extras from 2.274 to 5.127 events per innings versus 4.583 observed,
first-innings bias from -6.296 to -0.288 runs, and MAE from 31.205 to 30.153.
Winner LL changes from 0.6763 to 0.6742, with paired interval crossing zero.

The causal roster-aware bowling follow-up then passes all five structural PPC
gates: unique-bowler counts are 6.010/5.949 versus 6.125/6.000 observed;
storage symmetry is exact in 480/480 comparisons; first-innings MAE changes
only +0.156; winner LL has no CI-clean regression; and the -3.97-point paired
batting-first effect is compatible with the ordinary cohort's descriptive
order reference.

This closes the mechanical T4 simulator audit but does not revise the original
market rollout into a success. The correct statement is: **the original T4
winner/ROI experiment failed and was confounded; the repaired simulator now
passes its registered structural PPC and is eligible for fresh downstream
validation without calibration**.
