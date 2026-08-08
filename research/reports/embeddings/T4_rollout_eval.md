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
**This does not invalidate the per-ball results** (T1: 1.4372 val / 1.4288
test; xR: +0.02 LL from delivery info) — those were computed on real,
teacher-forced sequences and stand independently. What failed is the
*rollout* — using T1 as a Monte Carlo simulator to produce match win
probabilities.

## Calibration: the model is systematically overconfident

Binning by the transformer's own predicted win probability (team-1
orientation), n=255:

| predicted range | n | mean predicted | actual win rate |
|---|---|---|---|
| [0.5, 0.6) | 83 | 0.550 | 0.434 |
| [0.6, 0.7) | 114 | 0.640 | **0.482** |
| [0.7, 1.0) | 26 | 0.740 | 0.577 |

Every bucket above 50% overstates the true win rate by 10–16 points. For
contrast, the market's calibration on the same matches is close to
diagonal (e.g. predicted 0.843 / actual 0.861 in the top bucket). The
summary statistic agrees: **average signed edge −7.9% (overconfident)**.

## Diagnosis: exposure bias, empirically confirmed as a real contributor

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

The gap roughly quadruples from the first 10 balls to mid-innings,
consistent with compounding drift as sampled errors accumulate — the
signature of exposure bias, not random noise. This is a **real, measured
mechanism**, but the +0.027 LL degradation (ball-level, 6-class) is not
on its own large enough to fully explain the magnitude of the match-level
failure (a different metric on a different scale, and win-probability
rollout amplifies small per-ball biases nonlinearly over ~120+ balls).

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
confounded by at least one uncontrolled variable (bowler selector) and
one confirmed-but-unquantified mechanism (exposure bias).

Before re-attempting a clean verdict:
1. Pull `models/b10_usage_corpus.pkl` (or rebuild via
   `scripts/auto/b9_usage_baseline.py --rebuild-corpus`) from the other
   workstation and re-run with `--bowler-selector empirical` — the
   apples-to-apples comparison to production.
2. Increase `--n-sims` (100+) to match production protocol.
3. If exposure bias remains dominant after (1)–(2): mitigations to try are
   scheduled sampling during T1 training (mix teacher-forced and
   self-sampled prev-outcome tokens so the model sees its own error modes
   during training), or a post-hoc match-level recalibration layer
   (Platt/isotonic on the rollout output, same tooling as
   `scripts/calibration.py`) as a cheaper stopgap.
4. Re-run the exposure-bias check with state/EB features ALSO evolving
   under self-sampled outcomes (this run held them at ground truth) for a
   fuller estimate of total rollout drift, not just the outcome-token path.

## Honest scoreboard update

| layer | status |
|---|---|
| E-ladder (embeddings) | unaffected — FAILED as reported |
| Discriminability | unaffected — LANDED as reported |
| T1 per-ball | unaffected — LANDED as reported (1.4372/1.4288) |
| xR two-stage | unaffected — LANDED as reported (+0.02 LL) |
| **T4 rollout (win probability, ROI)** | **FAILED — confounded, needs clean re-run** |
