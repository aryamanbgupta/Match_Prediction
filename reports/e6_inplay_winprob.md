# E6 — Direct in-play win-probability model ✅ fair_blend LANDED as analytics artifact; crease-state extras add nothing

**Date**: 2026-06-09 · **Branch**: `improvement-experiments`
**Harness**: `scripts/e6_inplay_winprob.py` · **Artifact**: `models/inplay_winprob_v1/`
(fair_blend; `_full` variant alongside for reference)

## Hypothesis

The MLC 2025 in-play test showed the ball-by-ball *simulator* loses to
"chase math + pre-match team rating" (LL 0.6042 vs 0.4936 on 88 chase
states, n=32 matches). Apply the lesson that already worked for the winner
market: train P(batting side wins | ball state) **directly** on every
delivery of the corpus, and test whether live crease/momentum state adds
anything beyond chase math + rating.

## Setup

- Training rows: 1.81M deliveries / 7,727 matches (train split, through
  2024-12-30); val 120k / 517; test readout 181k / 780 (2025-07-01 →
  2026-04-16). Labels joined from cricsheet by (date, venue) with
  innings-1 legal-ball disambiguation for doubleheaders (the parquet's
  `innings_id` hash is irreproducible — flagged below). Ties, no-results
  and D/L-decided matches excluded.
- Nested ladder, identical XGBoost-hist hyperparams, early stop on val —
  deltas are feature value, not tuning:
  - **B0 prior_only**: pre-match team ELO/aggregates (static)
  - **B1 resource_only**: score / wickets / balls / target / RRR
  - **B2 fair_blend**: B0 + B1 (the baseline that beat the sim at MLC)
  - **E6 full**: B2 + crease & momentum (striker/non-striker form, ELO,
    partnership, last-10/30 balls, pressure index, phase flags)
- **A priori gate**: keep `full` iff it beats `fair_blend` on overall test
  LL with match-clustered bootstrap CI excluding 0.

## Results (test split, 780 matches, 181k states)

| model | val LL | test LL | test AUC |
|---|---:|---:|---:|
| prior_only | 0.6839 | 0.6703 | 0.604 |
| resource_only | 0.5400 | 0.5520 | 0.792 |
| fair_blend | **0.5355** | **0.5418** | **0.801** |
| full | 0.5355 | 0.5423 | 0.801 |

**GATE: Δ(full − fair_blend) = +0.0005, 95% CI [−0.0030, +0.0041] → FAIL.**

Per-checkpoint test LL (balls into innings):

| model | inn1@30 | inn1@60 | inn1@90 | inn2@30 | inn2@60 | inn2@90 |
|---|---:|---:|---:|---:|---:|---:|
| fair_blend | 0.6310 | 0.5965 | 0.5952 | 0.4897 | 0.4490 | 0.3961 |
| full | 0.6271 | 0.5982 | 0.5878 | 0.4999 | 0.4542 | 0.3935 |

## Conclusions

1. **The MLC finding replicates at 20× the sample**: who is at the crease,
   recent scoring, partnership state — none of it adds measurable win-prob
   signal beyond (score, wickets, balls, target) + pre-match rating. In-play
   win probability is resource math scaled by team strength.
2. **fair_blend is the keeper**: LL 0.5418 / AUC 0.80 across all states vs
   0.67 for the best static prior. It directly supersedes the simulator for
   any in-play probability use (the sim read 0.60 LL on its home turf,
   chase checkpoints, where fair_blend reads 0.40–0.49). Saved to
   `models/inplay_winprob_v1/` (9+10 features, retrain ~2 min).
3. **Product surface**: win-prob worm per match, "decision review" deltas
   (P(win) before/after an over), and pre-match scenario tables all come
   from this model with no Monte Carlo — and it is *validated* against
   780 held-out matches, which the sim never was.
4. **Betting**: this is the model an in-play market entry would start from;
   no in-play odds are captured yet, so no ROI claim is made. Forward
   capture of in-play prices is the C2-analogue prerequisite.

## Data-quality flag (pre-existing, now load-bearing)

`parsing_v2.py:1255` builds `innings_id` as `hash(json) % 100000` — salted
per process (irreproducible across runs) and collision-prone (~450 expected
collisions at corpus size). E6 works around it via (date, venue) joins +
innings-shape disambiguation (~5% of matches dropped at join). Worth fixing
at the source: emit the cricsheet filename stem instead. Filed in TODO.

## Artifacts

- `scripts/e6_inplay_winprob.py` (rebuilds everything; `--quick` for smoke)
- `models/inplay_winprob_v1/{model.pkl,feature_columns.txt}` + `_full/`
- `models/match_winner_map.json` (clean-match winner cache)
- `eval_out_e6_summary.json`
