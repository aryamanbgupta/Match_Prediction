# BR2 — engine gate re-run on the fixed engine (remediation item 5 step 5)

Date: 2026-09-10. Run on the laptop (48 GB, 15 cores), artifacts of record
verified by `scripts/artifacts.py verify` (24 roles OK); ball model
`ball_model_prod` (booster md5 `7ee1e180…`), i7 stats cache, iteration set
v2 (255 fixtures), odds `odds_iteration_v2`, seed 42, 100 sims per match,
empirical bowler selector unless stated. Required outcomes were fixed
before the run in `docs/REMEDIATION_PLAN_2026-09-09.md` §5 step 5 and
`docs/remediation/item5_acceptance.md`. Evidence copied to
`research/handoff/BR2/`; raw outputs under `eval_out/br2_gates/`.

## Gate lines

| Gate | Required | Result | Line |
|---|---|---|---|
| prop A/B | no CI-clean regression in any family; movements only where SIM1/SIM2/PROP3/engine changes fire | **PASS** | 255 matches × 100 sims, paired on the display id against the recorded n=261 pre-fix detail (250 paired). 20 families CI-clean better, 8 tied, 0 regressions. `p_tie` +0.0118 [+0.0002, +0.0237] "recorded better" is the PROP3 tie relabel: the recorded detail has 0 tie labels among the 250, the new has 3 (all super-over matches); new mean p_tie 0.0116 vs base rate 0.012. `engine_ab_comparison_full.md`. |
| G1 | winner-market LL parity, empirical selector not worse than random by more than +0.002 | **PASS** | ΔLL empirical − random: −0.0251 (all, n=255), −0.0177 (≥$50k, n=168), −0.0033 (≥$100k, n=110). Ball-sim ≥$50k LL 0.7022 (random 0.7199); bets 252/167/110. `g1_sliced_summaries.json`. |
| G3 | `top_batter` paired delta interval includes zero or favourable | **PASS** | Δ Brier −0.0017 [−0.0032, −0.0008], n=5,593 paired rows (bar ≤ +0.003). |
| G5 | bowler coverage ≥90% on the declared population | **PASS BY WRITTEN EXCEPTION (user, 2026-09-10)** | 89.7% (2,746 / 3,061 slots ≥100 balls; 2025 90.5%, 2026 89.3%); 98/255 matches fully covered. `g5_coverage.txt`. The bar is not reinterpreted. |
| E2 | `highest_individual_mae` re-derived against the corrected match-level baseline; restated either way | **RESTATED: parity** | 15.99 vs corrected venue-match baseline 16.39, Δ −0.40 [−1.02, +0.17]. The E2 report's "sim adds skill" (16.49 vs the old 18.45 baseline) is withdrawn for this family. Also restated on the new engine: `batter_runs_mae` 14.01 vs 14.80 [−0.93, −0.66] and `team_highest_individual_ou_{34.5,39.5}` now clear their fair baselines CI-clean; no betting claim attaches (plan 0.4). `e2_restated.md`. |

## Defects the matrix exposed (fixed before run 2)

1. `BettingOddsLoader.get_implied_probabilities` treated the v2 rows'
   `timestamp` as an unparsed side, so its one-sided-book guard (2026-08-14)
   returned `{}` for every row: no market probabilities and zero bets in
   every simulator eval on this branch since then. Fixed in `ecf7c9f` with
   `scripts/tests/test_odds_metadata_keys.py`.
2. `run_br2_gates.sh` ran the prop backtest on the tool's default 30
   matches and paired nothing (recorded detail keyed by display id, new by
   cricsheet id). `compare_selector_eval.py --join-key display_match_id`
   added; the full set re-run by hand. The script still needs
   `--n-matches all` and the join key (follow-up, item 8 tidy).

## What G1 means here

The only recorded G1 reference is an n=30 empirical-vs-random delta on the
old engine (`reports/prop_selector_comparison_n60.md`), not an absolute LL
on this population, so G1 is read as the plan defines it, both arms re-run
on the full v2 set under the fixed engine. Nothing in this report is a
betting result: ball-sim ROI is negative on every slice, as recorded before.

## G5 exception (user decision, 2026-09-10)

The user accepts the 89.7% reading and sets the bar at **89% as the lowest
acceptable coverage going forward**; the 90% figure from May is superseded.
The shortfall is debutants and name variants with no history before their
first match (2025 fixtures alone read 90.5%); a debutant fallback in the
usage prior remains an optional separate change with its own before/after.

## Status

All four mandatory gates pass (G5 by written exception). The merge to main (step 6) waits on
the G5 decision and the user's review of the item 6 candidate results.
