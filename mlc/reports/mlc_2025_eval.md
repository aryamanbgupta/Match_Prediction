# MLC — did our predictions hold up? (2025 backtest, 2026-06-07)

## What "our MLC predictions" means here

We generated **MLC 2026 Dallas-leg win probabilities today** (`fixtures/mlc_2026/`,
`reports/mlc_2026_dallas_leg.md`) — but those 7 fixtures are **2026-06-18 → 06-21,
unplayed**, so there is nothing to score yet. The only MLC matches with results are
past seasons. So this report backtests the **same production pipeline**
(`xgb_match_v3_m7_production` for the winner market; the v7 sim for ball-level) on
the most recent completed season, **MLC 2025** (33 matches, 2025-06-12 → 07-13), as
the closest read on whether the MLC picks are trustworthy.

**Honesty caveat — not fully out-of-sample.** Unlike the Blast golden pool, all 75
MLC matches were already in `data/t20s_json` and were materialized into the standard
splits. By date:

- **June (21 matches) → validation split** — seen at training/early-stop time. *In-sample.*
- **July (12 matches) → test split** — held out from fitting. *The clean subset.*

Lead with the July/test numbers; treat June as a sanity check, not evidence.

---

## 1. Winner market (match-level model)

`models/eval_mlc_match_level.py` → raw production probabilities, no Platt.

| slice | n | log loss | Brier | accuracy | vs coinflip (LL 0.693) |
|---|---:|---:|---:|---:|---|
| **July / test (held out)** | **12** | **0.679** | 0.243 | **8/12 = 67%** | beats coinflip |
| June / validation (in-sample) | 21 | 0.583 | 0.196 | 18/21 = 86% | inflated by leakage |
| all 33 | 33 | 0.618 | 0.213 | 26/33 = 79% | — |

**Read:** on the clean held-out July slice the model beats a coinflip and goes 8/12 —
the same modest-but-real directional signal we saw on the Blast golden set (64.7%).
Like Blast, it's **low-resolution**: MLC sides are tightly bunched in ELO, so picks
sit near 50% (the misses are coin-flips — 2025-07-01 SFU 57.5% lost, 07-13 MINY won as
a 60% underdog). Nothing here contradicts the 2026 Dallas-leg picks; it says treat the
near-50% calls as weak leans, which the report already does.

## 2. Player performances (v7 ball-level sim — prop backtest)

`prop_backtest.py`, 33 matches × 100 sims → `reports/mlc_2025_prop_report.md`. The
documented v7 pattern **reproduces cleanly on MLC**:

> ⚠️ **Correction (2026-06-08):** the "+0.069 skill / 27% vs 9% base" framing below is
> against a strawman base rate. A fair baseline check (`reports/mlc_baseline_check.md`)
> shows the sim's 27% top-scorer hit rate only **matches** "always back a top-order
> position" (27%), and the batter strike-rate rank-corr (+0.52) equals a pure
> career-stats lookup (+0.51). **The sim does not beat trivial baselines here.** Read the
> skill numbers below as "calibrated vs a flat prior", not "an edge over what a coach
> already knows".

**Where the sim shows Brier skill vs a flat (base-rate) prior — but NOT vs a real baseline:**
- **Top batter per team: +0.069** — #1 pick was the actual top scorer 18/66 = 27% of
  team-innings. *But see the correction above: 27% ≈ the positional baseline.* It did
  surface the stars (Faf ×3, Warner, Pooran, Ravindra, Monank Patel) — which is also
  exactly what a career-stats lookup does.
- Innings total O/U 160.5 (+0.040), team highest-individual O/U 39.5 (+0.097),
  batter-fours ≥1 (+0.011), batter 50+ (+0.027).
- Top bowler ranking weakly positive (+0.010; #1 pick right 10/66 = 15%).

**Where it over-states tails (negative skill → profitable *inverse* plays):**
- Total sixes O/U (−0.58), powerplay totals (−0.33 to −0.46), specific bowler wicket
  counts (−0.10 to −0.15), bowler economy O/U (−0.08).

So the sim is a **ranking engine, not a tail-probability engine** — exactly the
prop-framework conclusion, now confirmed on a league it wasn't tuned on.

## 3. Batter-vs-bowler interactions (teacher-forced ball replay)

`scripts/eval_mlc_matchups.py` replays all **7,736 actual MLC deliveries** through the
ball model and compares predicted vs actual, by player and by duel →
`reports/mlc_2025_matchups.md`.

**Headline calibration (full held-out test, not MLC-specific):** the raw ball model
runs **hot on tails** — predicts wicket 2.3×, six 1.6×, four 1.6× too often, and
under-predicts dots/ones. Absolute economy/wicket numbers are inflated; only relative
ranking is meaningful (the sim's match totals stay sane because extra boundaries and
extra wickets cancel).

**Can it rank who wins the duel?** (Spearman, predicted vs actual)

| level | n | metric | rank corr |
|---|---:|---|---:|
| per batter (≥15 balls) | 73 | strike rate | **+0.52** |
| per bowler (≥18 balls) | 65 | economy | +0.22 |
| per duel (≥9 balls) | 264 | economy | +0.11 (53% directional ≈ chance) |

**Read:** the model captures **batter quality** moderately (it knows who scores fast),
**bowler economy** weakly, and **specific batter-vs-bowler chemistry essentially not at
all** — duel-level is at coin-flip and, with only ~9–12 balls per duel, can't be
validated anyway. This matches the project's standing finding that the matchup cells
shrink hard toward each player's overall rate (lineup/role priors dominate; pairwise
edge is mostly noise). The "called-right duels" in the matchup report (e.g.
Sikandar Raza muzzling MG Bracewell; Onsmptu over-rate vs R Shepherd) are honest
illustrations but cherry-picked from agreement — not a measured edge. (Concrete
flavour: Sikandar Raza predicted + actual to muzzle MG Bracewell, ~7 vs 1.8 rpo;
R Shepherd predicted + actual to take down OC McCoy, ~17 vs 20.7 rpo.)

---

## Verdict

> **Final read after fair-baseline checks (2026-06-08): no validated edge — pre-match
> or in-play.** Every apparent edge collapsed against a sensible baseline:
> - Top-scorer call (27%) = positional baseline (27%); batter-SR corr (+0.52) = career
>   lookup (+0.51). See `reports/mlc_baseline_check.md`.
> - **In-play chase win-prob:** the sim ties a handicapped resource baseline (ΔLL within
>   noise) but **loses to a static pre-match team rating** (sim LL 0.604 / AUC 0.810 vs
>   prior 0.595 / 0.843) and is clearly worse than "chase math + team rating" (blend LL
>   0.494). See `reports/mlc_inplay_winprob.md`. Even where ball-resolution *should* help,
>   it doesn't beat team-rating + arithmetic.
>
> The honest framing: the sim re-derives career stats and team ratings; its tail-hot
> miscalibration then costs it. It's a scenario generator, not a source of edge.

- **Winner picks (incl. MLC 2026 Dallas leg):** modestly better than a coinflip,
  low-resolution. Trust the direction near the extremes; treat 49–56% calls as weak.
- **Player props:** the "rankings" only match career-stat / positional baselines (see
  above); fade the **tails** (sixes, powerplay totals, specific-bowler wickets).
- **Individual matchups:** no validated pairwise edge.

### Caveats
- June MLC is in-sample; the clean evidence is **n=12** (winner) and the held-out ball
  calibration. Underpowered — directional, not a verdict.
- Off-bat runs only (extras excluded) in the matchup replay.
- `venue_encoded=0` at serve (sim convention); real venue signal is in `venue_p*`.

### Artifacts (all under `mlc/`)
- `scripts/extract_mlc.py` → `data/mlc_2025/` (33 matches + `_manifest.json`)
- `scripts/eval_mlc_match_level.py` → `data/mlc_2025_predictions.json`
- `reports/mlc_2025_prop_report.md` + `reports/mlc_2025_prop_per_match/` (per-match drilldowns)
- `scripts/eval_mlc_matchups.py` → `reports/mlc_2025_matchups.md` (+ detail JSON)
- `scripts/sim_decision_demo.py` → `reports/mlc_decision_demo.md` (worked decision delta)
- `scripts/baseline_check.py` → `reports/mlc_baseline_check.md` (**fair-baseline check — the corrective**)
- ~~`EDGE_BRIEF.md`~~ — pitch draft, **superseded** by the baseline check; don't use as-is
