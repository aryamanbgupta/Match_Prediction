> ⛔ **SUPERSEDED — DO NOT PITCH AS-IS (2026-06-08).** A baseline check
> (`reports/mlc_baseline_check.md`) showed the headline claims below don't hold up:
> the top-scorer call (27%) only matches "back a top-order batter" (27%), and the
> strike-rate correlation (+0.52) is the same as a pure career-stats lookup (+0.51).
> The model is largely re-deriving career averages, not adding an edge. This brief is
> kept for history; see `reports/mlc_2025_eval.md` for the corrected verdict before
> using any of it.

# A ball-by-ball T20 engine for MLC — where the edge is, and where it isn't

**For:** an MLC franchise's cricket / analytics group.
**What we built:** a delivery-by-delivery T20 match simulator — a model that plays
out the match one ball at a time from each side's XI, venue, and match state. It was
trained on global T20 history plus all 75 MLC matches (2023–25), and we **validated it
out-of-sample on MLC 2025** before writing a word of this.

**The one-liner:** we won't sell you a crystal ball that "predicts the score." We sell
you something a planner can actually use — **trustworthy *rankings* of outcomes and
*deltas* between your options** — and we're precise about where the model is sharp and
where it runs hot. That honesty is the point: you can lean on the parts that are
validated and ignore the parts that aren't.

---

## The edge in one picture: a pre-toss danger board

Before a ball is bowled, the engine ranks every opposition batter by their probability
of being the innings' top scorer in *these* conditions against *your* attack. On MLC
2025 (out-of-sample), its **#1-ranked danger man was the actual top scorer 27% of the
time — 3× the ~9% you'd get by chance** — and it put the right names at the top:

> Faf du Plessis (TSK, flagged in 3 separate games), David Warner, Nicholas Pooran,
> Rachin Ravindra, Monank Patel — each correctly tagged as their team's top scorer.

Worked example, **TSK v MI New York, 2025-06-29** (`reports/mlc_decision_demo.md`):
the engine's pre-match read was **Faf du Plessis, P(top) = 0.41 → he top-scored.** That
is a concrete bowling-plan input: who to hold your best overs for, who to attack early,
where to set the ring.

## What it reliably does (all validated OOS on MLC 2025)

| Capability | Evidence | Use |
|---|---|---|
| **Rank opposition top-scorer / danger man** | #1 pick right 27% vs 9% base (Brier skill +0.069) | bowling plans, fields, who to remove early |
| **Rank your batters by impact** | strike-rate rank-corr **+0.52** predicted vs actual | role / order / matchup assignment |
| **Project innings totals & par bands** | positive O/U skill at 160.5; full distribution, not a point | chase planning, "is this a par night" |
| **Call the winner directionally** | held-out **8/12 = 67%**, beats a coin-flip | tilt close calls, not a betting oracle |

## How you'd actually use it: decision deltas, not forecasts

The engine's real product isn't a number on a scoreboard — it's a **ranking of your
choices**. Run option A and option B through the *same* model and compare: any
systematic bias the model has is in *both* runs, so it **cancels in the delta**. That
makes the comparison trustworthy even where the absolute number isn't.

Concretely, for any selection or in-game question we return a *signed, sized,
confidence-bounded* answer. Example (same 06-29 match): *"Promote Marcus Stoinis to
No.3?"* → **+3.0 runs (95% CI [−2.7, +8.5])** — i.e. roughly neutral, don't bother.
That "don't bother" **is** the value: we measured a real MLC finding that **batting-order
reshuffles are only worth a handful of runs**, so a coach can stop agonising over the
order and spend the optimisation budget where the levers are bigger (bowler phasing,
match-ups, chase tempo). The same machinery scores any XI, any order, any deployment.

## Where we're honest about the limits (so you know what to trust)

- **It runs hot on tail events.** Per-ball it over-predicts wickets ~2.3× and boundaries
  ~1.6×. So **don't** use its raw "X% chance of a specific bowler taking 3" — those are
  inflated (and, for a bettor, *fade* candidates). Rankings and deltas are unaffected.
- **It under-reacts to venue.** It projected ~160 at every MLC ground while reality swung
  120 (Lauderhill) to 190 (Dallas). Treat absolute totals as relative, not literal —
  another reason we pitch deltas.
- **No magic individual match-ups.** "This bowler owns that batter" did not validate
  (duel-level rank-corr +0.11 ≈ chance). We will not sell you that story; the signal is
  at the player-quality level, not the duel level.
- **Small samples.** MLC 2025's clean held-out slice is 12 matches — directional, not a
  verdict. More seasons / your internal data tighten everything.

## Why bring us into the room

This already works on your league using only public data. With **you** in the
conversation we get the inputs that move it from "useful" to "edge you can plan around":
your fitness/availability, role and intent data, net/training signals, and target
decisions (toss, phase plans, impact-sub timing). We tune the engine to your squad and
extend the decision-delta tooling to the levers you actually pull.

We're not asking you to trust a black box. We're showing you a validated ranking engine,
telling you exactly where it's sharp and where it's blunt, and offering to point it at
your decisions. Bring us in.

---

*Backing detail: `reports/mlc_2025_eval.md` (full backtest + caveats),
`reports/mlc_2025_prop_report.md` (player-performance skill),
`reports/mlc_2025_matchups.md` (ranking validation), `reports/mlc_decision_demo.md`
(worked decision delta). Reproduce via `README.md`.*
