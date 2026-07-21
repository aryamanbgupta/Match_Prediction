"""D14 unit check — batting/bowling-card attribution in `_simulate_innings`
plus the re-applied D2 extras semantics, verified as one unit.

Part 1 re-runs the D2 scripted-delivery check unchanged (26 assertions on
`MatchState.update` via the real `T20Rules.process_ball` path).

Part 2 drives the REAL `SimulationEngine._simulate_innings` with a
deterministic scripted rules subclass (no RNG, no model) and asserts the
D14 attribution contract:

  1. Card-vs-stats equality — for every batter, the innings batting card's
     (runs, balls faced) must equal the internal `batsman_stats` entry
     (which `update()` credits BEFORE rotation/replacement and is therefore
     correct). Pre-fix, a ONE was carded to the post-rotation non-striker
     and a WICKET carded its ball-faced to the INCOMING batter.
  2. Over-final-ball bowler attribution — `simulate_ball` reassigns
     `state.bowler_idx` to the NEXT over's bowler before the card read, so
     pre-fix the 6th legal delivery of every over (runs + any wicket) was
     credited to the next bowler. Post-fix each over's 6 balls sit on its
     own bowler's card.
  3. Over/ball labels — `state.balls` is incremented before BallResult was
     built, so pre-fix the 6th legal delivery of over k was labeled over
     k+1 (team_first_over/pp/highest_over extraction reads `b.over`).
     Post-fix the first-over sum includes all 6 legal deliveries + extras.
  4. BallResult.striker_idx is the batter who FACED the delivery.
  5. Conservation — bowling-card balls sum to the innings legal-ball count,
     wickets sum to 10, and card runs exclude extras exactly.

Run: uv run python scripts/auto/d14_unit_check.py
"""
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from sim_v1_2 import (MatchState, Outcome, Player, RandomBowlerSelector,  # noqa
                      SimulationEngine, T20Rules, TeamLineup)

FAILURES = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILURES.append(name)


def fresh_state():
    def lineup(team):
        return TeamLineup(team, [Player(f"{team}_p{i}", f"{team}_p{i}", team)
                                 for i in range(11)])
    return MatchState(team1_lineup=lineup("A"), team2_lineup=lineup("B"),
                      batting_first="A", venue="Test Ground",
                      match_date=datetime(2026, 1, 1))


class ScriptedRules(T20Rules):
    """Feed a fixed outcome sequence through the real process_ball/update
    path; once the script is exhausted, drain WICKETs so the innings ends
    deterministically. Bowler selection is a deterministic round-robin
    (over k -> bowler index k), mirroring simulate_ball's real post-over
    reassignment timing exactly."""

    def __init__(self, script):
        super().__init__(bowler_selector=RandomBowlerSelector())
        self.script = list(script)
        self.i = 0

    def select_next_bowler(self, state):
        return (state.balls // 6) % 11

    def simulate_ball(self, state, model):
        outcome = (self.script[self.i] if self.i < len(self.script)
                   else Outcome.WICKET)
        self.i += 1
        runs = self.process_ball(state, outcome)
        # Same post-over bowler reassignment as T20Rules.simulate_ball
        if state.balls % 6 == 0 and state.balls > 0 and not state.is_innings_over():
            state.bowler_idx = self.select_next_bowler(state)
        return outcome, runs


def main():
    print("== Part 1: D2 scripted-delivery semantics (unchanged) ==")
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts/auto/d2_unit_check.py")],
        capture_output=True, text=True)
    d2_ok = r.returncode == 0
    tail = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "(no output)"
    check("d2_unit_check.py passes", d2_ok, tail)

    print("\n== Part 2: _simulate_innings attribution on a scripted innings ==")
    # Over 0 (bowler 0): ONE(b0, rotate) WIDE SIX(b1) WICKET(b1 out, b2 in)
    #                    DOT(b2) NO_BALL TWO(b2) FOUR(b2, over ends)
    # Over 1 (bowler 1): ONE(b0, rotate) WICKET(b2 out, b3 in) DOT DOT
    # then WICKET drain (bowler 1 finishes over 1, bowler 2 bowls over 2)
    script = [Outcome.ONE, Outcome.WIDE, Outcome.SIX, Outcome.WICKET,
              Outcome.DOT, Outcome.NO_BALL, Outcome.TWO, Outcome.FOUR,
              Outcome.ONE, Outcome.WICKET, Outcome.DOT, Outcome.DOT]
    state = fresh_state()
    team = state.current_team_idx
    engine = SimulationEngine(model=None, rules=ScriptedRules(script))
    inn = engine._simulate_innings(state)

    check("innings ended all out", inn.total_wickets == 10,
          f"wickets={inn.total_wickets}")
    check("18 legal balls (3 overs)", inn.total_balls == 18,
          f"balls={inn.total_balls}")

    print("-- card-vs-stats equality (runs, balls faced) per batter --")
    keys = sorted(set(inn.batting_card) |
                  {idx for (t, idx) in state.batsman_stats if t == team})
    for idx in keys:
        card = inn.batting_card.get(idx, (0, 0, 0, 0))[:2]
        stats = state.batsman_stats.get((team, idx), (0, 0))
        check(f"batter {idx}: card {card} == stats {stats}", card == stats)

    print("-- explicit attribution cases --")
    check("ONE carded to the batter who faced it (b0), not the "
          "post-rotation striker", inn.batting_card.get(0, (0,) * 4)[:2] == (2, 3))
    check("WICKET ball-faced charged to the dismissed batter (b1), "
          "not the incoming one", inn.batting_card.get(1, (0,) * 4)[:2] == (6, 2))
    check("over-final FOUR carded to the striker who hit it (b2)",
          inn.batting_card.get(2, (0,) * 4) == (6, 4, 1, 0))
    check("non-striker never facing has no card entry", 5 not in inn.batting_card)

    print("-- bowler attribution --")
    check("over-0 bowler owns all 6 over-0 balls incl. the final FOUR",
          inn.bowling_card.get(0) == (6, 13, 1),
          f"card={inn.bowling_card.get(0)}")
    check("over-1 bowler owns its over-final drain WICKET",
          inn.bowling_card.get(1) == (6, 1, 3),
          f"card={inn.bowling_card.get(1)}")
    check("over-2 bowler credited 6 drain wickets",
          inn.bowling_card.get(2) == (6, 0, 6),
          f"card={inn.bowling_card.get(2)}")
    check("bowling-card balls sum to legal-ball count",
          sum(v[0] for v in inn.bowling_card.values()) == 18)
    check("bowling-card wickets sum to 10",
          sum(v[2] for v in inn.bowling_card.values()) == 10)

    print("-- over/ball labels --")
    over0 = [b for b in inn.balls if b.over == 0]
    check("over 0 has 6 legal + 2 extras = 8 deliveries", len(over0) == 8,
          f"n={len(over0)}")
    check("first-over runs (b.over==0) include the over-final boundary "
          "(15 = 13 off bat + 2 extras)", sum(b.runs for b in over0) == 15,
          f"sum={sum(b.runs for b in over0)}")
    check("6th legal delivery labeled over 0 ball 5",
          (over0[-1].over, over0[-1].ball) == (0, 5),
          f"got=({over0[-1].over},{over0[-1].ball})")
    check("first delivery labeled over 0 ball 0",
          (inn.balls[0].over, inn.balls[0].ball) == (0, 0))
    check("BallResult.striker_idx on ball 1 is the facing batter 0",
          inn.balls[0].striker_idx == 0)
    check("BallResult.bowler_idx on the over-final ball is the over-0 "
          "bowler", over0[-1].bowler_idx == 0)
    check("max over label is 2 (18 legal balls)",
          max(b.over for b in inn.balls) == 2)

    print("-- conservation --")
    card_runs = sum(v[0] for v in inn.batting_card.values())
    check("card runs == team total minus the 2 extras",
          card_runs == inn.total_runs - 2,
          f"card={card_runs}, total={inn.total_runs}")

    print()
    if FAILURES:
        print(f"UNIT CHECK FAILED: {len(FAILURES)} failure(s): {FAILURES}")
        sys.exit(1)
    print("UNIT CHECK PASSED: all assertions hold.")


if __name__ == "__main__":
    main()
