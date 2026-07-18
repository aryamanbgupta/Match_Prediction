"""D2 unit check — strike rotation + balls-faced on extras (`MatchState.update`).

Scripted deliveries through the real `T20Rules.process_ball` path assert the
post-fix semantics:

  1. WIDE / NO_BALL never rotate the strike (their runs=1 is an extra, not
     off the bat).
  2. WIDE / NO_BALL credit neither runs nor a ball faced to the striker
     (pre-fix: NO_BALL incremented balls faced and credited its run).
  3. Odd off-the-bat runs (ONE) still rotate; even runs / boundaries don't.
  4. WICKET replaces the striker with the next batter and counts a ball
     faced for the dismissed batter.
  5. Six legal balls end the over and rotate strike via `end_over`;
     wides/no-balls don't advance the over.
  6. Team score still includes extras (only attribution changes).

Also OBSERVES (does not assert — out of D2 scope) the batting-card
attribution in `_simulate_innings`, which reads `state.striker_idx` AFTER
`update()` has rotated/replaced the striker.

Run: uv run python scripts/auto/d2_unit_check.py
"""
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from sim_v1_2 import (MatchState, Outcome, Player, RandomBowlerSelector,  # noqa
                      T20Rules, TeamLineup)

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


def main():
    rules = T20Rules(bowler_selector=RandomBowlerSelector())
    st = fresh_state()
    team = st.current_team_idx

    print("== extras: no rotation, no balls-faced, team score keeps the run ==")
    runs = rules.process_ball(st, Outcome.WIDE)
    check("wide returns 1 run", runs == 1)
    check("wide does not rotate strike", st.striker_idx == 0 and st.non_striker_idx == 1,
          f"striker={st.striker_idx}")
    check("wide adds no ball to innings count", st.balls == 0)
    check("wide credits striker nothing", st.batsman_stats.get((team, 0), (0, 0)) == (0, 0))
    check("wide adds team run", st.runs[team] == 1)

    runs = rules.process_ball(st, Outcome.NO_BALL)
    check("no-ball returns 1 run", runs == 1)
    check("no-ball does not rotate strike", st.striker_idx == 0 and st.non_striker_idx == 1,
          f"striker={st.striker_idx}")
    check("no-ball adds no ball to innings count", st.balls == 0)
    check("no-ball credits striker nothing (pre-fix: 1 run + 1 ball faced)",
          st.batsman_stats.get((team, 0), (0, 0)) == (0, 0))
    check("no-ball adds team run", st.runs[team] == 2)

    print("== off-the-bat runs: odd rotates, even/boundary doesn't ==")
    rules.process_ball(st, Outcome.ONE)          # ball 1: striker 0 -> rotates
    check("single rotates strike", st.striker_idx == 1 and st.non_striker_idx == 0)
    check("single credited to facing batter", st.batsman_stats.get((team, 0)) == (1, 1))
    rules.process_ball(st, Outcome.TWO)          # ball 2: striker 1, no rotation
    check("two does not rotate", st.striker_idx == 1)
    check("two credited to facing batter", st.batsman_stats.get((team, 1)) == (2, 1))
    rules.process_ball(st, Outcome.FOUR)         # ball 3: striker 1
    rules.process_ball(st, Outcome.SIX)          # ball 4: striker 1
    check("boundaries do not rotate", st.striker_idx == 1)
    check("boundary runs credited", st.batsman_stats.get((team, 1)) == (12, 3))
    rules.process_ball(st, Outcome.DOT)          # ball 5: striker 1
    check("dot counts a ball faced", st.batsman_stats.get((team, 1)) == (12, 4))

    print("== wicket: striker replaced, dismissed batter keeps the ball faced ==")
    pre_striker = st.striker_idx                 # 1
    rules.process_ball(st, Outcome.WICKET)       # ball 6 -> over ends too
    check("dismissed batter charged the ball", st.batsman_stats.get((team, pre_striker)) == (12, 5))
    check("next batter comes in", 2 in (st.striker_idx, st.non_striker_idx),
          f"striker={st.striker_idx}, non_striker={st.non_striker_idx}")
    check("wicket recorded", st.wickets[team] == 1)

    print("== over accounting ==")
    check("six legal balls bowled (extras excluded)", st.balls == 6)
    check("end_over rotated strike (new batter 2 in, then swap)",
          st.striker_idx == 0 and st.non_striker_idx == 2,
          f"striker={st.striker_idx}, non_striker={st.non_striker_idx}")
    check("team score = 2 extras + 13 off the bat", st.runs[team] == 15)
    bat_total = sum(v[0] for k, v in st.batsman_stats.items() if k[0] == team)
    check("batter-credited runs exclude extras", bat_total == 13)

    print("== involution sanity: wide+no-ball mid-over leave over-parity intact ==")
    st2 = fresh_state()
    for oc in [Outcome.DOT, Outcome.WIDE, Outcome.DOT, Outcome.NO_BALL,
               Outcome.DOT, Outcome.DOT, Outcome.DOT]:
        rules.process_ball(st2, oc)
    check("5 legal balls after 7 deliveries", st2.balls == 5)
    check("over not ended early", len(st2.current_over) == 7)

    print("== OBSERVATION (out of D2 scope, for the report) ==")
    st3 = fresh_state()
    pre = st3.striker_idx
    rules.process_ball(st3, Outcome.ONE)
    post = st3.striker_idx
    print(f"  _simulate_innings batting card keys state.striker_idx AFTER update():")
    print(f"  a ONE scored by batter {pre} would be carded to batter {post} "
          f"(internal batsman_stats correctly credits {pre}).")
    print(f"  card-vs-stats attribution mismatch on odd runs: {pre != post}")

    print()
    if FAILURES:
        print(f"UNIT CHECK FAILED: {len(FAILURES)} failure(s): {FAILURES}")
        sys.exit(1)
    print("UNIT CHECK PASSED: all assertions hold.")


if __name__ == "__main__":
    main()
