"""D15 unit check — D2 extras semantics + D14 attribution snapshot + D4
run-out dismissal channel, verified as one unit.

Part 1 re-runs the D14 extended check unchanged (which itself re-runs D2's
26 scripted-delivery assertions). Both drive `process_ball` WITHOUT a
dismissal argument, so they exercise the legacy-deterministic path — they
must still pass byte-for-byte under D15.

Part 2 drives the REAL `SimulationEngine._simulate_innings` with scripted
(outcome, dismissal) pairs and asserts the run-out contract:

  1. A 'runout_nonstriker' WICKET dismisses the NON-striker (batsmen_out
     gets the non-striker; the incoming batsman takes the non-striker end;
     the striker keeps the strike) while the striker — who faced the
     delivery — is charged the ball on both card and internal stats.
  2. 'runout_striker' dismisses the striker like a bowler wicket, but
     neither run-out type credits the bowling card. Team wickets count ALL
     dismissals (total wicket rate unchanged): card wickets sum to
     team wickets minus run-outs — the exact actuals convention
     (prop_backtest counts bowler wickets only for kind != "run out").
  3. Card-vs-stats equality per batter still holds (D14 contract intact).
  4. Innings termination: an all-run-out innings ends 10-down with ZERO
     card wickets; a non-striker run-out as the 10th wicket ends the
     innings.

Part 3 seeds the RNG and pushes 20,000 wicket balls through the REAL
`T20Rules.simulate_ball` draw: empirical p_runout / nonstriker_share must
match the module constants within 3-sigma binomial tolerance (proves the
draw is wired in the live path, not just in the scripted subclass).

Run: uv run python scripts/auto/d15_unit_check.py
"""
import random
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from sim_v1_2 import (MatchState, Outcome, Player, RandomBowlerSelector,  # noqa
                      RUNOUT_NONSTRIKER_SHARE, RUNOUT_P, SimulationEngine,
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


class ScriptedRules(T20Rules):
    """Feed fixed (outcome, dismissal) pairs through the real
    process_ball/update path; once exhausted, drain with the given
    dismissal type so the innings ends deterministically. Bowler selection
    mirrors simulate_ball's real post-over reassignment timing."""

    def __init__(self, script, drain=('bowler',)):
        super().__init__(bowler_selector=RandomBowlerSelector())
        self.script = list(script)
        self.drain = list(drain)
        self.i = 0

    def select_next_bowler(self, state):
        return (state.balls // 6) % 11

    def simulate_ball(self, state, model):
        if self.i < len(self.script):
            outcome, dismissal = self.script[self.i]
        else:
            outcome = Outcome.WICKET
            dismissal = self.drain[(self.i - len(self.script)) % len(self.drain)]
        self.i += 1
        if outcome == Outcome.WICKET and dismissal is None:
            dismissal = 'bowler'
        runs = self.process_ball(state, outcome, dismissal=dismissal)
        if state.balls % 6 == 0 and state.balls > 0 and not state.is_innings_over():
            state.bowler_idx = self.select_next_bowler(state)
        return outcome, runs


class AlwaysWicketModel:
    """Stub model: every ball is a wicket (for the live-draw check)."""

    def extract_features(self, state):
        return None

    def predict_next_ball(self, features):
        return {'wicket': 1.0}


def main():
    print("== Part 1: D14 extended check (incl. D2's 26 assertions), "
          "legacy path unchanged ==")
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts/auto/d14_unit_check.py")],
        capture_output=True, text=True)
    d14_ok = r.returncode == 0
    tail = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "(no output)"
    check("d14_unit_check.py passes", d14_ok, tail)

    print("\n== Part 2a: scripted innings with run-out dismissals ==")
    # Over 0 (bowler 0):
    #   ONE(b0, rotate -> striker b1)
    #   WICKET runout_nonstriker (b1 faces+charged; NON-striker b0 out, b2 in
    #                             at the non-striker end; striker stays b1)
    #   WICKET bowler            (b1 out, b3 in; bowler credited)
    #   WICKET runout_striker    (b3 faces+charged+out, b4 in; NO credit)
    #   DOT(b4)  FOUR(b4, over ends -> rotate: striker b2)
    # Over 1 (bowler 1): 6 drain bowler-WICKETs (b2,b5,b6,b7,b8,b9 out)
    # Over 2 (bowler 2): 1 drain bowler-WICKET (b4 out -> all out)
    script = [(Outcome.ONE, None),
              (Outcome.WICKET, 'runout_nonstriker'),
              (Outcome.WICKET, 'bowler'),
              (Outcome.WICKET, 'runout_striker'),
              (Outcome.DOT, None),
              (Outcome.FOUR, None)]
    state = fresh_state()
    team = state.current_team_idx
    engine = SimulationEngine(model=None, rules=ScriptedRules(script))
    inn = engine._simulate_innings(state)

    check("innings ended all out (team wickets count run-outs)",
          inn.total_wickets == 10, f"wickets={inn.total_wickets}")
    check("13 legal balls", inn.total_balls == 13, f"balls={inn.total_balls}")
    check("non-striker (b0) is in batsmen_out after the run-out",
          0 in state.batsmen_out[team], f"out={state.batsmen_out[team]}")
    check("run-out of the non-striker recorded b0 out on ball 2 "
          "(order: b0 first)", state.batsmen_out[team][0] == 0,
          f"order={state.batsmen_out[team]}")
    check("striker b1 charged the run-out delivery (card runs,balls == (0,2))",
          inn.batting_card.get(1, (0,) * 4)[:2] == (0, 2),
          f"card={inn.batting_card.get(1)}")
    check("dismissed non-striker b0 keeps only his own faced ball (1,1)",
          inn.batting_card.get(0, (0,) * 4)[:2] == (1, 1),
          f"card={inn.batting_card.get(0)}")
    check("b2 (came in at non-striker end) faced no over-0 ball; first "
          "faced ball is the over-1 drain wicket (0,1)",
          inn.batting_card.get(2, (0,) * 4)[:2] == (0, 1),
          f"card={inn.batting_card.get(2)}")

    print("-- card-vs-stats equality (D14 contract intact) --")
    keys = sorted(set(inn.batting_card) |
                  {idx for (t, idx) in state.batsman_stats if t == team})
    for idx in keys:
        card = inn.batting_card.get(idx, (0, 0, 0, 0))[:2]
        stats = state.batsman_stats.get((team, idx), (0, 0))
        check(f"batter {idx}: card {card} == stats {stats}", card == stats)

    print("-- bowling-card credit --")
    check("bowler 0: 6 balls, 5 runs, ONLY the bowler-type wicket credited",
          inn.bowling_card.get(0) == (6, 5, 1),
          f"card={inn.bowling_card.get(0)}")
    check("bowler 1: 6 drain wickets credited", inn.bowling_card.get(1) == (6, 0, 6),
          f"card={inn.bowling_card.get(1)}")
    check("bowler 2: final wicket credited", inn.bowling_card.get(2) == (1, 0, 1),
          f"card={inn.bowling_card.get(2)}")
    card_wkts = sum(v[2] for v in inn.bowling_card.values())
    check("card wickets == team wickets minus the 2 run-outs (8 == 10-2)",
          card_wkts == 8, f"card_wkts={card_wkts}")
    check("bowling-card balls sum to legal-ball count",
          sum(v[0] for v in inn.bowling_card.values()) == 13)

    print("\n== Part 2b: termination edge cases ==")
    st2 = fresh_state()
    inn2 = SimulationEngine(
        model=None,
        rules=ScriptedRules([], drain=('runout_striker',)))._simulate_innings(st2)
    check("all-run-out innings ends 10 down", inn2.total_wickets == 10,
          f"wickets={inn2.total_wickets}")
    check("all-run-out innings: ZERO bowling-card wickets",
          sum(v[2] for v in inn2.bowling_card.values()) == 0,
          f"sum={sum(v[2] for v in inn2.bowling_card.values())}")

    st3 = fresh_state()
    script3 = [(Outcome.WICKET, 'bowler')] * 9 + \
              [(Outcome.WICKET, 'runout_nonstriker')]
    inn3 = SimulationEngine(
        model=None, rules=ScriptedRules(script3))._simulate_innings(st3)
    check("non-striker run-out as 10th wicket ends the innings",
          inn3.total_wickets == 10 and inn3.total_balls == 10,
          f"wickets={inn3.total_wickets}, balls={inn3.total_balls}")
    check("9 of 10 wickets bowler-credited in that innings",
          sum(v[2] for v in inn3.bowling_card.values()) == 9)

    print("\n== Part 3: live-path draw frequencies (seeded) ==")
    random.seed(4242)
    rules = T20Rules(bowler_selector=RandomBowlerSelector())
    model = AlwaysWicketModel()
    counts = Counter()
    N = 20000
    for _ in range(N):
        st = fresh_state()
        rules.simulate_ball(st, model)
        counts[st.last_dismissal] += 1
    n_ro = counts['runout_striker'] + counts['runout_nonstriker']
    p_hat = n_ro / N
    share_hat = counts['runout_nonstriker'] / max(1, n_ro)
    # 3-sigma binomial tolerances at N=20000 / n_ro~1500
    tol_p = 3 * (RUNOUT_P * (1 - RUNOUT_P) / N) ** 0.5
    tol_s = 3 * (RUNOUT_NONSTRIKER_SHARE * (1 - RUNOUT_NONSTRIKER_SHARE)
                 / max(1, n_ro)) ** 0.5
    check(f"live draw p_runout {p_hat:.4f} ~= {RUNOUT_P:.4f} (3s={tol_p:.4f})",
          abs(p_hat - RUNOUT_P) < tol_p, f"counts={dict(counts)}")
    check(f"live draw nonstriker_share {share_hat:.4f} ~= "
          f"{RUNOUT_NONSTRIKER_SHARE:.4f} (3s={tol_s:.4f})",
          abs(share_hat - RUNOUT_NONSTRIKER_SHARE) < tol_s)
    check("non-wicket balls leave last_dismissal None (spot)",
          fresh_state().last_dismissal is None)

    print()
    if FAILURES:
        print(f"UNIT CHECK FAILED: {len(FAILURES)} failure(s): {FAILURES}")
        sys.exit(1)
    print("UNIT CHECK PASSED: all assertions hold.")


if __name__ == "__main__":
    main()
