"""Pins the ball-119 rule removal (BR2 engine change, uncommitted).

The original rule — `is_legal_outcome` returning False for WIDE/NO_BALL at
balls == 119 — dates from the first "working simulation" commit and was
commented "(simplified)". It force-converted a sampled final-ball extra
into a DOT, which (a) deleted real runs including chase-winning wides in
tied finishes and (b) inflated final-ball dot probability by the extras
mass. It provided NO termination guarantee: extras never advance `balls`
at ANY position, so the re-bowl loop was always probabilistically bounded
on balls 0-118 with or without the rule, and the history buffer
auto-extends when full.

Committed together with the sim_v1_2.py engine change: it pins the
post-removal behavior and fails under the old rule. Lives in tests/ so the
embeddings-contracts CI workflow collects it.
"""

import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from sim_v1_2 import (  # noqa: E402
    BowlerSelector,
    MatchState,
    Outcome,
    Player,
    T20Rules,
    TeamLineup,
)


class _FirstAvailableSelector(BowlerSelector):
    def select_bowler(self, state, available):
        return available[0]


class _ScriptedModel:
    """Emits a scripted outcome sequence with certainty."""

    def __init__(self, outcomes):
        self._outcomes = list(outcomes)

    def extract_features(self, state):
        return {}

    def predict_next_ball(self, features):
        name = self._outcomes.pop(0)
        return {name: 1.0}


def _lineup(name):
    return TeamLineup(
        name, [Player(f"{name}_{i}", f"{name} P{i}", name) for i in range(11)]
    )


def _state_at_final_ball(innings=1, chasing_needs=None):
    state = MatchState(
        team1_lineup=_lineup("Alpha"),
        team2_lineup=_lineup("Beta"),
        batting_first="Alpha",
        venue="Test Ground",
        match_date=datetime(2026, 1, 1),
    )
    state.balls = 119
    state.last_bowler_idx = 10
    state.bowler_idx = 9
    if innings == 2:
        state.innings = 2
        # Alpha batted first; Beta is chasing. target = runs[0] + 1.
        state.runs[0] = 100
        state.runs[1] = 100 + 1 - chasing_needs  # needs `chasing_needs` to win
    return state


def test_final_ball_wide_is_rebowled_and_its_run_counts():
    rules = T20Rules(bowler_selector=_FirstAvailableSelector())
    state = _state_at_final_ball()
    model = _ScriptedModel(["wide", "dot"])

    outcome, runs = rules.simulate_ball(state, model)
    assert outcome == Outcome.WIDE, \
        "a final-ball wide must stay a wide (the old rule forced it to DOT)"
    assert state.balls == 119, "a wide never advances the legal-ball count"
    assert not state.is_innings_over(), \
        "the innings cannot end on a wide — the delivery is re-bowled"
    assert int(state.runs[0]) == 1, "the wide's run must count"

    outcome, _ = rules.simulate_ball(state, model)
    assert outcome == Outcome.DOT
    assert state.balls == 120
    assert state.is_innings_over(), \
        "the re-bowled legal delivery completes the 120-ball innings"


def test_chase_won_by_final_ball_wide_terminates_the_match():
    """Scores level, one ball left: a wide wins the chase. Under the old
    rule this exact outcome was force-converted to a DOT and the winning
    run deleted."""
    rules = T20Rules(bowler_selector=_FirstAvailableSelector())
    state = _state_at_final_ball(innings=2, chasing_needs=1)
    model = _ScriptedModel(["wide"])

    outcome, _ = rules.simulate_ball(state, model)
    assert outcome == Outcome.WIDE
    assert int(state.runs[1]) == state.target == 101
    assert state.is_innings_over(), \
        "target reached via the wide must end the innings"
    assert state.is_match_over()


def test_final_ball_wicket_still_legal_and_ends_innings():
    """Regression guard around the removal: the surviving is_legal_outcome
    rule (no wicket at 10 down) and normal final-ball termination are
    untouched."""
    rules = T20Rules(bowler_selector=_FirstAvailableSelector())
    state = _state_at_final_ball()
    model = _ScriptedModel(["wicket"])
    outcome, _ = rules.simulate_ball(state, model)
    assert outcome == Outcome.WICKET
    assert state.balls == 120
    assert state.is_innings_over()
