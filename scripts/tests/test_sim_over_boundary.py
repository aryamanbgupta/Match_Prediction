"""Regression tests for the 2026-08-14 review findings SIM1/SIM2.

SIM1: a WIDE / NO_BALL on the first delivery of an over must not re-trigger
end-of-over bowler selection (`state.balls` only advances on legal
deliveries, so the `balls % 6 == 0` condition still holds after an
over-leading extra — the legacy branch lacked the guard the I5 branch has
always carried, silently swapping bowlers mid-over).

SIM2: XGBoostModelV2.extract_features must produce every trained `basic`
group column — `is_toss_winner` / `is_batting_first` were silently
zero-filled by the feature bridge (the B1 `venue_encoded` bug class).
"""

import sys
from datetime import datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sim_v1_2 import (  # noqa: E402
    BowlerSelector,
    MatchState,
    Outcome,
    Player,
    T20Rules,
    TeamLineup,
)


class _SpySelector(BowlerSelector):
    """Counts selections; deterministic pick."""

    def __init__(self):
        self.calls = 0

    def select_bowler(self, state, available):
        self.calls += 1
        return available[0]


class _FixedModel:
    """Duck-typed PredictionModel returning one outcome with certainty."""

    def __init__(self, outcome_name: str):
        self._probs = {outcome_name: 1.0}

    def extract_features(self, state):
        return {}

    def predict_next_ball(self, features):
        return dict(self._probs)


def _lineup(name: str) -> TeamLineup:
    return TeamLineup(
        name,
        [Player(f"{name}_{i}", f"{name} P{i}", name) for i in range(11)],
    )


def _state_at_over_boundary() -> MatchState:
    state = MatchState(
        team1_lineup=_lineup("Alpha"),
        team2_lineup=_lineup("Beta"),
        batting_first="Alpha",
        venue="Test Ground",
        match_date=datetime(2026, 1, 1),
    )
    # Over 1 just completed: bowler 9 was selected for over 2 and has not
    # yet bowled a legal delivery.
    state.balls = 6
    state.last_bowler_idx = 10
    state.bowler_idx = 9
    return state


def test_wide_leading_an_over_does_not_replace_the_bowler():
    spy = _SpySelector()
    rules = T20Rules(bowler_selector=spy)
    state = _state_at_over_boundary()

    outcome, _ = rules.simulate_ball(state, _FixedModel("wide"))

    assert outcome == Outcome.WIDE
    assert state.balls == 6, "a wide must not advance the legal-ball count"
    assert spy.calls == 0, \
        "over-leading wide re-triggered end-of-over bowler selection"
    assert state.bowler_idx == 9, "bowler was replaced mid-over"


def test_no_ball_leading_an_over_does_not_replace_the_bowler():
    spy = _SpySelector()
    rules = T20Rules(bowler_selector=spy)
    state = _state_at_over_boundary()

    outcome, _ = rules.simulate_ball(state, _FixedModel("no_ball"))

    assert outcome == Outcome.NO_BALL
    assert spy.calls == 0
    assert state.bowler_idx == 9


def test_legal_delivery_completing_an_over_still_selects_new_bowler():
    spy = _SpySelector()
    rules = T20Rules(bowler_selector=spy)
    state = _state_at_over_boundary()
    state.balls = 5  # mid-over: next legal ball completes the over

    outcome, _ = rules.simulate_ball(state, _FixedModel("dot"))

    assert outcome == Outcome.DOT
    assert state.balls == 6
    assert spy.calls == 1, "end-of-over selection must still fire"


_V3_MODEL = ROOT.parent / "models" / "xgb_v3" / "xgboost_model_v3.pkl"


@pytest.mark.skipif(
    not _V3_MODEL.is_file()
    or not (ROOT.parent / "data" / "betting_test").is_dir(),
    reason="legacy v3 ball artifacts / betting_test data not on this checkout",
)
def test_extract_features_covers_toss_columns():
    """Artifact-gated end-to-end check: an innings-1 state must score
    is_batting_first = 1 (it was silently zero-filled pre-fix)."""
    from player_metadata import PlayerMetadataProvider
    from sim_eval.loaders import TestMatchLoader
    from sim_v1_2 import XGBoostModelV2
    from stats_provider import StatsProvider

    model = XGBoostModelV2(
        model_path='models/xgb_v3/xgboost_model_v3.pkl',
        batter_encoder_path='models/xgb_v3/batter_encoder_v3.pkl',
        bowler_encoder_path='models/xgb_v3/bowler_encoder_v3.pkl',
        feature_columns_path='models/xgb_v3/feature_columns_v3.txt',
        matchup_encoder_path='models/xgb_v3/matchup_encoder_v3.pkl',
        stats_provider=StatsProvider('models', version='v3'),
        player_metadata=PlayerMetadataProvider('data/all_players_enriched.csv'),
    )
    matches = TestMatchLoader().load_matches('data/betting_test')
    assert matches, "no test matches loaded"
    state = matches[0][1]
    buf = model.extract_features(state)
    columns = list(model.feature_columns)
    if "is_batting_first" in columns and state.innings == 1:
        assert buf[columns.index("is_batting_first")] == 1.0
