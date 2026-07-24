"""I5 tests for empirical extras fitting and simulator composition."""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from i5_extras import (  # noqa: E402
    EmpiricalExtrasProcess,
    I5_EXTRAS_CONTRACT,
    build_i5_extras_model,
)
from sim_v1_2 import (  # noqa: E402
    MatchState,
    Outcome,
    Player,
    RandomBowlerSelector,
    T20Rules,
    TeamLineup,
)


def _delivery(*, batter=0, extras=None, wicket=False):
    extras = extras or {}
    delivery = {
        "batter": "A0",
        "non_striker": "A1",
        "bowler": "B0",
        "runs": {
            "batter": batter,
            "extras": sum(extras.values()),
            "total": batter + sum(extras.values()),
        },
    }
    if extras:
        delivery["extras"] = extras
    if wicket:
        delivery["wickets"] = [{
            "player_out": "A0",
            "kind": "bowled",
        }]
    return delivery


def _write_match(path: Path, deliveries) -> None:
    match = {
        "info": {
            "dates": ["2025-01-15"],
            "gender": "male",
            "teams": ["A", "B"],
        },
        "innings": [{
            "team": "A",
            "overs": [{"over": 0, "deliveries": deliveries}],
        }],
    }
    path.write_text(json.dumps(match))


def test_extras_builder_uses_validation_only_and_preserves_channels(tmp_path):
    _write_match(tmp_path / "validation.json", [
        _delivery(extras={"wides": 2}),
        _delivery(batter=4, extras={"noballs": 1}),
        _delivery(extras={"byes": 2}),
        _delivery(extras={"legbyes": 1}),
        _delivery(),
        _delivery(batter=1),
        _delivery(wicket=True),
    ])

    model = build_i5_extras_model(tmp_path)

    assert model["contract"] == I5_EXTRAS_CONTRACT
    assert model["n_matches"] == 1
    assert model["n_deliveries"] == 7
    assert model["delivery_event_probabilities"] == {
        "legal": 5 / 7,
        "wide": 1 / 7,
        "no_ball": 1 / 7,
    }
    assert model["wide_team_runs_distribution"] == {"2": 1.0}
    assert model["noball_extras_distribution"] == {"1,0": 1.0}
    assert model["legal_dot_extra_probabilities"] == {
        "none": 1 / 3,
        "byes": 1 / 3,
        "legbyes": 1 / 3,
        "penalty": 0.0,
    }


def _lineup(team):
    return TeamLineup(
        team,
        [Player(f"{team}{idx}", f"{team}{idx}", team) for idx in range(11)],
    )


def _state():
    return MatchState(
        team1_lineup=_lineup("A"),
        team2_lineup=_lineup("B"),
        batting_first="A",
        venue="Test Ground",
        match_date=datetime(2026, 1, 1),
    )


def _extras_model(*, event, wide_runs=1, noball_runs=1,
                  dot_extra="none", dot_extra_runs=0):
    event_probs = {"legal": 0.0, "wide": 0.0, "no_ball": 0.0}
    event_probs[event] = 1.0
    dot_probs = {
        "none": 0.0, "byes": 0.0, "legbyes": 0.0, "penalty": 0.0,
    }
    dot_probs[dot_extra] = 1.0
    return {
        "contract": I5_EXTRAS_CONTRACT,
        "delivery_event_probabilities": event_probs,
        "wide_team_runs_distribution": {str(wide_runs): 1.0},
        "noball_extras_distribution": {f"{noball_runs},0": 1.0},
        "legal_dot_extra_probabilities": dot_probs,
        "legal_dot_extra_runs_distributions": {
            key: {str(dot_extra_runs if key == dot_extra else 0): 1.0}
            for key in dot_probs
        },
    }


class _Model:
    delivery_semantics = "legal_off_bat_v1"

    def __init__(self, probs, extras_model):
        self.probs = probs
        self.extras_process = EmpiricalExtrasProcess(extras_model)

    def extract_features(self, state):
        return None

    def predict_next_ball(self, features):
        return self.probs


def _rules():
    return T20Rules(bowler_selector=RandomBowlerSelector())


def test_no_ball_composes_batter_runs_without_consuming_ball():
    state = _state()
    model = _Model(
        {"dot": 0, "one": 0, "two": 0, "four": 1, "six": 0, "wicket": 0},
        _extras_model(event="no_ball", noball_runs=1),
    )

    outcome, runs = _rules().simulate_ball(state, model)

    assert outcome == Outcome.NO_BALL
    assert runs == 5
    assert state.balls == 0
    assert state.runs[0] == 5
    assert state.batsman_stats[(0, 0)] == (4, 0)
    assert state.last_batter_runs == 4
    assert state.last_bowler_runs == 5
    assert state.last_is_legal is False


def test_multi_run_wide_rotates_only_completed_runs():
    state = _state()
    model = _Model(
        {"dot": 1, "one": 0, "two": 0, "four": 0, "six": 0, "wicket": 0},
        _extras_model(event="wide", wide_runs=2),
    )

    outcome, runs = _rules().simulate_ball(state, model)

    assert outcome == Outcome.WIDE
    assert runs == 2
    assert state.balls == 0
    assert state.striker_idx == 1
    assert state.non_striker_idx == 0


def test_legal_dot_can_resolve_to_byes_without_batter_credit():
    state = _state()
    model = _Model(
        {"dot": 1, "one": 0, "two": 0, "four": 0, "six": 0, "wicket": 0},
        _extras_model(event="legal", dot_extra="byes", dot_extra_runs=3),
    )

    outcome, runs = _rules().simulate_ball(state, model)

    assert outcome == Outcome.BYE
    assert runs == 3
    assert state.balls == 1
    assert state.runs[0] == 3
    assert state.batsman_stats[(0, 0)] == (0, 1)
    assert state.last_batter_runs == 0
    assert state.last_bowler_runs == 0
    assert state.last_is_legal is True
    assert state.striker_idx == 1
    assert state.non_striker_idx == 0
