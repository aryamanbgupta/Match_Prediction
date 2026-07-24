"""Regression tests for the frozen-gated ball-v7 forward scorer."""
from __future__ import annotations

import builtins
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from forward_eval_contract import load_protocol  # noqa: E402
from score_forward_ball_v7 import (  # noqa: E402
    LINEUP_CONTRACT,
    build_match_state_from_info,
    build_prediction_artifact,
    load_context_batches,
    postprocess_winner_probabilities,
    pre_match_spec,
    selected_rows_by_cricsheet,
    validate_candidate_contract,
    validate_selected_identity,
    walk_context_and_score,
)
from score_forward_match_m7 import write_locked_artifact  # noqa: E402


PROTOCOL_PATH = (
    ROOT / "evaluation" / "forward_protocol_2026-06-01_2026-07-13.yaml"
)


def _match(
    match_id="001",
    date="2026-01-01",
    teams=("A", "B"),
    venue="Ground",
):
    names = {
        teams[0]: [f"{teams[0]} Player {index}" for index in range(11)],
        teams[1]: [f"{teams[1]} Player {index}" for index in range(11)],
    }
    registry = {
        name: f"{team.lower()}_{index}"
        for team, roster in names.items()
        for index, name in enumerate(roster)
    }
    return {
        "_test_id": match_id,
        "info": {
            "dates": [date],
            "teams": list(teams),
            "venue": venue,
            "team_type": "international",
            "event": {"name": "Synthetic"},
            "toss": {"winner": teams[0], "decision": "field"},
            "registry": {"people": registry},
            "players": names,
        },
        "innings": [{"forbidden": "outcome-bearing"}],
    }


def _selected(match_id="002", date="2026-01-01"):
    return {
        "match_id": f"{date}_A_B_Ground",
        "cricsheet_id": match_id,
        "date": date,
        "teams": ["A", "B"],
        "venue": "Ground",
    }


class _TraceStats:
    def __init__(self, trace):
        self.trace = trace

    def begin_date(self, date, documents):
        self.trace.append(f"begin_date:{date}:{len(documents)}")

    def begin_match(self, match_id, data, prediction_required):
        self.trace.append(f"begin:{match_id}:{prediction_required}")

    def lock_prediction(self, match_id):
        self.trace.append(f"lock:{match_id}")

    def advance_match(self, match_id, data):
        self.trace.append(f"advance:{match_id}")


def test_context_walk_scores_before_lock_and_replays_after_lock():
    trace = []
    batches = [
        (
            "2026-01-01",
            [("001", _match("001")), ("002", _match("002"))],
        )
    ]

    def build_state(data):
        trace.append(f"state:{data['_test_id']}")
        return data["_test_id"]

    def simulate(state):
        trace.append(f"simulate:{state}")
        return {
            "p_team1": 0.6,
            "p_team2": 0.4,
            "p_team1_raw": 0.6,
            "p_team2_raw": 0.4,
            "p_tie_raw": 0.0,
            "n_simulations": 100,
        }

    predictions, report = walk_context_and_score(
        batches,
        {"002": _selected()},
        _TraceStats(trace),
        build_state=build_state,
        simulate=simulate,
    )
    assert trace == [
        "begin_date:2026-01-01:2",
        "begin:001:False",
        "advance:001",
        "begin:002:True",
        "state:002",
        "simulate:002",
        "lock:002",
        "advance:002",
    ]
    assert predictions[0]["cricsheet_id"] == "002"
    assert report == {
        "context_matches_replayed": 2,
        "selected_matches_scored": 1,
    }


def test_pre_match_state_uses_roster_and_never_reads_innings():
    match = _match()

    class _ForbiddenInnings:
        def __iter__(self):
            raise AssertionError("innings were read")

        def __getitem__(self, key):
            raise AssertionError("innings were read")

    match["innings"] = _ForbiddenInnings()

    @dataclass
    class Player:
        player_id: str
        name: str
        team: str

    @dataclass
    class Lineup:
        team_name: str
        players: list

    class State:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    state = build_match_state_from_info(
        match,
        player_class=Player,
        lineup_class=Lineup,
        state_class=State,
        classify_context=lambda event, team_type, teams: {
            "match_importance": 3,
            "is_international": 1,
            "competition_tier": 2,
        },
    )
    assert state.team1_lineup.players[0].name == "A Player 0"
    assert state.team1_lineup.players[-1].name == "A Player 10"
    assert state.batting_first == "B"
    assert state.chose_to_bat == 0
    assert LINEUP_CONTRACT == "info_players_roster_order_only_v1"


def test_team_orientation_mismatch_fails_closed():
    match = _match(teams=("B", "A"))
    with pytest.raises(RuntimeError, match="team order mismatch"):
        validate_selected_identity(_selected(), "002", match)


def test_roster_integrity_fails_closed():
    match = _match()
    match["info"]["players"]["A"] = match["info"]["players"]["A"][:10]
    with pytest.raises(ValueError, match="at least 11"):
        pre_match_spec(match)


def test_winner_probability_postprocess_matches_landed_evaluator():
    result = postprocess_winner_probabilities(0.98, 0.01, 0.01)
    assert result["p_team1"] == pytest.approx(0.95)
    assert result["p_team2"] == pytest.approx(0.05)
    all_ties = postprocess_winner_probabilities(0.0, 0.0, 1.0)
    assert all_ties["p_team1"] == all_ties["p_team2"] == 0.5
    with pytest.raises(RuntimeError, match="sum to one"):
        postprocess_winner_probabilities(0.4, 0.4, 0.1)


def test_actual_context_is_complete_and_version_ordered():
    protocol = load_protocol(PROTOCOL_PATH)
    batches = load_context_batches(protocol)
    selected = selected_rows_by_cricsheet(protocol)
    flattened = [
        (date, match_id)
        for date, batch in batches
        for match_id, _ in batch
    ]
    assert len(flattened) == 401
    assert len(selected) == 137
    assert flattened == sorted(flattened)
    assert set(selected).issubset({match_id for _, match_id in flattened})
    for _date, batch in batches:
        for match_id, match_data in batch:
            if match_id in selected:
                validate_selected_identity(
                    selected[match_id],
                    match_id,
                    match_data,
                )


def test_protocol_recipe_must_match_implemented_ball_path():
    protocol = load_protocol(PROTOCOL_PATH)
    candidate = dict(protocol["candidates"]["ball_v7"])
    validate_candidate_contract(candidate)
    candidate["parallel_simulation"] = True
    with pytest.raises(RuntimeError, match="recipe is unsupported"):
        validate_candidate_contract(candidate)


def test_prediction_artifact_is_outcome_free_and_write_once(tmp_path):
    protocol = load_protocol(PROTOCOL_PATH)
    prediction = {
        "match_id": "2026-01-01_A_B_Ground",
        "cricsheet_id": "002",
        "date": "2026-01-01",
        "team1": "A",
        "team2": "B",
        "p_team1": 0.6,
        "p_team2": 0.4,
        "p_team1_raw": 0.6,
        "p_team2_raw": 0.4,
        "p_tie_raw": 0.0,
        "n_simulations": 100,
    }
    gate = {
        "protocol_sha256": "p",
        "holdout_fingerprint_sha256": "h",
        "state_fingerprint_sha256": "s",
    }
    artifact = build_prediction_artifact(
        protocol,
        gate,
        [prediction],
        {
            "context_matches_replayed": 2,
            "selected_matches_scored": 1,
        },
    )
    payload = json.dumps(artifact)
    assert '"actual_winner"' not in payload
    assert '"winner"' not in payload
    path = tmp_path / "predictions.json"
    write_locked_artifact(path, artifact)
    with pytest.raises(FileExistsError):
        write_locked_artifact(path, artifact)


def test_draft_gate_precedes_model_and_simulation_imports(tmp_path):
    output = tmp_path / "must_not_exist.json"
    code = f"""
import builtins
import sys
from pathlib import Path
sys.path.insert(0, {str(ROOT / 'scripts')!r})
blocked = {{
    'joblib', 'sim_v1_2', 'sim_eval.same_day_stats',
    'player_metadata', 'stats_provider'
}}
real_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name in blocked:
        raise AssertionError('blocked import occurred before frozen gate: ' + name)
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
from score_forward_ball_v7 import score
try:
    score(Path({str(PROTOCOL_PATH)!r}), Path({str(output)!r}))
except RuntimeError as exc:
    if 'model scoring is blocked' not in str(exc):
        raise
else:
    raise AssertionError('DRAFT protocol unexpectedly scored')
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert not output.exists()
    assert not output.with_suffix(".json.sha256").exists()
