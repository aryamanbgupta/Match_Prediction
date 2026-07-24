"""I5 regression tests for legal-ball, batter-run, and extras semantics."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from parsing_v2 import (  # noqa: E402
    I5_DELIVERY_SEMANTICS,
    PlayerStatsTracker,
    extract_delivery_semantics,
    parse_match_data_v2,
)
from build_stats_cache import _has_nonzero_match_stats  # noqa: E402


REGISTRY = {
    "Batter": "bat",
    "Non Striker": "non",
    "Bowler": "bowl",
}


def _delivery(*, batter_runs=0, extras=None, wickets=None, non_boundary=False):
    extras = extras or {}
    total = batter_runs + sum(extras.values())
    delivery = {
        "batter": "Batter",
        "non_striker": "Non Striker",
        "bowler": "Bowler",
        "runs": {
            "batter": batter_runs,
            "extras": sum(extras.values()),
            "total": total,
        },
    }
    if extras:
        delivery["extras"] = extras
    if wickets:
        delivery["wickets"] = wickets
    if non_boundary:
        delivery["runs"]["non_boundary"] = True
    return delivery


def _match(deliveries):
    return {
        "info": {
            "dates": ["2026-01-01"],
            "venue": "Test Ground",
            "teams": ["Team A", "Team B"],
            "team_type": "international",
            "registry": {"people": REGISTRY},
            "players": {
                "Team A": ["Batter", "Non Striker"],
                "Team B": ["Bowler"],
            },
            "toss": {"winner": "Team A", "decision": "bat"},
            "event": {"name": "Test Series"},
        },
        "innings": [{
            "team": "Team A",
            "overs": [{"over": 0, "deliveries": deliveries}],
        }],
    }


def test_extract_delivery_semantics_separates_run_channels():
    no_ball_four = extract_delivery_semantics(
        _delivery(batter_runs=4, extras={"noballs": 1}), REGISTRY)
    assert no_ball_four["team_runs"] == 5
    assert no_ball_four["batter_runs"] == 4
    assert no_ball_four["bowler_runs"] == 5
    assert no_ball_four["is_noball"] is True
    assert no_ball_four["is_legal"] is False
    assert no_ball_four["is_boundary"] is True

    byes = extract_delivery_semantics(
        _delivery(extras={"byes": 2}), REGISTRY)
    assert byes["team_runs"] == 2
    assert byes["batter_runs"] == 0
    assert byes["bowler_runs"] == 0
    assert byes["is_legal"] is True

    legbyes = extract_delivery_semantics(
        _delivery(extras={"legbyes": 1}), REGISTRY)
    assert legbyes["team_runs"] == 1
    assert legbyes["batter_runs"] == 0
    assert legbyes["bowler_runs"] == 0
    assert legbyes["is_legal"] is True


def test_non_boundary_flag_prevents_false_boundary():
    semantics = extract_delivery_semantics(
        _delivery(batter_runs=4, non_boundary=True), REGISTRY)
    assert semantics["is_boundary"] is False


def test_parser_excludes_illegal_rows_and_uses_off_bat_target():
    nonstriker_runout = [{
        "player_out": "Non Striker",
        "kind": "run out",
        "fielders": [{"name": "Bowler"}],
    }]
    deliveries = [
        _delivery(extras={"wides": 1}),
        _delivery(batter_runs=4, extras={"noballs": 1}),
        _delivery(extras={"byes": 2}),
        _delivery(extras={"legbyes": 1}),
        _delivery(batter_runs=3),
        _delivery(batter_runs=1, wickets=nonstriker_runout),
    ]
    tracker = PlayerStatsTracker()

    rows, totals, _, details, _ = parse_match_data_v2(
        json.dumps(_match(deliveries)),
        tracker,
        match_ref="i5_test",
        delivery_semantics=I5_DELIVERY_SEMANTICS,
    )

    # Wide and no-ball deliveries update state but are not model rows.
    assert len(rows) == 4
    assert totals == [13]
    assert [row["team_runs"] for row in rows] == [2, 1, 3, 1]
    assert [row["batter_runs"] for row in rows] == [0, 0, 3, 1]
    assert [row["ball_outcome"] for row in rows] == [0, 0, 2, -1]

    # The first legal row sees both earlier illegal-delivery runs, but no
    # legal ball has yet elapsed.
    assert rows[0]["score"] == 6
    assert rows[0]["balls_bowled"] == 0

    batter = tracker.batting_stats["bat"]
    assert batter["runs"] == 8
    assert batter["balls"] == 4
    assert batter["dismissals"] == 0
    assert tuple(batter[key] for key in ("c0", "c1", "c2", "c4", "c6", "cw")) == (
        2, 0, 1, 0, 0, 1)

    # A non-striker run-out belongs to the dismissed player, not the striker
    # or bowler, while the model's team-wicket target remains intact.
    assert tracker.batting_stats["non"]["dismissals"] == 1
    bowler = tracker.bowling_stats["bowl"]
    assert bowler["runs_given"] == 10
    assert bowler["balls_bowled"] == 4
    assert bowler["wickets"] == 0

    detail = details[0]
    assert detail["total_runs"] == 13
    assert detail["total_balls"] == 4
    assert detail["powerplay_runs"] == 13
    assert detail["powerplay_balls"] == 4
    assert tuple(detail[key] for key in ("c0", "c1", "c2", "c4", "c6", "cw")) == (
        2, 0, 1, 0, 0, 1)


def test_legacy_parser_contract_remains_the_default():
    deliveries = [
        _delivery(extras={"wides": 1}),
        _delivery(batter_runs=4, extras={"noballs": 1}),
        _delivery(extras={"byes": 2}),
    ]
    tracker = PlayerStatsTracker()

    rows, totals, _, details, _ = parse_match_data_v2(
        json.dumps(_match(deliveries)),
        tracker,
        match_ref="legacy_test",
    )

    assert len(rows) == 3
    assert [row["ball_outcome"] for row in rows] == [1, 4, 2]
    assert totals == [8]
    assert tracker.batting_stats["bat"]["runs"] == 8
    assert tracker.batting_stats["bat"]["balls"] == 3
    assert details[0]["total_runs"] == 2
    assert details[0]["total_balls"] == 1


def test_match_log_keeps_zero_ball_runs_and_dismissals():
    assert _has_nonzero_match_stats(
        {"runs": 0, "balls": 0, "dismissals": 1},
        ("runs", "balls", "dismissals"),
    )
    assert _has_nonzero_match_stats(
        {"runs": 4, "balls": 0, "dismissals": 0},
        ("runs", "balls", "dismissals"),
    )
    assert not _has_nonzero_match_stats(
        {"runs": 0, "balls": 0, "dismissals": 0},
        ("runs", "balls", "dismissals"),
    )
