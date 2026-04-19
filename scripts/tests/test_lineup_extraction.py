"""Unit tests for TestMatchLoader._extract_team_players and padding fallback.

Run:
    uv run python -m pytest scripts/tests/test_lineup_extraction.py -v
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sim_eval.loaders import TestMatchLoader  # noqa: E402


TEAM_A = "Team A"
TEAM_B = "Team B"
ROSTER_A = [f"A{i}" for i in range(1, 12)]  # A1..A11
ROSTER_B = [f"B{i}" for i in range(1, 12)]  # B1..B11
REGISTRY = {name: f"id_{name}" for name in ROSTER_A + ROSTER_B}


def _make_delivery(batter, non_striker, bowler, wickets=None):
    d = {"batter": batter, "non_striker": non_striker, "bowler": bowler}
    if wickets is not None:
        d["wickets"] = wickets
    return d


def _make_over(deliveries):
    return {"deliveries": deliveries}


def _innings_from_pairs(batting_team, pairs_and_bowlers):
    """Build an innings where `pairs_and_bowlers` is a list of
    (batter, non_striker, bowler, wicket_out_batter_or_None)."""
    deliveries = []
    for batter, ns, bowler, wicket_on in pairs_and_bowlers:
        wickets = [{"player_out": wicket_on}] if wicket_on else None
        deliveries.append(_make_delivery(batter, ns, bowler, wickets))
    return {"team": batting_team, "overs": [_make_over(deliveries)]}


def _base_info(roster_a=None, roster_b=None):
    return {
        "teams": [TEAM_A, TEAM_B],
        "dates": ["2025-01-01"],
        "venue": "Test Ground",
        "toss": {"winner": TEAM_A, "decision": "bat"},
        "registry": {"people": dict(REGISTRY)},
        "players": {
            TEAM_A: list(roster_a) if roster_a is not None else list(ROSTER_A),
            TEAM_B: list(roster_b) if roster_b is not None else list(ROSTER_B),
        },
        "event": {"name": "Test League"},
        "team_type": "club",
    }


@pytest.fixture
def loader():
    return TestMatchLoader()


def _names(players):
    return [p.name for p in players]


def test_full_match_all_batted(loader):
    """All 11 from each team bat; all 11 Bs bowl. Order: appearance in innings."""
    pairs_a = [(ROSTER_A[i], ROSTER_A[(i + 1) % 11], ROSTER_B[i % 11], None) for i in range(11)]
    pairs_b = [(ROSTER_B[i], ROSTER_B[(i + 1) % 11], ROSTER_A[i % 11], None) for i in range(11)]
    data = {
        "info": _base_info(),
        "innings": [
            _innings_from_pairs(TEAM_A, pairs_a),
            _innings_from_pairs(TEAM_B, pairs_b),
        ],
    }
    team_a = loader._extract_team_players(data, TEAM_A, REGISTRY, ROSTER_A)
    team_b = loader._extract_team_players(data, TEAM_B, REGISTRY, ROSTER_B)
    assert len(team_a) >= 11
    assert len(team_b) >= 11
    assert _names(team_a) == ROSTER_A
    assert _names(team_b) == ROSTER_B
    assert all(not p.player_id.startswith("player_") for p in team_a)


def test_chase_won_early(loader):
    """Chasing team: 7 batters appear, 5 bowled in defense (one overlap with batters).
    Tail-enders A8..A11 must be recovered from roster."""
    # Team A bowls first (Team B bats); Team A batters: A1..A7 (A5 also bowls in innings1)
    pairs_b = [(ROSTER_B[i], ROSTER_B[(i + 1) % 11], ROSTER_A[i], None) for i in range(5)]
    pairs_a = [(ROSTER_A[i], ROSTER_A[(i + 1) % 7], ROSTER_B[i % 11], None) for i in range(7)]
    data = {
        "info": _base_info(),
        "innings": [
            _innings_from_pairs(TEAM_B, pairs_b),
            _innings_from_pairs(TEAM_A, pairs_a),
        ],
    }
    team_a = loader._extract_team_players(data, TEAM_A, REGISTRY, ROSTER_A)
    assert len(team_a) >= 11
    # First 7 are batters in appearance order; rest are tail from roster
    assert _names(team_a)[:7] == ROSTER_A[:7]
    # Remaining must be A8..A11 (tail from roster, appended in roster order)
    assert set(_names(team_a)[7:]) == set(ROSTER_A[7:])
    assert all(not p.player_id.startswith("player_") for p in team_a)


def test_setting_team_five_down(loader):
    """Setting team batted 20 overs, 5 wickets — only 6 batters came to crease.
    Indices 6..10 fill from roster tail."""
    pairs_a = [(ROSTER_A[i], ROSTER_A[(i + 1) % 6], ROSTER_B[i % 11], None) for i in range(6)]
    pairs_b = [(ROSTER_B[i], ROSTER_B[(i + 1) % 11], ROSTER_A[i % 6], None) for i in range(11)]
    data = {
        "info": _base_info(),
        "innings": [
            _innings_from_pairs(TEAM_A, pairs_a),
            _innings_from_pairs(TEAM_B, pairs_b),
        ],
    }
    team_a = loader._extract_team_players(data, TEAM_A, REGISTRY, ROSTER_A)
    assert len(team_a) >= 11
    assert _names(team_a)[:6] == ROSTER_A[:6]
    assert set(_names(team_a)[6:]) == set(ROSTER_A[6:])


def test_no_result_abandoned(loader):
    """Rain-abandoned after 5 overs; Team B never bats. All Team B names come from roster."""
    pairs_a = [(ROSTER_A[i], ROSTER_A[(i + 1) % 2], ROSTER_B[i % 4], None) for i in range(2)]
    info = _base_info()
    info["outcome"] = {"result": "no result"}
    data = {
        "info": info,
        "innings": [_innings_from_pairs(TEAM_A, pairs_a)],
    }
    team_b = loader._extract_team_players(data, TEAM_B, REGISTRY, ROSTER_B)
    assert len(team_b) >= 11
    # First 4 are bowlers in appearance order, rest from roster in order
    assert _names(team_b)[:4] == ROSTER_B[:4]
    assert set(_names(team_b)) == set(ROSTER_B)


def test_missing_info_players_falls_back_to_dummy(loader, capsys):
    """If info.players is absent AND deliveries < 11, pad with dummies and warn."""
    pairs_a = [(ROSTER_A[0], ROSTER_A[1], ROSTER_B[0], None)]
    info = _base_info()
    info.pop("players")
    data = {"info": info, "innings": [_innings_from_pairs(TEAM_A, pairs_a)]}
    match_id, state = loader._create_match_state(data)
    captured = capsys.readouterr().out
    assert state is not None
    assert "Incomplete team lineups" in captured
    team1 = state.team1_lineup.players
    # Real A1, A2 at indices 0-1; dummy player_2..player_10 for the rest
    assert team1[0].name == "A1"
    assert team1[1].name == "A2"
    assert any(p.player_id.startswith("player_") for p in team1[2:])


def test_roster_size_ten_fallback(loader, capsys):
    """Roster = 10 names → pad 11th slot with dummy + warn."""
    short_roster = ROSTER_A[:10]
    pairs_a = [(short_roster[i], short_roster[(i + 1) % 10], ROSTER_B[i % 11], None) for i in range(10)]
    info = _base_info(roster_a=short_roster)
    data = {
        "info": info,
        "innings": [
            _innings_from_pairs(TEAM_A, pairs_a),
            _innings_from_pairs(TEAM_B, [(ROSTER_B[0], ROSTER_B[1], ROSTER_A[0], None)]),
        ],
    }
    match_id, state = loader._create_match_state(data)
    captured = capsys.readouterr().out
    assert state is not None
    assert "Incomplete team lineups" in captured
    team1 = state.team1_lineup.players
    assert len(team1) == 11
    assert team1[10].player_id.startswith("player_")


def test_name_not_in_registry(loader):
    """Player in roster but missing from registry → fallback ID = lowered_with_underscores."""
    missing = "Unregistered Player"
    roster = ROSTER_A[:10] + [missing]
    pairs_a = [(ROSTER_A[0], ROSTER_A[1], ROSTER_B[0], None)]
    data = {
        "info": _base_info(roster_a=roster),
        "innings": [_innings_from_pairs(TEAM_A, pairs_a)],
    }
    team_a = loader._extract_team_players(data, TEAM_A, REGISTRY, roster)
    last = next(p for p in team_a if p.name == missing)
    assert last.player_id == "unregistered_player"
    assert not last.player_id.startswith("player_")


def test_appearance_order_preserved(loader):
    """Openers always at indices 0,1; tail-enders at the back in roster order."""
    openers = ["A3", "A7"]
    pairs_a = [
        (openers[0], openers[1], ROSTER_B[0], None),
        (openers[1], openers[0], ROSTER_B[1], None),
        ("A1", openers[1], ROSTER_B[2], None),
        ("A5", "A1", ROSTER_B[3], None),
    ]
    data = {
        "info": _base_info(),
        "innings": [_innings_from_pairs(TEAM_A, pairs_a)],
    }
    team_a = loader._extract_team_players(data, TEAM_A, REGISTRY, ROSTER_A)
    assert _names(team_a)[:4] == ["A3", "A7", "A1", "A5"]
    assert len(team_a) >= 11
    tail = _names(team_a)[4:]
    # Tail contains only roster members not yet seen, in roster order
    seen = {"A3", "A7", "A1", "A5"}
    expected_tail = [n for n in ROSTER_A if n not in seen]
    assert tail == expected_tail


# ---- Option B: 12-man Impact Player squads ----

IMPACT_NAME = "A12"
ROSTER_A_12 = ROSTER_A + [IMPACT_NAME]
REGISTRY_12 = {**REGISTRY, IMPACT_NAME: f"id_{IMPACT_NAME}"}


def test_twelve_man_squad_preserved(loader):
    """IPL 2023+/ILT20/SMAT format: info.players[team] has 12 names. Only the
    XI appears in deliveries; the 12th (Impact Sub who never took the field)
    must still be present in the extracted lineup — with a real player_id."""
    pairs_a = [(ROSTER_A[i], ROSTER_A[(i + 1) % 11], ROSTER_B[i % 11], None) for i in range(11)]
    pairs_b = [(ROSTER_B[i], ROSTER_B[(i + 1) % 11], ROSTER_A[i % 11], None) for i in range(11)]
    info = _base_info()
    info["players"][TEAM_A] = list(ROSTER_A_12)
    info["registry"]["people"] = dict(REGISTRY_12)
    data = {
        "info": info,
        "innings": [
            _innings_from_pairs(TEAM_A, pairs_a),
            _innings_from_pairs(TEAM_B, pairs_b),
        ],
    }
    team_a = loader._extract_team_players(data, TEAM_A, REGISTRY_12, ROSTER_A_12)
    assert len(team_a) == 12
    # XI who actually played come first in appearance order; 12th is the
    # roster-tail Impact Sub.
    assert _names(team_a)[:11] == ROSTER_A
    assert _names(team_a)[11] == IMPACT_NAME
    assert all(not p.player_id.startswith("player_") for p in team_a)
    impact_player = team_a[11]
    assert impact_player.player_id == f"id_{IMPACT_NAME}"


def test_impact_sub_in_deliveries_and_roster(loader):
    """Impact Sub actually entered the XI (replaced a starter) and bowled /
    batted after the swap. Extractor must still return all 12 names, with
    appearance-order invariants preserved for the players who actually played."""
    # A1..A10 bat (A11 replaced mid-innings by A12 = Impact Sub). A12 bowls
    # later in opposing innings.
    pairs_a = [(ROSTER_A[i], ROSTER_A[(i + 1) % 10], ROSTER_B[i % 11], None) for i in range(10)]
    # A12 enters on the last ball of Team A's innings
    pairs_a.append((IMPACT_NAME, ROSTER_A[0], ROSTER_B[0], None))
    # Team B bats second; A12 bowls some deliveries
    pairs_b = [(ROSTER_B[i], ROSTER_B[(i + 1) % 11], ROSTER_A[i % 11], None) for i in range(9)]
    pairs_b.append((ROSTER_B[0], ROSTER_B[1], IMPACT_NAME, None))

    info = _base_info()
    info["players"][TEAM_A] = list(ROSTER_A_12)
    info["registry"]["people"] = dict(REGISTRY_12)
    # Cricsheet-style replacement event (not consulted by the extractor — this
    # is a forward-compatibility check that the fixture is realistic)
    data = {
        "info": info,
        "innings": [
            _innings_from_pairs(TEAM_A, pairs_a),
            _innings_from_pairs(TEAM_B, pairs_b),
        ],
    }
    team_a = loader._extract_team_players(data, TEAM_A, REGISTRY_12, ROSTER_A_12)
    assert len(team_a) == 12
    # Appearance-order: A1..A10 first (batting in innings 1), then A12
    # (entered as batter on the last ball of innings 1), then A11 (never
    # batted or bowled — comes from roster tail).
    names = _names(team_a)
    assert names[:10] == ROSTER_A[:10]
    assert names[10] == IMPACT_NAME
    assert names[11] == "A11"
    assert all(not p.player_id.startswith("player_") for p in team_a)
    # Spot-check ID lookup for the Impact Sub
    impact = next(p for p in team_a if p.name == IMPACT_NAME)
    assert impact.player_id == f"id_{IMPACT_NAME}"
