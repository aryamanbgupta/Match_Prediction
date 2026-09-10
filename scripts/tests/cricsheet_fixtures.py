"""Deterministic Cricsheet-style fixtures shared by the test suites."""

from __future__ import annotations

import csv
import json
from pathlib import Path


TEAMS = {
    "Alpha CC": [(f"a{i:02d}", f"Alpha P{i}") for i in range(1, 12)],
    "Beta CC": [(f"b{i:02d}", f"Beta P{i}") for i in range(1, 12)],
}
VENUE = "Parity Park"
EVENT = "Indian Premier League"
FIXTURE_DATE = "2025-08-01"


def delivery(
    batter: str = "Batter",
    bowler: str = "Bowler",
    runs: int | None = None,
    wicket: bool = False,
    *,
    batter_runs: int | None = None,
    non_striker: str | None = None,
    extras: dict[str, int] | None = None,
    wickets: list[dict] | None = None,
    non_boundary: bool = False,
    include_runs: bool = True,
) -> dict:
    """Build one delivery, with optional extras and dismissal details."""
    batter_runs = runs if batter_runs is None else batter_runs
    batter_runs = 0 if batter_runs is None else batter_runs
    extras = extras or {}
    result = {
        "batter": batter,
        "bowler": bowler,
        "non_striker": non_striker or "Non Striker",
    }
    if include_runs:
        extras_total = sum(extras.values())
        result["runs"] = {
            "batter": batter_runs,
            "extras": extras_total,
            "total": batter_runs + extras_total,
        }
        if non_boundary:
            result["runs"]["non_boundary"] = True
    if extras:
        result["extras"] = extras
    if wickets is not None:
        result["wickets"] = wickets
    elif wicket:
        result["wickets"] = [{"kind": "bowled", "player_out": batter}]
    return result


def innings(
    team: str,
    bowling_team: str | None = None,
    seed: int = 0,
    *,
    batter: str | None = None,
    bowler: str | None = None,
    runs_per_over: list[list[int]] | None = None,
    super_over: bool = False,
    pairs_and_bowlers: list[tuple[str, str, str, str | None]] | None = None,
    overs: list[dict] | None = None,
) -> dict:
    """Build an innings in canonical, explicit-over, or appearance-list form."""
    if overs is not None:
        built_overs = overs
    elif pairs_and_bowlers is not None:
        deliveries = []
        for striker, non_striker, delivery_bowler, wicket_on in pairs_and_bowlers:
            wickets = [{"player_out": wicket_on}] if wicket_on else None
            deliveries.append(delivery(
                striker,
                delivery_bowler,
                non_striker=non_striker,
                wickets=wickets,
                include_runs=False,
            ))
        built_overs = [{"over": 0, "deliveries": deliveries}]
    elif runs_per_over is not None:
        built_overs = [
            {
                "over": over_no,
                "deliveries": [delivery(batter or f"{team} batter", bowler or "Bowler", run)
                               for run in over_runs],
            }
            for over_no, over_runs in enumerate(runs_per_over)
        ]
    else:
        if team not in TEAMS or bowling_team not in TEAMS:
            raise ValueError("canonical innings requires teams from TEAMS")
        batter_names = [name for _, name in TEAMS[team]]
        bowler_names = [name for _, name in TEAMS[bowling_team]][7:11]
        runs_cycle = [0, 1, (seed % 3) + 1, 4, 6, 1]
        built_overs = []
        for over_no in range(5):
            deliveries = [
                delivery(
                    batter_names[over_no % 4],
                    bowler_names[over_no % 4],
                    runs_cycle[(index + seed) % 6],
                    wicket=(over_no == 3 and index == 5),
                    non_striker=batter_names[over_no % 4],
                )
                for index in range(6)
            ]
            built_overs.append({"over": over_no, "deliveries": deliveries})
    result = {"team": team, "overs": built_overs}
    if super_over:
        result["super_over"] = True
    return result


def match_json(
    date: str = "2026-01-01",
    home: str = "Alpha CC",
    away: str = "Beta CC",
    seed: int = 0,
    *,
    match_id: str | None = None,
    teams: list[str] | tuple[str, str] | None = None,
    winner: str | None = None,
    venue: str = VENUE,
    event: str = EVENT,
    top_scores: tuple[int, int] | None = None,
    star_runs: int | None = None,
    forward_stub: bool = False,
    serialize: bool = False,
    info: dict | None = None,
    innings_data: list[dict] | None = None,
) -> dict | str:
    """Build a match, with keyword variants for specialized regression tests."""
    if teams is not None:
        home, away = teams
    if info is not None:
        result = {"info": info, "innings": innings_data or []}
    elif forward_stub:
        names = {
            home: [f"{home} Player {index}" for index in range(11)],
            away: [f"{away} Player {index}" for index in range(11)],
        }
        registry = {
            name: f"{team.lower()}_{index}"
            for team, roster in names.items()
            for index, name in enumerate(roster)
        }
        result = {
            "_test_id": match_id or "001",
            "info": {
                "dates": [date], "teams": [home, away], "venue": venue,
                "team_type": "international", "event": {"name": "Synthetic"},
                "toss": {"winner": home, "decision": "field"},
                "registry": {"people": registry}, "players": names,
            },
            "innings": [{"forbidden": "outcome-bearing"}],
        }
    elif top_scores is not None:
        first, second = top_scores
        result = {
            "info": {
                "gender": "male", "match_type": "T20", "dates": [date],
                "teams": ["A", "B"],
                "players": {"A": ["A star", "A support"],
                            "B": ["B star", "B support"]},
                "venue": "Test Oval",
            },
            "innings": [
                innings("A", overs=[{"over": 0, "deliveries":
                    [delivery("A star", "B bowler", 6, non_striker="NS")] * (first // 6)
                    + [delivery("A support", "B bowler", 1, non_striker="NS")] * 3}]),
                innings("B", overs=[{"over": 0, "deliveries":
                    [delivery("B star", "A bowler", 6, non_striker="NS")] * (second // 6)
                    + [delivery("B support", "A bowler", 1, non_striker="NS")] * 3}]),
            ],
        }
    elif star_runs is not None:
        registry = {"Star Batter": "p_star", "NS X": "p_ns", "Some Bowler": "p_bowl"}
        result = {
            "info": {
                "match_type": "T20", "gender": "male", "dates": [date],
                "teams": ["Alpha", "Beta"],
                "players": {"Alpha": ["Star Batter", "NS X"], "Beta": ["Some Bowler"]},
                "registry": {"people": registry}, "venue": "Test Oval",
                "outcome": {"winner": "Alpha"},
                "toss": {"winner": "Alpha", "decision": "bat"},
            },
            "innings": [
                innings("Alpha", overs=[{"over": 0, "deliveries": [
                    delivery("Star Batter", "Some Bowler", 1, non_striker="NS X")
                ] * star_runs}]),
                innings("Beta", overs=[{"over": 0, "deliveries": [
                    delivery("Some Bowler", "Star Batter", 1, non_striker="NS X")
                ]}]),
            ],
        }
    elif home not in TEAMS or away not in TEAMS:
        result = {
            "info": {
                "dates": [date], "gender": "male", "teams": [home, away],
                "venue": venue, "outcome": {"winner": winner or home},
            }
        }
    else:
        registry = {name: pid for team in (home, away) for pid, name in TEAMS[team]}
        first, second = (home, away) if seed % 2 == 0 else (away, home)
        result = {
            "info": {
                "match_type": "T20", "gender": "male", "dates": [date],
                "teams": [home, away],
                "players": {team: [name for _, name in TEAMS[team]] for team in (home, away)},
                "registry": {"people": registry}, "venue": venue,
                "event": {"name": event}, "team_type": "club",
                "outcome": {"winner": winner or home},
                "toss": {"winner": first, "decision": "bat"},
            },
            "innings": [innings(first, second, seed), innings(second, first, seed + 1)],
        }
    return json.dumps(result) if serialize else result


def write_corpus(corpus: Path) -> list[str]:
    """Write the canonical seven-match train/serve parity corpus."""
    corpus.mkdir()
    dates = ["2024-05-01", "2024-06-01", "2024-07-01", "2024-08-01",
             "2024-09-01", "2024-10-01", FIXTURE_DATE]
    for index, date in enumerate(dates):
        (corpus / f"{9000 + index}.json").write_text(
            json.dumps(match_json(date, "Alpha CC", "Beta CC", seed=index))
        )
    return dates


def write_metadata_csv(path: Path) -> None:
    """Write metadata for the canonical Alpha/Beta player registry."""
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cricsheet_id", "name", "cricinfo_id", "unique_name",
                         "full_name", "country", "dob", "batting_style", "bowling_style"])
        for players in TEAMS.values():
            for player_id, name in players:
                writer.writerow([player_id, name, "", name, name, "Testland",
                                 "1995-01-01", "Right-hand bat", "Right-arm medium"])
