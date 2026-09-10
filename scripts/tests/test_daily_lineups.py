from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from daily.resolve_lineups import latest_lineups, resolve, resolve_with_counts


MINI = Path(__file__).parents[2] / "tests" / "fixtures" / "cricsheet_mini"


def test_projected_xi_is_each_teams_latest_prior_match():
    fixture = {"fixture_id": "market-1", "team1": "Alpha CC", "team2": "Beta CC",
               "scheduled_start": "2025-01-21T12:00:00+00:00"}
    rows = resolve([fixture], [MINI], now=datetime(2025, 1, 21, tzinfo=timezone.utc))
    assert rows[0]["lineup_mode"] == "projected"
    assert len(rows[0]["team1_lineup"]) == 11
    assert latest_lineups([MINI], {"Alpha CC", "Beta CC"}, "2025-01-21")["Alpha CC"]


def test_confirmed_lineups_override_projection():
    fixture = {"fixture_id": "m", "team1": "A", "team2": "B",
               "scheduled_start": "2025-01-21T12:00:00+00:00"}
    confirmed = {"m": {"A": ["a"] * 11, "B": ["b"] * 11}}
    row = resolve([fixture], [MINI], confirmed,
                  now=datetime(2025, 1, 21, tzinfo=timezone.utc))[0]
    assert row["lineup_mode"] == "confirmed"


def test_expired_missing_xi_is_skipped_before_upcoming_lineup_resolution():
    expired = {
        "fixture_id": "expired", "team1": "Never Seen", "team2": "Also Missing",
        "scheduled_start": "2025-01-20T12:00:00+00:00",
    }
    upcoming = {
        "fixture_id": "upcoming", "team1": "Alpha CC", "team2": "Beta CC",
        "scheduled_start": "2025-01-21T12:00:00+00:00",
    }
    rows, counts = resolve_with_counts(
        [expired, upcoming], [MINI],
        now=datetime(2025, 1, 21, 11, tzinfo=timezone.utc),
    )
    assert [row["fixture_id"] for row in rows] == ["upcoming"]
    assert counts == {"added": 1, "expired_skipped": 1}
