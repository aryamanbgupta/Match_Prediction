from __future__ import annotations

from copy import deepcopy

import pytest

from daily.score_daily import highest_settlements, select_scored_line


def line(attempt="2026-09-10T10:00:00+00:00", quote_ts="2026-09-10T10:00:00+00:00",
         run_kind="t60"):
    return {"cohort_id": "daily_v1", "fixture_id": "m", "run_kind": run_kind,
            "attempt_ts": attempt, "quote_ts": quote_ts,
            "scheduled_start": "2026-09-10T11:00:00+00:00",
            "quote": {"A": .5, "B": .5}}


@pytest.mark.parametrize("attempt", ["2026-09-10T09:45:00+00:00", "2026-09-10T10:15:00+00:00"])
def test_attempt_window_is_inclusive(attempt):
    selected, _ = select_scored_line([line(attempt, attempt)], "daily_v1", "m")
    assert selected is not None


def test_each_clause_excludes():
    cases = [
        line(run_kind="toss"),
        line("2026-09-10T09:44:59+00:00", "2026-09-10T10:00:00+00:00"),
        line("2026-09-10T10:00:00+00:00", "2026-09-10T10:00:01+00:00"),
        line("2026-09-10T10:00:00+00:00", "2026-09-10T09:44:59+00:00"),
        line("2026-09-10T11:00:00+00:00", "2026-09-10T10:00:00+00:00"),
    ]
    for candidate in cases:
        assert select_scored_line([candidate], "daily_v1", "m")[0] is None


def test_latest_wins_order_independently_and_scores_once():
    early = line("2026-09-10T09:55:00+00:00", "2026-09-10T09:50:00+00:00")
    late = line("2026-09-10T10:05:00+00:00", "2026-09-10T10:00:00+00:00")
    assert select_scored_line([early, late], "daily_v1", "m")[0] == late
    assert select_scored_line([late, early], "daily_v1", "m")[0] == late


def test_tied_latest_attempt_fails_closed():
    duplicate = deepcopy(line())
    duplicate["p_team1"] = .7
    with pytest.raises(ValueError, match="failing closed"):
        select_scored_line([line(), duplicate], "daily_v1", "m")


def test_settlement_conflicts_only_apply_to_highest_revision_order_independent():
    low_a = {"fixture_id": "m", "revision": 1, "winner": "A"}
    low_b = {"fixture_id": "m", "revision": 1, "winner": "B"}
    high = {"fixture_id": "m", "revision": 2, "winner": "A"}
    assert highest_settlements([low_a, low_b, high]) == {"m": high}
    assert highest_settlements([high, low_b, low_a]) == {"m": high}
