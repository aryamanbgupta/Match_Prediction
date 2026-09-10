from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

from daily.settle_daily import build_revisions, main, settle_locked


def test_settlement_revisions_append_and_highest_increments():
    fixture = {"fixture_id": "m", "market_id": "m", "team1": "A", "team2": "B",
               "scheduled_start": "2026-09-10T11:00:00+00:00"}
    outcome = {"cricsheet_id": "c", "date": "2026-09-10", "teams": ["A", "B"],
               "winner": "A", "void": False, "void_reason": None}
    first = build_revisions([fixture], [outcome], [])[0]
    assert first["revision"] == 1
    assert build_revisions([fixture], [outcome], [first]) == []
    corrected = {**outcome, "winner": None, "void": True, "void_reason": "no result"}
    second = build_revisions([fixture], [corrected], [first])[0]
    assert second["revision"] == 2 and second["void"] is True


def test_repeated_fixture_captures_allocate_only_one_revision_one():
    fixture = {"fixture_id": "m", "market_id": "m", "team1": "A", "team2": "B",
               "scheduled_start": "2026-09-10T11:00:00+00:00",
               "quote_ts": "2026-09-10T09:59:00+00:00"}
    later = {**fixture, "quote_ts": "2026-09-10T10:00:00+00:00"}
    outcome = {"cricsheet_id": "c", "date": "2026-09-10", "teams": ["A", "B"],
               "winner": "A", "void": False, "void_reason": None}
    rows = build_revisions([later, fixture, later], [outcome], [])
    assert len(rows) == 1
    assert rows[0]["revision"] == 1
    assert build_revisions([fixture, later], [outcome], rows) == []


def test_day_one_fixture_settles_when_result_arrives_on_day_three(tmp_path):
    fixtures = tmp_path / "daily" / "fixtures"
    fixtures.mkdir(parents=True)
    fixture = {"fixture_id": "m", "market_id": "m", "team1": "A", "team2": "B",
               "scheduled_start": "2026-09-10T11:00:00+00:00"}
    (fixtures / "2026-09-10.jsonl").write_text(json.dumps(fixture) + "\n")
    (fixtures / "2026-09-11.jsonl").write_text("")
    day_three = tmp_path / "daily" / "context" / "2026-09-12"
    day_three.mkdir(parents=True)
    (day_three / "c.json").write_text(json.dumps({
        "info": {"dates": ["2026-09-10"], "teams": ["A", "B"],
                 "outcome": {"winner": "A"}}
    }))
    output = tmp_path / "daily" / "settled.jsonl"
    assert main(["--fixtures-dir", str(fixtures), "--source-dir", str(day_three),
                 "--out", str(output)]) == 0
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [(row["fixture_id"], row["revision"], row["winner"])
            for row in rows] == [("m", 1, "A")]


def test_concurrent_settlement_allocates_one_revision(tmp_path):
    fixture = {"fixture_id": "m", "market_id": "m", "team1": "A", "team2": "B",
               "scheduled_start": "2026-09-10T11:00:00+00:00"}
    outcome = {"cricsheet_id": "c", "date": "2026-09-10", "teams": ["A", "B"],
               "winner": "A", "void": False, "void_reason": None}
    output = tmp_path / "settled.jsonl"
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: settle_locked(output, [fixture], [outcome]), range(2)))
    assert sorted(map(len, results)) == [0, 1]
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["revision"] for row in rows] == [1]
    assert output.with_name(output.name + ".lock").is_file()
