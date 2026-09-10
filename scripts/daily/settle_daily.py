#!/usr/bin/env python3
"""Append Cricsheet settlement revisions for daily fixtures."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def cricsheet_outcomes(source_dirs: list[Path]) -> list[dict]:
    rows = []
    for source in source_dirs:
        for path in source.glob("*.json"):
            data = json.loads(path.read_text())
            info = data.get("info", {})
            outcome = info.get("outcome") or {}
            teams, dates = info.get("teams") or [], info.get("dates") or []
            if len(teams) != 2 or not dates:
                continue
            winner = outcome.get("winner")
            result = str(outcome.get("result") or "").lower()
            void = not winner and result in {"no result", "abandoned", "cancelled", "postponed"}
            rows.append({"cricsheet_id": path.stem, "date": str(dates[0]),
                         "teams": teams, "winner": winner, "void": void,
                         "void_reason": result or None})
    return rows


def dedupe_fixtures(fixtures: list[dict]) -> list[dict]:
    """Keep one deterministic, latest capture for each fixture id."""
    selected: dict[str, dict] = {}
    for fixture in fixtures:
        fixture_id = fixture["fixture_id"]
        candidate_key = (
            str(fixture.get("quote_ts") or ""),
            json.dumps(fixture, sort_keys=True, separators=(",", ":")),
        )
        current = selected.get(fixture_id)
        if current is None:
            selected[fixture_id] = fixture
            continue
        current_key = (
            str(current.get("quote_ts") or ""),
            json.dumps(current, sort_keys=True, separators=(",", ":")),
        )
        if candidate_key > current_key:
            selected[fixture_id] = fixture
    return [selected[key] for key in sorted(selected)]


def build_revisions(fixtures: list[dict], outcomes: list[dict], existing: list[dict]) -> list[dict]:
    revisions = {}
    for row in existing:
        revisions[row["fixture_id"]] = max(revisions.get(row["fixture_id"], 0), int(row["revision"]))
    result = []
    latest_by_fixture: dict[str, dict] = {}
    for row in existing:
        key = row["fixture_id"]
        if key not in latest_by_fixture or int(row["revision"]) > int(
            latest_by_fixture[key]["revision"]
        ):
            latest_by_fixture[key] = row
    for fixture in dedupe_fixtures(fixtures):
        fixture_status = str(fixture.get("status") or "").lower()
        if fixture_status in {"postponed", "no result", "abandoned", "cancelled"}:
            matches = [{"cricsheet_id": None, "winner": None, "void": True,
                        "void_reason": fixture_status}]
        else:
            target_teams = {fixture["team1"], fixture["team2"]}
            matches = [row for row in outcomes if set(row["teams"]) == target_teams
                       and row["date"] == fixture["scheduled_start"][:10]]
        if len(matches) > 1:
            raise RuntimeError(f"ambiguous Cricsheet settlement for {fixture['fixture_id']}")
        if not matches:
            continue
        outcome = matches[0]
        latest = latest_by_fixture.get(fixture["fixture_id"])
        content = (outcome["cricsheet_id"], outcome["winner"], outcome["void"], outcome["void_reason"])
        if latest and content == (latest.get("cricsheet_id"), latest.get("winner"),
                                  latest.get("void"), latest.get("void_reason")):
            continue
        if (latest and latest.get("winner") and outcome.get("winner")
                and latest["winner"] != outcome["winner"]):
            raise RuntimeError("a settlement revision may add or void, not change a winner")
        revision = revisions.get(fixture["fixture_id"], 0) + 1
        row = {
            "cohort_id": "daily_v1",
            "fixture_id": fixture["fixture_id"],
            "market_id": fixture.get("market_id", fixture["fixture_id"]),
            "cricsheet_id": outcome["cricsheet_id"],
            "revision": revision,
            "winner": outcome["winner"],
            "void": bool(outcome["void"]),
            "void_reason": outcome["void_reason"],
            "settled_ts": datetime.now(timezone.utc).isoformat(),
        }
        result.append(row)
        revisions[fixture["fixture_id"]] = revision
        latest_by_fixture[fixture["fixture_id"]] = row
    return result


def settle_locked(output: Path, fixtures: list[dict], outcomes: list[dict]) -> list[dict]:
    """Lock the complete settlement read, revision allocation, and append."""
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output.with_name(output.name + ".lock")
    with lock_path.open("a+") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        existing = _jsonl(output)
        rows = build_revisions(fixtures, outcomes, existing)
        with output.open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return rows


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--fixtures", type=Path, action="append", default=[])
    p.add_argument("--fixtures-dir", type=Path)
    p.add_argument("--source-dir", type=Path, action="append", required=True)
    p.add_argument("--out", type=Path, default=REPO / "daily" / "settled.jsonl")
    a = p.parse_args(argv)
    fixture_paths = list(a.fixtures)
    if a.fixtures_dir:
        fixture_paths.extend(sorted(a.fixtures_dir.glob("*.jsonl")))
    if not fixture_paths:
        p.error("at least one --fixtures or --fixtures-dir is required")
    fixtures = [row for path in fixture_paths for row in _jsonl(path)]
    rows = settle_locked(a.out, fixtures, cricsheet_outcomes(a.source_dir))
    print(len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
