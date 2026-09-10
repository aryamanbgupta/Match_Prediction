#!/usr/bin/env python3
"""Append projected or hand-confirmed XI records for daily fixtures."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))


def _lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def latest_lineups(source_dirs: list[Path], teams: set[str], before: str) -> dict[str, list[str]]:
    found: dict[str, tuple[str, str, list[str]]] = {}
    for source in source_dirs:
        for path in source.glob("*.json"):
            data = json.loads(path.read_text())
            info = data.get("info", {})
            dates = info.get("dates") or []
            if not dates or str(dates[0]) >= before:
                continue
            registry = (info.get("registry") or {}).get("people") or {}
            for team, players in (info.get("players") or {}).items():
                if team not in teams:
                    continue
                ids = [str(registry.get(player, player)) for player in players]
                key = (str(dates[0]), path.stem)
                if team not in found or key > found[team][:2]:
                    found[team] = (key[0], key[1], ids)
    return {team: value[2] for team, value in found.items()}


def _parse_ts(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("daily timestamps must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def resolve_with_counts(fixtures: list[dict], source_dirs: list[Path],
                        confirmed: dict | None = None,
                        now: datetime | None = None) -> tuple[list[dict], dict[str, int]]:
    rows = []
    confirmed = confirmed or {}
    cutoff = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    expired_skipped = 0
    for fixture in fixtures:
        if _parse_ts(fixture["scheduled_start"]) <= cutoff:
            expired_skipped += 1
            continue
        supplied = confirmed.get(fixture["fixture_id"])
        if supplied:
            lineups = supplied.get("lineups", supplied)
            mode = "confirmed"
        else:
            lineups = latest_lineups(
                source_dirs, {fixture["team1"], fixture["team2"]},
                fixture["scheduled_start"][:10],
            )
            mode = "projected"
        if fixture["team1"] not in lineups or fixture["team2"] not in lineups:
            raise RuntimeError(f"no prior XI for fixture {fixture['fixture_id']}")
        rows.append({
            "fixture_id": fixture["fixture_id"],
            "lineup_mode": mode,
            "team1_lineup": lineups[fixture["team1"]],
            "team2_lineup": lineups[fixture["team2"]],
            "resolved_ts": datetime.now(timezone.utc).isoformat(),
        })
    return rows, {"added": len(rows), "expired_skipped": expired_skipped}


def resolve(fixtures: list[dict], source_dirs: list[Path], confirmed: dict | None = None,
            now: datetime | None = None) -> list[dict]:
    rows, _ = resolve_with_counts(fixtures, source_dirs, confirmed, now)
    return rows


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--fixtures", type=Path, required=True)
    p.add_argument("--source-dir", type=Path, action="append", required=True)
    p.add_argument("--confirmed", type=Path)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    confirmed = json.loads(a.confirmed.read_text()) if a.confirmed else None
    rows, counts = resolve_with_counts(_lines(a.fixtures), a.source_dir, confirmed)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    existing = set(a.out.read_text().splitlines()) if a.out.exists() else set()
    with a.out.open("a") as f:
        for row in rows:
            encoded = json.dumps(row, sort_keys=True)
            if encoded not in existing:
                f.write(encoded + "\n")
    print(json.dumps(counts, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
