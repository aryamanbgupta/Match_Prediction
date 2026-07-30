"""Build Cricsheet-shaped match records for The Hundred.

Two jobs:

1. ``--extract-history`` copies the men's Hundred matches out of a Cricsheet
   ``hnd_json`` pool into ``data/hundred/context_hnd_json/``. These are real
   Cricsheet files (with deliveries) and are used as *context state only* —
   tracker walks and backtests — never to rebuild the production stats cache.

2. ``--build-2026`` converts the hand-transcribed scorecard file
   ``data/hundred/season_2026_men_source.json`` into Cricsheet-shaped JSON in
   ``data/hundred/season_2026_men/``. Cricsheet had not published the 2026
   season when this was written, so the 2026 records carry ``info`` only: no
   deliveries, which is all the match-level model and the form/H2H/home
   trackers need.

Player names are resolved against ``data/all_players_enriched.csv`` on
``name``, ``full_name`` and ``unique_name`` (Cricsheet's initials form, e.g.
"WG Jacks"). Any unresolved name is a hard failure — a silently unresolved
player would quietly zero out that slot's ELO and career features.

Usage:
    uv run python scripts/build_hundred_matches.py --extract-history --build-2026
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_HISTORY_ZIP = Path(
    "/Users/aryamangupta/Projects/stat-generator/data/cricsheet/hnd_json.zip"
)
HISTORY_DIR = REPO / "data" / "hundred" / "context_hnd_json"
SOURCE_2026 = REPO / "data" / "hundred" / "season_2026_men_source.json"
OUT_2026 = REPO / "data" / "hundred" / "season_2026_men"
PLAYER_CSV = REPO / "data" / "all_players_enriched.csv"


def build_name_lookup(csv_path: Path = PLAYER_CSV) -> dict[str, str]:
    """Map every known spelling of a player to their Cricsheet ID.

    First write wins so that the canonical ``name`` column beats a colliding
    ``full_name`` from a different player.
    """
    lookup: dict[str, str] = {}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            pid = (row.get("cricsheet_id") or "").strip()
            if not pid:
                continue
            for column in ("name", "unique_name", "full_name"):
                value = (row.get(column) or "").strip()
                if value:
                    lookup.setdefault(value.lower(), pid)
    return lookup


def resolve_lineup(names: list[str],
                   lookup: dict[str, str]) -> tuple[list[str], list[str]]:
    resolved, missing = [], []
    for name in names:
        pid = lookup.get(str(name).strip().lower())
        if pid:
            resolved.append(pid)
        else:
            missing.append(name)
    return resolved, missing


def extract_history(zip_path: Path, out_dir: Path) -> int:
    """Copy men's Hundred matches out of a Cricsheet pool (zip or directory)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0

    def keep(name: str, payload: bytes) -> None:
        nonlocal written
        data = json.loads(payload)
        if (data.get("info") or {}).get("gender") != "male":
            return
        (out_dir / name).write_bytes(payload)
        written += 1

    if zip_path.is_dir():
        for path in sorted(zip_path.glob("*.json")):
            keep(path.name, path.read_bytes())
    else:
        with zipfile.ZipFile(zip_path) as archive:
            for entry in sorted(archive.namelist()):
                if entry.endswith(".json"):
                    keep(Path(entry).name, archive.read(entry))
    print(f"  wrote {written} men's Hundred matches -> {out_dir}")
    return written


def build_2026(source_path: Path, out_dir: Path,
               lookup: dict[str, str]) -> int:
    source = json.loads(source_path.read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    all_missing: list[str] = []
    written = 0

    for match in source["matches"]:
        teams = match["teams"]
        players, registry = {}, {}
        for team in teams:
            names = match["lineups"][team]
            if len(names) != 11:
                raise ValueError(
                    f"match {match['match_id']} {team}: expected 11 players, "
                    f"got {len(names)}"
                )
            ids, missing = resolve_lineup(names, lookup)
            all_missing.extend(f"{team}: {name}" for name in missing)
            # Cricsheet stores display names in `players` and the name -> id
            # map in `registry.people`. Keep the same shape so downstream code
            # can treat these records exactly like real Cricsheet files.
            players[team] = names
            registry.update(dict(zip(names, ids)))

        record = {
            "meta": {
                "data_version": "hundred_2026_transcribed_v1",
                "created": source["_provenance"]["transcribed_on"],
                "revision": 1,
            },
            "info": {
                "balls_per_over": 5,
                "city": match["venue"].split(", ")[-1],
                "dates": [match["date"]],
                "event": {
                    "name": source["event"],
                    "match_number": match["match_number"],
                },
                "gender": source["gender"],
                "match_type": "HND",
                "outcome": {"winner": match["winner"]},
                "overs": 20,
                "players": players,
                "registry": {"people": registry},
                "team_type": source["team_type"],
                "teams": teams,
                "toss": {
                    "decision": match["toss_decision"],
                    "winner": match["toss_winner"],
                },
                "venue": match["venue"],
            },
        }
        (out_dir / f"{match['match_id']}.json").write_text(
            json.dumps(record, indent=2)
        )
        written += 1

    if all_missing:
        print("  UNRESOLVED PLAYERS:")
        for entry in sorted(set(all_missing)):
            print(f"    - {entry}")
        raise SystemExit(
            f"{len(set(all_missing))} player names did not resolve; fix the "
            f"spelling in {source_path.name} before using these records"
        )
    print(f"  wrote {written} transcribed 2026 matches -> {out_dir}")
    return written


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract-history", action="store_true")
    ap.add_argument("--history-source", type=Path, default=DEFAULT_HISTORY_ZIP,
                    help="Cricsheet hnd_json zip or an already-extracted dir")
    ap.add_argument("--history-dir", type=Path, default=HISTORY_DIR)
    ap.add_argument("--build-2026", action="store_true")
    ap.add_argument("--source-2026", type=Path, default=SOURCE_2026)
    ap.add_argument("--out-2026", type=Path, default=OUT_2026)
    args = ap.parse_args()

    if not (args.extract_history or args.build_2026):
        ap.error("pass --extract-history and/or --build-2026")

    if args.extract_history:
        print(f"Extracting men's Hundred history from {args.history_source}")
        extract_history(args.history_source, args.history_dir)

    if args.build_2026:
        print(f"Building 2026 records from {args.source_2026}")
        build_2026(args.source_2026, args.out_2026, build_name_lookup())

    return 0


if __name__ == "__main__":
    sys.exit(main())
