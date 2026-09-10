#!/usr/bin/env python3
"""Build a causal empirical prior for intended T20 bowling-roster size.

Only regulation innings with at least 100 legal balls identify the intended
unit without conflating roster size with an early chase/all-out. Runtime lookup
uses complete calendar years strictly before the fixture year.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from registered_experiment import reject_sealed
from artifacts import artifact_path


def build(source: Path, min_legal_balls: int = 100) -> dict:
    by_year = defaultdict(Counter)
    files_seen = matches_seen = innings_seen = 0
    for path in sorted(source.glob("*.json")):
        files_seen += 1
        try:
            document = json.load(path.open())
        except (OSError, json.JSONDecodeError):
            continue
        info = document.get("info", {})
        if info.get("gender") != "male" or not info.get("dates"):
            continue
        year = int(str(info["dates"][0])[:4])
        matches_seen += 1
        for innings in document.get("innings", [])[:2]:
            bowlers, legal = set(), 0
            for over in innings.get("overs", []):
                for delivery in over.get("deliveries", []):
                    if delivery.get("bowler"):
                        bowlers.add(delivery["bowler"])
                    extras = delivery.get("extras", {}) or {}
                    legal += int(not extras.get("wides")
                                 and not extras.get("noballs"))
            if legal >= min_legal_balls and bowlers:
                by_year[year][len(bowlers)] += 1
                innings_seen += 1
    return {
        "schema_version": 1,
        "policy": "causal_intended_bowling_roster_size_v1",
        "source_dir": str(source),
        "gender": "male",
        "regulation_innings_only": True,
        "min_legal_balls": min_legal_balls,
        "runtime_year_contract": "complete years strictly before fixture year",
        # Weak hand-set Dirichlet-style prior for fixtures before the first
        # corpus year only (mode 6 bowlers, one pseudo-count of spread each
        # way); swamped by real counts the moment any prior year exists.
        "cold_start_counts": {"5": 1, "6": 3, "7": 1},
        "n_files_seen": files_seen,
        "n_matches_seen": matches_seen,
        "n_eligible_innings": innings_seen,
        "by_year": {
            str(year): {str(size): int(count)
                        for size, count in sorted(counts.items())}
            for year, counts in sorted(by_year.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("data/t20s_json"))
    parser.add_argument("--role", default="bowler_roster_policy")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--min-legal-balls", type=int, default=100)
    args = parser.parse_args()
    args.out = artifact_path(args.role, args.out)
    for path in (args.source, args.out):
        reject_sealed(path)
    result = build(args.source, args.min_legal_balls)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {args.out}: {result['n_eligible_innings']:,} innings")


if __name__ == "__main__":
    main()
