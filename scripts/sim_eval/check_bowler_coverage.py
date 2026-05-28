#!/usr/bin/env python3
"""Gate G5 diagnostic: what fraction of bowlers across the test set have
≥N historical balls in models/bowler_phase_usage.json (as-of fixture date)?

A low coverage means EmpiricalBowlerSelector falls back to the league prior
too often, defeating the point of the empirical signal.

Usage:
    uv run python scripts/sim_eval/check_bowler_coverage.py \
        --test-dir data/polymarket_test \
        --usage models/bowler_phase_usage.json \
        --threshold 100
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def cumulative_for_year(usage: dict, year: int) -> dict:
    """Return {cricsheet_id: total_balls} for all years strictly < year."""
    out: dict[str, int] = {}
    for cid, years in usage["by_player"].items():
        total = 0
        for y_str, counts in years.items():
            if int(y_str) < year:
                total += counts.get("total", 0)
        if total > 0:
            out[cid] = total
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", default="data/polymarket_test")
    ap.add_argument("--usage", default="models/bowler_phase_usage.json")
    ap.add_argument("--threshold", type=int, default=100,
                    help="A bowler is 'covered' if cumulative balls ≥ this.")
    ap.add_argument("--max-matches", type=int, default=None)
    args = ap.parse_args()

    with open(args.usage) as f:
        usage = json.load(f)

    files = sorted(Path(args.test_dir).glob("*.json"))
    if args.max_matches:
        files = files[: args.max_matches]

    # Tally: per-match-bowler-slot, was the bowler covered?
    cum_cache: dict[int, dict[str, int]] = {}
    n_bowler_slots = 0
    n_covered = 0
    unmatched_names: dict[str, int] = defaultdict(int)
    per_year_summary: dict[int, list[int]] = defaultdict(lambda: [0, 0])  # [covered, total]
    matches_with_full_coverage = 0

    for fp in files:
        with open(fp) as f:
            data = json.load(f)
        info = data.get("info", {})
        date_str = info["dates"][0]
        year = int(date_str.split("-")[0])
        registry = (info.get("registry", {}) or {}).get("people", {}) or {}

        if year not in cum_cache:
            cum_cache[year] = cumulative_for_year(usage, year)
        cum = cum_cache[year]

        # Identify "bowlers": players who actually delivered a ball in this
        # match (i.e. real bowlers, not the whole lineup — the 4-5 main
        # bowlers per side).
        bowler_names_this_match: set[str] = set()
        for inn in data.get("innings", []):
            for over in inn.get("overs", []):
                for d in over.get("deliveries", []):
                    bowler_names_this_match.add(d.get("bowler", ""))
        bowler_names_this_match.discard("")

        match_covered = True
        for name in bowler_names_this_match:
            cid = registry.get(name, name)
            n_balls = cum.get(cid, 0)
            n_bowler_slots += 1
            per_year_summary[year][1] += 1
            if n_balls >= args.threshold:
                n_covered += 1
                per_year_summary[year][0] += 1
            else:
                match_covered = False
                if n_balls == 0:
                    unmatched_names[name] += 1
        if match_covered:
            matches_with_full_coverage += 1

    print(f"Test matches scored:       {len(files)}")
    print(f"Bowler slots checked:      {n_bowler_slots:,}")
    print(f"Covered (≥{args.threshold} balls): {n_covered:,} ({n_covered/max(1,n_bowler_slots):.1%})")
    print(f"Matches with full coverage: {matches_with_full_coverage}/{len(files)} "
          f"({matches_with_full_coverage/max(1,len(files)):.1%})")
    print()
    print("Per-year coverage:")
    for year in sorted(per_year_summary):
        covered, total = per_year_summary[year]
        print(f"  {year}: {covered}/{total} ({covered/max(1,total):.1%})")

    if unmatched_names:
        print()
        print(f"Top 10 zero-history bowlers (likely new debutants or name "
              f"variants):")
        for name, ct in sorted(unmatched_names.items(), key=lambda kv: -kv[1])[:10]:
            print(f"  {name:30s} appears in {ct} test match(es)")

    gate_pass = (n_covered / max(1, n_bowler_slots)) >= 0.90
    print()
    print(f"Gate G5 (≥90% bowlers with ≥{args.threshold} balls): "
          f"{'✅ PASS' if gate_pass else '❌ FAIL'}")


if __name__ == "__main__":
    main()
