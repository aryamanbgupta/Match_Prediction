"""Build the GOLDEN polymarket odds set, isolated from the iteration set.

This is a thin wrapper around build_polymarket_odds.py that:
  - Reads the new fresh-snapshot polymarket file (cutoff 2026-04-17 onward)
  - Pairs against data/golden/t20s_json/ (extracted by extract_golden_cricsheet.py),
    UNIONED with the existing data/t20s_json/ pool to maximize match coverage —
    polymarket markets pre-cutoff would have already had cricsheet entries.
  - Writes everything under data/golden/* so nothing in the existing pipeline
    is touched.

Reuses load_polymarket_markets, filter_markets, match_markets, build_odds_entry
unmodified. Only paths and the date window are overridden.

Usage:
    uv run python scripts/build_polymarket_odds_golden.py                 # full
    uv run python scripts/build_polymarket_odds_golden.py --dry-run       # counts
    uv run python scripts/build_polymarket_odds_golden.py --verify-mapping
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import build_polymarket_odds as base  # noqa: E402


# === GOLDEN-ONLY OVERRIDES ====================================================
GOLDEN_POLYMARKET_PATH = Path(
    "/Users/aryamangupta/Projects/polymarket-cricket/data/"
    "polymarket_prematch_odds_2026-05-09.json"
)
GOLDEN_DIR = REPO_ROOT / "data" / "golden"
GOLDEN_CRICSHEET_DIR = GOLDEN_DIR / "t20s_json"
LIVE_CRICSHEET_DIR = REPO_ROOT / "data" / "t20s_json"  # union for matching only
GOLDEN_OUT_ODDS = GOLDEN_DIR / "betting_odds_golden.json"
GOLDEN_OUT_TEST_DIR = GOLDEN_DIR / "polymarket_test"
GOLDEN_OUT_UNMATCHED = GOLDEN_DIR / "build_unmatched.json"

GOLDEN_WINDOW_START = datetime(2026, 4, 17)
GOLDEN_WINDOW_END = datetime(2099, 12, 31)  # open-ended; capped only by file
GOLDEN_MIN_VOLUME_USD = 1000.0
# ==============================================================================


def _patch_constants() -> None:
    """Override module-level constants in build_polymarket_odds for this run.
    Patching is local to this process — the underlying module file is not edited.
    """
    base.POLYMARKET_PATH = GOLDEN_POLYMARKET_PATH
    base.OUT_ODDS_PATH = GOLDEN_OUT_ODDS
    base.OUT_TEST_DIR = GOLDEN_OUT_TEST_DIR
    base.OUT_UNMATCHED_PATH = GOLDEN_OUT_UNMATCHED
    base.TEST_WINDOW_START = GOLDEN_WINDOW_START
    base.TEST_WINDOW_END = GOLDEN_WINDOW_END
    base.MIN_VOLUME_USD = GOLDEN_MIN_VOLUME_USD
    # CRICSHEET_DIR is consumed by load_cricsheet_index via base.CRICSHEET_DIR;
    # we don't touch it here — we pass our golden dir directly into the loader.


def _union_cricsheet_index() -> dict:
    """Build a single date-indexed Cricsheet catalog spanning both the live
    pool (data/t20s_json/) and the golden pool (data/golden/t20s_json/).

    The matcher only reads from this index — it never modifies either pool —
    so unioning cannot leak golden matches back into training.
    """
    print(f"Indexing live Cricsheet pool: {LIVE_CRICSHEET_DIR}")
    live_index = base.load_cricsheet_index(LIVE_CRICSHEET_DIR)
    print(f"Indexing golden Cricsheet pool: {GOLDEN_CRICSHEET_DIR}")
    golden_index = base.load_cricsheet_index(GOLDEN_CRICSHEET_DIR)

    merged: dict[str, list[dict]] = defaultdict(list)
    for d, entries in live_index.items():
        merged[d].extend(entries)
    added_dates = 0
    added_entries = 0
    for d, entries in golden_index.items():
        if d not in merged:
            added_dates += 1
        merged[d].extend(entries)
        added_entries += len(entries)
    print(f"  union: {sum(len(v) for v in merged.values()):,} entries across "
          f"{len(merged):,} dates  (golden contributed +{added_entries} entries, "
          f"+{added_dates} new dates)")
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Counts + sample, no writes.")
    parser.add_argument("--verify-mapping", action="store_true",
                        help="Print Polymarket→Cricsheet team-name diff in window, exit.")
    args = parser.parse_args()

    _patch_constants()

    if not GOLDEN_POLYMARKET_PATH.exists():
        print(f"ERROR: golden polymarket file not found: {GOLDEN_POLYMARKET_PATH}",
              file=sys.stderr)
        sys.exit(1)
    if not GOLDEN_CRICSHEET_DIR.exists():
        print(f"ERROR: golden cricsheet dir not found: {GOLDEN_CRICSHEET_DIR}",
              file=sys.stderr)
        print("  Run scripts/extract_golden_cricsheet.py first.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading polymarket markets from {GOLDEN_POLYMARKET_PATH}")
    markets = base.load_polymarket_markets(GOLDEN_POLYMARKET_PATH)
    print(f"  {len(markets):,} markets in source file")

    cricsheet_index = _union_cricsheet_index()

    if args.verify_mapping:
        base.verify_team_mapping(markets, cricsheet_index)
        return

    filtered, stats = base.filter_markets(markets)
    print("\nFilter pipeline:")
    for k, v in stats.items():
        print(f"  {k:>22}: {v:>6,}")

    print("\nMatching polymarket → Cricsheet (live ∪ golden) ...")
    matched, unmatched = base.match_markets(filtered, cricsheet_index)
    print(f"  matched:    {len(matched):,}")
    print(f"  unmatched:  {len(unmatched):,}")

    by_reason = Counter(u["reason"] for u in unmatched)
    for reason, count in by_reason.most_common():
        print(f"    {reason:>28}: {count:>4}")

    # Diagnostic: how many matched entries resolve to the GOLDEN cricsheet pool
    # vs the live pool. A match that lands in the live pool was already there
    # before the golden window, so it shouldn't land in the golden eval set.
    golden_paths = {str(p) for p in GOLDEN_CRICSHEET_DIR.glob("*.json")}
    n_golden = sum(
        1 for m in matched if m["cricsheet"]["path"] in golden_paths)
    n_live = len(matched) - n_golden
    print(f"\n  matched-from-golden pool:  {n_golden}")
    print(f"  matched-from-live pool:    {n_live}  "
          "(should be ~0 with the 2026-04-17 cutoff)")

    if matched:
        print("\nSample matched entries (first 3):")
        for m in matched[:3]:
            mk = m["market"]
            cric = m["cricsheet"]
            print(f"  {mk['date']}  {mk['team1']} vs {mk['team2']}  "
                  f"→  {cric['teams'][0]} vs {cric['teams'][1]}  @ {cric['venue']}  "
                  f"(vol ${mk['volume_usd']:,.0f})")

    if args.dry_run:
        print("\nDry run — no writes.")
        return

    base.write_outputs(matched, unmatched)
    print(f"\nGolden artifacts written under {GOLDEN_DIR}")


if __name__ == "__main__":
    main()
