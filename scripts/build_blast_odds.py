"""Build the T20 Blast polymarket odds set — the betting analogue of the
Blast-golden cricsheet pool, mirroring build_polymarket_odds_golden.py.

Thin wrapper around build_polymarket_odds.py that:
  - Reads the Blast pre-match odds file (data/polymarket_t20blast_prematch_odds_*.json
    from the polymarket-cricket repo). It already ships the same `matches[]`
    schema the base loader expects (volume_usd, winner, prematch_price_team1/2,
    low_liquidity), and the upstream extractor emits one record per event,
    having already picked that event's head-to-head market. That claim is no
    longer taken on trust: since the 2026-08-05 toss-market fix the shared
    builder re-derives each record's Gamma market identity
    (`market_volume_exact` resolves the Blast capture, which persists
    market-level volume) and drops anything not structurally head-to-head.
  - Adds the one Blast-specific name alias the base TEAM_NAME_MAP is missing:
    Polymarket says "Warwickshire", Cricsheet's T20 brand is "Birmingham Bears".
  - Pairs against data/golden_blast/t20s_json/ (extracted by
    extract_blast_golden.py). Order-independent (date, team-set) matching +
    match_id dedup are inherited unchanged from the base module.
  - Writes everything under data/golden_blast/* so nothing else is touched,
    defaulting to the _v2 paths so the pre-fix evidence files survive.

Usage:
    uv run python scripts/build_blast_odds.py
    uv run python scripts/build_blast_odds.py --dry-run
    uv run python scripts/build_blast_odds.py --verify-mapping
    uv run python scripts/build_blast_odds.py --out-odds /tmp/blast_experiment.json
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import build_polymarket_odds as base  # noqa: E402

# === BLAST-ONLY OVERRIDES =====================================================
BLAST_ODDS_PATH = Path(
    "/Users/aryamangupta/Projects/polymarket-cricket/data/"
    "polymarket_t20blast_prematch_odds_2026-06-04.json"
)
BLAST_DIR = REPO_ROOT / "data" / "golden_blast"
BLAST_CRICSHEET_DIR = BLAST_DIR / "t20s_json"
# Defaults write to the _v2 (post-toss-fix) paths, mirroring
# build_polymarket_odds{,_golden}.py. The pre-2026-08-05
# data/golden_blast/betting_odds_blast.json + polymarket_test/ are frozen
# evidence of what was shipped under the defective market-selection rule
# (reports/market_benchmark_toss_defect_20260805.md) and must not be clobbered
# by a plain run of this script. Override with --out-odds et al.
BLAST_OUT_ODDS = BLAST_DIR / "betting_odds_blast_v2.json"
BLAST_OUT_TEST_DIR = BLAST_DIR / "polymarket_test_v2"
BLAST_OUT_UNMATCHED = BLAST_DIR / "build_unmatched_v2.json"

BLAST_WINDOW_START = datetime(2026, 4, 17)
BLAST_WINDOW_END = datetime(2099, 12, 31)
BLAST_MIN_VOLUME_USD = 1000.0

# Polymarket uses the county name; Cricsheet uses the T20 brand name.
BLAST_TEAM_ALIASES = {"Warwickshire": "Birmingham Bears"}
# ==============================================================================


def match_markets_canonical(markets: list[dict], cricsheet_index: dict):
    """Order-independent (date, team-set) match, but canonicalize BOTH sides
    via normalize_team before comparing. Cricsheet is internally inconsistent
    on Warwickshire (some files say "Birmingham Bears", newer ones "Warwickshire");
    the polymarket side always says "Warwickshire". Canonicalizing both makes
    either cricsheet spelling match.

    The emitted `matched` dict keeps the RAW cricsheet entry and sets
    mapped_team1/mapped_team2 to the RAW cricsheet names in the polymarket
    team order, so base.build_odds_entry aligns prices correctly and the
    output match_id uses raw cricsheet names — identical to the parquet/
    prediction keys, so the dashboard join holds.
    """
    canon = base.normalize_team
    matched, unmatched = [], []
    for market in markets:
        date_str = market.get("date")
        raw_t1 = (market.get("team1") or "").strip()
        raw_t2 = (market.get("team2") or "").strip()
        c1, c2 = canon(raw_t1), canon(raw_t2)
        hits = [
            c for c in cricsheet_index.get(date_str, [])
            if {c1, c2} == {canon(c["teams"][0]), canon(c["teams"][1])}
        ]
        if len(hits) == 1:
            cric_teams = hits[0]["teams"]
            # Raw cricsheet name corresponding to each polymarket team.
            raw_for_poly1 = cric_teams[0] if canon(cric_teams[0]) == c1 else cric_teams[1]
            raw_for_poly2 = cric_teams[0] if canon(cric_teams[0]) == c2 else cric_teams[1]
            matched.append({"market": market, "cricsheet": hits[0],
                            "mapped_team1": raw_for_poly1,
                            "mapped_team2": raw_for_poly2})
        else:
            unmatched.append({
                "reason": "no_cricsheet_match" if not hits else "multiple_cricsheet_matches",
                "date": date_str, "poly_team1": raw_t1, "poly_team2": raw_t2,
                "event_slug": market.get("event_slug"),
            })
    return matched, unmatched


def _patch_constants() -> None:
    base.POLYMARKET_PATH = BLAST_ODDS_PATH
    base.OUT_ODDS_PATH = BLAST_OUT_ODDS
    base.OUT_TEST_DIR = BLAST_OUT_TEST_DIR
    base.OUT_UNMATCHED_PATH = BLAST_OUT_UNMATCHED
    base.TEST_WINDOW_START = BLAST_WINDOW_START
    base.TEST_WINDOW_END = BLAST_WINDOW_END
    base.MIN_VOLUME_USD = BLAST_MIN_VOLUME_USD
    # In-place mutation so the module-level normalize_team() sees the alias.
    base.TEAM_NAME_MAP.update(BLAST_TEAM_ALIASES)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Counts + sample, no writes.")
    parser.add_argument("--verify-mapping", action="store_true",
                        help="Print Polymarket→Cricsheet team-name diff, exit.")
    base.add_output_arguments(parser)
    args = parser.parse_args()

    _patch_constants()
    base.apply_output_overrides(args)

    if not BLAST_ODDS_PATH.exists():
        print(f"ERROR: Blast odds file not found: {BLAST_ODDS_PATH}", file=sys.stderr)
        sys.exit(1)
    if not BLAST_CRICSHEET_DIR.exists():
        print(f"ERROR: Blast cricsheet dir not found: {BLAST_CRICSHEET_DIR}",
              file=sys.stderr)
        print("  Run scripts/extract_blast_golden.py first.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading polymarket markets from {BLAST_ODDS_PATH}")
    markets = base.load_polymarket_markets(BLAST_ODDS_PATH)
    print(f"  {len(markets):,} markets in source file")

    print(f"Indexing Blast cricsheet pool: {BLAST_CRICSHEET_DIR}")
    cricsheet_index = base.load_cricsheet_index(BLAST_CRICSHEET_DIR)

    if args.verify_mapping:
        base.verify_team_mapping(markets, cricsheet_index)
        return

    filtered, stats = base.filter_markets(markets)
    print("\nFilter pipeline:")
    for k, v in stats.items():
        print(f"  {k:>22}: {v:>6,}")

    print("\nMatching polymarket → Cricsheet (Blast golden pool, canonical) ...")
    matched, unmatched = match_markets_canonical(filtered, cricsheet_index)
    print(f"  matched:    {len(matched):,}")
    print(f"  unmatched:  {len(unmatched):,}")
    by_reason = Counter(u["reason"] for u in unmatched)
    for reason, count in by_reason.most_common():
        print(f"    {reason:>28}: {count:>4}")

    if matched:
        print("\nSample matched entries (first 3):")
        for m in matched[:3]:
            mk, cric = m["market"], m["cricsheet"]
            print(f"  {mk['date']}  {mk['team1']} vs {mk['team2']}  →  "
                  f"{cric['teams'][0]} vs {cric['teams'][1]}  @ {cric['venue']}  "
                  f"(vol ${mk['volume_usd']:,.0f})")

    if args.dry_run:
        print("\nDry run — no writes.")
        return

    base.write_outputs(matched, unmatched,
                       timestamp_guard=args.timestamp_guard,
                       restrict_to=(
                           base.load_manifest_identities(args.restrict_to_manifest)
                           if args.restrict_to_manifest else None))
    print(f"\nBlast odds artifacts written under {BLAST_DIR}")


if __name__ == "__main__":
    main()
