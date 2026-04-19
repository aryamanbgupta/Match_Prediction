"""Build betting_odds_polymarket.json by matching Polymarket pre-match markets to Cricsheet.

Reads:
  - /Users/aryamangupta/Projects/polymarket-cricket/data/polymarket_prematch_odds.json
  - data/t20s_json/*.json

Writes:
  - betting_odds_polymarket.json      (sim-eval-compatible odds file)
  - data/polymarket_test/*.json       (copies of matched Cricsheet JSONs)
  - data/polymarket_build_unmatched.json  (diagnostic: markets that didn't match)

Pipeline:
  1. Load Polymarket markets; apply filters (volume >= $1000, resolved, not low_liquidity,
     prematch prices present, date in [2025-07-01, 2026-04-16]).
  2. Normalize team names via TEAM_NAME_MAP (from docs/DATA_REFRESH_HANDOFF.md).
  3. For each filtered market, find the Cricsheet JSON with matching date + teams +
     match_type=T20 + gender=male.
  4. Emit odds file + copy matched JSONs into data/polymarket_test/.

CLI:
  uv run python scripts/build_polymarket_odds.py                   # full run
  uv run python scripts/build_polymarket_odds.py --dry-run         # counts + sample, no writes
  uv run python scripts/build_polymarket_odds.py --verify-mapping  # print team diff, exit
"""

from __future__ import annotations

import argparse
import difflib
import glob
import json
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
POLYMARKET_PATH = Path("/Users/aryamangupta/Projects/polymarket-cricket/data/polymarket_prematch_odds.json")
CRICSHEET_DIR = REPO_ROOT / "data" / "t20s_json"
OUT_ODDS_PATH = REPO_ROOT / "betting_odds_polymarket.json"
OUT_TEST_DIR = REPO_ROOT / "data" / "polymarket_test"
OUT_UNMATCHED_PATH = REPO_ROOT / "data" / "polymarket_build_unmatched.json"

TEST_WINDOW_START = datetime(2025, 7, 1)
TEST_WINDOW_END = datetime(2026, 4, 16)
MIN_VOLUME_USD = 1000.0


# ---------------------------------------------------------------------------
# Team name mapping (Polymarket → Cricsheet)
# ---------------------------------------------------------------------------

# Seeded from docs/DATA_REFRESH_HANDOFF.md. The script fails loudly if it
# encounters a Polymarket team name not covered here (use --verify-mapping to
# audit before a full run).
TEAM_NAME_MAP: dict[str, str] = {
    # International
    "USA": "United States of America",
    "UAE": "United Arab Emirates",
    "New Guinea": "Papua New Guinea",
    "Lanka": "Sri Lanka",
    "Kong": "Hong Kong",
    "Hong Kong, China": "Hong Kong",
    # IPL franchises
    "Hyderabad": "Sunrisers Hyderabad",
    "Chennai": "Chennai Super Kings",
    "Mumbai": "Mumbai Indians",
    "Kolkata": "Kolkata Knight Riders",
    "Delhi": "Delhi Capitals",
    "Lucknow": "Lucknow Super Giants",
    "Gujarat": "Gujarat Titans",
    "Punjab": "Punjab Kings",
    "Rajasthan": "Rajasthan Royals",
    "Bangalore": "Royal Challengers Bengaluru",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    # `Emirates` (alone) is the UAE national team in Polymarket — the ILT20
    # franchise is always written as `MI Emirates`, which already matches Cricsheet.
    "Emirates": "United Arab Emirates",
    # Kashmir Premier League
    "Rawalpindi Pindiz": "Rawalpindiz",
}


def normalize_team(name: str) -> str:
    """Strip whitespace and apply TEAM_NAME_MAP. Returns original (stripped) if unmapped."""
    s = (name or "").strip()
    return TEAM_NAME_MAP.get(s, s)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_polymarket_markets(path: Path) -> list[dict]:
    with open(path) as f:
        blob = json.load(f)
    return blob.get("matches", [])


def load_cricsheet_index(match_dir: Path) -> dict:
    """Build date-indexed Cricsheet catalog. Filters to T20 + male at load time.

    Returns {date_str: [{'path', 'teams', 'venue', 'event_name', 'winner'} ...]}.
    """
    index: dict[str, list[dict]] = defaultdict(list)
    skipped_format = skipped_gender = skipped_error = 0
    files = sorted(glob.glob(str(match_dir / "*.json")))
    for path in files:
        try:
            with open(path) as f:
                d = json.load(f)
        except Exception:
            skipped_error += 1
            continue
        info = d.get("info") or {}
        if info.get("match_type") != "T20":
            skipped_format += 1
            continue
        if info.get("gender") != "male":
            skipped_gender += 1
            continue
        dates = info.get("dates") or []
        teams = info.get("teams") or []
        if not dates or len(teams) < 2:
            continue
        event = info.get("event") or {}
        event_name = event.get("name", "") if isinstance(event, dict) else ""
        outcome = info.get("outcome") or {}
        winner = outcome.get("winner")
        index[dates[0]].append({
            "path": path,
            "teams": [t.strip() for t in teams],
            "venue": info.get("venue", "Unknown"),
            "event_name": event_name,
            "winner": winner,
        })
    print(
        f"Cricsheet index: {sum(len(v) for v in index.values()):,} T20-male matches "
        f"across {len(index):,} dates "
        f"(skipped {skipped_format:,} non-T20, {skipped_gender:,} non-male, {skipped_error:,} errors)"
    )
    return index


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def filter_markets(markets: list[dict]) -> tuple[list[dict], dict[str, int]]:
    """Apply Polymarket-side filters in order, returning survivors + per-stage counts."""
    stats: dict[str, int] = {"initial": len(markets)}
    stage = markets

    stage = [m for m in stage if (m.get("volume_usd") or 0) >= MIN_VOLUME_USD]
    stats["after_volume"] = len(stage)

    stage = [m for m in stage if not m.get("low_liquidity")]
    stats["after_liquidity"] = len(stage)

    stage = [m for m in stage if m.get("winner")]
    stats["after_resolved"] = len(stage)

    stage = [
        m for m in stage
        if m.get("prematch_price_team1") is not None and m.get("prematch_price_team2") is not None
    ]
    stats["after_prices"] = len(stage)

    def _in_window(m: dict) -> bool:
        try:
            dt = datetime.strptime(m["date"], "%Y-%m-%d")
        except (ValueError, KeyError):
            return False
        return TEST_WINDOW_START <= dt <= TEST_WINDOW_END

    stage = [m for m in stage if _in_window(m)]
    stats["after_date_window"] = len(stage)

    return stage, stats


# ---------------------------------------------------------------------------
# Verification modes
# ---------------------------------------------------------------------------

def verify_team_mapping(markets: list[dict], cricsheet_index: dict) -> None:
    """Print Polymarket team names that won't resolve against Cricsheet. Exits without writes."""
    filtered, stats = filter_markets(markets)
    print("\nFilter pipeline:")
    for k, v in stats.items():
        print(f"  {k:>22}: {v:>6,}")

    # Collect Cricsheet team names in the test window
    cricsheet_teams: set[str] = set()
    for date_str, entries in cricsheet_index.items():
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            continue
        if not (TEST_WINDOW_START <= dt <= TEST_WINDOW_END):
            continue
        for entry in entries:
            cricsheet_teams.update(entry["teams"])

    poly_names: Counter = Counter()
    for m in filtered:
        poly_names[(m.get("team1") or "").strip()] += 1
        poly_names[(m.get("team2") or "").strip()] += 1

    unmapped: list[tuple[str, int]] = []
    for raw, count in poly_names.most_common():
        mapped = normalize_team(raw)
        if mapped not in cricsheet_teams:
            unmapped.append((raw, count))

    print(f"\nDistinct Polymarket names in window: {len(poly_names)}")
    print(f"  Unmapped (mapped target not in Cricsheet team set): {len(unmapped)}")

    if not unmapped:
        print("\nAll Polymarket names resolve cleanly. Safe to run.")
        return

    print("\nUnmapped names with top-3 Cricsheet candidates:")
    cricsheet_team_list = sorted(cricsheet_teams)
    for raw, count in unmapped:
        mapped = normalize_team(raw)
        suggestions = difflib.get_close_matches(mapped, cricsheet_team_list, n=3, cutoff=0.55)
        print(f"  {count:>4}x  {raw!r:<35}  →  {mapped!r:<35}  suggestions: {suggestions}")


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def match_markets(markets: list[dict], cricsheet_index: dict) -> tuple[list[dict], list[dict]]:
    """For each market, find the Cricsheet entry matching (date, team-set).

    Returns (matched, unmatched). `matched` entries include both the Polymarket
    market and the resolved Cricsheet record.
    """
    matched: list[dict] = []
    unmatched: list[dict] = []

    for market in markets:
        date_str = market.get("date")
        raw_t1 = (market.get("team1") or "").strip()
        raw_t2 = (market.get("team2") or "").strip()
        mapped_t1 = normalize_team(raw_t1)
        mapped_t2 = normalize_team(raw_t2)

        candidates = cricsheet_index.get(date_str, [])
        hits = [
            c for c in candidates
            if {mapped_t1, mapped_t2} == {c["teams"][0], c["teams"][1]}
        ]

        if len(hits) == 1:
            matched.append({"market": market, "cricsheet": hits[0],
                            "mapped_team1": mapped_t1, "mapped_team2": mapped_t2})
        elif len(hits) == 0:
            unmatched.append({
                "reason": "no_cricsheet_match",
                "date": date_str,
                "poly_team1": raw_t1, "poly_team2": raw_t2,
                "mapped_team1": mapped_t1, "mapped_team2": mapped_t2,
                "event_slug": market.get("event_slug"),
            })
        else:
            # Tie-break on event name matching tournament substring
            tournament = (market.get("tournament") or "").lower()
            preferred = [c for c in hits if tournament and tournament in c["event_name"].lower()]
            if len(preferred) == 1:
                matched.append({"market": market, "cricsheet": preferred[0],
                                "mapped_team1": mapped_t1, "mapped_team2": mapped_t2})
            else:
                unmatched.append({
                    "reason": "multiple_cricsheet_matches",
                    "date": date_str,
                    "poly_team1": raw_t1, "poly_team2": raw_t2,
                    "mapped_team1": mapped_t1, "mapped_team2": mapped_t2,
                    "event_slug": market.get("event_slug"),
                    "candidate_paths": [c["path"] for c in hits],
                })

    return matched, unmatched


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def build_match_id(date_str: str, team1: str, team2: str, venue: str) -> str:
    return f"{date_str}_{team1}_{team2}_{venue}".replace(" ", "_")


def build_odds_entry(matched: dict) -> dict | None:
    """Construct a sim-eval-compatible odds entry. Returns None if winner disagreement."""
    m = matched["market"]
    cric = matched["cricsheet"]
    date_str = m["date"]

    # Use Cricsheet's canonical team ORDER (preserves info.teams order consumed elsewhere)
    cric_team1, cric_team2 = cric["teams"][0], cric["teams"][1]
    mapped_t1, mapped_t2 = matched["mapped_team1"], matched["mapped_team2"]

    # Polymarket prices are for (team1, team2) in Polymarket's order. Align to Cricsheet order.
    p1 = m["prematch_price_team1"]
    p2 = m["prematch_price_team2"]
    if mapped_t1 == cric_team1 and mapped_t2 == cric_team2:
        price_for_t1, price_for_t2 = p1, p2
    elif mapped_t1 == cric_team2 and mapped_t2 == cric_team1:
        price_for_t1, price_for_t2 = p2, p1
    else:
        return None  # unreachable — match() already verified set equality

    def _dec(price: float) -> float:
        return round(1.0 / price, 4) if price and price > 0 else 0.0

    venue = cric["venue"]
    entry = {
        "match_id": build_match_id(date_str, cric_team1, cric_team2, venue),
        "date": date_str,
        "team1": cric_team1,
        "team2": cric_team2,
        "venue": venue,
        "actual_winner": cric["winner"],  # Cricsheet is authoritative
        "odds": {
            "winner": {
                cric_team1: _dec(price_for_t1),
                cric_team2: _dec(price_for_t2),
                "timestamp": m.get("price_timestamp"),
            },
        },
        "source": "polymarket",
        "polymarket_event_slug": m.get("event_slug"),
        "polymarket_volume_usd": m.get("volume_usd"),
        "tournament": m.get("tournament"),
    }
    return entry


def write_outputs(matched: list[dict], unmatched: list[dict]) -> None:
    OUT_TEST_DIR.mkdir(parents=True, exist_ok=True)

    # Dedup by match_id. Polymarket runs each cricket fixture as two separate
    # binary YES/NO markets ("Will Team1 win?" and "Will Team2 win?"), so the
    # upstream extractor emits two records per fixture — one per binary market —
    # each carrying its own `winner` label and its own `prematch_price_team*`
    # orientation. Beyond the YES/NO split, the upstream "prematch" snapshots
    # occasionally contain in-play or post-match prices (top-side probability
    # near 1.0) — those are obviously not prematch even though they're labelled
    # as such. Naive highest-volume tiebreak selects them and inverts the
    # apparent edge on dozens of fixtures.
    #
    # Tiebreak, in priority order:
    #   1. plausible: max(prematch_price_team1, prematch_price_team2) <= 0.92
    #      (rejects in-play snapshots — genuine prematch T20 favourites rarely
    #      exceed this, so the cutoff trades a handful of lopsided legitimate
    #      prematch markets for discarding clearly-contaminated ones)
    #   2. `winner` matches the authoritative Cricsheet outcome
    #   3. highest volume
    #
    # When both siblings are implausible we still keep the best-scoring one
    # rather than dropping the fixture — log-loss from a noisy market price is
    # better than losing the eval match entirely.
    PLAUSIBLE_TOP_P = 0.92
    best_by_match: dict[str, tuple[dict, dict]] = {}
    winner_disagreements: list[dict] = []
    dup_dropped = 0

    def score(m: dict) -> tuple[int, int, float]:
        poly_winner = m["market"].get("winner")
        mapped = normalize_team(poly_winner) if poly_winner else None
        cric = m["cricsheet"]["winner"]
        matches_cric = 1 if (mapped and cric and mapped == cric) else 0
        p1 = m["market"].get("prematch_price_team1")
        p2 = m["market"].get("prematch_price_team2")
        if p1 is not None and p2 is not None:
            plausible = 1 if max(p1, p2) <= PLAUSIBLE_TOP_P else 0
        else:
            plausible = 0
        vol = m["market"].get("volume_usd") or 0
        return (plausible, matches_cric, vol)

    for m in matched:
        entry = build_odds_entry(m)
        if entry is None:
            continue
        poly_winner = m["market"].get("winner")
        mapped_winner = normalize_team(poly_winner) if poly_winner else None
        cric_winner = m["cricsheet"]["winner"]
        if mapped_winner and cric_winner and mapped_winner != cric_winner:
            winner_disagreements.append({
                "date": m["market"]["date"],
                "poly_winner": poly_winner,
                "mapped_poly_winner": mapped_winner,
                "cricsheet_winner": cric_winner,
                "event_slug": m["market"].get("event_slug"),
            })
        mid = entry["match_id"]
        if mid in best_by_match:
            prev_m, _ = best_by_match[mid]
            if score(m) <= score(prev_m):
                dup_dropped += 1
                continue
            dup_dropped += 1  # previous entry will be replaced
        best_by_match[mid] = (m, entry)

    odds_entries: list[dict] = []
    copied = 0
    already_present = 0
    residual_disagreements: list[dict] = []
    for mid, (m, entry) in best_by_match.items():
        odds_entries.append(entry)
        poly_winner = m["market"].get("winner")
        mapped = normalize_team(poly_winner) if poly_winner else None
        cric = m["cricsheet"]["winner"]
        if mapped and cric and mapped != cric:
            residual_disagreements.append({
                "match_id": mid,
                "poly_winner": poly_winner,
                "cricsheet_winner": cric,
                "event_slug": m["market"].get("event_slug"),
            })
        src = Path(m["cricsheet"]["path"])
        dst = OUT_TEST_DIR / src.name
        if dst.exists():
            already_present += 1
        else:
            shutil.copy2(src, dst)
            copied += 1

    output = {
        "source": "polymarket",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "total_matches": len(odds_entries),
        "filters": {
            "min_volume_usd": MIN_VOLUME_USD,
            "test_window_start": TEST_WINDOW_START.strftime("%Y-%m-%d"),
            "test_window_end": TEST_WINDOW_END.strftime("%Y-%m-%d"),
            "requires_resolved_winner": True,
            "requires_match_type_t20_male": True,
        },
        "matches": odds_entries,
    }
    tmp = OUT_ODDS_PATH.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(output, f, indent=2, default=str)
    tmp.replace(OUT_ODDS_PATH)

    unmatched_report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "total_unmatched": len(unmatched),
        "total_winner_disagreements_raw": len(winner_disagreements),
        "total_winner_disagreements_after_dedup": len(residual_disagreements),
        "unmatched": unmatched,
        "winner_disagreements_raw": winner_disagreements,
        "winner_disagreements_after_dedup": residual_disagreements,
    }
    tmp = OUT_UNMATCHED_PATH.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(unmatched_report, f, indent=2, default=str)
    tmp.replace(OUT_UNMATCHED_PATH)

    print(f"\nOdds file:       {OUT_ODDS_PATH}  ({len(odds_entries):,} matches)")
    print(f"Test dir:        {OUT_TEST_DIR}  (+{copied} copied, {already_present} already present)")
    print(f"Deduped by match_id: {dup_dropped} duplicate market(s) dropped")
    print(f"Unmatched report {OUT_UNMATCHED_PATH}  ({len(unmatched):,} unmatched, "
          f"{len(winner_disagreements):,} raw disagreements, "
          f"{len(residual_disagreements):,} residual after dedup)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Build Polymarket odds file by matching to Cricsheet")
    parser.add_argument("--dry-run", action="store_true", help="Report counts + sample, no writes")
    parser.add_argument("--verify-mapping", action="store_true",
                        help="Print Polymarket→Cricsheet team name diff and exit")
    args = parser.parse_args()

    if not POLYMARKET_PATH.exists():
        print(f"ERROR: Polymarket file not found at {POLYMARKET_PATH}", file=sys.stderr)
        sys.exit(1)
    if not CRICSHEET_DIR.exists():
        print(f"ERROR: Cricsheet dir not found at {CRICSHEET_DIR}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading Polymarket markets from {POLYMARKET_PATH} ...")
    markets = load_polymarket_markets(POLYMARKET_PATH)
    print(f"  {len(markets):,} markets")

    print(f"Indexing Cricsheet matches in {CRICSHEET_DIR} ...")
    cricsheet_index = load_cricsheet_index(CRICSHEET_DIR)

    if args.verify_mapping:
        verify_team_mapping(markets, cricsheet_index)
        return

    filtered, stats = filter_markets(markets)
    print("\nFilter pipeline:")
    for k, v in stats.items():
        print(f"  {k:>22}: {v:>6,}")

    print("\nMatching Polymarket → Cricsheet ...")
    matched, unmatched = match_markets(filtered, cricsheet_index)
    print(f"  matched:    {len(matched):,}")
    print(f"  unmatched:  {len(unmatched):,}")

    by_reason = Counter(u["reason"] for u in unmatched)
    for reason, count in by_reason.most_common():
        print(f"    {reason:>28}: {count:>4}")

    if matched:
        print("\nSample matched entries:")
        for m in matched[:3]:
            mk = m["market"]
            cric = m["cricsheet"]
            print(f"  {mk['date']}  {mk['team1']} vs {mk['team2']}  →  "
                  f"{cric['teams'][0]} vs {cric['teams'][1]}  @ {cric['venue']}  "
                  f"(vol ${mk['volume_usd']:,.0f})")

    if args.dry_run:
        print("\nDry run — no writes.")
        return

    write_outputs(matched, unmatched)


if __name__ == "__main__":
    main()
