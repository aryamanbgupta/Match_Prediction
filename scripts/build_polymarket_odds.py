"""Build betting_odds_polymarket.json by matching Polymarket pre-match markets to Cricsheet.

Reads:
  - /Users/aryamangupta/Projects/polymarket-cricket/data/polymarket_prematch_odds.json
  - data/polymarket_market_catalog.json  (Gamma event -> markets snapshot; see
    docs note in `load_market_catalog`)
  - data/t20s_json/*.json

Writes (defaults; the pre-2026-08-05 non-_v2 paths are frozen evidence and are
never written by default — see the OUT_* constants below):
  - betting_odds_polymarket_v2.json      (sim-eval-compatible odds file)
  - data/polymarket_test_v2/*.json       (copies of matched Cricsheet JSONs)
  - data/polymarket_build_unmatched_v2.json  (diagnostic: markets that didn't match)

Pipeline:
  1. Load Polymarket markets; resolve each capture record to the Gamma market that
     produced it and label it head-to-head / not (see "Market identity" below).
  2. Apply filters (volume >= $1000, resolved, not low_liquidity, prematch prices
     present, date in [2025-07-01, 2026-04-16]).
  3. Normalize team names via TEAM_NAME_MAP (from docs/DATA_REFRESH_HANDOFF.md).
  4. For each filtered market, find the Cricsheet JSON with matching date + teams +
     match_type=T20 + gender=male.
  5. Select ONE market per fixture with a structural, outcome-blind rule, then emit
     the odds file + copy matched JSONs into data/polymarket_test/.

Market identity (why step 1 exists):
  Each Gamma cricket event carries several binary markets — the head-to-head
  winner market, "Who wins the toss?", "Completed match?" — and the upstream
  prematch capture drops the market question/id, emitting one bare record per
  market. Selecting between those records therefore requires re-attaching the
  market identity from the Gamma catalog before anything else can be decided.

CLI:
  uv run python scripts/build_polymarket_odds.py                   # full run
  uv run python scripts/build_polymarket_odds.py --dry-run         # counts + sample, no writes
  uv run python scripts/build_polymarket_odds.py --verify-mapping  # print team diff, exit
  uv run python scripts/build_polymarket_odds.py \
      --out-odds /tmp/odds_experiment.json                         # write elsewhere
"""

from __future__ import annotations

import argparse
import difflib
import glob
import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from identity_maps import canonicalize_venue
from match_identity import (
    build_display_match_id,
    identity_contract,
    new_match_identity,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
POLYMARKET_PATH = Path("/Users/aryamangupta/Projects/polymarket-cricket/data/polymarket_prematch_odds.json")
CRICSHEET_DIR = REPO_ROOT / "data" / "t20s_json"
# Defaults write to the _v2 (post-toss-fix) paths, NOT to the pre-fix files
# shipped before 2026-08-05. `betting_odds_polymarket.json` and
# `data/polymarket_test/` are frozen evidence of what was shipped under the
# defective selection rule (reports/market_benchmark_toss_defect_20260805.md);
# a default that pointed at them meant any plain `python build_polymarket_odds.py`
# silently destroyed that evidence. Pass --out-odds/--out-test-dir/--out-unmatched
# to write somewhere else.
OUT_ODDS_PATH = REPO_ROOT / "betting_odds_polymarket_v2.json"
OUT_TEST_DIR = REPO_ROOT / "data" / "polymarket_test_v2"
OUT_UNMATCHED_PATH = REPO_ROOT / "data" / "polymarket_build_unmatched_v2.json"

# Gamma snapshot of every event id referenced by the prematch captures, and the
# ordered market list the legacy capture iterated over. Both are regenerable;
# see reports/market_benchmark_toss_defect_20260805.md § "Reproducing".
MARKET_CATALOG_PATH = REPO_ROOT / "data" / "polymarket_market_catalog.json"
RAW_MARKET_INDEX_PATH = Path(
    "/Users/aryamangupta/Projects/polymarket-cricket/data/polymarket_match_events.json"
)

TEST_WINDOW_START = datetime(2025, 7, 1)
TEST_WINDOW_END = datetime(2026, 4, 16)
MIN_VOLUME_USD = 1000.0

SELECTION_RULE_VERSION = "h2h_identity_outcome_blind_v2"
MONEYLINE_TYPE = "moneyline"
# `report` records how many selected rows carry a price stamped at/after the
# scheduled start without changing n; `enforce` drops them. Off by default so
# fixture counts are decided by the head-to-head rule alone.
TIMESTAMP_GUARD_MODES = ("off", "report", "enforce")


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
    """Load capture records and attach the market identity each one came from."""
    with open(path) as f:
        blob = json.load(f)
    markets = blob.get("matches", [])
    attach_market_identity(markets)
    return markets


# ---------------------------------------------------------------------------
# Market identity
# ---------------------------------------------------------------------------

def load_market_catalog(path: Path | None = None) -> dict[str, dict]:
    """Gamma `event_id -> {event_title, markets:[...]}` snapshot.

    Regenerate with a batched `GET /events?id=...` pull over every event id in
    the capture files; markets are closed, so their question/type/volume are
    frozen and the snapshot is stable.
    """
    path = Path(path or MARKET_CATALOG_PATH)
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f).get("events", {})


def load_raw_market_index(path: Path | None = None) -> list[dict]:
    """The ordered market list the legacy prematch capture iterated over.

    The capture appends one output record per surviving input market, in order,
    so this list lets a record be traced back to its exact market even though
    the capture itself persisted neither market id nor question.
    """
    path = Path(path or RAW_MARKET_INDEX_PATH)
    if not path.exists():
        return []
    with open(path) as f:
        rows = json.load(f)
    return [
        r for r in rows
        if r.get("closed") and (r.get("volume") or 0) > 0
        and r.get("token_id1") and r.get("token_id2")
    ]


def _raw_index_winner(row: dict) -> str | None:
    if row.get("price1") == 1.0:
        return row.get("team1")
    if row.get("price2") == 1.0:
        return row.get("team2")
    return None


def _align_to_raw_index(markets: list[dict], raw_index: list[dict]) -> dict[int, dict]:
    """Monotone alignment of capture records onto the ordered raw market list.

    The capture is an order-preserving subsequence of `raw_index`, so a single
    forward scan is exact even when two markets of one event are otherwise
    indistinguishable. Records that do not align are simply left unresolved.
    """
    hits: dict[int, dict] = {}
    cursor = 0
    for idx, record in enumerate(markets):
        key = (
            str(record.get("event_id")),
            record.get("team1"),
            record.get("team2"),
            record.get("winner"),
        )
        probe = cursor
        while probe < len(raw_index):
            row = raw_index[probe]
            row_key = (
                str(row.get("event_id")),
                row.get("team1"),
                row.get("team2"),
                _raw_index_winner(row),
            )
            if row_key == key:
                hits[idx] = row
                cursor = probe + 1
                break
            probe += 1
    return hits


def _classify_h2h(question, title, sports_market_type) -> tuple[bool | None, str | None]:
    """Structural head-to-head test. Never consults the fixture outcome.

    Two independent signals: Gamma's explicit `sportsMarketType` and the
    repository's established `market.question == event.title` identity rule.
    Where both exist they must agree; a conflict yields no verdict rather than
    a guess.
    """
    by_identity = None
    if question is not None and title is not None:
        by_identity = str(question).strip() == str(title).strip()
    by_type = None
    if sports_market_type:
        by_type = str(sports_market_type) == MONEYLINE_TYPE
    if by_type is None and by_identity is None:
        return None, "no_h2h_evidence"
    if by_type is not None and by_identity is not None and by_type != by_identity:
        return None, "h2h_rule_conflict"
    return (by_type if by_type is not None else by_identity), None


def attach_market_identity(markets: list[dict]) -> Counter:
    """Resolve each capture record to its Gamma market and label it H2H or not.

    Resolvers, in priority order:
      1. `ordered_capture_index` — exact provenance via the raw market list.
      2. `market_volume_exact`  — a unique market in the event whose volume
         equals the record's, used for captures that persist market-level
         volume (the legacy capture persists event-level volume instead, which
         is why this cannot be the primary resolver).
    """
    catalog = load_market_catalog()
    raw_index = load_raw_market_index()
    aligned = _align_to_raw_index(markets, raw_index) if raw_index else {}

    stats: Counter = Counter()
    for idx, record in enumerate(markets):
        event = catalog.get(str(record.get("event_id"))) or {}
        event_markets = event.get("markets") or []
        ident: dict = {
            "resolver": None,
            "market_id": None,
            "market_question": None,
            "event_title": event.get("event_title"),
            "sports_market_type": None,
            "market_volume_usd": None,
            "scheduled_start_timestamp": None,
            "is_h2h": None,
            "unresolved_reason": None,
        }

        raw_row = aligned.get(idx)
        catalog_row = None
        if raw_row is not None:
            ident["resolver"] = "ordered_capture_index"
            ident["market_id"] = str(raw_row.get("market_id") or "")
            ident["market_question"] = raw_row.get("question")
            ident["event_title"] = raw_row.get("event_title") or ident["event_title"]
            catalog_row = next(
                (m for m in event_markets if m.get("market_id") == ident["market_id"]),
                None,
            )
        else:
            want = record.get("volume_usd")
            candidates = [
                m for m in event_markets
                if want is not None
                and abs((m.get("volume") or 0.0) - float(want))
                <= 1e-6 * max(1.0, abs(m.get("volume") or 0.0))
            ]
            if len(candidates) == 1:
                catalog_row = candidates[0]
                ident["resolver"] = "market_volume_exact"
                ident["market_id"] = catalog_row.get("market_id")
                ident["market_question"] = catalog_row.get("question")
            elif len(candidates) > 1:
                ident["unresolved_reason"] = "ambiguous_volume_match"
            elif not event_markets:
                ident["unresolved_reason"] = "event_not_in_catalog"
            else:
                ident["unresolved_reason"] = "no_market_match"

        if catalog_row is not None:
            ident["sports_market_type"] = catalog_row.get("sports_market_type")
            ident["market_volume_usd"] = catalog_row.get("volume")
            ident["scheduled_start_timestamp"] = catalog_row.get("game_start_time")
            if ident["market_question"] is None:
                ident["market_question"] = catalog_row.get("question")

        if ident["resolver"] is not None:
            is_h2h, reason = _classify_h2h(
                ident["market_question"], ident["event_title"],
                ident["sports_market_type"],
            )
            ident["is_h2h"] = is_h2h
            if is_h2h is None:
                ident["unresolved_reason"] = reason

        record["_market_identity"] = ident
        stats[ident["resolver"] or f"unresolved:{ident['unresolved_reason']}"] += 1
        if ident["is_h2h"] is True:
            stats["labelled_h2h"] += 1
        elif ident["is_h2h"] is False:
            stats["labelled_non_h2h"] += 1

    print(
        f"Market identity: {stats['labelled_h2h']:,} head-to-head, "
        f"{stats['labelled_non_h2h']:,} non-H2H (toss/side markets), "
        f"{len(markets) - stats['labelled_h2h'] - stats['labelled_non_h2h']:,} unresolved"
    )
    for key, count in sorted(stats.items()):
        if key.startswith("unresolved:"):
            print(f"    {key}: {count}")
    return stats


def _market_id_sort_key(market_id: str | None) -> tuple[int, int, str]:
    text = str(market_id or "")
    if text.isdigit():
        return (0, int(text), "")
    return (1, 0, text)


def selection_key(matched: dict) -> tuple:
    """Outcome-blind tiebreak between head-to-head siblings of one fixture.

    Highest market volume first, then lowest market id. Capture order is never
    consulted, and neither is the resolved winner.
    """
    ident = matched["market"].get("_market_identity") or {}
    volume = ident.get("market_volume_usd")
    if volume is None:
        volume = matched["market"].get("volume_usd") or 0.0
    return (-float(volume), _market_id_sort_key(ident.get("market_id")))


_UTC_OFFSET_SHORT = re.compile(r"([+-]\d{2})$")


def _parse_ts(raw) -> datetime | None:
    """Parse Gamma/CLOB timestamps as UTC. Returns None when unparseable.

    Gamma writes `gameStartTime` as `YYYY-MM-DD HH:MM:SS+00` — a two-digit
    offset that `fromisoformat` does not accept before 3.11 — while the
    captures write `...Z`. Both are normalized here; a bare `.replace("+00",
    "+00:00")` would corrupt an already-full `+00:00` offset.
    """
    if not raw:
        return None
    text = str(raw).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    text = _UTC_OFFSET_SHORT.sub(r"\1:00", text)
    try:
        parsed = datetime.fromisoformat(text)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def price_is_prematch(matched: dict) -> bool | None:
    """True/False when both timestamps are known, else None (cannot judge)."""
    ident = matched["market"].get("_market_identity") or {}
    start = _parse_ts(ident.get("scheduled_start_timestamp"))
    priced = _parse_ts(matched["market"].get("price_timestamp"))
    if start is None or priced is None:
        return None
    return priced < start


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
            "cricsheet_id": Path(path).stem,
            "teams": [t.strip() for t in teams],
            "venue": canonicalize_venue(info.get("venue")),
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
    """Compatibility name for the historical display-ID builder."""
    return build_display_match_id(date_str, team1, team2, venue)


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
    identity = new_match_identity(
        cric["cricsheet_id"],
        date_text=date_str,
        team1=cric_team1,
        team2=cric_team2,
        venue=venue,
    )
    entry = {
        **identity.as_fields(),
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
        # Capture-level volume, kept as-is because downstream `--min-volume`
        # slices are defined against it. Market-level volume is reported
        # separately under `market_selection`.
        "polymarket_volume_usd": m.get("volume_usd"),
        "tournament": m.get("tournament"),
    }
    ident = m.get("_market_identity")
    if ident:
        entry["market_selection"] = {
            "market_id": ident.get("market_id"),
            "market_question": ident.get("market_question"),
            "event_title": ident.get("event_title"),
            "sports_market_type": ident.get("sports_market_type"),
            "market_volume_usd": ident.get("market_volume_usd"),
            "scheduled_start_timestamp": ident.get("scheduled_start_timestamp"),
            "resolver": ident.get("resolver"),
        }
    return entry


def load_manifest_identities(path: Path) -> set[str]:
    """Every identity string an existing odds manifest keys its fixtures by.

    Used by `--restrict-to-manifest` to rebuild a frozen benchmark's own
    fixtures under a new selection rule without re-deriving its membership.
    """
    with open(path) as f:
        blob = json.load(f)
    keys: set[str] = set()
    for row in blob.get("matches", []):
        for field in ("match_id", "cricsheet_id", "display_match_id"):
            value = row.get(field)
            if value:
                keys.add(str(value))
    return keys


def _entry_identities(entry: dict) -> set[str]:
    keys = {
        str(entry[field])
        for field in ("match_id", "cricsheet_id", "display_match_id")
        if entry.get(field)
    }
    keys.add(build_display_match_id(
        entry["date"], entry["team1"], entry["team2"], entry["venue"]))
    return keys


def write_outputs(
    matched: list[dict],
    unmatched: list[dict],
    timestamp_guard: str = "report",
    restrict_to: set[str] | None = None,
) -> None:
    OUT_TEST_DIR.mkdir(parents=True, exist_ok=True)

    # Selecting one market per fixture. Each Gamma cricket event carries several
    # binary markets (head-to-head winner, "Who wins the toss?", "Completed
    # match?"), and the capture emits one bare record per market, so a fixture
    # can arrive here several times over.
    #
    # The rule is structural and outcome-blind:
    #   1. keep only markets labelled head-to-head by `_classify_h2h`;
    #   2. break ties on market volume, then on market id.
    # A fixture with no surviving head-to-head market is DROPPED with a reason
    # rather than falling back to a side market. The resolved winner is never
    # consulted: doing so makes the benchmark a function of the outcome it is
    # supposed to be scored against.
    candidates_by_match: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    dropped_fixtures: dict[str, dict] = {}
    winner_disagreements: list[dict] = []
    dup_dropped = 0
    non_h2h_seen = 0
    unresolved_seen = 0
    non_prematch_selected: list[dict] = []

    restricted_out = 0
    for m in matched:
        entry = build_odds_entry(m)
        if entry is None:
            continue
        if restrict_to is not None and not (_entry_identities(entry) & restrict_to):
            restricted_out += 1
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
        ident = m["market"].get("_market_identity") or {}
        if ident.get("is_h2h") is not True:
            if ident.get("is_h2h") is False:
                non_h2h_seen += 1
                reason = "non_h2h_market"
            else:
                unresolved_seen += 1
                reason = f"market_identity_unresolved:{ident.get('unresolved_reason')}"
            dropped_fixtures.setdefault(mid, {
                "match_id": mid,
                "date": entry["date"],
                "team1": entry["team1"],
                "team2": entry["team2"],
                "reason": reason,
                "rejected_market_question": ident.get("market_question"),
                "rejected_sports_market_type": ident.get("sports_market_type"),
                "event_slug": m["market"].get("event_slug"),
            })
            continue
        candidates_by_match[mid].append((m, entry))

    best_by_match: dict[str, tuple[dict, dict]] = {}
    for mid, options in candidates_by_match.items():
        options.sort(key=lambda pair: selection_key(pair[0]))
        best_by_match[mid] = options[0]
        dup_dropped += len(options) - 1
        dropped_fixtures.pop(mid, None)

    if timestamp_guard != "off":
        for mid, (m, entry) in list(best_by_match.items()):
            if price_is_prematch(m) is False:
                non_prematch_selected.append({
                    "match_id": mid,
                    "date": entry["date"],
                    "price_timestamp": m["market"].get("price_timestamp"),
                    "scheduled_start_timestamp": (
                        m["market"]["_market_identity"].get("scheduled_start_timestamp")
                    ),
                })
                if timestamp_guard == "enforce":
                    del best_by_match[mid]
                    dropped_fixtures[mid] = {
                        "match_id": mid,
                        "date": entry["date"],
                        "team1": entry["team1"],
                        "team2": entry["team2"],
                        "reason": "price_not_strictly_prematch",
                    }

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
        "match_identity": identity_contract(),
        "total_matches": len(odds_entries),
        "winner_used_for_market_selection": False,
        "selection_rule": selection_rule_block(timestamp_guard, dropped_fixtures,
                                               non_prematch_selected),
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
        "total_dropped_fixtures": len(dropped_fixtures),
        "dropped_fixtures": sorted(dropped_fixtures.values(),
                                   key=lambda d: (d["date"], d["match_id"])),
        "non_prematch_selected_rows": non_prematch_selected,
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
    print(f"H2H siblings deduped: {dup_dropped} extra head-to-head market(s) dropped")
    print(f"Non-H2H records rejected: {non_h2h_seen}  "
          f"(unresolved identity: {unresolved_seen})")
    print(f"Fixtures DROPPED for lack of a head-to-head market: {len(dropped_fixtures)}")
    for row in sorted(dropped_fixtures.values(), key=lambda d: (d["date"], d["match_id"]))[:20]:
        print(f"    {row['date']}  {row['team1']} vs {row['team2']}  [{row['reason']}]")
    print(f"Selected rows whose price is not strictly pre-match: "
          f"{len(non_prematch_selected)}  (guard={timestamp_guard})")
    print(f"Unmatched report {OUT_UNMATCHED_PATH}  ({len(unmatched):,} unmatched, "
          f"{len(winner_disagreements):,} raw disagreements, "
          f"{len(residual_disagreements):,} residual after dedup)")


def selection_rule_block(
    timestamp_guard: str,
    dropped_fixtures: dict[str, dict],
    non_prematch_selected: list[dict],
) -> dict:
    """Provenance describing exactly how one market per fixture was chosen."""
    reasons: Counter = Counter(d["reason"].split(":")[0] for d in dropped_fixtures.values())
    return {
        "version": SELECTION_RULE_VERSION,
        "primary": (
            "keep only the event's head-to-head market: Gamma "
            f"sportsMarketType == '{MONEYLINE_TYPE}' when present, otherwise "
            "market.question == event.title; where both signals exist they "
            "must agree or the record is unresolved"
        ),
        "tiebreak": ["highest market volume_usd", "lowest market_id"],
        "capture_order_used_for_tiebreak": False,
        "winner_used_for_market_selection": False,
        "price_magnitude_filter": None,
        "timestamp_guard": timestamp_guard,
        "low_liquidity_filter": True,
        "min_volume_usd": MIN_VOLUME_USD,
        "market_catalog": str(MARKET_CATALOG_PATH),
        "raw_market_index": str(RAW_MARKET_INDEX_PATH),
        "dropped_fixture_count": len(dropped_fixtures),
        "dropped_fixture_reasons": dict(reasons),
        "non_prematch_selected_count": len(non_prematch_selected),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def add_output_arguments(parser: argparse.ArgumentParser) -> None:
    """Output-path and guard flags shared with build_polymarket_odds_golden.py."""
    parser.add_argument("--out-odds", type=Path, default=None,
                        help="Write the odds file here instead of the default path.")
    parser.add_argument("--out-test-dir", type=Path, default=None,
                        help="Write copied Cricsheet JSONs here instead of the default dir.")
    parser.add_argument("--out-unmatched", type=Path, default=None,
                        help="Write the diagnostic report here instead of the default path.")
    parser.add_argument("--timestamp-guard", choices=TIMESTAMP_GUARD_MODES,
                        default="report",
                        help="Handling of selected rows priced at/after the scheduled "
                             "start: off | report (default) | enforce (drop them).")
    parser.add_argument("--restrict-to-manifest", type=Path, default=None,
                        help="Emit only fixtures already present in this odds "
                             "manifest. Rebuilds a frozen benchmark's own rows "
                             "under the current selection rule without "
                             "re-deriving which fixtures belong to it.")


def apply_output_overrides(args: argparse.Namespace) -> None:
    global OUT_ODDS_PATH, OUT_TEST_DIR, OUT_UNMATCHED_PATH
    if getattr(args, "out_odds", None):
        OUT_ODDS_PATH = args.out_odds
    if getattr(args, "out_test_dir", None):
        OUT_TEST_DIR = args.out_test_dir
    if getattr(args, "out_unmatched", None):
        OUT_UNMATCHED_PATH = args.out_unmatched


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Polymarket odds file by matching to Cricsheet")
    parser.add_argument("--dry-run", action="store_true", help="Report counts + sample, no writes")
    parser.add_argument("--verify-mapping", action="store_true",
                        help="Print Polymarket→Cricsheet team name diff and exit")
    add_output_arguments(parser)
    args = parser.parse_args()
    apply_output_overrides(args)

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

    restrict_to = (
        load_manifest_identities(args.restrict_to_manifest)
        if args.restrict_to_manifest else None
    )
    write_outputs(matched, unmatched, timestamp_guard=args.timestamp_guard,
                  restrict_to=restrict_to)


if __name__ == "__main__":
    main()
