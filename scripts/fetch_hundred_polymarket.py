"""Pull Polymarket winner-market prices for The Hundred 2026 fixtures.

The repo's usual odds path (`build_polymarket_odds*.py`) reads a capture from
the sibling polymarket-cricket project. That capture stops at 2026-07-23, so
this script goes to Polymarket's public Gamma + CLOB APIs directly for the
in-flight Hundred season.

For a completed match the *current* price is the resolved 1/0, which is
useless as a benchmark, so the price is read from CLOB price history at a
cutoff strictly before the scheduled start (18:00 UK time / 17:00Z, matching
the sibling project's extractor). Prices are Polymarket binary prices and
already sum to ~1; they are stored as implied probabilities, not decimals.

Usage:
    uv run python scripts/fetch_hundred_polymarket.py
    uv run python scripts/fetch_hundred_polymarket.py --upcoming \
        --team1 "Southern Brave" --team2 "MI London" --date 2026-07-27
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
GAMMA = "https://gamma-api.polymarket.com/events"
CLOB_HISTORY = "https://clob.polymarket.com/prices-history"
SOURCE_2026 = REPO / "data" / "hundred" / "season_2026_men_source.json"
OUT = REPO / "data" / "hundred" / "polymarket_odds_2026.json"

# Polymarket's slug abbreviation for each 2026 men's franchise.
SLUG_ABBREV = {
    "Birmingham Phoenix": "bir",
    "London Spirit": "lon",
    "MI London": "mi",
    "Manchester Super Giants": "man",
    "Southern Brave": "sou",
    "Sunrisers Leeds": "sun",
    "Trent Rockets": "tre",
    "Welsh Fire": "wel",
}
# Cutoff hour (UTC) used as "pre-match": 17:00Z is an hour before the usual
# 18:30 UK start and matches the sibling extractor's timestamps.
CUTOFF_HOUR_UTC = 17


def get_json(url: str, retries: int = 3):
    # The API rejects urllib's default agent with a 403.
    request = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
        "Accept": "application/json",
    })
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read())
        except Exception as exc:  # noqa: BLE001 - network flakiness only
            if attempt == retries - 1:
                print(f"  request failed: {url} ({exc})")
                return None
            time.sleep(2)
    return None


def event_slug(team1: str, team2: str, date: str) -> str:
    return (f"crichundred-{SLUG_ABBREV[team1]}-{SLUG_ABBREV[team2]}-{date}")


def winner_market(event: dict) -> dict | None:
    """Return the head-to-head winner market, not toss / completed-match."""
    for market in event.get("markets") or []:
        question = market.get("question") or ""
        if " - " in question:  # side markets are suffixed
            continue
        outcomes = market.get("outcomes")
        if isinstance(outcomes, str):
            outcomes = json.loads(outcomes)
        if outcomes and len(outcomes) == 2:
            return market
    return None


def price_at_cutoff(token_id: str, cutoff_ts: int) -> tuple[float, int] | None:
    url = f"{CLOB_HISTORY}?market={token_id}&interval=max&fidelity=10"
    payload = get_json(url)
    points = (payload or {}).get("history") or []
    prior = [pt for pt in points if pt.get("t", 0) <= cutoff_ts]
    if not prior:
        return None
    last = prior[-1]
    return float(last["p"]), int(last["t"])


def fetch_match(team1: str, team2: str, date: str,
                completed: bool) -> dict:
    # Polymarket's slug usually lists the home side first, but not always;
    # try the reverse ordering before giving up.
    events, slug = [], None
    for candidate in (event_slug(team1, team2, date),
                      event_slug(team2, team1, date)):
        found = get_json(f"{GAMMA}?slug={candidate}") or []
        if found:
            events, slug = found, candidate
            break
    row = {"date": date, "team1": team1, "team2": team2, "slug": slug}
    if not events:
        row["error"] = "event not found"
        return row
    event = events[0]
    market = winner_market(event)
    if market is None:
        row["error"] = "no two-outcome winner market"
        return row

    outcomes = market.get("outcomes")
    if isinstance(outcomes, str):
        outcomes = json.loads(outcomes)
    prices = market.get("outcomePrices")
    if isinstance(prices, str):
        prices = json.loads(prices)
    tokens = market.get("clobTokenIds")
    if isinstance(tokens, str):
        tokens = json.loads(tokens)

    def as_float(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    row["outcomes"] = outcomes
    row["event_volume_usd"] = as_float(event.get("volume"))
    row["market_volume_usd"] = as_float(market.get("volume"))
    row["liquidity_usd"] = as_float(event.get("liquidity"))
    row["current_prices"] = [float(p) for p in prices] if prices else None
    row["closed"] = bool(market.get("closed"))

    if completed and tokens:
        idx = outcomes.index(team1) if team1 in outcomes else 0
        year, month, day = (int(part) for part in date.split("-"))
        # On double-header days the earlier match has already resolved by
        # 17:00Z, which would read back a 0/1 "price". Fall back to the
        # afternoon cutoff whenever the 17:00Z quote is degenerate.
        for hour, minute in ((CUTOFF_HOUR_UTC, 0), (12, 30)):
            cutoff = int(datetime(year, month, day, hour, minute,
                                  tzinfo=timezone.utc).timestamp())
            found = price_at_cutoff(tokens[idx], cutoff)
            if found is None:
                continue
            price, stamp = found
            if 0.03 < price < 0.97:
                row["prematch_prob_team1"] = price
                row["prematch_price_timestamp"] = datetime.fromtimestamp(
                    stamp, tz=timezone.utc).isoformat()
                row["cutoff_utc"] = f"{hour:02d}:{minute:02d}"
                break
        else:
            row["error"] = "no non-degenerate pre-match price"
    elif not completed and prices:
        idx = outcomes.index(team1) if team1 in outcomes else 0
        row["prematch_prob_team1"] = float(prices[idx])
        row["prematch_price_timestamp"] = datetime.now(
            timezone.utc).isoformat()
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=SOURCE_2026)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--upcoming", action="store_true",
                    help="Fetch one not-yet-played fixture's live price")
    ap.add_argument("--team1")
    ap.add_argument("--team2")
    ap.add_argument("--date")
    args = ap.parse_args()

    if args.upcoming:
        row = fetch_match(args.team1, args.team2, args.date, completed=False)
        print(json.dumps(row, indent=2))
        return 0

    source = json.loads(args.source.read_text())
    rows = []
    for match in source["matches"]:
        team1, team2 = match["teams"]
        print(f"  {match['date']}  {team1} vs {team2}")
        row = fetch_match(team1, team2, match["date"], completed=True)
        row["winner"] = match["winner"]
        rows.append(row)
        if "prematch_prob_team1" in row:
            print(f"     market P({team1}) = "
                  f"{row['prematch_prob_team1']*100:.1f}%  "
                  f"(vol ${row.get('market_volume_usd') or 0:,.0f})")
        else:
            print(f"     {row.get('error')}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"fetched_at": datetime.now(timezone.utc).isoformat(),
         "cutoff_hour_utc": CUTOFF_HOUR_UTC,
         "matches": rows}, indent=2))
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
