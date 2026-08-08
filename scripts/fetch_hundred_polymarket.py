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
# Toss is called ~30 min before the scheduled start; 60 min clears it.
PRETOSS_LEAD_SECONDS = 3600


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
    """Return the head-to-head winner market, not toss / completed-match.

    Polymarket ships each Hundred fixture as an event holding three binary
    markets: the moneyline (outcomes = the two team names), the toss winner
    (also two team names), and a "Completed match?" Yes/No. The moneyline is
    identified by `sportsMarketType == "moneyline"`, which is an explicit
    field rather than an ordering accident; the question-suffix test is kept
    as a fallback for events that predate that field.
    """
    markets = event.get("markets") or []
    for market in markets:
        if market.get("sportsMarketType") == "moneyline":
            return market
    for market in markets:
        question = market.get("question") or ""
        if " - " in question:  # side markets are suffixed
            continue
        outcomes = market.get("outcomes")
        if isinstance(outcomes, str):
            outcomes = json.loads(outcomes)
        if outcomes and len(outcomes) == 2:
            return market
    return None


def parse_iso(value: str | None) -> datetime | None:
    """Parse the mixed ISO shapes Gamma returns ('...Z' and '... +00:00')."""
    if not value:
        return None
    text = str(value).strip().replace("Z", "+00:00")
    if len(text) > 10 and text[10] == " ":
        text = text[:10] + "T" + text[11:]
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def scheduled_start(event: dict, market: dict) -> datetime | None:
    """Polymarket's own declared start for the fixture, in UTC."""
    for source in (market.get("gameStartTime"), event.get("startTime")):
        parsed = parse_iso(source)
        if parsed is not None:
            return parsed.astimezone(timezone.utc)
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
                completed: bool, start_mode: str = "fixed") -> dict:
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

    if start_mode == "gamma":
        start = scheduled_start(event, market)
        row["scheduled_start_utc"] = start.isoformat() if start else None
        if not tokens:
            row["error"] = "no clob tokens"
            return row
        if start is None:
            row["error"] = "no gameStartTime on market"
            return row
        idx = outcomes.index(team1) if team1 in outcomes else 0
        start_ts = int(start.timestamp())
        quotes, stamps = {}, {}
        for slot, token in enumerate(tokens[:2]):
            found = price_at_cutoff(token, start_ts - 1)
            if found is not None:
                quotes[slot], stamps[slot] = found
        if idx not in quotes:
            row["error"] = "no quote strictly before scheduled start"
            return row
        price, stamp = quotes[idx], stamps[idx]
        other = 1 - idx
        row["prematch_prob_team1"] = price
        row["prematch_prob_team2"] = (quotes[other] if other in quotes
                                      else round(1.0 - price, 6))
        row["prematch_prob_team2_is_complement"] = other not in quotes
        row["prematch_price_timestamp"] = datetime.fromtimestamp(
            stamp, tz=timezone.utc).isoformat()
        row["prematch_price_timestamp_team2"] = (
            datetime.fromtimestamp(stamps[other], tz=timezone.utc).isoformat()
            if other in stamps else None)
        row["quote_lead_seconds"] = start_ts - stamp
        row["cutoff_utc"] = "scheduled_start_minus_1s"
        row["quote_is_prematch"] = stamp < start_ts
        # The last quote before the scheduled start is taken AFTER the toss
        # (tossed ~30 min out), so it carries information a pre-match model
        # does not have. Record a pre-toss quote alongside it so a backtest
        # can pick the information set it actually wants to be judged against.
        pretoss = price_at_cutoff(tokens[idx], start_ts - PRETOSS_LEAD_SECONDS)
        if pretoss is not None:
            row["pretoss_prob_team1"] = pretoss[0]
            row["pretoss_price_timestamp"] = datetime.fromtimestamp(
                pretoss[1], tz=timezone.utc).isoformat()
            row["pretoss_lead_seconds"] = start_ts - pretoss[1]
        if not 0.0 < price < 1.0:
            row["warning"] = "degenerate pre-start price"
    elif completed and tokens:
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
        start = scheduled_start(event, market)
        row["scheduled_start_utc"] = start.isoformat() if start else None
        now = datetime.now(timezone.utc)
        row["prematch_prob_team1"] = float(prices[idx])
        row["prematch_prob_team2"] = float(prices[1 - idx])
        row["prematch_prob_team2_is_complement"] = False
        row["prematch_price_timestamp"] = now.isoformat()
        row["prematch_price_timestamp_team2"] = now.isoformat()
        row["cutoff_utc"] = "live_quote_at_fetch_time"
        row["quote_is_prematch"] = start is None or now < start
        row["quote_lead_seconds"] = (
            int((start - now).total_seconds()) if start else None)
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
    ap.add_argument("--start-mode", choices=("fixed", "gamma"), default="fixed",
                    help="fixed = legacy 17:00Z/12:30Z cutoff; gamma = last "
                         "quote strictly before the market's gameStartTime")
    ap.add_argument("--fixtures", type=Path,
                    help="Fixture list with per-match `status` "
                         "(played / in_progress / upcoming). Overrides "
                         "--source; unplayed fixtures get the live quote.")
    args = ap.parse_args()

    if args.upcoming:
        row = fetch_match(args.team1, args.team2, args.date, completed=False,
                          start_mode=args.start_mode)
        print(json.dumps(row, indent=2))
        return 0

    source = json.loads((args.fixtures or args.source).read_text())
    rows = []
    for match in source["matches"]:
        team1, team2 = match["teams"]
        status = match.get("status", "played")
        # An in-progress fixture must be priced from history like a played
        # one: its *current* quote is in-play, not pre-match.
        completed = status in ("played", "in_progress")
        print(f"  {match['date']}  {team1} vs {team2}  [{status}]")
        row = fetch_match(team1, team2, match["date"], completed=completed,
                          start_mode=args.start_mode if completed else "fixed")
        row["winner"] = match.get("winner")
        row["status"] = status
        if match.get("match_id"):
            row["cricsheet_match_id"] = match["match_id"]
        if match.get("venue"):
            row["venue"] = match["venue"]
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
         "start_mode": args.start_mode,
         "matches": rows}, indent=2))
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
