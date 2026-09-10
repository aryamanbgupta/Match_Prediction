#!/usr/bin/env python3
"""Append open men's T20 moneyline fixtures from Polymarket Gamma."""
from __future__ import annotations

import argparse
import hmac
import json
import os
import secrets
import stat
import sys
import urllib.parse
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from sim_eval.market_selection import classify_h2h  # noqa: E402

GAMMA_EVENTS = "https://gamma-api.polymarket.com/events"
FIXTURE_WRITER = "fetch_fixtures"
WRITER_KEY_BYTES = 32
HMAC_FIELDS = (
    "quote", "quote_ts", "market_volume_usd", "scheduled_start",
    "fixture_id", "market_id",
)


def writer_key_path() -> Path:
    override = os.environ.get("DAILY_WRITER_KEY")
    return Path(override) if override else REPO / "daily" / ".writer_key"


def load_writer_key(*, create: bool = False) -> bytes:
    """Read the machine-local writer key, creating it only for fixture capture."""
    path = writer_key_path()
    if create:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(path, flags, 0o600)
        except FileExistsError:
            pass
        else:
            try:
                os.fchmod(descriptor, 0o600)
                with os.fdopen(descriptor, "wb", closefd=False) as handle:
                    handle.write(secrets.token_bytes(WRITER_KEY_BYTES))
                    handle.flush()
                    os.fsync(handle.fileno())
            finally:
                os.close(descriptor)
    try:
        info = path.stat()
        if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600:
            raise RuntimeError(f"daily writer key must be a regular mode-0600 file: {path}")
        key = path.read_bytes()
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"daily writer key is missing; run fetch_fixtures first: {path}"
        ) from exc
    if len(key) != WRITER_KEY_BYTES:
        raise RuntimeError(f"daily writer key must contain exactly 32 bytes: {path}")
    return key


def fixture_line_hmac_sha256(row: dict, key: bytes | None = None) -> str:
    """Authenticate the canonical writer-owned fixture fields."""
    missing = [field for field in HMAC_FIELDS if field not in row]
    if missing:
        raise ValueError(f"fixture line lacks HMAC-owned fields: {', '.join(missing)}")
    payload = {field: row[field] for field in HMAC_FIELDS}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hmac.digest(key or load_writer_key(), encoded, "sha256").hex()


def verify_fixture_line(row: dict, key: bytes | None = None) -> None:
    if row.get("writer") != FIXTURE_WRITER:
        raise ValueError("fixture line was not written by fetch_fixtures")
    supplied = row.get("line_hmac_sha256")
    expected = fixture_line_hmac_sha256(row, key)
    if not isinstance(supplied, str) or not hmac.compare_digest(supplied, expected):
        raise ValueError("fixture line_hmac_sha256 verification failed")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_list(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    return value if isinstance(value, list) else []


def is_h2h_market(event: dict, market: dict) -> bool:
    """Apply the sealed extractor's structural, outcome-blind H2H rule."""
    verdict, _ = classify_h2h(
        market.get("question"), event.get("title"),
        market.get("sportsMarketType"),
    )
    return verdict is True


def _male_t20(event: dict, market: dict) -> bool:
    text = " ".join(str(x) for x in (
        event.get("title", ""), event.get("slug", ""),
        event.get("series", ""), market.get("question", ""),
    )).lower()
    tags = " ".join(str(t.get("label", t) if isinstance(t, dict) else t)
                    for t in event.get("tags", [])).lower()
    if any(word in text + " " + tags for word in ("women", "women's", "womens")):
        return False
    return any(word in text + " " + tags for word in (
        "t20", "ipl", "big bash", "psl", "bpl", "cpl", "lpl",
        "super smash", "sa20", "ilt20",
    ))


def _iso(value) -> str | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


def records_from_gamma(events: list[dict]) -> list[dict]:
    rows = []
    for event in events:
        candidates = [m for m in event.get("markets", [])
                      if not m.get("closed") and is_h2h_market(event, m)
                      and _male_t20(event, m)]
        if not candidates:
            continue
        market = sorted(candidates, key=lambda m: (
            -float(m.get("volume") or 0), str(m.get("id") or "")
        ))[0]
        outcomes = parse_list(market.get("outcomes"))
        prices = parse_list(market.get("outcomePrices"))
        if len(outcomes) != 2:
            continue
        numeric_prices = None
        try:
            if len(prices) == 2:
                numeric_prices = {str(outcomes[i]): float(prices[i]) for i in range(2)}
        except (TypeError, ValueError):
            numeric_prices = None
        start = _iso(market.get("gameStartTime") or event.get("startTime"))
        if start is None:
            continue
        rows.append({
            "fixture_id": str(market.get("id")),
            "market_id": str(market.get("id")),
            "cricsheet_id": None,
            "team1": str(outcomes[0]),
            "team2": str(outcomes[1]),
            "venue": event.get("venue") or market.get("venue"),
            "competition": event.get("series") or event.get("title"),
            "scheduled_start": start,
            "quote": numeric_prices,
            "market_volume_usd": (float(market["volume"])
                                  if market.get("volume") is not None else None),
        })
    return rows


def fetch_events(url: str = GAMMA_EVENTS) -> list[dict]:
    rows = []
    for offset in range(0, 1000, 100):
        query = urllib.parse.urlencode({"active": "true", "closed": "false",
                                        "tag_slug": "cricket", "limit": 100,
                                        "offset": offset})
        request = urllib.request.Request(f"{url}?{query}", headers={"User-Agent": "cricml-daily/1"})
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read())
        page = payload if isinstance(payload, list) else payload.get("events", [])
        rows.extend(page)
        if len(page) < 100:
            break
    return rows


def append_fixtures(rows: list[dict], output: Path) -> int:
    writer_key = load_writer_key(create=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    existing = set()
    if output.exists():
        existing = {line for line in output.read_text().splitlines() if line.strip()}
    added = 0
    with output.open("a", encoding="utf-8") as handle:
        for row in rows:
            if "quote_ts" in row:
                raise ValueError("quote_ts is writer-owned and may not be supplied")
            stamped = dict(row)
            stamped["quote_ts"] = (
                utc_now().astimezone(timezone.utc).isoformat()
                if stamped.get("quote") is not None else None
            )
            stamped["writer"] = FIXTURE_WRITER
            stamped["line_hmac_sha256"] = fixture_line_hmac_sha256(stamped, writer_key)
            encoded = json.dumps(stamped, sort_keys=True, separators=(",", ":"))
            if encoded in existing:
                continue
            handle.write(encoded + "\n")
            existing.add(encoded)
            added += 1
    return added


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--response", type=Path, help="Recorded Gamma response (no network)")
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    payload = json.loads(args.response.read_text()) if args.response else fetch_events()
    events = payload if isinstance(payload, list) else payload.get("events", [])
    out = args.out or REPO / "daily" / "fixtures" / f"{args.date}.jsonl"
    print(append_fixtures(records_from_gamma(events), out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
