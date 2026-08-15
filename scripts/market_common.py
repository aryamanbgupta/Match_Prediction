"""Shared market-join primitives (2026-08-14 review, Phase 4 item 5).

The three odds builders grew three timestamp parsers — one of which
reintroduced the exact `"+00"`-replace corruption another had explicitly
fixed, and another interpreted naive timestamps as LOCAL time via
`astimezone` on a tz-less datetime. Copy-drift in this layer produced the
toss-market defect's cousins; join primitives get ONE implementation here.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Optional

_UTC_OFFSET_SHORT = re.compile(r"([+-]\d{2})$")


def parse_market_timestamp(raw) -> Optional[datetime]:
    """Parse Gamma/CLOB timestamps as UTC; None when unparseable.

    Handles `YYYY-MM-DD HH:MM:SS+00` (two-digit offset Gamma emits, which
    pre-3.11 fromisoformat rejects), trailing `Z`, and full `+00:00`
    offsets. A bare `.replace("+00", "+00:00")` corrupts an already-full
    offset into `+00:00:00`; naive timestamps are UTC by contract, never
    local time.
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
