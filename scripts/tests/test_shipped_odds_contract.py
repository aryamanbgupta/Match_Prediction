"""Contracts for the two shipped v2 Polymarket evidence files.

Field mapping from the plan to the shipped schema:
* quote timestamp -> ``odds.winner.timestamp``
* scheduled start -> ``market_selection.scheduled_start_timestamp``
* implied probabilities -> reciprocals of the two decimal team odds in
  ``odds.winner`` (excluding its ``timestamp`` member)
* head-to-head metadata -> ``market_selection.market_question``,
  ``event_title``, and ``sports_market_type``
* unique fixture identity -> ``cricsheet_id``
"""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SHIPPED_FILES = [
    ROOT / "betting_odds_polymarket_v2.json",
    ROOT / "data" / "golden" / "betting_odds_golden_v2.json",
]


def _timestamp(value: str) -> datetime:
    value = value.replace("Z", "+00:00")
    value = re.sub(r"([+-]\d{2})$", r"\1:00", value)
    return datetime.fromisoformat(value)


@pytest.mark.parametrize("path", SHIPPED_FILES, ids=lambda path: path.name)
def test_every_shipped_row_is_valid_head_to_head_prestart_quote(path: Path):
    # Both files are tracked evidence (plan principle 0.3); a missing file is a
    # broken checkout, never a reason to skip.
    assert path.is_file(), f"tracked evidence file missing: {path}"
    rows = json.loads(path.read_text())["matches"]
    failures = []
    for row in rows:
        row_id = str(row["cricsheet_id"])
        selection = row["market_selection"]
        question = selection.get("market_question")
        is_head_to_head = bool(question) and (
            question == selection.get("event_title")
            or selection.get("sports_market_type") == "moneyline"
        )
        if not is_head_to_head:
            failures.append(f"{row_id}: not head-to-head")

        winner_odds = row["odds"]["winner"]
        team_odds = [float(value) for key, value in winner_odds.items()
                     if key != "timestamp"]
        implied_sum = sum(1.0 / decimal_odd for decimal_odd in team_odds)
        if len(team_odds) != 2 or not 0.98 <= implied_sum <= 1.02:
            failures.append(
                f"{row_id}: {len(team_odds)} teams, implied sum {implied_sum:.8f}"
            )

        quote = _timestamp(winner_odds["timestamp"])
        scheduled_start = _timestamp(selection["scheduled_start_timestamp"])
        if quote >= scheduled_start:
            failures.append(
                f"{row_id}: quote {quote.isoformat()} >= start {scheduled_start.isoformat()}"
            )

    duplicates = sorted(
        row_id for row_id, count in Counter(
            str(row["cricsheet_id"]) for row in rows
        ).items() if count > 1
    )
    if duplicates:
        failures.append(f"duplicate cricsheet ids: {duplicates}")
    assert not failures, "\n".join(failures)
