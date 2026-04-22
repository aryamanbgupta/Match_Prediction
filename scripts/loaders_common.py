"""Shared match-iteration helpers.

Extracted from parsing_v2.py:1249-1272 so both the Phase A parity harness
and the Phase B split scripts (build_stats_cache.py, materialize_features.py)
iterate matches identically. Keeping the chronological-sort + gender-filter
pattern in one place prevents drift between the pipeline and its tests.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Iterator, Tuple


def iter_matches_chronological(
    folder_path: str | Path,
    gender: str | None = "male",
) -> Iterator[Tuple[str, str, datetime]]:
    """Yield (match_id, json_text, match_date) in ascending date order.

    Args:
        folder_path: Directory containing cricsheet *.json match files.
        gender: If set (e.g. "male"), skip matches whose `info.gender`
            does not match. Pass None to disable filtering.

    Yields:
        match_id: Filename stem (e.g. "1234567" for 1234567.json).
        json_text: Raw JSON string — pass directly to `parse_match_data_v2`,
            which expects an unparsed string.
        match_date: Parsed date from `info.dates[0]`.

    Notes:
        * Sort is done once up front by parsing each file's first date. On
          ~11K matches this is ~3s of JSON decoding; acceptable as the
          corpus-level setup cost.
        * Files whose JSON fails to parse or lack `info.dates` are skipped
          silently to match parsing_v2.py's existing error-tolerant behavior.
    """
    folder = Path(folder_path)
    json_files = list(folder.glob("*.json"))

    # Gender filter is applied during the sort pass so we don't hold text
    # for skipped matches in memory. A second read at yield time hits the
    # OS page cache and is cheap.
    dated: list[tuple[datetime, Path]] = []
    for p in json_files:
        try:
            data = json.loads(p.read_text())
            if gender is not None and data["info"].get("gender", "male") != gender:
                continue
            date_str = data["info"]["dates"][0]
            dated.append((datetime.strptime(date_str, "%Y-%m-%d"), p))
        except (json.JSONDecodeError, KeyError, IndexError, ValueError):
            continue

    dated.sort(key=lambda t: t[0])

    for match_date, path in dated:
        yield path.stem, path.read_text(), match_date
