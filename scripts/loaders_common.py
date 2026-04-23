"""Shared match-iteration + metadata helpers.

Extracted from parsing_v2.py:1249-1272 so both the Phase A parity harness
and the Phase B split scripts (build_stats_cache.py, materialize_features.py)
iterate matches identically. Keeping the chronological-sort + gender-filter
pattern, and the match-metadata extraction, in one place prevents drift
between the pipeline and its tests.
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


# Monolith-inherited split cutoffs (parsing_v2.py:1210-1215). Used as
# defaults when a YAML config omits `data.splits` — every existing
# experiments/configs/*.yaml relies on these via the merge semantics
# in `effective_splits()`.
DEFAULT_SPLITS = {
    "train_end":    "2024-12-31",
    "val_end":      "2025-06-30",
    "test_end":     "2026-04-16",
    "golden_start": "2026-04-17",
}


def effective_splits(splits: dict | None) -> dict:
    """Merge YAML splits over DEFAULT_SPLITS to produce the canonical form.

    Both the materializer (write side: stored in `.feature_hash`) and the
    smart-cache check in run_experiment.py (read side) must use this
    function — otherwise the cached-vs-current comparison can disagree
    about missing keys and the cache silently breaks.
    """
    if splits is None:
        splits = {}
    merged = dict(DEFAULT_SPLITS)
    merged.update({k: v for k, v in splits.items() if v is not None})
    return merged


def extract_match_metadata(data: dict) -> dict:
    """Parse the event/team/venue fields both parsing scripts need.

    Shared by `build_stats_cache.py` and `materialize_features.py` so the
    `classify_match_k_factor` input contract is consulted in one place —
    if that function's signature or the cricsheet JSON schema changes,
    there's one callsite to fix instead of two.

    Returns a dict with:
        event_name: str   — `info.event.name`, or '' if missing/non-dict.
        team_type:  str   — `info.team_type`, default 'unknown'.
        teams:      list  — `info.teams`, default [].
        k_factor:   float — computed via classify_match_k_factor.
        venue:      str   — `info.venue`, default 'unknown'.
    """
    # Local import avoids a module-load-time cycle: parsing_v2.py imports
    # `loaders_common` for iteration helpers in some contexts; keeping the
    # classify_match_k_factor reference lazy keeps us independent.
    from parsing_v2 import classify_match_k_factor

    info = data["info"]
    event_info = info.get("event", {})
    event_name = (
        event_info.get("name", "") if isinstance(event_info, dict) else ""
    )
    team_type = info.get("team_type", "unknown")
    teams = info.get("teams", [])
    return {
        "event_name": event_name,
        "team_type": team_type,
        "teams": teams,
        "k_factor": classify_match_k_factor(event_name, team_type, teams),
        "venue": info.get("venue", "unknown"),
    }
