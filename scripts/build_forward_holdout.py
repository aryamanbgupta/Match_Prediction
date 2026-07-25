#!/usr/bin/env python3
"""Build an isolated, sealed forward cricket evaluation set.

This builder consumes:

* strict match-winner odds from
  ``polymarket-cricket/extract_match_prematch_odds_strict.py``; and
* refreshed Cricsheet ``*_json.zip`` archives.

It never modifies existing training, validation, iteration, golden, odds, or
model artifacts. It performs no model scoring. The output contains a locked
dataset, provenance manifest, diagnostics, chronological bridge/context JSONs,
and a ``SEALED`` marker for a later one-time terminal evaluation.

Duplicate market selection is outcome-blind:

1. all strict integrity guards must pass;
2. highest market volume;
3. latest valid quote strictly before scheduled start;
4. stable market ID.

Example:

    uv run python scripts/build_forward_holdout.py \
      --market-json /Users/.../polymarket_match_odds_strict_2026-06-01_2026-07-23.json \
      --start 2026-06-01 --end 2026-07-13 --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from build_polymarket_odds import (  # noqa: E402
    TEAM_NAME_MAP,
    build_match_id,
)
from identity_maps import canonicalize_venue, venue_alias_contract  # noqa: E402

DEFAULT_ZIP_DIR = Path(
    "/Users/aryamangupta/Projects/stat-generator/data/cricsheet"
)
DEFAULT_CONTEXT_START = "2026-04-17"
REQUIRED_SELECTION_CONTRACT = {
    "h2h_rule": "market.question == event.title",
    "requires_resolved": True,
    "requires_explicit_game_start": True,
    "price_rule": "latest valid CLOB tick strictly before gameStartTime",
    "deduplicated_by_fixture": False,
    "winner_used_for_market_selection": False,
}
EXISTING_EVALUATED_DIRS = (
    ROOT / "data" / "t20s_json",
    ROOT / "data" / "polymarket_test",
    ROOT / "data" / "golden" / "polymarket_test",
    ROOT / "data" / "golden_blast" / "polymarket_test",
)
_WOMENS_MARKERS = re.compile(
    r"(^|[^a-z0-9])(women|womens|woman|female|wpl)([^a-z0-9]|$)",
    re.IGNORECASE,
)
_WOMENS_SLUG_MARKERS = ("t20blastw-", "womens-", "women-", "wpl-")
_NON_TEAM_OUTCOMES = {"yes", "no", "over", "under", "draw"}
_TERMINAL_SLUG_DATE = re.compile(r"(20\d{2}-\d{2}-\d{2})$")

# Shared historical mappings plus deterministic names observed in the
# refreshed forward sources. These aliases change spelling only; they do not
# use outcomes or fuzzy matching.
_TEAM_ALIASES = dict(TEAM_NAME_MAP)
_TEAM_ALIASES.update(
    {
        "Warwickshire": "Birmingham Bears",
        "Mi New York": "MI New York",
        "Czechia": "Czech Republic",
        "Turkiye": "Turkey",
    }
)


@dataclass(frozen=True)
class CricsheetMatch:
    cricsheet_id: str
    date: str
    teams: tuple[str, str]
    venue: str
    winner: str
    event_name: str
    source_zip: Path
    member_name: str
    payload: bytes
    payload_sha256: str


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(128 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_ts(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(
            str(raw).replace("+00", "+00:00").replace("Z", "+00:00")
        ).astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def canonical_team(name: str | None) -> str:
    value = str(name or "").strip()
    seen: set[str] = set()
    while value in _TEAM_ALIASES and value not in seen:
        seen.add(value)
        value = _TEAM_ALIASES[value]
    return value


def is_womens_market(row: dict) -> bool:
    title = str(row.get("event_title") or "")
    slug = str(row.get("event_slug") or "").lower()
    return bool(_WOMENS_MARKERS.search(title)) or any(
        marker in slug for marker in _WOMENS_SLUG_MARKERS
    )


def fixture_date_from_market(row: dict) -> str | None:
    """Return the local fixture date encoded in the Polymarket event slug.

    ``gameStartTime`` is UTC, while Cricsheet's ``info.dates`` is the local
    match date. For late-evening fixtures in the Americas those dates differ
    by one day. The strict pull persists both fields, so the slug date is the
    appropriate join key and the UTC timestamp remains the quote-time guard.
    """
    match = _TERMINAL_SLUG_DATE.search(str(row.get("event_slug") or ""))
    if match is None:
        return None
    try:
        datetime.strptime(match.group(1), "%Y-%m-%d")
    except ValueError:
        return None
    return match.group(1)


def strict_market_rejection(
    row: dict,
    start: str,
    end: str,
    min_volume: float,
) -> str | None:
    required = (
        "event_id",
        "event_slug",
        "market_id",
        "event_title",
        "market_question",
        "scheduled_start_timestamp",
        "price_timestamp",
        "team1",
        "team2",
        "winner",
        "prematch_price_team1",
        "prematch_price_team2",
        "volume_usd",
    )
    if any(row.get(key) in (None, "") for key in required):
        return "missing_required_provenance"
    if row.get("gender_scope") != "male":
        return "not_male_scope"
    if is_womens_market(row):
        return "womens_market"
    if str(row["market_question"]).strip() != str(row["event_title"]).strip():
        return "not_exact_h2h"

    team1 = canonical_team(row["team1"])
    team2 = canonical_team(row["team2"])
    if (
        not team1
        or not team2
        or team1.casefold() == team2.casefold()
        or team1.casefold() in _NON_TEAM_OUTCOMES
        or team2.casefold() in _NON_TEAM_OUTCOMES
    ):
        return "invalid_team_outcomes"
    if canonical_team(row["winner"]) not in {team1, team2}:
        return "unresolved_or_invalid_winner"

    scheduled = parse_ts(row["scheduled_start_timestamp"])
    priced = parse_ts(row["price_timestamp"])
    if scheduled is None or priced is None:
        return "invalid_timestamp"
    if not priced < scheduled:
        return "not_strictly_prematch"
    scheduled_utc_date = scheduled.strftime("%Y-%m-%d")
    if row.get("date") != scheduled_utc_date:
        return "date_start_mismatch"
    fixture_date = fixture_date_from_market(row)
    if fixture_date is None:
        return "missing_or_invalid_fixture_date"
    date_delta = (
        datetime.strptime(fixture_date, "%Y-%m-%d")
        - datetime.strptime(scheduled_utc_date, "%Y-%m-%d")
    ).days
    if abs(date_delta) > 1:
        return "fixture_date_start_mismatch"
    if not (start <= fixture_date <= end):
        return "outside_holdout_window"
    try:
        lag = int(row.get("price_lag_seconds"))
    except (TypeError, ValueError):
        return "invalid_price_lag"
    if lag != int((scheduled - priced).total_seconds()) or lag <= 0:
        return "invalid_price_lag"
    try:
        price1 = float(row["prematch_price_team1"])
        price2 = float(row["prematch_price_team2"])
        volume = float(row["volume_usd"])
    except (TypeError, ValueError):
        return "invalid_numeric_field"
    if not (0.0 < price1 < 1.0 and 0.0 < price2 < 1.0):
        return "invalid_probability"
    if abs((price1 + price2) - 1.0) > 1e-5:
        return "probabilities_not_complementary"
    if volume < min_volume:
        return "below_min_volume"
    if bool(row.get("low_liquidity")):
        return "low_liquidity"
    return None


def market_selection_key(row: dict) -> tuple[float, datetime, str]:
    """Outcome-blind duplicate selection key; do not add winner/result."""
    priced = parse_ts(row["price_timestamp"])
    assert priced is not None
    return (
        float(row["volume_usd"]),
        priced,
        str(row["market_id"]),
    )


def _read_cricsheet_member(
    zip_path: Path,
    member_name: str,
    payload: bytes,
    context_start: str,
    end: str,
) -> tuple[CricsheetMatch | None, str | None]:
    try:
        data = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None, "invalid_json"
    info = data.get("info") or {}
    dates = info.get("dates") or []
    teams = info.get("teams") or []
    if info.get("match_type") != "T20":
        return None, "non_t20"
    if info.get("gender") != "male":
        return None, "non_male"
    if not dates or not (context_start <= dates[0] <= end):
        return None, "outside_context_window"
    if len(teams) != 2:
        return None, "invalid_teams"
    outcome = info.get("outcome") or {}
    winner = outcome.get("winner")
    if not winner and outcome.get("result") == "tie":
        winner = outcome.get("eliminator")
    if not winner or winner not in teams:
        return None, "no_valid_winner"
    event = info.get("event") or {}
    event_name = event.get("name", "") if isinstance(event, dict) else ""
    return (
        CricsheetMatch(
            cricsheet_id=Path(member_name).stem,
            date=dates[0],
            teams=(str(teams[0]).strip(), str(teams[1]).strip()),
            venue=canonicalize_venue(info.get("venue")),
            winner=str(winner),
            event_name=str(event_name),
            source_zip=zip_path,
            member_name=member_name,
            payload=payload,
            payload_sha256=sha256_bytes(payload),
        ),
        None,
    )


def load_cricsheet_archives(
    zip_dir: Path,
    context_start: str,
    end: str,
) -> tuple[dict[str, CricsheetMatch], dict]:
    archives = sorted(zip_dir.glob("*_json.zip"))
    if not archives:
        raise FileNotFoundError(f"no *_json.zip archives under {zip_dir}")
    records: dict[str, CricsheetMatch] = {}
    stats = Counter()
    duplicate_sources: list[dict] = []
    for zip_path in archives:
        with zipfile.ZipFile(zip_path) as archive:
            for member_name in archive.namelist():
                if not member_name.endswith(".json"):
                    continue
                payload = archive.read(member_name)
                record, reason = _read_cricsheet_member(
                    zip_path, member_name, payload, context_start, end
                )
                if record is None:
                    stats[reason or "unknown_rejection"] += 1
                    continue
                existing = records.get(record.cricsheet_id)
                if existing is not None:
                    if existing.payload_sha256 != record.payload_sha256:
                        raise RuntimeError(
                            "Cricsheet ID collision with different payloads: "
                            f"{record.cricsheet_id} in {existing.source_zip.name} "
                            f"and {record.source_zip.name}"
                        )
                    duplicate_sources.append(
                        {
                            "cricsheet_id": record.cricsheet_id,
                            "kept": existing.source_zip.name,
                            "duplicate": record.source_zip.name,
                        }
                    )
                    stats["identical_duplicate_source"] += 1
                    continue
                records[record.cricsheet_id] = record
                stats["accepted_context_match"] += 1
    return records, {
        "counts": dict(sorted(stats.items())),
        "identical_duplicate_sources": duplicate_sources,
        "archives": [
            {
                "path": str(path),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in archives
        ],
    }


def existing_evaluated_ids() -> tuple[set[str], dict[str, int]]:
    found: set[str] = set()
    counts: dict[str, int] = {}
    for directory in EXISTING_EVALUATED_DIRS:
        ids = {path.stem for path in directory.glob("*.json")}
        found.update(ids)
        counts[str(directory.relative_to(ROOT))] = len(ids)
    return found, counts


def match_market_rows(
    rows: Iterable[dict],
    cricsheet: dict[str, CricsheetMatch],
) -> tuple[dict[str, list[dict]], list[dict]]:
    by_date: dict[str, list[CricsheetMatch]] = defaultdict(list)
    for record in cricsheet.values():
        by_date[record.date].append(record)
    grouped: dict[str, list[dict]] = defaultdict(list)
    unmatched: list[dict] = []
    for row in rows:
        fixture_date = fixture_date_from_market(row)
        if fixture_date is None:
            raise RuntimeError(
                "accepted strict market unexpectedly lacks fixture date"
            )
        market_teams = {
            canonical_team(row["team1"]),
            canonical_team(row["team2"]),
        }
        hits = [
            record
            for record in by_date.get(fixture_date, [])
            if {canonical_team(team) for team in record.teams} == market_teams
        ]
        if len(hits) != 1:
            unmatched.append(
                {
                    "reason": (
                        "no_cricsheet_match"
                        if not hits
                        else "ambiguous_cricsheet_match"
                    ),
                    "fixture_date": fixture_date,
                    "scheduled_utc_date": row["date"],
                    "team1": row["team1"],
                    "team2": row["team2"],
                    "market_id": row["market_id"],
                    "event_slug": row.get("event_slug"),
                    "candidate_cricsheet_ids": [
                        record.cricsheet_id for record in hits
                    ],
                }
            )
            continue
        grouped[hits[0].cricsheet_id].append(row)
    return grouped, unmatched


def _aligned_prices(row: dict, record: CricsheetMatch) -> tuple[float, float]:
    market_team1 = canonical_team(row["team1"])
    market_team2 = canonical_team(row["team2"])
    cric_team1 = canonical_team(record.teams[0])
    cric_team2 = canonical_team(record.teams[1])
    p1 = float(row["prematch_price_team1"])
    p2 = float(row["prematch_price_team2"])
    if (market_team1, market_team2) == (cric_team1, cric_team2):
        return p1, p2
    if (market_team1, market_team2) == (cric_team2, cric_team1):
        return p2, p1
    raise RuntimeError("team alignment failed after exact set match")


def _decimal(price: float) -> float:
    return round(1.0 / price, 4)


def _write_json(path: Path, payload: dict | list) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as handle:
        json.dump(payload, handle, indent=2, default=str)
    tmp.replace(path)


def build(args: argparse.Namespace) -> dict:
    for value, label in (
        (args.context_start, "--context-start"),
        (args.start, "--start"),
        (args.end, "--end"),
    ):
        datetime.strptime(value, "%Y-%m-%d")
    if not (args.context_start <= args.start <= args.end):
        raise ValueError("require context_start <= start <= end")
    if not args.market_json.exists():
        raise FileNotFoundError(args.market_json)

    market_blob = json.loads(args.market_json.read_text())
    if market_blob.get("schema_version") != 1:
        raise RuntimeError("strict market file schema_version must be 1")
    contract = market_blob.get("selection_contract") or {}
    if contract != REQUIRED_SELECTION_CONTRACT:
        raise RuntimeError(
            "strict market selection contract mismatch; refuse to weaken "
            f"holdout provenance\nexpected={REQUIRED_SELECTION_CONTRACT}\n"
            f"actual={contract}"
        )

    cricsheet, cric_report = load_cricsheet_archives(
        args.zip_dir, args.context_start, args.end
    )
    eligible_holdout = {
        match_id: record
        for match_id, record in cricsheet.items()
        if args.start <= record.date <= args.end
    }
    if not eligible_holdout:
        raise RuntimeError("no eligible Cricsheet matches in holdout window")

    market_rejections: list[dict] = []
    rejection_counts = Counter()
    accepted_markets: list[dict] = []
    for row in market_blob.get("matches") or []:
        reason = strict_market_rejection(
            row, args.start, args.end, args.min_volume
        )
        if reason:
            rejection_counts[reason] += 1
            market_rejections.append(
                {
                    "reason": reason,
                    "market_id": row.get("market_id"),
                    "event_id": row.get("event_id"),
                    "date": row.get("date"),
                    "team1": row.get("team1"),
                    "team2": row.get("team2"),
                }
            )
            continue
        accepted_markets.append(row)

    grouped, unmatched = match_market_rows(
        accepted_markets, eligible_holdout
    )
    selected: dict[str, dict] = {}
    duplicate_report: list[dict] = []
    for cricsheet_id, candidates in grouped.items():
        ordered = sorted(candidates, key=market_selection_key, reverse=True)
        selected[cricsheet_id] = ordered[0]
        if len(ordered) > 1:
            duplicate_report.append(
                {
                    "cricsheet_id": cricsheet_id,
                    "selection_rule": [
                        "highest_volume",
                        "latest_valid_prematch_quote",
                        "stable_market_id",
                    ],
                    "selected_market_id": ordered[0]["market_id"],
                    "candidates": [
                        {
                            "market_id": row["market_id"],
                            "volume_usd": row["volume_usd"],
                            "price_timestamp": row["price_timestamp"],
                        }
                        for row in ordered
                    ],
                }
            )

    existing_ids, existing_counts = existing_evaluated_ids()
    overlaps = sorted(set(selected) & existing_ids)
    if overlaps:
        raise RuntimeError(
            "forward holdout overlaps an existing evaluated/training pool: "
            + ", ".join(overlaps[:20])
        )

    winner_disagreements: list[dict] = []
    for cricsheet_id, row in selected.items():
        record = eligible_holdout[cricsheet_id]
        if canonical_team(row["winner"]) != canonical_team(record.winner):
            winner_disagreements.append(
                {
                    "cricsheet_id": cricsheet_id,
                    "market_id": row["market_id"],
                    "market_winner": row["winner"],
                    "cricsheet_winner": record.winner,
                }
            )
    if winner_disagreements:
        raise RuntimeError(
            "selected market winner disagrees with Cricsheet after "
            "outcome-blind selection; see dry-run diagnostics: "
            + json.dumps(winner_disagreements[:5])
        )

    selected_records = [eligible_holdout[mid] for mid in selected]
    selected_records.sort(key=lambda record: (record.date, record.cricsheet_id))
    date_min = min(record.date for record in selected_records)
    date_max = max(record.date for record in selected_records)
    output_dir = args.out_root / f"{args.start}_{args.end}"

    event_counts = Counter(
        record.event_name or record.source_zip.stem
        for record in selected_records
    )
    selected_markets = [selected[record.cricsheet_id] for record in selected_records]
    volume_50k = sum(float(row["volume_usd"]) >= 50_000 for row in selected_markets)
    volume_100k = sum(float(row["volume_usd"]) >= 100_000 for row in selected_markets)
    integrity = {
        "status": "PASS",
        "model_scoring_performed": False,
        "holdout_window": {"start": args.start, "end": args.end},
        "matched_date_range": {"start": date_min, "end": date_max},
        "eligible_cricsheet_matches": len(eligible_holdout),
        "strict_market_rows_in_source": len(market_blob.get("matches") or []),
        "strict_market_rows_after_guards": len(accepted_markets),
        "selected_unique_matches": len(selected),
        "liquidity_slices": {
            "min_1k": len(selected),
            "min_50k": volume_50k,
            "min_100k": volume_100k,
        },
        "duplicate_fixture_groups": len(duplicate_report),
        "unmatched_market_rows": len(unmatched),
        "market_rejection_counts": dict(sorted(rejection_counts.items())),
        "winner_disagreements_after_outcome_blind_selection": 0,
        "overlap_with_existing_evaluated_pools": 0,
        "existing_pool_counts": existing_counts,
        "timestamp_invariant": (
            "all selected price_timestamp < scheduled_start_timestamp"
        ),
        "competition_counts": dict(event_counts.most_common()),
        "extreme_top_price_counts": {
            "gt_0.92": sum(
                max(
                    float(row["prematch_price_team1"]),
                    float(row["prematch_price_team2"]),
                )
                > 0.92
                for row in selected_markets
            ),
            "gt_0.98": sum(
                max(
                    float(row["prematch_price_team1"]),
                    float(row["prematch_price_team2"]),
                )
                > 0.98
                for row in selected_markets
            ),
        },
    }
    diagnostics = {
        "market_rejections": market_rejections,
        "unmatched_markets": unmatched,
        "duplicate_fixture_groups": duplicate_report,
        "winner_disagreements": winner_disagreements,
        "cricsheet_archive_report": cric_report,
    }

    manifest_matches: list[dict] = []
    odds_entries: list[dict] = []
    for record in selected_records:
        row = selected[record.cricsheet_id]
        price_for_team1, price_for_team2 = _aligned_prices(row, record)
        match_id = build_match_id(
            record.date, record.teams[0], record.teams[1], record.venue
        )
        odds_entries.append(
            {
                "match_id": match_id,
                "date": record.date,
                "team1": record.teams[0],
                "team2": record.teams[1],
                "venue": record.venue,
                "actual_winner": record.winner,
                "odds": {
                    "winner": {
                        record.teams[0]: _decimal(price_for_team1),
                        record.teams[1]: _decimal(price_for_team2),
                        "timestamp": row["price_timestamp"],
                        "scheduled_start_timestamp": row[
                            "scheduled_start_timestamp"
                        ],
                    }
                },
                "source": "polymarket_strict",
                "polymarket_event_id": row["event_id"],
                "polymarket_market_id": row["market_id"],
                "polymarket_event_slug": row.get("event_slug"),
                "polymarket_scheduled_utc_date": row["date"],
                "polymarket_volume_usd": row["volume_usd"],
                "tournament": record.event_name or row.get("tournament"),
            }
        )
        manifest_matches.append(
            {
                "match_id": match_id,
                "cricsheet_id": record.cricsheet_id,
                "date": record.date,
                "teams": list(record.teams),
                "venue": record.venue,
                "competition": record.event_name,
                "cricsheet_source_zip": record.source_zip.name,
                "cricsheet_member": record.member_name,
                "cricsheet_sha256": record.payload_sha256,
                "polymarket_event_id": row["event_id"],
                "polymarket_market_id": row["market_id"],
                "polymarket_fixture_date": fixture_date_from_market(row),
                "scheduled_start_timestamp": row[
                    "scheduled_start_timestamp"
                ],
                "price_timestamp": row["price_timestamp"],
                "price_lag_seconds": row["price_lag_seconds"],
                "volume_usd": row["volume_usd"],
            }
        )

    odds_blob = {
        "source": "polymarket_strict",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_matches": len(odds_entries),
        "filters": {
            "min_volume_usd": args.min_volume,
            "test_window_start": args.start,
            "test_window_end": args.end,
            "requires_resolved_winner": True,
            "requires_match_type_t20_male": True,
            "requires_strictly_prematch_timestamp": True,
            "duplicate_selection_uses_winner": False,
        },
        "matches": odds_entries,
        "venue_identity": venue_alias_contract(),
    }
    manifest = {
        "schema_version": 1,
        "purpose": "sealed forward audit; terminal evaluation only",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "venue_identity": venue_alias_contract(),
        "model_scoring_performed": False,
        "holdout_window": {"start": args.start, "end": args.end},
        "context_window": {
            "start": args.context_start,
            "end": args.end,
            "purpose": (
                "chronological pre-match state only; never model fitting"
            ),
        },
        "source_market_file": {
            "path": str(args.market_json),
            "size": args.market_json.stat().st_size,
            "sha256": sha256_file(args.market_json),
            "selection_contract": REQUIRED_SELECTION_CONTRACT,
        },
        "source_cricsheet_archives": cric_report["archives"],
        "selected_match_count": len(manifest_matches),
        "liquidity_slices": integrity["liquidity_slices"],
        "matches": manifest_matches,
    }

    if args.dry_run:
        print(json.dumps(integrity, indent=2))
        print(f"Dry run: would write sealed dataset to {output_dir}")
        return {
            "integrity": integrity,
            "diagnostics": diagnostics,
            "manifest": manifest,
            "output_dir": str(output_dir),
        }

    if output_dir.exists():
        raise FileExistsError(
            f"{output_dir} already exists; sealed datasets are immutable"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}.staging-{os.getpid()}"
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir()
    try:
        (staging / "raw").mkdir()
        (staging / "polymarket_test").mkdir()
        (staging / "context_t20s_json").mkdir()
        shutil.copy2(args.market_json, staging / "raw" / args.market_json.name)
        for record in selected_records:
            (staging / "polymarket_test" / f"{record.cricsheet_id}.json").write_bytes(
                record.payload
            )
        for record in sorted(
            cricsheet.values(), key=lambda item: (item.date, item.cricsheet_id)
        ):
            (
                staging
                / "context_t20s_json"
                / f"{record.cricsheet_id}.json"
            ).write_bytes(record.payload)
        _write_json(staging / "betting_odds.json", odds_blob)
        _write_json(staging / "integrity_report.json", integrity)
        _write_json(staging / "diagnostics.json", diagnostics)
        _write_json(staging / "manifest.json", manifest)
        (staging / "SEALED").write_text(
            "SEALED FOR TERMINAL EVALUATION ONLY\n"
            "No model scoring was performed during construction.\n"
            "Freeze candidate models, betting rules, metrics, and slices "
            "before opening this holdout.\n"
        )
        staging.replace(output_dir)
    except Exception:
        # Retain staging on failure for forensic inspection; never partially
        # publish it as the final immutable directory.
        raise

    print(json.dumps(integrity, indent=2))
    print(f"Sealed forward holdout written to {output_dir}")
    return {
        "integrity": integrity,
        "diagnostics": diagnostics,
        "manifest": manifest,
        "output_dir": str(output_dir),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market-json", type=Path, required=True)
    parser.add_argument("--zip-dir", type=Path, default=DEFAULT_ZIP_DIR)
    parser.add_argument("--context-start", default=DEFAULT_CONTEXT_START)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--min-volume", type=float, default=1000.0)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=ROOT / "data" / "forward_holdout",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.min_volume <= 0:
        raise SystemExit("--min-volume must be positive")
    build(args)


if __name__ == "__main__":
    main()
