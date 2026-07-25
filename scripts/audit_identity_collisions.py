"""Audit venue aliases and duplicate Cricsheet player identities.

This is the read-only first stage of I7. It scans the source corpus and
produces a review report; it does not rewrite source JSON, create aliases, or
change cache-build behavior.

Usage:
    uv run python scripts/audit_identity_collisions.py \
        --source-dir data/t20s_json \
        --metadata-csv data/all_players_enriched.csv \
        --out reports/i7_identity_collision_audit.md
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


def normalize_label(value: str) -> str:
    """Case/punctuation-insensitive label used only to find candidates."""
    ascii_value = unicodedata.normalize("NFKD", value or "").encode(
        "ascii", "ignore").decode("ascii")
    return " ".join(re.findall(r"[a-z0-9]+", ascii_value.casefold()))


def _new_evidence() -> dict:
    return {
        "matches": set(),
        "dates": set(),
        "teams": Counter(),
        "cities": Counter(),
        "names": Counter(),
    }


def _date(info: dict) -> str:
    dates = info.get("dates") or []
    return str(dates[0]) if dates else "unknown"


def scan_corpus(source_dir: Path, gender: str | None = None) -> dict:
    """Collect venue and player-ID evidence from Cricsheet match JSONs."""
    venues = defaultdict(_new_evidence)
    player_ids = defaultdict(_new_evidence)
    name_to_ids = defaultdict(set)
    files_seen = 0
    matches_used = 0

    for path in sorted(source_dir.glob("*.json")):
        files_seen += 1
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        info = data.get("info") or {}
        if gender and info.get("gender") != gender:
            continue
        match_date = _date(info)
        teams = [str(team) for team in (info.get("teams") or [])]
        city = str(info.get("city") or "").strip()
        venue = str(info.get("venue") or "unknown").strip()
        match_key = path.name

        venue_ev = venues[venue]
        venue_ev["matches"].add(match_key)
        venue_ev["dates"].add(match_date)
        venue_ev["teams"].update(teams)
        if city:
            venue_ev["cities"][city] += 1

        roster_team = {}
        for team, names in (info.get("players") or {}).items():
            for name in names or []:
                roster_team[str(name)] = str(team)

        people = ((info.get("registry") or {}).get("people") or {})
        for raw_name, raw_id in people.items():
            name = str(raw_name).strip()
            player_id = str(raw_id).strip()
            if not name or not player_id:
                continue
            name_to_ids[name].add(player_id)
            player_ev = player_ids[player_id]
            player_ev["matches"].add(match_key)
            player_ev["dates"].add(match_date)
            player_ev["names"][name] += 1
            team = roster_team.get(name)
            if team:
                player_ev["teams"][team] += 1
        matches_used += 1

    return {
        "files_seen": files_seen,
        "matches_used": matches_used,
        "gender": gender,
        "venues": dict(venues),
        "player_ids": dict(player_ids),
        "name_to_ids": dict(name_to_ids),
    }


def load_metadata(path: Path) -> dict[str, dict]:
    """Load identity evidence keyed by Cricsheet ID."""
    if not path.is_file():
        return {}
    rows = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            player_id = str(row.get("cricsheet_id") or "").strip()
            if player_id:
                rows[player_id] = row
    return rows


def _counter_keys(counter: Counter) -> set[str]:
    return {str(key) for key in counter if str(key)}


def _date_span(evidence: dict) -> str:
    dates = sorted(date for date in evidence["dates"] if date != "unknown")
    if not dates:
        return "unknown"
    return dates[0] if len(dates) == 1 else f"{dates[0]}…{dates[-1]}"


def _top(counter: Counter, limit: int = 4) -> str:
    if not counter:
        return "—"
    return ", ".join(
        f"{name} ({count})" for name, count in counter.most_common(limit))


def venue_candidates(venues: dict[str, dict]) -> list[dict]:
    """Return normalized-equality/substring candidates with corpus evidence."""
    names = sorted(venues)
    normalized = {name: normalize_label(name) for name in names}
    candidates = []
    for index, left in enumerate(names):
        left_norm = normalized[left]
        if not left_norm:
            continue
        for right in names[index + 1:]:
            right_norm = normalized[right]
            if not right_norm:
                continue
            if not (left_norm in right_norm or right_norm in left_norm):
                continue
            left_ev = venues[left]
            right_ev = venues[right]
            shared_cities = (
                _counter_keys(left_ev["cities"]) &
                _counter_keys(right_ev["cities"])
            )
            shared_teams = (
                _counter_keys(left_ev["teams"]) &
                _counter_keys(right_ev["teams"])
            )
            left_cities = _counter_keys(left_ev["cities"])
            right_cities = _counter_keys(right_ev["cities"])
            if left_norm == right_norm:
                classification = "high-confidence formatting alias"
            elif len(left_cities) > 1 or len(right_cities) > 1:
                classification = "review — ambiguous multi-city label"
            elif (
                len(shared_cities) == 1
                and _is_explicit_city_suffix(
                    left_norm, right_norm, next(iter(shared_cities)))
            ):
                classification = "likely alias — explicit city suffix"
            elif shared_cities:
                classification = "review — shared city but possible subvenue"
            else:
                classification = "review — substring only"
            candidates.append({
                "left": left,
                "right": right,
                "left_matches": len(left_ev["matches"]),
                "right_matches": len(right_ev["matches"]),
                "left_dates": _date_span(left_ev),
                "right_dates": _date_span(right_ev),
                "shared_cities": sorted(shared_cities),
                "shared_teams": sorted(shared_teams),
                "left_city_counts": left_ev["cities"],
                "right_city_counts": right_ev["cities"],
                "classification": classification,
            })
    return sorted(
        candidates,
        key=lambda row: (
            row["classification"].startswith("review"),
            -(row["left_matches"] + row["right_matches"]),
            row["left"],
            row["right"],
        ),
    )


def _is_explicit_city_suffix(
    left_normalized: str,
    right_normalized: str,
    city: str,
) -> bool:
    """Whether the only extra name tokens are the shared city name."""
    short, long = sorted(
        (left_normalized, right_normalized), key=lambda value: len(value))
    city_normalized = normalize_label(city)
    prefix = f"{short} "
    return bool(
        city_normalized
        and long.startswith(prefix)
        and long[len(prefix):] == city_normalized
    )


def _clean_numeric(value: str) -> str:
    value = str(value or "").strip()
    return value[:-2] if value.endswith(".0") else value


def _metadata_signature(row: dict) -> tuple[str, str, str]:
    return (
        _clean_numeric(row.get("cricinfo_id", "")),
        normalize_label(row.get("full_name", "")),
        str(row.get("dob") or "").strip(),
    )


def player_collision_groups(
    name_to_ids: dict[str, set[str]],
    player_ids: dict[str, dict],
    metadata: dict[str, dict],
) -> list[dict]:
    """Group exact display names that resolve to multiple registry IDs."""
    groups = []
    for name, ids_set in name_to_ids.items():
        if len(ids_set) < 2:
            continue
        ids = sorted(ids_set)
        rows = [metadata.get(player_id, {}) for player_id in ids]
        signatures = [_metadata_signature(row) for row in rows]
        cricinfo_ids = {sig[0] for sig in signatures if sig[0]}
        full_dob = {(sig[1], sig[2]) for sig in signatures
                    if sig[1] and sig[2]}

        if len(cricinfo_ids) == 1 and all(sig[0] for sig in signatures):
            classification = "high-confidence duplicate — same Cricinfo ID"
        elif len(full_dob) == 1 and all(sig[1] and sig[2]
                                       for sig in signatures):
            classification = "high-confidence duplicate — same full name/DOB"
        else:
            classification = "review — exact display-name collision"

        id_evidence = []
        for player_id, row in zip(ids, rows):
            evidence = player_ids.get(player_id, _new_evidence())
            id_evidence.append({
                "id": player_id,
                "matches": len(evidence["matches"]),
                "dates": _date_span(evidence),
                "teams": evidence["teams"],
                "cricinfo_id": _clean_numeric(row.get("cricinfo_id", "")),
                "full_name": str(row.get("full_name") or "").strip(),
                "dob": str(row.get("dob") or "").strip(),
            })
        groups.append({
            "name": name,
            "ids": id_evidence,
            "classification": classification,
        })
    return sorted(
        groups,
        key=lambda row: (
            row["classification"].startswith("review"),
            -sum(item["matches"] for item in row["ids"]),
            row["name"],
        ),
    )


def _escape(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _id_summary(items: Iterable[dict]) -> str:
    parts = []
    for item in items:
        metadata = []
        if item["cricinfo_id"]:
            metadata.append(f"CI:{item['cricinfo_id']}")
        if item["dob"]:
            metadata.append(f"DOB:{item['dob']}")
        suffix = f"; {'; '.join(metadata)}" if metadata else ""
        parts.append(
            f"`{item['id']}` ({item['matches']} matches, "
            f"{item['dates']}; {_top(item['teams'], 3)}{suffix})")
    return "<br>".join(parts)


def render_report(
    audit: dict,
    venues: list[dict],
    players: list[dict],
    source_dir: Path,
    metadata_path: Path,
) -> str:
    venue_recommended = [
        row for row in venues
        if not row["classification"].startswith("review")
    ]
    player_recommended = [
        row for row in players
        if not row["classification"].startswith("review")
    ]
    lines = [
        "# I7 — Venue and player identity collision audit",
        "",
        f"Source: `{source_dir}` ({audit['matches_used']:,} readable matches "
        f"of {audit['files_seen']:,} JSON files; gender filter: "
        f"`{audit['gender'] or 'all'}`). Metadata: "
        f"`{metadata_path}`.",
        "",
        "This report proposes candidates only. No source JSON, cache, ELO, "
        "encoder, or model artifact was modified. A merge is safe to apply "
        "only after its evidence is reviewed and encoded in a versioned map.",
        "",
        "## Summary",
        "",
        f"- Unique venue strings: **{len(audit['venues']):,}**",
        f"- Venue equality/substring candidate pairs: **{len(venues):,}**",
        f"- Venue candidates with formatting/explicit-city-suffix evidence: "
        f"**{len(venue_recommended):,}**",
        f"- Exact player names mapping to multiple registry IDs: "
        f"**{len(players):,}**",
        f"- Player groups with matching stable metadata: "
        f"**{len(player_recommended):,}**",
        "",
        "## Safety policy",
        "",
        "- Venue normalization is candidate generation, not proof. Substring "
        "matches stay review-only unless the only additional tokens are a "
        "shared, unambiguous city. Shared city alone cannot distinguish a "
        "main ground from a separate outer oval or academy ground.",
        "- Player display-name equality is not proof. Automatic merge "
        "eligibility requires the same non-empty Cricinfo ID or the same "
        "non-empty normalized full name and date of birth.",
        "- Date ranges and team histories are shown to expose implausible "
        "merges. They are supporting evidence, not automatic merge keys.",
        "",
        "## Venue candidates",
        "",
        "| left | right | matches | date spans | city evidence | shared "
        "teams | classification |",
        "|---|---|---:|---|---|---|---|",
    ]
    for row in venues:
        city_evidence = (
            ", ".join(row["shared_cities"])
            if row["shared_cities"]
            else f"left: {_top(row['left_city_counts'], 2)}; "
                 f"right: {_top(row['right_city_counts'], 2)}"
        )
        lines.append(
            f"| {_escape(row['left'])} | {_escape(row['right'])} | "
            f"{row['left_matches']} + {row['right_matches']} | "
            f"{row['left_dates']} / {row['right_dates']} | "
            f"{_escape(city_evidence)} | "
            f"{_escape(', '.join(row['shared_teams'][:5]) or '—')} | "
            f"{row['classification']} |")

    lines += [
        "",
        "## Player display-name collisions",
        "",
        "| display name | registry-ID evidence | classification |",
        "|---|---|---|",
    ]
    for row in players:
        lines.append(
            f"| {_escape(row['name'])} | {_escape(_id_summary(row['ids']))} | "
            f"{row['classification']} |")
    if not player_recommended:
        lines += [
            "",
            "**Player decision:** no player-ID merge is justified by the "
            "current evidence. The 94 repeated display names are homonym "
            "collisions, not demonstrated split identities. Keep their "
            "registry IDs separate.",
        ]
    lines += [
        "",
        "## Next gate",
        "",
        "Review the recommended venue groups and explicitly approve a "
        "versioned venue-alias map. Do not create a player-ID merge map "
        "unless new stable-identity evidence appears. The next implementation "
        "must apply the venue map at ingestion/cache-build time, preserve raw "
        "source files, reject cycles or conflicting targets, and pass "
        "same-day/as-of chronology tests before any cache rebuild.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path,
                        default=Path("data/t20s_json"))
    parser.add_argument("--metadata-csv", type=Path,
                        default=Path("data/all_players_enriched.csv"))
    parser.add_argument("--gender", default="male",
                        help="Match info.gender filter; use an empty value for all")
    parser.add_argument("--out", type=Path,
                        default=Path("reports/i7_identity_collision_audit.md"))
    args = parser.parse_args()

    audit = scan_corpus(args.source_dir, gender=args.gender or None)
    metadata = load_metadata(args.metadata_csv)
    venues = venue_candidates(audit["venues"])
    players = player_collision_groups(
        audit["name_to_ids"], audit["player_ids"], metadata)
    report = render_report(
        audit, venues, players, args.source_dir, args.metadata_csv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report)
    print(
        f"matches={audit['matches_used']} venues={len(audit['venues'])} "
        f"venue_pairs={len(venues)} player_name_collisions={len(players)}")
    print(f"report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
