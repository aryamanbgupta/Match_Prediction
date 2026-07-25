"""Freeze I7's conservative venue candidates into a review-only v1 map.

The generated CSV is deliberately inactive. Cache builders must not consume it
until the proposal has been reviewed and its status changed in a later,
separate implementation commit.

Usage:
    uv run python scripts/propose_venue_aliases.py \
        --source-dir data/t20s_json \
        --csv-out config/identity/venue_aliases_v1.csv \
        --report-out reports/i7_venue_alias_proposal.md
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from audit_identity_collisions import (
    normalize_label,
    scan_corpus,
    venue_candidates,
)


RECOMMENDED_CLASSES = {
    "high-confidence formatting alias",
    "likely alias — explicit city suffix",
}
MAP_VERSION = "venue_aliases_v1"


class DisjointSet:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, value: str) -> str:
        self.parent.setdefault(value, value)
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _latest_date(evidence: dict) -> str:
    dates = [date for date in evidence["dates"] if date != "unknown"]
    return max(dates, default="")


def choose_canonical(component: set[str], venues: dict[str, dict]) -> str:
    """Prefer the most specific label, then recent usage and volume."""
    return max(
        component,
        key=lambda name: (
            len(normalize_label(name).split()),
            _latest_date(venues[name]),
            len(venues[name]["matches"]),
            len(name),
            name,
        ),
    )


def build_proposal(
    venues: dict[str, dict],
    candidates: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Return alias rows and component summaries from conservative edges."""
    selected = [
        row for row in candidates
        if row["classification"] in RECOMMENDED_CLASSES
    ]
    dsu = DisjointSet()
    edge_classes = defaultdict(set)
    for row in selected:
        dsu.union(row["left"], row["right"])
        edge_classes[row["left"]].add(row["classification"])
        edge_classes[row["right"]].add(row["classification"])

    components = defaultdict(set)
    for name in dsu.parent:
        components[dsu.find(name)].add(name)

    aliases = []
    summaries = []
    for component in components.values():
        canonical = choose_canonical(component, venues)
        canonical_count = len(venues[canonical]["matches"])
        total_matches = sum(len(venues[name]["matches"]) for name in component)
        city_names = sorted({
            city
            for name in component
            for city in venues[name]["cities"]
        })
        component_aliases = sorted(component - {canonical})
        summaries.append({
            "canonical": canonical,
            "aliases": component_aliases,
            "total_matches": total_matches,
            "canonical_matches": canonical_count,
            "recovered_matches": total_matches - canonical_count,
            "cities": city_names,
            "last_date": _latest_date(venues[canonical]),
        })
        for alias in component_aliases:
            aliases.append({
                "version": MAP_VERSION,
                "alias": alias,
                "canonical": canonical,
                "status": "proposed",
                "evidence": "; ".join(sorted(edge_classes[alias])),
                "alias_matches": len(venues[alias]["matches"]),
                "canonical_matches": canonical_count,
                "component_matches": total_matches,
                "cities": "; ".join(city_names),
            })

    aliases.sort(key=lambda row: (row["canonical"], row["alias"]))
    summaries.sort(
        key=lambda row: (-row["recovered_matches"], row["canonical"]))
    validate_proposal(aliases)
    return aliases, summaries


def validate_proposal(rows: list[dict]) -> None:
    """Reject cycles, conflicting targets, and canonical-as-alias chains."""
    targets = {}
    for row in rows:
        alias = row["alias"]
        canonical = row["canonical"]
        if alias == canonical:
            raise ValueError(f"self-alias is not allowed: {alias}")
        prior = targets.setdefault(alias, canonical)
        if prior != canonical:
            raise ValueError(
                f"conflicting targets for {alias}: {prior} vs {canonical}")
    aliases = set(targets)
    chained = aliases & set(targets.values())
    if chained:
        raise ValueError(
            f"canonical targets must not also be aliases: {sorted(chained)}")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "version", "alias", "canonical", "status", "evidence",
        "alias_matches", "canonical_matches", "component_matches", "cities",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def render_report(
    summaries: list[dict],
    rows: list[dict],
    source_dir: Path,
    csv_path: Path,
) -> str:
    recovered = sum(row["recovered_matches"] for row in summaries)
    lines = [
        "# I7 — Proposed venue alias map v1",
        "",
        f"Source: `{source_dir}`. Frozen proposal: `{csv_path}`.",
        "",
        "This map is **review-only and inactive**. Every CSV row has "
        "`status=proposed`; no cache builder or model consumes it.",
        "",
        "## Summary",
        "",
        f"- Canonical venue components: **{len(summaries)}**",
        f"- Alias strings proposed: **{len(rows)}**",
        f"- Historical matches outside the selected current canonical labels: "
        f"**{recovered:,}**",
        "- Canonical choice: most specific spelling, then most recent corpus "
        "usage and match volume.",
        "- Excluded: generic/multi-city labels, substring-only pairs, and "
        "possible subvenues.",
        "",
        "## Components",
        "",
        "| canonical | aliases | matches after merge | history recovered | "
        "cities | last canonical use |",
        "|---|---|---:|---:|---|---|",
    ]
    for row in summaries:
        aliases = "<br>".join(row["aliases"])
        cities = ", ".join(row["cities"]) or "—"
        lines.append(
            f"| {row['canonical']} | {aliases} | {row['total_matches']} | "
            f"{row['recovered_matches']} | {cities} | {row['last_date']} |")
    lines += [
        "",
        "## Activation gate",
        "",
        "Review every component, with special attention to components listing "
        "multiple city spellings. Activation requires a separate commit that "
        "changes approved rows from `proposed` to `active`, validates the map "
        "at load time, and applies it consistently to ball parsing, match "
        "features, cache snapshots, live fixtures, and evaluation match IDs.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path,
                        default=Path("data/t20s_json"))
    parser.add_argument("--gender", default="male")
    parser.add_argument("--csv-out", type=Path,
                        default=Path("config/identity/venue_aliases_v1.csv"))
    parser.add_argument("--report-out", type=Path,
                        default=Path("reports/i7_venue_alias_proposal.md"))
    args = parser.parse_args()

    audit = scan_corpus(args.source_dir, gender=args.gender or None)
    candidates = venue_candidates(audit["venues"])
    aliases, summaries = build_proposal(audit["venues"], candidates)
    write_csv(args.csv_out, aliases)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(
        render_report(
            summaries, aliases, args.source_dir, args.csv_out))
    print(
        f"components={len(summaries)} aliases={len(aliases)} "
        f"proposal={args.csv_out}")
    print(f"report -> {args.report_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
