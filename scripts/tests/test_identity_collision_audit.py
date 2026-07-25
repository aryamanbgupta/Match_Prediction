"""Tests for I7's read-only identity-collision audit."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audit_identity_collisions import (
    load_metadata,
    normalize_label,
    player_collision_groups,
    scan_corpus,
    venue_candidates,
)


def _write_match(
    path: Path,
    *,
    date: str,
    venue: str,
    city: str,
    people: dict[str, str],
    gender: str = "male",
) -> None:
    names = list(people)
    path.write_text(json.dumps({
        "info": {
            "dates": [date],
            "gender": gender,
            "venue": venue,
            "city": city,
            "teams": ["A", "B"],
            "players": {"A": names, "B": []},
            "registry": {"people": people},
        },
        "innings": [],
    }))


def _write_metadata(path: Path) -> None:
    columns = [
        "cricsheet_id", "name", "cricinfo_id", "unique_name",
        "full_name", "country", "dob", "batting_style", "bowling_style",
    ]
    rows = [
        ["id1", "A Player", "100", "", "Alpha Player", "", "1990-01-01",
         "", ""],
        ["id2", "A Player", "100", "", "Alpha Player", "", "1990-01-01",
         "", ""],
        ["id3", "Common Name", "300", "", "Common One", "", "1991-01-01",
         "", ""],
        ["id4", "Common Name", "400", "", "Common Two", "", "1992-01-01",
         "", ""],
    ]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)


def test_normalize_label_removes_case_punctuation_and_accents():
    assert normalize_label("  Bay Óval, Mount-Maunganui ") == (
        "bay oval mount maunganui")


def test_audit_classifies_only_stable_player_metadata_as_high_confidence(
    tmp_path,
):
    _write_match(
        tmp_path / "m1.json", date="2024-01-01", venue="Bay Oval",
        city="Mount Maunganui",
        people={"A Player": "id1", "Common Name": "id3"},
    )
    _write_match(
        tmp_path / "m2.json", date="2024-02-01",
        venue="Bay Oval, Mount Maunganui", city="Mount Maunganui",
        people={"A Player": "id2", "Common Name": "id4"},
    )
    metadata_path = tmp_path / "players.csv"
    _write_metadata(metadata_path)

    audit = scan_corpus(tmp_path)
    metadata = load_metadata(metadata_path)
    groups = player_collision_groups(
        audit["name_to_ids"], audit["player_ids"], metadata)
    by_name = {row["name"]: row for row in groups}

    assert len(groups) == 2
    assert by_name["A Player"]["classification"].startswith(
        "high-confidence duplicate")
    assert by_name["Common Name"]["classification"].startswith("review")


def test_venue_substring_with_shared_city_is_likely_alias(tmp_path):
    _write_match(
        tmp_path / "m1.json", date="2024-01-01", venue="Bay Oval",
        city="Mount Maunganui", people={"A": "id1"},
    )
    _write_match(
        tmp_path / "m2.json", date="2024-02-01",
        venue="Bay Oval, Mount Maunganui", city="Mount Maunganui",
        people={"B": "id2"},
    )
    _write_match(
        tmp_path / "m3.json", date="2024-03-01", venue="Oval",
        city="Elsewhere", people={"C": "id3"},
    )

    audit = scan_corpus(tmp_path)
    candidates = venue_candidates(audit["venues"])
    by_pair = {
        frozenset((row["left"], row["right"])): row
        for row in candidates
    }

    bay_pair = by_pair[frozenset(
        ("Bay Oval", "Bay Oval, Mount Maunganui"))]
    assert bay_pair["classification"] == (
        "likely alias — explicit city suffix")
    unrelated = by_pair[frozenset(("Bay Oval", "Oval"))]
    assert unrelated["classification"].startswith("review")


def test_shared_city_does_not_merge_distinct_subvenues(tmp_path):
    _write_match(
        tmp_path / "m1.json", date="2024-01-01", venue="Eden Park",
        city="Auckland", people={"A": "id1"},
    )
    _write_match(
        tmp_path / "m2.json", date="2024-02-01",
        venue="Eden Park Outer Oval", city="Auckland",
        people={"B": "id2"},
    )

    audit = scan_corpus(tmp_path)
    candidate = venue_candidates(audit["venues"])[0]
    assert candidate["classification"] == (
        "review — shared city but possible subvenue")


def test_gender_filter_matches_cache_build_scope(tmp_path):
    _write_match(
        tmp_path / "male.json", date="2024-01-01", venue="Male Ground",
        city="A", people={"A": "id1"}, gender="male",
    )
    _write_match(
        tmp_path / "female.json", date="2024-01-02", venue="Female Ground",
        city="B", people={"B": "id2"}, gender="female",
    )

    audit = scan_corpus(tmp_path, gender="male")
    assert audit["matches_used"] == 1
    assert set(audit["venues"]) == {"Male Ground"}
