"""Tests for I7's inactive, versioned venue-map proposal."""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from propose_venue_aliases import (
    build_proposal,
    choose_canonical,
    validate_proposal,
)


def _venue(matches, dates, cities):
    return {
        "matches": set(matches),
        "dates": set(dates),
        "teams": Counter(),
        "cities": Counter(cities),
        "names": Counter(),
    }


def test_choose_canonical_prefers_most_recent_label():
    venues = {
        "Bay Oval": _venue(["a", "b"], ["2021-01-01"], ["Mount Maunganui"]),
        "Bay Oval, Mount Maunganui": _venue(
            ["c"], ["2026-01-01"], ["Mount Maunganui"]),
    }
    assert choose_canonical(set(venues), venues) == (
        "Bay Oval, Mount Maunganui")


def test_choose_canonical_prefers_specific_city_even_if_short_name_is_newer():
    venues = {
        "Buffalo Park": _venue(
            ["a"], ["2026-01-01"], ["East London"]),
        "Buffalo Park, East London": _venue(
            ["b", "c"], ["2025-01-01"], ["East London"]),
    }
    assert choose_canonical(set(venues), venues) == (
        "Buffalo Park, East London")


def test_proposal_excludes_review_only_subvenue_edge():
    venues = {
        "Bay Oval": _venue(["a"], ["2021-01-01"], ["Mount Maunganui"]),
        "Bay Oval, Mount Maunganui": _venue(
            ["b"], ["2026-01-01"], ["Mount Maunganui"]),
        "Eden Park": _venue(["c"], ["2025-01-01"], ["Auckland"]),
        "Eden Park Outer Oval": _venue(
            ["d"], ["2026-01-01"], ["Auckland"]),
    }
    candidates = [
        {
            "left": "Bay Oval",
            "right": "Bay Oval, Mount Maunganui",
            "classification": "likely alias — explicit city suffix",
        },
        {
            "left": "Eden Park",
            "right": "Eden Park Outer Oval",
            "classification": "review — shared city but possible subvenue",
        },
    ]

    rows, summaries = build_proposal(venues, candidates)
    assert [(row["alias"], row["canonical"]) for row in rows] == [
        ("Bay Oval", "Bay Oval, Mount Maunganui"),
    ]
    assert len(summaries) == 1
    assert summaries[0]["recovered_matches"] == 1


def test_validate_proposal_rejects_conflicts_and_chains():
    with pytest.raises(ValueError, match="conflicting targets"):
        validate_proposal([
            {"alias": "A", "canonical": "B"},
            {"alias": "A", "canonical": "C"},
        ])
    with pytest.raises(ValueError, match="must not also be aliases"):
        validate_proposal([
            {"alias": "A", "canonical": "B"},
            {"alias": "B", "canonical": "C"},
        ])
