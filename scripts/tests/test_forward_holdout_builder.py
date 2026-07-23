"""Guardrail tests for scripts/build_forward_holdout.py."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import build_forward_holdout as forward


def _row(**overrides):
    row = {
        "event_id": "e1",
        "event_slug": "crict20blast-a-b-2026-06-01",
        "event_title": "A vs B",
        "market_id": "m1",
        "market_question": "A vs B",
        "date": "2026-06-01",
        "scheduled_start_timestamp": "2026-06-01T12:00:00Z",
        "team1": "A",
        "team2": "B",
        "winner": "A",
        "prematch_price_team1": 0.55,
        "prematch_price_team2": 0.45,
        "price_timestamp": "2026-06-01T11:00:00Z",
        "price_lag_seconds": 3600,
        "volume_usd": 100_000,
        "gender_scope": "male",
        "low_liquidity": False,
    }
    row.update(overrides)
    return row


def test_valid_strict_row_passes():
    assert (
        forward.strict_market_rejection(
            _row(), "2026-06-01", "2026-07-13", 1000
        )
        is None
    )


def test_post_start_quote_is_rejected():
    assert (
        forward.strict_market_rejection(
            _row(
                price_timestamp="2026-06-01T12:00:01Z",
                price_lag_seconds=-1,
            ),
            "2026-06-01",
            "2026-07-13",
            1000,
        )
        == "not_strictly_prematch"
    )


def test_womens_market_is_rejected_even_with_same_team_names():
    assert (
        forward.strict_market_rejection(
            _row(
                event_title="A Women vs B Women",
                market_question="A Women vs B Women",
            ),
            "2026-06-01",
            "2026-07-13",
            1000,
        )
        == "womens_market"
    )


def test_props_are_rejected():
    assert (
        forward.strict_market_rejection(
            _row(
                event_title="Total runs",
                market_question="Total runs",
                team1="Over",
                team2="Under",
                winner="Over",
            ),
            "2026-06-01",
            "2026-07-13",
            1000,
        )
        == "invalid_team_outcomes"
    )


def test_duplicate_selection_key_is_outcome_blind():
    low = _row(market_id="m-low", volume_usd=10_000, winner="A")
    high = _row(market_id="m-high", volume_usd=20_000, winner="B")
    assert forward.market_selection_key(high) > forward.market_selection_key(
        low
    )
    assert forward.market_selection_key(
        _row(winner="A")
    ) == forward.market_selection_key(_row(winner="B"))


def test_warwickshire_alias_matches_birmingham_bears():
    assert forward.canonical_team("Warwickshire") == "Birmingham Bears"


def test_forward_source_spelling_aliases_are_deterministic():
    assert forward.canonical_team("Mi New York") == "MI New York"
    assert forward.canonical_team("Czechia") == "Czech Republic"
    assert forward.canonical_team("Turkiye") == "Turkey"


def test_slug_fixture_date_handles_utc_midnight_rollover():
    row = _row(
        event_slug="crint-wst2-lka2-2026-06-13",
        date="2026-06-14",
        scheduled_start_timestamp="2026-06-14T00:30:00Z",
        price_timestamp="2026-06-14T00:00:00Z",
        price_lag_seconds=1800,
    )
    assert forward.fixture_date_from_market(row) == "2026-06-13"
    assert (
        forward.strict_market_rejection(
            row, "2026-06-01", "2026-07-13", 1000
        )
        is None
    )


def test_slug_fixture_date_cannot_be_far_from_scheduled_start():
    assert (
        forward.strict_market_rejection(
            _row(event_slug="crict20blast-a-b-2026-06-03"),
            "2026-06-01",
            "2026-07-13",
            1000,
        )
        == "fixture_date_start_mismatch"
    )
