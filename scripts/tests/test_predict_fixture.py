"""Operational-contract tests for live fixture prediction."""

from __future__ import annotations

import pickle
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from predict_fixture import (  # noqa: E402
    assess_state_freshness,
    compute_bet,
    read_sqlite_state_metadata,
    read_tracker_state_metadata,
)


def _write_sqlite(path: Path, as_of: str, source_count: int) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE dates (id INTEGER PRIMARY KEY, date TEXT)")
        conn.execute("INSERT INTO dates(date) VALUES (?)", (as_of,))
        conn.execute("CREATE TABLE _meta (key TEXT PRIMARY KEY, value TEXT)")
        conn.executemany(
            "INSERT INTO _meta(key, value) VALUES (?, ?)",
            [
                ("source_match_count", str(source_count)),
                ("build_timestamp", "2026-07-20T00:00:00Z"),
                (
                    "same_day_order_version",
                    "date_then_match_id_lexicographic_v1",
                ),
            ],
        )


def _write_snapshot(path: Path, as_of: str, source_count: int) -> None:
    with path.open("wb") as handle:
        pickle.dump(
            {
                "as_of": as_of,
                "n_matches_walked": source_count,
                "built_at": "2026-07-20T00:00:00Z",
                "source_dirs": ["/sealed/source"],
                "same_day_order_version": (
                    "date_then_match_id_lexicographic_v1"
                ),
            },
            handle,
        )


def test_state_metadata_and_freshness_use_older_component(tmp_path: Path):
    sqlite_path = tmp_path / "player_stats_cache_v3.sqlite"
    snapshot_path = tmp_path / "tracker.pkl"
    _write_sqlite(sqlite_path, "2026-07-20", 100)
    _write_snapshot(snapshot_path, "2026-07-18", 100)

    sqlite_state = read_sqlite_state_metadata(tmp_path)
    tracker_state = read_tracker_state_metadata(snapshot_path)
    assessment = assess_state_freshness(
        "2026-07-24",
        sqlite_state,
        tracker_state,
        max_state_age_days=7,
    )

    assert assessment["status"] == "fresh"
    assert assessment["state_available_through"] == "2026-07-18"
    assert assessment["query_as_of"] == "2026-07-24"
    assert assessment["age_days"] == 6
    assert assessment["sqlite"]["age_days"] == 4
    assert assessment["tracker"]["age_days"] == 6


def test_state_older_than_budget_is_stale():
    assessment = assess_state_freshness(
        "2026-07-24",
        {"as_of": "2026-07-01", "source_match_count": 10},
        {"as_of": "2026-07-01", "source_match_count": 10},
        max_state_age_days=14,
    )
    assert assessment["status"] == "stale"
    assert assessment["age_days"] == 23


def test_state_newer_than_fixture_is_filtered_not_stale():
    assessment = assess_state_freshness(
        "2026-07-10",
        {"as_of": "2026-07-20", "source_match_count": 10},
        {"as_of": "2026-07-20", "source_match_count": 10},
        max_state_age_days=0,
    )
    assert assessment["status"] == "fresh"
    assert assessment["age_days"] == 0


def test_sqlite_tracker_source_count_mismatch_fails_closed():
    with pytest.raises(RuntimeError, match="source-count mismatch"):
        assess_state_freshness(
            "2026-07-24",
            {"as_of": "2026-07-20", "source_match_count": 100},
            {"as_of": "2026-07-20", "source_match_count": 99},
        )


def test_negative_age_budget_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        assess_state_freshness(
            "2026-07-24",
            {"as_of": "2026-07-20", "source_match_count": 100},
            {"as_of": "2026-07-20", "source_match_count": 100},
            max_state_age_days=-1,
        )


def test_a7_close_fixture_uses_strictly_positive_edge_and_normalized_market():
    decision = compute_bet(
        "A",
        "B",
        0.55,
        {"A": 2.10, "B": 1.80},
        top6_batting_elo_diff=5.0,
        polymarket_volume_usd=50_000,
    )
    inverse_sum = 1 / 2.10 + 1 / 1.80
    assert decision["market_implied_prob"]["A"] == pytest.approx(
        (1 / 2.10) / inverse_sum
    )
    assert decision["elo_regime"] == "close"
    assert decision["edge_threshold_pp"] == 0.0
    assert decision["shadow_bet_placed"] is True
    assert decision["shadow_bet_team"] == "A"
    assert decision["execution_authorized"] is False
    assert decision["bet_team"] is None


@pytest.mark.parametrize(
    ("model_probability", "expected_placed"),
    [
        (0.60, False),
        (0.600001, True),
    ],
)
def test_a7_mismatch_requires_edge_strictly_above_ten_points(
    model_probability: float,
    expected_placed: bool,
):
    decision = compute_bet(
        "A",
        "B",
        model_probability,
        {"A": 2.0, "B": 2.0},
        top6_batting_elo_diff=5.000001,
        polymarket_volume_usd=50_000,
    )
    assert decision["elo_regime"] == "mismatch"
    assert decision["edge_threshold_pp"] == 10.0
    assert decision["shadow_bet_placed"] is expected_placed


def test_a7_requires_primary_liquidity_and_fresh_state():
    low_volume = compute_bet(
        "A",
        "B",
        0.65,
        {"A": 2.0, "B": 2.0},
        top6_batting_elo_diff=2.0,
        polymarket_volume_usd=49_999.99,
    )
    assert low_volume["shadow_bet_placed"] is False
    assert low_volume["suppression_reasons"] == [
        "below_minimum_liquidity"
    ]

    stale = compute_bet(
        "A",
        "B",
        0.65,
        {"A": 2.0, "B": 2.0},
        top6_batting_elo_diff=2.0,
        polymarket_volume_usd=50_000,
        state_eligible=False,
    )
    assert stale["shadow_bet_placed"] is False
    assert stale["suppression_reasons"] == ["state_not_fresh"]


def test_a7_missing_volume_or_elo_suppresses_shadow_candidate():
    decision = compute_bet(
        "A",
        "B",
        0.65,
        {"A": 2.0, "B": 2.0},
    )
    assert decision["shadow_bet_placed"] is False
    assert decision["suppression_reasons"] == [
        "missing_top6_batting_elo_diff",
        "missing_liquidity",
    ]
