"""Focused schema-v5 reader and shrinkage tests for I8."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from stats_sqlite_backend import (  # noqa: E402
    I8_SCHEMA_VERSION,
    SCHEMA_SQL,
    SCHEMA_SQL_V5,
    SCHEMA_VERSION,
    _SQLiteBackend,
)


PRIOR = (0.4, 0.3, 0.1, 0.1, 0.05, 0.05)


def _create_db(path: Path, schema_version: int) -> None:
    with sqlite3.connect(path) as conn:
        conn.executescript(
            SCHEMA_SQL_V5
            if schema_version == I8_SCHEMA_VERSION
            else SCHEMA_SQL
        )
        conn.execute("INSERT INTO players VALUES (0, 'bat')")
        conn.execute("INSERT INTO players VALUES (1, 'bowl')")
        conn.execute("INSERT INTO dates VALUES (0, '2026-01-01')")
        meta = [("schema_version", str(schema_version))]
        meta.extend(
            (f"prior_p{suffix}", str(value))
            for suffix, value in zip(("0", "1", "2", "4", "6", "w"), PRIOR)
        )
        conn.executemany("INSERT INTO _meta VALUES (?, ?)", meta)


def _insert_i8_history(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute(
            "INSERT INTO batting "
            "(player_id,date_id,runs,balls,dismissals,c0,c1,c2,c4,c6,cw) "
            "VALUES (0,0,50,10,1,2,4,1,1,1,1)"
        )
        conn.execute(
            "INSERT INTO bowling "
            "(player_id,date_id,runs_given,balls_bowled,wickets,"
            "c0,c1,c2,c4,c6,cw) "
            "VALUES (1,0,40,10,1,6,1,1,1,0,1)"
        )
        conn.execute(
            "INSERT INTO batting_phase VALUES (0,0,0,6,2,1,1,0,0)"
        )
        conn.execute(
            "INSERT INTO bowling_phase VALUES (1,0,0,8,1,0,0,0,1)"
        )
        conn.execute(
            "INSERT INTO h2h "
            "(batter_id,bowler_id,date_id,runs,balls,dismissals,"
            "c0,c1,c2,c4,c6,cw) "
            "VALUES (0,1,0,12,10,1,7,1,0,1,0,1)"
        )


def test_schema_v5_phase_and_h2h_getters_follow_frozen_hierarchy(
    tmp_path: Path,
):
    path = tmp_path / "i8.sqlite"
    _create_db(path, I8_SCHEMA_VERSION)
    _insert_i8_history(path)
    backend = _SQLiteBackend(path)

    batter_parent = backend._shrink((2, 4, 1, 1, 1, 1), PRIOR, 30.0)
    bowler_parent = backend._shrink((6, 1, 1, 1, 0, 1), PRIOR, 30.0)
    expected_batter = backend._shrink(
        (6, 2, 1, 1, 0, 0), batter_parent, 30.0)
    expected_bowler = backend._shrink(
        (8, 1, 0, 0, 0, 1), bowler_parent, 30.0)
    h2h_parent = tuple(
        (bat + bowl) / 2
        for bat, bowl in zip(batter_parent, bowler_parent)
    )
    expected_h2h = backend._shrink(
        (7, 1, 0, 1, 0, 1), h2h_parent, 60.0)

    batter = backend.get_batter_phase_outcome_dist(
        "bat", "2026-01-02", 0)
    bowler = backend.get_bowler_phase_outcome_dist(
        "bowl", "2026-01-02", 35)
    h2h = backend.get_h2h_outcome_dist(
        "bat", "bowl", "2026-01-02")

    assert tuple(batter.values()) == pytest.approx(expected_batter)
    assert tuple(bowler.values()) == pytest.approx(expected_bowler)
    assert tuple(h2h.values()) == pytest.approx(expected_h2h)
    assert sum(batter.values()) == pytest.approx(1.0)
    assert sum(bowler.values()) == pytest.approx(1.0)
    assert sum(h2h.values()) == pytest.approx(1.0)


def test_unseen_phase_cell_falls_back_to_shrunk_player_parent(
    tmp_path: Path,
):
    path = tmp_path / "i8.sqlite"
    _create_db(path, I8_SCHEMA_VERSION)
    _insert_i8_history(path)
    backend = _SQLiteBackend(path)

    expected = backend._shrink((2, 4, 1, 1, 1, 1), PRIOR, 30.0)
    death = backend.get_batter_phase_outcome_dist(
        "bat", "2026-01-02", 96)

    assert tuple(death.values()) == pytest.approx(expected)


def test_schema_v4_remains_readable_but_rejects_i8_getters(tmp_path: Path):
    path = tmp_path / "v4.sqlite"
    _create_db(path, SCHEMA_VERSION)
    backend = _SQLiteBackend(path)

    assert backend.get_batter_outcome_dist(
        "unknown", "2026-01-02")["batter_p0"] == pytest.approx(PRIOR[0])
    with pytest.raises(RuntimeError, match="requires SQLite schema 5"):
        backend.get_batter_phase_outcome_dist(
            "unknown", "2026-01-02", 0)
    with pytest.raises(RuntimeError, match="requires SQLite schema 5"):
        backend.get_h2h_outcome_dist(
            "unknown", "other", "2026-01-02")
