"""Schema v3 match-log unit tests.

Validates the `batting_match_log` / `bowling_match_log` tables + their
getters on `_SQLiteBackend`, introduced in Phase B deliverable §1.

Uses a tiny synthetic SQLite (temp file) built from the production
`SCHEMA_SQL`. Keeps the tests self-contained — no dependency on the
full ~50 MB production cache file.

Six assertions per the Phase B plan:

  1. get_batting_match_log_recent returns rows newest-first ordered by
     (date_id DESC, intra_date_idx DESC).
  2. `limit` parameter honored (request 3, get 3 even when 5 available).
  3. Strict `date_id < ?` — querying for date D returns matches BEFORE
     D, not on D.
  4. Sum of log rows equals the denormalized recent_* counters stored
     on the batting/bowling row for the corresponding snapshot date
     (consistency between log and sum columns).
  5. EXPLAIN QUERY PLAN reports `SEARCH TABLE ... USING INDEX` (no
     full scan — the primary-key scan is essential for p50 < 10 µs).
  6. Same-day matches correctly ordered by intra_date_idx within a
     date (tie-break rule for the ORDER BY).
"""
from __future__ import annotations

import sqlite3
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from stats_sqlite_backend import (  # noqa: E402
    SCHEMA_SQL,
    SCHEMA_VERSION,
    _Q_BATTING_LOG_RECENT,
    _Q_BOWLING_LOG_RECENT,
    _SQLiteBackend,
)


def _build_fixture_db(tmp_path: Path) -> Path:
    """Tiny fixture: 1 player (pid=0), 6 match dates, batting + bowling log
    rows across them. Populates lookup tables so _SQLiteBackend lookups
    work. Returns the SQLite file path."""
    db_path = tmp_path / "fixture.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(SCHEMA_SQL)

    # Lookup tables
    conn.execute("INSERT INTO players (id, player_id) VALUES (0, 'p0')")
    conn.execute("INSERT INTO players (id, player_id) VALUES (1, 'p1')")

    # Dates 2024-01-01 through 2024-01-06 with date_id 0..5
    date_rows = [(i, f"2024-01-0{i + 1}") for i in range(6)]
    conn.executemany("INSERT INTO dates (id, date) VALUES (?, ?)", date_rows)

    # Batting match log — p0 plays on each date. Two matches on 2024-01-05
    # (date_id=4) to test the intra_date_idx ordering.
    # (pid, date_id, intra_idx, runs, balls, dismissals)
    batting_log = [
        (0, 0, 0, 10, 12, 0),   # 2024-01-01
        (0, 1, 0, 20, 18, 0),   # 2024-01-02
        (0, 2, 0, 30, 22, 1),   # 2024-01-03
        (0, 3, 0, 40, 26, 0),   # 2024-01-04
        (0, 4, 0, 50, 30, 0),   # 2024-01-05 match 1
        (0, 4, 1, 60, 34, 1),   # 2024-01-05 match 2
    ]
    conn.executemany(
        "INSERT INTO batting_match_log VALUES (?, ?, ?, ?, ?, ?)",
        batting_log,
    )

    # Bowling log — p1 bowls at the same matches.
    # (pid, date_id, intra_idx, runs_given, balls_bowled, wickets)
    bowling_log = [
        (1, 0, 0, 30, 24, 1),
        (1, 1, 0, 35, 24, 0),
        (1, 2, 0, 28, 24, 2),
        (1, 3, 0, 40, 24, 1),
        (1, 4, 0, 32, 24, 0),
        (1, 4, 1, 38, 24, 1),
    ]
    conn.executemany(
        "INSERT INTO bowling_match_log VALUES (?, ?, ?, ?, ?, ?)",
        bowling_log,
    )

    # Denormalized recent_* on batting row for p0 on 2024-01-06 (date_id=5).
    # Should equal the sum of the 5 most recent log rows (date_id < 5):
    # (0,1,2,3,4-intra0,4-intra1) → last 5 = (1,2,3,4-intra0,4-intra1)
    #   runs:        20+30+40+50+60 = 200
    #   balls:       18+22+26+30+34 = 130
    #   dismissals:   0+ 1+ 0+ 0+ 1 =   2
    conn.execute(
        "INSERT INTO batting VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (0, 5, 210, 142, 2, 200, 130, 2),
    )
    # Denormalized recent_* on bowling row for p1 on 2024-01-06.
    # last 5 = (1,2,3,4-intra0,4-intra1)
    #   runs_given:    35+28+40+32+38 = 173
    #   balls_bowled:  24*5            = 120
    #   wickets:        0+ 2+ 1+ 0+ 1  =   4
    conn.execute(
        "INSERT INTO bowling VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (1, 5, 203, 144, 5, 173, 120, 4),
    )

    # _meta with schema_version.
    conn.execute(
        "INSERT INTO _meta VALUES (?, ?)",
        ("schema_version", str(SCHEMA_VERSION)),
    )

    conn.commit()
    conn.close()
    return db_path


# ---------------------------------------------------------------------------
# Assertions

def test_schema_version_is_3():
    """SCHEMA_VERSION must have been bumped from 2 to 3."""
    assert SCHEMA_VERSION == 3, (
        f"SCHEMA_VERSION is {SCHEMA_VERSION}, expected 3 for Phase B"
    )


def test_log_returns_newest_first():
    """Assertion 1: result ordered by (date_id DESC, intra_date_idx DESC)."""
    with tempfile.TemporaryDirectory() as tmp:
        db = _build_fixture_db(Path(tmp))
        backend = _SQLiteBackend(str(db))
        rows = backend.get_batting_match_log_recent("p0", "2024-01-06", limit=5)

    # 5 most recent before 2024-01-06: newest-first.
    # date_id 4 intra_idx 1  → runs=60
    # date_id 4 intra_idx 0  → runs=50
    # date_id 3             → runs=40
    # date_id 2             → runs=30
    # date_id 1             → runs=20
    assert [r["runs"] for r in rows] == [60, 50, 40, 30, 20], (
        f"rows not newest-first: {[r['runs'] for r in rows]}"
    )


def test_limit_parameter_honored():
    """Assertion 2: request 3 rows, get 3 even when 5 available."""
    with tempfile.TemporaryDirectory() as tmp:
        db = _build_fixture_db(Path(tmp))
        backend = _SQLiteBackend(str(db))
        rows = backend.get_batting_match_log_recent("p0", "2024-01-06", limit=3)

    assert len(rows) == 3
    # Newest 3: date_id 4 intra_idx 1, date_id 4 intra_idx 0, date_id 3
    assert [r["runs"] for r in rows] == [60, 50, 40]


def test_strict_date_exclusion():
    """Assertion 3: date_id < ? is strict; target-date matches excluded."""
    with tempfile.TemporaryDirectory() as tmp:
        db = _build_fixture_db(Path(tmp))
        backend = _SQLiteBackend(str(db))
        # Query at 2024-01-05 (date_id=4). Both same-day log rows have
        # date_id=4, so they must NOT appear.
        rows = backend.get_batting_match_log_recent("p0", "2024-01-05", limit=5)
    assert len(rows) == 4, (
        f"expected 4 rows (date_id 0..3), got {len(rows)}: "
        f"{[r['runs'] for r in rows]}"
    )
    assert [r["runs"] for r in rows] == [40, 30, 20, 10], (
        f"same-date matches leaked into result: {[r['runs'] for r in rows]}"
    )


def test_log_sum_matches_denorm_recent_columns():
    """Assertion 4: sum of 5 most recent log rows == denormalized recent_*
    stored on the batting/bowling row at the same snapshot date. This is
    the consistency contract between the log and the sum columns."""
    with tempfile.TemporaryDirectory() as tmp:
        db = _build_fixture_db(Path(tmp))
        backend = _SQLiteBackend(str(db))
        bat = backend.get_batting_match_log_recent("p0", "2024-01-06", 5)
        bowl = backend.get_bowling_match_log_recent("p1", "2024-01-06", 5)

        # Read the denormalized recent_* directly off the batting/bowling
        # row for date 2024-01-06.
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        b_row = conn.execute(
            "SELECT recent_runs, recent_balls, recent_dismissals "
            "FROM batting WHERE player_id=0 AND date_id=5"
        ).fetchone()
        w_row = conn.execute(
            "SELECT recent_runs_given, recent_balls_bowled, recent_wickets "
            "FROM bowling WHERE player_id=1 AND date_id=5"
        ).fetchone()
        conn.close()

    sum_runs = sum(r["runs"] for r in bat)
    sum_balls = sum(r["balls"] for r in bat)
    sum_dism = sum(r["dismissals"] for r in bat)
    assert (sum_runs, sum_balls, sum_dism) == b_row, (
        f"batting log-sum {(sum_runs, sum_balls, sum_dism)} != "
        f"denormalized recent_* {b_row}"
    )

    sum_rg = sum(r["runs_given"] for r in bowl)
    sum_bb = sum(r["balls_bowled"] for r in bowl)
    sum_wk = sum(r["wickets"] for r in bowl)
    assert (sum_rg, sum_bb, sum_wk) == w_row, (
        f"bowling log-sum {(sum_rg, sum_bb, sum_wk)} != "
        f"denormalized recent_* {w_row}"
    )


def test_query_plan_uses_index():
    """Assertion 5: EXPLAIN QUERY PLAN must report `USING INDEX` (PK scan,
    not full scan). A full scan here would blow past the p50<10µs budget."""
    with tempfile.TemporaryDirectory() as tmp:
        db = _build_fixture_db(Path(tmp))
        # Open read-write so ANALYZE can write the sqlite_stat1 table.
        # The production build path does this once and then the reader
        # opens read-only.
        conn = sqlite3.connect(str(db))
        conn.execute("ANALYZE")
        conn.commit()

        for name, query in (
            ("batting_match_log", _Q_BATTING_LOG_RECENT),
            ("bowling_match_log", _Q_BOWLING_LOG_RECENT),
        ):
            plan = conn.execute(
                f"EXPLAIN QUERY PLAN {query}", (0, 5, 5)
            ).fetchall()
            plan_text = " | ".join(row[3] for row in plan)
            # SQLite reports "USING PRIMARY KEY" for WITHOUT ROWID tables
            # (which both match logs are). Either phrase is acceptable.
            assert (
                "USING INDEX" in plan_text
                or "USING PRIMARY KEY" in plan_text
            ), f"{name} query plan lacks index usage: {plan_text}"
            # A full-table scan would say "SCAN TABLE <t>" without
            # "USING ..." — guard against that.
            assert "SEARCH" in plan_text, (
                f"{name} query plan isn't an indexed SEARCH: {plan_text}"
            )
        conn.close()


def test_same_day_intra_date_ordering():
    """Assertion 6: within a single date, rows are ordered by intra_date_idx
    DESC (matches the ORDER BY tie-break). Verifies the fixture's two
    2024-01-05 rows come out in the right order and intra_idx=1 wins."""
    with tempfile.TemporaryDirectory() as tmp:
        db = _build_fixture_db(Path(tmp))
        backend = _SQLiteBackend(str(db))
        # Limit=2 at 2024-01-06 must return the two 2024-01-05 matches.
        rows = backend.get_batting_match_log_recent("p0", "2024-01-06", limit=2)

    assert len(rows) == 2
    # intra_idx=1 (runs=60) before intra_idx=0 (runs=50) — newest first.
    assert rows[0]["runs"] == 60
    assert rows[1]["runs"] == 50


if __name__ == "__main__":
    # Allow standalone runs for fast dev iteration.
    import traceback
    tests = [
        test_schema_version_is_3,
        test_log_returns_newest_first,
        test_limit_parameter_honored,
        test_strict_date_exclusion,
        test_log_sum_matches_denorm_recent_columns,
        test_query_plan_uses_index,
        test_same_day_intra_date_ordering,
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
            traceback.print_exc()
    if failed:
        sys.exit(1)
    print(f"\nAll {len(tests)} tests passed.")
