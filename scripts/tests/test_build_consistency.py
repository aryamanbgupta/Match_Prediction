"""Verify that `batting_match_log` / `bowling_match_log` rows sum to the
denormalized `batting.recent_*` / `bowling.recent_*` columns in the
current `models/player_stats_cache_v3.sqlite`.

This is the permanent form of the build-time check in
`scripts/build_stats_cache.py:_verify_log_denormalized_consistency`. The
build-time version aborts a poisoned build; this test catches any drift
introduced by:

  * A schema change that doesn't go through build_stats_cache.py (e.g.
    someone manually ALTER TABLE'ing or a migration script).
  * A change to `PlayerStatsTracker.end_match` ordering that updates
    recent_* but not the log, or vice versa.
  * Accidental writes to the DB after the build (shouldn't happen —
    the reader path is read-only — but the test is cheap).

Samples 500 batting + 500 bowling rows with non-zero recent-form sums,
and for each sums the 5 most recent log rows strictly before that
date_id. Same contract as the build-time check.
"""
from __future__ import annotations

import random
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "models" / "player_stats_cache_v3.sqlite"

SPECS = (
    # (stats_table, log_table, denorm_cols, log_cols)
    ("batting", "batting_match_log",
     ("recent_runs", "recent_balls", "recent_dismissals"),
     ("runs", "balls", "dismissals")),
    ("bowling", "bowling_match_log",
     ("recent_runs_given", "recent_balls_bowled", "recent_wickets"),
     ("runs_given", "balls_bowled", "wickets")),
)


def check_consistency(conn, sample_n: int = 500, seed: int = 0xC0FFEE) -> int:
    """Returns the number of rows sampled. Raises AssertionError on any
    mismatch with the offending (pid, date_id) in the message."""
    rng = random.Random(seed)
    total_checked = 0
    for stats_table, log_table, denorm_cols, log_cols in SPECS:
        rows = conn.execute(
            f"SELECT player_id, date_id, {', '.join(denorm_cols)} "
            f"FROM {stats_table} "
            f"WHERE {denorm_cols[1]} > 0 "
            f"ORDER BY RANDOM() LIMIT ?",
            (sample_n,),
        ).fetchall()
        for row in rows:
            pid, did = row[0], row[1]
            expected = tuple(row[2:5])
            log_rows = conn.execute(
                f"SELECT {', '.join(log_cols)} FROM {log_table} "
                f"WHERE player_id=? AND date_id<? "
                f"ORDER BY date_id DESC, intra_date_idx DESC LIMIT 5",
                (pid, did),
            ).fetchall()
            actual = tuple(sum(r[i] for r in log_rows) for i in range(3))
            assert actual == expected, (
                f"{stats_table}/{log_table} drift at pid={pid} date_id={did}: "
                f"denormalized {denorm_cols}={expected}, "
                f"log sum-of-last-5 {log_cols}={actual}"
            )
        total_checked += len(rows)
    return total_checked


def test_build_consistency():
    assert DB_PATH.exists(), (
        f"expected SQLite cache at {DB_PATH}. Run "
        f"`uv run python scripts/build_stats_cache.py` first."
    )
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        n = check_consistency(conn, sample_n=500)
    finally:
        conn.close()
    # Both tables have enough non-zero rows in the current corpus that
    # we expect sample_n * 2 ≈ 1000. If either came back empty, the DB
    # may be mis-built.
    assert n >= 500, f"sampled only {n} rows — DB may be missing data"


if __name__ == "__main__":
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        n = check_consistency(conn, sample_n=500)
        print(f"PASS: {n} (pid, date_id) rows match — log sum ≡ denormalized.")
    except AssertionError as e:
        print(f"FAIL: {e}")
        sys.exit(1)
    finally:
        conn.close()
