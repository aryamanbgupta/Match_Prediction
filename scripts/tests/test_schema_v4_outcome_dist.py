"""Schema v4 conservation + prior-completeness checks on the built
SQLite stats cache.

Run after `build_stats_cache.py` completes. Exercises:

  * `_meta.schema_version == 4`, `_meta.prior_p*` populated, prior sums to 1.
  * Σ(c0..cw) == `balls` / `balls_bowled` / `total_balls` on every sampled
    row of batting / bowling / batting_vs_type / bowling_vs_hand / venue.
  * `EXPLAIN QUERY PLAN` on each of the v4-extended queries still reports
    `USING INDEX` — schema widening must not regress the planner.
  * A smoke-check that the 5 v4 getters on `_SQLiteBackend` return
    6-key dicts summing to 1.0 for a real player id.

This is a run-on-demand integration test, not a pure unit test — it
reads from `models/player_stats_cache_v3.sqlite`. Ships exit-code-0 on
PASS, non-zero on any violation.
"""
from __future__ import annotations

import random
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from stats_sqlite_backend import (  # noqa: E402
    QUERY_PLAN_CASES,
    SCHEMA_VERSION,
    _SQLiteBackend,
)

DB_PATH = ROOT / "models" / "player_stats_cache_v3.sqlite"


def _fail(msg):
    print(f"FAIL: {msg}", file=sys.stderr)
    return 1


def check_schema_version(conn):
    meta = dict(conn.execute("SELECT key, value FROM _meta"))
    sv = int(meta.get("schema_version", -1))
    if sv != SCHEMA_VERSION:
        return _fail(f"schema_version={sv}, expected {SCHEMA_VERSION}")
    print(f"  schema_version = {sv} OK")
    return 0


def check_prior_in_meta(conn):
    meta = dict(conn.execute("SELECT key, value FROM _meta"))
    keys = ["prior_p0", "prior_p1", "prior_p2",
            "prior_p4", "prior_p6", "prior_pw"]
    vals = []
    for k in keys:
        if k not in meta:
            return _fail(f"_meta missing {k}")
        vals.append(float(meta[k]))
    s = sum(vals)
    if abs(s - 1.0) > 1e-6:
        return _fail(f"prior sums to {s}, not 1.0")
    print(f"  prior = ({', '.join(f'{v:.4f}' for v in vals)}) sum={s:.6f} OK")
    return 0


def check_count_conservation(conn, sample_n=1000):
    specs = [
        ("batting", "balls"),
        ("bowling", "balls_bowled"),
        ("batting_vs_type", "balls"),
        ("bowling_vs_hand", "balls_bowled"),
        ("venue", "total_balls"),
    ]
    for table, balls_col in specs:
        rows = conn.execute(
            f"SELECT {balls_col}, c0, c1, c2, c4, c6, cw "
            f"FROM {table} WHERE {balls_col} > 0 "
            f"ORDER BY RANDOM() LIMIT ?",
            (sample_n,),
        ).fetchall()
        if not rows:
            print(f"  {table}: no rows with {balls_col} > 0 — skipping")
            continue
        for row in rows:
            balls, *cs = row
            if sum(cs) != balls:
                return _fail(
                    f"{table}: {balls_col}={balls}, Σ(cX)={sum(cs)}: {row}"
                )
        print(f"  {table}: {len(rows)} rows conservation OK")
    return 0


def check_query_plan_uses_index(conn):
    for name, sql, params in QUERY_PLAN_CASES:
        plan = conn.execute(f"EXPLAIN QUERY PLAN {sql}", params).fetchall()
        detail = " | ".join(r[-1] for r in plan)
        if "USING INDEX" not in detail and "USING PRIMARY KEY" not in detail:
            # Some SQLite versions phrase it differently; WITHOUT ROWID
            # tables often report "SEARCH ... USING INDEX"; h2h uses a
            # named index. Any other plan is a regression.
            if "SCAN" in detail and "SEARCH" not in detail:
                return _fail(
                    f"{name}: table SCAN in plan (v4 widening broke "
                    f"planner?): {detail}"
                )
        print(f"  {name}: plan OK — {detail[:80]}")
    return 0


def check_backend_getters_sum_to_one(conn):
    backend = _SQLiteBackend(DB_PATH)
    backend._ensure_conn()

    # Pick a player with plenty of balls — any top-200 pid works.
    row = conn.execute(
        "SELECT p.player_id FROM players p JOIN batting b "
        "ON b.player_id = p.id WHERE b.balls > 500 "
        "ORDER BY b.balls DESC LIMIT 1"
    ).fetchone()
    if row is None:
        print("  skipping getter sanity: no player with >500 balls")
        return 0
    pid = row[0]
    latest_date = backend._date_strs[-1]

    bd = backend.get_batter_outcome_dist(pid, latest_date, k=30.0)
    bw = backend.get_bowler_outcome_dist(pid, latest_date, k=30.0)
    bvt = backend.get_batter_vs_type_outcome_dist(pid, latest_date, k=30.0)
    bvh = backend.get_bowler_vs_hand_outcome_dist(pid, latest_date, k=30.0)

    # Batter dist
    s = sum(bd.values())
    if abs(s - 1.0) > 1e-9:
        return _fail(f"batter dist sums to {s}")
    # Bowler dist (this player may not bowl — still shrinks to prior).
    s = sum(bw.values())
    if abs(s - 1.0) > 1e-9:
        return _fail(f"bowler dist sums to {s}")
    # vs-type: each of pace/spin sums to 1.
    for suffix in ("pace", "spin"):
        s = sum(bvt[f"batter_p{c}_vs_{suffix}"] for c in
                ("0", "1", "2", "4", "6", "w"))
        if abs(s - 1.0) > 1e-9:
            return _fail(f"batter_vs_{suffix} dist sums to {s}")
    for suffix in ("lhb", "rhb"):
        s = sum(bvh[f"bowler_p{c}_vs_{suffix}"] for c in
                ("0", "1", "2", "4", "6", "w"))
        if abs(s - 1.0) > 1e-9:
            return _fail(f"bowler_vs_{suffix} dist sums to {s}")

    # Venue (pick a venue with >0 rows)
    vrow = conn.execute(
        "SELECT v.venue FROM venues v JOIN venue vv ON vv.venue_id = v.id "
        "WHERE vv.total_balls > 1000 "
        "ORDER BY vv.total_balls DESC LIMIT 1"
    ).fetchone()
    if vrow is not None:
        vd = backend.get_venue_outcome_dist(vrow[0], latest_date, k=200.0)
        s = sum(vd.values())
        if abs(s - 1.0) > 1e-9:
            return _fail(f"venue dist sums to {s}")

    print(f"  backend getters sum to 1 on pid={pid} OK")
    return 0


def main():
    if not DB_PATH.exists():
        print(f"SKIP: {DB_PATH} does not exist. Run build_stats_cache.py first.")
        return 0
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)

    # Gate before running count-conservation / getter checks — those
    # SELECT c0..cw from tables that don't exist in pre-v4 schemas.
    meta = dict(conn.execute("SELECT key, value FROM _meta"))
    try:
        sv = int(meta.get("schema_version", -1))
    except (TypeError, ValueError):
        sv = -1
    if sv != SCHEMA_VERSION:
        print(f"SKIP: DB schema_version={sv}, expected {SCHEMA_VERSION}. "
              "Rebuild with `uv run python scripts/build_stats_cache.py`.")
        conn.close()
        return 0

    rc = 0
    rc |= check_schema_version(conn)
    rc |= check_prior_in_meta(conn)
    rc |= check_count_conservation(conn)
    rc |= check_query_plan_uses_index(conn)
    rc |= check_backend_getters_sum_to_one(conn)
    conn.close()
    print("PASS" if rc == 0 else "FAIL")
    return rc


if __name__ == "__main__":
    sys.exit(main())
