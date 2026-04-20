"""Phase-2 equivalence harness: SQLite backend vs legacy StatsProvider chunks.

Asserts every getter returns bit-exactly the same output as the chunk-based
StatsProvider. Float results must compare equal via `==`, not `math.isclose`:
both backends do the same sequence of arithmetic operations on the same int
counters, so any divergence is a real bug.

Structure:

  * `test_stratified` — hand-picked edge cases for every getter:
    boundary dates (before min, exact min, exact max, after max), unknown
    player_id, unknown venue, H2H pair that never occurred, date exactly
    on a snapshot vs 1 day before.

  * `test_random_sample` — 10_000 (method, args, date) triples drawn at
    random from the actual data (real player_ids, real dates, real
    venues). Any divergence is printed with both outputs, then the test
    fails.

Run:
    uv run python scripts/tests/test_sqlite_equivalence.py
or via pytest (will pick up the `test_*` functions).
"""
from __future__ import annotations

import random
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))

from stats_provider import StatsProvider  # noqa: E402
from stats_sqlite_backend import _SQLiteBackend  # noqa: E402


DB_PATH = ROOT / 'models' / 'player_stats_cache_v3.sqlite'
RANDOM_N = 10_000
RANDOM_SEED = 0xC0DE


# ---------------------------------------------------------------------------
# Helpers

def _load_both():
    print("loading chunked StatsProvider (v3) ...", flush=True)
    t0 = time.time()
    chunks = StatsProvider(str(ROOT / 'models'), version='v3')
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

    print("opening SQLite backend ...", flush=True)
    t0 = time.time()
    sql = _SQLiteBackend(str(DB_PATH))
    sql._ensure_conn()
    print(f"  opened in {time.time()-t0:.2f}s", flush=True)
    return chunks, sql


def _eq_dict(a: dict, b: dict) -> bool:
    if a.keys() != b.keys():
        return False
    return all(a[k] == b[k] for k in a)


# ---------------------------------------------------------------------------
# Stratified cases

def _stratified_cases(chunks: StatsProvider, sql: _SQLiteBackend):
    """Yield (label, chunks_fn, sql_fn) triples to compare."""
    dates_sorted = list(chunks.dates)  # already sorted
    min_d, max_d = dates_sorted[0], dates_sorted[-1]
    mid_d = dates_sorted[len(dates_sorted) // 2]

    before_min = (datetime.strptime(min_d, '%Y-%m-%d')
                  - timedelta(days=5)).strftime('%Y-%m-%d')
    after_max = (datetime.strptime(max_d, '%Y-%m-%d')
                 + timedelta(days=365)).strftime('%Y-%m-%d')
    one_before_mid = (datetime.strptime(mid_d, '%Y-%m-%d')
                      - timedelta(days=1)).strftime('%Y-%m-%d')

    # Sample a real, well-populated player by peeking at the SQLite
    # batting table (most rows = most queries in eval).
    import sqlite3
    rdb = sqlite3.connect(f'file:{DB_PATH}?mode=ro', uri=True)
    pid_int, = rdb.execute(
        "SELECT player_id FROM batting GROUP BY player_id "
        "ORDER BY COUNT(*) DESC LIMIT 1"
    ).fetchone()
    player_main, = rdb.execute(
        "SELECT player_id FROM players WHERE id = ?", (pid_int,)
    ).fetchone()
    pid_other_int, = rdb.execute(
        "SELECT player_id FROM batting WHERE player_id != ? "
        "GROUP BY player_id ORDER BY COUNT(*) DESC LIMIT 1", (pid_int,)
    ).fetchone()
    player_other, = rdb.execute(
        "SELECT player_id FROM players WHERE id = ?", (pid_other_int,)
    ).fetchone()
    venue_main, = rdb.execute(
        "SELECT v.venue FROM venues v JOIN venue vn ON vn.venue_id = v.id "
        "GROUP BY v.id ORDER BY COUNT(*) DESC LIMIT 1"
    ).fetchone()
    rdb.close()

    UNKNOWN_P = 'player_does_not_exist_xyz'
    UNKNOWN_V = 'venue_does_not_exist_xyz'

    def add(label, method, args):
        cases.append((label, method, args))

    cases = []

    # Batting edge cases
    for d_label, d in [('min', min_d), ('max', max_d), ('mid', mid_d),
                       ('before_min', before_min), ('after_max', after_max),
                       ('one_before_mid', one_before_mid)]:
        add(f'batting({d_label})', 'get_batting_stats', (player_main, d))
        add(f'bowling({d_label})', 'get_bowling_stats', (player_main, d))
        add(f'h2h({d_label})', 'get_h2h_stats',
            (player_main, player_other, d))
        add(f'venue_avg({d_label})', 'get_venue_avg_score', (venue_main, d))
        add(f'venue_profile({d_label})', 'get_venue_profile', (venue_main, d))
        add(f'bat_vs_type({d_label})', 'get_batting_vs_type_stats',
            (player_main, d))
        add(f'bowl_vs_hand({d_label})', 'get_bowling_vs_hand_stats',
            (player_main, d))
        add(f'bat_elo({d_label})', 'get_batting_elo', (player_main, d))
        add(f'bowl_elo({d_label})', 'get_bowling_elo', (player_main, d))

    # Unknown player / venue / H2H pair
    add('batting(unknown_p)', 'get_batting_stats', (UNKNOWN_P, mid_d))
    add('bowling(unknown_p)', 'get_bowling_stats', (UNKNOWN_P, mid_d))
    add('h2h(unknown_batter)', 'get_h2h_stats', (UNKNOWN_P, player_other, mid_d))
    add('h2h(unknown_bowler)', 'get_h2h_stats', (player_main, UNKNOWN_P, mid_d))
    add('venue_avg(unknown_v)', 'get_venue_avg_score', (UNKNOWN_V, mid_d))
    add('venue_profile(unknown_v)', 'get_venue_profile', (UNKNOWN_V, mid_d))
    add('bat_elo(unknown)', 'get_batting_elo', (UNKNOWN_P, mid_d))
    add('bowl_elo(unknown)', 'get_bowling_elo', (UNKNOWN_P, mid_d))

    # Datetime vs string date equivalence
    mid_dt = datetime.strptime(mid_d, '%Y-%m-%d')
    add('batting(datetime)', 'get_batting_stats', (player_main, mid_dt))
    add('bat_elo(datetime)', 'get_batting_elo', (player_main, mid_dt))

    return cases


def test_stratified():
    chunks, sql = _load_both()
    cases = _stratified_cases(chunks, sql)
    print(f"\nstratified: {len(cases)} cases", flush=True)
    failures = []
    for label, method, args in cases:
        a = getattr(chunks, method)(*args)
        b = getattr(sql, method)(*args)
        eq = _eq_dict(a, b) if isinstance(a, dict) else (a == b)
        if not eq:
            failures.append((label, method, args, a, b))
    if failures:
        print(f"FAIL: {len(failures)} stratified mismatches")
        for label, method, args, a, b in failures[:10]:
            print(f"  [{label}] {method}{args}")
            print(f"    chunks: {a}")
            print(f"    sqlite: {b}")
        raise AssertionError(f"{len(failures)} stratified mismatches")
    print(f"  PASS {len(cases)} cases")


# ---------------------------------------------------------------------------
# Random sample

def test_random_sample():
    chunks, sql = _load_both()

    rng = random.Random(RANDOM_SEED)
    sql._ensure_conn()

    # Pull the universe from SQLite so we sample real IDs.
    players = list(sql._player_id_map.keys())
    venues = list(sql._venue_id_map.keys())
    # Sample from full date range *and* a few out-of-range dates.
    dates = list(sql._date_strs)
    stretched_dates = dates + [
        (datetime.strptime(dates[0], '%Y-%m-%d') - timedelta(days=10)).strftime('%Y-%m-%d'),
        (datetime.strptime(dates[-1], '%Y-%m-%d') + timedelta(days=200)).strftime('%Y-%m-%d'),
    ]

    methods = [
        ('get_batting_stats',
         lambda: (rng.choice(players), rng.choice(stretched_dates))),
        ('get_bowling_stats',
         lambda: (rng.choice(players), rng.choice(stretched_dates))),
        ('get_h2h_stats',
         lambda: (rng.choice(players), rng.choice(players),
                  rng.choice(stretched_dates))),
        ('get_venue_avg_score',
         lambda: (rng.choice(venues), rng.choice(stretched_dates))),
        ('get_venue_profile',
         lambda: (rng.choice(venues), rng.choice(stretched_dates))),
        ('get_batting_vs_type_stats',
         lambda: (rng.choice(players), rng.choice(stretched_dates))),
        ('get_bowling_vs_hand_stats',
         lambda: (rng.choice(players), rng.choice(stretched_dates))),
        ('get_batting_elo',
         lambda: (rng.choice(players), rng.choice(stretched_dates))),
        ('get_bowling_elo',
         lambda: (rng.choice(players), rng.choice(stretched_dates))),
    ]

    print(f"\nrandom sample: {RANDOM_N} queries across "
          f"{len(methods)} methods", flush=True)

    # Sort by date so the chunks backend amortises chunk I/O (5-chunk
    # LRU otherwise churns on a uniform-random workload; 10K queries
    # take 15+ min). The SQLite backend doesn't care about ordering.
    sample = []
    for _ in range(RANDOM_N):
        name, build = rng.choice(methods)
        args = build()
        # Last positional arg is always the date string.
        sample.append((args[-1], name, args))
    sample.sort(key=lambda x: x[0])

    failures = []
    t0 = time.time()
    for i, (_, name, args) in enumerate(sample):
        a = getattr(chunks, name)(*args)
        b = getattr(sql, name)(*args)
        eq = _eq_dict(a, b) if isinstance(a, dict) else (a == b)
        if not eq:
            failures.append((i, name, args, a, b))
            if len(failures) >= 5:
                break
        if i > 0 and i % 2000 == 0:
            print(f"  ... {i}/{RANDOM_N} ({time.time()-t0:.1f}s elapsed)",
                  flush=True)

    if failures:
        print(f"FAIL: {len(failures)}+ random mismatches (first 5):")
        for i, name, args, a, b in failures:
            print(f"  #{i} {name}{args}")
            print(f"    chunks: {a}")
            print(f"    sqlite: {b}")
        raise AssertionError(f"{len(failures)}+ random mismatches")
    print(f"  PASS {RANDOM_N} queries")


# ---------------------------------------------------------------------------

if __name__ == '__main__':
    test_stratified()
    test_random_sample()
    print("\nALL EQUIVALENCE CHECKS PASSED")
