"""Phase-1 gate benchmark for the SQLite stats backend.

Measures four things, in order:

  1. EXPLAIN QUERY PLAN for every canonical getter shape — must report
     `SEARCH ... USING INDEX` (or `USING PRIMARY KEY`), never `SCAN`.
  2. Per-process steady-state RSS after a warm query pass.
  3. Query latency p50 / p99 over a random workload that touches every
     getter shape.
  4. 4-worker combined RSS against the same file (mmap page-share check).
     If combined RSS scales ~linearly with worker count, the OS isn't
     sharing the file's pages across workers — which would invalidate
     the structural premise of this migration.

Run:
    uv run python scripts/tests/bench_sqlite_backend.py \
        --db models/player_stats_cache_v3_poc.sqlite

Phase-1 gates (from the approved plan):
    - Per-process RSS  ≤ 100 MB steady state
    - 4-worker combined RSS ≤ 500 MB
    - p50 ≤ 20 µs, p99 ≤ 100 µs
    - Every getter uses an index
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import random
import statistics
import sys
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))

from stats_sqlite_backend import QUERY_PLAN_CASES, _SQLiteBackend  # noqa: E402


N_QUERIES = 100_000
N_WORKERS_PARALLEL = 4
WORKER_QUERIES = 50_000


def _rss_mb(pid: int) -> float:
    return psutil.Process(pid).memory_info().rss / (1024 * 1024)


def _sample_args(backend: _SQLiteBackend, rng: random.Random):
    """Build a random (method, args) workload touching every getter."""
    backend._ensure_conn()
    players = list(backend._player_id_map.keys())
    venues = list(backend._venue_id_map.keys())
    dates = backend._date_strs
    methods = [
        ('batting', lambda: (rng.choice(players), rng.choice(dates))),
        ('bowling', lambda: (rng.choice(players), rng.choice(dates))),
        ('h2h', lambda: (rng.choice(players), rng.choice(players), rng.choice(dates))),
        ('batting_vs_type', lambda: (rng.choice(players), rng.choice(dates))),
        ('bowling_vs_hand', lambda: (rng.choice(players), rng.choice(dates))),
        ('venue_profile', lambda: (rng.choice(venues), rng.choice(dates))),
        ('batting_elo', lambda: (rng.choice(players), rng.choice(dates))),
        ('bowling_elo', lambda: (rng.choice(players), rng.choice(dates))),
    ]
    return methods


def _dispatch(backend: _SQLiteBackend, method: str, args):
    if method == 'batting':
        return backend.get_batting_stats(*args)
    if method == 'bowling':
        return backend.get_bowling_stats(*args)
    if method == 'h2h':
        return backend.get_h2h_stats(*args)
    if method == 'batting_vs_type':
        return backend.get_batting_vs_type_stats(*args)
    if method == 'bowling_vs_hand':
        return backend.get_bowling_vs_hand_stats(*args)
    if method == 'venue_profile':
        return backend.get_venue_profile(*args)
    if method == 'batting_elo':
        return backend.get_batting_elo(*args)
    if method == 'bowling_elo':
        return backend.get_bowling_elo(*args)
    raise KeyError(method)


def check_query_plans(db_path: str) -> bool:
    import sqlite3
    conn = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
    all_ok = True
    print("=== EXPLAIN QUERY PLAN ===")
    for name, sql, args in QUERY_PLAN_CASES:
        plan = conn.execute('EXPLAIN QUERY PLAN ' + sql, args).fetchall()
        detail = ' | '.join(row[-1] for row in plan)
        ok = 'SEARCH' in detail and ('INDEX' in detail or 'PRIMARY KEY' in detail)
        mark = 'OK ' if ok else 'BAD'
        all_ok = all_ok and ok
        print(f"  [{mark}] {name:18s} {detail}")
    conn.close()
    return all_ok


def bench_single(db_path: str, n_queries: int) -> dict:
    rng = random.Random(0xC0FFEE)
    backend = _SQLiteBackend(db_path)
    backend._ensure_conn()
    methods = _sample_args(backend, rng)

    # Warm pass — first N queries pull pages into the OS page cache.
    for _ in range(5_000):
        name, build = rng.choice(methods)
        _dispatch(backend, name, build())

    rss_before = _rss_mb(os.getpid())

    latencies: list[float] = []
    t0 = time.perf_counter()
    for _ in range(n_queries):
        name, build = rng.choice(methods)
        args = build()
        q0 = time.perf_counter_ns()
        _dispatch(backend, name, args)
        latencies.append(time.perf_counter_ns() - q0)
    total_s = time.perf_counter() - t0

    rss_after = _rss_mb(os.getpid())

    latencies.sort()
    p50_us = latencies[len(latencies) // 2] / 1e3
    p99_us = latencies[int(len(latencies) * 0.99)] / 1e3
    mean_us = statistics.mean(latencies) / 1e3

    return {
        'rss_before_mb': rss_before,
        'rss_after_mb': rss_after,
        'n_queries': n_queries,
        'total_s': total_s,
        'qps': n_queries / total_s,
        'p50_us': p50_us,
        'p99_us': p99_us,
        'mean_us': mean_us,
    }


def _worker(db_path: str, seed: int, n_queries: int, rss_q):
    rng = random.Random(seed)
    backend = _SQLiteBackend(db_path)
    backend._ensure_conn()
    methods = _sample_args(backend, rng)
    for _ in range(n_queries):
        name, build = rng.choice(methods)
        _dispatch(backend, name, build())
    # Let the OS settle, then read own RSS.
    time.sleep(0.1)
    rss_q.put((os.getpid(), _rss_mb(os.getpid())))


def bench_parallel(db_path: str, n_workers: int, n_queries: int) -> dict:
    ctx = mp.get_context('spawn')
    rss_q: mp.Queue = ctx.Queue()
    procs = []
    for w in range(n_workers):
        p = ctx.Process(target=_worker, args=(db_path, 1000 + w, n_queries, rss_q))
        p.start()
        procs.append(p)

    # Wait until all workers have reported RSS, then join.
    reports = []
    for _ in range(n_workers):
        reports.append(rss_q.get(timeout=120))
    for p in procs:
        p.join(timeout=60)
    return {
        'n_workers': n_workers,
        'per_worker_rss_mb': [r[1] for r in reports],
        'combined_rss_mb': sum(r[1] for r in reports),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', default='models/player_stats_cache_v3_poc.sqlite')
    ap.add_argument('--n-queries', type=int, default=N_QUERIES)
    ap.add_argument('--workers', type=int, default=N_WORKERS_PARALLEL)
    ap.add_argument('--worker-queries', type=int, default=WORKER_QUERIES)
    args = ap.parse_args()

    db_size_mb = Path(args.db).stat().st_size / (1024 * 1024)
    print(f"db: {args.db}  ({db_size_mb:.1f} MB on disk)\n")

    plans_ok = check_query_plans(args.db)
    print()

    print("=== single-process RSS + latency ===")
    r = bench_single(args.db, args.n_queries)
    print(f"  rss before:   {r['rss_before_mb']:7.1f} MB")
    print(f"  rss after:    {r['rss_after_mb']:7.1f} MB")
    print(f"  {r['n_queries']} queries in {r['total_s']:.2f}s "
          f"({r['qps']:,.0f} qps)")
    print(f"  latency  mean {r['mean_us']:6.2f} µs  "
          f"p50 {r['p50_us']:6.2f} µs  p99 {r['p99_us']:6.2f} µs")
    print()

    print(f"=== {args.workers}-worker parallel RSS (spawn) ===")
    pr = bench_parallel(args.db, args.workers, args.worker_queries)
    for i, rss in enumerate(pr['per_worker_rss_mb']):
        print(f"  worker {i}: {rss:6.1f} MB")
    print(f"  combined:  {pr['combined_rss_mb']:6.1f} MB "
          f"(would be {args.workers * r['rss_after_mb']:.1f} MB if no sharing)")
    print()

    # --- gate check ---
    gates = {
        'plans_index': plans_ok,
        'single_rss_le_100': r['rss_after_mb'] <= 100,
        'parallel_rss_le_500': pr['combined_rss_mb'] <= 500,
        'p50_le_20us': r['p50_us'] <= 20,
        'p99_le_100us': r['p99_us'] <= 100,
    }
    print("=== Phase-1 gates ===")
    for name, ok in gates.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if not all(gates.values()):
        sys.exit(1)


if __name__ == '__main__':
    main()
