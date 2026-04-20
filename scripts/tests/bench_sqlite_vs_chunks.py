"""Side-by-side profile: SQLite backend vs chunked StatsProvider.

Runs both backends through identical workloads and reports init time,
steady-state RSS, and per-query latency. Intended to answer "what's the
actual speedup and RAM savings from the SQLite migration?"

Workload: 10K random (method, args, date) triples, shared across both
backends. Queries are sorted by date so the chunk backend amortises its
disk I/O — otherwise its LRU thrashes and the comparison becomes an
I/O benchmark rather than a query benchmark.

Caveat: chunked RSS includes up to 5 cached chunks (~500 MB). The first
few hundred queries pull the chunks in; later queries all hit RAM. SQLite
keeps everything on-disk and reads via mmap.

Run:
    uv run python scripts/tests/bench_sqlite_vs_chunks.py
"""
from __future__ import annotations

import gc
import os
import random
import statistics
import sys
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))

from stats_provider import StatsProvider  # noqa: E402
from stats_sqlite_backend import _SQLiteBackend  # noqa: E402


N_QUERIES = 10_000
RANDOM_SEED = 0xC0DE


def _rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def _build_methods(sql: _SQLiteBackend, rng: random.Random, dates_pool):
    players = list(sql._player_id_map.keys())
    venues = list(sql._venue_id_map.keys())
    return [
        ('get_batting_stats',
         lambda: (rng.choice(players), rng.choice(dates_pool))),
        ('get_bowling_stats',
         lambda: (rng.choice(players), rng.choice(dates_pool))),
        ('get_h2h_stats',
         lambda: (rng.choice(players), rng.choice(players),
                  rng.choice(dates_pool))),
        ('get_venue_profile',
         lambda: (rng.choice(venues), rng.choice(dates_pool))),
        ('get_batting_vs_type_stats',
         lambda: (rng.choice(players), rng.choice(dates_pool))),
        ('get_bowling_vs_hand_stats',
         lambda: (rng.choice(players), rng.choice(dates_pool))),
        ('get_batting_elo',
         lambda: (rng.choice(players), rng.choice(dates_pool))),
        ('get_bowling_elo',
         lambda: (rng.choice(players), rng.choice(dates_pool))),
    ]


def _make_cross_date_workload(sql: _SQLiteBackend, n: int):
    """Random dates across the full range — worst case for chunks backend
    because the LRU can't hold a hot set."""
    rng = random.Random(RANDOM_SEED)
    sql._ensure_conn()
    dates = list(sql._date_strs)
    methods = _build_methods(sql, rng, dates)
    workload = []
    for _ in range(n):
        name, build = rng.choice(methods)
        args = build()
        workload.append((args[-1], name, args))
    workload.sort(key=lambda x: x[0])  # date-sort for chunk amortisation
    return [(name, args) for _, name, args in workload]


def _make_single_date_workload(sql: _SQLiteBackend, n: int):
    """All queries target the same recent date — closer to real eval
    where one match = one fixed date × many player/ball queries."""
    rng = random.Random(RANDOM_SEED + 1)
    sql._ensure_conn()
    # Pick a date ~75% through the range (late enough that all chunks
    # exist; not the final chunk so we avoid edge effects).
    dates = sql._date_strs
    pinned = dates[int(len(dates) * 0.75)]
    methods = _build_methods(sql, rng, [pinned])
    workload = []
    for _ in range(n):
        name, build = rng.choice(methods)
        workload.append((name, build()))
    return workload


def _time_queries(backend, workload):
    latencies = []
    t0 = time.perf_counter()
    for name, args in workload:
        q0 = time.perf_counter_ns()
        getattr(backend, name)(*args)
        latencies.append(time.perf_counter_ns() - q0)
    total_s = time.perf_counter() - t0
    latencies.sort()
    return {
        'total_s': total_s,
        'qps': len(workload) / total_s,
        'mean_us': statistics.mean(latencies) / 1e3,
        'p50_us': latencies[len(latencies) // 2] / 1e3,
        'p99_us': latencies[int(len(latencies) * 0.99)] / 1e3,
    }


def bench_sqlite(workload):
    gc.collect()
    rss_before = _rss_mb()
    t0 = time.perf_counter()
    backend = _SQLiteBackend(str(ROOT / 'models' / 'player_stats_cache_v3.sqlite'))
    backend._ensure_conn()
    init_s = time.perf_counter() - t0
    # Warm up the OS page cache + prepared-statement cache.
    for name, args in workload[:500]:
        getattr(backend, name)(*args)
    rss_warm = _rss_mb()
    stats = _time_queries(backend, workload)
    rss_after = _rss_mb()
    return {'backend': 'sqlite', 'init_s': init_s,
            'rss_before_mb': rss_before, 'rss_warm_mb': rss_warm,
            'rss_after_mb': rss_after, **stats}


def bench_chunks(workload):
    gc.collect()
    rss_before = _rss_mb()
    t0 = time.perf_counter()
    provider = StatsProvider(str(ROOT / 'models'), version='v3')
    init_s = time.perf_counter() - t0
    # Warm: pull first few chunks. The chunk backend is slower on warmup
    # (pickle expansion) but amortises over long runs — mirror the
    # sqlite warmup count for a fair steady-state comparison.
    for name, args in workload[:500]:
        getattr(provider, name)(*args)
    rss_warm = _rss_mb()
    stats = _time_queries(provider, workload)
    rss_after = _rss_mb()
    return {'backend': 'chunks', 'init_s': init_s,
            'rss_before_mb': rss_before, 'rss_warm_mb': rss_warm,
            'rss_after_mb': rss_after, **stats}


def _run_both(workload, label):
    print(f"\n{'='*62}\n{label}\n{'='*62}")
    print("--- SQLite ---", flush=True)
    sql = bench_sqlite(workload)
    for k, v in sql.items():
        print(f"  {k:16s} {v:>12.3f}" if isinstance(v, float)
              else f"  {k:16s} {v}")
    gc.collect()
    print("\n--- chunks ---", flush=True)
    ch = bench_chunks(workload)
    for k, v in ch.items():
        print(f"  {k:16s} {v:>12.3f}" if isinstance(v, float)
              else f"  {k:16s} {v}")
    print(f"\n--- comparison ({label}) ---")
    rows = [
        ('init_s',       'lower is better'),
        ('rss_after_mb', 'lower is better'),
        ('total_s',      'lower is better'),
        ('qps',          'higher is better'),
        ('mean_us',      'lower is better'),
        ('p50_us',       'lower is better'),
        ('p99_us',       'lower is better'),
    ]
    print(f"  {'metric':<16s} {'sqlite':>12s} {'chunks':>12s} "
          f"{'ratio':>10s}  direction")
    for key, direction in rows:
        s_val, c_val = sql[key], ch[key]
        ratio = (s_val / c_val) if key == 'qps' else (c_val / s_val)
        print(f"  {key:<16s} {s_val:>12.3f} {c_val:>12.3f} "
              f"{ratio:>9.2f}x  {direction}")
    return sql, ch


def main():
    sql_tmp = _SQLiteBackend(str(ROOT / 'models' / 'player_stats_cache_v3.sqlite'))

    print(f"building cross-date workload of {N_QUERIES:,} queries ...",
          flush=True)
    cross = _make_cross_date_workload(sql_tmp, N_QUERIES)
    print(f"building single-date workload of {N_QUERIES:,} queries ...",
          flush=True)
    single = _make_single_date_workload(sql_tmp, N_QUERIES)
    del sql_tmp
    gc.collect()

    # Cross-date first: stress-tests chunks LRU eviction + pickle-load cost.
    _run_both(cross, "cross-date workload (worst case for chunks)")
    # Single-date: realistic eval pattern (one match = one date).
    _run_both(single, "single-date workload (realistic eval shape)")


if __name__ == '__main__':
    main()
