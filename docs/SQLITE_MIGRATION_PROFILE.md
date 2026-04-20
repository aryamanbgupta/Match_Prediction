# SQLite stats-cache migration — profile

Measured 2026-04-19 on the `main` branch after Phase 2 of the migration plan.
Target of the migration: replace 11 GB of pickle chunks (which expand ~12×
on load and forced `--parallel=True` to OOM the 16 GB machine) with a
single mmap-backed SQLite file.

Reproduce:

```bash
uv run python scripts/tests/bench_sqlite_vs_chunks.py
```

Both workloads exercise all 8 getter shapes through 10 000 random
queries. The SQLite DB is `models/player_stats_cache_v3.sqlite`
(full 75-chunk build, 39.7 MB on disk vs 11 GB of pickle chunks).

## Disk

| | chunks | sqlite | ratio |
|---|---:|---:|---:|
| on-disk size | 11 GB | 39.7 MB | **276× smaller** |
| build time (full) | ~15 min (parsing_v2) | 5 min 43 s | ~2.6× |

Row counts in the full SQLite file:

```
players           7,516
dates             3,664
venues              467
batting         149,942
bowling         114,802
h2h             473,773
batting_vs_type 232,774
bowling_vs_hand 190,875
venue             7,971
batting_elo     147,354
bowling_elo     111,417
```

Delta compression does most of the size reduction — stats are stored only
when a counter actually changes.

## Cross-date workload (worst case for chunks)

10 K queries spanning the full date range. Sorted by date so the chunk
backend's 5-chunk LRU walks forward instead of thrashing.

| metric | sqlite | chunks | ratio |
|---|---:|---:|---:|
| RSS after run | 51 MB | 3,225 MB | **63× less** |
| wall-clock total | 0.08 s | 533 s | **6581× faster** |
| qps | 123 400 | 19 | **6581×** |
| mean latency | 8 µs | 53 ms | 6702× |
| p50 latency | 2.7 µs | 1.9 µs | 0.7× (chunks wins p50 when warm) |
| p99 latency | 146 µs | 448 µs | 3.1× |

The chunks backend re-pays pickle-deserialisation cost on every new
chunk access; SQLite walks primary-key b-trees with no per-chunk load.

## Single-date workload (realistic eval shape)

All 10 K queries against a single fixed date, i.e. one simulated match.
Representative of the actual eval hot path.

| metric | sqlite | chunks | ratio |
|---|---:|---:|---:|
| RSS during run (working set) | ~30 MB | ~3.8 GB | **~100× less** |
| wall-clock total | 0.08 s | 0.011 s | 0.14× (chunks wins) |
| qps | 127 K | 914 K | 0.14× |
| mean latency | 7.7 µs | 1.0 µs | 0.13× |
| p50 latency | 2.9 µs | 0.96 µs | 0.33× |
| p99 latency | 145 µs | 1.75 µs | 0.01× |

Chunks wins raw per-query latency once the needed chunks are resident:
a Python dict lookup is faster than a SQLite prepared-statement
execution. But the chunk backend pays this speed in RAM — it holds the
entire working set in RSS, not just recently-touched rows.

## Read this as

- **Absolute throughput is a tie on the realistic hot path.** A single
  match simulation never spans many chunks, so the chunk backend's
  in-memory dict lookups are a few µs faster per call.
- **The real win is RAM.** Steady-state working set drops from GB to
  tens of MB. That's what unblocks `--parallel=True` inside one eval,
  and running two eval worktrees simultaneously without OOM (the
  `feedback_no_parallel_sim_eval` incident).
- **Cold paths are 6000× faster.** Anything that crosses many dates
  (training-data validation, cross-match analysis) pays the pickle load
  once per chunk under the old backend — 100 ms × 75 chunks = 7.5 s of
  I/O per pass. SQLite doesn't pay this.
- **Single-process RSS is 60–100× lower; mmap sharing multiplies that
  across workers** — Phase 1 bench already showed 4 workers combined =
  75.8 MB on the 5-chunk POC (vs 4 × chunk-backend RSS = 16 GB+ today).

The production-relevant number is the full-eval wall-clock comparison
planned for Phase 4 — both backends, 261 matches × 100 sims. That's what
tells us the end-to-end user-visible speedup. The micro-bench above is
mostly a correctness + RAM check.

## Phase 4 — end-to-end eval (261 matches × 100 sims, polymarket_test)

Both backends run through the full production eval path
(`scripts/sim_eval/run_sim_eval.py --test-dir data/polymarket_test
--odds betting_odds_polymarket.json --n-sims 100`).

| metric | sqlite | chunks |
|---|---:|---:|
| matches | 261 | 261 |
| wall-clock (serial) | **36 min** (2 161 s) | 38 min (2 289 s) |
| avg_log_loss | 0.7210459782407803 | 0.7210459782407803 |
| avg_brier_score | 0.26022053595610395 | 0.26022053595610395 |
| avg_edge | 0.18152773078062487 | 0.18152773078062487 |

**All 261 matches bit-identical on `simulated_prob`** (every key, every
value, float-exact). Verified via
`scripts/tests/compare_phase4_evals.py`.

### Parallel (2-worktree) eval — the original crash scenario

Two concurrent full evals from separate shells against the same SQLite
DB, sampled via `scripts/tests/sample_rss_by_name.py` (5 s interval):

| metric | value |
|---|---|
| peak combined RSS | **1 736 MB** (vs ~10 GB that OOM'd on chunks) |
| per-eval wall-clock | ~48 min each (vs 36 min serial; CPU contention) |
| combined throughput | 2 evals in 48 min (≈ 1.5× serial) |
| OOM / swap | none |
| output vs serial | **bit-identical** (`simulated_prob` to 16 dp) |

The chunks backend's per-process ~5 GB working set put 2 concurrent
evals at ~10 GB RSS on a 16 GB box — that's the crash documented in
`feedback_no_parallel_sim_eval`. SQLite's mmap-backed, shared-page-cache
design collapses that to 1.7 GB combined. `--parallel=True`
(inner-process multiprocessing inside a single eval) is still untested —
only the 2-outer-worktree scenario is validated.

## Phase 1 micro-benchmark (5-chunk POC, for reference)

Also in `scripts/tests/bench_sqlite_backend.py`:

| gate | target | measured |
|---|---|---:|
| query plans | every getter uses index | 8/8 index/PK searches |
| single-proc RSS | ≤ 100 MB | 22.6 MB |
| 4-worker combined RSS | ≤ 500 MB | 75.8 MB |
| p50 latency | ≤ 20 µs | 2.2 µs |
| p99 latency | ≤ 100 µs | 4.8 µs |
