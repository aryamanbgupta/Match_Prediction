"""Single-process RAM benchmark for StatsProvider.

Tight + unbuffered. Measures per-process RSS cost under a realistic eval
pattern (sequential dates within a late-season window). Avoids pathological
random-access thrash.
"""
import gc
import os
import sys
import time
from pathlib import Path

import psutil

# Unbuffered prints so we see progress live
sys.stdout.reconfigure(line_buffering=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

proc = psutil.Process(os.getpid())

def rss_mb() -> float:
    return proc.memory_info().rss / (1024 * 1024)

def mark(label: str, before: float) -> float:
    now = rss_mb()
    print(f"{label:<50s} RSS={now:8.1f} MB   delta={now-before:+7.1f} MB", flush=True)
    return now

print(f"available_mb={psutil.virtual_memory().available/1e6:.0f}", flush=True)
base = rss_mb()
print(f"{'baseline (python + psutil only)':<50s} RSS={base:8.1f} MB", flush=True)

t0 = time.time()
from stats_provider import StatsProvider  # noqa: E402
after_import = mark(f"after stats_provider import ({time.time()-t0:.1f}s)", base)

t0 = time.time()
provider = StatsProvider('models', max_cached_chunks=5, version='v3')
after_init = mark(f"after StatsProvider init ({time.time()-t0:.1f}s)", after_import)
print(f"  chunks_on_disk={provider.metadata['num_chunks']}  total_dates={len(provider.dates):,}", flush=True)

# Realistic pattern: sorted eval access. betting_test matches are recent (2024-06+).
# We pick ~10 distinct recent dates, query each once — simulates the per-match
# stat-snapshot touches an eval does BEFORE running the 100 sims (the snapshot
# is cached per-match inside sim_v1_2).
recent_dates = provider.dates[-30:]  # last 30 snapshot dates (~late 2024)
sampled = recent_dates[::3]  # 10 evenly spaced dates
print(f"  sampling {len(sampled)} dates from {sampled[0]} to {sampled[-1]}", flush=True)

t0 = time.time()
for d in sampled:
    _ = provider._get_snapshot_for_date(d)
after_seq = mark(f"after {len(sampled)} sequential-date queries ({time.time()-t0:.1f}s)", after_init)
print(f"  chunks_resident_in_lru={len(provider.chunk_cache)}", flush=True)

# Size of one late chunk resident
late_idx = provider.metadata['num_chunks'] - 2
chunk_file = Path('models') / provider.metadata['chunks'][late_idx]['file']
disk_mb = chunk_file.stat().st_size / (1024*1024)
print(f"  late_chunk_disk={disk_mb:.1f}MB  (pickle typically expands 1.5-2x in RAM)", flush=True)

# Reclaim
provider.chunk_cache.clear()
gc.collect()
after_free = mark("after clearing LRU + gc", after_seq)

# Extrapolation table
per_proc_steady = after_seq - base
print("\n--- single-process cost ---", flush=True)
print(f"metadata only (always resident):  ~{after_init-after_import:.0f} MB", flush=True)
print(f"steady-state eval (LRU=5 late):   ~{per_proc_steady:.0f} MB", flush=True)
print("\n--- N parallel worktrees (pickle is private heap → no sharing) ---", flush=True)
for n in (1, 2, 3, 4):
    print(f"  N={n}: ~{n*per_proc_steady/1024:.1f} GB stats cache alone "
          f"(+ model + pandas + simulation state per process)", flush=True)

print("\nDONE", flush=True)
