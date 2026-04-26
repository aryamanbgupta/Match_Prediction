"""Estimate how small the cache becomes if we go columnar + per-entity timelines.

Loads ONE late chunk (the realistic worst case), measures:
  - RAM of the pickled Python dict (current format)
  - How many rows a normalized "event log" representation would have
  - Size of the same data as a numpy/parquet columnar table
"""
import gc
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import psutil

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

proc = psutil.Process(os.getpid())
def rss_mb(): return proc.memory_info().rss / 1024 / 1024

# Pick one late chunk (near-last has broadest player coverage)
late_file = Path('models/cache_chunks_v3/cache_chunk_73.pkl')
print(f"probing: {late_file} ({late_file.stat().st_size/1e6:.0f} MB on disk)", flush=True)

base = rss_mb()
print(f"baseline RSS: {base:.1f} MB", flush=True)

t0 = time.time()
with open(late_file, 'rb') as f:
    data = pickle.load(f)
load_t = time.time() - t0
after = rss_mb()
print(f"after pickle.load: RSS={after:.1f} MB (+{after-base:.1f} MB, {load_t:.1f}s)", flush=True)
print(f"pickle expansion factor: {(after-base)/(late_file.stat().st_size/1e6):.1f}x", flush=True)

dates = sorted(data.keys())
print(f"dates_in_chunk: {len(dates)}  range: {dates[0]} → {dates[-1]}", flush=True)

# Look at the latest snapshot — worst case
last_snap = data[dates[-1]]
print(f"\n=== snapshot at {dates[-1]} ===", flush=True)
for k, v in last_snap.items():
    if isinstance(v, dict):
        print(f"  {k}: {len(v):,} entries", flush=True)

# How many unique players total? How many H2H pairs?
n_batters = len(last_snap.get('batting', {}))
n_bowlers = len(last_snap.get('bowling', {}))
n_h2h = len(last_snap.get('h2h', {}))
print(f"\nlatest snapshot totals: batters={n_batters:,} bowlers={n_bowlers:,} h2h={n_h2h:,}", flush=True)

# Columnar estimate — if we stored each as a flat numpy table keyed by player_id:
#   batting: 3 int64 per player (runs, balls, dismissals) = 24 bytes
#   bowling: 3 int64 per player = 24 bytes
#   h2h: 3 int64 + 2 int32 keys per pair = 20 bytes packed (id_b, id_bw packed int32)
col_batting_mb = n_batters * 24 / 1e6
col_bowling_mb = n_bowlers * 24 / 1e6
col_h2h_mb = n_h2h * 20 / 1e6
print(f"\n--- one snapshot, columnar (numpy) estimate ---", flush=True)
print(f"  batting table:   {col_batting_mb:6.2f} MB", flush=True)
print(f"  bowling table:   {col_bowling_mb:6.2f} MB", flush=True)
print(f"  h2h table:       {col_h2h_mb:6.2f} MB", flush=True)
print(f"  one snap total:  {col_batting_mb+col_bowling_mb+col_h2h_mb:6.2f} MB", flush=True)

# If we had per-entity timelines (not per-date snapshots), size = total events, not dates * entities.
# Proxy: count UNIQUE (player, date) with non-zero stats across the chunk's 50 dates
prev_batting = {}
delta_rows = 0
for d in dates:
    snap = data[d]
    for pid, stats in snap.get('batting', {}).items():
        prev = prev_batting.get(pid)
        if prev != (stats['runs'], stats['balls'], stats['dismissals']):
            delta_rows += 1
            prev_batting[pid] = (stats['runs'], stats['balls'], stats['dismissals'])
print(f"\n--- delta-compression analysis (batting only, this chunk) ---", flush=True)
print(f"  naive snapshot rows (all players × all dates): {n_batters * len(dates):,}", flush=True)
print(f"  delta rows (only when a player's stats changed): {delta_rows:,}", flush=True)
print(f"  compression ratio: {(n_batters*len(dates))/max(delta_rows,1):.1f}x", flush=True)

# Extrapolate to full cache (75 chunks)
total_chunks = 75
est_full_col_mb = (col_batting_mb + col_bowling_mb + col_h2h_mb) * len(dates) * total_chunks / len(dates)
# ^ misleading; that's still "snapshot-per-date". Better estimate uses delta compression:
# Across full cache, delta_rows grows with unique player-match events, roughly 4M balls worth of state.
# Very rough: 4M ball events × ~40 bytes (date,pid,runs,balls,dismissals) = 160 MB total
print(f"\n--- full cache estimates ---", flush=True)
print(f"  current format:                     ~11 GB disk, ~5.3 GB/process RAM", flush=True)
print(f"  columnar per-snapshot (numpy):      ~{est_full_col_mb:.0f} MB disk (compressed ~50-100MB)", flush=True)
print(f"  per-entity delta timelines:         ~100-300 MB disk, ~300-500 MB RAM", flush=True)

del data
gc.collect()
print("\nDONE", flush=True)
