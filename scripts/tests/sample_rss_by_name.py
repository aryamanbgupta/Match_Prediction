"""Sample combined RSS of all processes whose cmdline matches a pattern.

Polls every N seconds, prints one line per sample:
    timestamp  n_procs  combined_rss_mb  per_proc_rss_mb_list

Exits when the pattern stops matching any process.

Usage:
    uv run python scripts/tests/sample_rss_by_name.py run_sim_eval.py 2.0
"""
from __future__ import annotations

import sys
import time

import psutil


def _matches(proc: psutil.Process, pattern: str) -> bool:
    try:
        cmdline = ' '.join(proc.cmdline())
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False
    return pattern in cmdline


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: sample_rss_by_name.py <pattern> <interval_s>")
        return 2
    pattern = sys.argv[1]
    interval = float(sys.argv[2])

    peak_combined = 0.0
    peak_per_proc: dict[int, float] = {}
    empty_streak = 0

    while True:
        matches = [p for p in psutil.process_iter(['pid', 'cmdline'])
                   if _matches(p, pattern)]
        if not matches:
            empty_streak += 1
            # Exit after 3 empty intervals — processes have finished.
            if empty_streak >= 3:
                break
            time.sleep(interval)
            continue
        empty_streak = 0
        rss_list = []
        combined = 0.0
        for p in matches:
            try:
                rss = p.memory_info().rss / (1024 * 1024)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
            rss_list.append((p.pid, rss))
            combined += rss
            peak_per_proc[p.pid] = max(peak_per_proc.get(p.pid, 0.0), rss)
        peak_combined = max(peak_combined, combined)
        ts = time.strftime('%H:%M:%S')
        per_fmt = ','.join(f"{pid}:{rss:.0f}" for pid, rss in rss_list)
        print(f"{ts}  n={len(rss_list)}  combined={combined:.0f}MB  "
              f"[{per_fmt}]", flush=True)
        time.sleep(interval)

    print(f"\n--- SUMMARY ---")
    print(f"peak combined RSS: {peak_combined:.0f} MB")
    for pid, rss in peak_per_proc.items():
        print(f"  pid {pid}: peak {rss:.0f} MB")
    return 0


if __name__ == '__main__':
    sys.exit(main())
