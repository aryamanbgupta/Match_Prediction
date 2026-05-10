"""Two-process parallel-eval test.

Launches two eval processes on disjoint match subsets, samples combined
RSS while both run, reports total wall, per-process RSS peak, and
combined RSS peak.

Usage:
    uv run python perf_runs/run_parallel_test.py
"""
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parent.parent
SPLIT_A = ROOT / "perf_runs" / "split_a"
SPLIT_B = ROOT / "perf_runs" / "split_b"
OUT_A = ROOT / "perf_runs" / "par_a"
OUT_B = ROOT / "perf_runs" / "par_b"


def build_cmd(test_dir: Path, out_dir: Path):
    return [
        sys.executable,
        "scripts/sim_eval/run_sim_eval.py",
        "--model-type", "xgboost",
        "--test-dir", str(test_dir),
        "--odds", "betting_odds_polymarket.json",
        "--max-matches", "10",
        "--n-sims", "100",
        "--output-dir", str(out_dir),
    ]


# Cap each child's OpenMP/BLAS thread count. Without this, each XGBoost
# process spawns N threads where N = #cores, and multiple processes
# oversubscribe — turning a should-be-2x speedup into a serialization.
THREAD_LIMIT_ENV = {
    "OMP_NUM_THREADS": "2",
    "OPENBLAS_NUM_THREADS": "2",
    "MKL_NUM_THREADS": "2",
    "VECLIB_MAXIMUM_THREADS": "2",
    "NUMEXPR_NUM_THREADS": "2",
}


def child_rss_mb(parent: psutil.Process):
    """Sum of RSS over the parent + all descendants."""
    total = 0
    procs = [parent]
    try:
        procs += parent.children(recursive=True)
    except psutil.NoSuchProcess:
        pass
    for p in procs:
        try:
            total += p.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return total / (1024 * 1024)


env = {**os.environ, **THREAD_LIMIT_ENV}

t0 = time.perf_counter()
proc_a = subprocess.Popen(build_cmd(SPLIT_A, OUT_A),
                          stdout=open(ROOT / "perf_runs/par_a.log", "wb"),
                          stderr=subprocess.STDOUT, cwd=str(ROOT), env=env)
proc_b = subprocess.Popen(build_cmd(SPLIT_B, OUT_B),
                          stdout=open(ROOT / "perf_runs/par_b.log", "wb"),
                          stderr=subprocess.STDOUT, cwd=str(ROOT), env=env)

ps_a = psutil.Process(proc_a.pid)
ps_b = psutil.Process(proc_b.pid)

peak_a = peak_b = peak_combined = 0.0
samples = []

try:
    while proc_a.poll() is None or proc_b.poll() is None:
        time.sleep(0.5)
        try:
            rss_a = child_rss_mb(ps_a) if proc_a.poll() is None else 0
        except psutil.NoSuchProcess:
            rss_a = 0
        try:
            rss_b = child_rss_mb(ps_b) if proc_b.poll() is None else 0
        except psutil.NoSuchProcess:
            rss_b = 0
        combined = rss_a + rss_b
        peak_a = max(peak_a, rss_a)
        peak_b = max(peak_b, rss_b)
        peak_combined = max(peak_combined, combined)
        samples.append((time.perf_counter() - t0, rss_a, rss_b, combined))
except KeyboardInterrupt:
    proc_a.terminate(); proc_b.terminate()
    raise

wall = time.perf_counter() - t0

print(f"\n=== parallel 2x10x100 (disjoint subsets) ===")
print(f"  wall:           {wall:.1f}s")
print(f"  exit_a:         {proc_a.returncode}")
print(f"  exit_b:         {proc_b.returncode}")
print(f"  peak RSS A:     {peak_a:.0f} MB")
print(f"  peak RSS B:     {peak_b:.0f} MB")
print(f"  peak combined:  {peak_combined:.0f} MB")
print(f"  samples taken:  {len(samples)}")
if samples:
    # Print 5 evenly-spaced samples
    n = len(samples)
    print(f"\n  sample timeline (combined RSS):")
    for i in range(0, n, max(1, n // 6)):
        t, a, b, c = samples[i]
        print(f"    t={t:5.1f}s  a={a:6.0f}  b={b:6.0f}  combined={c:6.0f} MB")
