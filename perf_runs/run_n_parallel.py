"""N-process parallel-eval test.

Splits the polymarket test set into N disjoint subsets, runs N eval
processes simultaneously with OMP capped to control oversubscription,
samples combined RSS, and reports throughput vs serial baseline.

Usage:
    uv run python perf_runs/run_n_parallel.py 4 10 100
    # = 4 processes, 10 matches each, 100 sims each
"""
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parent.parent

if len(sys.argv) != 4:
    print("usage: run_n_parallel.py <n_procs> <matches_per_proc> <n_sims>")
    sys.exit(2)

n_procs = int(sys.argv[1])
matches_per_proc = int(sys.argv[2])
n_sims = int(sys.argv[3])

# OMP per proc to avoid oversubscription.  hw.logicalcpu = 10 here.
omp = max(1, 10 // n_procs)
print(f"=== Parallel test: {n_procs} procs x {matches_per_proc} matches "
      f"x {n_sims} sims  (OMP_NUM_THREADS={omp}) ===")

# Build N disjoint match subsets via symlinks. Special meaning:
#   matches_per_proc == -1  -> split the FULL test set across n_procs
#                              (last proc takes the remainder).
all_matches = sorted((ROOT / "data/polymarket_test").glob("*.json"))
if matches_per_proc == -1:
    matches_per_proc = len(all_matches) // n_procs
    use_remainder = True
else:
    use_remainder = False
    if len(all_matches) < n_procs * matches_per_proc:
        print(f"warning: only {len(all_matches)} matches; need "
              f"{n_procs * matches_per_proc}")
        matches_per_proc = len(all_matches) // n_procs

split_dirs = []
proc_match_counts = []
for i in range(n_procs):
    d = ROOT / f"perf_runs/par_split_{i}"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    if use_remainder and i == n_procs - 1:
        chunk = all_matches[i * matches_per_proc:]
    else:
        chunk = all_matches[i * matches_per_proc:(i + 1) * matches_per_proc]
    proc_match_counts.append(len(chunk))
    for src in chunk:
        os.symlink(src, d / src.name)
    split_dirs.append(d)
print(f"  matches per proc: {proc_match_counts}  (total: "
      f"{sum(proc_match_counts)}/{len(all_matches)})")

env = {
    **os.environ,
    "OMP_NUM_THREADS": str(omp),
    "OPENBLAS_NUM_THREADS": str(omp),
    "MKL_NUM_THREADS": str(omp),
    "VECLIB_MAXIMUM_THREADS": str(omp),
    "NUMEXPR_NUM_THREADS": str(omp),
}


def build_cmd(test_dir: Path, out_dir: Path, n_matches: int):
    cmd = [
        sys.executable,
        "scripts/sim_eval/run_sim_eval.py",
        "--model-type", "xgboost",
        "--test-dir", str(test_dir),
        "--odds", "betting_odds_polymarket.json",
        "--n-sims", str(n_sims),
        "--output-dir", str(out_dir),
    ]
    # Only cap matches if requested; with -1 (full split), let each
    # process consume its entire shard.
    if not use_remainder:
        cmd.extend(["--max-matches", str(n_matches)])
    return cmd


def descendant_rss_mb(parent: psutil.Process) -> float:
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


t0 = time.perf_counter()
procs = []
ps_handles = []
for i in range(n_procs):
    out_dir = ROOT / f"perf_runs/par_out_{i}"
    log_path = ROOT / f"perf_runs/par_out_{i}.log"
    p = subprocess.Popen(
        build_cmd(split_dirs[i], out_dir, matches_per_proc),
        stdout=open(log_path, "wb"),
        stderr=subprocess.STDOUT,
        cwd=str(ROOT),
        env=env,
    )
    procs.append(p)
    ps_handles.append(psutil.Process(p.pid))

peak_combined = 0.0
peak_per_proc = [0.0] * n_procs
samples = 0
while any(p.poll() is None for p in procs):
    time.sleep(0.5)
    rss = []
    for i, (p, h) in enumerate(zip(procs, ps_handles)):
        if p.poll() is None:
            try:
                r = descendant_rss_mb(h)
            except psutil.NoSuchProcess:
                r = 0
        else:
            r = 0
        rss.append(r)
        peak_per_proc[i] = max(peak_per_proc[i], r)
    peak_combined = max(peak_combined, sum(rss))
    samples += 1

wall = time.perf_counter() - t0

print(f"\n  wall:           {wall:.1f}s")
print(f"  exits:          {[p.returncode for p in procs]}")
print(f"  peak per-proc:  {[f'{r:.0f}' for r in peak_per_proc]} MB")
print(f"  peak combined:  {peak_combined:.0f} MB")
print(f"  samples:        {samples}")

# Cleanup the symlink dirs
for d in split_dirs:
    if d.exists():
        shutil.rmtree(d)
