# perf_runs/

Local scratch space for benchmarking and perf experiments. Only the three
Python drivers are tracked; everything else (logs, intermediate eval JSONs,
profile dumps, split-symlink dirs) is gitignored.

## Drivers

| Script | Purpose |
|---|---|
| [`measure.py`](measure.py) | Wrap any command with wall-clock + max-RSS reporting. Uses `resource.getrusage(RUSAGE_CHILDREN)`. macOS-aware (RSS reported in bytes). |
| [`diff_eval.py`](diff_eval.py) | Bit-identical comparison of two `run_sim_eval.py` output JSONs. Checks `simulated_win_prob`, `log_loss`, `brier_score`, `edge`, `realized_pnl`, `simulated_scores` per match plus four summary metrics. |
| [`run_n_parallel.py`](run_n_parallel.py) | N-process parallel eval. Splits the polymarket test set into N disjoint shards via symlinks, launches N `run_sim_eval.py` processes with `OMP_NUM_THREADS` capped, samples combined RSS in real time, reports throughput. |

## Common recipes

```bash
# Wall + max-RSS for any command
uv run python perf_runs/measure.py "label" -- <cmd> <args>

# Bit-identical diff
uv run python perf_runs/diff_eval.py <baseline.json> <new.json>

# Multi-process parallel eval (the recommended way to use multiple cores)
uv run python perf_runs/run_n_parallel.py 2 -1 100   # 2 procs over full 261 matches
uv run python perf_runs/run_n_parallel.py 4 -1 100   # 4 procs, faster but diminishing returns
```

For the production-recipe and the speedup numbers see
[`docs/OPERATIONS.md`](../docs/OPERATIONS.md) § "Multi-process parallel eval"
and [`IMPROVEMENTS.md`](../IMPROVEMENTS.md) § "Performance Pass".

## What gets reclaimed locally

`perf_runs/` accumulates output dirs, profile files, and run logs.
None of it is load-bearing — `rm -rf perf_runs/*` (then `git checkout
perf_runs/` to restore the tracked drivers) is the cleanup recipe if it
gets large.
