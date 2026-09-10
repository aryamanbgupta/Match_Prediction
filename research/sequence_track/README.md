# Sequence-track queue runner

Deterministic runner for pre-registered overnight jobs (stage 1+ of the
sequence and embeddings track). No model in the loop, no verdicts, no
source-control commands: the runner launches the commands listed in
`queue.yaml`, one at a time, and writes marker files.

## Run

```bash
# what would happen, launching nothing
./research/sequence_track/run_queue.sh --dry-run

# the real thing, laptop jobs (default), inside tmux
./research/sequence_track/run_queue.sh 2>&1 | tee -a research/sequence_track/queue.log

# the mini's jobs
./research/sequence_track/run_queue.sh --machine mini

# a different queue file
./research/sequence_track/run_queue.sh --queue /path/to/other_queue.yaml
```

Stop the queue with `touch research/sequence_track/STOP` (the path in
`defaults.stop_file`). It is checked before each job starts, never mid-job.
Remove it and re-run to resume: finished jobs are skipped by config hash.

## Queue file

```yaml
defaults:
  memory_floor_gb: 8     # required; refuse to launch below this
  memory_cap_gb: 36      # required; kill a job's tree above this
  poll_seconds: 30       # optional (default 30); watchdog interval
  stop_file: research/sequence_track/STOP   # optional (this default)

jobs:
  - id: stage1_a          # unique, [A-Za-z0-9][A-Za-z0-9._-]*
    config: experiments/configs/seq_stage1_sim_v1.yaml   # hashed for skip-if-complete
    command: uv run --no-sync python scripts/... # single line, /bin/bash -c
    output_dir: eval_out/seq_stage1/a                    # markers + run.log
    expected_hours: 6                                    # timeout = 2x, in seconds
    machine: laptop                                      # matched against --machine
    notes: optional free text
```

A job missing a field, a duplicate `id`, or a malformed value makes the
runner exit 2 before launching anything. Paths are repo-root relative unless
absolute; jobs run with the repo root as the working directory. A config file
that does not exist yet is recorded as `absent:<path>` rather than a hash, so
the job re-runs once the real config lands.

## Per job

- `caffeinate -i` wrapper, and a `perl -e 'alarm shift; exec @ARGV'` timeout of
  `int(expected_hours * 2 * 3600)` seconds (minimum 1).
- Available memory (`vm_stat`: free + inactive + speculative pages x page size,
  reported in GiB) is read and logged before **every attempt**, the retry
  included. Below `memory_floor_gb`, or if the reading cannot be taken or
  parsed, the attempt is refused: `REFUSED_MEMORY` is written, no `FAILED`
  marker, and the queue moves on. Refusing on an unreadable machine is
  deliberate — an unreadable machine is not evidence that memory is free.
- The whole process tree's RSS is polled every `poll_seconds` and the tree is
  killed (`SIGTERM`, then `SIGKILL`) above `memory_cap_gb`.
- The same watchdog enforces the deadline on the whole tree, a grace period
  (`max(2 s, poll_seconds)`) after the alarm. The alarm only signals the one
  process it exec'd, so a job that backgrounds a child, or that handles
  `SIGALRM` itself, would otherwise survive it and hold the log pipe open.
  Either path writes `TIMEOUT`.
- One retry on any non-zero exit, including timeout and memory kill. A second
  failure marks the job FAILED and the queue moves on.

## Markers in `output_dir`

| file | meaning |
|---|---|
| `COMPLETE` | exited 0; holds `config_sha256` and the end stamp |
| `TIMEOUT` | killed by the alarm |
| `KILLED_MEMORY` | killed by the watchdog; holds the observed RSS |
| `REFUSED_MEMORY` | attempt not launched; memory below the floor, or unreadable |
| `FAILED` | failed twice; holds the exit code and the failure reason |
| `run.log` | job stdout+stderr, appended across retries, with start/exit/end stamps |

## Files

- `run_queue.sh` — the runner.
- `queue.yaml` — the example queue (one placeholder stage-1 job).
- `scripts/sequence_track/queue_lib.py` — YAML parsing/validation, the
  `vm_stat` reading, and the memory/deadline watchdog. Set `QUEUE_VM_STAT_CMD`
  to substitute the memory reading with another command (tests only).
- `scripts/tests/test_run_queue.py` — the acceptance tests
  (`uv run --no-sync pytest -q scripts/tests/test_run_queue.py`).
