#!/usr/bin/env python3
"""Stage 1 step 1d runner: the full 255-fixture run, sharded across processes.

This script OWNS the mechanics of 1d and nothing else. It renders nothing of
its own: every job is one of the per-shard command lines
`pin_stage1.py --write` pinned into `arms.<arm>.full_run.shards[k].command`,
which differ from the arm's whole-set `full_run.command` only in
`--fixture-dir` and `--output-dir`. 40 jobs = 4 arms x 10 shards.

Why shards at all: `run_arm.py` refuses `--parallel` (the same-day replay
lifecycle is strictly sequential), so disjoint fixture shards in separate
one-thread processes are the only parallelism stage 1 has. The 1b thread
probe (D9 addendum) measured that threads buy nothing per process, so every
registered command already caps its own BLAS/OMP pools at 1 and the useful
knob is how many processes run at once (`--concurrency`, default 10).

Scheduling: the T1 arm (C) is ~2.3x slower per simulation than the XGBoost
arms, so its ten shards are queued FIRST and the others fill in behind them
(`--arms C B A50 A`). The last T1 shard therefore starts as early as the
pool allows, which is what sets the wall clock.

What it records, per (arm, shard), into `<full root>/full_record.json`:

* wall seconds, exit code, start/end UTC timestamps;
* `Total simulation time: <x>s` as parsed from the run's own log — the ONLY
  number read back out of a run log;
* peak resident memory of the whole process TREE, sampled every
  `--poll-seconds` (the registered command is `env ... uv run ...`: `env`
  execs into `uv` and `uv` forks the interpreter that does the work, so the
  direct child's own rss would measure the launcher).

What it deliberately never does: print, parse, or store any log loss, Brier
score, ROI or edge. Subprocess output goes straight to `<output_dir>/run.log`
(run_arm.py writes no log of its own) and is never echoed here.

After the jobs, each shard's four arm directories are handed to
`audit_cross_arm.py --expected-fixtures <shard fixture dir>` and the
pass/fail is recorded in the same file. The audit's own output is written to
`<full root>/audit/` rather than echoed, for the same reason.

Usage (real run; Fable launches, this module never self-starts):

    uv run --no-sync python scripts/sequence_track/run_full_1d.py --dry-run
    nohup uv run --no-sync python scripts/sequence_track/run_full_1d.py &

A `STOP` file in the full root stops the runner between job LAUNCHES: jobs
already running are allowed to finish, nothing new starts.
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# 1b's mechanics are reused rather than restated: the completion test, the
# STOP test, the `ps` tree sampler, the timing-line reader, the atomic record
# writer and the exit codes are the same contracts, and a second copy of any
# of them could drift from the one 1b ran under.
from sequence_track.run_timing_1b import (  # noqa: E402
    AUDIT_DIRNAME, COMPLETION_ARTIFACTS, DEFAULT_POLL_SECONDS, EXIT_OK,
    EXIT_REFUSED, EXIT_RUN_FAILED, EXIT_STOPPED, LOG_FILENAME, STOP_FILENAME,
    _utc_now, is_complete, load_config, load_record, ps_table,
    read_total_simulation_seconds, sha256_file, stop_requested,
    tree_rss_bytes, write_record)

RECORD_CONTRACT = "sequence_track_full_1d_v1"
RECORD_FILENAME = "full_record.json"

DEFAULT_CONFIG = "experiments/configs/seq_stage1_sim_v1.yaml"
BLOCK = "full_run"

# The T1 arm first: it is the slowest per simulation, so the wall clock is
# set by when its last shard FINISHES, and queueing it first is the only
# lever this script has on that.
DEFAULT_ARMS = ("C", "B", "A50", "A")
DEFAULT_CONCURRENCY = 10

# `pin_stage1.CONVERGENCE_PLACEHOLDER`: the full-run n_sims was unchosen
# until the 1b convergence read. A config still carrying it is not launchable.
N_SIMS_PLACEHOLDER = "to_be_filled_before_1d"


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------


def full_run_block(config: dict, arm: str) -> dict:
    """The arm's registered `full_run` block, or a clear refusal."""
    arms = config.get("arms") or {}
    if arm not in arms:
        raise KeyError("config has no arm %r (has: %s)"
                       % (arm, ", ".join(sorted(arms))))
    block = (arms[arm] or {}).get(BLOCK)
    if not block:
        raise KeyError("arm %s has no %s block in the config" % (arm, BLOCK))
    for required in ("n_sims", "base_seeds", "output_dir", "shards"):
        if required not in block:
            raise KeyError("arm %s %s block has no %r" % (arm, BLOCK,
                                                          required))
    return block


def check_launchable(config: dict, arms: Sequence[str]) -> int:
    """The registered n_sims, refusing the pre-1b placeholder."""
    values = set()
    for arm in arms:
        n_sims = full_run_block(config, arm)["n_sims"]
        if not isinstance(n_sims, int) or isinstance(n_sims, bool):
            raise ValueError(
                "arm %s %s.n_sims is %r, not a launchable count: the full-run "
                "count is chosen by the 1b convergence read and re-pinned "
                "(pin_stage1.py --write) before 1d" % (arm, BLOCK, n_sims))
        values.add(int(n_sims))
    if len(values) != 1:
        raise ValueError("arms disagree on %s.n_sims: %r" % (BLOCK,
                                                             sorted(values)))
    return next(iter(values))


def shard_entries(config: dict, arm: str) -> Dict[int, dict]:
    """`{shard index: entry}` for one arm, refusing a malformed partition."""
    entries: Dict[int, dict] = {}
    for entry in (full_run_block(config, arm)["shards"] or []):
        if not isinstance(entry, dict):
            raise ValueError("arm %s: shard entry %r is not a mapping"
                             % (arm, entry))
        for required in ("index", "fixture_dir", "output_dir", "command"):
            if not entry.get(required) and entry.get(required) != 0:
                raise ValueError("arm %s: shard entry %r has no %r"
                                 % (arm, entry.get("index"), required))
        index = int(entry["index"])
        if index in entries:
            raise ValueError("arm %s: shard %d is registered twice"
                             % (arm, index))
        entries[index] = entry
    if not entries:
        raise ValueError(
            "arm %s registers no %s shards; the 1d partition is materialised "
            "and pinned by pin_stage1.py --write" % (arm, BLOCK))
    return entries


def shard_indices(config: dict, arms: Sequence[str]) -> List[int]:
    """The shard indices every arm registers, refusing any disagreement."""
    per_arm = {arm: sorted(shard_entries(config, arm)) for arm in arms}
    distinct = {tuple(value) for value in per_arm.values()}
    if len(distinct) != 1:
        raise ValueError(
            "arms register different shard sets, so they would not be "
            "covering the same partition: %r" % (per_arm,))
    return list(next(iter(distinct)))


def shard_fixture_dirs(config: dict, arms: Sequence[str]) -> Dict[int, str]:
    """`{shard index: fixture dir}`, identical across arms by construction."""
    dirs: Dict[int, str] = {}
    for index in shard_indices(config, arms):
        per_arm = {arm: str(shard_entries(config, arm)[index]["fixture_dir"])
                   for arm in arms}
        distinct = set(per_arm.values())
        if len(distinct) != 1:
            raise ValueError(
                "arms disagree on the fixture dir of shard %d: %r"
                % (index, per_arm))
        dirs[index] = next(iter(distinct))
    return dirs


def full_root(config: dict, arms: Sequence[str]) -> Path:
    """`models/embeddings/seq_stage1/full`, read off the output dirs.

    Each arm's `full_run.output_dir` is `<full root>/<arm>`, so the root is
    its parent. Deriving it keeps the record, the STOP file and the audit
    output inside the registered namespace without a second hard-coded copy
    of that path.
    """
    roots = {arm: str(Path(full_run_block(config, arm)["output_dir"]).parent)
             for arm in arms}
    distinct = set(roots.values())
    if len(distinct) != 1:
        raise ValueError("arms disagree on the full-run root: %r" % (roots,))
    return Path(next(iter(distinct)))


# --------------------------------------------------------------------------
# jobs
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class JobSpec:
    arm: str
    shard: int
    command: str
    output_dir: Path
    fixture_dir: str

    @property
    def argv(self) -> List[str]:
        return shlex.split(self.command)

    @property
    def log_path(self) -> Path:
        return self.output_dir / LOG_FILENAME

    @property
    def key(self) -> str:
        return "%s/shard%d" % (self.arm, self.shard)


def build_job_spec(config: dict, arm: str, shard: int) -> JobSpec:
    entry = shard_entries(config, arm)[int(shard)]
    return JobSpec(arm=arm, shard=int(shard), command=str(entry["command"]),
                   output_dir=Path(str(entry["output_dir"])),
                   fixture_dir=str(entry["fixture_dir"]))


def plan_jobs(config: dict, arms: Sequence[str],
              shards: Sequence[int]) -> List[JobSpec]:
    """Every job, arm-major in the given arm order, then by shard index.

    Arm-major is the schedule, not a formatting choice: the pool takes jobs
    off the front, so putting the slow T1 arm first starts all ten of its
    shards before any XGBoost shard is queued.
    """
    return [build_job_spec(config, arm, shard)
            for arm in arms
            for shard in sorted(int(value) for value in shards)]


@dataclass
class JobResult:
    arm: str
    shard: int
    command: str
    output_dir: str
    log_path: str
    started_at: str
    ended_at: str
    wall_seconds: float
    exit_code: int
    total_simulation_seconds: Optional[float]
    peak_rss_bytes: Optional[int]

    @property
    def key(self) -> str:
        return "%s/shard%d" % (self.arm, self.shard)

    def as_record(self) -> dict:
        return {
            "arm": self.arm,
            "shard": self.shard,
            "command": self.command,
            "output_dir": self.output_dir,
            "log_path": self.log_path,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "wall_seconds": round(self.wall_seconds, 3),
            "exit_code": self.exit_code,
            "total_simulation_seconds": self.total_simulation_seconds,
            "peak_rss_bytes": self.peak_rss_bytes,
            "peak_rss_mb": (None if self.peak_rss_bytes is None
                            else round(self.peak_rss_bytes / (1024 * 1024), 1)),
            "skipped": False,
        }


def launch_jobs(specs: Sequence[JobSpec], *,
                concurrency: int = DEFAULT_CONCURRENCY,
                poll_seconds: float = DEFAULT_POLL_SECONDS,
                stop_file=None, cwd: Path = REPO_ROOT,
                on_result: Callable[[JobResult], None] = None):
    """Run the specs through a pool of `concurrency` subprocesses.

    Returns `(results, stopped)`. The STOP file is tested before every
    LAUNCH, never mid-job: a 1d shard is hours long and killing one would
    leave a half-written output dir that `is_complete` would then re-run
    anyway. Each subprocess writes its own combined stdout/stderr to
    `<output_dir>/run.log`; nothing is streamed back here. `on_result` is
    called as each job finishes so the caller can persist the record while
    the rest are still running.
    """
    pending = list(specs)
    running: Dict[str, dict] = {}
    results: List[JobResult] = []
    stopped = False
    limit = max(1, int(concurrency))

    def _start(spec: JobSpec) -> dict:
        spec.output_dir.mkdir(parents=True, exist_ok=True)
        handle = spec.log_path.open("a", encoding="utf-8", errors="replace")
        handle.write("=== run_full_1d %s shard %d %s ===\n"
                     % (spec.arm, spec.shard, _utc_now()))
        handle.write("$ %s\n" % spec.command)
        handle.flush()
        process = subprocess.Popen(
            spec.argv, cwd=str(cwd), stdout=handle,
            stderr=subprocess.STDOUT, env=os.environ.copy())
        return {"spec": spec, "process": process, "handle": handle,
                "started_at": _utc_now(),
                "started_monotonic": time.monotonic(), "peak": 0}

    try:
        while pending or running:
            while pending and len(running) < limit and not stopped:
                if stop_file is not None and stop_requested(stop_file):
                    stopped = True
                    print("STOP file present (%s): launching nothing further; "
                          "%d job(s) still running, %d not started"
                          % (stop_file, len(running), len(pending)))
                    break
                spec = pending.pop(0)
                running[spec.key] = _start(spec)
                print("  [%s] launch: %s" % (spec.key, spec.output_dir))

            if not running:
                break

            table = ps_table()
            finished = []
            for key, job in running.items():
                if job["process"].poll() is None:
                    if table:
                        job["peak"] = max(
                            job["peak"],
                            tree_rss_bytes(table, job["process"].pid))
                else:
                    finished.append(key)

            for key in finished:
                job = running.pop(key)
                job["handle"].close()
                spec = job["spec"]
                result = JobResult(
                    arm=spec.arm, shard=spec.shard, command=spec.command,
                    output_dir=str(spec.output_dir),
                    log_path=str(spec.log_path),
                    started_at=job["started_at"], ended_at=_utc_now(),
                    wall_seconds=time.monotonic() - job["started_monotonic"],
                    exit_code=job["process"].returncode,
                    total_simulation_seconds=read_total_simulation_seconds(
                        spec.log_path),
                    peak_rss_bytes=job["peak"] or None)
                results.append(result)
                if on_result is not None:
                    on_result(result)

            if running or (pending and not stopped):
                time.sleep(max(0.05, float(poll_seconds)))
    except BaseException:
        for job in running.values():
            if job["process"].poll() is None:
                job["process"].terminate()
        for job in running.values():
            try:
                job["process"].wait(timeout=30)
            except Exception:  # pragma: no cover - best effort teardown
                job["process"].kill()
            job["handle"].close()
        raise

    return results, stopped


# --------------------------------------------------------------------------
# cross-arm audit, per shard
# --------------------------------------------------------------------------


def audit_command(arm_dirs: Sequence[Path], fixture_dir: str) -> List[str]:
    return (["uv", "run", "--no-sync", "python",
             "scripts/sequence_track/audit_cross_arm.py"]
            + [str(directory) for directory in arm_dirs]
            + ["--expected-fixtures", str(fixture_dir)])


def run_shard_audit(specs: Sequence[JobSpec], shard: int, fixture_dir: str,
                    audit_dir: Path, cwd: Path = REPO_ROOT) -> dict:
    """Audit one shard's arm dirs; record pass/fail only, never a metric."""
    arm_dirs = [spec.output_dir for spec in specs]
    missing = [str(directory) for directory in arm_dirs
               if not is_complete(directory)]
    if missing:
        return {"status": "skipped",
                "reason": "incomplete arm dirs: %s" % ", ".join(missing),
                "checked_at": _utc_now()}
    if len(arm_dirs) < 2:
        return {"status": "skipped",
                "reason": "a cross-arm audit needs at least two arms",
                "checked_at": _utc_now()}
    command = audit_command(arm_dirs, fixture_dir)
    audit_dir.mkdir(parents=True, exist_ok=True)
    report_path = audit_dir / ("shard%d.txt" % int(shard))
    completed = subprocess.run(command, cwd=str(cwd), capture_output=True,
                               text=True)
    report_path.write_text(
        "$ %s\n%s\n%s" % (" ".join(shlex.quote(part) for part in command),
                          completed.stdout, completed.stderr),
        encoding="utf-8")
    return {
        "status": "pass" if completed.returncode == 0 else "fail",
        "exit_code": completed.returncode,
        "command": " ".join(shlex.quote(part) for part in command),
        "report_path": str(report_path),
        "arm_dirs": [str(directory) for directory in arm_dirs],
        "expected_fixtures": str(fixture_dir),
        "checked_at": _utc_now(),
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage 1 step 1d: the full sharded run, 4 arms x 10 "
                    "shards.")
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help="registered stage-1 config (default %s)"
                             % DEFAULT_CONFIG)
    parser.add_argument("--arms", nargs="+", default=list(DEFAULT_ARMS),
                        help="arms to run, IN SCHEDULING ORDER (default: %s "
                             "— the slow T1 arm first)"
                             % " ".join(DEFAULT_ARMS))
    parser.add_argument("--shards", nargs="+", type=int, default=None,
                        help="shard indices to run (default: every shard the "
                             "config registers)")
    parser.add_argument("--concurrency", type=int,
                        default=DEFAULT_CONCURRENCY,
                        help="one-thread processes running at once "
                             "(default %d)" % DEFAULT_CONCURRENCY)
    parser.add_argument("--record", default=None,
                        help="run record JSON (default <full root>/%s)"
                             % RECORD_FILENAME)
    parser.add_argument("--stop-file", default=None,
                        help="stop launching new jobs when this exists "
                             "(default <full root>/%s)" % STOP_FILENAME)
    parser.add_argument("--poll-seconds", type=float,
                        default=DEFAULT_POLL_SECONDS,
                        help="resident-memory sampling interval (default 2 s)")
    parser.add_argument("--force", action="store_true",
                        help="re-run jobs whose output dirs already hold "
                             "%s" % " + ".join(COMPLETION_ARTIFACTS))
    parser.add_argument("--skip-audit", action="store_true",
                        help="do not run audit_cross_arm.py afterwards")
    parser.add_argument("--audit-only", action="store_true",
                        help="launch nothing; just run the per-shard "
                             "cross-arm audit and record the result")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the rendered commands and exit")
    return parser


def main(argv=None, *, launch: Callable[..., tuple] = launch_jobs,
         audit: Callable[..., dict] = run_shard_audit) -> int:
    args = build_parser().parse_args(argv)
    arms = list(args.arms)

    try:
        config = load_config(args.config)
        n_sims = check_launchable(config, arms)
        registered = shard_indices(config, arms)
        shards = ([int(value) for value in args.shards]
                  if args.shards is not None else list(registered))
        unregistered = [value for value in shards if value not in registered]
        if unregistered:
            raise ValueError(
                "shard(s) %s are not registered in %s.shards %s"
                % (", ".join(str(value) for value in unregistered), BLOCK,
                   registered))
        fixture_dirs = shard_fixture_dirs(config, arms)
        root = full_root(config, arms)
        jobs = plan_jobs(config, arms, shards)
    except (KeyError, ValueError) as exc:
        sys.stderr.write("refused: %s\n" % exc)
        return EXIT_REFUSED

    record_path = Path(args.record) if args.record else root / RECORD_FILENAME
    stop_file = (Path(args.stop_file) if args.stop_file
                 else root / STOP_FILENAME)
    audit_dir = root / AUDIT_DIRNAME

    if args.dry_run:
        print("full root:        %s" % root)
        print("run record:       %s" % record_path)
        print("stop file:        %s" % stop_file)
        print("n_sims:           %d" % n_sims)
        print("arms (in order):  %s" % " ".join(arms))
        print("shards:           %s" % " ".join(str(v) for v in shards))
        print("concurrency:      %d" % max(1, int(args.concurrency)))
        print("jobs:             %d" % len(jobs))
        for spec in jobs:
            state = ("complete (would skip)"
                     if is_complete(spec.output_dir) and not args.force
                     else "would run")
            print("\n[%s] %s" % (spec.key, state))
            print("  fixture-dir: %s (%s json)"
                  % (spec.fixture_dir,
                     len(sorted(Path(spec.fixture_dir).glob("*.json")))
                     if Path(spec.fixture_dir).is_dir() else "absent"))
            print("  output-dir:  %s" % spec.output_dir)
            print("  log:         %s" % spec.log_path)
            print("  $ %s" % spec.command)
        return EXIT_OK

    record = load_record(record_path)
    record.update({
        "contract": RECORD_CONTRACT,
        "config_path": str(args.config),
        "config_sha256": sha256_file(args.config),
        "arms": arms,
        "shards": shards,
        "n_sims": n_sims,
        "concurrency": max(1, int(args.concurrency)),
        "shard_fixture_dirs": {str(index): fixture_dirs[index]
                               for index in shards},
        "updated_at": _utc_now(),
    })
    record.setdefault("jobs", {})
    record.setdefault("audits", {})
    write_record(record_path, record)

    failures = 0
    pending = []
    for spec in jobs:
        if args.audit_only:
            continue
        if not args.force and is_complete(spec.output_dir):
            print("  [%s] skip: %s already has %s"
                  % (spec.key, spec.output_dir,
                     " + ".join(COMPLETION_ARTIFACTS)))
            if record["jobs"].get(spec.key) is None:
                record["jobs"][spec.key] = {
                    "arm": spec.arm, "shard": spec.shard,
                    "command": spec.command,
                    "output_dir": str(spec.output_dir),
                    "log_path": str(spec.log_path),
                    "skipped": True,
                    "total_simulation_seconds":
                        read_total_simulation_seconds(spec.log_path),
                    "noted_at": _utc_now(),
                }
            continue
        pending.append(spec)

    stopped = False
    if pending:
        print("\n=== %d job(s) to run at concurrency %d ==="
              % (len(pending), max(1, int(args.concurrency))))

        def _persist(result: JobResult) -> None:
            record["jobs"][result.key] = result.as_record()
            record["updated_at"] = _utc_now()
            write_record(record_path, record)
            print("  [%s] %s in %.1fs (exit %s, sim %s s, peak RSS %s MB)"
                  % (result.key,
                     "ok" if result.exit_code == 0 else "FAILED",
                     result.wall_seconds, result.exit_code,
                     result.total_simulation_seconds,
                     "n/a" if result.peak_rss_bytes is None
                     else round(result.peak_rss_bytes / (1024 * 1024), 1)))

        results, stopped = launch(
            pending, concurrency=max(1, int(args.concurrency)),
            poll_seconds=args.poll_seconds, stop_file=stop_file,
            on_result=_persist)
        for result in results:
            record["jobs"].setdefault(result.key, result.as_record())
            if result.exit_code != 0:
                failures += 1
        if stopped:
            record["stopped_at"] = _utc_now()
            record["stopped_launching"] = True
        write_record(record_path, record)

    if not args.skip_audit:
        by_shard: Dict[int, List[JobSpec]] = {}
        for spec in jobs:
            by_shard.setdefault(spec.shard, []).append(spec)
        for shard in sorted(by_shard):
            outcome = audit(by_shard[shard], shard, fixture_dirs[shard],
                            audit_dir)
            record["audits"]["shard%d" % shard] = outcome
            print("  [audit shard %d] %s%s"
                  % (shard, outcome["status"],
                     "" if outcome["status"] != "skipped"
                     else ": " + outcome.get("reason", "")))
            if outcome["status"] == "fail":
                failures += 1
        record["updated_at"] = _utc_now()
        write_record(record_path, record)

    record["updated_at"] = _utc_now()
    write_record(record_path, record)
    print("\nrun record: %s" % record_path)
    if stopped:
        return EXIT_STOPPED
    return EXIT_RUN_FAILED if failures else EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
