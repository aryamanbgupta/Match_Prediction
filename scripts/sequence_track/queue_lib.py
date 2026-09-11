#!/usr/bin/env python3
"""Helpers for the sequence-track queue runner (``research/sequence_track/run_queue.sh``).

Three jobs are done here rather than in shell, because each is fragile in bash:

``resolve``
    Parse and validate a queue file, then print the defaults and the fully
    resolved job list as tab-separated records the runner can read.
``mem``
    Read available memory from ``vm_stat`` (free + inactive + speculative pages
    times the page size) and optionally compare it against a floor. Exit 0 when
    the floor is met, 3 when it is not, and 4 when the reading itself failed --
    the runner fails closed on both non-zero cases. Tests can substitute the
    reading by setting ``QUEUE_VM_STAT_CMD`` to a command that prints
    ``vm_stat``-shaped output.
``watch``
    Poll a running job's whole process tree: kill it when its resident memory
    exceeds a cap (writing ``KILLED_MEMORY``) and kill it when it outlives its
    deadline (writing ``TIMEOUT``). The deadline is the second line of defence
    behind the per-job alarm, which only signals the one process it exec'd --
    a job that backgrounds a child, or that handles SIGALRM itself, otherwise
    leaves survivors holding the log pipe open.

This helper never invokes a model runner, a source-control command, or the
research verdict logger. It only reads the queue file, reads process/memory
state, and writes marker files.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]

GIB = 1024 ** 3
DEFAULT_POLL_SECONDS = 30.0
DEFAULT_STOP_FILE = "research/sequence_track/STOP"

# Test seam: a command printing vm_stat-shaped output, used instead of vm_stat.
VM_STAT_COMMAND_ENV = "QUEUE_VM_STAT_CMD"

# `mem` exit codes. 0 = floor met.
MEM_BELOW_FLOOR = 3
MEM_READING_FAILED = 4

REQUIRED_DEFAULT_FIELDS = ("memory_floor_gb", "memory_cap_gb")
KNOWN_DEFAULT_FIELDS = REQUIRED_DEFAULT_FIELDS + ("poll_seconds", "stop_file")
REQUIRED_JOB_FIELDS = (
    "id",
    "config",
    "command",
    "output_dir",
    "expected_hours",
    "machine",
)
KNOWN_JOB_FIELDS = REQUIRED_JOB_FIELDS + ("notes",)

ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

# Kill grace period between SIGTERM and SIGKILL for a job process tree.
KILL_GRACE_SECONDS = 3.0

# How long after the nominal deadline the watchdog waits before killing the
# tree itself, so the per-job alarm gets the first attempt.
TIMEOUT_GRACE_SECONDS = 2.0


# --------------------------------------------------------------------------
# small utilities
# --------------------------------------------------------------------------


def now_stamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def resolve_path(raw: str) -> Path:
    """Resolve a queue path: absolute as given, relative against the repo root."""
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def sha256_file(path: Path) -> str:
    digester = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digester.update(chunk)
    return digester.hexdigest()


def timeout_seconds(expected_hours: float) -> int:
    """Alarm seconds for a job: twice the expected wall time, integer, min 1."""
    return max(1, int(float(expected_hours) * 2 * 3600))


# --------------------------------------------------------------------------
# queue file parsing and validation
# --------------------------------------------------------------------------


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_clean_text(value: Any) -> bool:
    return isinstance(value, str) and value.strip() != "" and "\t" not in value and "\n" not in value


def load_queue(queue_path: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[str], List[str]]:
    """Return ``(defaults, jobs, errors, warnings)`` for a queue file.

    ``errors`` non-empty means the runner must exit non-zero before launching
    anything.
    """
    import yaml  # imported here so --help works without the dependency

    errors: List[str] = []
    warnings: List[str] = []

    if not queue_path.is_file():
        return {}, [], ["queue file not found: %s" % queue_path], warnings

    try:
        with open(queue_path, "r", encoding="utf-8") as handle:
            document = yaml.safe_load(handle)
    except Exception as exc:  # noqa: BLE001 - surfaced as a validation error
        return {}, [], ["queue file is not valid YAML: %s" % exc], warnings

    if not isinstance(document, dict):
        return {}, [], ["queue file must be a mapping with 'defaults' and 'jobs'"], warnings

    raw_defaults = document.get("defaults")
    if raw_defaults is None:
        errors.append("missing top-level 'defaults' mapping")
        raw_defaults = {}
    elif not isinstance(raw_defaults, dict):
        errors.append("'defaults' must be a mapping")
        raw_defaults = {}

    for field in REQUIRED_DEFAULT_FIELDS:
        if field not in raw_defaults:
            errors.append("defaults: missing required field '%s'" % field)
        elif not _is_number(raw_defaults[field]) or float(raw_defaults[field]) <= 0:
            errors.append("defaults: '%s' must be a positive number" % field)

    poll_seconds = raw_defaults.get("poll_seconds", DEFAULT_POLL_SECONDS)
    if not _is_number(poll_seconds) or float(poll_seconds) <= 0:
        errors.append("defaults: 'poll_seconds' must be a positive number")
        poll_seconds = DEFAULT_POLL_SECONDS

    stop_file = raw_defaults.get("stop_file", DEFAULT_STOP_FILE)
    if not _is_clean_text(stop_file):
        errors.append("defaults: 'stop_file' must be a non-empty single-line string")
        stop_file = DEFAULT_STOP_FILE

    for field in sorted(set(raw_defaults) - set(KNOWN_DEFAULT_FIELDS)):
        warnings.append("defaults: ignoring unknown field '%s'" % field)

    defaults = {
        "memory_floor_gb": float(raw_defaults.get("memory_floor_gb", 0) or 0),
        "memory_cap_gb": float(raw_defaults.get("memory_cap_gb", 0) or 0),
        "poll_seconds": float(poll_seconds),
        "stop_file": str(stop_file),
    }

    raw_jobs = document.get("jobs")
    if raw_jobs is None:
        errors.append("missing top-level 'jobs' list")
        raw_jobs = []
    elif not isinstance(raw_jobs, list):
        errors.append("'jobs' must be a list")
        raw_jobs = []
    elif not raw_jobs:
        warnings.append("'jobs' list is empty")

    jobs: List[Dict[str, Any]] = []
    seen_ids: Set[str] = set()

    for index, raw_job in enumerate(raw_jobs):
        label = "job[%d]" % index
        if not isinstance(raw_job, dict):
            errors.append("%s: must be a mapping" % label)
            continue

        job_id = raw_job.get("id")
        if _is_clean_text(job_id):
            label = "job '%s'" % job_id

        missing = [field for field in REQUIRED_JOB_FIELDS if raw_job.get(field) is None]
        if missing:
            errors.append("%s: missing required field(s): %s" % (label, ", ".join(missing)))
            continue

        ok = True
        if not _is_clean_text(job_id) or not ID_PATTERN.match(str(job_id)):
            errors.append(
                "%s: 'id' must match [A-Za-z0-9][A-Za-z0-9._-]* (it names log lines and markers)"
                % label
            )
            ok = False
        elif job_id in seen_ids:
            errors.append("%s: duplicate id" % label)
            ok = False

        for field in ("config", "command", "output_dir", "machine"):
            if not _is_clean_text(raw_job.get(field)):
                errors.append(
                    "%s: '%s' must be a non-empty string without tabs or newlines" % (label, field)
                )
                ok = False

        expected_hours = raw_job.get("expected_hours")
        if not _is_number(expected_hours) or float(expected_hours) <= 0:
            errors.append("%s: 'expected_hours' must be a positive number" % label)
            ok = False

        for field in sorted(set(raw_job) - set(KNOWN_JOB_FIELDS)):
            warnings.append("%s: ignoring unknown field '%s'" % (label, field))

        if not ok:
            continue

        seen_ids.add(str(job_id))
        jobs.append(
            {
                "id": str(job_id),
                "config": str(raw_job["config"]),
                "command": str(raw_job["command"]),
                "output_dir": str(raw_job["output_dir"]),
                "expected_hours": float(expected_hours),
                "machine": str(raw_job["machine"]),
            }
        )

    return defaults, jobs, errors, warnings


def config_signature(config_path: Path, raw_value: str) -> Tuple[str, Optional[str]]:
    """sha256 of a job's config file, or a stable sentinel when it is absent.

    A job whose config has not been written yet still has a stable signature, so
    a completed placeholder run is not re-run on every pass; when the real file
    lands the signature changes and the job re-runs, which is the behaviour the
    skip-if-complete rule wants.
    """
    if config_path.is_file():
        return sha256_file(config_path), None
    return (
        "absent:%s" % raw_value,
        "config file not found: %s (signature recorded as a sentinel)" % config_path,
    )


# --------------------------------------------------------------------------
# vm_stat memory reading
# --------------------------------------------------------------------------


class MemoryReadingError(RuntimeError):
    """The available-memory reading could not be taken or could not be parsed."""


def read_available_memory() -> Dict[str, float]:
    """Available memory from ``vm_stat``: (free + inactive + speculative) pages.

    Raises ``MemoryReadingError`` rather than guessing, so the runner can fail
    closed: an unreadable machine is not evidence that memory is available.
    """
    command = shlex.split(os.environ.get(VM_STAT_COMMAND_ENV, "").strip()) or ["vm_stat"]
    try:
        completed = subprocess.run(command, capture_output=True, text=True)
    except OSError as exc:
        raise MemoryReadingError("could not run %s: %s" % (command[0], exc))
    if completed.returncode != 0:
        raise MemoryReadingError(
            "%s exited %d: %s"
            % (command[0], completed.returncode, completed.stderr.strip()[:200] or "no stderr")
        )
    output = completed.stdout

    page_size = 4096
    header = re.search(r"page size of (\d+) bytes", output)
    if header:
        page_size = int(header.group(1))

    pages: Dict[str, int] = {}
    for line in output.splitlines():
        match = re.match(r"^Pages\s+([A-Za-z ]+):\s+(\d+)\.", line.strip())
        if match:
            pages[match.group(1).strip().lower()] = int(match.group(2))

    if not {"free", "inactive", "speculative"} & set(pages):
        raise MemoryReadingError(
            "%s printed no page counts: %r" % (command[0], output.strip()[:200])
        )

    free = pages.get("free", 0)
    inactive = pages.get("inactive", 0)
    speculative = pages.get("speculative", 0)
    total_pages = free + inactive + speculative

    return {
        "page_size": float(page_size),
        "free_gb": free * page_size / GIB,
        "inactive_gb": inactive * page_size / GIB,
        "speculative_gb": speculative * page_size / GIB,
        "available_gb": total_pages * page_size / GIB,
    }


# --------------------------------------------------------------------------
# process-tree resident memory watchdog
# --------------------------------------------------------------------------


def ps_snapshot() -> Dict[int, Tuple[int, int, int]]:
    """``{pid: (ppid, pgid, rss_kib)}`` for every process on the machine."""
    try:
        output = subprocess.run(
            ["ps", "-Ao", "pid=,ppid=,pgid=,rss="], capture_output=True, text=True
        ).stdout
    except OSError:
        return {}

    table: Dict[int, Tuple[int, int, int]] = {}
    for line in output.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            pid, ppid, pgid, rss = (int(part) for part in parts[:4])
        except ValueError:
            continue
        table[pid] = (ppid, pgid, rss)
    return table


def descendants(table: Dict[int, Tuple[int, int, int]], root_pid: int) -> Set[int]:
    children: Dict[int, List[int]] = {}
    for pid, (ppid, _pgid, _rss) in table.items():
        children.setdefault(ppid, []).append(pid)

    found: Set[int] = set()
    pending = [root_pid]
    while pending:
        pid = pending.pop()
        if pid in found or pid not in table:
            continue
        found.add(pid)
        pending.extend(children.get(pid, []))
    return found


def tree_pids(
    table: Dict[int, Tuple[int, int, int]], root_pid: int, target_pgid: Optional[int]
) -> Set[int]:
    if target_pgid:
        group = {pid for pid, (_ppid, pgid, _rss) in table.items() if pgid == target_pgid}
        if group:
            return group
    return descendants(table, root_pid)


def tree_rss_bytes(table: Dict[int, Tuple[int, int, int]], pids: Iterable[int]) -> int:
    return sum(table[pid][2] for pid in pids if pid in table) * 1024


def kill_tree(root_pid: int, pids: Set[int], target_pgid: Optional[int]) -> None:
    """SIGTERM the whole tree, then SIGKILL anything still alive."""
    for signal_number in (signal.SIGTERM, signal.SIGKILL):
        sent = False
        if target_pgid:
            try:
                os.killpg(target_pgid, signal_number)
                sent = True
            except OSError:
                sent = False
        if not sent:
            for pid in sorted(pids):
                try:
                    os.kill(pid, signal_number)
                except OSError:
                    pass

        if signal_number is signal.SIGTERM:
            deadline = time.time() + KILL_GRACE_SECONDS
            while time.time() < deadline:
                if not tree_pids(ps_snapshot(), root_pid, target_pgid):
                    return
                time.sleep(0.2)


# --------------------------------------------------------------------------
# subcommands
# --------------------------------------------------------------------------


def cmd_resolve(args: argparse.Namespace) -> int:
    queue_path = resolve_path(args.queue)
    defaults, jobs, errors, warnings = load_queue(queue_path)

    if errors:
        for message in errors:
            sys.stderr.write("queue error: %s\n" % message)
        return 2

    records: List[str] = []
    records.append("DEFAULT\tmemory_floor_gb\t%.6g" % defaults["memory_floor_gb"])
    records.append("DEFAULT\tmemory_cap_gb\t%.6g" % defaults["memory_cap_gb"])
    records.append("DEFAULT\tpoll_seconds\t%.6g" % defaults["poll_seconds"])
    records.append("DEFAULT\tstop_file\t%s" % resolve_path(defaults["stop_file"]))

    for message in warnings:
        records.append("WARN\t%s" % message)

    for job in jobs:
        config_path = resolve_path(job["config"])
        signature, warning = config_signature(config_path, job["config"])
        if warning:
            records.append("WARN\tjob '%s': %s" % (job["id"], warning))
        records.append(
            "JOB\t%s\t%s\t%.6g\t%d\t%s\t%s\t%s\t%s"
            % (
                job["id"],
                job["machine"],
                job["expected_hours"],
                timeout_seconds(job["expected_hours"]),
                config_path,
                signature,
                resolve_path(job["output_dir"]),
                job["command"],
            )
        )

    sys.stdout.write("\n".join(records) + "\n")
    return 0


def cmd_mem(args: argparse.Namespace) -> int:
    try:
        reading = read_available_memory()
    except MemoryReadingError as exc:
        sys.stdout.write("memory: reading failed: %s\n" % exc)
        return MEM_READING_FAILED

    line = (
        "memory: available %.2f GiB (free %.2f + inactive %.2f + speculative %.2f; "
        "page size %d B)"
        % (
            reading["available_gb"],
            reading["free_gb"],
            reading["inactive_gb"],
            reading["speculative_gb"],
            int(reading["page_size"]),
        )
    )
    if args.floor_gb is None:
        sys.stdout.write(line + "\n")
        return 0

    below = reading["available_gb"] < float(args.floor_gb)
    sys.stdout.write(
        "%s; floor %.2f GiB -> %s\n" % (line, float(args.floor_gb), "BELOW" if below else "OK")
    )
    return MEM_BELOW_FLOOR if below else 0


def _write_marker(path: str, body: str) -> None:
    marker = Path(path)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(body, encoding="utf-8")


def cmd_watch(args: argparse.Namespace) -> int:
    cap_bytes = int(float(args.cap_gb) * GIB)
    poll = max(0.05, float(args.poll))
    own_pgid = os.getpgid(0)
    target_pgid: Optional[int] = None

    timeout = float(args.timeout_seconds or 0)
    # The alarm fires at `timeout`; the watchdog only steps in once the grace
    # period has passed and the tree is somehow still alive.
    deadline = (
        time.monotonic() + timeout + max(TIMEOUT_GRACE_SECONDS, poll) if timeout > 0 else None
    )

    while True:
        table = ps_snapshot()

        if target_pgid is None and args.pid in table:
            candidate = table[args.pid][1]
            if candidate and candidate != own_pgid:
                target_pgid = candidate

        pids = tree_pids(table, args.pid, target_pgid)
        if not pids:
            return 0

        observed = tree_rss_bytes(table, pids)
        if observed > cap_bytes:
            stamp = now_stamp()
            _write_marker(
                args.marker,
                "job_id: %s\n"
                "observed_rss_bytes: %d\n"
                "observed_rss_gb: %.3f\n"
                "cap_gb: %.3f\n"
                "pids: %s\n"
                "killed: %s\n"
                % (
                    args.job_id,
                    observed,
                    observed / GIB,
                    float(args.cap_gb),
                    ",".join(str(pid) for pid in sorted(pids)),
                    stamp,
                ),
            )
            sys.stdout.write(
                "=== watchdog %s: tree RSS %.3f GiB above cap %.3f GiB, killing %d process(es) %s ===\n"
                % (args.job_id, observed / GIB, float(args.cap_gb), len(pids), stamp)
            )
            sys.stdout.flush()
            kill_tree(args.pid, pids, target_pgid)
            return 1

        if deadline is not None and time.monotonic() >= deadline:
            stamp = now_stamp()
            if args.timeout_marker:
                _write_marker(
                    args.timeout_marker,
                    "job_id: %s\n"
                    "timeout_seconds: %d\n"
                    "killed_by: watchdog\n"
                    "pids: %s\n"
                    "killed: %s\n"
                    % (
                        args.job_id,
                        int(timeout),
                        ",".join(str(pid) for pid in sorted(pids)),
                        stamp,
                    ),
                )
            sys.stdout.write(
                "=== watchdog %s: still alive %.0fs past the %ds timeout "
                "(the alarm did not reach every process), killing %d process(es) %s ===\n"
                % (
                    args.job_id,
                    max(TIMEOUT_GRACE_SECONDS, poll),
                    int(timeout),
                    len(pids),
                    stamp,
                )
            )
            sys.stdout.flush()
            kill_tree(args.pid, pids, target_pgid)
            return 2

        sleep_for = poll
        if deadline is not None:
            sleep_for = min(sleep_for, max(0.05, deadline - time.monotonic()))
        time.sleep(sleep_for)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="command", required=True)

    resolve_parser = subparsers.add_parser(
        "resolve", help="validate the queue file and print defaults + jobs as TSV"
    )
    resolve_parser.add_argument("--queue", required=True)
    resolve_parser.set_defaults(func=cmd_resolve)

    mem_parser = subparsers.add_parser("mem", help="read available memory from vm_stat")
    mem_parser.add_argument("--floor-gb", default=None)
    mem_parser.set_defaults(func=cmd_mem)

    watch_parser = subparsers.add_parser(
        "watch", help="memory and deadline watchdog for a job process tree"
    )
    watch_parser.add_argument("--pid", type=int, required=True)
    watch_parser.add_argument("--cap-gb", required=True)
    watch_parser.add_argument("--poll", default=DEFAULT_POLL_SECONDS)
    watch_parser.add_argument("--marker", required=True, help="KILLED_MEMORY marker path")
    watch_parser.add_argument(
        "--timeout-seconds",
        default=0,
        help="kill the tree this long (plus a grace period) after launch; 0 disables",
    )
    watch_parser.add_argument("--timeout-marker", default="", help="TIMEOUT marker path")
    watch_parser.add_argument("--job-id", default="job")
    watch_parser.set_defaults(func=cmd_watch)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
