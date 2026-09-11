#!/usr/bin/env python3
"""Stage 1 step 1b runner: timing and convergence batches for all four arms.

This script OWNS the mechanics of 1b and nothing else. It renders each arm's
registered `timing_1b.command_template` for one (n_sims, base seed) pair,
launches the FOUR ARMS CONCURRENTLY (each command already caps its own BLAS
threads at 4), waits for the group, and moves on to the next pair. Groups run
in ascending `n_sims`, then in registered base-seed order, so the cheap
candidates land first and a stop between groups always leaves a complete set
of batches behind.

What it records, per (arm, n_sims, base_seed), into
`<timing root>/timing_record.json`:

* wall seconds, exit code, start/end UTC timestamps;
* `Total simulation time: <x>s` as parsed from the run's own log — the ONLY
  number read back out of a run log;
* peak resident memory, sampled every `--poll-seconds` while the group runs.

What it deliberately never does: print, parse, or store any log loss, Brier
score, ROI or edge. 1b chooses `n_sims`; it is not allowed to see the result
that choice will be used to compute. Subprocess output goes straight to
`<output_dir>/run.log` (run_arm.py writes no log of its own — the smoke
round's logs came from shell redirection), it is never echoed here, and the
only line matched against it is the timing line above.

Peak RSS is the resident size of the arm's whole process TREE. The registered
command is `env ... uv run --no-sync python ...`: `env` execs into `uv`, and
`uv` forks the interpreter that does the work, so the direct child's own
`rss` would measure the launcher rather than the simulation.

After the batches, the four arm directories of each (n_sims, base_seed) group
are handed to `audit_cross_arm.py --expected-fixtures <shard dir>` and the
pass/fail is recorded in the same file. The audit's own output is written to
`<timing root>/audit/` rather than echoed, for the same reason as above.

Usage (real run; Fable launches, this module never self-starts):

    uv run --no-sync python scripts/sequence_track/run_timing_1b.py \
        --candidates 50 100 200 400 800

    uv run --no-sync python scripts/sequence_track/run_timing_1b.py \
        --candidates 50 --dry-run

A `STOP` file in the timing root stops the runner between groups.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import yaml

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

RECORD_CONTRACT = "sequence_track_timing_1b_v1"
RECORD_FILENAME = "timing_record.json"
STOP_FILENAME = "STOP"
LOG_FILENAME = "run.log"
AUDIT_DIRNAME = "audit"

DEFAULT_CONFIG = "experiments/configs/seq_stage1_sim_v1.yaml"
DEFAULT_ARMS = ("A", "A50", "B", "C")
DEFAULT_CANDIDATES = (50, 100, 200, 400, 800)
DEFAULT_POLL_SECONDS = 2.0
BLOCK = "timing_1b"

# run_arm.py's two completion artifacts. A directory holding both is a run
# that finished; anything less is re-run rather than trusted.
COMPLETION_ARTIFACTS = ("eval.json", "arm_provenance.json")

# `match_evaluator.py` prints exactly `Total simulation time: 1.2s`.
TOTAL_SIM_TIME_RE = re.compile(
    r"^Total simulation time:\s*([0-9]+(?:\.[0-9]+)?)\s*s\s*$", re.MULTILINE
)

EXIT_OK = 0
EXIT_RUN_FAILED = 1
EXIT_REFUSED = 2
EXIT_STOPPED = 3

KIB = 1024


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------


def load_config(config_path) -> dict:
    with Path(config_path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def timing_block(config: dict, arm: str) -> dict:
    """The arm's registered `timing_1b` block, or a clear refusal."""
    arms = config.get("arms") or {}
    if arm not in arms:
        raise KeyError("config has no arm %r (has: %s)"
                       % (arm, ", ".join(sorted(arms))))
    block = (arms[arm] or {}).get(BLOCK)
    if not block:
        raise KeyError("arm %s has no %s block in the config" % (arm, BLOCK))
    for required in ("command_template", "output_dir_template", "n_sims",
                     "base_seeds", "fixture_dir"):
        if required not in block:
            raise KeyError("arm %s %s block has no %r"
                           % (arm, BLOCK, required))
    return block


def _agreed(config: dict, arms: Sequence[str], key: str):
    """One value of `key` shared by every arm's timing block, or a refusal."""
    values = {arm: timing_block(config, arm)[key] for arm in arms}
    distinct = {json.dumps(value, sort_keys=True) for value in values.values()}
    if len(distinct) != 1:
        rendered = "; ".join("%s=%r" % (arm, value)
                             for arm, value in sorted(values.items()))
        raise ValueError(
            "arms disagree on %s.%s, so they would not be running the same "
            "1b block: %s" % (BLOCK, key, rendered))
    return next(iter(values.values()))


def permitted_n_sims(config: dict, arms: Sequence[str]) -> Tuple[int, ...]:
    return tuple(int(value) for value in _agreed(config, arms, "n_sims"))


def registered_base_seeds(config: dict, arms: Sequence[str]) -> Tuple[int, ...]:
    return tuple(int(value) for value in _agreed(config, arms, "base_seeds"))


def shard_fixture_dir(config: dict, arms: Sequence[str]) -> str:
    return str(_agreed(config, arms, "fixture_dir"))


def expected_fixture_count(config: dict, arms: Sequence[str]) -> Optional[int]:
    counts = {arm: timing_block(config, arm).get("fixture_count_expected")
              for arm in arms}
    distinct = {value for value in counts.values() if value is not None}
    if len(distinct) > 1:
        raise ValueError("arms disagree on %s.fixture_count_expected: %r"
                         % (BLOCK, counts))
    return int(next(iter(distinct))) if distinct else None


def timing_root(config: dict, arms: Sequence[str]) -> Path:
    """`models/embeddings/seq_stage1/timing`, read off the output templates.

    The templates end `<timing root>/<arm>/seed<base_seed>_n<n_sims>`, so the
    root is the grandparent. Deriving it keeps the record, the STOP file and
    the audit output inside the registered namespace without this script
    hard-coding a second copy of that path.
    """
    roots = {}
    for arm in arms:
        template = str(timing_block(config, arm)["output_dir_template"])
        roots[arm] = str(Path(template).parent.parent)
    distinct = set(roots.values())
    if len(distinct) != 1:
        raise ValueError("arms disagree on the timing root: %r" % (roots,))
    return Path(next(iter(distinct)))


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

# `<base_seed>` is NOT a substring of `<batch_base_seed>` (the character
# before `base_seed>` there is `_`, not `<`), so plain replacement of each
# token is unambiguous.
TOKENS = ("<candidate>", "<batch_base_seed>", "<n_sims>", "<base_seed>")


def render_template(template: str, n_sims: int, base_seed: int) -> str:
    """Substitute the four registered 1b tokens, refusing any leftover."""
    rendered = (str(template)
                .replace("<candidate>", str(n_sims))
                .replace("<batch_base_seed>", str(base_seed))
                .replace("<n_sims>", str(n_sims))
                .replace("<base_seed>", str(base_seed)))
    leftover = re.findall(r"<[a-z_]+>", rendered)
    if leftover:
        raise ValueError(
            "unresolved token(s) %s in rendered template %r"
            % (", ".join(sorted(set(leftover))), rendered))
    return rendered


@dataclass(frozen=True)
class RunSpec:
    arm: str
    n_sims: int
    base_seed: int
    command: str
    output_dir: Path

    @property
    def argv(self) -> List[str]:
        return shlex.split(self.command)

    @property
    def log_path(self) -> Path:
        return self.output_dir / LOG_FILENAME

    @property
    def key(self) -> str:
        return "%s/n%d/seed%d" % (self.arm, self.n_sims, self.base_seed)


@dataclass
class Group:
    n_sims: int
    base_seed: int
    specs: List[RunSpec] = field(default_factory=list)

    @property
    def key(self) -> str:
        return "n%d/seed%d" % (self.n_sims, self.base_seed)


def build_run_spec(config: dict, arm: str, n_sims: int,
                   base_seed: int) -> RunSpec:
    block = timing_block(config, arm)
    command = render_template(block["command_template"], n_sims, base_seed)
    output_dir = render_template(block["output_dir_template"], n_sims,
                                 base_seed)
    return RunSpec(arm=arm, n_sims=n_sims, base_seed=base_seed,
                   command=command, output_dir=Path(output_dir))


def plan_groups(config: dict, arms: Sequence[str], candidates: Sequence[int],
                base_seeds: Sequence[int]) -> List[Group]:
    """Every group to run, in ascending n_sims then registered seed order."""
    groups: List[Group] = []
    for n_sims in sorted(int(value) for value in candidates):
        for base_seed in base_seeds:
            group = Group(n_sims=n_sims, base_seed=int(base_seed))
            for arm in arms:
                group.specs.append(
                    build_run_spec(config, arm, n_sims, int(base_seed)))
            groups.append(group)
    return groups


def check_candidates(config: dict, arms: Sequence[str],
                     candidates: Sequence[int]) -> None:
    """Refuse any n_sims the registered block does not permit."""
    permitted = permitted_n_sims(config, arms)
    unpermitted = [int(value) for value in candidates
                   if int(value) not in permitted]
    if unpermitted:
        raise ValueError(
            "n_sims %s not permitted by %s.n_sims %s: the seed-overlap screen "
            "registers which candidates the batch seeds can carry, so an "
            "unregistered candidate has to be re-pinned (pin_stage1.py), not "
            "passed on the command line"
            % (", ".join(str(value) for value in unpermitted), BLOCK,
               list(permitted)))


def check_base_seeds(config: dict, arms: Sequence[str],
                     base_seeds: Sequence[int]) -> None:
    registered = registered_base_seeds(config, arms)
    unregistered = [int(value) for value in base_seeds
                    if int(value) not in registered]
    if unregistered:
        raise ValueError(
            "base seed(s) %s are not registered in %s.base_seeds %s"
            % (", ".join(str(value) for value in unregistered), BLOCK,
               list(registered)))


# --------------------------------------------------------------------------
# completion, stop file, log parsing
# --------------------------------------------------------------------------


def is_complete(output_dir) -> bool:
    """True when a previous run left both of run_arm's completion artifacts."""
    directory = Path(output_dir)
    return all((directory / name).is_file() for name in COMPLETION_ARTIFACTS)


def stop_requested(stop_file) -> bool:
    return Path(stop_file).exists()


def parse_total_simulation_seconds(text: str) -> Optional[float]:
    """The evaluator's own `Total simulation time: <x>s`, or None.

    The last match wins: a forced re-run appends to the same log.
    """
    matches = TOTAL_SIM_TIME_RE.findall(text or "")
    return float(matches[-1]) if matches else None


def read_total_simulation_seconds(log_path) -> Optional[float]:
    path = Path(log_path)
    if not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    return parse_total_simulation_seconds(text)


# --------------------------------------------------------------------------
# resident memory sampling
# --------------------------------------------------------------------------


def ps_table() -> Dict[int, Tuple[int, int]]:
    """`{pid: (ppid, rss_bytes)}` from one `ps` snapshot."""
    try:
        completed = subprocess.run(["ps", "-Ao", "pid=,ppid=,rss="],
                                   capture_output=True, text=True)
    except OSError:
        return {}
    table: Dict[int, Tuple[int, int]] = {}
    for line in completed.stdout.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            pid, ppid, rss_kib = (int(part) for part in parts[:3])
        except ValueError:
            continue
        table[pid] = (ppid, rss_kib * KIB)
    return table


def tree_rss_bytes(table: Dict[int, Tuple[int, int]], root_pid: int) -> int:
    """Resident bytes of `root_pid` plus every descendant in the snapshot."""
    children: Dict[int, List[int]] = {}
    for pid, (ppid, _rss) in table.items():
        children.setdefault(ppid, []).append(pid)
    total = 0
    seen = set()
    pending = [root_pid]
    while pending:
        pid = pending.pop()
        if pid in seen or pid not in table:
            continue
        seen.add(pid)
        total += table[pid][1]
        pending.extend(children.get(pid, ()))
    return total


# --------------------------------------------------------------------------
# running one group
# --------------------------------------------------------------------------


@dataclass
class RunResult:
    arm: str
    n_sims: int
    base_seed: int
    command: str
    output_dir: str
    log_path: str
    started_at: str
    ended_at: str
    wall_seconds: float
    exit_code: int
    total_simulation_seconds: Optional[float]
    peak_rss_bytes: Optional[int]

    def as_record(self) -> dict:
        return {
            "arm": self.arm,
            "n_sims": self.n_sims,
            "base_seed": self.base_seed,
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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def launch_group(specs: Sequence[RunSpec], *, poll_seconds: float = DEFAULT_POLL_SECONDS,
                 cwd: Path = REPO_ROOT) -> List[RunResult]:
    """Run every spec CONCURRENTLY, one subprocess per arm, and wait.

    Each subprocess writes its own combined stdout/stderr to
    `<output_dir>/run.log`; nothing is streamed back here. Peak tree RSS is
    sampled every `poll_seconds` while any process is alive.
    """
    if not specs:
        return []

    processes: Dict[str, subprocess.Popen] = {}
    handles = []
    started_monotonic: Dict[str, float] = {}
    started_at: Dict[str, str] = {}
    ended_monotonic: Dict[str, float] = {}
    ended_at: Dict[str, str] = {}
    peak_rss: Dict[str, int] = {spec.arm: 0 for spec in specs}
    threads: List[threading.Thread] = []

    def _wait_for(arm: str, process: subprocess.Popen) -> None:
        process.wait()
        ended_monotonic[arm] = time.monotonic()
        ended_at[arm] = _utc_now()

    try:
        for spec in specs:
            spec.output_dir.mkdir(parents=True, exist_ok=True)
            handle = spec.log_path.open("a", encoding="utf-8", errors="replace")
            handles.append(handle)
            handle.write("=== run_timing_1b %s n_sims=%d base_seed=%d %s ===\n"
                         % (spec.arm, spec.n_sims, spec.base_seed, _utc_now()))
            handle.write("$ %s\n" % spec.command)
            handle.flush()
            started_at[spec.arm] = _utc_now()
            started_monotonic[spec.arm] = time.monotonic()
            processes[spec.arm] = subprocess.Popen(
                spec.argv, cwd=str(cwd), stdout=handle,
                stderr=subprocess.STDOUT, env=os.environ.copy())
            thread = threading.Thread(
                target=_wait_for, args=(spec.arm, processes[spec.arm]),
                daemon=True)
            thread.start()
            threads.append(thread)

        while any(thread.is_alive() for thread in threads):
            table = ps_table()
            if table:
                for arm, process in processes.items():
                    if process.poll() is None:
                        peak_rss[arm] = max(peak_rss[arm],
                                            tree_rss_bytes(table, process.pid))
            time.sleep(max(0.05, float(poll_seconds)))

        for thread in threads:
            thread.join()
    except BaseException:
        for process in processes.values():
            if process.poll() is None:
                process.terminate()
        for process in processes.values():
            try:
                process.wait(timeout=30)
            except Exception:  # pragma: no cover - best effort teardown
                process.kill()
        raise
    finally:
        for handle in handles:
            handle.close()

    results: List[RunResult] = []
    for spec in specs:
        arm = spec.arm
        end = ended_monotonic.get(arm, time.monotonic())
        results.append(RunResult(
            arm=arm,
            n_sims=spec.n_sims,
            base_seed=spec.base_seed,
            command=spec.command,
            output_dir=str(spec.output_dir),
            log_path=str(spec.log_path),
            started_at=started_at.get(arm, ""),
            ended_at=ended_at.get(arm, _utc_now()),
            wall_seconds=end - started_monotonic[arm],
            exit_code=processes[arm].returncode,
            total_simulation_seconds=read_total_simulation_seconds(spec.log_path),
            peak_rss_bytes=peak_rss.get(arm) or None,
        ))
    return results


# --------------------------------------------------------------------------
# the timing record
# --------------------------------------------------------------------------


def load_record(record_path) -> dict:
    path = Path(record_path)
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_record(record_path, record: dict) -> None:
    path = Path(record_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temp.replace(path)


def merge_run(record: dict, key: str, entry: dict) -> dict:
    record.setdefault("runs", {})[key] = entry
    return record


# --------------------------------------------------------------------------
# cross-arm audit
# --------------------------------------------------------------------------


def audit_command(arm_dirs: Sequence[Path], fixture_dir: str) -> List[str]:
    return (["uv", "run", "--no-sync", "python",
             "scripts/sequence_track/audit_cross_arm.py"]
            + [str(directory) for directory in arm_dirs]
            + ["--expected-fixtures", str(fixture_dir)])


def run_audit(group: Group, fixture_dir: str, audit_dir: Path,
              cwd: Path = REPO_ROOT) -> dict:
    """Audit one (n_sims, seed) group's four arm dirs; record pass/fail only.

    The audit's stdout is written to a file rather than echoed: it is a
    provenance report, but 1b keeps a blanket no-echo rule on run output.
    """
    arm_dirs = [spec.output_dir for spec in group.specs]
    missing = [str(directory) for directory in arm_dirs
               if not is_complete(directory)]
    if missing:
        return {"status": "skipped", "reason": "incomplete arm dirs: %s"
                % ", ".join(missing), "checked_at": _utc_now()}
    command = audit_command(arm_dirs, fixture_dir)
    audit_dir.mkdir(parents=True, exist_ok=True)
    report_path = audit_dir / ("n%d_seed%d.txt" % (group.n_sims,
                                                   group.base_seed))
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
        description="Stage 1 step 1b: timing and convergence batches.")
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help="registered stage-1 config (default %s)"
                             % DEFAULT_CONFIG)
    parser.add_argument("--arms", nargs="+", default=list(DEFAULT_ARMS),
                        help="arms to run concurrently in each group")
    parser.add_argument("--candidates", nargs="+", type=int,
                        default=list(DEFAULT_CANDIDATES),
                        help="n_sims candidates; every one must appear in the "
                             "config's timing_1b.n_sims")
    parser.add_argument("--base-seeds", nargs="+", type=int, default=None,
                        help="batch base seeds (default: the registered ones)")
    parser.add_argument("--record", default=None,
                        help="timing record JSON (default <timing root>/%s)"
                             % RECORD_FILENAME)
    parser.add_argument("--stop-file", default=None,
                        help="stop between groups when this exists "
                             "(default <timing root>/%s)" % STOP_FILENAME)
    parser.add_argument("--expected-fixtures", default=None,
                        help="shard directory passed to the cross-arm audit "
                             "(default: the registered timing_1b.fixture_dir)")
    parser.add_argument("--poll-seconds", type=float,
                        default=DEFAULT_POLL_SECONDS,
                        help="resident-memory sampling interval (default 2 s)")
    parser.add_argument("--force", action="store_true",
                        help="re-run groups whose output dirs already hold "
                             "eval.json and arm_provenance.json")
    parser.add_argument("--skip-audit", action="store_true",
                        help="do not run audit_cross_arm.py afterwards")
    parser.add_argument("--audit-only", action="store_true",
                        help="launch nothing; just re-run the cross-arm audit "
                             "for every planned group and record the result")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the rendered commands and exit")
    return parser


def main(argv=None, *, launch: Callable[..., List[RunResult]] = launch_group,
         audit: Callable[..., dict] = run_audit) -> int:
    args = build_parser().parse_args(argv)
    arms = list(args.arms)

    try:
        config = load_config(args.config)
        check_candidates(config, arms, args.candidates)
        base_seeds = (list(registered_base_seeds(config, arms))
                      if args.base_seeds is None else list(args.base_seeds))
        check_base_seeds(config, arms, base_seeds)
        root = timing_root(config, arms)
        fixture_dir = args.expected_fixtures or shard_fixture_dir(config, arms)
        expected_count = expected_fixture_count(config, arms)
        groups = plan_groups(config, arms, args.candidates, base_seeds)
    except (KeyError, ValueError) as exc:
        sys.stderr.write("refused: %s\n" % exc)
        return EXIT_REFUSED

    record_path = Path(args.record) if args.record else root / RECORD_FILENAME
    stop_file = Path(args.stop_file) if args.stop_file else root / STOP_FILENAME
    audit_dir = root / AUDIT_DIRNAME

    counted = (len(sorted(Path(fixture_dir).glob("*.json")))
               if Path(fixture_dir).is_dir() else None)
    if counted is None:
        print("WARNING: shard directory %s does not exist yet" % fixture_dir)
    elif expected_count is not None and counted != expected_count:
        print("WARNING: shard %s holds %d fixtures, the config expects %d "
              "(re-pin with pin_stage1.py)"
              % (fixture_dir, counted, expected_count))

    if args.dry_run:
        print("timing root:      %s" % root)
        print("shard fixtures:   %s (%s json)"
              % (fixture_dir, "absent" if counted is None else counted))
        print("timing record:    %s" % record_path)
        print("stop file:        %s" % stop_file)
        print("groups:           %d (%d arms each)"
              % (len(groups), len(arms)))
        for group in groups:
            print("\n=== group %s ===" % group.key)
            for spec in group.specs:
                state = ("complete (would skip)"
                         if is_complete(spec.output_dir) and not args.force
                         else "would run")
                print("  [%s] %s" % (spec.arm, state))
                print("    output-dir: %s" % spec.output_dir)
                print("    log:        %s" % spec.log_path)
                print("    $ %s" % spec.command)
        return EXIT_OK

    record = load_record(record_path)
    record.update({
        "contract": RECORD_CONTRACT,
        "config_path": str(args.config),
        "config_sha256": sha256_file(args.config),
        "shard_fixture_dir": str(fixture_dir),
        "expected_fixture_count": expected_count,
        "shard_fixture_count": counted,
        "arms": arms,
        "base_seeds": base_seeds,
        "candidates": sorted(int(value) for value in args.candidates),
        "updated_at": _utc_now(),
    })
    record.setdefault("runs", {})
    record.setdefault("audits", {})
    write_record(record_path, record)

    failures = 0
    stopped = False
    for group in groups:
        if stop_requested(stop_file):
            print("STOP file present (%s): stopping before group %s"
                  % (stop_file, group.key))
            record["stopped_at"] = _utc_now()
            record["stopped_before_group"] = group.key
            stopped = True
            break

        pending = ([] if args.audit_only else
                   [spec for spec in group.specs
                    if args.force or not is_complete(spec.output_dir)])
        skipped = ([] if args.audit_only else
                   [spec for spec in group.specs if spec not in pending])
        if args.audit_only:
            print("\n=== group %s: audit only ===" % group.key)
        else:
            print("\n=== group %s: %d to run, %d complete ==="
                  % (group.key, len(pending), len(group.specs) - len(pending)))
        for spec in skipped:
            print("  [%s] skip: %s already has %s"
                  % (spec.arm, spec.output_dir,
                     " + ".join(COMPLETION_ARTIFACTS)))
            entry = record["runs"].get(spec.key)
            if entry is None:
                record["runs"][spec.key] = {
                    "arm": spec.arm, "n_sims": spec.n_sims,
                    "base_seed": spec.base_seed, "command": spec.command,
                    "output_dir": str(spec.output_dir),
                    "log_path": str(spec.log_path),
                    "skipped": True,
                    "total_simulation_seconds":
                        read_total_simulation_seconds(spec.log_path),
                    "noted_at": _utc_now(),
                }

        if pending:
            for spec in pending:
                print("  [%s] launch: %s" % (spec.arm, spec.output_dir))
            results = launch(pending, poll_seconds=args.poll_seconds)
            for result in results:
                key = "%s/n%d/seed%d" % (result.arm, result.n_sims,
                                         result.base_seed)
                merge_run(record, key, result.as_record())
                status = "ok" if result.exit_code == 0 else "FAILED"
                print("  [%s] %s in %.1fs (exit %s, sim %s s, peak RSS %s MB)"
                      % (result.arm, status, result.wall_seconds,
                         result.exit_code,
                         result.total_simulation_seconds,
                         "n/a" if result.peak_rss_bytes is None
                         else round(result.peak_rss_bytes / (1024 * 1024), 1)))
                if result.exit_code != 0:
                    failures += 1
            write_record(record_path, record)

        if not args.skip_audit:
            outcome = audit(group, fixture_dir, audit_dir)
            record["audits"][group.key] = outcome
            print("  [audit] %s%s" % (outcome["status"],
                                      "" if outcome["status"] != "skipped"
                                      else ": " + outcome.get("reason", "")))
            if outcome["status"] == "fail":
                failures += 1
        record["updated_at"] = _utc_now()
        write_record(record_path, record)

    record["updated_at"] = _utc_now()
    write_record(record_path, record)
    print("\ntiming record: %s" % record_path)
    if stopped:
        return EXIT_STOPPED
    return EXIT_RUN_FAILED if failures else EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
