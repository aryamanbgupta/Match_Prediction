"""Acceptance tests for the sequence-track queue runner.

Covers stage 0 D2 checks 2.1, 2.3, 2.4, 2.5, 2.7, 2.8, 2.9, 2.10, 2.11 and the
resume-after-STOP behaviour of 2.13, using dummy jobs of a few seconds. No
repository artifact is touched: every queue, config and output directory lives
under pytest's tmp_path.

Run: uv run --no-sync pytest -q scripts/tests/test_run_queue.py
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "research" / "sequence_track" / "run_queue.sh"
QUEUE_LIB = REPO_ROOT / "scripts" / "sequence_track" / "queue_lib.py"
EXAMPLE_QUEUE = REPO_ROOT / "research" / "sequence_track" / "queue.yaml"

# Generous per-run ceiling; the slowest test here is the timeout job (~7 s).
RUN_TIMEOUT_SECONDS = 120


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def write_queue(
    tmp_path: Path,
    jobs: List[Dict[str, Any]],
    *,
    memory_floor_gb: float = 0.001,
    memory_cap_gb: float = 64,
    poll_seconds: float = 1,
    stop_file: Optional[Path] = None,
    name: str = "queue.yaml",
    omit_defaults: bool = False,
) -> Path:
    document: Dict[str, Any] = {"jobs": jobs}
    if not omit_defaults:
        document["defaults"] = {
            "memory_floor_gb": memory_floor_gb,
            "memory_cap_gb": memory_cap_gb,
            "poll_seconds": poll_seconds,
            "stop_file": str(stop_file or (tmp_path / "STOP")),
        }
    queue_path = tmp_path / name
    queue_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return queue_path


def make_job(
    tmp_path: Path,
    job_id: str,
    command: str,
    *,
    expected_hours: float = 0.01,
    machine: str = "laptop",
    config_text: str = "config-v1\n",
    config_name: Optional[str] = None,
) -> Dict[str, Any]:
    config_path = tmp_path / (config_name or ("%s.cfg" % job_id))
    if not config_path.exists():
        config_path.write_text(config_text, encoding="utf-8")
    return {
        "id": job_id,
        "config": str(config_path),
        "command": command,
        "output_dir": str(tmp_path / ("out_%s" % job_id)),
        "expected_hours": expected_hours,
        "machine": machine,
    }


def run_queue(
    queue_path: Path, *extra: str, env: Optional[Dict[str, str]] = None
) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(RUNNER), "--queue", str(queue_path), *extra],
        capture_output=True,
        text=True,
        timeout=RUN_TIMEOUT_SECONDS,
        cwd=str(REPO_ROOT),
        env=dict(os.environ, **(env or {})),
    )


def fake_vm_stat(tmp_path: Path, starve_when_exists: Optional[Path] = None) -> Path:
    """A stand-in for vm_stat: plenty of free memory, or almost none.

    With `starve_when_exists`, the reading flips to starved as soon as that
    file appears, which is how a test makes memory "run out" between attempts.
    Page size 16384: 524288 free pages is 8 GiB, 3277 pages is 0.05 GiB.
    """
    script = tmp_path / "fake_vm_stat.sh"
    trigger = str(starve_when_exists) if starve_when_exists else ""
    script.write_text(
        "#!/bin/bash\n"
        'free=524288\n'
        'if [ -n "%s" ] && [ -e "%s" ]; then free=3277; fi\n'
        'echo "Mach Virtual Memory Statistics: (page size of 16384 bytes)"\n'
        'echo "Pages free:                               ${free}."\n'
        'echo "Pages active:                             100."\n'
        'echo "Pages inactive:                           0."\n'
        'echo "Pages speculative:                        0."\n' % (trigger, trigger),
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script


def survivors(token: str) -> List[str]:
    """Command lines of any process still carrying a job's unique token."""
    found = subprocess.run(
        ["pgrep", "-fl", token], capture_output=True, text=True, timeout=30
    ).stdout.strip()
    return [line for line in found.splitlines() if line.strip()]


def reap(token: str) -> None:
    """Last-resort cleanup so a failing test cannot leak a sleeping process."""
    subprocess.run(["pkill", "-9", "-f", token], capture_output=True, text=True, timeout=30)


def wait_for_no_survivors(token: str, seconds: float = 5.0) -> List[str]:
    """Give SIGKILL a moment to land, then report whatever is still running."""
    deadline = time.monotonic() + seconds
    remaining = survivors(token)
    while remaining and time.monotonic() < deadline:
        time.sleep(0.2)
        remaining = survivors(token)
    return remaining


def marker(tmp_path: Path, job_id: str, name: str) -> Path:
    return tmp_path / ("out_%s" % job_id) / name


def read_marker_field(path: Path, field: str) -> Optional[str]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("%s:" % field):
            return line.split(":", 1)[1].strip()
    return None


# ---------------------------------------------------------------------------
# 2.1 schema validation
# ---------------------------------------------------------------------------


def test_missing_job_field_exits_non_zero_before_launching(tmp_path: Path) -> None:
    """(a) A job missing a required field aborts the queue before any launch."""
    good = make_job(tmp_path, "good", "touch %s" % (tmp_path / "ran.txt"))
    broken = make_job(tmp_path, "broken", "echo hi")
    del broken["command"]
    queue_path = write_queue(tmp_path, [good, broken])

    result = run_queue(queue_path)

    assert result.returncode != 0, result.stdout
    assert "command" in result.stderr
    assert not (tmp_path / "ran.txt").exists()
    assert not (tmp_path / "out_good").exists()


def test_duplicate_job_id_exits_non_zero(tmp_path: Path) -> None:
    """(a) Duplicate ids abort the queue before any launch."""
    first = make_job(tmp_path, "twin", "touch %s" % (tmp_path / "ran.txt"))
    second = dict(first)
    second["output_dir"] = str(tmp_path / "out_twin_b")
    queue_path = write_queue(tmp_path, [first, second])

    result = run_queue(queue_path)

    assert result.returncode != 0
    assert "duplicate id" in result.stderr
    assert not (tmp_path / "ran.txt").exists()


def test_missing_defaults_exits_non_zero(tmp_path: Path) -> None:
    """(a) A queue with no defaults block never launches."""
    job = make_job(tmp_path, "solo", "touch %s" % (tmp_path / "ran.txt"))
    queue_path = write_queue(tmp_path, [job], omit_defaults=True)

    result = run_queue(queue_path)

    assert result.returncode != 0
    assert "defaults" in result.stderr
    assert not (tmp_path / "ran.txt").exists()


# ---------------------------------------------------------------------------
# 2.3 skip-if-complete / re-run on config change
# ---------------------------------------------------------------------------


def test_skip_if_complete_and_rerun_on_config_change(tmp_path: Path) -> None:
    """(b) COMPLETE with a matching config hash skips; a changed config re-runs."""
    counter = tmp_path / "counter.txt"
    job = make_job(tmp_path, "counted", "printf x >> %s" % counter)
    queue_path = write_queue(tmp_path, [job])

    first = run_queue(queue_path)
    assert first.returncode == 0, first.stdout + first.stderr
    assert counter.read_text() == "x"
    complete = marker(tmp_path, "counted", "COMPLETE")
    assert complete.is_file()
    first_sha = read_marker_field(complete, "config_sha256")
    assert first_sha and len(first_sha) == 64

    second = run_queue(queue_path)
    assert second.returncode == 0
    assert counter.read_text() == "x", "job re-ran despite an unchanged config"
    assert "COMPLETE with matching config_sha256" in second.stdout

    Path(job["config"]).write_text("config-v2\n", encoding="utf-8")
    third = run_queue(queue_path)
    assert third.returncode == 0
    assert counter.read_text() == "xx", "changed config did not re-run the job"
    assert read_marker_field(complete, "config_sha256") not in (None, first_sha)


# ---------------------------------------------------------------------------
# 2.4 + 2.7 timeout, retry once
# ---------------------------------------------------------------------------


def test_timeout_writes_marker_and_retries_once(tmp_path: Path) -> None:
    """(c) expected_hours 0.0005 -> 3 s alarm; the job is killed and retried once."""
    job = make_job(tmp_path, "sleeper", "sleep 30", expected_hours=0.0005)
    queue_path = write_queue(tmp_path, [job])

    result = run_queue(queue_path)

    assert result.returncode == 0, result.stdout + result.stderr
    timeout_marker = marker(tmp_path, "sleeper", "TIMEOUT")
    assert timeout_marker.is_file()
    assert read_marker_field(timeout_marker, "timeout_seconds") == "3"
    assert marker(tmp_path, "sleeper", "FAILED").is_file()
    assert not marker(tmp_path, "sleeper", "COMPLETE").exists()

    log_text = marker(tmp_path, "sleeper", "run.log").read_text(encoding="utf-8")
    assert log_text.count("attempt 1 start") == 1
    assert log_text.count("attempt 2 start") == 1
    assert log_text.count("exit=142") == 2, log_text


def test_timeout_kills_a_grandchild_job(tmp_path: Path) -> None:
    """(c) The real job shape -- uv spawning python -- is killed on time."""
    token = "SEQQPROBE_GRANDCHILD_TIMEOUT"
    command = (
        "bash -c 'uv run --no-sync python -c \"import time; %s = 1; time.sleep(60)\"'" % token
    )
    job = make_job(tmp_path, "gcsleep", command, expected_hours=0.0005)
    queue_path = write_queue(tmp_path, [job], poll_seconds=1)

    try:
        started = time.monotonic()
        result = run_queue(queue_path)
        elapsed = time.monotonic() - started

        assert result.returncode == 0, result.stdout + result.stderr
        # Two attempts of a 3 s timeout; a job that outran the timeout would
        # sit here for 60 s per attempt.
        assert elapsed < 30, "both attempts took %.1f s" % elapsed
        assert marker(tmp_path, "gcsleep", "TIMEOUT").is_file()
        assert marker(tmp_path, "gcsleep", "FAILED").is_file()
        assert not marker(tmp_path, "gcsleep", "COMPLETE").exists()
        assert wait_for_no_survivors(token) == []
    finally:
        reap(token)


def test_timeout_kills_a_child_that_outlives_the_alarm(tmp_path: Path) -> None:
    """(c) The alarm only reaches the process it exec'd; the watchdog finishes the job.

    A backgrounded child survives SIGALRM and keeps the log pipe open, so
    without the watchdog deadline the runner blocks until the child ends on
    its own -- 60 s here, hours for a real arm.
    """
    token = "SEQQPROBE_ORPHAN_TIMEOUT"
    command = (
        "uv run --no-sync python -c \"import time; %s = 1; time.sleep(60)\" & wait" % token
    )
    job = make_job(tmp_path, "orphan", command, expected_hours=0.0005)
    queue_path = write_queue(tmp_path, [job], poll_seconds=1)

    try:
        started = time.monotonic()
        result = run_queue(queue_path)
        elapsed = time.monotonic() - started

        assert result.returncode == 0, result.stdout + result.stderr
        assert elapsed < 30, "both attempts took %.1f s" % elapsed
        timeout_marker = marker(tmp_path, "orphan", "TIMEOUT")
        assert timeout_marker.is_file()
        assert read_marker_field(timeout_marker, "killed_by") == "watchdog"
        assert read_marker_field(timeout_marker, "timeout_seconds") == "3"
        assert marker(tmp_path, "orphan", "FAILED").is_file()
        assert "killed by the watchdog" in result.stdout

        log_text = marker(tmp_path, "orphan", "run.log").read_text(encoding="utf-8")
        assert log_text.count("past the 3s timeout") == 2, log_text
        assert wait_for_no_survivors(token) == []
    finally:
        reap(token)


# ---------------------------------------------------------------------------
# 2.5 + 2.13 STOP file and resume
# ---------------------------------------------------------------------------


def test_stop_file_present_launches_nothing(tmp_path: Path) -> None:
    """(d) A pre-existing stop file exits 0 without launching."""
    stop_file = tmp_path / "STOP"
    stop_file.write_text("halt\n", encoding="utf-8")
    ran = tmp_path / "ran.txt"
    job = make_job(tmp_path, "never", "touch %s" % ran)
    queue_path = write_queue(tmp_path, [job], stop_file=stop_file)

    result = run_queue(queue_path)

    assert result.returncode == 0
    assert not ran.exists()
    assert not (tmp_path / "out_never").exists()


def test_stop_created_by_first_job_halts_queue_then_resumes(tmp_path: Path) -> None:
    """(d) A job that writes the stop file stops the queue; resume runs the rest."""
    stop_file = tmp_path / "STOP"
    first_ran = tmp_path / "first.txt"
    second_ran = tmp_path / "second.txt"
    jobs = [
        make_job(tmp_path, "first", "printf x >> %s; touch %s" % (first_ran, stop_file)),
        make_job(tmp_path, "second", "printf y >> %s" % second_ran),
    ]
    queue_path = write_queue(tmp_path, jobs, stop_file=stop_file)

    first_pass = run_queue(queue_path)
    assert first_pass.returncode == 0, first_pass.stdout + first_pass.stderr
    assert first_ran.read_text() == "x"
    assert marker(tmp_path, "first", "COMPLETE").is_file()
    assert not second_ran.exists(), "second job launched after the stop file appeared"
    assert not marker(tmp_path, "second", "COMPLETE").exists()

    stop_file.unlink()
    second_pass = run_queue(queue_path)
    assert second_pass.returncode == 0, second_pass.stdout + second_pass.stderr
    assert first_ran.read_text() == "x", "completed job re-ran on resume"
    assert second_ran.read_text() == "y"
    assert marker(tmp_path, "second", "COMPLETE").is_file()


# ---------------------------------------------------------------------------
# 2.7 failure handling
# ---------------------------------------------------------------------------


def test_non_zero_exit_retries_once_then_fails_and_queue_continues(tmp_path: Path) -> None:
    """(e) Two failures write FAILED; the next job still runs."""
    after = tmp_path / "after.txt"
    jobs = [
        make_job(tmp_path, "flaky", "echo boom >&2; exit 3"),
        make_job(tmp_path, "after", "touch %s" % after),
    ]
    queue_path = write_queue(tmp_path, jobs)

    result = run_queue(queue_path)

    assert result.returncode == 0, result.stdout + result.stderr
    failed = marker(tmp_path, "flaky", "FAILED")
    assert failed.is_file()
    assert read_marker_field(failed, "exit_code") == "3"
    assert read_marker_field(failed, "attempts") == "2"
    log_text = marker(tmp_path, "flaky", "run.log").read_text(encoding="utf-8")
    assert log_text.count("exit=3") == 2, log_text
    # The job's own stderr is captured once per attempt (the "command:" stamp
    # also echoes the command text, so match whole lines).
    assert log_text.splitlines().count("boom") == 2, log_text

    assert after.exists(), "queue aborted instead of continuing after a FAILED job"
    assert marker(tmp_path, "after", "COMPLETE").is_file()


# ---------------------------------------------------------------------------
# 2.8 memory floor
# ---------------------------------------------------------------------------


def test_memory_floor_refuses_to_launch(tmp_path: Path) -> None:
    """(f) An unreachable floor writes REFUSED_MEMORY and launches nothing."""
    ran = tmp_path / "ran.txt"
    jobs = [
        make_job(tmp_path, "hungry", "touch %s" % ran),
        make_job(tmp_path, "next", "touch %s" % (tmp_path / "next.txt")),
    ]
    queue_path = write_queue(tmp_path, jobs, memory_floor_gb=1_000_000)

    result = run_queue(queue_path)

    assert result.returncode == 0, result.stdout + result.stderr
    refused = marker(tmp_path, "hungry", "REFUSED_MEMORY")
    assert refused.is_file()
    assert "available" in read_marker_field(refused, "reading")
    assert not ran.exists()
    assert not marker(tmp_path, "hungry", "COMPLETE").exists()
    # The refusal is per job and does not abort the queue.
    assert marker(tmp_path, "next", "REFUSED_MEMORY").is_file()
    assert "BELOW" in result.stdout
    # A refusal is not a failed run.
    assert not marker(tmp_path, "hungry", "FAILED").exists()


def test_memory_floor_is_rechecked_before_the_retry(tmp_path: Path) -> None:
    """(a) The floor is read again before attempt 2, which is refused."""
    starved = tmp_path / "memory_gone"
    vm_stat = fake_vm_stat(tmp_path, starve_when_exists=starved)
    after = tmp_path / "after.txt"
    jobs = [
        # The first attempt fails and, like a real memory hog, leaves the
        # machine short: the retry must see the new reading, not the old one.
        make_job(tmp_path, "greedy", "touch %s; exit 3" % starved),
        make_job(tmp_path, "after", "touch %s" % after),
    ]
    queue_path = write_queue(tmp_path, jobs, memory_floor_gb=4)

    result = run_queue(queue_path, env={"QUEUE_VM_STAT_CMD": str(vm_stat)})

    assert result.returncode == 0, result.stdout + result.stderr
    refused = marker(tmp_path, "greedy", "REFUSED_MEMORY")
    assert refused.is_file(), "the retry launched without a fresh memory reading"
    assert read_marker_field(refused, "attempt") == "2"
    assert "below the 4 GiB floor" in (read_marker_field(refused, "refusal") or "")
    # Refused, not failed: no FAILED marker, and the queue keeps going -- the
    # machine is still starved, so the next job reaches its own memory check
    # and is refused there rather than the queue aborting.
    assert not marker(tmp_path, "greedy", "FAILED").exists()
    assert marker(tmp_path, "after", "REFUSED_MEMORY").is_file()
    assert not after.exists()

    log_text = marker(tmp_path, "greedy", "run.log").read_text(encoding="utf-8")
    assert log_text.count("attempt 1 start") == 1
    assert "attempt 2 start" not in log_text, "attempt 2 ran despite the floor"
    assert log_text.count("memory check") == 2, log_text


def test_failed_memory_reading_refuses_the_launch(tmp_path: Path) -> None:
    """(b) A reading that cannot be parsed fails closed instead of launching."""
    ran = tmp_path / "ran.txt"
    job = make_job(tmp_path, "unmeasured", "touch %s" % ran)
    queue_path = write_queue(tmp_path, [job], memory_floor_gb=4)

    result = run_queue(
        queue_path, env={"QUEUE_VM_STAT_CMD": "/bin/echo definitely-not-vm-stat-output"}
    )

    assert result.returncode == 0, result.stdout + result.stderr
    refused = marker(tmp_path, "unmeasured", "REFUSED_MEMORY")
    assert refused.is_file()
    assert "reading failed" in (read_marker_field(refused, "refusal") or "")
    assert "printed no page counts" in (read_marker_field(refused, "reading") or "")
    assert not ran.exists(), "the job launched on an unreadable memory reading"
    assert not marker(tmp_path, "unmeasured", "COMPLETE").exists()
    assert not marker(tmp_path, "unmeasured", "FAILED").exists()


def test_missing_vm_stat_command_refuses_the_launch(tmp_path: Path) -> None:
    """(b) A reading command that cannot even run also fails closed."""
    ran = tmp_path / "ran.txt"
    job = make_job(tmp_path, "noreader", "touch %s" % ran)
    queue_path = write_queue(tmp_path, [job], memory_floor_gb=4)

    result = run_queue(
        queue_path, env={"QUEUE_VM_STAT_CMD": str(tmp_path / "no_such_vm_stat")}
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert marker(tmp_path, "noreader", "REFUSED_MEMORY").is_file()
    assert not ran.exists()


# ---------------------------------------------------------------------------
# 2.9 memory watchdog
# ---------------------------------------------------------------------------


def test_memory_watchdog_kills_tree_and_writes_marker(tmp_path: Path) -> None:
    """(g) A 300 MB allocator under a 0.1 GiB cap is killed with KILLED_MEMORY."""
    allocator = (
        'uv run --no-sync python -c "import time; b = bytearray(300 * 1024 * 1024); '
        'time.sleep(20); print(len(b))"'
    )
    job = make_job(tmp_path, "hog", allocator, expected_hours=0.01)
    queue_path = write_queue(tmp_path, [job], memory_cap_gb=0.1, poll_seconds=1)

    result = run_queue(queue_path)

    assert result.returncode == 0, result.stdout + result.stderr
    killed = marker(tmp_path, "hog", "KILLED_MEMORY")
    assert killed.is_file()
    observed = read_marker_field(killed, "observed_rss_bytes")
    assert observed is not None and int(observed) > 0.1 * 1024 ** 3
    assert read_marker_field(killed, "cap_gb") == "0.100"
    assert not marker(tmp_path, "hog", "COMPLETE").exists()
    # A memory kill is a failure like any other: retried once, then FAILED.
    assert marker(tmp_path, "hog", "FAILED").is_file()
    log_text = marker(tmp_path, "hog", "run.log").read_text(encoding="utf-8")
    assert log_text.count("watchdog hog") == 2, log_text


def test_memory_watchdog_kills_a_grandchild_allocator(tmp_path: Path) -> None:
    """(g) The kill is by process group, so it reaches allocators two levels down."""
    token = "SEQQPROBE_GRANDCHILD_MEMORY"
    command = (
        "bash -c 'uv run --no-sync python -c \"import time; %s = bytearray(300 * 1024 * 1024); "
        'time.sleep(20)"\'' % token
    )
    job = make_job(tmp_path, "gchog", command, expected_hours=0.01)
    queue_path = write_queue(tmp_path, [job], memory_cap_gb=0.1, poll_seconds=1)

    try:
        started = time.monotonic()
        result = run_queue(queue_path)
        elapsed = time.monotonic() - started

        assert result.returncode == 0, result.stdout + result.stderr
        assert elapsed < 30, "both attempts took %.1f s" % elapsed
        assert marker(tmp_path, "gchog", "KILLED_MEMORY").is_file()
        assert not marker(tmp_path, "gchog", "COMPLETE").exists()
        assert wait_for_no_survivors(token) == []
    finally:
        reap(token)


# ---------------------------------------------------------------------------
# 2.2 / 2.11 machine tag and dry run
# ---------------------------------------------------------------------------


def test_machine_tag_filters_jobs(tmp_path: Path) -> None:
    """2.2 A job tagged for another machine is skipped with a log line."""
    mini_ran = tmp_path / "mini.txt"
    laptop_ran = tmp_path / "laptop.txt"
    jobs = [
        make_job(tmp_path, "mini_job", "touch %s" % mini_ran, machine="mini"),
        make_job(tmp_path, "laptop_job", "touch %s" % laptop_ran, machine="laptop"),
    ]
    queue_path = write_queue(tmp_path, jobs)

    result = run_queue(queue_path)

    assert result.returncode == 0, result.stdout + result.stderr
    assert not mini_ran.exists()
    assert laptop_ran.exists()
    assert "does not match 'laptop'" in result.stdout


def test_dry_run_launches_nothing(tmp_path: Path) -> None:
    """2.11 --dry-run prints decisions and the memory reading, launching nothing."""
    ran = tmp_path / "ran.txt"
    job = make_job(tmp_path, "planned", "touch %s" % ran)
    queue_path = write_queue(tmp_path, [job])

    result = run_queue(queue_path, "--dry-run")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "decision=run" in result.stdout
    assert "memory: available" in result.stdout
    assert "nothing launched" in result.stdout
    assert not ran.exists()
    assert not (tmp_path / "out_planned").exists()


def test_dry_run_reports_skip_for_completed_job(tmp_path: Path) -> None:
    """2.11 The dry run's decision matches the skip-if-complete rule."""
    counter = tmp_path / "counter.txt"
    job = make_job(tmp_path, "done", "printf x >> %s" % counter)
    queue_path = write_queue(tmp_path, [job])

    assert run_queue(queue_path).returncode == 0
    result = run_queue(queue_path, "--dry-run")

    assert "decision=skip" in result.stdout
    assert counter.read_text() == "x"


def test_example_queue_dry_runs(tmp_path: Path) -> None:
    """The shipped example queue validates and dry-runs (2.1, 2.11)."""
    result = subprocess.run(
        [str(RUNNER), "--dry-run"],
        capture_output=True,
        text=True,
        timeout=RUN_TIMEOUT_SECONDS,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "stage1_placeholder" in result.stdout
    assert "command=echo placeholder" in result.stdout


# ---------------------------------------------------------------------------
# 2.10 no side channels
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", [RUNNER, QUEUE_LIB, EXAMPLE_QUEUE])
def test_no_side_channel_invocations(path: Path) -> None:
    """2.10 No model, source-control, or verdict-logging invocation anywhere."""
    text = path.read_text(encoding="utf-8")
    for token in ("claude", "codex", "log_verdict"):
        assert token not in text, "%s mentions %s" % (path.name, token)
    assert re.search(r"\bgit\b", text) is None, "%s invokes a version-control command" % path.name
