"""Tests for stage-1 step 1d: the sharded full-run driver (D11 check 11.3).

Synthetic only. No simulation is launched: `run_full_1d.main` takes its
process launcher and its per-shard cross-arm audit as injectable callables,
and the one test that uses the REAL launcher runs `/bin/echo`.

What has to hold before 40 real jobs are started:

* every job is a command the config pinned, never one this script composed;
* the slow T1 arm's shards are queued first, so the pool starts them before
  any XGBoost shard;
* a finished job is skipped, not re-run, and the STOP file stops LAUNCHING
  without killing anything already running;
* the record carries wall, exit, the evaluator's own timing line and peak
  tree RSS — and no metric, ever.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
import yaml

from sequence_track import run_full_1d
from sequence_track.run_full_1d import (EXIT_OK, EXIT_REFUSED, EXIT_RUN_FAILED,
                                        EXIT_STOPPED, JobResult)

ARMS = ("C", "B", "A50", "A")
SHARDS = (0, 1, 2)
N_SIMS = 1600


# --------------------------------------------------------------------------
# synthetic config
# --------------------------------------------------------------------------


def write_config(tmp_path: Path, *, arms=ARMS, shards=SHARDS, n_sims=N_SIMS,
                 command="/bin/echo", mutate=None) -> Path:
    """A config with the same shape `pin_stage1` writes for `full_run`."""
    root = tmp_path / "full"
    payload = {"arms": {}}
    for arm in arms:
        entries = []
        for index in shards:
            fixtures = tmp_path / "shards" / str(index) / "fixtures"
            fixtures.mkdir(parents=True, exist_ok=True)
            (fixtures / f"150000{index}.json").write_text("{}")
            output_dir = root / arm / ("shard%d" % index)
            entries.append({
                "index": index,
                "fixture_dir": str(fixtures),
                "fixture_dir_md5": "0" * 32,
                "fixture_count": 1,
                "output_dir": str(output_dir),
                "command": ("%s arm=%s shard=%d --fixture-dir %s "
                            "--output-dir %s"
                            % (command, arm, index, fixtures, output_dir)),
            })
        payload["arms"][arm] = {
            "full_run": {
                "fixture_dir": "data/polymarket_test_v2",
                "fixture_count": 255,
                "n_sims": n_sims,
                "base_seeds": [20260910],
                "output_dir": str(root / arm),
                "shard_count": len(shards),
                "shards": entries,
            }
        }
    if mutate is not None:
        mutate(payload)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


class Recorder:
    """Stands in for the subprocess pool; records what would have run."""

    def __init__(self, stop_file: Path = None, exit_code: int = 0):
        self.launched = []
        self.concurrency = None
        self.stop_file = stop_file
        self.exit_code = exit_code

    def __call__(self, specs, *, concurrency=10, poll_seconds=2.0,
                 stop_file=None, cwd=None, on_result=None):
        self.concurrency = concurrency
        self.launched = [spec.key for spec in specs]
        if self.stop_file is not None:
            self.stop_file.parent.mkdir(parents=True, exist_ok=True)
            self.stop_file.write_text("stop\n", encoding="utf-8")
        results = [
            JobResult(arm=spec.arm, shard=spec.shard, command=spec.command,
                      output_dir=str(spec.output_dir),
                      log_path=str(spec.log_path),
                      started_at="2026-09-11T00:00:00+00:00",
                      ended_at="2026-09-11T02:00:00+00:00",
                      wall_seconds=7200.0, exit_code=self.exit_code,
                      total_simulation_seconds=6900.5,
                      peak_rss_bytes=512 * 1024 * 1024)
            for spec in specs
        ]
        for result in results:
            if on_result is not None:
                on_result(result)
        return results, self.stop_file is not None


def no_audit(specs, shard, fixture_dir, audit_dir):
    return {"status": "pass", "exit_code": 0}


def mark_complete(config_path: Path, arm: str, shard: int) -> Path:
    config = run_full_1d.load_config(config_path)
    spec = run_full_1d.build_job_spec(config, arm, shard)
    spec.output_dir.mkdir(parents=True, exist_ok=True)
    for name in run_full_1d.COMPLETION_ARTIFACTS:
        (spec.output_dir / name).write_text("{}", encoding="utf-8")
    return spec.output_dir


def read_record(tmp_path: Path) -> dict:
    path = tmp_path / "full" / run_full_1d.RECORD_FILENAME
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# the plan: jobs come from the config, in the registered order
# --------------------------------------------------------------------------


def test_every_job_is_a_command_the_config_pinned(tmp_path):
    config = run_full_1d.load_config(write_config(tmp_path))
    jobs = run_full_1d.plan_jobs(config, ARMS, SHARDS)
    assert len(jobs) == len(ARMS) * len(SHARDS)
    for spec in jobs:
        entry = run_full_1d.shard_entries(config, spec.arm)[spec.shard]
        assert spec.command == entry["command"]
        assert str(spec.output_dir) == entry["output_dir"]
        assert spec.fixture_dir == entry["fixture_dir"]
        assert spec.log_path == spec.output_dir / "run.log"


def test_the_t1_arm_is_queued_first(tmp_path):
    """C's shards must all be queued before the first XGBoost shard."""
    config = run_full_1d.load_config(write_config(tmp_path))
    keys = [spec.key for spec in run_full_1d.plan_jobs(config, ARMS, SHARDS)]
    assert keys[:len(SHARDS)] == ["C/shard%d" % index for index in SHARDS]
    arms_in_order = []
    for key in keys:
        arm = key.split("/")[0]
        if arm not in arms_in_order:
            arms_in_order.append(arm)
    assert arms_in_order == list(ARMS)
    assert run_full_1d.DEFAULT_ARMS == ("C", "B", "A50", "A")


def test_the_full_root_is_read_off_the_output_dirs(tmp_path):
    config = run_full_1d.load_config(write_config(tmp_path))
    assert run_full_1d.full_root(config, ARMS) == tmp_path / "full"


def test_forty_jobs_on_the_registered_shape():
    """4 arms x 10 shards is the registered 1d plan."""
    assert run_full_1d.DEFAULT_CONCURRENCY == 10
    assert len(run_full_1d.DEFAULT_ARMS) == 4


# --------------------------------------------------------------------------
# refusals
# --------------------------------------------------------------------------


def test_a_placeholder_n_sims_is_refused(tmp_path, capsys):
    path = write_config(tmp_path, n_sims=run_full_1d.N_SIMS_PLACEHOLDER)
    assert run_full_1d.main(["--config", str(path), "--dry-run"],
                            launch=Recorder(), audit=no_audit) == EXIT_REFUSED
    assert "not a launchable count" in capsys.readouterr().err


def test_an_unregistered_shard_is_refused(tmp_path, capsys):
    path = write_config(tmp_path)
    assert run_full_1d.main(
        ["--config", str(path), "--shards", "9", "--dry-run"],
        launch=Recorder(), audit=no_audit) == EXIT_REFUSED
    assert "not registered" in capsys.readouterr().err


def test_arms_that_shard_differently_are_refused(tmp_path, capsys):
    def drop(payload):
        payload["arms"]["A"]["full_run"]["shards"].pop()

    path = write_config(tmp_path, mutate=drop)
    assert run_full_1d.main(["--config", str(path), "--dry-run"],
                            launch=Recorder(), audit=no_audit) == EXIT_REFUSED
    assert "different shard sets" in capsys.readouterr().err


def test_arms_that_disagree_on_a_shard_fixture_dir_are_refused(tmp_path,
                                                               capsys):
    def move(payload):
        payload["arms"]["A"]["full_run"]["shards"][0]["fixture_dir"] = "/tmp/x"

    path = write_config(tmp_path, mutate=move)
    assert run_full_1d.main(["--config", str(path), "--dry-run"],
                            launch=Recorder(), audit=no_audit) == EXIT_REFUSED
    assert "fixture dir of shard 0" in capsys.readouterr().err


def test_an_unsharded_config_is_refused(tmp_path, capsys):
    def unshard(payload):
        for arm in payload["arms"].values():
            arm["full_run"]["shards"] = []

    path = write_config(tmp_path, mutate=unshard)
    assert run_full_1d.main(["--config", str(path), "--dry-run"],
                            launch=Recorder(), audit=no_audit) == EXIT_REFUSED
    assert "registers no full_run shards" in capsys.readouterr().err


# --------------------------------------------------------------------------
# dry run
# --------------------------------------------------------------------------


def test_dry_run_prints_every_job_and_launches_nothing(tmp_path, capsys):
    path = write_config(tmp_path)
    recorder = Recorder()
    assert run_full_1d.main(["--config", str(path), "--dry-run"],
                            launch=recorder, audit=no_audit) == EXIT_OK
    out = capsys.readouterr().out
    assert recorder.launched == []
    assert "jobs:             %d" % (len(ARMS) * len(SHARDS)) in out
    assert "concurrency:      10" in out
    assert out.index("[C/shard0]") < out.index("[A/shard0]")
    for arm in ARMS:
        for shard in SHARDS:
            assert "[%s/shard%d]" % (arm, shard) in out
    assert not (tmp_path / "full" / run_full_1d.RECORD_FILENAME).exists()


def test_dry_run_marks_a_complete_job(tmp_path, capsys):
    path = write_config(tmp_path)
    mark_complete(path, "B", 1)
    run_full_1d.main(["--config", str(path), "--dry-run"],
                     launch=Recorder(), audit=no_audit)
    out = capsys.readouterr().out
    assert "[B/shard1] complete (would skip)" in out
    assert "[B/shard0] would run" in out


# --------------------------------------------------------------------------
# skip, stop, record
# --------------------------------------------------------------------------


def test_complete_jobs_are_skipped_not_relaunched(tmp_path):
    path = write_config(tmp_path)
    mark_complete(path, "C", 0)
    mark_complete(path, "A", 2)
    recorder = Recorder()
    assert run_full_1d.main(["--config", str(path)], launch=recorder,
                            audit=no_audit) == EXIT_OK
    assert "C/shard0" not in recorder.launched
    assert "A/shard2" not in recorder.launched
    assert len(recorder.launched) == len(ARMS) * len(SHARDS) - 2

    record = read_record(tmp_path)
    assert record["jobs"]["C/shard0"]["skipped"] is True
    assert record["jobs"]["C/shard1"]["skipped"] is False


def test_force_relaunches_a_complete_job(tmp_path):
    path = write_config(tmp_path)
    mark_complete(path, "C", 0)
    recorder = Recorder()
    run_full_1d.main(["--config", str(path), "--force"], launch=recorder,
                     audit=no_audit)
    assert "C/shard0" in recorder.launched


def test_the_stop_file_stops_launching(tmp_path):
    path = write_config(tmp_path)
    stop_file = tmp_path / "full" / run_full_1d.STOP_FILENAME
    recorder = Recorder(stop_file=stop_file)
    assert run_full_1d.main(["--config", str(path)], launch=recorder,
                            audit=no_audit) == EXIT_STOPPED
    record = read_record(tmp_path)
    assert record["stopped_launching"] is True
    assert record["stopped_at"]


def test_a_failing_job_is_recorded_and_exits_non_zero(tmp_path):
    path = write_config(tmp_path)
    assert run_full_1d.main(["--config", str(path)],
                            launch=Recorder(exit_code=2),
                            audit=no_audit) == EXIT_RUN_FAILED
    record = read_record(tmp_path)
    assert record["jobs"]["C/shard0"]["exit_code"] == 2


def test_the_record_carries_the_registered_identity_and_per_job_facts(
        tmp_path):
    path = write_config(tmp_path)
    run_full_1d.main(["--config", str(path)], launch=Recorder(),
                     audit=no_audit)
    record = read_record(tmp_path)

    assert record["contract"] == run_full_1d.RECORD_CONTRACT
    assert record["config_path"] == str(path)
    assert len(record["config_sha256"]) == 64
    assert record["arms"] == list(ARMS)
    assert record["shards"] == list(SHARDS)
    assert record["n_sims"] == N_SIMS
    assert record["concurrency"] == 10
    assert set(record["shard_fixture_dirs"]) == {"0", "1", "2"}
    assert len(record["jobs"]) == len(ARMS) * len(SHARDS)

    job = record["jobs"]["C/shard2"]
    assert job["arm"] == "C" and job["shard"] == 2
    assert job["wall_seconds"] == 7200.0
    assert job["exit_code"] == 0
    assert job["total_simulation_seconds"] == 6900.5
    assert job["peak_rss_bytes"] == 512 * 1024 * 1024
    assert job["peak_rss_mb"] == 512.0
    assert job["log_path"].endswith("/C/shard2/run.log")


def test_the_record_carries_no_metric(tmp_path):
    """1d must not read what it is producing: the record is timing only."""
    path = write_config(tmp_path)
    run_full_1d.main(["--config", str(path)], launch=Recorder(),
                     audit=no_audit)
    text = (tmp_path / "full" / run_full_1d.RECORD_FILENAME).read_text().lower()
    for banned in ("log_loss", "logloss", "brier", "\"roi\"", "edge",
                   "win_probability", "accuracy"):
        assert banned not in text, banned


# --------------------------------------------------------------------------
# the per-shard cross-arm audit
# --------------------------------------------------------------------------


def test_the_audit_runs_once_per_shard_over_the_four_arm_dirs(tmp_path):
    path = write_config(tmp_path)
    seen = []

    def spy(specs, shard, fixture_dir, audit_dir):
        seen.append((shard, tuple(spec.arm for spec in specs),
                     str(fixture_dir)))
        return {"status": "pass", "exit_code": 0}

    assert run_full_1d.main(["--config", str(path)], launch=Recorder(),
                            audit=spy) == EXIT_OK
    assert [row[0] for row in seen] == list(SHARDS)
    for _shard, arms, fixture_dir in seen:
        assert arms == ARMS
        assert fixture_dir.endswith("/fixtures")
    record = read_record(tmp_path)
    assert set(record["audits"]) == {"shard0", "shard1", "shard2"}
    assert record["audits"]["shard1"]["status"] == "pass"


def test_a_failing_shard_audit_exits_non_zero(tmp_path):
    path = write_config(tmp_path)

    def failing(specs, shard, fixture_dir, audit_dir):
        return {"status": "fail", "exit_code": 1}

    assert run_full_1d.main(["--config", str(path)], launch=Recorder(),
                            audit=failing) == EXIT_RUN_FAILED


def test_skip_audit_leaves_the_audits_empty(tmp_path):
    path = write_config(tmp_path)

    def explode(*_args, **_kwargs):  # pragma: no cover - must not be called
        raise AssertionError("the audit ran under --skip-audit")

    run_full_1d.main(["--config", str(path), "--skip-audit"],
                     launch=Recorder(), audit=explode)
    assert read_record(tmp_path)["audits"] == {}


def test_audit_only_launches_nothing(tmp_path):
    path = write_config(tmp_path)
    recorder = Recorder()
    assert run_full_1d.main(["--config", str(path), "--audit-only"],
                            launch=recorder, audit=no_audit) == EXIT_OK
    assert recorder.launched == []


def test_the_audit_is_skipped_while_an_arm_dir_is_incomplete(tmp_path):
    path = write_config(tmp_path)
    config = run_full_1d.load_config(path)
    specs = [run_full_1d.build_job_spec(config, arm, 0) for arm in ARMS]
    outcome = run_full_1d.run_shard_audit(
        specs, 0, str(specs[0].fixture_dir), tmp_path / "audit")
    assert outcome["status"] == "skipped"
    assert "incomplete arm dirs" in outcome["reason"]


def test_the_audit_command_is_the_cross_arm_auditor(tmp_path):
    command = run_full_1d.audit_command([Path("a"), Path("b")], "fixtures")
    assert command[:4] == ["uv", "run", "--no-sync", "python"]
    assert command[4] == "scripts/sequence_track/audit_cross_arm.py"
    assert command[-2:] == ["--expected-fixtures", "fixtures"]


# --------------------------------------------------------------------------
# the real launcher, on /bin/echo
# --------------------------------------------------------------------------


def test_the_real_pool_runs_every_job_and_logs_each_one(tmp_path):
    path = write_config(tmp_path, arms=("C", "A"), shards=(0, 1))
    config = run_full_1d.load_config(path)
    specs = run_full_1d.plan_jobs(config, ("C", "A"), (0, 1))
    results, stopped = run_full_1d.launch_jobs(
        specs, concurrency=2, poll_seconds=0.05, cwd=tmp_path)

    assert stopped is False
    assert {result.key for result in results} == {
        "C/shard0", "C/shard1", "A/shard0", "A/shard1"}
    for result in results:
        assert result.exit_code == 0
        assert result.wall_seconds >= 0
        # /bin/echo prints no timing line, and nothing else is parsed.
        assert result.total_simulation_seconds is None
        text = Path(result.log_path).read_text()
        assert "$ /bin/echo" in text
        assert "arm=" in text


def _fixed_command(command):
    def mutate(payload):
        for arm in payload["arms"].values():
            for entry in arm["full_run"]["shards"]:
                entry["command"] = command
    return mutate


def test_the_real_pool_reads_the_evaluator_timing_line(tmp_path):
    """The one line 1d parses out of a run log, end to end."""
    path = write_config(
        tmp_path, arms=("C",), shards=(0,),
        mutate=_fixed_command("/bin/echo Total simulation time: 6900.5s"))
    config = run_full_1d.load_config(path)
    specs = run_full_1d.plan_jobs(config, ("C",), (0,))
    results, _stopped = run_full_1d.launch_jobs(
        specs, concurrency=1, poll_seconds=0.05, cwd=tmp_path)
    assert results[0].total_simulation_seconds == 6900.5


def test_the_real_pool_honours_the_stop_file_between_launches(tmp_path):
    path = write_config(tmp_path, arms=("C", "A"), shards=(0, 1))
    config = run_full_1d.load_config(path)
    specs = run_full_1d.plan_jobs(config, ("C", "A"), (0, 1))
    stop_file = tmp_path / "STOP"
    stop_file.write_text("stop\n")
    results, stopped = run_full_1d.launch_jobs(
        specs, concurrency=1, poll_seconds=0.05, stop_file=stop_file,
        cwd=tmp_path)
    assert stopped is True
    assert results == []


def _sleep_wall(tmp_path, concurrency, seconds="0.4"):
    path = write_config(tmp_path, arms=("C",), shards=(0, 1),
                        mutate=_fixed_command("/bin/sleep %s" % seconds))
    config = run_full_1d.load_config(path)
    specs = run_full_1d.plan_jobs(config, ("C",), (0, 1))
    started = time.monotonic()
    results, _stopped = run_full_1d.launch_jobs(
        specs, concurrency=concurrency, poll_seconds=0.05, cwd=tmp_path)
    assert [result.exit_code for result in results] == [0, 0]
    return time.monotonic() - started


def test_the_real_pool_respects_the_concurrency_limit(tmp_path):
    """Two 0.4 s jobs: serial at a pool of one, together at a pool of two."""
    assert _sleep_wall(tmp_path / "serial", 1) >= 0.8
    assert _sleep_wall(tmp_path / "parallel", 2) < 0.8


@pytest.mark.parametrize("name", list(run_full_1d.COMPLETION_ARTIFACTS))
def test_completion_needs_both_of_run_arms_artifacts(tmp_path, name):
    directory = tmp_path / "arm"
    directory.mkdir()
    (directory / name).write_text("{}")
    assert run_full_1d.is_complete(directory) is False
    for other in run_full_1d.COMPLETION_ARTIFACTS:
        (directory / other).write_text("{}")
    assert run_full_1d.is_complete(directory) is True
