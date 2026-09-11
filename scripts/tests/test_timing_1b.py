"""Tests for stage-1 step 1b: the timing runner and the convergence report.

Synthetic only. No simulation is launched: `run_timing_1b.main` takes its
process launcher and its cross-arm audit as injectable callables, and the
convergence tests are driven by hand-written `eval.json` files whose
per-fixture values make every range, SD and fit checkable by hand.

The last two tests are the ones that matter for the protocol: 1b picks
`n_sims`, so its outputs must carry spreads and never a log-loss level.
"""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import pytest
import yaml

from sequence_track import convergence_1b, run_timing_1b
from sequence_track.run_timing_1b import (EXIT_OK, EXIT_REFUSED, EXIT_STOPPED,
                                          RunResult)

ARMS = ("A", "A50", "B", "C")
SEEDS = (20260910, 20260911, 20260912)
FIXTURES = ("1500001", "1500002")


# --------------------------------------------------------------------------
# synthetic config
# --------------------------------------------------------------------------


def write_config(tmp_path: Path, *, n_sims=(50, 100, 200, 400, 800),
                 base_seeds=SEEDS) -> Path:
    """A config with the same token shapes as the registered stage-1 one."""
    root = tmp_path / "timing"
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir(exist_ok=True)
    arms = {}
    for arm in ARMS:
        output_template = "%s/%s/seed<base_seed>_n<n_sims>" % (root, arm)
        arms[arm] = {
            "timing_1b": {
                "fixture_dir": str(fixtures),
                "fixture_count_expected": len(FIXTURES),
                "base_seeds": list(base_seeds),
                "n_sims": list(n_sims),
                "output_dir_template": output_template,
                "command_template": (
                    "/bin/echo --arm %s --n-sims <candidate> "
                    "--base-seed <batch_base_seed> --output-dir %s"
                    % (arm, output_template)),
            }
        }
    payload = {"arms": arms,
               "convergence_protocol": {"threshold": 0.002,
                                        "batch_base_seeds": list(base_seeds)}}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


class Recorder:
    """Stands in for the subprocess launcher; records what would have run."""

    def __init__(self, stop_file: Path = None, exit_code: int = 0):
        self.groups = []
        self.stop_file = stop_file
        self.exit_code = exit_code

    def __call__(self, specs, poll_seconds=2.0):
        self.groups.append([spec.key for spec in specs])
        if self.stop_file is not None:
            self.stop_file.parent.mkdir(parents=True, exist_ok=True)
            self.stop_file.write_text("stop\n", encoding="utf-8")
        return [
            RunResult(arm=spec.arm, n_sims=spec.n_sims,
                      base_seed=spec.base_seed, command=spec.command,
                      output_dir=str(spec.output_dir),
                      log_path=str(spec.log_path),
                      started_at="2026-09-11T00:00:00+00:00",
                      ended_at="2026-09-11T00:01:00+00:00",
                      wall_seconds=60.0, exit_code=self.exit_code,
                      total_simulation_seconds=42.5,
                      peak_rss_bytes=512 * 1024 * 1024)
            for spec in specs
        ]


def no_audit(group, fixture_dir, audit_dir):
    return {"status": "pass", "exit_code": 0}


def mark_complete(config_path: Path, arm: str, n_sims: int,
                  base_seed: int) -> Path:
    config = run_timing_1b.load_config(config_path)
    spec = run_timing_1b.build_run_spec(config, arm, n_sims, base_seed)
    spec.output_dir.mkdir(parents=True, exist_ok=True)
    for name in run_timing_1b.COMPLETION_ARTIFACTS:
        (spec.output_dir / name).write_text("{}", encoding="utf-8")
    return spec.output_dir


# --------------------------------------------------------------------------
# command rendering
# --------------------------------------------------------------------------


def test_render_template_substitutes_every_token():
    template = ("run --n-sims <candidate> --base-seed <batch_base_seed> "
                "--output-dir timing/A/seed<base_seed>_n<n_sims>")
    rendered = run_timing_1b.render_template(template, 400, 20260911)
    assert rendered == ("run --n-sims 400 --base-seed 20260911 "
                        "--output-dir timing/A/seed20260911_n400")


def test_render_template_refuses_an_unresolved_token():
    with pytest.raises(ValueError, match="unresolved token"):
        run_timing_1b.render_template("run --n-sims <chosen_n_sims>", 50, 1)


def test_build_run_spec_from_the_registered_config():
    """The real config's templates render to a runnable command."""
    config = run_timing_1b.load_config(
        "experiments/configs/seq_stage1_sim_v1.yaml")
    spec = run_timing_1b.build_run_spec(config, "A", 50, 20260910)
    assert "--n-sims 50" in spec.command
    assert "--base-seed 20260910" in spec.command
    assert "<" not in spec.command
    assert spec.output_dir == Path(
        "models/embeddings/seq_stage1/timing/A/seed20260910_n50")
    assert spec.argv[0] == "env"
    assert "--arm" in spec.argv and spec.argv[spec.argv.index("--arm") + 1] == "A"
    assert spec.key == "A/n50/seed20260910"


def test_all_four_arms_share_the_shard_and_the_candidate_list():
    config = run_timing_1b.load_config(
        "experiments/configs/seq_stage1_sim_v1.yaml")
    assert run_timing_1b.shard_fixture_dir(config, ARMS)
    assert 50 in run_timing_1b.permitted_n_sims(config, ARMS)
    assert run_timing_1b.registered_base_seeds(config, ARMS) == SEEDS
    assert run_timing_1b.timing_root(config, ARMS) == Path(
        "models/embeddings/seq_stage1/timing")


def test_groups_run_ascending_n_sims_then_seed(tmp_path):
    config = run_timing_1b.load_config(write_config(tmp_path))
    groups = run_timing_1b.plan_groups(config, ARMS, [200, 50], SEEDS)
    assert [group.key for group in groups] == [
        "n50/seed20260910", "n50/seed20260911", "n50/seed20260912",
        "n200/seed20260910", "n200/seed20260911", "n200/seed20260912"]
    assert [spec.arm for spec in groups[0].specs] == list(ARMS)


# --------------------------------------------------------------------------
# refusals
# --------------------------------------------------------------------------


def test_unpermitted_n_sims_is_refused(tmp_path):
    config_path = write_config(tmp_path, n_sims=(50, 100))
    config = run_timing_1b.load_config(config_path)
    with pytest.raises(ValueError, match="not permitted"):
        run_timing_1b.check_candidates(config, ARMS, [50, 300])
    recorder = Recorder()
    code = run_timing_1b.main(
        ["--config", str(config_path), "--candidates", "300"],
        launch=recorder, audit=no_audit)
    assert code == EXIT_REFUSED
    assert recorder.groups == []


def test_unregistered_base_seed_is_refused(tmp_path):
    config_path = write_config(tmp_path)
    recorder = Recorder()
    code = run_timing_1b.main(
        ["--config", str(config_path), "--candidates", "50",
         "--base-seeds", "12345"], launch=recorder, audit=no_audit)
    assert code == EXIT_REFUSED
    assert recorder.groups == []


def test_arms_that_disagree_on_the_block_are_refused(tmp_path):
    config_path = write_config(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["arms"]["B"]["timing_1b"]["n_sims"] = [50]
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="disagree"):
        run_timing_1b.permitted_n_sims(
            run_timing_1b.load_config(config_path), ARMS)


# --------------------------------------------------------------------------
# skip-if-complete and the STOP file
# --------------------------------------------------------------------------


def test_skip_if_complete_and_force(tmp_path):
    config_path = write_config(tmp_path)
    mark_complete(config_path, "A", 50, 20260910)
    mark_complete(config_path, "B", 50, 20260910)

    recorder = Recorder()
    code = run_timing_1b.main(
        ["--config", str(config_path), "--candidates", "50",
         "--base-seeds", "20260910"], launch=recorder, audit=no_audit)
    assert code == EXIT_OK
    assert recorder.groups == [["A50/n50/seed20260910", "C/n50/seed20260910"]]

    forced = Recorder()
    code = run_timing_1b.main(
        ["--config", str(config_path), "--candidates", "50",
         "--base-seeds", "20260910", "--force"], launch=forced, audit=no_audit)
    assert code == EXIT_OK
    assert forced.groups == [["%s/n50/seed20260910" % arm for arm in ARMS]]


def test_skipped_runs_are_still_noted_in_the_record(tmp_path):
    config_path = write_config(tmp_path)
    for arm in ARMS:
        mark_complete(config_path, arm, 50, 20260910)
    recorder = Recorder()
    run_timing_1b.main(
        ["--config", str(config_path), "--candidates", "50",
         "--base-seeds", "20260910"], launch=recorder, audit=no_audit)
    assert recorder.groups == []
    record = json.loads((tmp_path / "timing" / "timing_record.json")
                        .read_text(encoding="utf-8"))
    assert set(record["runs"]) == {"%s/n50/seed20260910" % arm for arm in ARMS}
    assert all(entry["skipped"] for entry in record["runs"].values())


def test_stop_file_stops_between_groups(tmp_path):
    config_path = write_config(tmp_path)
    stop_file = tmp_path / "timing" / "STOP"
    recorder = Recorder(stop_file=stop_file)
    code = run_timing_1b.main(
        ["--config", str(config_path), "--candidates", "50"],
        launch=recorder, audit=no_audit)
    assert code == EXIT_STOPPED
    # The first group ran to completion; the second was never launched.
    assert len(recorder.groups) == 1
    record = json.loads((tmp_path / "timing" / "timing_record.json")
                        .read_text(encoding="utf-8"))
    assert record["stopped_before_group"] == "n50/seed20260911"


def test_stop_file_present_up_front_launches_nothing(tmp_path):
    config_path = write_config(tmp_path)
    stop_file = tmp_path / "timing" / "STOP"
    stop_file.parent.mkdir(parents=True, exist_ok=True)
    stop_file.write_text("stop\n", encoding="utf-8")
    recorder = Recorder()
    code = run_timing_1b.main(
        ["--config", str(config_path), "--candidates", "50"],
        launch=recorder, audit=no_audit)
    assert code == EXIT_STOPPED
    assert recorder.groups == []


# --------------------------------------------------------------------------
# the record
# --------------------------------------------------------------------------


def test_record_holds_timing_exit_codes_and_audits(tmp_path):
    config_path = write_config(tmp_path)
    audits = []

    def fake_audit(group, fixture_dir, audit_dir):
        audits.append((group.key, str(fixture_dir)))
        return {"status": "pass", "exit_code": 0,
                "expected_fixtures": str(fixture_dir)}

    run_timing_1b.main(
        ["--config", str(config_path), "--candidates", "50",
         "--base-seeds", "20260910"], launch=Recorder(), audit=fake_audit)
    record = json.loads((tmp_path / "timing" / "timing_record.json")
                        .read_text(encoding="utf-8"))
    entry = record["runs"]["A/n50/seed20260910"]
    assert entry["wall_seconds"] == 60.0
    assert entry["exit_code"] == 0
    assert entry["total_simulation_seconds"] == 42.5
    assert entry["peak_rss_mb"] == 512.0
    assert entry["started_at"] and entry["ended_at"]
    assert record["audits"]["n50/seed20260910"]["status"] == "pass"
    assert audits == [("n50/seed20260910", str(tmp_path / "fixtures"))]
    assert record["contract"] == run_timing_1b.RECORD_CONTRACT


def test_a_failing_run_makes_the_runner_exit_non_zero(tmp_path):
    config_path = write_config(tmp_path)
    code = run_timing_1b.main(
        ["--config", str(config_path), "--candidates", "50",
         "--base-seeds", "20260910"],
        launch=Recorder(exit_code=1), audit=no_audit)
    assert code == run_timing_1b.EXIT_RUN_FAILED


def test_dry_run_prints_commands_and_writes_nothing(tmp_path, capsys):
    config_path = write_config(tmp_path)
    code = run_timing_1b.main(
        ["--config", str(config_path), "--candidates", "50",
         "--base-seeds", "20260910", "--dry-run"],
        launch=Recorder(), audit=no_audit)
    out = capsys.readouterr().out
    assert code == EXIT_OK
    assert "--n-sims 50" in out and "--base-seed 20260910" in out
    assert not (tmp_path / "timing" / "timing_record.json").exists()


# --------------------------------------------------------------------------
# log parsing and RSS sampling
# --------------------------------------------------------------------------


def test_launch_group_runs_the_arms_concurrently_and_captures_the_log(tmp_path):
    """Real subprocesses, no simulation: four shells that sleep and log."""
    root = tmp_path / "timing"
    specs = []
    for arm in ARMS:
        output_dir = root / arm / "seed20260910_n50"
        script = ("sleep 0.3; echo 'Avg Log Loss: 0.83'; "
                  "echo 'Total simulation time: 3.0s'; "
                  "echo '{}' > %s/eval.json; "
                  "echo '{}' > %s/arm_provenance.json" % (output_dir, output_dir))
        specs.append(run_timing_1b.RunSpec(
            arm=arm, n_sims=50, base_seed=20260910,
            command="/bin/sh -c %r" % script, output_dir=output_dir))

    import time
    start = time.monotonic()
    results = run_timing_1b.launch_group(specs, poll_seconds=0.05)
    elapsed = time.monotonic() - start

    assert [result.exit_code for result in results] == [0, 0, 0, 0]
    # Concurrent, not serial: four 0.3 s sleeps finish well inside 1.2 s.
    assert elapsed < 1.0
    for result, spec in zip(results, specs):
        assert result.total_simulation_seconds == 3.0
        assert run_timing_1b.is_complete(spec.output_dir)
        log = spec.log_path.read_text(encoding="utf-8")
        assert "Total simulation time: 3.0s" in log
        assert result.peak_rss_bytes is None or result.peak_rss_bytes > 0


def test_launch_group_reports_a_non_zero_exit(tmp_path):
    spec = run_timing_1b.RunSpec(
        arm="A", n_sims=50, base_seed=20260910, command="/bin/sh -c 'exit 3'",
        output_dir=tmp_path / "A" / "seed20260910_n50")
    result = run_timing_1b.launch_group([spec], poll_seconds=0.05)[0]
    assert result.exit_code == 3
    assert result.total_simulation_seconds is None


def test_parse_total_simulation_time_takes_the_last_line():
    text = ("Matches evaluated: 10\nTotal simulation time: 12.3s\n"
            "...\nTotal simulation time: 45.6s\n")
    assert run_timing_1b.parse_total_simulation_seconds(text) == 45.6
    assert run_timing_1b.parse_total_simulation_seconds("no timing here") is None


def test_parse_total_simulation_time_ignores_metric_lines():
    """Only the timing line is read back out of a run log."""
    text = ("Avg Log Loss: 0.8311\nFlat ROI: -100.00%\n"
            "Total simulation time: 7.0s\n")
    assert run_timing_1b.parse_total_simulation_seconds(text) == 7.0


def test_tree_rss_sums_the_descendants():
    table = {10: (1, 100), 11: (10, 200), 12: (11, 300), 20: (1, 999)}
    assert run_timing_1b.tree_rss_bytes(table, 10) == 600
    assert run_timing_1b.tree_rss_bytes(table, 99) == 0


# --------------------------------------------------------------------------
# convergence_1b: synthetic runs with known spreads
# --------------------------------------------------------------------------

# Per-fixture winner log losses chosen so every reported statistic is
# checkable by hand. Shard means: A is 0.55 in every batch; B is 0.65, 0.66,
# 0.68, so the B-A paired differences are 0.10, 0.11, 0.13 (range 0.03).
LEVELS = {
    20260910: {"A": [0.50, 0.60], "B": [0.60, 0.70]},
    20260911: {"A": [0.52, 0.58], "B": [0.58, 0.74]},
    20260912: {"A": [0.50, 0.60], "B": [0.66, 0.70]},
}
EXPECTED_DIFFERENCES = [0.10, 0.11, 0.13]


def write_eval(output_dir: Path, values) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    matches = []
    for fixture, value in zip(FIXTURES, values):
        matches.append({
            "match_id": fixture, "cricsheet_id": fixture,
            "actual_winner": "Home",
            "simulated_prob": {"Home": math.exp(-value),
                               "Away": 1 - math.exp(-value)},
            "log_loss": value,
        })
    (output_dir / "eval.json").write_text(
        json.dumps({"summary": {"slice": "all"}, "matches": matches}),
        encoding="utf-8")
    (output_dir / "arm_provenance.json").write_text("{}", encoding="utf-8")


def build_timing_tree(tmp_path: Path, *, n_sims_list=(100, 400),
                      scale_by_n=True, sliced=False) -> Path:
    """A timing record plus eval.json files for every (arm, n, seed)."""
    root = tmp_path / "timing"
    runs = {}
    for n_sims in n_sims_list:
        # Spreads shrink as 1/sqrt(n) when scale_by_n: the batch offsets are
        # divided by sqrt(n / n_first).
        shrink = (math.sqrt(n_sims_list[0] / n_sims) if scale_by_n else 1.0)
        for base_seed in SEEDS:
            for arm in ARMS:
                source = "A" if arm in ("A", "A50") else "B"
                reference = LEVELS[20260910][source]
                values = [
                    reference[index]
                    + (LEVELS[base_seed][source][index] - reference[index]) * shrink
                    for index in range(len(FIXTURES))
                ]
                output_dir = root / arm / ("seed%d_n%d" % (base_seed, n_sims))
                write_eval(output_dir, values)
                if sliced:
                    sliced_dir = output_dir / "sliced"
                    sliced_dir.mkdir(parents=True, exist_ok=True)
                    write_eval(sliced_dir, values)
                    (sliced_dir / "eval.json").rename(
                        sliced_dir / "x_all_20260911_000000_min_volume_50000.json")
                runs["%s/n%d/seed%d" % (arm, n_sims, base_seed)] = {
                    "arm": arm, "n_sims": n_sims, "base_seed": base_seed,
                    "output_dir": str(output_dir), "exit_code": 0,
                    "wall_seconds": 20.0 * n_sims / 100.0,
                    "total_simulation_seconds": 18.0 * n_sims / 100.0,
                    "peak_rss_bytes": 1024 * 1024 * 1024,
                    "skipped": False,
                }
    record = {"contract": run_timing_1b.RECORD_CONTRACT,
              "base_seeds": list(SEEDS),
              "expected_fixture_count": len(FIXTURES),
              "runs": runs}
    record_path = root / "timing_record.json"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record), encoding="utf-8")
    return record_path


def run_convergence(record_path: Path, *extra) -> dict:
    code = convergence_1b.main(
        ["--record", str(record_path), "--config", "does/not/exist.yaml",
         "--threshold", "0.002", "--base-seeds",
         *[str(seed) for seed in SEEDS], *extra])
    assert code == 0
    return json.loads((record_path.parent / "convergence_1b.json")
                      .read_text(encoding="utf-8"))


def test_convergence_ranges_match_the_hand_computed_values(tmp_path):
    payload = run_convergence(build_timing_tree(tmp_path, n_sims_list=(100,),
                                                scale_by_n=False))
    rows = {row["contrast"]: row for row in payload["spread_table"]
            if row["n_sims"] == 100}
    row = rows["B-A"]
    assert row["batch_count"] == 3
    assert row["fixture_count"] == 2
    assert row["range_95"] == pytest.approx(0.03)
    assert row["sd"] == pytest.approx(statistics.stdev(EXPECTED_DIFFERENCES))
    assert row["below_threshold"] is False
    scale = math.sqrt(2 / 255)
    assert payload["scale_factor"] == pytest.approx(scale)
    assert row["scaled_range_95"] == pytest.approx(0.03 * scale)
    # A50 mirrors A and C mirrors B, so A50-A has no spread at all and C-B
    # is identically zero: a clean below-threshold case.
    assert rows["A50-A"]["range_95"] == pytest.approx(0.0)
    assert rows["A50-A"]["below_threshold"] is True
    assert rows["C-B"]["range_95"] == pytest.approx(0.0)
    assert rows["C-A"]["range_95"] == pytest.approx(0.03)


def test_variability_rerun_sd_matches_the_hand_computed_value(tmp_path):
    payload = run_convergence(build_timing_tree(tmp_path, n_sims_list=(100,),
                                                scale_by_n=False))
    row = next(entry for entry in payload["variability_rerun"]
               if entry["contrast"] == "B-A" and entry["n_sims"] == 100)
    # Per fixture, (B-A at seed 20260910) - (B-A at seed 20260911) is
    # [0.10 - 0.06, 0.10 - 0.16] = [0.04, -0.06]; its sample SD is 0.1/sqrt(2).
    assert row["seed_pair"] == [20260910, 20260911]
    assert row["fixture_count"] == 2
    assert row["per_fixture_paired_diff_sd"] == pytest.approx(0.1 / math.sqrt(2))
    assert row["shard_mean_paired_diff_sd"] == pytest.approx(
        0.1 / math.sqrt(2) / math.sqrt(2))
    assert row["scaled_paired_diff_sd"] == pytest.approx(
        0.1 / 2 * math.sqrt(2 / 255))


def test_noise_curve_fit_recovers_a_and_the_crossing():
    fit = convergence_1b.fit_noise_curve([(100, 0.1), (400, 0.05)],
                                         threshold=0.002, scale=1.0)
    assert fit["fitted_a"] == pytest.approx(1.0)
    assert fit["n_at_threshold"] == pytest.approx(250000.0)
    assert fit["fit_relative_rmse"] == pytest.approx(0.0, abs=1e-12)
    scaled = convergence_1b.fit_noise_curve([(100, 0.1), (400, 0.05)],
                                            threshold=0.002, scale=0.2)
    assert scaled["n_at_threshold_scaled"] == pytest.approx(10000.0)


def test_noise_curve_runs_over_every_candidate(tmp_path):
    payload = run_convergence(build_timing_tree(
        tmp_path, n_sims_list=(100, 400), scale_by_n=True))
    fit = next(row for row in payload["noise_curve"]
               if row["contrast"] == "B-A")
    assert fit["points"] == 2
    assert fit["n_values"] == [100, 400]
    # The synthetic offsets shrink exactly as 1/sqrt(n), so the fit is exact.
    assert fit["fitted_a"] == pytest.approx(0.03 * math.sqrt(100))
    assert fit["fit_relative_rmse"] == pytest.approx(0.0, abs=1e-12)


def test_timing_table_extrapolates_per_match_and_concurrently(tmp_path):
    payload = run_convergence(build_timing_tree(tmp_path, n_sims_list=(100,),
                                                scale_by_n=False))
    row = next(entry for entry in payload["timing_per_arm"]
               if entry["arm"] == "A" and entry["n_sims"] == 100)
    assert row["batch_count"] == 3
    assert row["elapsed_seconds_mean"] == pytest.approx(20.0)
    assert row["seconds_per_match_mean"] == pytest.approx(10.0)  # 20s / 2
    assert row["seconds_per_simulation"] == pytest.approx(0.1)
    assert row["serial_seconds_255"] == pytest.approx(2550.0)
    assert row["peak_rss_mb_max"] == pytest.approx(1024.0)
    concurrent = payload["timing_concurrent"][0]
    assert concurrent["arms_measured"] == 4
    assert concurrent["concurrent_seconds_255"] == pytest.approx(2550.0)
    assert concurrent["serial_seconds_255_sum"] == pytest.approx(4 * 2550.0)


def test_number_formatting_keeps_integer_magnitudes():
    assert convergence_1b._fmt(2550.0, 0) == "2550"
    assert convergence_1b._fmt(0.1, 4) == "0.1"
    assert convergence_1b._fmt(0.0, 4) == "0"
    assert convergence_1b._fmt(None) == "n/a"
    assert convergence_1b._fmt(True) == "yes"


def test_slice_is_all_when_the_shard_is_not_sliceable(tmp_path):
    payload = run_convergence(build_timing_tree(tmp_path, n_sims_list=(100,),
                                                scale_by_n=False))
    assert payload["slice_used"] == "all"
    assert "not sliceable" in payload["slice_reason"]


def test_slice_is_used_when_the_shard_is_sliceable(tmp_path):
    payload = run_convergence(build_timing_tree(tmp_path, n_sims_list=(100,),
                                                scale_by_n=False, sliced=True))
    assert payload["slice_used"] == "min_volume_50000"
    assert "is sliceable" in payload["slice_reason"]


def test_a_stale_sliced_file_is_refused(tmp_path, capsys):
    record_path = build_timing_tree(tmp_path, n_sims_list=(100,),
                                    scale_by_n=False, sliced=True)
    stale = (record_path.parent / "A" / "seed20260910_n100" / "sliced"
             / "x_all_20260911_000000_min_volume_50000.json")
    payload = json.loads(stale.read_text(encoding="utf-8"))
    payload["matches"][0]["log_loss"] = 9.9
    stale.write_text(json.dumps(payload), encoding="utf-8")
    code = convergence_1b.main(["--record", str(record_path),
                                "--config", "does/not/exist.yaml"])
    assert code == 2
    assert "stale" in capsys.readouterr().err


# --------------------------------------------------------------------------
# the no-levels guard
# --------------------------------------------------------------------------


@pytest.mark.parametrize("payload", [
    {"rows": [{"avg_log_loss": 0.5}]},
    {"ll_mean": 0.5},
    {"mean_ll": 0.5},
    {"batch_1_delta": 0.1},
    {"flat_roi_pct": 1.0},
    {"avg_brier_score": 0.2},
])
def test_assert_no_level_leak_catches_a_level_field(payload):
    with pytest.raises(AssertionError, match="never log-loss levels"):
        convergence_1b.assert_no_level_leak(payload)


def test_assert_no_level_leak_allows_spread_fields():
    convergence_1b.assert_no_level_leak({"range_95": 0.03, "ll_range": 0.03,
                                         "paired_diff_sd": 0.01,
                                         "seconds_per_match_mean": 1.0})


def test_outputs_carry_no_level_anywhere(tmp_path):
    record_path = build_timing_tree(tmp_path, n_sims_list=(100, 400))
    payload = run_convergence(record_path)
    json_text = (record_path.parent / "convergence_1b.json").read_text(
        encoding="utf-8")
    markdown = (record_path.parent / "convergence_1b.md").read_text(
        encoding="utf-8")

    for banned in ("log_loss", "ll_mean", "mean_ll", "avg_ll", "delta"):
        assert banned not in json_text, "%r leaked into the JSON" % banned
        assert banned not in markdown, "%r leaked into the markdown" % banned

    # No key anywhere carries a level, and no markdown column does either.
    convergence_1b.assert_no_level_leak(payload)
    for line in markdown.splitlines():
        if line.startswith("|") and "---" not in line:
            convergence_1b.assert_headers_clean(
                [cell.strip() for cell in line.strip("|").split("|")],
                "markdown row")

    # The internal levels really are 0.50-0.74; none of them is printed.
    for level in ("0.5", "0.58", "0.6", "0.66", "0.7", "0.74"):
        assert ("| %s |" % level) not in markdown
    assert payload["candidate_notes"]["50"].startswith("noise-curve point only")
    assert "spreads only" in payload["reported_statistics"]


def test_markdown_states_the_slice_and_the_range_95_definition(tmp_path):
    record_path = build_timing_tree(tmp_path, n_sims_list=(100,),
                                    scale_by_n=False)
    run_convergence(record_path)
    markdown = (record_path.parent / "convergence_1b.md").read_text(
        encoding="utf-8")
    assert "Slice used: **all**" in markdown
    assert "95% range" in markdown
    assert "noise-curve point only" in markdown
