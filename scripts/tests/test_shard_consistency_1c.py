"""Tests for stage-1 step 1c: the shard-consistency comparison.

Synthetic only. No simulation is launched: the comparison is driven by
hand-written `eval.json`, `raw_sims.jsonl` and `arm_provenance.json` files,
and the launcher is never called. The tests that matter for the protocol are
the last three: 1c must report a difference by FIELD NAME and never by
value, and its outputs must be refused if a number that is not a count ever
reaches them.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from sequence_track import shard_consistency_1c as mod
from sequence_track.shard_consistency_1c import (CASE_SET, ConsistencyError,
                                                 build_report,
                                                 check_case_set,
                                                 check_matches_advanced_decomposition,
                                                 compare_fixture,
                                                 compare_run_identity,
                                                 first_difference, load_run,
                                                 ordered_cases,
                                                 render_command,
                                                 render_markdown,
                                                 shard_assignment,
                                                 shard_members)

ARM = "A"


# --------------------------------------------------------------------------
# synthetic run outputs
# --------------------------------------------------------------------------


def eval_record(fixture: str) -> dict:
    return {
        "match_id": fixture,
        "cricsheet_id": fixture,
        "display_match_id": f"{fixture}_display",
        "teams": ["India", "Australia"],
        "actual_winner": "India",
        "simulated_prob": {"India": 0.6, "Australia": 0.4},
        "market_prob": {"India": 0.55, "Australia": 0.45},
        "log_loss": 0.5108256,
        "brier_score": 0.16,
        "bet_placed": True,
        "bet_team": "India",
    }


def raw_row(fixture: str, date: str) -> dict:
    return {
        "match_id": fixture,
        "date": date,
        "first_batting_team": "India",
        "actual_winner": "India",
        "fixture_seed": 1234567,
        "n_sims": 2,
        "simulations": [
            {"winner": "India", "scores": {"India": 180, "Australia": 170},
             "tie": False},
            {"winner": "Australia", "scores": {"India": 150,
                                               "Australia": 151},
             "tie": False},
        ],
    }


def provenance_row(fixture: str, date: str, *, scored: bool = True,
                   advanced_before=(), matches_advanced: int = 0) -> dict:
    return {
        "cricsheet_id": fixture,
        "match_date": date,
        "as_of": {
            "date": date,
            "same_day_advanced_before": list(advanced_before),
            "matches_advanced": matches_advanced,
        },
        "eligibility": {
            "odds_row_found": scored,
            "odds_row_id": ({"cricsheet_id": fixture,
                             "display_match_id": None} if scored else None),
            "odds_row_sha256": "abc123" if scored else None,
            "odds_actual_winner": "India" if scored else None,
            "cricsheet_resolved": True,
            "cricsheet_winner": "India",
            "male_t20": True,
            "scored": scored,
            "skip_reason": None if scored else "no_odds_row",
        },
        "fixture_seed": 1234567,
        "sub_seeds": {"outcome": 11, "extras": 22, "selector": 33},
        "n_sims": 2,
    }


def run_block(**overrides) -> dict:
    block = {
        "arm": ARM,
        "model_dir": "models/xgb_i7_noweights_production",
        "model_dir_hash": "a6ac7671",
        "checkpoint_path": "models/xgb_i7_noweights_production/m.pkl",
        "checkpoint_md5": "7ee1e180",
        "stats_version": "i7",
        "stats_cache_path": "models/player_stats_cache_i7.sqlite",
        "stats_cache_md5": "671ac820",
        "selector_class": "RosterEmpiricalBowlerSelector",
        "bowler_usage_path": "models/bowler_phase_usage.json",
        "bowler_usage_md5": "2e650423",
        "bowler_roster_policy_path": "models/bowler_roster_policy.json",
        "bowler_roster_policy_md5": "f2a5efa2",
        "extras_graft_path": "models/auto/b18/extras_graft_v1.json",
        "extras_graft_sha256": "ad6e863b",
        "extras_graft_applied_via": "attribute",
        "runout_p": 0.075077,
        "clip": {"low": 0.01, "high": 0.99, "seam_enabled": True},
        "base_seed": 20260910,
        "n_sims": 2,
        "engine_md5": "d7e7a70d",
        "runner_md5": "1111",
        "threads": 1,
        "device": "cpu",
        "context_dir": "data/t20s_json",
        "context_dir_hash": "6da2f2d7",
        "odds": "betting_odds_polymarket_v2.json",
        "odds_sha256": "dc36aef3",
        "player_metadata_sha256": "e5a657b7",
        "fixture_dir": "fixtures",
        "fixture_count": 1,
        "started_at": "2026-09-11T00:00:00Z",
    }
    block.update(overrides)
    return block


def write_run(directory: Path, fixtures, *, unscored=(), run_overrides=None,
              eval_mutator=None, raw_mutator=None, prov_mutator=None) -> Path:
    """Write one synthetic arm output dir.

    `fixtures` is a sequence of `(cricsheet_id, date, matches_advanced,
    advanced_before)` tuples.
    """
    directory.mkdir(parents=True, exist_ok=True)
    matches = []
    raw_rows = []
    provenance_rows = []
    for fixture, date, advanced, before in fixtures:
        scored = fixture not in set(unscored)
        row = provenance_row(fixture, date, scored=scored,
                             advanced_before=before,
                             matches_advanced=advanced)
        if prov_mutator is not None:
            prov_mutator(fixture, row)
        provenance_rows.append(row)
        if not scored:
            continue
        record = eval_record(fixture)
        if eval_mutator is not None:
            eval_mutator(fixture, record)
        matches.append(record)
        raw = raw_row(fixture, date)
        if raw_mutator is not None:
            raw_mutator(fixture, raw)
        raw_rows.append(raw)

    (directory / "eval.json").write_text(json.dumps({
        "match_identity": {"match_identity_version": "v2",
                           "primary_key": "cricsheet_id",
                           "display_key": "display_match_id"},
        "summary": {"model_type": "xgboost", "slice": "all",
                    "min_volume": None,
                    "cost_model": {"spread_bps": 0.0, "fee_bps": 0.0,
                                   "fee_basis": "winnings"},
                    "price_basis": "mid", "volume_basis": "event",
                    "n_matches": len(matches),
                    "n_matches_evaluated": len(matches),
                    "avg_log_loss": 0.5108256,
                    "avg_brier_score": 0.16,
                    "flat_betting_roi_pct": 3.38,
                    "bootstrap_seed": 42, "bootstrap_resamples": 10000,
                    "bootstrap_contract": "tournament_time_block_v1",
                    "calibration_method": None,
                    "ball_calibration_enabled": False,
                    "ball_calibrator_path": None},
        "matches": matches,
    }, indent=2))
    with (directory / "raw_sims.jsonl").open("w") as handle:
        for row in raw_rows:
            handle.write(json.dumps(row) + "\n")
    block = run_block(fixture_count=len(fixtures))
    block.update(run_overrides or {})
    (directory / "arm_provenance.json").write_text(json.dumps({
        "contract": "sequence_track_arm_provenance_v2",
        "arm": block["arm"],
        "run": block,
        "fixtures": provenance_rows,
    }, indent=2))
    return directory


SERIAL_FIXTURES = [
    ("1000001", "2025-09-10", 0, []),
    ("1000002", "2025-09-10", 1, ["1000001"]),
    ("1000003", "2026-01-27", 2, []),
]
# shard 0 holds 1000002 only: it starts on 2025-09-10 with the same same-day
# predecessor advanced as context, but its run-cumulative counter is 1 too.
SHARD_FIXTURES = [("1000002", "2025-09-10", 1, ["1000001"])]
# shard 1 holds 1000003, whose run-cumulative counter is 0 because this
# shard never visited the earlier date.
SHARD_LATE_FIXTURES = [("1000003", "2026-01-27", 0, [])]


# --------------------------------------------------------------------------
# the case set and the shard rule (check 10.1)
# --------------------------------------------------------------------------


def test_case_set_covers_every_registered_boundary_case():
    assert check_case_set(CASE_SET, 3) == []
    labels = {label for case in CASE_SET for label in case.cases}
    assert {"a", "b", "c", "d", "e"} <= labels


def test_shard_rule_splits_the_registered_same_day_pair():
    assignment = shard_assignment(CASE_SET, 3)
    assert assignment["1512766"] != assignment["1525160"]


def test_shard_members_partition_the_case_set():
    members = shard_members(CASE_SET, 3)
    flat = [fixture for ids in members.values() for fixture in ids]
    assert sorted(flat) == sorted(case.cricsheet_id for case in CASE_SET)
    assert len(flat) == len(set(flat))


def test_first_fixture_of_every_shard_is_named():
    members = shard_members(CASE_SET, 3)
    firsts = {index: ids[0] for index, ids in members.items()}
    assert len(firsts) == 3
    for index, ids in members.items():
        assert ids[0] == firsts[index]


def test_case_set_check_refuses_a_single_shard():
    with pytest.raises(ConsistencyError):
        check_case_set(CASE_SET, 1)


def test_case_set_check_reports_a_rule_that_never_splits_a_pair():
    """A set with one fixture per date cannot cover boundary case (a)."""
    cases = tuple(case for case in CASE_SET
                  if case.match_date in ("2026-01-27", "2026-01-29"))
    problems = check_case_set(cases, 2)
    assert any("same-day pair" in problem for problem in problems)


# --------------------------------------------------------------------------
# command rendering (check 10.2)
# --------------------------------------------------------------------------


REGISTERED = (
    "env -u T1_SIM_PREFIX_CACHE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "
    "uv run --no-sync python scripts/sequence_track/run_arm.py --arm B "
    "--config experiments/configs/seq_stage1_sim_v1.yaml "
    "--model-dir models/embeddings/seq_stage1/retrain_i7/mlp/seed_101 "
    "--fixture-dir models/embeddings/seq_stage1/smoke/fixtures "
    "--context-dir data/t20s_json --n-sims 10 --base-seed 20260910 "
    "--threads 1 --output-dir models/embeddings/seq_stage1/smoke/B")


def test_render_command_drops_config_and_substitutes_three_tokens():
    argv = render_command(REGISTERED, fixture_dir="case/fixtures", n_sims=20,
                          output_dir="case/B", base_seed=20260910)
    assert "--config" not in argv
    assert argv[argv.index("--fixture-dir") + 1] == "case/fixtures"
    assert argv[argv.index("--n-sims") + 1] == "20"
    assert argv[argv.index("--output-dir") + 1] == "case/B"
    # everything else verbatim
    assert argv[argv.index("--model-dir") + 1] == (
        "models/embeddings/seq_stage1/retrain_i7/mlp/seed_101")
    assert argv[argv.index("--context-dir") + 1] == "data/t20s_json"
    assert argv[argv.index("--threads") + 1] == "1"
    assert argv[0] == "env"
    assert "OMP_NUM_THREADS=1" in argv


def test_render_command_refuses_a_command_without_config():
    with pytest.raises(ConsistencyError):
        render_command(REGISTERED.replace(
            "--config experiments/configs/seq_stage1_sim_v1.yaml ", ""),
            fixture_dir="x", n_sims=1, output_dir="y")


def test_render_command_refuses_a_thread_count_other_than_one():
    with pytest.raises(ConsistencyError):
        render_command(REGISTERED.replace("--threads 1", "--threads 4"),
                       fixture_dir="x", n_sims=1, output_dir="y")


def test_render_command_refuses_a_base_seed_it_does_not_record():
    with pytest.raises(ConsistencyError):
        render_command(REGISTERED, fixture_dir="x", n_sims=1, output_dir="y",
                       base_seed=20260911)


# --------------------------------------------------------------------------
# the comparison (check 10.3)
# --------------------------------------------------------------------------


def test_identical_outputs_pass_every_artefact(tmp_path):
    serial = write_run(tmp_path / "serial", SERIAL_FIXTURES)
    shard = write_run(tmp_path / "shard0", SHARD_FIXTURES)
    results = compare_fixture(load_run(serial), load_run(shard), "1000002",
                              shard_index=0)
    assert [row.artefact for row in results] == list(mod.ARTEFACTS)
    assert {row.status for row in results} == {"PASS"}


def test_a_changed_eval_field_fails_and_names_it(tmp_path):
    serial = write_run(tmp_path / "serial", SERIAL_FIXTURES)

    def mutate(fixture, record):
        record["log_loss"] = record["log_loss"] + 0.25

    shard = write_run(tmp_path / "shard0", SHARD_FIXTURES,
                      eval_mutator=mutate)
    results = {row.artefact: row
               for row in compare_fixture(load_run(serial), load_run(shard),
                                          "1000002", shard_index=0)}
    assert results["eval_record"].status == "FAIL"
    assert results["eval_record"].detail == "first difference: log_loss"
    # the value never appears in the detail
    assert "0.7" not in results["eval_record"].detail
    assert results["provenance"].status == "PASS"


def test_a_changed_raw_simulation_fails_and_names_the_path(tmp_path):
    serial = write_run(tmp_path / "serial", SERIAL_FIXTURES)

    def mutate(fixture, row):
        row["simulations"][1]["winner"] = "India"

    shard = write_run(tmp_path / "shard0", SHARD_FIXTURES, raw_mutator=mutate)
    results = {row.artefact: row
               for row in compare_fixture(load_run(serial), load_run(shard),
                                          "1000002", shard_index=0)}
    assert results["raw_sims_row"].status == "FAIL"
    assert results["raw_sims_row"].detail == (
        "first difference: simulations[1].winner")


def test_a_changed_fixture_seed_fails(tmp_path):
    serial = write_run(tmp_path / "serial", SERIAL_FIXTURES)

    def mutate(fixture, row):
        row["fixture_seed"] = 999

    shard = write_run(tmp_path / "shard0", SHARD_FIXTURES, prov_mutator=mutate)
    results = {row.artefact: row
               for row in compare_fixture(load_run(serial), load_run(shard),
                                          "1000002", shard_index=0)}
    assert results["provenance"].status == "FAIL"
    assert results["provenance"].detail == "first difference: fixture_seed"


def test_a_changed_same_day_predecessor_list_fails(tmp_path):
    serial = write_run(tmp_path / "serial", SERIAL_FIXTURES)

    def mutate(fixture, row):
        row["as_of"]["same_day_advanced_before"] = []

    shard = write_run(tmp_path / "shard0", SHARD_FIXTURES, prov_mutator=mutate)
    results = {row.artefact: row
               for row in compare_fixture(load_run(serial), load_run(shard),
                                          "1000002", shard_index=0)}
    assert results["provenance"].status == "FAIL"
    assert results["provenance"].detail.startswith(
        "first difference: as_of.same_day_advanced_before")


def test_matches_advanced_is_excluded_because_it_is_run_cumulative(tmp_path):
    """The counter differs across a shard boundary and must not fail 1c."""
    serial = write_run(tmp_path / "serial", SERIAL_FIXTURES)
    shard = write_run(tmp_path / "shard1", SHARD_LATE_FIXTURES)
    results = {row.artefact: row
               for row in compare_fixture(load_run(serial), load_run(shard),
                                          "1000003", shard_index=1)}
    serial_value = load_run(serial).fixtures["1000003"]["as_of"][
        "matches_advanced"]
    shard_value = load_run(shard).fixtures["1000003"]["as_of"][
        "matches_advanced"]
    assert serial_value != shard_value          # the reason for the exclusion
    assert results["provenance"].status == "PASS"
    assert "as_of.matches_advanced" in mod.PROVENANCE_EXCLUSIONS


def test_an_unscored_fixture_is_absent_from_both_and_passes(tmp_path):
    fixtures = [("1000001", "2025-09-10", 0, [])]
    serial = write_run(tmp_path / "serial", fixtures, unscored=("1000001",))
    shard = write_run(tmp_path / "shard0", fixtures, unscored=("1000001",))
    results = {row.artefact: row
               for row in compare_fixture(load_run(serial), load_run(shard),
                                          "1000001", shard_index=0)}
    assert {row.status for row in results.values()} == {"PASS"}
    assert results["eval_record"].detail.startswith("absent in both")
    prov = load_run(serial).fixtures["1000001"]["eligibility"]
    assert prov["odds_row_found"] is False and prov["scored"] is False


def test_a_fixture_scored_in_one_run_only_fails(tmp_path):
    fixtures = [("1000001", "2025-09-10", 0, [])]
    serial = write_run(tmp_path / "serial", fixtures)
    shard = write_run(tmp_path / "shard0", fixtures, unscored=("1000001",))
    results = {row.artefact: row
               for row in compare_fixture(load_run(serial), load_run(shard),
                                          "1000001", shard_index=0)}
    assert results["eval_record"].status == "FAIL"
    assert "present in one run only" in results["eval_record"].detail


def test_run_identity_difference_is_named(tmp_path):
    serial = write_run(tmp_path / "serial", SERIAL_FIXTURES)
    shard = write_run(tmp_path / "shard0", SHARD_FIXTURES,
                      run_overrides={"checkpoint_md5": "deadbeef"})
    ok, detail = compare_run_identity(load_run(serial), load_run(shard))
    assert ok is False
    assert detail == "run.checkpoint_md5"


def test_run_identity_ignores_the_per_shard_fields(tmp_path):
    serial = write_run(tmp_path / "serial", SERIAL_FIXTURES)
    shard = write_run(tmp_path / "shard0", SHARD_FIXTURES,
                      run_overrides={"fixture_dir": "elsewhere",
                                     "started_at": "2026-09-11T09:00:00Z"})
    ok, detail = compare_run_identity(load_run(serial), load_run(shard))
    assert ok is True and detail is None


# --------------------------------------------------------------------------
# the decomposition of the excluded counter
# --------------------------------------------------------------------------


def test_decomposition_passes_on_a_consistent_run(tmp_path):
    serial = write_run(tmp_path / "serial", SERIAL_FIXTURES)
    assert check_matches_advanced_decomposition(load_run(serial)) == []


def test_decomposition_catches_a_same_day_disagreement(tmp_path):
    fixtures = [("1000001", "2025-09-10", 0, []),
                ("1000002", "2025-09-10", 5, ["1000001"])]
    run = write_run(tmp_path / "serial", fixtures)
    problems = check_matches_advanced_decomposition(load_run(run))
    assert problems and "2025-09-10" in problems[0]


def test_decomposition_catches_a_negative_counter(tmp_path):
    fixtures = [("1000002", "2025-09-10", 0, ["1000001"])]
    run = write_run(tmp_path / "serial", fixtures)
    problems = check_matches_advanced_decomposition(load_run(run))
    assert any("more same-day matches" in problem for problem in problems)


# --------------------------------------------------------------------------
# first_difference
# --------------------------------------------------------------------------


def test_first_difference_returns_a_path_not_a_value():
    left = {"a": {"b": [1, 2, 3]}}
    right = {"a": {"b": [1, 9, 3]}}
    assert first_difference(left, right) == "a.b[1]"


def test_first_difference_reports_a_missing_key_and_a_length():
    assert first_difference({"a": 1}, {}) == "a"
    assert first_difference({"a": [1]}, {"a": [1, 2]}) == "a[len]"
    assert first_difference({"a": 1}, {"a": 1}) is None


def test_all_differences_lists_every_path_not_just_the_first():
    left = {"a": 1, "b": {"c": 2}, "d": 3}
    right = {"a": 9, "b": {"c": 8}, "d": 3}
    assert mod.all_differences(left, right) == ["a", "b.c"]
    assert mod.all_differences(left, right, exclude={"a"}) == ["b.c"]


def test_diagnose_names_the_known_shard_dependent_field():
    assert mod.diagnose(["competition_cluster_id"]) is not None
    assert "load_competition_clusters" in mod.diagnose(
        ["competition_cluster_id"])
    assert mod.diagnose(["simulated_prob.India"]) is None


def test_a_cluster_id_difference_is_reported_and_diagnosed(tmp_path):
    serial = write_run(tmp_path / "serial", SERIAL_FIXTURES)

    def mutate(fixture, record):
        record["competition_cluster_id"] = "event:X|block_start:2026-01-29"

    def stamp(fixture, record):
        record["competition_cluster_id"] = "event:X|block_start:2026-01-27"

    write_run(tmp_path / "serial", SERIAL_FIXTURES, eval_mutator=stamp)
    shard = write_run(tmp_path / "shard0", SHARD_FIXTURES,
                      eval_mutator=mutate)
    row = {result.artefact: result
           for result in compare_fixture(load_run(tmp_path / "serial"),
                                         load_run(shard), "1000002",
                                         shard_index=0)}["eval_record"]
    assert row.status == "FAIL"
    assert row.differences == ("competition_cluster_id",)
    assert row.diagnosis is not None


def test_first_difference_honours_exclusions():
    left = {"as_of": {"matches_advanced": 3, "date": "2025-09-10"}}
    right = {"as_of": {"matches_advanced": 7, "date": "2025-09-10"}}
    assert first_difference(left, right) == "as_of.matches_advanced"
    assert first_difference(left, right,
                            exclude={"as_of.matches_advanced"}) is None


# --------------------------------------------------------------------------
# the report and its metric guard (check 10.6)
# --------------------------------------------------------------------------


def _report(results=(), identity_rows=(), problems=(), merge_rows=()):
    return build_report(cases=CASE_SET, n_shards=3, n_sims=20,
                        base_seed=20260910, specs=[], runs=[],
                        results=list(results),
                        identity_rows=list(identity_rows),
                        decomposition_problems=list(problems),
                        merge_rows=list(merge_rows))


def test_runs_record_keeps_wall_times_across_a_skip(tmp_path):
    """A compare-only rerun must not report every run as taking no time."""
    record = tmp_path / "runs_record.json"
    mod._write_runs_record(record, [
        mod.RunResult(label="serial/A", arm="A", shard=None,
                      output_dir=str(tmp_path / "serial" / "A"),
                      exit_code=0, wall_seconds=10.2)])
    output = tmp_path / "serial" / "A"
    write_run(output, SERIAL_FIXTURES)
    spec = mod.RunSpec(label="serial/A", arm="A", shard=None,
                       fixture_dir=tmp_path, output_dir=output,
                       argv=["/bin/echo"])
    results = mod.launch([spec], record_path=record)
    assert results[0].skipped is True
    assert results[0].wall_seconds == 10.2


def test_report_has_a_case_table_and_a_verdict():
    payload = _report()
    assert payload["summary"]["verdict"] == "PASS"
    assert len(payload["cases"]) == len(CASE_SET)
    assert payload["cases"][0]["cricsheet_id"] == ordered_cases(
        CASE_SET)[0].cricsheet_id
    assert payload["same_day_pairs_split"]
    markdown = render_markdown(payload)
    assert "# Stage 1 step 1c" in markdown
    assert "1512766" in markdown


def test_report_carries_no_metric_value():
    payload = _report()
    text = json.dumps(payload)
    for banned in ("avg_log_loss", "brier_score", "roi_pct"):
        assert banned not in text
    # and the guard itself refuses one
    with pytest.raises(ConsistencyError):
        mod.assert_no_metric_values({"avg_log_loss": 0.62})
    with pytest.raises(ConsistencyError):
        mod.assert_no_metric_values({"results": [{"flat_betting_roi_pct": 3.0}]})


def test_report_counts_failures():
    failing = mod.ArtefactResult(ARM, "1000002", "eval_record", "FAIL",
                                 "first difference: log_loss", 0)
    payload = _report(results=[failing])
    assert payload["summary"]["verdict"] == "FAIL"
    assert payload["summary"]["n_fail"] == 1
