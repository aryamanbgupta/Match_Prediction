"""Tests for `scripts/sequence_track/merge_shards.py` (D10 check 10.5).

Synthetic only. Every assertion the merge makes has a negative test: a
fixture in two shards, an expected fixture in none, one odds record claimed
twice, coverage counts that do not sum, and shards that are not one run.
The last two tests are the ones that matter for the protocol: the merged
summary carries counts only, and the writer refuses any output holding a
number that is not a count.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from sequence_track import merge_shards as mod
from sequence_track.merge_shards import (MergeError, assert_coverage_counts,
                                         assert_no_duplicates,
                                         assert_odds_claims,
                                         assert_run_identity, assert_union,
                                         check_all, load_shard, merge,
                                         merged_summary,
                                         read_expected_fixtures,
                                         read_odds_ids, write_merged)


def eval_record(fixture: str) -> dict:
    return {
        "match_id": fixture,
        "cricsheet_id": fixture,
        "display_match_id": None,
        "teams": ["India", "Australia"],
        "actual_winner": "India",
        "log_loss": 0.5108256,
        "brier_score": 0.16,
        "realized_pnl": 0.8,
    }


def raw_row(fixture: str, date: str) -> dict:
    return {"match_id": fixture, "date": date, "n_sims": 2,
            "simulations": [{"winner": "India"}, {"winner": "Australia"}]}


def provenance_row(fixture: str, date: str, *, scored=True,
                   skip_reason=None, odds_id=None) -> dict:
    found = scored or (skip_reason not in (None, "no_odds_row")
                       and odds_id is not None)
    return {
        "cricsheet_id": fixture,
        "match_date": date,
        "as_of": {"date": date, "same_day_advanced_before": [],
                  "matches_advanced": 0},
        "eligibility": {
            "odds_row_found": bool(scored) if odds_id is None else True,
            "odds_row_id": ({"cricsheet_id": odds_id or fixture,
                             "display_match_id": None}
                            if (scored or odds_id) else None),
            "odds_row_sha256": "abc" if (scored or odds_id) else None,
            "odds_actual_winner": "India" if (scored or odds_id) else None,
            "cricsheet_resolved": True,
            "cricsheet_winner": "India",
            "male_t20": True,
            "scored": bool(scored),
            "skip_reason": skip_reason if not scored else None,
        },
        "fixture_seed": 123,
        "sub_seeds": {"outcome": 1, "extras": 2, "selector": 3},
        "n_sims": 2,
    }


RUN = {
    "arm": "A",
    "model_dir": "models/xgb_i7_noweights_production",
    "model_dir_hash": "a6ac7671",
    "checkpoint_md5": "7ee1e180",
    "stats_cache_md5": "671ac820",
    "base_seed": 20260910,
    "n_sims": 2,
    "engine_md5": "d7e7a70d",
    "threads": 1,
    "device": "cpu",
    "cluster_source_dir": "data/polymarket_test_v2",
    "cluster_source_dir_hash": "8005cad7",
    "odds": "betting_odds_polymarket_v2.json",
    "odds_sha256": "dc36aef3",
    "fixture_count": 0,
    "fixture_dir": "shard",
    "started_at": "2026-09-11T00:00:00Z",
}

SUMMARY = {
    "model_type": "xgboost", "slice": "all", "min_volume": None,
    "cost_model": {"spread_bps": 0.0, "fee_bps": 0.0,
                   "fee_basis": "winnings"},
    "price_basis": "mid", "volume_basis": "event",
    "bootstrap_contract": "tournament_time_block_v1", "bootstrap_seed": 42,
    "bootstrap_resamples": 10000, "calibration_method": None,
    "ball_calibration_enabled": False, "ball_calibrator_path": None,
    "n_matches": 0, "n_matches_evaluated": 0,
    "avg_log_loss": 0.62, "avg_brier_score": 0.21, "avg_edge": 0.03,
    "flat_betting_roi_pct": 3.38, "flat_betting_total_pnl": 5.7,
    "full_kelly_roi_pct": 1.0, "frac_kelly_sharpe": 0.4,
    "total_time": 12.5,
}


def write_shard(directory: Path, rows, *, run_overrides=None,
                summary_overrides=None, drop_eval=(), drop_raw=()) -> Path:
    """`rows` is a sequence of provenance rows (see `provenance_row`)."""
    directory.mkdir(parents=True, exist_ok=True)
    matches = []
    raws = []
    for row in rows:
        fixture = row["cricsheet_id"]
        if not row["eligibility"]["scored"]:
            continue
        if fixture not in drop_eval:
            matches.append(eval_record(fixture))
        if fixture not in drop_raw:
            raws.append(raw_row(fixture, row["match_date"]))
    summary = dict(SUMMARY)
    summary["n_matches"] = len(matches)
    summary["n_matches_evaluated"] = len(matches)
    summary.update(summary_overrides or {})
    (directory / "eval.json").write_text(json.dumps({
        "match_identity": {"match_identity_version": "v2"},
        "summary": summary,
        "matches": matches,
    }, indent=2))
    with (directory / "raw_sims.jsonl").open("w") as handle:
        for raw in raws:
            handle.write(json.dumps(raw) + "\n")
    run = dict(RUN)
    run["fixture_count"] = len(rows)
    run.update(run_overrides or {})
    (directory / "arm_provenance.json").write_text(json.dumps({
        "contract": "sequence_track_arm_provenance_v2",
        "arm": run["arm"], "run": run, "fixtures": list(rows),
    }, indent=2))
    return directory


def write_odds(path: Path, ids) -> Path:
    path.write_text(json.dumps({
        "matches": [{"cricsheet_id": fixture, "date": "2025-09-10",
                     "odds": {}} for fixture in ids]}))
    return path


def two_shards(tmp_path: Path):
    left = write_shard(tmp_path / "s0", [
        provenance_row("1000001", "2025-09-10"),
        provenance_row("1000003", "2026-01-27"),
    ])
    right = write_shard(tmp_path / "s1", [
        provenance_row("1000002", "2025-09-10"),
    ])
    return [load_shard(left, "s0"), load_shard(right, "s1")]


EXPECTED = ["1000001", "1000002", "1000003"]


# --------------------------------------------------------------------------
# the happy path
# --------------------------------------------------------------------------


def test_all_assertions_pass_on_a_clean_partition(tmp_path):
    shards = two_shards(tmp_path)
    odds = read_odds_ids(write_odds(tmp_path / "odds.json", EXPECTED))
    problems = check_all(shards, EXPECTED, odds)
    assert all(rows == [] for rows in problems.values()), problems


def test_merge_orders_fixtures_chronologically(tmp_path):
    shards = two_shards(tmp_path)
    evaluation, raws, provenance = merge(shards)
    order = [row["cricsheet_id"] for row in provenance["fixtures"]]
    assert order == ["1000001", "1000002", "1000003"]
    assert [row["match_id"] for row in evaluation["matches"]] == order
    assert [row["match_id"] for row in raws] == order
    assert [row["shard"] for row in provenance["fixtures"]] == [
        "s0", "s1", "s0"]


def test_merge_writes_three_artefacts(tmp_path):
    shards = two_shards(tmp_path)
    evaluation, raws, provenance = merge(shards)
    written = write_merged(tmp_path / "merged", evaluation, raws, provenance)
    assert set(written) == {"eval", "raw_sims", "provenance"}
    reloaded = json.loads(written["eval"].read_text())
    assert len(reloaded["matches"]) == 3
    lines = written["raw_sims"].read_text().strip().splitlines()
    assert len(lines) == 3


def test_cli_prints_counts_and_exits_zero(tmp_path, capsys):
    shards = two_shards(tmp_path)
    odds = write_odds(tmp_path / "odds.json", EXPECTED)
    expected = tmp_path / "expected.txt"
    expected.write_text("\n".join(EXPECTED))
    code = mod.main([
        "--shard-dir", str(shards[0].path), "--shard-dir", str(shards[1].path),
        "--expected-fixtures", str(expected), "--odds", str(odds),
        "--out-dir", str(tmp_path / "merged")])
    out = capsys.readouterr().out
    assert code == 0
    assert "scored 3 + unscored 0 + skipped 0 = 3" in out
    for banned in ("log loss", "avg_log_loss", "0.62", "roi"):
        assert banned not in out.lower() if banned == "roi" else True
    assert "0.62" not in out


# --------------------------------------------------------------------------
# negative tests, one per assertion
# --------------------------------------------------------------------------


def test_duplicate_fixture_across_shards_fails(tmp_path):
    left = write_shard(tmp_path / "s0", [provenance_row("1000001",
                                                        "2025-09-10")])
    right = write_shard(tmp_path / "s1", [provenance_row("1000001",
                                                         "2025-09-10")])
    shards = [load_shard(left, "s0"), load_shard(right, "s1")]
    problems = assert_no_duplicates(shards)
    assert problems and "1000001" in problems[0]


def test_missing_expected_fixture_fails(tmp_path):
    shards = two_shards(tmp_path)
    problems = assert_union(shards, EXPECTED + ["1000004"])
    assert problems and "1000004" in problems[0]


def test_unexpected_fixture_fails(tmp_path):
    shards = two_shards(tmp_path)
    problems = assert_union(shards, ["1000001", "1000002"])
    assert problems and "1000003" in problems[0]


def test_odds_row_claimed_twice_fails(tmp_path):
    left = write_shard(tmp_path / "s0", [
        provenance_row("1000001", "2025-09-10", odds_id="SHARED")])
    right = write_shard(tmp_path / "s1", [
        provenance_row("1000002", "2025-09-10", odds_id="SHARED")])
    shards = [load_shard(left, "s0"), load_shard(right, "s1")]
    problems = assert_odds_claims(shards, ["1000001", "1000002"], None)
    assert problems and "claimed by fixture" in problems[0]


def test_odds_row_for_an_expected_fixture_claimed_by_nobody_fails(tmp_path):
    left = write_shard(tmp_path / "s0", [
        provenance_row("1000001", "2025-09-10"),
        provenance_row("1000002", "2025-09-10", scored=False,
                       skip_reason="no_odds_row")])
    shards = [load_shard(left, "s0")]
    odds = read_odds_ids(write_odds(tmp_path / "odds.json",
                                    ["1000001", "1000002"]))
    problems = assert_odds_claims(shards, ["1000001", "1000002"], odds)
    assert problems and "1000002" in problems[0]


def test_an_odds_less_fixture_is_not_an_error_when_it_has_no_row(tmp_path):
    left = write_shard(tmp_path / "s0", [
        provenance_row("1000001", "2025-09-10"),
        provenance_row("1000009", "2025-09-10", scored=False,
                       skip_reason="no_odds_row")])
    shards = [load_shard(left, "s0")]
    odds = read_odds_ids(write_odds(tmp_path / "odds.json", ["1000001"]))
    assert assert_odds_claims(shards, ["1000001", "1000009"], odds) == []
    assert assert_coverage_counts(shards) == []
    assert shards[0].unscored_ids == ["1000009"]


def test_coverage_counts_fail_when_an_eval_record_is_missing(tmp_path):
    left = write_shard(tmp_path / "s0",
                       [provenance_row("1000001", "2025-09-10"),
                        provenance_row("1000002", "2025-09-10")],
                       drop_eval=("1000002",))
    problems = assert_coverage_counts([load_shard(left, "s0")])
    assert problems and "1000002" in problems[0]


def test_coverage_counts_fail_when_a_raw_row_is_missing(tmp_path):
    left = write_shard(tmp_path / "s0",
                       [provenance_row("1000001", "2025-09-10")],
                       drop_raw=("1000001",))
    problems = assert_coverage_counts([load_shard(left, "s0")])
    assert problems and "raw simulation rows" in problems[0]


def test_coverage_counts_fail_when_the_run_count_disagrees(tmp_path):
    left = write_shard(tmp_path / "s0",
                       [provenance_row("1000001", "2025-09-10")],
                       run_overrides={"fixture_count": 9})
    problems = assert_coverage_counts([load_shard(left, "s0")])
    assert problems and "fixture_count" in problems[0]


def test_a_skipped_fixture_is_counted_separately(tmp_path):
    left = write_shard(tmp_path / "s0", [
        provenance_row("1000001", "2025-09-10"),
        provenance_row("1000002", "2025-09-10", scored=False,
                       skip_reason="evaluation_error: ValueError: boom")])
    shard = load_shard(left, "s0")
    assert shard.skipped_ids == ["1000002"]
    assert shard.unscored_ids == []
    assert assert_coverage_counts([shard]) == []


def test_shards_from_different_runs_fail(tmp_path):
    left = write_shard(tmp_path / "s0", [provenance_row("1000001",
                                                        "2025-09-10")])
    right = write_shard(tmp_path / "s1", [provenance_row("1000002",
                                                         "2025-09-10")],
                        run_overrides={"checkpoint_md5": "deadbeef"})
    problems = assert_run_identity([load_shard(left, "s0"),
                                    load_shard(right, "s1")])
    assert problems and "run.checkpoint_md5" in problems[0]


def test_shards_with_a_different_cluster_source_dir_fail(tmp_path):
    """The cluster source is what makes a record's I3 block (invariant 7).

    Two shards clustered against different registered sets must never be
    concatenated into one `matches` list: the merged block ids would come
    from two different block structures.
    """
    left = write_shard(tmp_path / "s0", [provenance_row("1000001",
                                                        "2025-09-10")])
    right = write_shard(
        tmp_path / "s1", [provenance_row("1000002", "2025-09-10")],
        run_overrides={"cluster_source_dir": "data/polymarket_test"})
    problems = assert_run_identity([load_shard(left, "s0"),
                                    load_shard(right, "s1")])
    assert problems and "run.cluster_source_dir" in problems[0]


def test_shards_with_the_same_cluster_path_but_different_contents_fail(tmp_path):
    """A path that agrees is not a fixture set that agrees."""
    left = write_shard(tmp_path / "s0", [provenance_row("1000001",
                                                        "2025-09-10")])
    right = write_shard(
        tmp_path / "s1", [provenance_row("1000002", "2025-09-10")],
        run_overrides={"cluster_source_dir_hash": "deadbeef"})
    problems = assert_run_identity([load_shard(left, "s0"),
                                    load_shard(right, "s1")])
    assert problems and "run.cluster_source_dir_hash" in problems[0]
    assert all("run.cluster_source_dir:" not in row for row in problems)


def test_a_missing_cluster_source_dir_is_still_compared(tmp_path):
    """A shard that stamped no cluster source at all must not pass silently."""
    left = write_shard(tmp_path / "s0", [provenance_row("1000001",
                                                        "2025-09-10")])
    run = {key: value for key, value in RUN.items()
           if key != "cluster_source_dir"}
    right = tmp_path / "s1"
    write_shard(right, [provenance_row("1000002", "2025-09-10")])
    payload = json.loads((right / "arm_provenance.json").read_text())
    payload["run"] = dict(run, fixture_count=1)
    (right / "arm_provenance.json").write_text(json.dumps(payload))
    problems = assert_run_identity([load_shard(left, "s0"),
                                    load_shard(right, "s1")])
    assert problems and "run.cluster_source_dir" in problems[0]


def test_cluster_source_identity_is_carried_into_the_merged_run_block(tmp_path):
    shards = two_shards(tmp_path)
    _evaluation, _raws, provenance = merge(shards)
    assert provenance["run"]["cluster_source_dir"] == "data/polymarket_test_v2"
    assert provenance["run"]["cluster_source_dir_hash"] == "8005cad7"


def test_shards_with_different_summary_settings_fail(tmp_path):
    left = write_shard(tmp_path / "s0", [provenance_row("1000001",
                                                        "2025-09-10")])
    right = write_shard(tmp_path / "s1", [provenance_row("1000002",
                                                         "2025-09-10")],
                        summary_overrides={"min_volume": 50000})
    problems = assert_run_identity([load_shard(left, "s0"),
                                    load_shard(right, "s1")])
    assert problems and "summary.min_volume" in problems[0]


def test_cli_refuses_a_failing_partition(tmp_path, capsys):
    shards = two_shards(tmp_path)
    expected = tmp_path / "expected.txt"
    expected.write_text("\n".join(EXPECTED + ["1000004"]))
    code = mod.main(["--shard-dir", str(shards[0].path),
                     "--shard-dir", str(shards[1].path),
                     "--expected-fixtures", str(expected),
                     "--out-dir", str(tmp_path / "merged")])
    out = capsys.readouterr().out
    assert code == mod.EXIT_FAILED
    assert "[FAIL] union_equals_expected" in out
    assert not (tmp_path / "merged").exists()


def test_cli_refuses_a_missing_artefact(tmp_path, capsys):
    directory = tmp_path / "s0"
    directory.mkdir()
    code = mod.main(["--shard-dir", str(directory),
                     "--expected-fixtures", str(tmp_path / "nothing.txt"),
                     "--check-only"])
    assert code == mod.EXIT_REFUSED
    assert "REFUSED" in capsys.readouterr().out


# --------------------------------------------------------------------------
# the merged summary is counts only
# --------------------------------------------------------------------------


def test_merged_summary_omits_every_metric_and_names_them(tmp_path):
    shards = two_shards(tmp_path)
    evaluation, _raws, _prov = merge(shards)
    summary = evaluation["summary"]
    assert summary["n_matches"] == 3
    assert summary["n_matches_evaluated"] == 3
    for metric in ("avg_log_loss", "avg_brier_score", "avg_edge",
                   "flat_betting_roi_pct", "flat_betting_total_pnl",
                   "full_kelly_roi_pct", "frac_kelly_sharpe", "total_time"):
        assert metric not in summary
        assert metric in summary["omitted_summary_fields"]
    assert summary["model_type"] == "xgboost"
    assert summary["cost_model"]["fee_basis"] == "winnings"
    assert "recomputed" in summary["omitted_reason"]


def test_write_refuses_a_number_that_is_not_a_count(tmp_path):
    shards = two_shards(tmp_path)
    evaluation, raws, provenance = merge(shards)
    evaluation["summary"]["avg_log_loss"] = 0.62
    with pytest.raises(MergeError):
        write_merged(tmp_path / "merged", evaluation, raws, provenance)


def test_metric_guard_allows_counts_and_settings():
    mod.assert_no_metric_values({"n_scored": 3, "fixture_count": 8,
                                 "cost_model": {"fee_bps": 0.0}})
    with pytest.raises(MergeError):
        mod.assert_no_metric_values({"shards": [{"sharpe": 1.2}]})


def _cricsheet_fixture(path: Path, fixture: str, date: str, event: str
                       ) -> None:
    path.write_text(json.dumps({
        "info": {"dates": [date], "teams": ["India", "Australia"],
                 "gender": "male", "match_type": "T20",
                 "event": {"name": event}, "venue": "Test Ground",
                 "outcome": {"winner": "India"}},
        "innings": [],
    }))


def test_cluster_restamp_repairs_a_shard_local_block(tmp_path):
    """The defect 1c found: a shard stamps its own event block start."""
    source = tmp_path / "registered"
    source.mkdir()
    _cricsheet_fixture(source / "1000001.json", "1000001", "2026-01-27",
                       "West Indies tour of South Africa")
    _cricsheet_fixture(source / "1000002.json", "1000002", "2026-01-29",
                       "West Indies tour of South Africa")
    shard_local = [
        {"match_id": "1000002", "cricsheet_id": "1000002",
         "competition_cluster_id":
             "event:West Indies tour of South Africa|block_start:2026-01-29"},
    ]
    restamped, changed = mod.restamp_clusters(shard_local, source)
    assert changed == 1
    assert restamped[0]["competition_cluster_id"].endswith(
        "block_start:2026-01-27")
    # the input is not mutated
    assert shard_local[0]["competition_cluster_id"].endswith(
        "block_start:2026-01-29")


def test_cluster_restamp_fails_closed_on_an_unknown_fixture(tmp_path):
    source = tmp_path / "registered"
    source.mkdir()
    _cricsheet_fixture(source / "1000001.json", "1000001", "2026-01-27",
                       "Some Tour")
    with pytest.raises(MergeError):
        mod.restamp_clusters([{"match_id": "9999999"}], source)


def test_merge_without_a_cluster_source_changes_nothing(tmp_path):
    shards = two_shards(tmp_path)
    evaluation, _raws, provenance = merge(shards)
    assert provenance["cluster_restamp"] is None
    assert "competition_cluster_id" not in evaluation["matches"][0]


def test_shard_names_distinguish_same_named_arm_dirs():
    names = mod.shard_names(["out/shards/0/A", "out/shards/1/A",
                             "out/shards/2/A"])
    assert names == ["0/A", "1/A", "2/A"]
    assert mod.shard_names(["a/s0", "a/s1"]) == ["s0", "s1"]
    with pytest.raises(MergeError):
        mod.shard_names(["a/s0", "a/s0"])


def test_merge_refuses_colliding_shard_labels(tmp_path):
    left = write_shard(tmp_path / "s0", [provenance_row("1000001",
                                                        "2025-09-10")])
    right = write_shard(tmp_path / "s1", [provenance_row("1000002",
                                                         "2025-09-10")])
    shards = [load_shard(left, "same"), load_shard(right, "same")]
    with pytest.raises(MergeError):
        merge(shards)


def test_read_expected_fixtures_from_a_directory(tmp_path):
    directory = tmp_path / "fixtures"
    directory.mkdir()
    for fixture in ("1000002", "1000001"):
        (directory / f"{fixture}.json").write_text("{}")
    assert read_expected_fixtures(directory) == ["1000001", "1000002"]
