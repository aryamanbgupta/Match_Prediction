"""Tests for the stage-1 realism scorer (D7 check 7.4).

Synthetic `raw_sims.jsonl` plus a synthetic cricsheet fixture: no model,
no stats cache, no simulation, no artifact of record. The arithmetic
checks pin the two numbers the check names — first-innings bias and the
P10–P90 coverage indicator — on values chosen so the answers are exact.
"""
from __future__ import annotations

import json

import pytest

from cricsheet_fixtures import TEAMS, match_json
from sequence_track import score_realism
from t1_ppc_common import actual_innings

PHASE_RUNS = {"powerplay": 40, "middle": 50, "death": 30}


def _simulation(first_team, second_team, totals, *, winner="__first__",
                wickets=(6, 4), extras=(5, 3), bowlers=(6, 5),
                with_winner=True):
    innings = []
    for index, (team, total) in enumerate(
            ((first_team, totals[0]), (second_team, totals[1]))):
        batting = TEAMS[team][:4]
        bowling = TEAMS[second_team if index == 0 else first_team][7:11]
        innings.append({
            "batting_team": team,
            "total_runs": int(total),
            "wickets": int(wickets[index]),
            "legal_balls": 120,
            "deliveries": 120 + int(extras[index]),
            "extras_events": int(extras[index]),
            "phase_runs": dict(PHASE_RUNS),
            "unique_bowlers": int(bowlers[index]),
            "batter_runs": {player_id: 10 + position
                            for position, (player_id, _) in enumerate(batting)},
            "bowler_wickets": {player_id: position % 2
                               for position, (player_id, _) in enumerate(bowling)},
        })
    record = {"innings": innings, "tie": False,
              "scores": {first_team: totals[0], second_team: totals[1]},
              "wickets": {first_team: wickets[0], second_team: wickets[1]}}
    if with_winner:
        record["winner"] = (first_team if winner == "__first__" else winner)
    return record


def _write_run(tmp_path, plans, *, arm="A", provenance=True):
    """Build a fixture dir + raw_sims.jsonl from per-fixture plans.

    Each plan is (match_id, seed, offsets, kwargs) where `offsets` are the
    per-simulation first/second innings totals relative to the actual.
    """
    fixture_dir = tmp_path / "fixtures"
    fixture_dir.mkdir()
    arm_dir = tmp_path / arm
    arm_dir.mkdir()
    lines = []
    for match_id, seed, offsets, kwargs in plans:
        document = match_json("2026-01-01", seed=seed, match_id=match_id)
        if kwargs.pop("no_result", False):
            document["info"]["outcome"] = {"result": "no result"}
        (fixture_dir / f"{match_id}.json").write_text(json.dumps(document))
        observed = actual_innings(document)
        first_team = document["innings"][0]["team"]
        second_team = document["innings"][1]["team"]
        simulations = [
            _simulation(
                first_team, second_team,
                (observed[0]["total_runs"] + first_offset,
                 observed[1]["total_runs"] + second_offset),
                **kwargs)
            for first_offset, second_offset in offsets
        ]
        lines.append(json.dumps({
            "match_id": match_id,
            "date": "2026-01-01",
            "first_batting_team": first_team,
            "actual_winner": (document["info"].get("outcome") or {}).get(
                "winner"),
            "actual_innings": observed,
            "fixture_seed": 1234,
            "n_sims": len(simulations),
            "simulations": simulations,
        }))
    (arm_dir / score_realism.RAW_SIMS_FILENAME).write_text(
        "\n".join(lines) + "\n")
    if provenance:
        (arm_dir / score_realism.PROVENANCE_FILENAME).write_text(json.dumps({
            "contract": "sequence_track_arm_provenance_v1",
            "arm": arm,
            "run": {"arm": arm, "fixture_dir": str(fixture_dir),
                    "model_dir_hash": "deadbeef", "stats_version": "i7",
                    "base_seed": 20260910, "n_sims": len(plans)},
            "fixtures": [],
        }))
    return arm_dir, fixture_dir


# Fixture 1: sims sit 10 and 20 above the actual  -> bias +15, outside the
# P10-P90 band. Fixture 2: sims straddle the actual -> bias 0, covered.
BIAS_PLANS = [
    ("2000001", 0, [(10, 4), (20, 8)], {}),
    ("2000002", 2, [(-10, -6), (0, 0), (10, 6)], {}),
]


def test_top_level_and_aggregate_keys(tmp_path):
    arm_dir, fixture_dir = _write_run(tmp_path, BIAS_PLANS)
    score_realism.main(["--arm-dir", str(arm_dir)])
    payload = json.loads(
        (arm_dir / score_realism.DEFAULT_OUT_NAME).read_text())

    assert payload["contract"] == score_realism.CONTRACT
    for key in ("arm", "arm_dir", "raw_sims", "fixture_dir", "bootstrap",
                "n_fixtures_in_raw_sims", "n_fixtures_scored",
                "excluded_fixtures", "aggregate", "fixtures"):
        assert key in payload, key
    assert payload["fixture_dir"] == str(fixture_dir)
    assert payload["bootstrap"] == {
        "reps": 2000, "seed": 29,
        "generator": "numpy.random.default_rng", "resample": "by fixture"}

    aggregate = payload["aggregate"]
    for key in ("n_fixtures", "first_innings", "second_innings",
                "second_innings_completed_chases",
                "extras_events_pooled_over_innings", "batting_order"):
        assert key in aggregate, key
    for key in ("bias", "abs_bias", "coverage_p10_p90", "quantile_means",
                "fields"):
        assert key in aggregate["first_innings"], key
    for field in ("total_runs", "wickets", "extras_events", "unique_bowlers",
                  "legal_balls", "deliveries"):
        assert field in aggregate["first_innings"]["fields"], field
        assert field in aggregate["second_innings"]["fields"], field
        block = aggregate["second_innings"]["fields"][field]
        assert set(block) == {"actual_mean", "sim_posterior_mean", "bias"}

    row = payload["fixtures"][0]
    for key in ("match_id", "n_sims", "first_batting_team", "actual_winner",
                "chase_successful", "sim_first_batting_win_prob",
                "innings_1", "innings_2"):
        assert key in row, key
    assert set(row["innings_1"]) == {
        "sim_p10", "sim_p50", "sim_p90", "covered_p10_p90", "fields"}


def test_first_innings_bias_and_coverage_arithmetic(tmp_path):
    arm_dir, _ = _write_run(tmp_path, BIAS_PLANS)
    payload = score_realism.score(
        score_realism.load_jsonl(
            arm_dir / score_realism.RAW_SIMS_FILENAME),
        tmp_path / "fixtures")

    first, second = payload["fixtures"]
    # +10 and +20 above the actual: mean bias +15, band [+11, +19].
    assert first["innings_1"]["fields"]["total_runs"]["bias"] == 15.0
    assert first["innings_1"]["covered_p10_p90"] == 0
    actual_first = first["innings_1"]["fields"]["total_runs"]["actual"]
    assert first["innings_1"]["sim_p10"] == actual_first + 11.0
    assert first["innings_1"]["sim_p50"] == actual_first + 15.0
    assert first["innings_1"]["sim_p90"] == actual_first + 19.0
    # -10 / 0 / +10 around the actual: bias 0, band [-8, +8], covered.
    assert second["innings_1"]["fields"]["total_runs"]["bias"] == 0.0
    assert second["innings_1"]["covered_p10_p90"] == 1
    # Second innings offsets +4/+8 then -6/0/+6.
    assert first["innings_2"]["fields"]["total_runs"]["bias"] == 6.0
    assert second["innings_2"]["fields"]["total_runs"]["bias"] == 0.0

    aggregate = payload["aggregate"]
    assert aggregate["n_fixtures"] == 2
    assert aggregate["first_innings"]["bias"]["mean"] == 7.5
    assert aggregate["first_innings"]["abs_bias"]["mean"] == 7.5
    assert aggregate["first_innings"]["coverage_p10_p90"]["mean"] == 0.5
    low, high = aggregate["first_innings"]["bias"]["ci95"]
    assert low <= 7.5 <= high
    # Extras/wickets/unique bowlers are constants in the synthetic sims.
    fields = aggregate["first_innings"]["fields"]
    assert fields["extras_events"]["sim_posterior_mean"] == 5.0
    assert fields["wickets"]["sim_posterior_mean"] == 6.0
    assert fields["unique_bowlers"]["sim_posterior_mean"] == 6.0
    assert fields["wickets"]["bias"]["mean"] == pytest.approx(
        6.0 - fields["wickets"]["actual_mean"])


def test_second_innings_chase_split_and_batting_order(tmp_path):
    arm_dir, _ = _write_run(tmp_path, BIAS_PLANS)
    payload = score_realism.score(
        score_realism.load_jsonl(
            arm_dir / score_realism.RAW_SIMS_FILENAME),
        tmp_path / "fixtures")
    chases = payload["aggregate"]["second_innings_completed_chases"]
    assert chases["all_completed"]["n"] == 2
    for key in ("chase_successful", "chase_unsuccessful"):
        assert key in chases
    counted = sum(block["n"] for key, block in chases.items()
                  if key in ("chase_successful", "chase_unsuccessful")
                  and block)
    assert counted == 2

    order = payload["aggregate"]["batting_order"]
    # Every synthetic simulation is won by the batting-first side.
    assert order["simulations"]["pooled_first_batting_win_rate"] == 1.0
    assert order["simulations"]["pooled_chasing_win_rate"] == 0.0
    assert order["simulations"]["pooled_first_batting_minus_chasing"] == 1.0
    assert order["simulations"]["per_fixture_first_batting_win_rate"][
        "mean"] == 1.0
    assert order["actual_reference"]["n_decided_matches"] == 2


def test_batting_order_omitted_when_sims_carry_no_winner(tmp_path):
    plans = [(match_id, seed, offsets, {"with_winner": False})
             for match_id, seed, offsets, _ in BIAS_PLANS]
    arm_dir, _ = _write_run(tmp_path, plans)
    payload = score_realism.score(
        score_realism.load_jsonl(
            arm_dir / score_realism.RAW_SIMS_FILENAME),
        tmp_path / "fixtures")
    assert payload["aggregate"]["batting_order"] is None
    assert "omitted" in payload["aggregate"]["batting_order_note"]
    assert payload["fixtures"][0]["sim_first_batting_win_prob"] is None
    # The innings realism numbers are unaffected.
    assert payload["aggregate"]["first_innings"]["bias"]["mean"] == 7.5


def test_no_result_fixture_is_excluded_not_scored(tmp_path):
    plans = list(BIAS_PLANS) + [("2000003", 4, [(0, 0)], {"no_result": True})]
    arm_dir, _ = _write_run(tmp_path, plans)
    payload = score_realism.score(
        score_realism.load_jsonl(
            arm_dir / score_realism.RAW_SIMS_FILENAME),
        tmp_path / "fixtures")
    assert payload["n_fixtures_in_raw_sims"] == 3
    assert payload["n_fixtures_scored"] == 2
    assert payload["excluded_fixtures"] == [
        {"match_id": "2000003", "reason": "outcome.result=no result"}]


def test_mismatched_fixture_dir_fails_closed(tmp_path):
    arm_dir, fixture_dir = _write_run(tmp_path, BIAS_PLANS)
    document = json.loads((fixture_dir / "2000001.json").read_text())
    document["innings"][0]["overs"][0]["deliveries"][0]["runs"] = {
        "batter": 6, "extras": 0, "total": 6}
    (fixture_dir / "2000001.json").write_text(json.dumps(document))
    with pytest.raises(SystemExit) as excinfo:
        score_realism.score(
            score_realism.load_jsonl(
                arm_dir / score_realism.RAW_SIMS_FILENAME),
            fixture_dir)
    assert "does not match the run" in str(excinfo.value)


@pytest.mark.parametrize("sealed", ["data/golden/polymarket_test_v2",
                                    "data/forward_holdout/2026-06-01"])
def test_sealed_paths_are_refused_before_anything_is_opened(tmp_path, sealed):
    arm_dir, fixture_dir = _write_run(tmp_path, BIAS_PLANS)

    # Sealed --fixture-dir: refused before the existence check, so the
    # message is the guard's, not "fixture dir not found".
    with pytest.raises(SystemExit) as excinfo:
        score_realism.main(["--arm-dir", str(arm_dir),
                            "--fixture-dir", sealed])
    assert str(excinfo.value) == f"refusing sealed path: {sealed}"

    # Sealed arm dir: refused before the raw-sims existence check.
    with pytest.raises(SystemExit) as excinfo:
        score_realism.main(["--arm-dir", sealed,
                            "--fixture-dir", str(fixture_dir)])
    assert str(excinfo.value) == f"refusing sealed path: {sealed}"

    # Sealed --out: refused before the run is scored.
    with pytest.raises(SystemExit) as excinfo:
        score_realism.main(["--arm-dir", str(arm_dir),
                            "--out", f"{sealed}/realism.json"])
    assert "refusing sealed path" in str(excinfo.value)

    # Sealed fixture dir recorded in provenance: refused on resolution.
    provenance = json.loads(
        (arm_dir / score_realism.PROVENANCE_FILENAME).read_text())
    provenance["run"]["fixture_dir"] = sealed
    (arm_dir / score_realism.PROVENANCE_FILENAME).write_text(
        json.dumps(provenance))
    with pytest.raises(SystemExit) as excinfo:
        score_realism.main(["--arm-dir", str(arm_dir)])
    assert str(excinfo.value) == f"refusing sealed path: {sealed}"

    # And the library entry points guard themselves.
    with pytest.raises(SystemExit):
        score_realism.load_jsonl(f"{sealed}/raw_sims.jsonl")
    with pytest.raises(SystemExit):
        score_realism.score([], sealed)
    assert not (arm_dir / score_realism.DEFAULT_OUT_NAME).exists()


def _retarget_match_id(arm_dir, old_id, new_id):
    raw_path = arm_dir / score_realism.RAW_SIMS_FILENAME
    rows = score_realism.load_jsonl(raw_path)
    for row in rows:
        if row["match_id"] == old_id:
            row["match_id"] = new_id
    raw_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


@pytest.mark.parametrize("bad_id", ["../outside", "..%2foutside",
                                    "/etc/hosts", "dir/2000001",
                                    "2000001.json"])
def test_traversal_and_absolute_match_ids_are_refused(tmp_path, bad_id):
    arm_dir, fixture_dir = _write_run(tmp_path, BIAS_PLANS)
    # A readable file the traversal would reach if the id were trusted.
    (tmp_path / "outside.json").write_text(
        (fixture_dir / "2000001.json").read_text())
    _retarget_match_id(arm_dir, "2000001", bad_id)
    with pytest.raises(SystemExit) as excinfo:
        score_realism.main(["--arm-dir", str(arm_dir)])
    assert "refusing match id" in str(excinfo.value)
    assert not (arm_dir / score_realism.DEFAULT_OUT_NAME).exists()


def test_symlinked_fixture_into_a_sealed_dir_is_refused(tmp_path):
    arm_dir, fixture_dir = _write_run(tmp_path, BIAS_PLANS)
    sealed_dir = tmp_path / "data" / "golden" / "polymarket_test_v2"
    sealed_dir.mkdir(parents=True)
    target = sealed_dir / "sealed.json"
    target.write_text((fixture_dir / "2000001.json").read_text())
    link = fixture_dir / "2000001.json"
    link.unlink()
    link.symlink_to(target)

    # The directory sweep refuses it (this is also the path
    # `ordinary_test_order_reference` would glob open).
    with pytest.raises(SystemExit) as excinfo:
        score_realism.main(["--arm-dir", str(arm_dir)])
    assert "refusing sealed path" in str(excinfo.value)
    assert not (arm_dir / score_realism.DEFAULT_OUT_NAME).exists()

    # And the per-id resolution refuses it on its own.
    with pytest.raises(SystemExit) as excinfo:
        score_realism.fixture_document_path("2000001", fixture_dir)
    assert "refusing sealed path" in str(excinfo.value)


def test_symlinked_fixture_outside_the_dir_is_refused(tmp_path):
    arm_dir, fixture_dir = _write_run(tmp_path, BIAS_PLANS)
    target = tmp_path / "elsewhere.json"
    target.write_text((fixture_dir / "2000001.json").read_text())
    link = fixture_dir / "2000001.json"
    link.unlink()
    link.symlink_to(target)
    with pytest.raises(SystemExit) as excinfo:
        score_realism.main(["--arm-dir", str(arm_dir)])
    assert "resolves outside" in str(excinfo.value)
    with pytest.raises(SystemExit):
        score_realism.fixture_document_path("2000001", fixture_dir)


def test_missing_provenance_requires_explicit_fixture_dir(tmp_path):
    arm_dir, fixture_dir = _write_run(tmp_path, BIAS_PLANS, provenance=False)
    with pytest.raises(SystemExit) as excinfo:
        score_realism.main(["--arm-dir", str(arm_dir)])
    assert "records none" in str(excinfo.value)
    out = tmp_path / "elsewhere" / "realism.json"
    score_realism.main(["--arm-dir", str(arm_dir),
                        "--fixture-dir", str(fixture_dir),
                        "--out", str(out)])
    assert json.loads(out.read_text())["n_fixtures_scored"] == 2
