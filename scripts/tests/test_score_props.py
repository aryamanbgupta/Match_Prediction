"""Tests for the stage-1 prop scorer (D7 check 7.5).

Synthetic `raw_sims.jsonl`, a synthetic cricsheet fixture and a synthetic
fair-baseline corpus: no model, no simulation, no artifact of record. One
test that resolves the real `prop_fair_baseline_corpus_v2` role is marked
`needs_artifacts`.

The arithmetic checks pin the family Brier scores on values chosen so the
answers are exact, and pin the skip inventory: every family
`prop_backtest.build_observations` defines is either scored or listed
with the per-simulation field `raw_sims.jsonl` does not carry.
"""
from __future__ import annotations

import json

import pytest

from cricsheet_fixtures import TEAMS, match_json
from sequence_track import score_props

CORPUS_KEYS = ("batter", "bowler", "bowling_usage", "venue_inn",
               "venue_match", "pos_top")


def _empty_logs(venue_match_arity=4):
    """A corpus with the right shape and no history at all.

    Every as-of rate then falls back to the global prior over an empty
    log, which is 0.0 — so each family's baseline probability is exactly
    zero and the paired difference is the sim's Brier score.
    """
    logs = {key: ({} if key != "pos_top" else []) for key in CORPUS_KEYS}
    if venue_match_arity:
        row = ("1900-01-01",) + (0,) * (venue_match_arity - 1)
        logs["venue_match"] = {"Parity Park": [row]}
        logs["venue_inn"] = {"Parity Park": [("1900-01-01",) + (0,) * 6]}
    return logs


def _simulation(document, first_total, second_total, powerplay,
                batter_runs, bowler_wickets):
    first_team = document["innings"][0]["team"]
    second_team = document["innings"][1]["team"]
    innings = []
    for index, (team, total) in enumerate(
            ((first_team, first_total), (second_team, second_total))):
        bowling_team = second_team if index == 0 else first_team
        innings.append({
            "batting_team": team,
            "total_runs": int(total),
            "wickets": 6,
            "legal_balls": 120,
            "deliveries": 124,
            "extras_events": 4,
            "phase_runs": {"powerplay": int(powerplay[index]),
                           "middle": 50, "death": 30},
            "unique_bowlers": 4,
            # The canonical fixture's batters are the team's first four
            # names and its bowlers the opponent's 8th-11th.
            "batter_runs": {player_id: int(batter_runs[index])
                            for player_id, _ in TEAMS[team][:4]},
            "bowler_wickets": {player_id: int(bowler_wickets[index])
                               for player_id, _ in TEAMS[bowling_team][7:11]},
        })
    return {"winner": first_team, "tie": False, "innings": innings,
            "scores": {first_team: first_total, second_team: second_total},
            "wickets": {first_team: 6, second_team: 6}}


# Two simulations per fixture: one high, one low, on every quantity, so
# every family's simulated probability is exactly 0.5.
SIM_PLAN = (
    dict(first_total=200, second_total=190, powerplay=(60, 60),
         batter_runs=(60, 60), bowler_wickets=(3, 3)),
    dict(first_total=100, second_total=90, powerplay=(30, 30),
         batter_runs=(10, 10), bowler_wickets=(0, 0)),
)


def _write_run(tmp_path, match_ids=("3000001", "3000002"), *,
               provenance=True):
    fixture_dir = tmp_path / "fixtures"
    fixture_dir.mkdir()
    arm_dir = tmp_path / "A"
    arm_dir.mkdir()
    lines = []
    for seed, match_id in enumerate(match_ids):
        document = match_json("2026-01-01", seed=2 * seed)
        (fixture_dir / f"{match_id}.json").write_text(json.dumps(document))
        simulations = [_simulation(document, **plan) for plan in SIM_PLAN]
        lines.append(json.dumps({
            "match_id": match_id,
            "date": "2026-01-01",
            "first_batting_team": document["innings"][0]["team"],
            "actual_winner": document["info"]["outcome"]["winner"],
            "actual_innings": [],
            "fixture_seed": 4321,
            "n_sims": len(simulations),
            "simulations": simulations,
        }))
    (arm_dir / score_props.RAW_SIMS_FILENAME).write_text(
        "\n".join(lines) + "\n")
    if provenance:
        (arm_dir / score_props.PROVENANCE_FILENAME).write_text(json.dumps({
            "contract": "sequence_track_arm_provenance_v1",
            "arm": "A",
            "run": {"arm": "A", "fixture_dir": str(fixture_dir),
                    "model_dir_hash": "deadbeef", "stats_version": "i7",
                    "base_seed": 20260910, "n_sims": 2},
            "fixtures": [],
        }))
    return arm_dir, fixture_dir


def _built(tmp_path, **kwargs):
    arm_dir, fixture_dir = _write_run(tmp_path, **kwargs)
    raw = score_props.load_jsonl(arm_dir / score_props.RAW_SIMS_FILENAME)
    return arm_dir, fixture_dir, score_props.build_detail(
        raw, fixture_dir, set(score_props.MISSING_FIELD))


def test_family_inventory_is_scored_or_skipped_with_the_missing_field(
        tmp_path):
    _arm_dir, _fixture_dir, built = _built(tmp_path)
    payload = score_props.score(built, _empty_logs())

    universe = set(payload["family_universe"])
    scored = set(payload["families_scored"])
    skipped = {row["family"] for row in payload["families_skipped"]}
    assert scored | skipped == universe
    assert not scored & skipped

    # Everything the raw file cannot support names the missing quantity.
    for row in payload["families_skipped"]:
        if row["reason"] == "input absent from raw_sims.jsonl":
            assert row["missing_field"]
    assert "top_bowler" in skipped
    missing = {row["family"]: row.get("missing_field")
               for row in payload["families_skipped"]}
    assert "runs conceded" in missing["top_bowler"]
    assert "fours" in missing["batter_fours_2plus"]
    assert "sixes" in missing["match_total_sixes_ou_20_5"]
    assert "ball log" in missing["first_wicket_runs_ou_30_5"]

    for family in ("top_batter", "batter_50plus", "batter_runs_mae",
                   "innings_runs_ou_160_5", "pp_total_ou_45_5",
                   "bowler_wkts_1plus", "team_highest_individual_ou_29_5",
                   "highest_individual_mae"):
        assert family in scored, family

    # `build_observations` would otherwise emit one fabricated p=0
    # top-bowler row per rostered player; they are dropped, not scored.
    assert built["dropped_fabricated_rows"]["top_bowler"] == 2 * 2 * 11
    assert payload["raw_fields_present"] == sorted([
        score_props.RAW_BATTER_RUNS, score_props.RAW_BOWLER_WICKETS,
        score_props.RAW_POWERPLAY, score_props.RAW_TOTAL_RUNS])


def test_brier_arithmetic_against_a_zero_baseline(tmp_path):
    _arm_dir, _fixture_dir, built = _built(tmp_path)
    payload = score_props.score(built, _empty_logs())

    # One simulation clears every line and one clears none, so every
    # simulated probability is 0.5 and every sim Brier is 0.25; the
    # synthetic corpus has no history, so every fair-baseline probability
    # is 0.0 and its Brier is the outcome itself. The canonical fixture
    # scores under every runs/wickets line here (y=0) but over both
    # powerplay lines (y=1).
    expected = {
        "innings_runs_ou_160_5": 0.0,
        "team_highest_individual_ou_39_5": 0.0,
        "bowler_wkts_3plus": 0.0,
        "batter_50plus": 0.0,
        "pp_total_ou_55_5": 1.0,
    }
    for family, baseline in expected.items():
        block = payload["families"][family]
        assert block["metric"] == "brier"
        assert block["sim"] == pytest.approx(0.25)
        assert block["baseline"] == pytest.approx(baseline)
        assert block["delta_sim_minus_baseline"] == pytest.approx(
            0.25 - baseline)
        assert block["delta_ci95"] == pytest.approx(
            [0.25 - baseline, 0.25 - baseline])
        assert block["n_matches"] == 2
        assert set(block) == {"metric", "n_rows", "n_matches", "sim",
                              "baseline", "delta_sim_minus_baseline",
                              "delta_ci95"}

    # 4 rows per fixture (2 innings x 2 lines is 2 rows per team here).
    assert payload["families"]["innings_runs_ou_160_5"]["n_rows"] == 4
    # MAE families report an absolute-error metric on the same pairing.
    mae = payload["families"]["batter_runs_mae"]
    assert mae["metric"] == "mae"
    assert mae["sim"] > 0


def test_scores_do_not_depend_on_observation_row_order(tmp_path):
    """`baseline_rows` emits per-team rows in set-iteration order."""
    _arm_dir, _fixture_dir, built = _built(tmp_path)
    payload = score_props.score(built, _empty_logs())
    for match in built["detail"]:
        for family, rows in match["obs"].items():
            match["obs"][family] = list(reversed(rows))
    reversed_payload = score_props.score(built, _empty_logs())
    assert reversed_payload["families"] == payload["families"]


def test_corpus_without_the_match_top_score_column_skips_that_family(
        tmp_path):
    _arm_dir, _fixture_dir, built = _built(tmp_path)
    payload = score_props.score(built, _empty_logs(venue_match_arity=3))
    assert "highest_individual_mae" not in payload["families_scored"]
    reasons = {row["family"]: row["reason"]
               for row in payload["families_skipped"]}
    assert "venue_match rows carry 3 fields" in reasons[
        "highest_individual_mae"]
    assert payload["corpus_venue_match_arity"] == 3


def test_families_filter(tmp_path):
    _arm_dir, _fixture_dir, built = _built(tmp_path)
    payload = score_props.score(built, _empty_logs(),
                                {"innings_runs_ou_170_5"})
    assert payload["families_scored"] == ["innings_runs_ou_170_5"]
    reasons = {row["family"]: row["reason"]
               for row in payload["families_skipped"]}
    assert reasons["pp_total_ou_45_5"] == "not requested via --families"


def test_raw_file_without_bowler_wickets_reports_that_field(tmp_path):
    arm_dir, fixture_dir = _write_run(tmp_path)
    raw_path = arm_dir / score_props.RAW_SIMS_FILENAME
    rows = score_props.load_jsonl(raw_path)
    for row in rows:
        for simulation in row["simulations"]:
            for innings in simulation["innings"]:
                innings.pop("bowler_wickets")
    raw_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n")

    built = score_props.build_detail(
        score_props.load_jsonl(raw_path), fixture_dir,
        set(score_props.MISSING_FIELD))
    payload = score_props.score(built, _empty_logs())
    assert score_props.RAW_BOWLER_WICKETS not in payload["raw_fields_present"]
    skipped = {row["family"]: row.get("missing_field")
               for row in payload["families_skipped"]}
    for family in ("bowler_wkts_1plus", "bowler_wkts_2plus",
                   "bowler_wkts_3plus"):
        assert family not in payload["families_scored"]
        assert skipped[family] == score_props.RAW_BOWLER_WICKETS
    assert "innings_runs_ou_160_5" in payload["families_scored"]


def test_void_fixture_is_not_scored(tmp_path):
    arm_dir, fixture_dir = _write_run(tmp_path)
    document = json.loads((fixture_dir / "3000002.json").read_text())
    document["info"]["outcome"]["method"] = "D/L"
    (fixture_dir / "3000002.json").write_text(json.dumps(document))
    built = score_props.build_detail(
        score_props.load_jsonl(arm_dir / score_props.RAW_SIMS_FILENAME),
        fixture_dir, set(score_props.MISSING_FIELD))
    payload = score_props.score(built, _empty_logs())
    assert payload["voided_fixtures"] == [
        {"match_id": "3000002", "reason": "outcome.method=D/L"}]
    assert payload["n_fixtures_scored"] == 1
    assert payload["families"]["innings_runs_ou_160_5"]["n_matches"] == 1


def test_unknown_player_id_fails_closed(tmp_path):
    arm_dir, fixture_dir = _write_run(tmp_path)
    raw_path = arm_dir / score_props.RAW_SIMS_FILENAME
    rows = score_props.load_jsonl(raw_path)
    innings = rows[0]["simulations"][0]["innings"][0]
    innings["batter_runs"] = {"not-a-player": 25}
    raw_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(SystemExit) as excinfo:
        score_props.build_detail(
            score_props.load_jsonl(raw_path), fixture_dir,
            set(score_props.MISSING_FIELD))
    assert "not-a-player" in str(excinfo.value)


def test_cli_writes_props_json(tmp_path):
    import pickle

    arm_dir, _fixture_dir = _write_run(tmp_path)
    corpus = tmp_path / "corpus.pkl"
    corpus.write_bytes(pickle.dumps(_empty_logs()))
    score_props.main(["--arm-dir", str(arm_dir), "--corpus", str(corpus)])
    payload = json.loads(
        (arm_dir / score_props.DEFAULT_OUT_NAME).read_text())
    assert payload["contract"] == score_props.CONTRACT
    for key in ("arm", "arm_dir", "raw_sims", "fixture_dir", "corpus",
                "families", "families_scored", "families_skipped",
                "family_universe", "fixtures", "bootstrap",
                "baseline_version", "n_fixtures_scored"):
        assert key in payload, key
    assert payload["corpus"]["path"] == str(corpus)
    assert payload["bootstrap"]["seed"] == 29
    assert payload["fixtures"][0]["venue_resolved"] is True


def _retarget_match_id(arm_dir, old_id, new_id):
    raw_path = arm_dir / score_props.RAW_SIMS_FILENAME
    rows = score_props.load_jsonl(raw_path)
    for row in rows:
        if row["match_id"] == old_id:
            row["match_id"] = new_id
    raw_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


@pytest.mark.parametrize("bad_id", ["../outside", "..%2foutside",
                                    "/etc/hosts", "dir/3000001",
                                    "3000001.json"])
def test_traversal_and_absolute_match_ids_are_refused(tmp_path, bad_id):
    arm_dir, fixture_dir = _write_run(tmp_path)
    # A readable file the traversal would reach if the id were trusted.
    (tmp_path / "outside.json").write_text(
        (fixture_dir / "3000001.json").read_text())
    _retarget_match_id(arm_dir, "3000001", bad_id)
    with pytest.raises(SystemExit) as excinfo:
        score_props.build_detail(
            score_props.load_jsonl(arm_dir / score_props.RAW_SIMS_FILENAME),
            fixture_dir, set(score_props.MISSING_FIELD))
    assert "refusing match id" in str(excinfo.value)

    import pickle
    corpus = tmp_path / "corpus.pkl"
    corpus.write_bytes(pickle.dumps(_empty_logs()))
    with pytest.raises(SystemExit) as excinfo:
        score_props.main(["--arm-dir", str(arm_dir),
                          "--corpus", str(corpus)])
    assert "refusing match id" in str(excinfo.value)
    assert not (arm_dir / score_props.DEFAULT_OUT_NAME).exists()


def test_symlinked_fixture_into_a_sealed_dir_is_refused(tmp_path):
    arm_dir, fixture_dir = _write_run(tmp_path)
    sealed_dir = tmp_path / "data" / "golden" / "polymarket_test_v2"
    sealed_dir.mkdir(parents=True)
    target = sealed_dir / "sealed.json"
    target.write_text((fixture_dir / "3000001.json").read_text())
    link = fixture_dir / "3000001.json"
    link.unlink()
    link.symlink_to(target)

    with pytest.raises(SystemExit) as excinfo:
        score_props.build_detail(
            score_props.load_jsonl(arm_dir / score_props.RAW_SIMS_FILENAME),
            fixture_dir, set(score_props.MISSING_FIELD))
    assert "refusing sealed path" in str(excinfo.value)
    with pytest.raises(SystemExit) as excinfo:
        score_props.fixture_document_path("3000001", fixture_dir)
    assert "refusing sealed path" in str(excinfo.value)
    assert not (arm_dir / score_props.DEFAULT_OUT_NAME).exists()


def test_symlinked_fixture_outside_the_dir_is_refused(tmp_path):
    arm_dir, fixture_dir = _write_run(tmp_path)
    target = tmp_path / "elsewhere.json"
    target.write_text((fixture_dir / "3000001.json").read_text())
    link = fixture_dir / "3000001.json"
    link.unlink()
    link.symlink_to(target)
    with pytest.raises(SystemExit) as excinfo:
        score_props.build_detail(
            score_props.load_jsonl(arm_dir / score_props.RAW_SIMS_FILENAME),
            fixture_dir, set(score_props.MISSING_FIELD))
    assert "resolves outside" in str(excinfo.value)


@pytest.mark.parametrize("sealed", ["data/golden/polymarket_test_v2",
                                    "data/forward_holdout/2026-06-01"])
def test_sealed_paths_are_refused_before_anything_is_opened(tmp_path, sealed):
    arm_dir, fixture_dir = _write_run(tmp_path)

    # Sealed --fixture-dir: refused before the existence check, so the
    # message is the guard's, not "fixture dir not found".
    with pytest.raises(SystemExit) as excinfo:
        score_props.main(["--arm-dir", str(arm_dir),
                          "--fixture-dir", sealed])
    assert str(excinfo.value) == f"refusing sealed path: {sealed}"

    # Sealed arm dir: refused before the raw-sims existence check.
    with pytest.raises(SystemExit) as excinfo:
        score_props.main(["--arm-dir", sealed,
                          "--fixture-dir", str(fixture_dir)])
    assert str(excinfo.value) == f"refusing sealed path: {sealed}"

    # Sealed --corpus and --out.
    with pytest.raises(SystemExit) as excinfo:
        score_props.main(["--arm-dir", str(arm_dir),
                          "--corpus", f"{sealed}/corpus.pkl"])
    assert "refusing sealed path" in str(excinfo.value)
    with pytest.raises(SystemExit) as excinfo:
        score_props.main(["--arm-dir", str(arm_dir),
                          "--out", f"{sealed}/props.json"])
    assert "refusing sealed path" in str(excinfo.value)

    # Sealed fixture dir recorded in provenance: refused on resolution.
    provenance = json.loads(
        (arm_dir / score_props.PROVENANCE_FILENAME).read_text())
    provenance["run"]["fixture_dir"] = sealed
    (arm_dir / score_props.PROVENANCE_FILENAME).write_text(
        json.dumps(provenance))
    with pytest.raises(SystemExit) as excinfo:
        score_props.main(["--arm-dir", str(arm_dir)])
    assert str(excinfo.value) == f"refusing sealed path: {sealed}"

    # And the library entry points guard themselves.
    with pytest.raises(SystemExit):
        score_props.load_jsonl(f"{sealed}/raw_sims.jsonl")
    with pytest.raises(SystemExit):
        score_props.load_corpus(f"{sealed}/corpus.pkl")
    with pytest.raises(SystemExit):
        score_props.build_detail([], sealed, set(score_props.MISSING_FIELD))
    assert not (arm_dir / score_props.DEFAULT_OUT_NAME).exists()


@pytest.mark.needs_artifacts
def test_corpus_role_resolves_to_the_manifest_path():
    from artifacts import artifact_path
    path = artifact_path(score_props.CORPUS_ROLE)
    assert path.name.endswith(".pkl")
    logs = score_props.load_corpus(score_props.REPO_ROOT / path)
    assert set(logs) >= set(CORPUS_KEYS)
