"""Negative tests for the stage-1 cross-arm audit (D6 check 6.6).

Every way the arms can silently stop solving the same problem must fail,
naming the arm (and, where the defect is per fixture, the fixture):

* a fixture missing from one arm, a moved as-of stamp, a flipped
  eligibility flag, a changed sidecar hash (the four D6 6.6 cases);
* a changed odds row, a fixture scored in one arm and skipped in another
  (Astra round 1, item 3);
* an empty fixture set, a field missing from EVERY arm, a fixture set that
  does not match the fixture directory (item 2);
* a run that never verified the registered config (item 4).

Synthetic provenance only: no model, no cache, no simulation.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from artifacts import md5_directory
from sequence_track.run_arm import fixture_seed, sub_seeds

from sequence_track.audit_cross_arm import (PROVENANCE_CONTRACT, audit,
                                            load_arm, main)


BASE_SEED = 20260910
SMOKE_IDS = ("1478908", "1501896")
TIMING_IDS = tuple(f"150000{index}" for index in range(10))
ODDS_ROW_SHA = {"1478908": "a" * 64, "1501896": "b" * 64}


def _eligibility(**overrides):
    row = {
        "odds_row_found": True,
        "odds_row_id": {"cricsheet_id": "1478908",
                        "display_match_id": "2025-10-31_India_Australia"},
        "odds_row_sha256": "a" * 64,
        "odds_actual_winner": "Australia",
        "cricsheet_resolved": True,
        "cricsheet_winner": "Australia",
        "male_t20": True,
        "scored": True,
        "skip_reason": None,
    }
    row.update(overrides)
    return row


def _fixture_row(cricsheet_id, date="2025-10-31", predecessors=(),
                 matches_advanced=10, n_sims=10, base_seed=BASE_SEED,
                 eligibility=None):
    """A row whose seeds are the ones the runner would really derive."""
    seed = fixture_seed(cricsheet_id, base_seed)
    return {
        "cricsheet_id": cricsheet_id,
        "match_date": date,
        "as_of": {
            "date": date,
            "same_day_advanced_before": list(predecessors),
            "matches_advanced": matches_advanced,
        },
        "eligibility": eligibility or _eligibility(),
        "fixture_seed": seed,
        "sub_seeds": sub_seeds(seed),
        "n_sims": n_sims,
    }


def _make_dir(tmp_path, name, ids):
    directory = tmp_path / name
    directory.mkdir(exist_ok=True)
    for fixture_id in ids:
        (directory / f"{fixture_id}.json").write_text("{}")
    return directory


def _fixture_dir(tmp_path):
    return _make_dir(tmp_path, "fixtures", SMOKE_IDS)


def _timing_dir(tmp_path):
    return _make_dir(tmp_path, "timing_fixtures", TIMING_IDS)


# The synthetic registered config: the audit re-derives every arm's pins
# from this file, so the test fixture has to carry the same shape the real
# `pin_stage1` template writes.
# Since the 2026-09-11 retrain (D6 check 6.3) every arm serves the i7 cache
# and B resolves from the retrain namespace, so the audit's synthetic pins
# carry one cache hash across both arms.
ARM_PINS = {
    "A": {"model_dir": "models/xgb_i7_noweights_production",
          "checkpoint_md5": "A-booster", "stats_version": "i7",
          "stats_cache_md5": "i7-cache"},
    "B": {"model_dir": "models/embeddings/seq_stage1/retrain_i7/mlp/seed_101",
          "checkpoint_md5": "B-booster", "stats_version": "i7",
          "stats_cache_md5": "i7-cache"},
}
PERMITTED_TIMING_N_SIMS = [100, 200, 400, 800, 1600, 3200]


def _config_payload(tmp_path):
    smoke_dir = _fixture_dir(tmp_path)
    timing_dir = _timing_dir(tmp_path)
    arms = {}
    for arm, pins in ARM_PINS.items():
        arms[arm] = {
            "model_dir": pins["model_dir"],
            "checkpoint_md5": pins["checkpoint_md5"],
            "stats_version": pins["stats_version"],
            "stats_cache_md5": pins["stats_cache_md5"],
            "extras_graft_sha256": "ad6e863b",
            "bowler_usage_md5": "usage-md5",
            "roster_policy_md5": "roster-md5",
            "bowler_selector": "RosterEmpiricalBowlerSelector",
            "player_metadata_sha256": "m" * 64,
            "run_out": {"constant": "RUNOUT_P", "value": 0.075077},
            "odds": {"path": "betting_odds_polymarket_v2.json",
                     "sha256": "o" * 64},
            "context_dir": "data/t20s_json",
            "context_dir_md5": "context-md5",
            "cluster_source_dir": "data/polymarket_test_v2",
            "cluster_source_dir_md5": "cluster-md5",
            "clip": [0.01, 0.99],
            "threads": 1,
            "device": "cpu",
            "base_seed": BASE_SEED,
            "smoke_1a": {"fixture_dir": str(smoke_dir),
                         "fixture_dir_md5": md5_directory(smoke_dir),
                         "fixture_count": len(SMOKE_IDS),
                         "n_sims": 10, "base_seeds": [BASE_SEED]},
            "full_run": {"fixture_dir": "data/polymarket_test_v2",
                         "fixture_dir_md5": "full-md5",
                         "fixture_count": 255,
                         "n_sims": "to_be_filled_before_1d",
                         "base_seeds": [BASE_SEED]},
            "timing_1b": {
                "fixture_dir": str(timing_dir),
                "fixture_dir_md5": md5_directory(timing_dir),
                "fixture_count": len(TIMING_IDS),
                "n_sims": list(PERMITTED_TIMING_N_SIMS),
                "candidates_screened": PERMITTED_TIMING_N_SIMS + [6400],
                "base_seeds": [20260910, 20260911, 20260912]},
        }
    return {
        "arms": arms,
        "provenance": {"source_md5": {
            "scripts/sim_v1_2.py": "engine-md5",
            "scripts/sequence_track/run_arm.py": "runner-md5"}},
    }


def _config_file(tmp_path, mutate=None):
    payload = _config_payload(tmp_path)
    if mutate is not None:
        mutate(payload)
    path = tmp_path / "seq_stage1_sim_v1.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return str(path), digest


def _payload(arm, fixture_dir, config=None, model_dir_hash="deadbeef",
             block="smoke_1a", ids=SMOKE_IDS, n_sims=10,
             base_seed=BASE_SEED):
    pins = ARM_PINS[arm]
    config_path, config_sha = config or (None, None)
    rows = []
    for index, fixture_id in enumerate(ids):
        eligibility = _eligibility(
            odds_row_id={"cricsheet_id": fixture_id,
                         "display_match_id": f"2025-10-31_{fixture_id}"},
            odds_row_sha256=ODDS_ROW_SHA.get(fixture_id, fixture_id * 4))
        rows.append(_fixture_row(
            fixture_id, predecessors=list(ids[:index]),
            matches_advanced=10 + index, n_sims=n_sims, base_seed=base_seed,
            eligibility=eligibility))
    return {
        "contract": PROVENANCE_CONTRACT,
        "arm": arm,
        "run": {
            "arm": arm,
            "model_dir": pins["model_dir"],
            "model_dir_hash": model_dir_hash,
            "checkpoint_md5": pins["checkpoint_md5"],
            "stats_version": pins["stats_version"],
            "stats_cache_md5": pins["stats_cache_md5"],
            "selector_class": "RosterEmpiricalBowlerSelector",
            "bowler_usage_md5": "usage-md5",
            "bowler_roster_policy_md5": "roster-md5",
            "extras_graft_sha256": "ad6e863b",
            "runout_p": 0.075077,
            "clip": {"low": 0.01, "high": 0.99, "seam_enabled": True},
            "base_seed": base_seed,
            "n_sims": n_sims,
            "engine_md5": "engine-md5",
            "runner_md5": "runner-md5",
            "threads": 1,
            "device": "cpu",
            "config_path": config_path,
            "config_sha256": config_sha,
            "config_verified": config_path is not None,
            "config_block": block,
            "fixture_dir": str(fixture_dir),
            "fixture_dir_hash": md5_directory(fixture_dir),
            "fixture_count": len(ids),
            "player_metadata_sha256": "m" * 64,
            "context_dir": "data/t20s_json",
            "context_dir_hash": "context-md5",
            "cluster_source_dir": "data/polymarket_test_v2",
            "cluster_source_dir_hash": "cluster-md5",
            "odds": "betting_odds_polymarket_v2.json",
            "odds_sha256": "o" * 64,
        },
        "fixtures": rows,
    }


def _write(tmp_path, arm, payload):
    directory = tmp_path / arm
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "arm_provenance.json").write_text(json.dumps(payload))
    return directory


def _arms(tmp_path, mutate=None, mutate_a=None, config=None):
    fixture_dir = _fixture_dir(tmp_path)
    config = config or _config_file(tmp_path)
    a = _payload("A", fixture_dir, config)
    b = _payload("B", fixture_dir, config, model_dir_hash="cafef00d")
    if mutate is not None:
        mutate(b)
    if mutate_a is not None:
        mutate_a(a)
    return [load_arm(_write(tmp_path, "A", a)),
            load_arm(_write(tmp_path, "B", b))]


# ---------------------------------------------------------------------------
# The identical case, and the asymmetries that are allowed to differ
# ---------------------------------------------------------------------------

def test_identical_arms_pass(tmp_path):
    assert audit(_arms(tmp_path)) == []


def test_expected_asymmetries_do_not_fail(tmp_path):
    """Model dir and checkpoint differ by design; the stats cache no longer
    does, because the 2026-09-11 retrain put every arm on the i7 cache."""
    arms = _arms(tmp_path)
    assert arms[0].run["stats_cache_md5"] == arms[1].run["stats_cache_md5"]
    assert arms[0].run["model_dir_hash"] != arms[1].run["model_dir_hash"]
    assert audit(arms) == []


def test_a_cross_arm_cache_difference_is_listed_not_asserted(tmp_path):
    """The audit checks each arm against ITS OWN block; a cache difference
    between arms is a listed field, so a future arm on another cache is
    reported by the table rather than refused here."""
    def use_another_cache(payload):
        payload["arms"]["B"]["stats_version"] = "v3"
        payload["arms"]["B"]["stats_cache_md5"] = "v3-cache"

    config = _config_file(tmp_path, use_another_cache)

    def drift(payload):
        payload["run"]["stats_version"] = "v3"
        payload["run"]["stats_cache_md5"] = "v3-cache"

    arms = _arms(tmp_path, drift, config=config)
    assert arms[0].run["stats_cache_md5"] != arms[1].run["stats_cache_md5"]
    assert audit(arms) == []


# ---------------------------------------------------------------------------
# D6 6.2-6.5 — the four required negative cases
# ---------------------------------------------------------------------------

def test_missing_fixture_fails_naming_arm_and_fixture(tmp_path):
    def drop(payload):
        payload["fixtures"] = [payload["fixtures"][0]]

    failures = audit(_arms(tmp_path, drop))
    assert any("6.2" in message and "arm B" in message
               and "1501896" in message for message in failures)


def test_changed_as_of_stamp_fails_naming_arm_and_fixture(tmp_path):
    def move_stamp(payload):
        payload["fixtures"][1]["as_of"]["same_day_advanced_before"] = []
        payload["fixtures"][1]["as_of"]["matches_advanced"] = 10

    failures = audit(_arms(tmp_path, move_stamp))
    assert any("6.3" in message and "arm B" in message
               and "1501896" in message for message in failures)
    assert any("matches_advanced" in message for message in failures)


def test_flipped_eligibility_flag_fails_naming_arm_and_fixture(tmp_path):
    def flip(payload):
        payload["fixtures"][0]["eligibility"]["odds_row_found"] = False

    failures = audit(_arms(tmp_path, flip))
    assert any("6.4" in message and "arm B" in message
               and "1478908" in message and "odds_row_found" in message
               for message in failures)


def test_changed_sidecar_hash_fails_naming_arm(tmp_path):
    def resign(payload):
        payload["run"]["extras_graft_sha256"] = "0000dead"

    failures = audit(_arms(tmp_path, resign))
    # A run-level defect has no fixture to name; it must name the arm and
    # the field.
    assert any("6.5" in message and "arm B" in message
               and "extras_graft_sha256" in message for message in failures)


# ---------------------------------------------------------------------------
# Astra item 3 — audited eligibility is the eligibility that was scored
# ---------------------------------------------------------------------------

def test_a_different_odds_row_fails(tmp_path):
    def repriced(payload):
        payload["fixtures"][0]["eligibility"]["odds_row_sha256"] = "f" * 64

    failures = audit(_arms(tmp_path, repriced))
    assert any("6.4" in message and "odds_row_sha256" in message
               and "1478908" in message for message in failures)


def test_a_different_resolved_odds_row_id_fails(tmp_path):
    def rejoined(payload):
        payload["fixtures"][0]["eligibility"]["odds_row_id"] = {
            "cricsheet_id": "9999999", "display_match_id": None}

    failures = audit(_arms(tmp_path, rejoined))
    assert any("odds_row_id" in message and "1478908" in message
               for message in failures)


def test_a_fixture_scored_in_one_arm_only_fails(tmp_path):
    def unscored(payload):
        payload["fixtures"][1]["eligibility"]["scored"] = False
        payload["fixtures"][1]["eligibility"]["skip_reason"] = (
            "evaluation_error: RuntimeError: boom")

    failures = audit(_arms(tmp_path, unscored))
    assert any("scored" in message and "1501896" in message
               for message in failures)
    assert any("skip_reason" in message for message in failures)


def test_a_disagreeing_scored_winner_fails(tmp_path):
    def rewinner(payload):
        payload["fixtures"][0]["eligibility"]["odds_actual_winner"] = "India"

    failures = audit(_arms(tmp_path, rewinner))
    assert any("odds_actual_winner" in message for message in failures)


# ---------------------------------------------------------------------------
# Astra item 2 — the audit cannot pass without evidence
# ---------------------------------------------------------------------------

def test_identical_empty_fixture_sets_fail(tmp_path):
    fixture_dir = _fixture_dir(tmp_path)
    config = _config_file(tmp_path)
    arms = []
    for arm in ("A", "B"):
        payload = _payload(arm, fixture_dir, config)
        payload["fixtures"] = []
        arms.append(load_arm(_write(tmp_path, arm, payload)))
    failures = audit(arms)
    assert any("no fixtures recorded" in message for message in failures)
    assert any("arm A" in message for message in failures)
    assert any("arm B" in message for message in failures)


@pytest.mark.parametrize("field", [
    "extras_graft_sha256", "config_sha256", "odds_sha256",
    "context_dir_hash", "engine_md5", "runner_md5", "clip", "base_seed",
    "selector_class", "bowler_usage_md5", "n_sims",
    "cluster_source_dir", "cluster_source_dir_hash",
])
def test_a_field_missing_from_every_arm_fails(tmp_path, field):
    fixture_dir = _fixture_dir(tmp_path)
    config = _config_file(tmp_path)
    arms = []
    for arm in ("A", "B"):
        payload = _payload(arm, fixture_dir, config)
        del payload["run"][field]
        arms.append(load_arm(_write(tmp_path, arm, payload)))
    failures = audit(arms)
    assert any("schema" in message and field in message
               for message in failures), failures


@pytest.mark.parametrize("field", [
    "odds_row_sha256", "scored", "skip_reason", "cricsheet_resolved",
    "male_t20", "odds_row_id", "odds_actual_winner", "cricsheet_winner",
])
def test_an_eligibility_field_missing_from_every_arm_fails(tmp_path, field):
    fixture_dir = _fixture_dir(tmp_path)
    config = _config_file(tmp_path)
    arms = []
    for arm in ("A", "B"):
        payload = _payload(arm, fixture_dir, config)
        for row in payload["fixtures"]:
            del row["eligibility"][field]
        arms.append(load_arm(_write(tmp_path, arm, payload)))
    failures = audit(arms)
    assert any("schema" in message and field in message
               for message in failures), failures


def test_a_wrongly_typed_field_fails(tmp_path):
    def retype(payload):
        payload["run"]["base_seed"] = "20260910"
        payload["fixtures"][0]["eligibility"]["scored"] = "yes"

    failures = audit(_arms(tmp_path, retype))
    assert any("schema" in message and "base_seed" in message
               for message in failures)
    assert any("schema" in message and "scored" in message
               for message in failures)


def test_fixture_set_must_equal_the_fixture_directory(tmp_path):
    fixture_dir = _fixture_dir(tmp_path)
    (fixture_dir / "1505127.json").write_text("{}")
    failures = audit(_arms(tmp_path))
    assert any("1505127" in message
               and "but not in this arm's provenance" in message
               for message in failures), failures
    assert any("arm A" in message for message in failures)
    assert any("arm B" in message for message in failures)


def test_extra_fixture_not_in_the_directory_fails(tmp_path):
    fixture_dir = _fixture_dir(tmp_path)
    config = _config_file(tmp_path)
    arms = []
    for arm in ("A", "B"):
        payload = _payload(arm, fixture_dir, config)
        payload["fixtures"].append(_fixture_row("7777777"))
        payload["run"]["fixture_count"] = len(payload["fixtures"])
        arms.append(load_arm(_write(tmp_path, arm, payload)))
    failures = audit(arms)
    assert any("7777777" in message and "not in" in message
               for message in failures)


def test_disagreeing_fixture_dirs_fail_without_an_explicit_expectation(
        tmp_path):
    def elsewhere(payload):
        payload["run"]["fixture_dir"] = str(tmp_path / "other_fixtures")

    failures = audit(_arms(tmp_path, elsewhere))
    assert any("different fixture dirs" in message for message in failures)


def test_expected_fixtures_overrides_the_recorded_dir(tmp_path):
    other = tmp_path / "other_fixtures"
    other.mkdir()
    (other / "1478908.json").write_text("{}")

    def elsewhere(payload):
        payload["run"]["fixture_dir"] = str(other)

    arms = _arms(tmp_path, elsewhere)
    failures = audit(arms, expected_fixtures=str(other))
    # Both arms carry 1501896, which `other` does not hold.
    assert any("1501896" in message for message in failures)
    assert not any("different fixture dirs" in message
                   for message in failures)


def test_expected_fixtures_must_exist(tmp_path):
    with pytest.raises(SystemExit):
        audit(_arms(tmp_path), expected_fixtures=str(tmp_path / "absent"))


# ---------------------------------------------------------------------------
# Astra item 4 — unregistered runs are refused unless asked for
# ---------------------------------------------------------------------------

def test_unregistered_run_fails_by_default(tmp_path):
    def unregistered(payload):
        payload["run"]["config_verified"] = False
        payload["run"]["config_sha256"] = None
        payload["run"]["config_path"] = None

    failures = audit(_arms(tmp_path, unregistered))
    assert any("config" in message and "arm B" in message
               for message in failures)


def test_unregistered_run_is_allowed_with_the_flag(tmp_path):
    fixture_dir = _fixture_dir(tmp_path)
    config = _config_file(tmp_path)
    arms = []
    for arm in ("A", "B"):
        payload = _payload(arm, fixture_dir, config)
        payload["run"]["config_verified"] = False
        payload["run"]["config_sha256"] = None
        payload["run"]["config_path"] = None
        arms.append(load_arm(_write(tmp_path, arm, payload)))
    assert audit(arms) != []
    assert audit(arms, allow_unregistered=True) == []


def test_a_different_registered_config_fails(tmp_path):
    def repinned(payload):
        payload["run"]["config_sha256"] = "d" * 64

    failures = audit(_arms(tmp_path, repinned))
    assert any("config_sha256" in message and "arm B" in message
               for message in failures)


@pytest.mark.parametrize("field", [
    "odds_sha256", "context_dir_hash", "engine_md5", "runner_md5",
    # 1c / D10: the corpus the I3 block ids were stamped from.
    "cluster_source_dir_hash"])
def test_shared_inputs_must_be_identical(tmp_path, field):
    def drift(payload):
        payload["run"][field] = "moved"

    failures = audit(_arms(tmp_path, drift))
    assert any(field in message and "arm B" in message
               for message in failures)


def test_an_arm_that_stamped_clusters_from_its_own_shard_is_refused(tmp_path):
    """Every arm agrees, and every arm used the wrong cluster corpus."""
    fixture_dir = _fixture_dir(tmp_path)
    config = _config_file(tmp_path)
    arms = []
    for arm in ("A", "B"):
        payload = _payload(arm, fixture_dir, config)
        payload["run"]["cluster_source_dir"] = str(fixture_dir)
        payload["run"]["cluster_source_dir_hash"] = md5_directory(fixture_dir)
        arms.append(load_arm(_write(tmp_path, arm, payload)))
    # Cross-arm equality alone is satisfied …
    assert (arms[0].run["cluster_source_dir_hash"]
            == arms[1].run["cluster_source_dir_hash"])
    # … and the audit still refuses both, against the registered pin.
    failures = audit(arms)
    for arm in ("A", "B"):
        assert any(f"[config] arm {arm}: cluster_source_dir" in message
                   for message in failures), arm


# ---------------------------------------------------------------------------
# Run-level identity, loading, and the CLI
# ---------------------------------------------------------------------------

def test_other_run_level_identity_fields_are_asserted(tmp_path):
    for field, value in (("selector_class", "EmpiricalBowlerSelector"),
                         ("bowler_usage_md5", "other"),
                         ("bowler_roster_policy_md5", None),
                         ("runout_p", 0.08),
                         ("n_sims", 20),
                         ("base_seed", 1),
                         ("clip", {"low": 0.05, "high": 0.95,
                                   "seam_enabled": False})):
        def mutate(payload, field=field, value=value):
            payload["run"][field] = value

        failures = audit(_arms(tmp_path, mutate))
        assert any(field in message and "arm B" in message
                   for message in failures), field


def test_per_fixture_seed_and_n_sims_are_asserted(tmp_path):
    def reseed(payload):
        payload["fixtures"][0]["fixture_seed"] = 42

    failures = audit(_arms(tmp_path, reseed))
    assert any("6.5" in message and "1478908" in message
               and "fixture_seed" in message for message in failures)


def test_cli_exits_zero_on_agreement_and_one_on_mismatch(tmp_path, capsys):
    fixture_dir = _fixture_dir(tmp_path)
    config = _config_file(tmp_path)
    a = _payload("A", fixture_dir, config)
    b = _payload("B", fixture_dir, config, model_dir_hash="cafef00d")
    dir_a = _write(tmp_path, "A", a)
    dir_b = _write(tmp_path, "B", b)
    assert main([str(dir_a), str(dir_b)]) == 0
    assert "CROSS-ARM AUDIT PASSED" in capsys.readouterr().out

    b["fixtures"][0]["eligibility"]["cricsheet_resolved"] = False
    _write(tmp_path, "B", b)
    assert main([str(dir_a), str(dir_b)]) == 1
    out = capsys.readouterr().out
    assert "CROSS-ARM AUDIT FAILED" in out
    assert "1478908" in out


def test_missing_or_wrong_contract_fails_closed(tmp_path):
    missing = tmp_path / "empty"
    missing.mkdir()
    with pytest.raises(SystemExit):
        load_arm(missing)

    bad = _payload("A", _fixture_dir(tmp_path), _config_file(tmp_path))
    bad["contract"] = "sequence_track_arm_provenance_v1"
    with pytest.raises(SystemExit):
        load_arm(_write(tmp_path, "bad", bad))


def test_duplicate_fixture_ids_fail_closed(tmp_path):
    payload = _payload("A", _fixture_dir(tmp_path), _config_file(tmp_path))
    payload["fixtures"].append(copy.deepcopy(payload["fixtures"][0]))
    with pytest.raises(SystemExit):
        load_arm(_write(tmp_path, "dup", payload))


def test_a_single_arm_is_refused(tmp_path):
    with pytest.raises(SystemExit):
        audit([load_arm(_write(
            tmp_path, "A",
            _payload("A", _fixture_dir(tmp_path), _config_file(tmp_path))))])


# ---------------------------------------------------------------------------
# Astra round 2, item 4b — the audit re-derives registration, never trusts it
# ---------------------------------------------------------------------------

def test_null_config_identity_fails_even_when_verified_is_true(tmp_path):
    """`config_verified: true` with no config to check is not evidence."""
    fixture_dir = _fixture_dir(tmp_path)
    config = _config_file(tmp_path)
    arms = []
    for arm in ("A", "B"):
        payload = _payload(arm, fixture_dir, config)
        payload["run"]["config_path"] = None
        payload["run"]["config_sha256"] = None
        payload["run"]["config_verified"] = True
        arms.append(load_arm(_write(tmp_path, arm, payload)))
    failures = audit(arms)
    assert any("config_path/config_sha256 is null" in message
               for message in failures)
    assert any("arm A" in message for message in failures)
    assert any("arm B" in message for message in failures)


def test_uniformly_wrong_hashes_no_longer_pass(tmp_path):
    """Astra's exact bypass: every arm agrees, and every arm is wrong."""
    fixture_dir = _fixture_dir(tmp_path)
    config = _config_file(tmp_path)
    arms = []
    for arm in ("A", "B"):
        payload = _payload(arm, fixture_dir, config)
        payload["run"]["odds_sha256"] = "wrong-odds"
        payload["run"]["context_dir_hash"] = "wrong-context"
        payload["run"]["engine_md5"] = "wrong-engine"
        payload["run"]["runner_md5"] = "wrong-runner"
        arms.append(load_arm(_write(tmp_path, arm, payload)))
    # Cross-arm equality alone is satisfied …
    assert arms[0].run["engine_md5"] == arms[1].run["engine_md5"]
    # … and the audit still fails, per arm and per field.
    failures = audit(arms)
    for field in ("odds_sha256", "context_dir_hash", "engine_md5",
                  "runner_md5"):
        assert any(f"[config] arm A: {field}" in message
                   for message in failures), field
        assert any(f"[config] arm B: {field}" in message
                   for message in failures), field


def test_a_missing_config_file_fails(tmp_path):
    config_path, config_sha = _config_file(tmp_path)
    arms = _arms(tmp_path, config=(config_path, config_sha))
    Path(config_path).unlink()
    failures = audit(arms)
    assert any("does not exist" in message for message in failures)


def test_a_config_changed_after_the_run_fails(tmp_path):
    config_path, config_sha = _config_file(tmp_path)
    arms = _arms(tmp_path, config=(config_path, config_sha))
    Path(config_path).write_text(Path(config_path).read_text() + "\n# edit\n")
    failures = audit(arms)
    assert any("hashes to" in message and "changed after the run" in message
               for message in failures)


@pytest.mark.parametrize("field,value", [
    ("model_dir", "models/somewhere_else"),
    ("checkpoint_md5", "moved"),
    ("stats_version", "v3"),
    ("stats_cache_md5", "moved"),
    ("extras_graft_sha256", "moved"),
    ("bowler_usage_md5", "moved"),
    ("bowler_roster_policy_md5", "moved"),
    ("threads", 8),
    ("device", "mps"),
    ("selector_class", "EmpiricalBowlerSelector"),
    ("n_sims", 999),
    ("base_seed", 12345),
])
def test_one_arm_drifted_from_its_registered_block_fails(
        tmp_path, field, value):
    def drift(payload):
        payload["run"][field] = value

    failures = audit(_arms(tmp_path, drift))
    assert any(message.startswith("[config] arm B") and field in message
               for message in failures), failures


def test_clip_seam_disabled_against_a_pinned_clip_fails(tmp_path):
    def unclip(payload):
        payload["run"]["clip"] = {"low": 0.01, "high": 0.99,
                                  "seam_enabled": False}

    failures = audit(_arms(tmp_path, unclip))
    assert any("clip seam is disabled" in message for message in failures)


def test_an_unregistered_config_block_fails(tmp_path):
    def rename(payload):
        payload["run"]["config_block"] = "some_other_block"

    failures = audit(_arms(tmp_path, rename))
    assert any("config_block" in message and "not a registered run block"
               in message for message in failures)


def test_a_mixed_registered_and_unregistered_set_is_refused(tmp_path):
    def unregister(payload):
        payload["run"]["config_verified"] = False
        payload["run"]["config_path"] = None
        payload["run"]["config_sha256"] = None

    arms = _arms(tmp_path, unregister)
    failures = audit(arms, allow_unregistered=True)
    assert any("--allow-unregistered is for a set where NO arm" in message
               for message in failures)


# ---------------------------------------------------------------------------
# Astra round 2, item 4c — the registered 1b batches audit as registered
# ---------------------------------------------------------------------------

def _timing_arms(tmp_path, base_seed=20260911, n_sims=200):
    """Arms that ran the registered 1b timing shard, not the smoke set."""
    config = _config_file(tmp_path)
    timing_dir = _timing_dir(tmp_path)
    arms = []
    for arm in ("A", "B"):
        payload = _payload(arm, timing_dir, config, block="timing_1b",
                           ids=TIMING_IDS, n_sims=n_sims,
                           base_seed=base_seed)
        arms.append(load_arm(_write(tmp_path, arm, payload)))
    return arms


def test_a_registered_batch_seed_audits_clean(tmp_path):
    assert audit(_timing_arms(tmp_path, base_seed=20260911,
                              n_sims=200)) == []
    assert audit(_timing_arms(tmp_path, base_seed=20260912,
                              n_sims=400)) == []


def test_an_unregistered_batch_seed_fails(tmp_path):
    failures = audit(_timing_arms(tmp_path, base_seed=777, n_sims=200))
    assert any("base_seed" in message and "permits" in message
               for message in failures)


def test_an_unregistered_batch_n_sims_fails(tmp_path):
    failures = audit(_timing_arms(tmp_path, base_seed=20260911, n_sims=999))
    assert any("n_sims" in message and "permits" in message
               for message in failures)


def test_the_full_run_block_is_not_launchable_while_n_sims_is_a_placeholder(
        tmp_path):
    def full(payload):
        payload["run"]["config_block"] = "full_run"

    failures = audit(_arms(tmp_path, full))
    assert any("not a launchable value" in message for message in failures)


# ---------------------------------------------------------------------------
# Astra round 3, item 1 — the claimed block is bound to its fixture set
# ---------------------------------------------------------------------------

def test_smoke_records_relabelled_as_timing_are_refused(tmp_path):
    """Astra's exact case: relabel the smoke records as `timing_1b` with
    counts 100 and they used to pass, four smoke fixtures and all."""
    fixture_dir = _fixture_dir(tmp_path)
    config = _config_file(tmp_path)
    arms = []
    for arm in ("A", "B"):
        payload = _payload(arm, fixture_dir, config)
        payload["run"]["config_block"] = "timing_1b"
        payload["run"]["n_sims"] = 100
        for row in payload["fixtures"]:
            row["n_sims"] = 100
        arms.append(load_arm(_write(tmp_path, arm, payload)))
    failures = audit(arms)
    assert any("fixture_dir" in message and "timing_1b block registers"
               in message for message in failures), failures
    assert any("fixture_dir_hash" in message for message in failures)
    assert any("fixture_count" in message for message in failures)


def test_a_block_whose_fixture_set_is_unpinned_cannot_be_claimed(tmp_path):
    """The 1b shard is absent until 1b; nothing may claim it before then."""
    def unpin(payload):
        for arm in payload["arms"].values():
            arm["timing_1b"]["fixture_dir_md5"] = None
            arm["timing_1b"]["fixture_dir_status"] = "absent"

    config = _config_file(tmp_path, unpin)
    timing_dir = _timing_dir(tmp_path)
    arms = []
    for arm in ("A", "B"):
        payload = _payload(arm, timing_dir, config, block="timing_1b",
                           ids=TIMING_IDS, n_sims=200, base_seed=20260911)
        arms.append(load_arm(_write(tmp_path, arm, payload)))
    failures = audit(arms)
    assert any("pins no fixture directory hash" in message
               for message in failures), failures


def test_a_directory_edited_after_the_run_fails_on_inventory(tmp_path):
    """The run and the pin agree with each other but not with the disk."""
    arms = _arms(tmp_path)
    (tmp_path / "fixtures" / "9999999.json").write_text("{}")
    failures = audit(arms)
    assert any("9999999" in message and "not in this arm" in message
               for message in failures), failures


def test_a_run_that_read_another_directory_fails_on_the_hash(tmp_path):
    """The run's recorded inventory hash is not the block's."""
    def elsewhere(payload):
        payload["run"]["fixture_dir_hash"] = "some-other-directory"

    failures = audit(_arms(tmp_path, elsewhere))
    assert any("fixture_dir_hash" in message and "arm B" in message
               for message in failures), failures


# ---------------------------------------------------------------------------
# Astra round 3, item 2 — the jointly permitted ceiling is enforced
# ---------------------------------------------------------------------------

def test_a_jointly_forbidden_candidate_is_refused(tmp_path):
    """6400 is screened but NOT permitted: it collides across batches."""
    failures = audit(_timing_arms(tmp_path, base_seed=20260911,
                                  n_sims=6400))
    assert any("n_sims" in message and "permits" in message
               for message in failures), failures


def test_every_permitted_candidate_still_audits_clean(tmp_path):
    for candidate in PERMITTED_TIMING_N_SIMS:
        assert audit(_timing_arms(tmp_path, base_seed=20260912,
                                  n_sims=candidate)) == [], candidate


# ---------------------------------------------------------------------------
# Astra round 3, item 3 — an internally inconsistent record fails
# ---------------------------------------------------------------------------

def test_uniform_fixture_counts_that_contradict_the_run_fail(tmp_path):
    """Every fixture count 999 while the run says 10 used to pass."""
    fixture_dir = _fixture_dir(tmp_path)
    config = _config_file(tmp_path)
    arms = []
    for arm in ("A", "B"):
        payload = _payload(arm, fixture_dir, config)
        for row in payload["fixtures"]:
            row["n_sims"] = 999
        arms.append(load_arm(_write(tmp_path, arm, payload)))
    failures = audit(arms)
    assert any("[internal]" in message and "n_sims = 999" in message
               for message in failures), failures
    assert any("the smoke_1a block permits" in message
               for message in failures)


def test_uniform_wrong_seeds_fail(tmp_path):
    """Every fixture_seed 42 and every sub_seeds {} used to pass."""
    fixture_dir = _fixture_dir(tmp_path)
    config = _config_file(tmp_path)
    arms = []
    for arm in ("A", "B"):
        payload = _payload(arm, fixture_dir, config)
        for row in payload["fixtures"]:
            row["fixture_seed"] = 42
            row["sub_seeds"] = {}
        arms.append(load_arm(_write(tmp_path, arm, payload)))
    failures = audit(arms)
    assert any("[internal]" in message and "fixture_seed = 42" in message
               for message in failures), failures
    assert any("sha256 of" in message for message in failures)


def test_wrong_sub_seeds_alone_fail(tmp_path):
    def resub(payload):
        payload["fixtures"][0]["sub_seeds"] = {
            "outcome": 1, "extras": 2, "selector": 3}

    failures = audit(_arms(tmp_path, resub))
    assert any("[internal]" in message and "sub_seeds" in message
               for message in failures)


def test_a_run_fixture_count_that_contradicts_the_rows_fails(tmp_path):
    def miscount(payload):
        payload["run"]["fixture_count"] = 99

    failures = audit(_arms(tmp_path, miscount))
    assert any("[internal]" in message and "fixture_count" in message
               for message in failures)


@pytest.mark.parametrize("field,value", [
    ("player_metadata_sha256", "wrong"),
    ("runout_p", 0.9),
])
def test_uniformly_changed_pins_fail(tmp_path, field, value):
    """Astra changed both uniformly across arms and the audit passed."""
    fixture_dir = _fixture_dir(tmp_path)
    config = _config_file(tmp_path)
    arms = []
    for arm in ("A", "B"):
        payload = _payload(arm, fixture_dir, config)
        payload["run"][field] = value
        arms.append(load_arm(_write(tmp_path, arm, payload)))
    failures = audit(arms)
    assert any(f"[config] arm A: {field}" in message
               for message in failures), failures
    assert any(f"[config] arm B: {field}" in message for message in failures)


# ---------------------------------------------------------------------------
# D11 check 11.2 — a 1d shard run is bound to its shard entry, not the block
# ---------------------------------------------------------------------------

SHARD_IDS = ("1600001", "1600002", "1600003")
SHARD_N_SIMS = 1600


def _shard_dir(tmp_path, index=0, ids=SHARD_IDS):
    return _make_dir(tmp_path, f"shard{index}_fixtures", ids)


def _shard_config(tmp_path, index=0, mutate=None):
    """The synthetic config with a launchable, sharded `full_run` block."""
    directory = _shard_dir(tmp_path, index)

    def _add_shards(payload):
        for arm in payload["arms"].values():
            arm["full_run"]["n_sims"] = SHARD_N_SIMS
            arm["full_run"]["shards"] = [{
                "index": index,
                "fixture_dir": str(directory),
                "fixture_dir_md5": md5_directory(directory),
                "fixture_count": len(SHARD_IDS),
                "output_dir": f"models/embeddings/seq_stage1/full/A/shard"
                              f"{index}",
                "command": "env uv run",
            }]
        if mutate is not None:
            mutate(payload)

    return directory, _config_file(tmp_path, _add_shards)


def _shard_arms(tmp_path, index=0, mutate=None, config=None, directory=None):
    if config is None:
        directory, config = _shard_config(tmp_path, index)
    arms = []
    for arm in ("A", "B"):
        payload = _payload(arm, directory, config, block="full_run",
                           ids=SHARD_IDS, n_sims=SHARD_N_SIMS)
        payload["run"]["config_shard"] = index
        if mutate is not None:
            mutate(payload)
        arms.append(load_arm(_write(tmp_path, arm, payload)))
    return arms


def test_a_shard_run_binds_to_its_shard_entry(tmp_path):
    assert audit(_shard_arms(tmp_path)) == []


def test_a_shard_run_whose_fixture_dir_hash_moved_fails(tmp_path):
    directory, config = _shard_config(tmp_path)

    def moved(payload):
        payload["run"]["fixture_dir_hash"] = "0" * 32

    failures = audit(_shard_arms(tmp_path, config=config, directory=directory,
                                 mutate=moved))
    assert any("fixture_dir_hash" in message
               and "full_run shard 0 block pins" in message
               for message in failures), failures


def test_a_shard_run_whose_fixture_count_moved_fails(tmp_path):
    directory, config = _shard_config(tmp_path)

    def moved(payload):
        payload["run"]["fixture_count"] = 255

    failures = audit(_shard_arms(tmp_path, config=config, directory=directory,
                                 mutate=moved))
    assert any("fixture_count" in message
               and "full_run shard 0 block registers" in message
               for message in failures), failures


def test_a_shard_run_claiming_the_whole_set_fails(tmp_path):
    """Without the shard binding this read the 255-fixture pins and passed."""
    directory, config = _shard_config(tmp_path)

    def unshard(payload):
        payload["run"]["config_shard"] = None

    failures = audit(_shard_arms(tmp_path, config=config, directory=directory,
                                 mutate=unshard))
    assert any("fixture_dir" in message and "full_run block registers" in
               message for message in failures), failures


def test_an_unregistered_shard_index_fails(tmp_path):
    directory, config = _shard_config(tmp_path)

    def relabel(payload):
        payload["run"]["config_shard"] = 7

    failures = audit(_shard_arms(tmp_path, config=config, directory=directory,
                                 mutate=relabel))
    assert any("config_shard" in message and "not a registered shard" in
               message for message in failures), failures


def test_two_arms_on_different_shards_fail(tmp_path):
    directory, config = _shard_config(tmp_path)
    arms = _shard_arms(tmp_path, config=config, directory=directory)
    arms[1].run["config_shard"] = 1
    failures = audit(arms)
    assert any("config_shard" in message and "arm B" in message
               for message in failures), failures
