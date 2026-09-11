"""Tests for the stage-2 provenance pin (D3 check 3.2, 3.13).

Three things have to hold for the pinned block to mean anything:

* every recorded fact is compared, and a drift in ANY of them is reported by
  name — the frame's declaration and parquet identity, the stats cache, the
  feature order, the base logits, the source closure, the cohort freeze and
  the config's own body hash;
* the two fields that change without the config's meaning changing
  (`pins_generated_at`, `git_head_short`) are recorded and NOT compared;
* the cohort freeze is immutable once it exists: a `FROZEN.json` that changes
  after the pin fails `--verify`, because the untouched cohort is opened once.

The drift cases come in two flavours. The synthetic ones drive
`diff_provenance` directly and touch no artifact. The ones marked
`needs_artifacts` tamper with a COPY of the committed config and let
`verify()` recompute the truth from the repository's own frame, cache and
base logits.
"""
from __future__ import annotations

import copy
import json

import pytest
import yaml

from sequence_track import pin_stage2
from sequence_track.pin_stage2 import (
    COHORT_NOT_FROZEN,
    NOT_VERIFIED,
    PinError,
    build_provenance,
    cohort_block,
    config_body_sha256,
    diff_provenance,
    flatten,
    load_config,
    main,
    verify,
)

CONFIG_PATH = pin_stage2.CONFIG_PATH


# ---------------------------------------------------------------------------
# the diff machinery: one synthetic drift case per pinned field
# ---------------------------------------------------------------------------

BASELINE = {
    "pins_generated_at": "2026-09-11T00:00:00Z",
    "git_head_short": "abc1234",
    "not_verified": list(NOT_VERIFIED),
    "config_body_sha256": "0" * 64,
    "frame": {
        "dir": "data/xgb_data_i7",
        "version": "i7",
        "feature_hash": {"version": "i7", "n_features": 114,
                         "venue_alias_version": "venue_aliases_v1"},
        "splits": {
            "train": {"md5": "a" * 32, "n_rows": 1876971,
                      "match_date_min": "2005-02-17",
                      "match_date_max": "2024-12-30"},
            "validation": {"md5": "b" * 32, "n_rows": 124292,
                           "match_date_min": "2024-12-31",
                           "match_date_max": "2025-06-29"},
            "test": {"md5": "c" * 32},
        },
    },
    "stats_cache": {"role": "stats_cache_i7", "md5": "d" * 32,
                    "venue_alias_version": "venue_aliases_v1"},
    "features": {"count": 50, "names": ["a", "b"], "sha256": "e" * 64},
    "base_logits": {"splits": {"train": {"npz_md5": "f" * 32,
                                         "parquet_md5": "a" * 32}},
                    "train_validation_digest": "1" * 32},
    "sources": {"count": 22, "source_sha256": {"scripts/transformer_t1.py":
                                              "2" * 64},
                "source_sha256_absent": []},
    "cohort": {"status": "frozen", "frozen_json_sha256": "3" * 64,
               "eligible_match_count": 210},
}


def _mutate(path_keys, value):
    payload = copy.deepcopy(BASELINE)
    node = payload
    for key in path_keys[:-1]:
        node = node[key]
    node[path_keys[-1]] = value
    return payload


@pytest.mark.parametrize("keys,value", [
    (["config_body_sha256"], "9" * 64),
    (["frame", "version"], "v3"),
    (["frame", "feature_hash", "n_features"], 50),
    (["frame", "splits", "train", "md5"], "9" * 32),
    (["frame", "splits", "train", "n_rows"], 1876970),
    (["frame", "splits", "validation", "match_date_max"], "2025-06-30"),
    (["frame", "splits", "test", "md5"], "9" * 32),
    (["stats_cache", "md5"], "9" * 32),
    (["stats_cache", "venue_alias_version"], "venue_aliases_v2"),
    (["features", "count"], 49),
    (["features", "sha256"], "9" * 64),
    (["base_logits", "train_validation_digest"], "9" * 32),
    (["base_logits", "splits", "train", "npz_md5"], "9" * 32),
    (["sources", "count"], 7),
    (["cohort", "frozen_json_sha256"], "9" * 64),
    (["cohort", "eligible_match_count"], 209),
])
def test_every_pinned_field_is_compared_and_named(keys, value):
    problems = diff_provenance(BASELINE, _mutate(keys, value))
    expected_key = "provenance." + ".".join(str(key) for key in keys)
    assert problems, expected_key
    assert any(expected_key in problem for problem in problems), problems


def test_a_dropped_field_and_an_added_field_are_both_reported():
    dropped = copy.deepcopy(BASELINE)
    dropped["stats_cache"].pop("md5")
    assert any("MISSING  provenance.stats_cache.md5" in problem
               for problem in diff_provenance(BASELINE, dropped))
    added = copy.deepcopy(BASELINE)
    added["stats_cache"]["extra"] = 1
    assert any("EXTRA    provenance.stats_cache.extra" in problem
               for problem in diff_provenance(BASELINE, added))


def test_a_reordered_feature_list_is_a_difference():
    permuted = _mutate(["features", "names"], ["b", "a"])
    problems = diff_provenance(BASELINE, permuted)
    assert any("provenance.features.names[0]" in problem
               for problem in problems)


def test_timestamp_and_head_are_recorded_but_not_compared():
    moved = copy.deepcopy(BASELINE)
    moved["pins_generated_at"] = "2027-01-01T00:00:00Z"
    moved["git_head_short"] = "deadbee"
    assert diff_provenance(BASELINE, moved) == []
    assert NOT_VERIFIED == ("provenance.pins_generated_at",
                           "provenance.git_head_short")


def test_empty_containers_are_their_own_leaf():
    flat = flatten({"a": [], "b": {}}, "provenance")
    assert flat["provenance.a"] == "<empty list>"
    assert flat["provenance.b"] == "<empty mapping>"
    # an empty list that became an empty mapping must not slip past
    assert diff_provenance({"a": []}, {"a": {}}) != []


# ---------------------------------------------------------------------------
# the cohort freeze
# ---------------------------------------------------------------------------

def test_cohort_before_the_freeze(tmp_path):
    block = cohort_block(tmp_path / "FROZEN.json")
    assert block["status"] == COHORT_NOT_FROZEN
    assert "frozen_json_sha256" not in block


def test_cohort_frozen_then_changed_is_a_refusal(tmp_path):
    frozen = tmp_path / "FROZEN.json"
    frozen.write_text(json.dumps({"eligible_match_count": 210,
                                  "eligible_match_ids": ["1", "2"],
                                  "frozen_at_utc": "2026-09-11T17:23:20Z"}))
    pinned = cohort_block(frozen)
    assert pinned["status"] == "frozen"
    assert pinned["eligible_match_count"] == 210
    assert pinned["eligible_ids_recorded"] == 2

    # someone re-runs the builder and the eligible set moves
    frozen.write_text(json.dumps({"eligible_match_count": 209,
                                  "eligible_match_ids": ["1"],
                                  "frozen_at_utc": "2026-09-12T09:00:00Z"}))
    problems = diff_provenance({"cohort": pinned},
                               {"cohort": cohort_block(frozen)})
    assert any("provenance.cohort.frozen_json_sha256" in problem
               for problem in problems), problems


def test_freezing_after_the_pin_forces_a_repin(tmp_path):
    frozen = tmp_path / "FROZEN.json"
    before = cohort_block(frozen)
    frozen.write_text(json.dumps({"eligible_match_count": 1,
                                  "eligible_match_ids": ["1"]}))
    assert diff_provenance({"cohort": before},
                           {"cohort": cohort_block(frozen)}) != []


# ---------------------------------------------------------------------------
# the config body hash
# ---------------------------------------------------------------------------

def test_config_body_hash_ignores_the_provenance_block():
    body = {"experiment": {"name": "seq_stage2_v1"}, "training": {"seeds": [7]}}
    with_block = dict(body, provenance={"pins_generated_at": "x"})
    assert config_body_sha256(body) == config_body_sha256(with_block)


def test_config_body_hash_moves_when_the_seed_list_moves():
    body = {"training": {"seeds": [7, 13]}}
    assert config_body_sha256(body) != config_body_sha256(
        {"training": {"seeds": [7, 13, 29]}})


def test_load_config_refuses_a_missing_file(tmp_path):
    with pytest.raises(PinError):
        load_config(tmp_path / "nope.yaml")


def test_verify_reports_a_config_with_no_provenance_block(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"experiment": {"name": "x"}}))
    problems = verify(path)
    assert len(problems) == 1
    assert "carries no provenance block" in problems[0]


# ---------------------------------------------------------------------------
# the real tree
# ---------------------------------------------------------------------------

@pytest.mark.needs_artifacts
def test_verify_passes_on_the_committed_config():
    assert CONFIG_PATH.is_file()
    assert verify(CONFIG_PATH) == []
    assert main(["--verify"]) == 0


@pytest.mark.needs_artifacts
@pytest.mark.parametrize("keys", [
    ["frame", "splits", "train", "md5"],
    ["frame", "feature_hash", "version"],
    ["stats_cache", "md5"],
    ["features", "sha256"],
    ["base_logits", "train_validation_digest"],
    ["sources", "source_sha256", "scripts/transformer_t1.py"],
    ["cohort", "frozen_json_sha256"],
    ["config_body_sha256"],
])
def test_tampered_field_fails_verify_and_names_it(tmp_path, keys):
    config = load_config(CONFIG_PATH)
    node = config["provenance"]
    for key in keys[:-1]:
        node = node[key]
    node[keys[-1]] = "tampered"
    path = tmp_path / "seq_stage2_v1.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    problems = verify(path)
    expected_key = "provenance." + ".".join(keys)
    assert any(expected_key in problem for problem in problems), problems
    assert main(["--verify", "--config", str(path)]) == 1


@pytest.mark.needs_artifacts
def test_an_edited_body_key_fails_verify(tmp_path):
    config = load_config(CONFIG_PATH)
    config["training"]["seeds"] = [7, 13, 29]
    path = tmp_path / "seq_stage2_v1.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    problems = verify(path)
    assert any("provenance.config_body_sha256" in problem
               for problem in problems), problems


@pytest.mark.needs_artifacts
def test_provenance_block_covers_every_recomputed_group():
    provenance = load_config(CONFIG_PATH)["provenance"]
    assert set(provenance) >= {
        "pins_generated_at", "git_head_short", "not_verified", "pinned_by",
        "config_body_sha256", "frame", "stats_cache", "features",
        "base_logits", "sources", "cohort", "runtime"}
    assert provenance["features"]["count"] == 50
    assert len(provenance["features"]["names"]) == 50
    assert provenance["frame"]["dir"] == "data/xgb_data_i7"
    assert set(provenance["frame"]["splits"]) == {"train", "validation",
                                                 "test"}
    # the test split is hashed and nothing more
    assert "n_rows" not in provenance["frame"]["splits"]["test"]
    for source in ("scripts/transformer_t1.py",
                   "scripts/sequence_track/recurrent_arms.py",
                   "scripts/sequence_track/build_base_logits.py",
                   "scripts/sequence_track/build_cohort_stage2.py",
                   "scripts/sequence_track/retrain_stage2.py",
                   "scripts/sequence_track/pin_stage2.py",
                   # Astra SHOULD 3: the imported modules that move stage-2
                   # behaviour without any stage-2 script changing.
                   "scripts/embeddings_e1.py",
                   "scripts/artifacts.py",
                   "scripts/calibration.py",
                   "scripts/registered_experiment.py",
                   "scripts/materialize_features.py",
                   "scripts/build_stats_cache.py",
                   "scripts/feature_registry.py",
                   "scripts/parsing_v2.py",
                   "scripts/tracker_rehydration.py",
                   "scripts/stats_provider.py",
                   "scripts/stats_sqlite_backend.py",
                   "scripts/loaders_common.py",
                   "scripts/identity_maps.py",
                   "scripts/elo_update.py",
                   "scripts/player_metadata.py"):
        assert source in provenance["sources"]["source_sha256"]


def test_the_closure_covers_what_the_stage_2_paths_import():
    """The closure must not claim more than it covers (Astra SHOULD 3).

    Every top-level `scripts/` module imported by a stage-2 script has to be
    in the closure, or a change to it moves stage-2 behaviour while --verify
    still passes.
    """
    import ast

    closure = set(pin_stage2.PROVENANCE_SOURCE_CLOSURE)
    repo_root = pin_stage2.REPO_ROOT
    scripts_dir = repo_root / "scripts"
    stage2_paths = [
        "scripts/transformer_t1.py",
        "scripts/sequence_track/recurrent_arms.py",
        "scripts/sequence_track/build_base_logits.py",
        "scripts/sequence_track/build_cohort_stage2.py",
        "scripts/sequence_track/retrain_stage2.py",
        "scripts/sequence_track/ownership_dependency_test.py",
    ]
    missing = []
    for relative in stage2_paths:
        tree = ast.parse((repo_root / relative).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                head = name.split(".")[0]
                candidate = "scripts/%s.py" % head
                if (scripts_dir / ("%s.py" % head)).is_file():
                    if candidate not in closure:
                        missing.append((relative, candidate))
    assert missing == [], missing


@pytest.mark.needs_artifacts
def test_base_logits_sidecars_agree_with_the_live_parquets():
    # build_provenance itself refuses a stale base-logits build; this asserts
    # the recorded parquet md5s are the live ones rather than trusting it.
    provenance = build_provenance(
        {key: value for key, value in load_config(CONFIG_PATH).items()
         if key != "provenance"})
    splits = provenance["frame"]["splits"]
    for split, record in provenance["base_logits"]["splits"].items():
        assert record["parquet_md5"] == splits[split]["md5"]
