"""Tests for the stage-2 multi-seed driver (D3 check 3.12, 3.13).

Four things have to hold for a stage-2 night to mean anything:

* the registered config validates, and validates for the reasons the arm
  register gives: sixteen ids, one queue order each, the trainer's own wiring
  and history tables, and `--k` / `--base-logits-dir` exactly where the
  trainer accepts them;
* `--seeds` can narrow a configuration's seed list and can never widen it, so
  a candidate cannot end the night with more seeds than its control;
* a completed checkpoint is reused only when its `arm_params` and config id
  are this invocation's. Sixteen configurations share one architecture and
  one optimiser, so `k`, the wiring, the history input, the bias, the
  residual L2 and the base-logits digest are what distinguish them;
* the argv the driver runs is the argv the config registers.

Astra's stage-2 code review added four more, each with its own section below:

* the seed-independent TRAINING SIGNATURE (MUST-FIX 2/11) covers the
  configuration, the whole `arm_params` identity including `key_construction`,
  the frame, the cache, the base logits and the sha256 of the implementation
  files the arm actually runs; a reuse whose stored signature differs is
  refused naming the component that moved;
* preflight COMPARES the config's registered `provenance` block instead of
  only measuring (MUST-FIX 2), refuses when it is absent, and verifies with
  NO cohort artifact on disk, because the mini has none;
* a run is complete only with all four artefacts, an interrupted one is
  retried rather than refused or silently accepted, and the completion record
  is written atomically (MUST-FIX 3);
* peak RSS is sampled per child with its method recorded, and the earlier
  `RUSAGE_CHILDREN` delta observations are marked invalid rather than
  reinterpreted or rewritten (MUST-FIX 4).

Astra's round-2 review (and two round-1 items that were still open) added
four more:

* the signature is MACHINE- AND CHECKOUT-INDEPENDENT (round-2 MUST-FIX 1):
  seed 7 trains from a git worktree whose frame and cache are symlinks into
  the main checkout, so the signature hashes the configured frame spelling
  plus content hashes, never a resolved path. Two fake checkout roots, one
  with the frame as a real directory and one with it as a symlink to the
  first, must produce one signature and must consolidate;
* `--consolidate` is verification-and-summary only (round-2 MUST-FIX 3): a
  half-transferred seed refuses and is named, nothing is deleted and nothing
  is trained, so recovery stays on the machine that owns that seed;
* every run records its MACHINE while it runs (round-2 MUST-FIX 4, D8.8):
  hostname, label, chip, OS, python/torch, device, all four thread caps, the
  repository root and which paths were symlinks;
* the training sources include the trainer's imported feature contract and
  manifest resolver, and they are compared against the pin's REGISTERED
  hashes (round-1 MUST-FIX 2); `model.pt` is loaded and its recorded manifest
  enforced on reuse, with absent / malformed / disagreeing `COMPLETE.json`
  distinguished (round-1 MUST-FIX 3).

The last section covers the split-night flow the fixes exist for: seed 7
trained on the laptop, seed 13 on the mini, rsynced into one tree, then one
`--consolidate --seeds 7,13` invocation that verifies and reuses both and
rewrites `summary.yaml` over the two.

Nothing here touches a repository artifact except the `needs_artifacts` test
that compares the committed pin: every checkpoint is synthetic and every path
is under pytest's tmp_path. The one other exception is READING the committed
config file, which is a text read of a registered file. No test trains.
"""
from __future__ import annotations

import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

import transformer_t1 as t1
from sequence_track import retrain_stage2 as driver
from sequence_track.retrain_stage2 import (
    COMPLETION_RECORD,
    RUN_ARTIFACTS,
    RetrainError,
    build_summary,
    command_for,
    configurations,
    effective_settings,
    load_config,
    narrow_seeds,
    registered_commands,
    verify_checkpoint,
)

CONFIG_PATH = driver.DEFAULT_CONFIG
REGISTERED_ORDER = [
    "mlp", "full", "fixed_decay", "fox", "aligned_hist", "recency_k30",
    "same_entity_k30", "aligned_hist_rf", "same_entity_k0", "same_entity_k6",
    "same_entity_k12", "same_entity_unr", "lstm", "xlstm", "residual_mlp",
    "residual_t1",
]


@pytest.fixture(scope="module")
def config() -> dict:
    return load_config(CONFIG_PATH)


# ---------------------------------------------------------------------------
# configuration validation
# ---------------------------------------------------------------------------

def test_sixteen_configurations_in_the_registered_order(config):
    assert list(configurations(config)) == REGISTERED_ORDER
    assert [int(entry["queue_order"])
            for entry in sorted(config["configurations"],
                                key=lambda row: int(row["queue_order"]))
            ] == list(range(1, 17))


def test_training_block_and_seeds(config):
    training = config["training"]
    assert training["seeds"] == [7, 13]
    assert (training["dmodel"], training["layers"], training["heads"]) == (
        128, 2, 4)
    assert (training["batch"], training["epochs"], training["patience"]) == (
        128, 30, 3)
    assert training["learning_rate"] == 0.0003
    assert training["device"] == "mps"
    assert training["aux"] is False
    assert training["no_kit"] is True
    assert training["score_test"] is False
    assert training["save_predictions"] is True
    assert "seed_policy" in training


def test_params_match_the_trainer_flag_contract(config):
    for config_id, entry in configurations(config).items():
        params = entry["_params"]
        arm = entry["arm"]
        assert (params["k"] is not None) == (arm in t1.ARMS_NEEDING_K), (
            config_id)
        assert bool(params["base_logits_dir"]) == (
            arm in t1.ARMS_NEEDING_BASE_LOGITS), config_id
        if arm in t1.ARMS_NEEDING_BASE_LOGITS:
            assert params["residual_l2"] == pytest.approx(0.001)
        else:
            assert params["residual_l2"] is None
        assert entry["wiring"] == t1.ARM_WIRING[arm]
        assert entry["history_input"] == t1.ARM_HISTORY[arm]


def test_access_row_is_declared_for_every_configuration(config):
    for config_id, entry in configurations(config).items():
        access = entry["access"]
        assert set(access) == {"features", "history", "identity",
                               "prod_logits"}, config_id
        assert access["features"] is True
        assert access["prod_logits"] is (
            entry["arm"] in t1.ARMS_NEEDING_BASE_LOGITS)
        assert access["history"] is (
            t1.ARM_HISTORY[entry["arm"]] != "none")


def test_k_sweep_cells_are_the_registered_windows(config):
    entries = configurations(config)
    assert entries["same_entity_k0"]["_params"]["k"] == 0
    assert entries["same_entity_k6"]["_params"]["k"] == 6
    assert entries["same_entity_k12"]["_params"]["k"] == 12
    assert entries["same_entity_k30"]["_params"]["k"] == 30
    assert entries["same_entity_unr"]["_params"]["k"] == "unr"
    assert entries["recency_k30"]["_params"]["k"] == 30


def _write_config(tmp_path: Path, config: dict, mutate) -> Path:
    payload = copy.deepcopy(config)
    for entry in payload["configurations"]:
        entry.pop("_params", None)
    mutate(payload)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return path


@pytest.mark.parametrize("mutate,fragment", [
    (lambda c: c["configurations"].__setitem__(
        0, {**c["configurations"][0], "id": "full"}), "duplicate config id"),
    (lambda c: c["configurations"][0].__setitem__("arm", "nope"),
     "unknown arm"),
    (lambda c: c["configurations"][0].__setitem__("wiring", "standard"),
     "is not the trainer's"),
    (lambda c: c["configurations"][0].__setitem__(
        "history_input", "innings_previous"), "is not the trainer's"),
    (lambda c: c["configurations"][0].__setitem__("params", {"k": 6}),
     "params.k is not accepted"),
    (lambda c: c["configurations"][5].__setitem__("params", {}),
     "requires params.k"),
    (lambda c: c["configurations"][14].__setitem__("params", {}),
     "requires params.base_logits_dir"),
    (lambda c: c["configurations"][0].__setitem__("queue_order", 99),
     "queue_order"),
    (lambda c: c["training"].__setitem__("seeds", [7, 7]), "duplicate seed"),
    (lambda c: c["training"].__setitem__("seeds", [7, 8]),
     "outside the registered pool"),
    (lambda c: c["training"].__setitem__("score_test", True),
     "score_test' must be false"),
    (lambda c: c["training"].__setitem__("save_predictions", False),
     "save_predictions' must be true"),
    (lambda c: c["training"].__setitem__("no_kit", False),
     "no_kit' must be true"),
])
def test_config_refusals(tmp_path, config, mutate, fragment):
    path = _write_config(tmp_path, config, mutate)
    with pytest.raises(RetrainError) as error:
        load_config(path)
    assert fragment in str(error.value)


# ---------------------------------------------------------------------------
# seed narrowing
# ---------------------------------------------------------------------------

def test_seeds_may_narrow():
    assert narrow_seeds([7, 13], None) == [7, 13]
    assert narrow_seeds([7, 13], [13]) == [13]
    # order follows the registered list, not the command line
    assert narrow_seeds([7, 13], [13, 7]) == [7, 13]


def test_seeds_may_never_widen():
    with pytest.raises(RetrainError) as error:
        narrow_seeds([7, 13], [7, 29])
    assert "would widen the seed list" in str(error.value)
    assert "[29]" in str(error.value)


def test_main_refuses_a_widening_seed_flag(config):
    with pytest.raises(RetrainError):
        driver.main(["--config", str(CONFIG_PATH), "--seeds", "29",
                     "--dry-run"])


# ---------------------------------------------------------------------------
# argv rendering
# ---------------------------------------------------------------------------

def test_argv_carries_k_and_base_logits_exactly_where_the_trainer_wants(
        config):
    out_root = Path("/tmp/out")
    for config_id, entry in configurations(config).items():
        effective = effective_settings(config, entry)
        argv = command_for(config, entry, 7,
                           out_root / config_id / "seed_7", effective)
        assert argv[1].endswith("transformer_t1.py")
        assert ("--k" in argv) == (entry["arm"] in t1.ARMS_NEEDING_K)
        assert ("--base-logits-dir" in argv) == (
            entry["arm"] in t1.ARMS_NEEDING_BASE_LOGITS)
        assert ("--residual-l2" in argv) == (
            entry["arm"] in t1.ARMS_NEEDING_BASE_LOGITS)
        # every run, without exception
        assert "--no-kit" in argv
        assert "--save-predictions" in argv
        assert "--score-test" not in argv
        assert "--aux" not in argv
        assert argv[argv.index("--arm") + 1] == entry["arm"]
        assert argv[argv.index("--seed") + 1] == "7"
        assert argv[argv.index("--epochs") + 1] == "30"
        assert argv[argv.index("--stats-cache-role") + 1] == "stats_cache_i7"


def test_registered_commands_match_the_config(config):
    out_root = driver.REPO / config["outputs"]["directory"]
    for config_id, entry in configurations(config).items():
        assert entry["command"] == registered_commands(
            config, entry, out_root), config_id
        assert len(entry["command"]) == len(config["training"]["seeds"])
        for command in entry["command"]:
            assert command.startswith("uv run --no-sync python "
                                      "scripts/transformer_t1.py ")
            assert (f"--out models/embeddings/seq_stage2/runs/{config_id}/"
                    "seed_") in command


def test_epoch_override_changes_the_argv(config):
    entry = configurations(config)["mlp"]
    effective = effective_settings(config, entry, epochs=1)
    argv = command_for(config, entry, 7, Path("/tmp/x"), effective)
    assert argv[argv.index("--epochs") + 1] == "1"


# ---------------------------------------------------------------------------
# reuse: arm_params and config id
# ---------------------------------------------------------------------------

SPLIT_FILES = {
    "train": {"path": "data/xgb_data_i7/cricket_data_i7_train.parquet",
              "md5": "a" * 32, "n_rows": 10, "match_date_min": "2005-02-17",
              "match_date_max": "2024-12-30"},
    "validation": {
        "path": "data/xgb_data_i7/cricket_data_i7_validation.parquet",
        "md5": "b" * 32, "n_rows": 5, "match_date_min": "2024-12-31",
        "match_date_max": "2025-06-29"},
}
FEATURE_HASH = {"version": "i7", "venue_alias_version": "venue_aliases_v1"}


PROVENANCE_CHECK = {"pinned_by": "scripts/sequence_track/pin_stage2.py",
                    "pins_generated_at": "2026-09-11T17:57:48Z",
                    "config_body_sha256": "7" * 64,
                    "compared": ["frame.splits"], "skipped": {}}


def _state_dict(arm: str = "mlp") -> dict:
    """A real, minimal state dict for `model.pt`.

    Astra round-1 MUST-FIX 3: `b"not a real checkpoint"` used to pass, so a
    corrupted transfer could be admitted. The driver now LOADS the file, so a
    synthetic checkpoint has to be a genuine one — with the parameter names
    the arm's own wiring produces.
    """
    import torch

    if t1.ARM_WIRING[arm] == "recurrent":
        return {"recurrent.cell.weight": torch.zeros(2, 2),
                "recurrent.head.bias": torch.zeros(2)}
    return {"feat_proj.weight": torch.zeros(2, 2),
            "feat_proj.bias": torch.zeros(2),
            "head.weight": torch.zeros(6, 2),
            "head.bias": torch.zeros(6)}


def _write_state_dict(path: Path, arm: str = "mlp") -> None:
    import torch

    torch.save(_state_dict(arm), path)


def _resolved() -> dict:
    return {
        "frame_dir": driver.REPO / "data/xgb_data_i7",
        # Astra round-2 MUST-FIX 1: the configured spelling, which is what the
        # signature hashes; identical on the laptop worktree and the mini.
        "frame_dir_configured": "data/xgb_data_i7",
        "frame_version": "i7",
        "feature_hash": dict(FEATURE_HASH),
        "split_files": copy.deepcopy(SPLIT_FILES),
        "stats_cache": {"role": "stats_cache_i7", "md5": "c" * 32},
        "base_logits": {"dir": "models/embeddings/seq_stage2/base_logits",
                        "digest": "d" * 32, "splits": {}},
        "provenance": copy.deepcopy(PROVENANCE_CHECK),
    }


def _checkpoint(tmp_path: Path, config: dict, entry: dict, seed: int,
                effective: dict, *, arm_params_patch=None,
                config_id=None, resolved=None, signature=None,
                artifacts=RUN_ARTIFACTS, completion=False, ll=None,
                machine=None) -> Path:
    """A synthetic finished run: all four artefacts plus its signature.

    `artifacts` narrows the set, which is how an interrupted run is
    reproduced (Astra MUST-FIX 3).
    """
    if signature is None:
        signature = driver.signature_for(effective,
                                         resolved or _resolved())
    out_dir = tmp_path / str(entry["id"]) / f"seed_{seed}"
    out_dir.mkdir(parents=True)
    arm_params = dict(effective["arm_params"])
    arm_params.update({"positional_embedding": True, "n_parameters": 123})
    if arm_params_patch:
        arm_params.update(arm_params_patch)
    metrics = {
        "arm_params": arm_params,
        "validation_ll": 1.234567890123 if ll is None else ll,
        "training_contract": {
            "contract_version": t1.CONTRACT_VERSION,
            "frame_dir": "data/xgb_data_i7",
            "frame_version": "i7",
            "feature_hash": dict(FEATURE_HASH),
            "split_files": copy.deepcopy(SPLIT_FILES),
            "kit_used": False,
            "test_split_scored": False,
            "stats_cache": {"role": "stats_cache_i7", "md5": "c" * 32},
            "architecture": {**effective["arch"], "arm": entry["arm"]},
            "optimiser": dict(effective["optimiser"]),
            "seed": seed,
            "best_epoch": 4,
        },
    }
    written = {
        "metrics.json": lambda p: p.write_text(json.dumps(metrics)),
        # Astra round-1 MUST-FIX 3: a real state dict, because the driver
        # loads it and checks the arm's parameter names.
        "model.pt": lambda p: _write_state_dict(p, str(entry["arm"])),
        "run_record.json": lambda p: p.write_text(json.dumps({
            "config_id": config_id or str(entry["id"]),
            "overrides": dict(effective["overrides"]),
            "wall_seconds": 12.5,
            # Astra round-2 MUST-FIX 4: the machine that trained this seed.
            "machine_provenance": machine,
            "peak_rss_bytes_sampled": 3 * 1024 ** 3,
            "peak_rss_bytes_valid": True,
            "rss_method": "psutil_process_tree_sampled",
            "peak_rss_unit": "bytes",
            "omp_num_threads": "4",
            "training_signature": signature["training_signature"],
            "training_signature_components":
                signature["training_signature_components"],
        })),
        f"predictions_{driver.SELECTION_SPLIT}.npz":
            lambda p: np.savez_compressed(p, probs=np.zeros((5, 6),
                                                            dtype=np.float32)),
    }
    for name in artifacts:
        written[name](out_dir / name)
    if completion:
        driver.write_completion_record(
            out_dir, seed, config_id or str(entry["id"]), signature, machine,
            str(entry["arm"]))
    return out_dir


def test_matching_checkpoint_is_reused(tmp_path, config):
    entry = configurations(config)["same_entity_k30"]
    effective = effective_settings(config, entry, base_logits_digest="d" * 32)
    out_dir = _checkpoint(tmp_path, config, entry, 7, effective)
    contract = verify_checkpoint(out_dir, json.loads(
        (out_dir / "metrics.json").read_text()), config, _resolved(), entry, 7,
        effective)
    assert contract["best_epoch"] == 4


@pytest.mark.parametrize("patch,fragment", [
    ({"k": 6}, "arm_params.k"),
    ({"wiring": "standard"}, "arm_params.wiring"),
    ({"history_input": "innings_previous"}, "arm_params.history_input"),
    ({"bias": "alibi"}, "arm_params.bias"),
    ({"residual_l2": 0.01}, "arm_params.residual_l2"),
    ({"base_logits_md5": "e" * 32}, "arm_params.base_logits_md5"),
    ({"arm": "recency"}, "arm_params.arm"),
])
def test_arm_params_drift_refuses_reuse(tmp_path, config, patch, fragment):
    entry = configurations(config)["same_entity_k30"]
    effective = effective_settings(config, entry, base_logits_digest="d" * 32)
    out_dir = _checkpoint(tmp_path, config, entry, 7, effective,
                          arm_params_patch=patch)
    with pytest.raises(RetrainError) as error:
        verify_checkpoint(out_dir, json.loads(
            (out_dir / "metrics.json").read_text()), config, _resolved(),
            entry, 7, effective)
    assert fragment in str(error.value)
    assert "refusing to reuse" in str(error.value)


def test_residual_arm_pins_the_base_logits_digest(tmp_path, config):
    entry = configurations(config)["residual_t1"]
    effective = effective_settings(config, entry, base_logits_digest="d" * 32)
    assert effective["arm_params"]["base_logits_md5"] == "d" * 32
    out_dir = _checkpoint(tmp_path, config, entry, 13, effective)
    # a rebuilt base-logits directory changes the digest and must refuse
    rebuilt = effective_settings(config, entry, base_logits_digest="f" * 32)
    with pytest.raises(RetrainError) as error:
        verify_checkpoint(out_dir, json.loads(
            (out_dir / "metrics.json").read_text()), config, _resolved(),
            entry, 13, rebuilt)
    assert "arm_params.base_logits_md5" in str(error.value)


def test_non_residual_arm_records_no_base_logits_digest(config):
    entry = configurations(config)["mlp"]
    effective = effective_settings(config, entry, base_logits_digest="d" * 32)
    assert effective["arm_params"]["base_logits_md5"] is None
    assert effective["arm_params"]["residual_l2"] is None


def test_config_id_drift_refuses_reuse(tmp_path, config):
    entry = configurations(config)["same_entity_k6"]
    effective = effective_settings(config, entry)
    out_dir = _checkpoint(tmp_path, config, entry, 7, effective,
                          config_id="same_entity_k12")
    with pytest.raises(RetrainError) as error:
        verify_checkpoint(out_dir, json.loads(
            (out_dir / "metrics.json").read_text()), config, _resolved(),
            entry, 7, effective)
    assert "config_id" in str(error.value)


def test_missing_arm_params_block_refuses_reuse(tmp_path, config):
    entry = configurations(config)["mlp"]
    effective = effective_settings(config, entry)
    out_dir = _checkpoint(tmp_path, config, entry, 7, effective)
    metrics = json.loads((out_dir / "metrics.json").read_text())
    metrics.pop("arm_params")
    with pytest.raises(RetrainError) as error:
        verify_checkpoint(out_dir, metrics, config, _resolved(), entry, 7,
                          effective)
    assert "no arm_params block" in str(error.value)


# --- the stage 4 identity fields (rungs 4b / C114 / 4d) --------------------
# The trainer records these nested inside its own blocks and under its own
# names; `expected_arm_params` carries them flat. Each kind is exercised with
# a recorded value that agrees and one that has drifted, and a configuration
# that sets none of them must see byte-identical behaviour.

STAGE4_CASES = [
    # (kind, expected flat keys, recorded block, drifted recorded block,
    #  the fragment the problem must name)
    ("4b_base_probs_digests",
     {"base_probs_sha256": {"train": "a" * 64, "validation": "b" * 64}},
     {"base_probs": {"dir": "models/embeddings/stage4/refs", "name": "eb_ctx",
                     "sha256_by_split": {"train": "a" * 64,
                                         "validation": "b" * 64}}},
     {"base_probs": {"dir": "models/embeddings/stage4/refs", "name": "eb_ctx",
                     "sha256_by_split": {"train": "a" * 64,
                                         "validation": "c" * 64}}},
     "arm_params.base_probs.sha256_by_split"),
    ("4b_residual_lambda",
     {"residual_lambda": 0.001},
     {"base_probs": {"residual_lambda": 0.001}},
     {"base_probs": {"residual_lambda": 0.01}},
     "arm_params.base_probs.residual_lambda"),
    ("c114_feature_contract",
     {"feature_contract": "v7_114", "feature_contract_n": 114,
      "feature_contract_sha256": "d" * 64},
     {"feature_contract": {"name": "v7_114", "n_features": 114,
                           "feature_names_sha256": "d" * 64}},
     {"feature_contract": {"name": "v7_114", "n_features": 114,
                           "feature_names_sha256": "e" * 64}},
     "arm_params.feature_contract.feature_names_sha256"),
    ("4d_extra_features",
     {"extra_features": {"dir": "models/embeddings/stage4/exposure",
                         "cols": ["batter_N_asof"], "col_set": "exposure",
                         "sha256": {"train": "f" * 64,
                                    "validation": "0" * 64}}},
     {"extra_features": {"dir": "models/embeddings/stage4/exposure",
                         "n_cols": 1,
                         "sha256_by_split": {"train": "f" * 64,
                                             "validation": "0" * 64}}},
     {"extra_features": {"dir": "models/embeddings/stage4/exposure",
                         "n_cols": 1,
                         "sha256_by_split": {"train": "f" * 64,
                                             "validation": "1" * 64}}},
     "arm_params.extra_features.sha256_by_split"),
]


def _stage4_effective(config, want_patch):
    """A stage 2 `effective` whose arm_params also carry a stage 4 field."""
    entry = configurations(config)["mlp"]
    effective = effective_settings(config, entry)
    effective["arm_params"] = {**effective["arm_params"], **want_patch}
    return effective


def _stage4_recorded(effective, block_patch):
    """What the trainer's metrics.json would carry for that run."""
    recorded = {key: value for key, value in effective["arm_params"].items()
                if key in driver.ARM_PARAM_FIELDS}
    recorded.update({"positional_embedding": False, "n_parameters": 123})
    recorded.update(copy.deepcopy(block_patch))
    return {"arm_params": recorded}


@pytest.mark.parametrize(
    "want_patch,block", [(case[1], case[2]) for case in STAGE4_CASES],
    ids=[case[0] for case in STAGE4_CASES])
def test_stage4_identity_field_matching_recorded_value_passes(
        config, want_patch, block):
    effective = _stage4_effective(config, want_patch)
    metrics = _stage4_recorded(effective, block)
    assert driver.arm_params_problems(metrics, effective) == []


@pytest.mark.parametrize(
    "want_patch,drifted,fragment",
    [(case[1], case[3], case[4]) for case in STAGE4_CASES],
    ids=[case[0] for case in STAGE4_CASES])
def test_stage4_identity_field_mismatch_is_a_problem(
        config, want_patch, drifted, fragment):
    effective = _stage4_effective(config, want_patch)
    metrics = _stage4_recorded(effective, drifted)
    problems = driver.arm_params_problems(metrics, effective)
    assert [p for p in problems if p.startswith(fragment)], problems


@pytest.mark.parametrize(
    "want_patch,block", [(case[1], case[2]) for case in STAGE4_CASES],
    ids=[case[0] for case in STAGE4_CASES])
def test_stage4_identity_field_missing_block_is_a_problem(
        config, want_patch, block):
    """A checkpoint from before the rung records no block at all."""
    effective = _stage4_effective(config, want_patch)
    metrics = _stage4_recorded(effective, block)
    for name in block:
        metrics["arm_params"].pop(name)
    assert driver.arm_params_problems(metrics, effective)


def test_stage2_config_is_untouched_by_the_stage4_checks(config):
    """A configuration that sets none of them compares exactly as before."""
    entry = configurations(config)["same_entity_k30"]
    effective = effective_settings(config, entry, base_logits_digest="d" * 32)
    for path, _, _ in driver.NESTED_ARM_PARAM_FIELDS:
        assert path[0] not in effective["arm_params"]
    metrics = _stage4_recorded(effective, {})
    assert driver.arm_params_problems(metrics, effective) == []


def test_override_drift_refuses_reuse(tmp_path, config):
    entry = configurations(config)["mlp"]
    smoke = effective_settings(config, entry, epochs=1, overrides={"epochs": 1})
    out_dir = _checkpoint(tmp_path, config, entry, 7, smoke)
    registered = effective_settings(config, entry)
    with pytest.raises(RetrainError) as error:
        verify_checkpoint(out_dir, json.loads(
            (out_dir / "metrics.json").read_text()), config, _resolved(),
            entry, 7, registered)
    message = str(error.value)
    assert "overrides" in message and "optimiser.epochs" in message


# ---------------------------------------------------------------------------
# summary schema
# ---------------------------------------------------------------------------

def _summary_row(signature: dict, **patch) -> dict:
    row = {
        "seed": 7, "ll": 1.2345678901234, "best_epoch": 4,
        "overrides": {}, "wall_seconds": 12.5,
        "peak_rss_bytes": 3 * 1024 ** 3, "peak_rss_bytes_valid": True,
        "rss_method": "psutil_process_tree_sampled",
        "peak_rss_unit": "bytes",
        "omp_num_threads": "4", "checkpoint_md5": "0" * 32,
        "checkpoint_dir": "models/embeddings/seq_stage2/runs/recency_k30/"
                          "seed_7",
        "training_signature": signature["training_signature"],
        "machine_provenance": {"machine": "laptop", "hostname": "laptop.local"},
        "_split_files": copy.deepcopy(SPLIT_FILES),
    }
    row.update(patch)
    return row


def test_summary_schema(tmp_path, config):
    entry = configurations(config)["recency_k30"]
    effective = effective_settings(config, entry)
    resolved = _resolved()
    signature = driver.signature_for(effective, resolved)
    rows = [_summary_row(signature)]
    summary = build_summary(config, CONFIG_PATH, resolved, entry, rows, {},
                            tmp_path, effective, (5, 2), signature)
    assert summary["experiment"]["config_id"] == "recency_k30"
    assert summary["experiment"]["runs_expected"] == 2
    assert summary["experiment"]["runs_recorded"] == 1
    assert summary["experiment"]["complete"] is False
    assert summary["experiment"]["test_split_scored"] is False
    assert summary["arm"]["arm_params_expected"]["k"] == 30
    assert summary["frame"]["version"] == "i7"
    assert set(summary["frame"]["split_md5s"]) == {"train", "validation"}
    per_seed = summary["splits"]["validation"]["per_seed"][0]
    assert set(per_seed) == set(driver.PER_SEED_FIELDS)
    assert summary["experiment"]["training_signature"] == signature[
        "training_signature"]
    assert summary["experiment"]["provenance_check"]["pinned_by"] == (
        "scripts/sequence_track/pin_stage2.py")
    # full precision, never the rounded four-decimal field
    assert per_seed["ll"] == 1.2345678901234
    assert yaml.safe_load(yaml.safe_dump(summary))["arm"]["queue_order"] == 6


def test_summary_refuses_a_row_from_another_frame(tmp_path, config):
    entry = configurations(config)["mlp"]
    effective = effective_settings(config, entry)
    resolved = _resolved()
    signature = driver.signature_for(effective, resolved)
    stale = copy.deepcopy(SPLIT_FILES)
    stale["train"]["md5"] = "9" * 32
    rows = [_summary_row(signature, checkpoint_dir="x", _split_files=stale)]
    with pytest.raises(RetrainError) as error:
        build_summary(config, CONFIG_PATH, resolved, entry, rows, {},
                      tmp_path, effective, (5, 2), signature)
    assert "split_files.train.md5" in str(error.value)


# ---------------------------------------------------------------------------
# output confinement
# ---------------------------------------------------------------------------

def test_output_root_is_confined_to_the_stage_2_namespace(tmp_path):
    with pytest.raises(RetrainError) as error:
        driver.assert_writable(tmp_path)
    assert "refusing to write outside" in str(error.value)
    assert driver.assert_writable(
        driver.SEQ_STAGE2_ROOT / "smoke") == (
            driver.SEQ_STAGE2_ROOT / "smoke").resolve()


# ---------------------------------------------------------------------------
# Astra MUST-FIX 2: the seed-independent training signature
# ---------------------------------------------------------------------------

def test_signature_is_seed_independent(config):
    """The whole point: two seeds of one configuration share one signature."""
    entry = configurations(config)["mlp"]
    resolved = _resolved()
    seven = driver.signature_for(effective_settings(config, entry), resolved)
    thirteen = driver.signature_for(effective_settings(config, entry),
                                    resolved)
    assert seven["training_signature"] == thirteen["training_signature"]
    assert set(seven["training_signature_components"]) == set(
        driver.SIGNATURE_COMPONENTS)


def test_signature_separates_two_configurations(config):
    entries = configurations(config)
    resolved = _resolved()
    k6 = driver.signature_for(
        effective_settings(config, entries["same_entity_k6"]), resolved)
    k12 = driver.signature_for(
        effective_settings(config, entries["same_entity_k12"]), resolved)
    assert k6["training_signature"] != k12["training_signature"]


def test_signature_covers_key_construction(config):
    """`key_construction` is identity, not a label (it lives in arm_params)."""
    entry = configurations(config)["same_entity_k30"]
    resolved = _resolved()
    effective = effective_settings(config, entry)
    assert effective["arm_params"]["key_construction"] == (
        t1.ARM_KEY_CONSTRUCTION[entry["arm"]])
    before = driver.signature_for(effective, resolved)
    twisted = copy.deepcopy(effective)
    twisted["arm_params"]["key_construction"] = "shifted_history"
    after = driver.signature_for(twisted, resolved)
    assert before["training_signature"] != after["training_signature"]
    assert before["training_signature_components"]["arm_params"] != (
        after["training_signature_components"]["arm_params"])


def test_recurrent_arms_hash_their_own_implementation_file():
    # Astra round-1 MUST-FIX 2: the imported feature contract and the manifest
    # resolver are in every arm's set; the recurrent module is in theirs.
    # C114 added `feature_registry.py`: it resolves the alternative contract's
    # ordered 114 columns, so it changes a training input with no trainer edit.
    common = [driver.TRAINER_SOURCE, driver.FEATURE_CONTRACT_SOURCE,
              driver.ARTIFACT_RESOLVER_SOURCE, driver.FEATURE_REGISTRY_SOURCE]
    assert driver.implementation_sources("lstm") == common + [
        driver.RECURRENT_SOURCE]
    assert driver.implementation_sources("xlstm") == common + [
        driver.RECURRENT_SOURCE]
    assert driver.implementation_sources("mlp") == common
    identity = driver.implementation_identity("lstm")
    assert set(identity) == set(common) | {driver.RECURRENT_SOURCE}
    assert all(len(value) == 64 for value in identity.values())


def test_reuse_refused_when_the_feature_contract_changed(tmp_path, config,
                                                         monkeypatch):
    """An edited `embeddings_e1.py` changes the training inputs."""
    entry = configurations(config)["mlp"]
    resolved = _resolved()
    effective = effective_settings(config, entry)
    out_dir = _checkpoint(tmp_path, config, entry, 7, effective,
                          resolved=resolved)
    stale = driver.signature_for(effective, resolved)
    monkeypatch.setattr(driver, "FEATURE_CONTRACT_SOURCE",
                        driver.RECURRENT_SOURCE)
    edited = driver.signature_for(effective, resolved)
    assert edited["training_signature"] != stale["training_signature"]
    with pytest.raises(RetrainError) as error:
        verify_checkpoint(out_dir, json.loads(
            (out_dir / "metrics.json").read_text()), config, resolved, entry, 7,
            effective, edited)
    assert "training_signature component 'implementation'" in str(error.value)


def test_reuse_refused_when_key_construction_changed(tmp_path, config):
    entry = configurations(config)["same_entity_k30"]
    resolved = _resolved()
    effective = effective_settings(config, entry)
    out_dir = _checkpoint(tmp_path, config, entry, 7, effective,
                          resolved=resolved)
    # the same configuration, retrained after the arm's key construction moved
    twisted = copy.deepcopy(effective)
    twisted["arm_params"]["key_construction"] = "shifted_history"
    with pytest.raises(RetrainError) as error:
        verify_checkpoint(out_dir, json.loads(
            (out_dir / "metrics.json").read_text()), config, resolved,
            entry, 7,
            twisted, driver.signature_for(twisted, resolved))
    message = str(error.value)
    assert "training_signature component 'arm_params'" in message
    assert "refusing to reuse" in message


def test_reuse_refused_when_an_implementation_file_changed(
        tmp_path, config, monkeypatch):
    """An edited model file must not be able to reuse the old seeds."""
    entry = configurations(config)["mlp"]
    resolved = _resolved()
    effective = effective_settings(config, entry)
    out_dir = _checkpoint(tmp_path, config, entry, 7, effective,
                          resolved=resolved)
    stale = driver.signature_for(effective, resolved)
    # a different real file stands in for an edited trainer, so the hashing
    # path under test is the real one
    monkeypatch.setattr(driver, "TRAINER_SOURCE", driver.RECURRENT_SOURCE)
    edited = driver.signature_for(effective, resolved)
    assert edited["training_signature"] != stale["training_signature"]
    with pytest.raises(RetrainError) as error:
        verify_checkpoint(out_dir, json.loads(
            (out_dir / "metrics.json").read_text()), config, resolved,
            entry, 7,
            effective, edited)
    assert "training_signature component 'implementation'" in str(error.value)


def test_reuse_refused_when_the_frame_md5_changed(tmp_path, config):
    entry = configurations(config)["mlp"]
    resolved = _resolved()
    effective = effective_settings(config, entry)
    out_dir = _checkpoint(tmp_path, config, entry, 7, effective,
                          resolved=resolved)
    moved = _resolved()
    moved["split_files"]["train"]["md5"] = "9" * 32
    with pytest.raises(RetrainError) as error:
        verify_checkpoint(out_dir, json.loads(
            (out_dir / "metrics.json").read_text()), config, moved, entry, 7,
            effective, driver.signature_for(effective, moved))
    message = str(error.value)
    assert "training_signature component 'frame'" in message


def test_reuse_refused_when_the_run_record_has_no_signature(tmp_path, config):
    entry = configurations(config)["mlp"]
    resolved = _resolved()
    effective = effective_settings(config, entry)
    out_dir = _checkpoint(tmp_path, config, entry, 7, effective,
                          resolved=resolved)
    record = json.loads((out_dir / "run_record.json").read_text())
    record.pop("training_signature")
    record.pop("training_signature_components")
    (out_dir / "run_record.json").write_text(json.dumps(record))
    with pytest.raises(RetrainError) as error:
        verify_checkpoint(out_dir, json.loads(
            (out_dir / "metrics.json").read_text()), config, resolved,
            entry, 7,
            effective)
    assert "carries no training_signature" in str(error.value)


def test_completion_record_signature_must_match(tmp_path, config):
    entry = configurations(config)["mlp"]
    resolved = _resolved()
    effective = effective_settings(config, entry)
    out_dir = _checkpoint(tmp_path, config, entry, 7, effective,
                          resolved=resolved, completion=True)
    payload = json.loads((out_dir / COMPLETION_RECORD).read_text())
    payload["training_signature"] = "0" * 64
    (out_dir / COMPLETION_RECORD).write_text(json.dumps(payload))
    with pytest.raises(RetrainError) as error:
        verify_checkpoint(out_dir, json.loads(
            (out_dir / "metrics.json").read_text()), config, resolved,
            entry, 7,
            effective)
    assert f"{COMPLETION_RECORD} training_signature" in str(error.value)


def test_signature_is_written_into_the_run_record_and_summary(
        tmp_path, config):
    entry = configurations(config)["mlp"]
    resolved = _resolved()
    effective = effective_settings(config, entry)
    signature = driver.signature_for(effective, resolved)
    out_dir = _checkpoint(tmp_path, config, entry, 7, effective,
                          resolved=resolved, completion=True)
    record = json.loads((out_dir / "run_record.json").read_text())
    completion = json.loads((out_dir / COMPLETION_RECORD).read_text())
    assert record["training_signature"] == signature["training_signature"]
    assert completion["training_signature"] == signature["training_signature"]
    summary = build_summary(config, CONFIG_PATH, resolved, entry,
                            [_summary_row(signature)], {}, tmp_path, effective,
                            (5, 2), signature)
    assert summary["experiment"]["training_signature"] == signature[
        "training_signature"]
    assert summary["experiment"]["training_signature_sources"] == (
        driver.implementation_sources("mlp"))


def test_summary_refuses_two_differently_signed_runs(tmp_path, config):
    entry = configurations(config)["mlp"]
    resolved = _resolved()
    effective = effective_settings(config, entry)
    signature = driver.signature_for(effective, resolved)
    rows = [_summary_row(signature),
            _summary_row(signature, seed=13, training_signature="0" * 64)]
    with pytest.raises(RetrainError) as error:
        build_summary(config, CONFIG_PATH, resolved, entry, rows, {}, tmp_path,
                      effective, (5, 2), signature)
    assert "training_signature" in str(error.value)


# ---------------------------------------------------------------------------
# Astra MUST-FIX 2: preflight COMPARES the registered provenance
# ---------------------------------------------------------------------------

LIVE_CACHE = {"role": "stats_cache_i7", "md5": "c" * 32,
              "path": "models/player_stats_cache_i7.sqlite"}


def _live_source_hashes(arm: str = "mlp") -> dict:
    """The live sha256 of an arm's training sources, as a pin would record."""
    return dict(driver.implementation_identity(arm))


def _pinned(**patch) -> dict:
    """A provenance block that matches SPLIT_FILES / FEATURE_HASH exactly."""
    block = {
        "pinned_by": "scripts/sequence_track/pin_stage2.py",
        "pins_generated_at": "2026-09-11T17:57:48Z",
        "config_body_sha256": "7" * 64,
        "frame": {
            "dir": "data/xgb_data_i7",
            "version": "i7",
            "feature_hash": dict(FEATURE_HASH),
            "splits": {
                split: {field: facts[field]
                        for field in driver.SPLIT_IDENTITY_FIELDS}
                for split, facts in SPLIT_FILES.items()},
        },
        "stats_cache": dict(LIVE_CACHE),
        # the cohort is pinned, and is deliberately NOT required to exist
        "cohort": {"frozen": "models/embeddings/seq_stage2/cohort/FROZEN.json",
                   "frozen_json_sha256": "a" * 64, "status": "frozen"},
        # the real hashes of the real training sources, so the round-1
        # MUST-FIX 2 comparison passes on an unedited checkout
        "sources": {"count": 1, "source_sha256": _live_source_hashes()},
        "runtime": {"python": "3.9.25"},
        "features": {"count": 50},
    }
    block.update(patch)
    return block


def _provenance_check(pinned, *, base_logits=None, cache=None,
                      training_sources=("mlp",)):
    sources = sorted({name for arm in training_sources
                      for name in driver.implementation_sources(arm)})
    return driver.provenance_check(
        {"provenance": pinned} if pinned is not None else {},
        CONFIG_PATH, "i7", dict(FEATURE_HASH), copy.deepcopy(SPLIT_FILES),
        dict(cache or LIVE_CACHE), base_logits, sources)


def test_provenance_matching_the_live_measurement_passes():
    check = _provenance_check(_pinned())
    assert "frame.splits" in check["compared"]
    assert "stats_cache.md5" in check["compared"]
    assert check["config_body_sha256"] == "7" * 64


def test_provenance_absent_refuses_and_names_the_pin():
    with pytest.raises(RetrainError) as error:
        _provenance_check(None)
    message = str(error.value)
    assert "carries no 'provenance' block" in message
    assert "pin_stage2.py --write" in message


@pytest.mark.parametrize("mutate,fragment", [
    (lambda p: p["frame"]["splits"]["train"].__setitem__("md5", "9" * 32),
     "train parquet md5"),
    (lambda p: p["frame"]["splits"]["validation"].__setitem__("n_rows", 999),
     "validation parquet n_rows"),
    (lambda p: p["frame"]["splits"]["train"].__setitem__(
        "match_date_max", "2030-01-01"), "train parquet match_date_max"),
    (lambda p: p["frame"]["feature_hash"].__setitem__("version", "v3"),
     "frame .feature_hash version"),
    (lambda p: p["frame"].__setitem__("version", "v3"),
     "provenance.frame.version"),
    (lambda p: p["stats_cache"].__setitem__("md5", "9" * 32),
     "stats cache md5"),
    (lambda p: p["stats_cache"].__setitem__("role", "stats_cache_v3"),
     "stats cache role"),
    (lambda p: p["frame"].pop("splits"), "provenance.frame.splits"),
])
def test_provenance_mismatch_refuses(mutate, fragment):
    pinned = _pinned()
    mutate(pinned)
    with pytest.raises(RetrainError) as error:
        _provenance_check(pinned)
    message = str(error.value)
    assert fragment in message
    assert "are not the files this config pins" in message


def test_provenance_verification_needs_no_cohort_artifact(tmp_path):
    """The mini has no cohort; verification must still work there."""
    pinned = _pinned()
    pinned["cohort"]["frozen"] = str(tmp_path / "definitely-absent.json")
    assert not Path(pinned["cohort"]["frozen"]).exists()
    check = _provenance_check(pinned)
    assert "cohort" in check["skipped"]
    assert "mini" in check["skipped"]["cohort"]
    assert not any(name.startswith("cohort") for name in check["compared"])


def test_provenance_skips_the_non_training_closure_with_a_note():
    check = _provenance_check(_pinned())
    assert set(check["skipped"]) >= {
        "cohort", "sources_outside_the_training_closure", "runtime",
        "features"}
    assert all(check["skipped"].values())


# ---------------------------------------------------------------------------
# Astra round-1 MUST-FIX 2: agreement with the pin's REGISTERED source hashes
# ---------------------------------------------------------------------------

def test_training_sources_are_the_import_closure_of_the_trainer():
    """The set is derived, not guessed: the AST closure over `scripts/`.

    `embeddings_e1.py` defines the 50-feature contract the trainer imports at
    module scope and `artifacts.py` resolves the stats cache by manifest role,
    so both change training inputs with no trainer edit.
    """
    import ast

    def imported(path: Path) -> set[str]:
        names = set()
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                names.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and (
                    node.level == 0):
                names.add(node.module)
        return names

    def as_source(module: str):
        candidate = driver.REPO / "scripts" / (
            module.replace(".", "/") + ".py")
        return (candidate.relative_to(driver.REPO).as_posix()
                if candidate.is_file() else None)

    closure, stack = set(), list(driver.IMPLEMENTATION_CLOSURE_ROOTS)
    while stack:
        current = stack.pop()
        if current in closure:
            continue
        closure.add(current)
        for module in imported(driver.REPO / current):
            resolved = as_source(module)
            if resolved:
                stack.append(resolved)
    assert closure == set(driver.implementation_sources("lstm"))
    assert set(driver.implementation_sources("mlp")) == closure - {
        driver.RECURRENT_SOURCE}
    assert driver.FEATURE_CONTRACT_SOURCE in closure
    assert driver.ARTIFACT_RESOLVER_SOURCE in closure


def test_training_sources_are_compared_against_the_pin():
    check = _provenance_check(_pinned())
    assert "sources.source_sha256 (training closure)" in check["compared"]
    assert set(check["training_sources_compared"]) == set(
        driver.implementation_sources("mlp"))
    assert check["training_sources_not_pinned"] == []


def test_a_training_source_disagreeing_with_the_pin_refuses():
    pinned = _pinned()
    pinned["sources"]["source_sha256"][
        driver.FEATURE_CONTRACT_SOURCE] = "0" * 64
    with pytest.raises(RetrainError) as error:
        _provenance_check(pinned)
    message = str(error.value)
    assert driver.FEATURE_CONTRACT_SOURCE in message
    assert "not the code the pin registered" in message


def test_a_source_outside_the_training_closure_cannot_refuse_a_night():
    """An edited analysis script is not a reason to refuse to train."""
    pinned = _pinned()
    pinned["sources"]["source_sha256"][
        "scripts/sequence_track/stage2_stats.py"] = "0" * 64
    check = _provenance_check(pinned)
    assert check["training_sources_not_pinned"] == []


def test_a_training_source_the_pin_never_recorded_is_named_not_passed():
    pinned = _pinned()
    pinned["sources"]["source_sha256"].pop(driver.ARTIFACT_RESOLVER_SOURCE)
    check = _provenance_check(pinned)
    assert check["training_sources_not_pinned"] == [
        driver.ARTIFACT_RESOLVER_SOURCE]
    assert "sources_not_pinned" in check["skipped"]
    assert driver.ARTIFACT_RESOLVER_SOURCE in check["skipped"][
        "sources_not_pinned"]


def test_an_unpinnable_sources_block_refuses():
    pinned = _pinned()
    pinned["sources"] = {"count": 22}
    with pytest.raises(RetrainError) as error:
        _provenance_check(pinned)
    assert "source_sha256" in str(error.value)


def test_provenance_compares_the_base_logits_digest_only_when_read():
    live = {"dir": "models/embeddings/seq_stage2/base_logits",
            "digest": "d" * 32,
            "splits": {"train": {"npz_md5": "1" * 32},
                       "validation": {"npz_md5": "2" * 32}}}
    pinned = _pinned(base_logits={
        "train_validation_digest": "d" * 32,
        "splits": {"train": {"npz_md5": "1" * 32},
                   "validation": {"npz_md5": "2" * 32}}})
    check = _provenance_check(pinned, base_logits=live)
    assert "base_logits.train_validation_digest" in check["compared"]

    stale = copy.deepcopy(pinned)
    stale["base_logits"]["train_validation_digest"] = "e" * 32
    with pytest.raises(RetrainError) as error:
        _provenance_check(stale, base_logits=live)
    assert "base-logits digest" in str(error.value)

    rebuilt = copy.deepcopy(live)
    rebuilt["splits"]["validation"]["npz_md5"] = "9" * 32
    with pytest.raises(RetrainError) as error:
        _provenance_check(pinned, base_logits=rebuilt)
    assert "base logits validation npz md5" in str(error.value)

    # no residual arm selected: nothing is read, so nothing is compared
    check = _provenance_check(pinned, base_logits=None)
    assert "base_logits" in check["skipped"]


NIGHT3_CONFIG_PATH = driver.REPO / "experiments/configs/seq_stage3_night3_v1.yaml"


@pytest.mark.needs_artifacts
def test_preflight_compares_the_committed_pin():
    """The ACTIVE config, the real frame: the comparison must pass as shipped.

    Night 3 (2026-09-13): the active config is the night-3 one. The sealed
    stage 2 configs keep the provenance of the sources AS TRAINED, so after
    a later trainer edit their preflight is expected to refuse — that is the
    behaviour the next test locks in — and their runs are anchored by the
    statistics tool to the pin's hashes of each run's own source list.
    """
    config = driver.load_config(NIGHT3_CONFIG_PATH)
    entries = configurations(config)
    resolved = driver.preflight(
        config, [entries["mlp_pool"], entries["mlp_E_tiercond"]],
        NIGHT3_CONFIG_PATH)
    check = resolved["provenance"]
    assert "frame.splits" in check["compared"]
    assert "cohort" in check["skipped"]


@pytest.mark.needs_artifacts
def test_a_sealed_stage_config_refuses_preflight_after_a_trainer_edit(config):
    """The stage 2 pin records the trainer as trained; the committed trainer
    has moved since (night 3), so preflight must refuse rather than train
    stage 2 configurations under a different implementation."""
    entries = configurations(config)
    with pytest.raises(driver.RetrainError, match="transformer_t1.py"):
        driver.preflight(config, [entries["mlp"]], CONFIG_PATH)


# ---------------------------------------------------------------------------
# Astra MUST-FIX 3: completion needs all four artefacts, atomically recorded
# ---------------------------------------------------------------------------

def test_all_four_artefacts_are_required(config, tmp_path):
    assert set(RUN_ARTIFACTS) == {"metrics.json", "model.pt",
                                  "run_record.json",
                                  "predictions_validation.npz"}
    entry = configurations(config)["mlp"]
    effective = effective_settings(config, entry)
    out_dir = _checkpoint(tmp_path, config, entry, 7, effective)
    assert driver.artifact_problems(out_dir) == []
    assert driver.is_complete(out_dir) is True


@pytest.mark.parametrize("missing", list(RUN_ARTIFACTS))
def test_a_run_missing_any_artefact_is_incomplete(config, tmp_path, missing):
    entry = configurations(config)["mlp"]
    effective = effective_settings(config, entry)
    out_dir = _checkpoint(
        tmp_path, config, entry, 7, effective,
        artifacts=[name for name in RUN_ARTIFACTS if name != missing])
    problems = driver.artifact_problems(out_dir)
    assert problems == [f"{missing} is missing"]
    assert driver.is_complete(out_dir) is False


def test_a_run_interrupted_after_model_pt_is_incomplete(config, tmp_path):
    """The exact interruption Astra named: model.pt, no run_record.json."""
    entry = configurations(config)["mlp"]
    effective = effective_settings(config, entry)
    out_dir = _checkpoint(tmp_path, config, entry, 7, effective,
                          artifacts=["metrics.json", "model.pt",
                                     "predictions_validation.npz"])
    assert driver.is_complete(out_dir) is False
    assert driver.artifact_problems(out_dir) == ["run_record.json is missing"]


def test_an_empty_or_truncated_artefact_is_incomplete(config, tmp_path):
    entry = configurations(config)["mlp"]
    effective = effective_settings(config, entry)
    out_dir = _checkpoint(tmp_path, config, entry, 7, effective)
    (out_dir / "model.pt").write_bytes(b"")
    assert driver.artifact_problems(out_dir) == ["model.pt is empty"]
    out_dir = _checkpoint(tmp_path / "b", config, entry, 13, effective)
    (out_dir / "metrics.json").write_text("{not json")
    assert "metrics.json is unreadable" in driver.artifact_problems(out_dir)[0]
    out_dir = _checkpoint(tmp_path / "c", config, entry, 13, effective)
    (out_dir / "predictions_validation.npz").write_bytes(b"PK\x03\x04garbage")
    assert "predictions_validation.npz is unreadable" in (
        driver.artifact_problems(out_dir)[0])


def test_predictions_without_probs_is_incomplete(config, tmp_path):
    entry = configurations(config)["mlp"]
    effective = effective_settings(config, entry)
    out_dir = _checkpoint(tmp_path, config, entry, 7, effective)
    np.savez_compressed(out_dir / "predictions_validation.npz",
                        y=np.zeros(5, dtype=np.int64))
    assert driver.artifact_problems(out_dir) == [
        "predictions_validation.npz carries no 'probs' array"]


def test_completion_record_holds_the_signature_and_the_artefacts(
        config, tmp_path):
    entry = configurations(config)["mlp"]
    resolved = _resolved()
    effective = effective_settings(config, entry)
    signature = driver.signature_for(effective, resolved)
    out_dir = _checkpoint(tmp_path, config, entry, 7, effective,
                          resolved=resolved)
    payload = driver.write_completion_record(out_dir, 7, "mlp", signature)
    assert payload["seed"] == 7 and payload["config_id"] == "mlp"
    assert set(payload["artifacts"]) == set(RUN_ARTIFACTS)
    for name, facts in payload["artifacts"].items():
        assert facts["bytes"] == (out_dir / name).stat().st_size
        assert len(facts["md5"]) == 32
    assert payload["training_signature"] == signature["training_signature"]
    assert json.loads((out_dir / COMPLETION_RECORD).read_text()) == payload


def test_completion_record_refuses_an_incomplete_run(config, tmp_path):
    entry = configurations(config)["mlp"]
    resolved = _resolved()
    effective = effective_settings(config, entry)
    signature = driver.signature_for(effective, resolved)
    out_dir = _checkpoint(tmp_path, config, entry, 7, effective,
                          resolved=resolved,
                          artifacts=["metrics.json", "model.pt"])
    with pytest.raises(RetrainError) as error:
        driver.write_completion_record(out_dir, 7, "mlp", signature)
    assert "did not leave a complete set of artefacts" in str(error.value)
    assert not (out_dir / COMPLETION_RECORD).exists()


def test_completion_record_is_atomic(config, tmp_path, monkeypatch):
    """A crash between validating and publishing leaves NO COMPLETE.json."""
    entry = configurations(config)["mlp"]
    resolved = _resolved()
    effective = effective_settings(config, entry)
    signature = driver.signature_for(effective, resolved)
    out_dir = _checkpoint(tmp_path, config, entry, 7, effective,
                          resolved=resolved)

    def boom(src, dst):
        raise OSError("power cut between validate and publish")

    monkeypatch.setattr(driver.os, "replace", boom)
    with pytest.raises(OSError):
        driver.write_completion_record(out_dir, 7, "mlp", signature)
    assert not (out_dir / COMPLETION_RECORD).exists()
    assert driver.read_completion_record(out_dir) is None
    # the temp file is the only trace, and it is not the record
    assert [p.name for p in out_dir.iterdir() if p.name.startswith(".")] == [
        f".{COMPLETION_RECORD}.tmp"]


def test_clear_run_artifacts_removes_a_partial_attempt(config, tmp_path):
    entry = configurations(config)["mlp"]
    effective = effective_settings(config, entry)
    out_dir = _checkpoint(tmp_path, config, entry, 7, effective,
                          artifacts=["metrics.json", "model.pt"])
    (out_dir / f".{COMPLETION_RECORD}.tmp").write_text("{}")
    removed = driver.clear_run_artifacts(out_dir)
    assert set(removed) == {"metrics.json", "model.pt",
                            f".{COMPLETION_RECORD}.tmp"}
    assert list(out_dir.iterdir()) == []


# ---------------------------------------------------------------------------
# Astra MUST-FIX 4: per-child peak RSS
# ---------------------------------------------------------------------------

def test_sampled_rss_is_recorded_with_its_method(config, tmp_path):
    record = {"peak_rss_bytes_sampled": 5 * 1024 ** 3,
              "rss_method": "psutil_process_tree_sampled"}
    read = driver.read_peak_rss(record)
    assert read["peak_rss_bytes"] == 5 * 1024 ** 3
    assert read["peak_rss_bytes_valid"] is True
    assert read["rss_method"] == "psutil_process_tree_sampled"
    assert read["peak_rss_unit"] == "bytes"


def test_a_legacy_rusage_delta_is_marked_invalid_not_reinterpreted(tmp_path):
    """The D6 smoke records: preserved, marked, never turned into a peak."""
    path = tmp_path / "run_record.json"
    legacy = {
        "peak_rss_bytes": 10059776,
        "peak_rss_unit": "bytes",
        "peak_rss_source": ("resource.getrusage(RUSAGE_CHILDREN).ru_maxrss "
                            "delta across this child; macOS reports bytes"),
        "children_max_rss_bytes": 4745101312,
    }
    path.write_text(json.dumps(legacy))
    before = path.read_bytes()
    read = driver.read_peak_rss(json.loads(path.read_text()))
    assert read["peak_rss_bytes"] is None
    assert read["peak_rss_bytes_valid"] is False
    assert read["rss_method"] == driver.INVALID_RSS_METHOD
    assert read["peak_rss_bytes_observed_invalid"] == 10059776
    assert "high-water mark" in read["peak_rss_invalid_reason"]
    # the historical record itself is untouched
    assert path.read_bytes() == before


def test_a_record_with_no_rss_at_all_is_not_invented():
    read = driver.read_peak_rss({})
    assert read == {"peak_rss_bytes": None, "peak_rss_bytes_valid": False,
                    "rss_method": None, "peak_rss_unit": "bytes"}
    assert driver.read_peak_rss(None)["peak_rss_bytes"] is None


def test_invalid_rss_travels_into_the_summary(tmp_path, config):
    entry = configurations(config)["mlp"]
    resolved = _resolved()
    effective = effective_settings(config, entry)
    signature = driver.signature_for(effective, resolved)
    row = _summary_row(signature)
    row.update(driver.read_peak_rss({"peak_rss_bytes": 10059776}))
    summary = build_summary(config, CONFIG_PATH, resolved, entry, [row], {},
                            tmp_path, effective, (5, 2), signature)
    per_seed = summary["splits"]["validation"]["per_seed"][0]
    assert per_seed["peak_rss_bytes"] is None
    assert per_seed["peak_rss_bytes_valid"] is False
    assert per_seed["rss_method"] == driver.INVALID_RSS_METHOD
    assert per_seed["peak_rss_bytes_observed_invalid"] == 10059776
    driver.print_table("mlp", summary)  # must not raise on a missing number


def test_sampler_measures_one_real_child():
    """A real child process, sampled while it lives. No training involved."""
    child = subprocess.Popen(
        [sys.executable, "-c",
         "x = bytearray(80 * 1024 * 1024)\n"
         "import time; time.sleep(1.0)\n"
         "len(x)"])
    sampler = driver.ChildRssSampler(child.pid, interval=0.05).start()
    child.wait()
    payload = sampler.stop()
    assert payload["peak_rss_bytes_valid"] is True
    assert payload["peak_rss_bytes_sampled"] > 20 * 1024 ** 2
    assert payload["rss_samples"] >= 1
    assert payload["rss_method"].endswith("_process_tree_sampled")
    assert payload["peak_rss_unit"] == "bytes"
    assert "getrusage" in payload["rss_method_note"]


def test_sampler_records_no_peak_when_nothing_was_observed():
    sampler = driver.ChildRssSampler(2 ** 30, interval=10.0)
    payload = sampler.start().stop()
    assert payload["peak_rss_bytes_sampled"] is None
    assert payload["peak_rss_bytes_valid"] is False
    assert "no sample landed" in payload["peak_rss_invalid_reason"]


def test_tree_rss_of_this_process_is_positive():
    import os as _os
    assert driver.tree_rss_bytes(_os.getpid()) > 0
    assert driver._tree_rss_bytes_ps(_os.getpid()) > 0


# ---------------------------------------------------------------------------
# the consolidated two-machine rewrite: seed 7 (laptop) + seed 13 (mini)
# ---------------------------------------------------------------------------

def _two_machine_tree(tmp_path, config, monkeypatch, config_id="mlp"):
    """One output tree assembled from two separately-trained seeds.

    Each seed was produced by its own one-seed invocation, so each already
    has its own `summary.yaml` naming only that seed — exactly what the rsync
    of the laptop and the mini trees produces.
    """
    monkeypatch.setattr(driver, "SEQ_STAGE2_ROOT", tmp_path / "seq_stage2")
    out_root = tmp_path / "seq_stage2" / "runs"
    entry = configurations(config)[config_id]
    resolved = _resolved()
    # `--out-root` is a recorded deviation, so the runs each machine wrote
    # carry it too; the consolidation must agree with them.
    overrides = {"out_root": driver.rel(out_root)}
    effective = effective_settings(config, entry, None, overrides,
                                   resolved["base_logits"]["digest"])
    signature = driver.signature_for(effective, resolved)
    lls = {7: 1.4250123456789, 13: 1.4319876543210}
    machines = {7: {"machine": "laptop", "hostname": "laptop.local",
                    "chip": "Apple M5 Pro",
                    "thread_caps": {"OMP_NUM_THREADS": "2"}},
                13: {"machine": "mini", "hostname": "mini.local",
                     "chip": "Apple M2", "thread_caps":
                         {"OMP_NUM_THREADS": "4"}}}
    for seed in (7, 13):
        # `ll` is passed in rather than patched afterwards, because the
        # completion record's manifest now pins metrics.json's md5 (Astra
        # round-1 MUST-FIX 3).
        _checkpoint(out_root, config, entry, seed, effective,
                    resolved=resolved, signature=signature, completion=True,
                    ll=lls[seed], machine=machines[seed])
        # the one-seed summary that machine wrote
        (out_root / config_id / "summary.yaml").write_text(yaml.safe_dump(
            {"experiment": {"runs_recorded": 1, "complete": False},
             "splits": {"validation": {"per_seed": [{"seed": seed}]}}}))
    monkeypatch.setattr(driver, "preflight",
                        lambda *a, **k: copy.deepcopy(resolved))
    monkeypatch.setattr(driver, "validation_match_count",
                        lambda *a, **k: (5, 2))
    return out_root, signature, lls


def test_consolidated_two_seed_invocation_reuses_both_and_rewrites_summary(
        tmp_path, config, monkeypatch, capsys):
    out_root, signature, lls = _two_machine_tree(tmp_path, config, monkeypatch)
    assert driver.main(["--config", str(CONFIG_PATH), "--out-root",
                        str(out_root), "--config-ids", "mlp",
                        "--consolidate", "--seeds", "7,13"]) == 0
    printed = capsys.readouterr().out
    # both seeds verified and reused; nothing trained, nothing deleted
    for seed in (7, 13):
        assert f"mlp seed {seed}: verified" in printed
    summary = yaml.safe_load((out_root / "mlp" / "summary.yaml").read_text())
    per_seed = summary["splits"]["validation"]["per_seed"]
    assert [row["seed"] for row in per_seed] == [7, 13]
    # full precision, not the trainer's rounded field
    assert [row["ll"] for row in per_seed] == [lls[7], lls[13]]
    assert summary["experiment"]["runs_recorded"] == 2
    assert summary["experiment"]["runs_expected"] == 2
    assert summary["experiment"]["complete"] is True
    assert summary["experiment"]["training_signature"] == signature[
        "training_signature"]
    assert all(row["training_signature"] == signature["training_signature"]
               for row in per_seed)
    # Astra round-2 MUST-FIX 4: each row says which machine trained that seed
    assert [row["machine_provenance"]["machine"] for row in per_seed] == [
        "laptop", "mini"]
    assert per_seed[1]["machine_provenance"]["thread_caps"] == {
        "OMP_NUM_THREADS": "4"}


def test_consolidated_invocation_refuses_a_seed_from_other_code(
        tmp_path, config, monkeypatch):
    """The mini ran an edited trainer: the consolidation must not merge it."""
    out_root, signature, _ = _two_machine_tree(tmp_path, config, monkeypatch)
    record_path = out_root / "mlp" / "seed_13" / "run_record.json"
    record = json.loads(record_path.read_text())
    record["training_signature"] = "0" * 64
    record["training_signature_components"] = dict(
        signature["training_signature_components"],
        implementation="9" * 64)
    record_path.write_text(json.dumps(record))
    with pytest.raises(RetrainError) as error:
        driver.main(["--config", str(CONFIG_PATH), "--out-root",
                     str(out_root), "--config-ids", "mlp", "--consolidate",
                     "--seeds", "7,13"])
    assert "training_signature component 'implementation'" in str(error.value)


# ---------------------------------------------------------------------------
# Astra round-2 MUST-FIX 3: --consolidate never trains and never deletes
# ---------------------------------------------------------------------------

def _is_training(command) -> bool:
    """True for a trainer launch, false for the `sysctl` chip read (D8.8)."""
    return any("transformer_t1.py" in str(part) for part in command)


def _no_launch(monkeypatch) -> list:
    """Fail loudly if consolidation ever tries to start a training child."""
    launched = []
    real_popen = driver.subprocess.Popen

    def fake_popen(command, **kwargs):
        if _is_training(command):
            launched.append(command)
            raise AssertionError("--consolidate launched training")
        return real_popen(command, **kwargs)

    monkeypatch.setattr(driver.subprocess, "Popen", fake_popen)
    return launched


def test_consolidate_refuses_a_half_transferred_seed_without_training_it(
        tmp_path, config, monkeypatch):
    """Seed 13 arrived half-rsynced.

    The old behaviour cleared it and retrained it ON THE LAPTOP, which
    silently breaks the immutable seed-to-machine assignment (D8.3). It must
    refuse, name the run, delete nothing and launch nothing; recovery is
    queued on the mini.
    """
    out_root, _, _ = _two_machine_tree(tmp_path, config, monkeypatch)
    launched = _no_launch(monkeypatch)
    seed13 = out_root / "mlp" / "seed_13"
    (seed13 / "run_record.json").unlink()
    (seed13 / COMPLETION_RECORD).unlink()
    survivors = sorted(p.name for p in seed13.iterdir())
    with pytest.raises(RetrainError) as error:
        driver.main(["--config", str(CONFIG_PATH), "--out-root",
                     str(out_root), "--config-ids", "mlp", "--consolidate",
                     "--seeds", "7,13"])
    message = str(error.value)
    assert "no artefact was deleted and no training was launched" in message
    assert "mlp seed 13" in message and "run_record.json is missing" in message
    assert "machine its seed is assigned to" in message
    assert launched == []
    # nothing deleted, and the sibling seed and the summary are untouched
    assert sorted(p.name for p in seed13.iterdir()) == survivors
    assert driver.is_complete(out_root / "mlp" / "seed_7", "mlp") is True
    assert yaml.safe_load((out_root / "mlp" / "summary.yaml").read_text())[
        "experiment"]["runs_recorded"] == 1


def test_consolidate_refuses_force_and_dry_run():
    with pytest.raises(RetrainError) as error:
        driver.main(["--config", str(CONFIG_PATH), "--consolidate", "--force"])
    assert "contradictory" in str(error.value)
    with pytest.raises(RetrainError) as error:
        driver.main(["--config", str(CONFIG_PATH), "--consolidate",
                     "--dry-run"])
    assert "launches nothing" in str(error.value)


def test_ordinary_invocation_still_retries_an_incomplete_seed(
        tmp_path, config, monkeypatch, capsys):
    """The training path keeps its retry semantics; only --consolidate is new.

    This is the behaviour the old `--seeds 7,13` test asserted, kept for the
    case it is actually correct for: a one-machine invocation retrying its own
    interrupted run.
    """
    out_root, _, _ = _two_machine_tree(tmp_path, config, monkeypatch)
    (out_root / "mlp" / "seed_13" / "run_record.json").unlink()
    (out_root / "mlp" / "seed_13" / COMPLETION_RECORD).unlink()
    launched = []

    class _FakeChild:
        pid = 1
        returncode = 0

        def wait(self):
            return 0

    real_popen = driver.subprocess.Popen

    def fake_popen(command, **kwargs):
        if not _is_training(command):  # the sysctl chip read (D8.8)
            return real_popen(command, **kwargs)
        launched.append(command)
        return _FakeChild()

    monkeypatch.setattr(driver.subprocess, "Popen", fake_popen)
    # the fake child writes nothing, so completion validation must refuse
    with pytest.raises(RetrainError) as error:
        driver.main(["--config", str(CONFIG_PATH), "--out-root",
                     str(out_root), "--config-ids", "mlp", "--seeds", "7,13"])
    printed = capsys.readouterr().out
    assert "mlp seed 7: complete; not overwriting" in printed
    assert "mlp seed 13: incomplete" in printed
    assert "retraining" in printed
    assert len(launched) == 1
    assert "--seed" in launched[0] and "13" in launched[0]
    assert "did not leave a complete set of artefacts" in str(error.value)


# ---------------------------------------------------------------------------
# Astra round-2 MUST-FIX 1: one signature across two checkout roots
# ---------------------------------------------------------------------------

def _fake_checkout(root: Path, frame_source: Path | None = None) -> Path:
    """A checkout root whose frame is either a real directory or a symlink.

    The implementation files are symlinked in from the real repository, so the
    signature's `implementation` component hashes identical CONTENT on both
    roots — which is what a git worktree pinned to one commit gives.
    """
    (root / "scripts" / "sequence_track").mkdir(parents=True)
    for name in driver.implementation_sources("lstm"):
        (root / name).symlink_to(driver.REPO / name)
    frame = root / "data" / "xgb_data_i7"
    frame.parent.mkdir(parents=True, exist_ok=True)
    if frame_source is None:
        frame.mkdir()
    else:
        frame.symlink_to(frame_source)
    return root


def _resolved_in(root: Path) -> dict:
    resolved = _resolved()
    resolved["frame_dir"] = root / "data/xgb_data_i7"
    return resolved


def test_one_signature_across_a_real_frame_and_a_symlinked_frame(
        tmp_path, config, monkeypatch):
    """The laptop worktree and the mini must compute the SAME signature.

    In the worktree `data/xgb_data_i7` is a symlink into the main checkout, so
    a signature over resolved physical paths gives the two machines different
    hashes and neither checkout can consolidate both seeds.
    """
    entry = configurations(config)["mlp"]
    effective = effective_settings(config, entry)
    main_root = _fake_checkout(tmp_path / "Match_Prediction")
    worktree = _fake_checkout(tmp_path / "MP_train_s7",
                              frame_source=main_root / "data" / "xgb_data_i7")
    assert (worktree / "data/xgb_data_i7").is_symlink()
    assert not (main_root / "data/xgb_data_i7").is_symlink()

    signatures = {}
    for root in (main_root, worktree):
        monkeypatch.setattr(driver, "REPO", root)
        signatures[root] = driver.signature_for(effective, _resolved_in(root))
    monkeypatch.undo()
    assert signatures[main_root]["training_signature"] == (
        signatures[worktree]["training_signature"])
    assert signatures[main_root]["training_signature_components"] == (
        signatures[worktree]["training_signature_components"])
    # and the logical components are the configured spellings, not paths
    assert driver.logical_frame_dir(_resolved_in(worktree)) == (
        "data/xgb_data_i7")
    assert driver.logical_stats_cache(_resolved_in(worktree)) == {
        "role": "stats_cache_i7", "md5": "c" * 32}


def test_a_signature_cannot_be_computed_without_the_configured_frame_dir():
    resolved = _resolved()
    resolved.pop("frame_dir_configured")
    with pytest.raises(RetrainError) as error:
        driver.logical_frame_dir(resolved)
    assert "machine-specific resolved path" in str(error.value)


def test_two_checkouts_consolidate_one_tree_written_under_the_other(
        tmp_path, config, monkeypatch):
    """Seed 7 written from the worktree, seed 13 from the main checkout.

    Both are verified and reused by one `--consolidate` invocation run from
    the main checkout, which is the D8.7 flow.
    """
    main_root = _fake_checkout(tmp_path / "Match_Prediction")
    worktree = _fake_checkout(tmp_path / "MP_train_s7",
                              frame_source=main_root / "data" / "xgb_data_i7")
    entry = configurations(config)["mlp"]
    out_root = tmp_path / "seq_stage2" / "runs"
    out_root.mkdir(parents=True)
    lls = {7: 1.4250123456789, 13: 1.4319876543210}
    written_by = {7: worktree, 13: main_root}
    for seed, root in written_by.items():
        monkeypatch.setattr(driver, "REPO", root)
        resolved = _resolved_in(root)
        effective = effective_settings(config, entry, None,
                                       {"out_root": driver.rel(out_root)},
                                       resolved["base_logits"]["digest"])
        _checkpoint(out_root, config, entry, seed, effective,
                    resolved=resolved,
                    signature=driver.signature_for(effective, resolved),
                    completion=True, ll=lls[seed],
                    machine={"machine": "laptop" if seed == 7 else "mini"})
    monkeypatch.undo()

    # consolidating from the MAIN checkout: the frame it measures is the real
    # directory, the frame seed 7 recorded was reached through a symlink
    monkeypatch.setattr(driver, "REPO", main_root)
    monkeypatch.setattr(driver, "SEQ_STAGE2_ROOT", tmp_path / "seq_stage2")
    resolved = _resolved_in(main_root)
    monkeypatch.setattr(driver, "preflight",
                        lambda *a, **k: copy.deepcopy(resolved))
    monkeypatch.setattr(driver, "validation_match_count",
                        lambda *a, **k: (5, 2))
    assert driver.main(["--config", str(CONFIG_PATH), "--out-root",
                        str(out_root), "--config-ids", "mlp", "--consolidate",
                        "--seeds", "7,13"]) == 0
    summary = yaml.safe_load((out_root / "mlp" / "summary.yaml").read_text())
    per_seed = summary["splits"]["validation"]["per_seed"]
    assert [row["seed"] for row in per_seed] == [7, 13]
    assert [row["ll"] for row in per_seed] == [lls[7], lls[13]]
    assert summary["experiment"]["complete"] is True
    assert [row["machine_provenance"]["machine"] for row in per_seed] == [
        "laptop", "mini"]


# ---------------------------------------------------------------------------
# Astra round-1 MUST-FIX 3: the checkpoint is read, the manifest enforced
# ---------------------------------------------------------------------------

def test_a_checkpoint_that_is_not_a_state_dict_is_incomplete(config, tmp_path):
    entry = configurations(config)["mlp"]
    effective = effective_settings(config, entry)
    out_dir = _checkpoint(tmp_path, config, entry, 7, effective)
    (out_dir / "model.pt").write_bytes(b"not a real checkpoint")
    problems = driver.artifact_problems(out_dir, "mlp")
    assert problems and "model.pt does not load as a checkpoint" in problems[0]
    assert driver.is_complete(out_dir, "mlp") is False


def test_a_truncated_checkpoint_is_refused_on_reuse(config, tmp_path):
    """The exact rsync failure: a complete-looking run with a cut model.pt."""
    entry = configurations(config)["mlp"]
    resolved = _resolved()
    effective = effective_settings(config, entry)
    out_dir = _checkpoint(tmp_path, config, entry, 7, effective,
                          resolved=resolved, completion=True)
    whole = (out_dir / "model.pt").read_bytes()
    (out_dir / "model.pt").write_bytes(whole[:len(whole) // 2])
    assert driver.is_complete(out_dir, "mlp") is False
    problems = driver.artifact_problems(out_dir, "mlp")
    assert any("model.pt" in problem for problem in problems)


def test_a_checkpoint_from_another_arm_is_refused(config, tmp_path):
    """A recurrent checkpoint in a transformer arm's directory, and back."""
    entry = configurations(config)["mlp"]
    effective = effective_settings(config, entry)
    out_dir = _checkpoint(tmp_path, config, entry, 7, effective)
    _write_state_dict(out_dir / "model.pt", "lstm")
    assert "recurrent." in driver.artifact_problems(out_dir, "mlp")[0]
    lstm = configurations(config)["lstm"]
    out_dir = _checkpoint(tmp_path / "b", config, lstm, 7,
                          effective_settings(config, lstm))
    _write_state_dict(out_dir / "model.pt", "mlp")
    assert "recurrent." in driver.artifact_problems(out_dir, "lstm")[0]
    # without an arm the structural state-dict check still runs
    assert driver.artifact_problems(out_dir) == []


def test_reuse_enforces_the_recorded_artifact_manifest(config, tmp_path):
    """A silently replaced artefact must not be reusable after transfer."""
    entry = configurations(config)["mlp"]
    resolved = _resolved()
    effective = effective_settings(config, entry)
    out_dir = _checkpoint(tmp_path, config, entry, 7, effective,
                          resolved=resolved, completion=True)
    predictions = out_dir / f"predictions_{driver.SELECTION_SPLIT}.npz"
    np.savez_compressed(predictions, probs=np.ones((5, 6), dtype=np.float32))
    with pytest.raises(RetrainError) as error:
        verify_checkpoint(out_dir, json.loads(
            (out_dir / "metrics.json").read_text()), config, resolved, entry, 7,
            effective)
    message = str(error.value)
    assert predictions.name in message
    assert "manifest" in message or "bytes" in message


def test_the_three_completion_record_states_are_distinguished(config,
                                                              tmp_path):
    entry = configurations(config)["mlp"]
    resolved = _resolved()
    effective = effective_settings(config, entry)

    # absent: a legacy run, allowed
    out_dir = _checkpoint(tmp_path / "legacy", config, entry, 7, effective,
                          resolved=resolved)
    status = driver.completion_status(out_dir)
    assert status["state"] == driver.COMPLETION_ABSENT
    assert "predates" in status["reason"]
    assert verify_checkpoint(out_dir, json.loads(
        (out_dir / "metrics.json").read_text()), config, resolved, entry, 7,
        effective)["best_epoch"] == 4

    # malformed: present but unreadable — unverifiable, NOT legacy
    out_dir = _checkpoint(tmp_path / "malformed", config, entry, 7, effective,
                          resolved=resolved, completion=True)
    (out_dir / COMPLETION_RECORD).write_text("{ truncated")
    assert driver.completion_status(out_dir)["state"] == (
        driver.COMPLETION_MALFORMED)
    with pytest.raises(RetrainError) as error:
        verify_checkpoint(out_dir, json.loads(
            (out_dir / "metrics.json").read_text()), config, resolved, entry, 7,
            effective)
    assert "unverifiable" in str(error.value)

    # malformed the other way: parses, but claims nothing about the artefacts
    out_dir = _checkpoint(tmp_path / "fieldless", config, entry, 7, effective,
                          resolved=resolved, completion=True)
    payload = json.loads((out_dir / COMPLETION_RECORD).read_text())
    payload.pop("artifacts")
    (out_dir / COMPLETION_RECORD).write_text(json.dumps(payload))
    status = driver.completion_status(out_dir)
    assert status["state"] == driver.COMPLETION_MALFORMED
    assert "artifacts" in status["reason"]

    # present and disagreeing: refused naming the field
    out_dir = _checkpoint(tmp_path / "disagreeing", config, entry, 7, effective,
                          resolved=resolved, completion=True)
    payload = json.loads((out_dir / COMPLETION_RECORD).read_text())
    payload["seed"] = 13
    (out_dir / COMPLETION_RECORD).write_text(json.dumps(payload))
    assert driver.completion_status(out_dir)["state"] == (
        driver.COMPLETION_PRESENT)
    with pytest.raises(RetrainError) as error:
        verify_checkpoint(out_dir, json.loads(
            (out_dir / "metrics.json").read_text()), config, resolved, entry, 7,
            effective)
    assert f"{COMPLETION_RECORD} seed" in str(error.value)


# ---------------------------------------------------------------------------
# Astra round-2 MUST-FIX 4 / D8.8: machine provenance, captured at run time
# ---------------------------------------------------------------------------

def test_machine_provenance_records_the_host_the_caps_and_the_symlinks(
        tmp_path, monkeypatch):
    for name in driver.THREAD_CAP_VARS:
        monkeypatch.setenv(name, "2")
    monkeypatch.delenv("VECLIB_MAXIMUM_THREADS")
    real = tmp_path / "frame"
    real.mkdir()
    link = tmp_path / "linked_frame"
    link.symlink_to(real)
    record = driver.machine_provenance(
        "laptop", device_requested="mps", device_used="mps",
        paths={"frame_dir": link, "stats_cache": real / "cache.sqlite"})
    assert record["machine"] == "laptop"
    assert record["machine_label_source"] == "--machine"
    assert record["hostname"]
    assert record["chip"]
    assert record["os"] and record["os_release"]
    assert record["python_version"] and record["torch_version"]
    assert isinstance(record["torch_mps_available"], bool)
    assert isinstance(record["torch_mps_built"], bool)
    assert record["device_requested"] == "mps"
    assert record["device_used"] == "mps"
    assert set(record["thread_caps"]) == set(driver.THREAD_CAP_VARS)
    assert record["thread_caps"]["OMP_NUM_THREADS"] == "2"
    assert record["thread_caps"]["VECLIB_MAXIMUM_THREADS"] is None
    assert record["repo_root"] == str(driver.REPO)
    assert isinstance(record["repo_is_worktree"], bool)
    assert record["paths"]["frame_dir"]["is_symlink"] is True
    assert record["paths"]["frame_dir"]["resolved"] == real.resolve().as_posix()
    assert record["paths"]["stats_cache"]["is_symlink"] is False


def test_machine_label_defaults_to_the_hostname():
    record = driver.machine_provenance()
    assert record["machine"] == record["hostname"]
    assert record["machine_label_source"] == "platform.node()"


def test_machine_provenance_is_not_part_of_the_signature(config):
    """Machine is exactly what the two halves of a split night may differ in."""
    entry = configurations(config)["mlp"]
    resolved = _resolved()
    signature = driver.signature_for(effective_settings(config, entry),
                                     resolved)
    blob = json.dumps(signature)
    for token in ("hostname", "machine", "thread_caps", "chip"):
        assert token not in blob


def test_a_run_records_its_machine_in_the_completion_record(config, tmp_path):
    entry = configurations(config)["mlp"]
    resolved = _resolved()
    effective = effective_settings(config, entry)
    signature = driver.signature_for(effective, resolved)
    machine = driver.machine_provenance("mini", device_requested="mps")
    out_dir = _checkpoint(tmp_path, config, entry, 7, effective,
                          resolved=resolved, machine=machine)
    payload = driver.write_completion_record(out_dir, 7, "mlp", signature,
                                             machine, "mlp")
    assert payload["machine_provenance"]["machine"] == "mini"
    assert json.loads((out_dir / "run_record.json").read_text())[
        "machine_provenance"]["machine"] == "mini"


def test_the_summary_carries_the_consolidating_machine_and_the_paths(
        tmp_path, config):
    entry = configurations(config)["mlp"]
    resolved = _resolved()
    resolved["physical_paths"] = driver.path_link_facts(
        {"frame_dir": resolved["frame_dir"]})
    effective = effective_settings(config, entry)
    signature = driver.signature_for(effective, resolved)
    machine = driver.machine_provenance("laptop")
    summary = build_summary(config, CONFIG_PATH, resolved, entry,
                            [_summary_row(signature)], {}, tmp_path, effective,
                            (5, 2), signature, machine)
    assert summary["experiment"]["consolidating_machine"]["machine"] == "laptop"
    assert "frame_dir" in summary["experiment"]["physical_paths"]
    per_seed = summary["splits"]["validation"]["per_seed"][0]
    assert per_seed["machine_provenance"]["machine"] == "laptop"
    driver.print_table("mlp", summary)


# ---------------------------------------------------------------------------
# stage 3: replay compatibility of the training signature
# ---------------------------------------------------------------------------
#
# Stage 3 adds five default-off params (`tier_embed`, `train_tier`,
# `train_match_list`, `max_steps`, `eval_every`) and an optional `role` key.
# None of them may move an EXISTING stage 2 configuration's signature, or
# every stage 2 checkpoint on both machines becomes unverifiable and the
# five-seed tables stop consolidating.
#
# The `implementation` component hashes `scripts/transformer_t1.py` itself, so
# the whole-signature hash necessarily moves when the trainer is edited — that
# is the component doing its job. What CAN be pinned, and is pinned here, is
# every component the new params could touch: `arm_params` and
# `training_block`. The digests below were captured from this config file
# BEFORE the stage 3 edits.
STAGE2_SIGNATURE_PIN = {
    "mlp": ("8246d96dbb617e41b49cda423d11ac5b213a07a84d97b2f8bee503753c373517",
            "3c17a16c7ba69b401a218b1f08ae2b6a9c17b4d82c62ab1939cfdacca8327002"),
    "full": (
        "9c58a995d3f528e3f8b4f233ad0122d57e4d5b2ab7d54cddc14e111b88be55a1",
        "3c17a16c7ba69b401a218b1f08ae2b6a9c17b4d82c62ab1939cfdacca8327002"),
    "fixed_decay": (
        "10ceed67d81171f8a99e665f7b4d2d8af80068aeca8f7ae172fbc87d75fd9a5e",
        "3c17a16c7ba69b401a218b1f08ae2b6a9c17b4d82c62ab1939cfdacca8327002"),
    "fox": ("9829c841a93a3cbbdff426d93ded00a035dfb55432c728726cec8b139b22e930",
            "3c17a16c7ba69b401a218b1f08ae2b6a9c17b4d82c62ab1939cfdacca8327002"),
    "aligned_hist": (
        "59892037b49fba55b22f98522ca9df0acad8c4804aaf0978340b624c84250dc8",
        "3c17a16c7ba69b401a218b1f08ae2b6a9c17b4d82c62ab1939cfdacca8327002"),
    "recency_k30": (
        "6914a8223b1ad1f6d4239467a6d3d01af1ea41ff3f2face8d9148f8b26a2ccca",
        "3c17a16c7ba69b401a218b1f08ae2b6a9c17b4d82c62ab1939cfdacca8327002"),
    "same_entity_k30": (
        "7005eb60bf2c1cce6deab75a6852d01e4c731015e3cf16957015669ff8865129",
        "3c17a16c7ba69b401a218b1f08ae2b6a9c17b4d82c62ab1939cfdacca8327002"),
    "aligned_hist_rf": (
        "8cae81ac9959e1832e3cc310dc874a24f51e98b35cae1ad72025bd2d6cfe83bb",
        "3c17a16c7ba69b401a218b1f08ae2b6a9c17b4d82c62ab1939cfdacca8327002"),
    "same_entity_k0": (
        "a122ed92bee322ef51137293027037dcb2a0871692d00241bfaa9c7ae77d6bdd",
        "3c17a16c7ba69b401a218b1f08ae2b6a9c17b4d82c62ab1939cfdacca8327002"),
    "same_entity_k6": (
        "90e334b43d9930a8ae817c40f5a77633e1e6893f592dab95c6fc898144126ade",
        "3c17a16c7ba69b401a218b1f08ae2b6a9c17b4d82c62ab1939cfdacca8327002"),
    "same_entity_k12": (
        "3bc33eeea5955f4c52c1d9435466a44090f96ea02ba87a3b654921fc726ee7d3",
        "3c17a16c7ba69b401a218b1f08ae2b6a9c17b4d82c62ab1939cfdacca8327002"),
    "same_entity_unr": (
        "31d2c375e32af691c29e427b7d7aab5828f009fc952c908842ac68cb84854391",
        "3c17a16c7ba69b401a218b1f08ae2b6a9c17b4d82c62ab1939cfdacca8327002"),
    "lstm": (
        "b6791758ebf6921bf09ed38e941f911a87055279679da2e9b181082b0597b65b",
        "3c17a16c7ba69b401a218b1f08ae2b6a9c17b4d82c62ab1939cfdacca8327002"),
    "xlstm": (
        "1f77aa9d9ed6b88681b0c55460735973c32e6186a384098ecf08a609db1ff616",
        "3c17a16c7ba69b401a218b1f08ae2b6a9c17b4d82c62ab1939cfdacca8327002"),
    "residual_mlp": (
        "f5a8bc3c06e72c76245e4e3141063144a80199d8855baa3ec85cb05acaf374f0",
        "3c17a16c7ba69b401a218b1f08ae2b6a9c17b4d82c62ab1939cfdacca8327002"),
    "residual_t1": (
        "8d9a5e94c4ceb3877daff4043290d58e6091d22b52309b3986f3cf4ff48ec14a",
        "3c17a16c7ba69b401a218b1f08ae2b6a9c17b4d82c62ab1939cfdacca8327002"),
}
# The base-logits digest the pin was captured with. It is an input to the
# residual arms' `arm_params`, so the pin fixes it rather than reading a
# directory that may not exist on this machine.
PIN_BASE_LOGITS_DIGEST = "DUMMY"


def _pinned_components(config: dict, config_id: str) -> tuple[str, str]:
    entry = configurations(config)[config_id]
    digest = (PIN_BASE_LOGITS_DIGEST
              if entry["arm"] in driver.t1.ARMS_NEEDING_BASE_LOGITS else None)
    effective = effective_settings(config, entry, base_logits_digest=digest)
    training_block = {"arch": effective["arch"],
                      "optimiser": effective["optimiser"]}
    if effective.get("schedule"):
        training_block["schedule"] = effective["schedule"]
    return (driver._digest(effective["arm_params"]),
            driver._digest(training_block))


def test_the_pin_covers_every_registered_configuration(config):
    assert sorted(STAGE2_SIGNATURE_PIN) == sorted(configurations(config))


@pytest.mark.parametrize("config_id", REGISTERED_ORDER)
def test_stage2_signature_components_are_unchanged(config, config_id):
    """The replay contract: stage 3's default-off params must serialise to
    exactly the canonical JSON stage 2 serialised to."""
    assert _pinned_components(config, config_id) == (
        STAGE2_SIGNATURE_PIN[config_id])


def test_no_stage2_configuration_sets_a_stage3_param(config):
    """The pin above only means something while this holds."""
    for config_id, entry in configurations(config).items():
        for name in driver.STAGE3_PARAMS:
            assert entry["_params"].get(name) is None, (config_id, name)
        assert "schedule" not in effective_settings(
            config, entry,
            base_logits_digest=PIN_BASE_LOGITS_DIGEST)


def test_report_match_lists_is_reporting_only(config, tmp_path):
    """Reviewer MUST-FIX 2: the exposure flag is rendered for EVERY
    configuration and enters NO identity block, so a pooled checkpoint stays
    shareable across both target families."""
    lists = ["experiments/stage3a/target_P_matches.json",
             "experiments/stage3a/target_E_matches.json"]
    for value in lists:
        if not (driver.REPO / value).is_file():
            pytest.skip("the frozen stage 3a match lists are not committed")
    raw = yaml.safe_load(CONFIG_PATH.read_text())
    raw["training"][driver.REPORT_MATCH_LISTS_KEY] = lists
    path = tmp_path / "report_lists.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    loaded = load_config(path)
    assert loaded["training"][driver.REPORT_MATCH_LISTS_KEY] == lists

    for config_id in configurations(loaded):
        # ...the pinned identity components are bit-for-bit what they were.
        assert _pinned_components(loaded, config_id) == (
            STAGE2_SIGNATURE_PIN[config_id]), config_id
        entry = configurations(loaded)[config_id]
        digest = (PIN_BASE_LOGITS_DIGEST
                  if entry["arm"] in driver.t1.ARMS_NEEDING_BASE_LOGITS
                  else None)
        effective = effective_settings(loaded, entry,
                                       base_logits_digest=digest)
        # It is in no param table, no arm_params block and no schedule.
        assert driver.REPORT_MATCH_LISTS_KEY not in driver.STAGE3_PARAMS
        assert driver.REPORT_MATCH_LISTS_KEY not in effective["arm_params"]
        assert driver.REPORT_MATCH_LISTS_KEY not in effective
        assert "schedule" not in effective
        # ...but it IS rendered, for every configuration, or nothing measures.
        args = driver._trainer_args(loaded, entry, 7, Path("out"), effective)
        assert driver.REPORT_MATCH_LISTS_FLAG in args
        assert ",".join(lists) in args
        # ...and the signature is the one the config without it produces.
        assert driver.signature_for(
            effective, _resolved())["training_signature"] == (
            driver.signature_for(
                effective_settings(config, configurations(config)[config_id],
                                   base_logits_digest=digest),
                _resolved())["training_signature"]), config_id


def test_report_match_lists_is_validated(tmp_path):
    raw = yaml.safe_load(CONFIG_PATH.read_text())
    for bad in ([], "one.json", [""], ["no/such/list.json"],
                ["experiments/stage3a/target_P_matches.json",
                 "experiments/stage3a/target_P_matches.json"]):
        raw["training"][driver.REPORT_MATCH_LISTS_KEY] = bad
        path = tmp_path / "bad.yaml"
        path.write_text(yaml.safe_dump(raw, sort_keys=False))
        with pytest.raises(driver.RetrainError,
                           match=driver.REPORT_MATCH_LISTS_KEY):
            load_config(path)


def test_stage2_configurations_render_an_unchanged_argv(config):
    """No stage 3 flag appears in an existing configuration's command."""
    stage3_flags = {flag for _, flag, _ in driver.STAGE3_PARAMS.values()}
    for config_id, entry in configurations(config).items():
        args = driver._trainer_args(
            config, entry, 7, Path("out"),
            effective_settings(config, entry,
                               base_logits_digest=PIN_BASE_LOGITS_DIGEST))
        assert not stage3_flags & set(args), config_id


# ------------------------------------------- stage 3 Block B driver params

def _stage3_entry(config: dict, **params) -> dict:
    """A copy of the `mlp` entry with stage 3 params, run through the checks."""
    entry = dict(configurations(config)["mlp"])
    entry["params"] = dict(params)
    entry["_params"] = driver._check_params(CONFIG_PATH, entry)
    return entry


def test_stage3_params_are_admitted_and_typed(config):
    entry = _stage3_entry(config, tier_embed=8, train_tier=3,
                          max_steps=3840, eval_every=128)
    assert entry["_params"]["tier_embed"] == 8
    assert entry["_params"]["max_steps"] == 3840
    for bad in ({"tier_embed": "8"}, {"train_tier": 3.0},
                {"max_steps": True}, {"train_match_list": 5}):
        with pytest.raises(driver.RetrainError, match="params."):
            _stage3_entry(config, **bad)


def test_step_budget_must_be_a_pair(config):
    with pytest.raises(driver.RetrainError, match="must be set together"):
        _stage3_entry(config, max_steps=3840)
    with pytest.raises(driver.RetrainError, match="must be set together"):
        _stage3_entry(config, eval_every=128)


def test_tier_embed_is_refused_for_a_non_token_mlp_arm(config):
    entry = dict(configurations(config)["full"])
    entry["params"] = {"tier_embed": 8}
    with pytest.raises(driver.RetrainError, match="token_mlp"):
        driver._check_params(CONFIG_PATH, entry)


def test_train_tier_range_is_checked(config):
    with pytest.raises(driver.RetrainError, match="train_tier"):
        _stage3_entry(config, train_tier=0)
    with pytest.raises(driver.RetrainError, match="train_tier"):
        _stage3_entry(config, train_tier=9)


def test_missing_match_list_is_refused_before_the_run(config):
    with pytest.raises(driver.RetrainError, match="does not exist"):
        _stage3_entry(config, train_match_list="experiments/stage3a/nope.json")


def test_match_list_sha256_enters_the_arm_params_component(config, tmp_path):
    path = tmp_path / "list.json"
    path.write_text('{"match_ids": ["1", "2"]}')
    relative = path.relative_to(driver.REPO) if str(path).startswith(
        str(driver.REPO)) else None
    if relative is None:  # tmp_path is outside the repo on this machine
        relative = Path("experiments/stage3a/target_P_matches.json")
        if not (driver.REPO / relative).is_file():
            pytest.skip("no frozen match list available to hash")
    entry = _stage3_entry(config, tier_embed=8,
                          train_match_list=relative.as_posix())
    params = entry["_params"]
    assert len(params["train_match_list_sha256"]) == 64
    block = driver.expected_arm_params(entry, None)
    assert block["train_match_list_sha256"] == (
        params["train_match_list_sha256"])
    assert block["tier_embed"] == 8
    # ...and a different list is a different identity.
    assert driver._digest(block) != driver._digest(
        driver.expected_arm_params(configurations(config)["mlp"], None))


def test_step_budget_joins_training_block_not_the_optimiser(config):
    entry = _stage3_entry(config, max_steps=3840, eval_every=128)
    effective = effective_settings(config, entry)
    # The optimiser block is compared field-by-field against the checkpoint's
    # training_contract, which the trainer does not grow, so the budget must
    # NOT be in it.
    assert "max_steps" not in effective["optimiser"]
    assert effective["schedule"] == {"max_steps": 3840, "eval_every": 128}
    args = driver._trainer_args(config, entry, 7, Path("out"), effective)
    assert "--max-steps" in args and "3840" in args
    assert "--eval-every" in args and "128" in args


def test_role_is_optional_and_validated(tmp_path):
    raw = yaml.safe_load(CONFIG_PATH.read_text())
    raw["configurations"][0]["role"] = "control"
    raw["configurations"][1]["role"] = "candidate"
    path = tmp_path / "roles.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    loaded = load_config(path)
    assert configurations(loaded)["mlp"]["role"] == "control"
    # ...and it changes nothing about what the configuration trains.
    assert _pinned_components(loaded, "mlp") == STAGE2_SIGNATURE_PIN["mlp"]

    raw["configurations"][0]["role"] = "shared_control"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    with pytest.raises(driver.RetrainError, match="role"):
        load_config(path)


# ------------------------------- stage 4 rung 4b / 4d driver params

def test_target_tier_is_not_part_of_the_signature(config):
    """Code gate finding 4: reporting-only, so a pooled checkpoint is shared
    across both target families instead of forking per measurement."""
    assert driver.STAGE3_PARAMS["target_tier"][2] == "reporting"
    plain = _stage3_entry(config, tier_embed=8)
    measured = _stage3_entry(config, tier_embed=8, target_tier=3)
    assert driver._digest(driver.expected_arm_params(plain, None)) == (
        driver._digest(driver.expected_arm_params(measured, None)))
    assert "target_tier" not in driver.expected_arm_params(measured, None)
    assert "schedule" not in effective_settings(config, measured)
    # ...but it IS rendered, or the trainer would never measure anything.
    args = driver._trainer_args(config, measured, 7, Path("out"),
                                effective_settings(config, measured))
    assert "--target-tier" in args and "3" in args


def test_target_tier_may_not_contradict_train_tier(config):
    with pytest.raises(driver.RetrainError, match="contradicts"):
        _stage3_entry(config, train_tier=3, target_tier=2)
    entry = _stage3_entry(config, train_tier=3, target_tier=3)
    assert entry["_params"]["target_tier"] == 3


def test_extra_features_param_resolves_a_registered_set(config):
    entry = _stage3_entry(config, extra_features={
        "dir": "models/embeddings/stage4/exposure", "cols": "counts"})
    resolved = entry["_params"]["extra_features"]
    assert resolved["cols"] == ["batter_N_asof", "bowler_N_asof"]
    assert resolved["col_set"] == "counts"
    assert set(resolved["sha256"]) == set(driver.CONTRACT_SPLITS)
    block = driver.expected_arm_params(entry, None)
    assert block["extra_features"]["cols"] == resolved["cols"]
    # The ordered column list is identity: spread_recency is a different model.
    other = _stage3_entry(config, extra_features={
        "dir": "models/embeddings/stage4/exposure",
        "cols": "spread_recency"})
    assert driver._digest(block) != driver._digest(
        driver.expected_arm_params(other, None))
    args = driver._trainer_args(config, entry, 7, Path("out"),
                                effective_settings(config, entry))
    assert "--extra-features" in args
    assert "batter_N_asof,bowler_N_asof" in args


def test_spread_recency_set_is_counts_plus_sd_and_recency(config):
    counts = driver.EXTRA_FEATURE_SETS["counts"]
    spread = driver.EXTRA_FEATURE_SETS["spread_recency"]
    assert spread[:2] == counts
    assert len([c for c in spread if "_var_p" in c]) == 12
    assert spread[-2:] == ["batter_recent_N", "bowler_recent_N"]
    assert len(spread) == 16


def test_extra_features_param_is_validated(config):
    for bad in ({"dir": "models/embeddings/stage4/exposure"},
                {"dir": "", "cols": "counts"},
                {"dir": "models/embeddings/stage4/exposure", "cols": "nope"},
                {"dir": "models/embeddings/stage4/exposure", "cols": []},
                {"dir": "x", "cols": "counts", "extra": 1}):
        with pytest.raises(driver.RetrainError):
            _stage3_entry(config, extra_features=bad)
    with pytest.raises(driver.RetrainError, match="is missing"):
        _stage3_entry(config, extra_features={"dir": "no/such/dir",
                                              "cols": "counts"})


def test_extra_features_refused_for_a_sequence_arm(config):
    entry = dict(configurations(config)["full"])
    entry["params"] = {"extra_features": {
        "dir": "models/embeddings/stage4/exposure", "cols": "counts"}}
    with pytest.raises(driver.RetrainError, match="token_mlp"):
        driver._check_params(CONFIG_PATH, entry)


def _identity_entry(config, **params):
    entry = dict(configurations(config)["mlp"])
    entry["arm"] = "identity_residual"
    entry["params"] = dict(params)
    entry["_params"] = driver._check_params(CONFIG_PATH, entry)
    return entry


def test_identity_residual_requires_its_frozen_reference(config):
    with pytest.raises(driver.RetrainError, match="base_probs_dir"):
        _identity_entry(config)
    refs = "models/embeddings/stage4/refs"
    if not (driver.REPO / refs / "eb_ctx_train_probs.npz").is_file():
        pytest.skip("the frozen reference's train probabilities are not built")
    entry = _identity_entry(config, base_probs_dir=refs,
                            residual_lambda=0.001)
    block = driver.expected_arm_params(entry, None)
    assert set(block["base_probs_sha256"]) == set(driver.CONTRACT_SPLITS)
    assert block["residual_lambda"] == 0.001
    # Two lambdas are two different models.
    other = _identity_entry(config, base_probs_dir=refs,
                            residual_lambda=0.01)
    assert driver._digest(block) != driver._digest(
        driver.expected_arm_params(other, None))
    args = driver._trainer_args(config, entry, 7, Path("out"),
                                effective_settings(config, entry))
    assert "--base-probs-dir" in args and "--residual-lambda" in args


def test_base_probs_params_are_refused_for_other_arms(config):
    with pytest.raises(driver.RetrainError, match="not accepted by arm"):
        _stage3_entry(config, base_probs_dir="models/embeddings/stage4/refs")
    with pytest.raises(driver.RetrainError, match="meaningless"):
        _stage3_entry(config, residual_lambda=0.001)


# ------------------------- reviewer MUST-FIX 1: frozen-input drift

NIGHT3_CONFIG = driver.REPO / "experiments/configs/seq_stage3_night3_v1.yaml"


def _night3_raw():
    if not NIGHT3_CONFIG.is_file():
        pytest.skip("the night 3 config is not registered in this checkout")
    if not (driver.REPO / "experiments/stage3a/manifest.json").is_file():
        pytest.skip("the stage 3a freeze manifest is not committed")
    return yaml.safe_load(NIGHT3_CONFIG.read_text())


def _write(tmp_path: Path, raw: dict, name="night3.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return path


def test_stage2_config_reads_no_freeze_manifest(config):
    """A config that pins no frozen input is untouched: nothing is read and
    no freeze record is produced."""
    assert config.get("_freeze") is None


def test_night3_config_verifies_every_frozen_list(tmp_path):
    loaded = load_config(NIGHT3_CONFIG) if NIGHT3_CONFIG.is_file() else None
    if loaded is None:
        pytest.skip("the night 3 config is not registered in this checkout")
    freeze = loaded["_freeze"]
    assert freeze["manifest"] == "experiments/stage3a/manifest.json"
    assert len(freeze["manifest_sha256"]) == 64
    # Every list the config names was compared, not merely hashed.
    named = {Path(entry["_params"]["train_match_list"]).name
             for entry in loaded["configurations"]
             if entry["_params"].get("train_match_list")}
    assert named and named <= set(freeze["lists_verified"])
    assert freeze["steps"]["max_steps"] == 3840
    assert freeze["steps"]["eval_every"] == 128


def test_a_drifted_training_list_is_refused(tmp_path):
    """The MUST-FIX itself: an edited list must be REJECTED, not re-hashed."""
    raw = _night3_raw()
    stage3a = tmp_path / "stage3a"
    stage3a.mkdir()
    source = driver.REPO / "experiments/stage3a"
    for item in source.glob("*.json"):
        shutil.copy2(item, stage3a / item.name)
    # ...drift one list by one match id.
    victim = stage3a / "target_E_matches.json"
    payload = json.loads(victim.read_text())
    payload["match_ids"] = payload["match_ids"][:-1]
    victim.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    relative = stage3a.relative_to(driver.REPO) if str(stage3a).startswith(
        str(driver.REPO)) else None
    if relative is None:
        pytest.skip("tmp_path is outside the repository on this machine")
    raw["experiment"]["freeze_manifest"] = (
        relative / "manifest.json").as_posix()
    raw.pop("freeze", None)
    for entry in raw["configurations"]:
        listed = (entry.get("params") or {}).get("train_match_list")
        if listed:
            entry["params"]["train_match_list"] = (
                relative / Path(listed).name).as_posix()
    with pytest.raises(driver.RetrainError, match="has drifted"):
        load_config(_write(tmp_path, raw))


def test_a_step_budget_that_is_not_the_frozen_one_is_refused(tmp_path):
    raw = _night3_raw()
    raw["configurations"][0].setdefault("params", {})["max_steps"] = 1920
    with pytest.raises(driver.RetrainError, match="frozen budget"):
        load_config(_write(tmp_path, raw))


def test_a_list_the_manifest_does_not_record_is_refused(tmp_path):
    raw = _night3_raw()
    stray = driver.REPO / "experiments/stage3a/_stray_list.json"
    stray.write_text('{"match_ids": ["1"]}')
    try:
        for entry in raw["configurations"]:
            if (entry.get("params") or {}).get("train_match_list"):
                entry["params"]["train_match_list"] = (
                    "experiments/stage3a/_stray_list.json")
                break
        with pytest.raises(driver.RetrainError, match="records no list file"):
            load_config(_write(tmp_path, raw))
    finally:
        stray.unlink()


def test_a_moved_manifest_is_refused(tmp_path):
    raw = _night3_raw()
    raw["experiment"]["freeze_manifest"] = "experiments/stage3a/nope.json"
    with pytest.raises(driver.RetrainError, match="does not exist"):
        load_config(_write(tmp_path, raw))
