"""Tests for `scripts/sequence_track/stage2_stats.py` (D9, D10.1-D10.7, D10.11).

Synthetic fixtures only: a tiny fake frame, a tiny fake runs tree, a tiny
fake cricsheet corpus for the block lookup.  No real run, no real corpus, no
cohort, no smoke, no sealed holdout is touched.
"""
from __future__ import annotations

import ast
import builtins
import hashlib
import io
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from registered_experiment import (  # noqa: E402
    match_cluster_ci,
    seed_mean_match_cluster_ci,
)
from sequence_track import stage2_stats as st  # noqa: E402

SEEDS = (7, 13)
# Four innings per "match", eight matches over two events: enough rows for a
# block bootstrap and few enough to build by hand.
N_MATCHES = 8
BALLS_PER_INNINGS = 12


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _frame_rows(n_matches: int = N_MATCHES) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(3)
    for match in range(n_matches):
        match_id = f"90000{match}"
        for innings in (1, 2):
            for ball in range(BALLS_PER_INNINGS):
                death = 1 if ball >= BALLS_PER_INNINGS - 3 else 0
                middle = 1 if 3 <= ball < BALLS_PER_INNINGS - 3 else 0
                rows.append({
                    "innings_id": f"{innings}_{match_id}",
                    "inning_idx": innings,
                    "is_powerplay": 1 if ball < 3 else 0,
                    "is_middle_overs": middle,
                    "is_death_overs": death,
                    "chase_target": 0.0 if innings == 1 else 150.0,
                    "ball_outcome": int(rng.choice([0, 1, 2, 4, 6, -1])),
                })
    return pd.DataFrame(rows)


# The frame directory of the test currently running. A fixture run's signature
# components are anchored to that frame's pin exactly as a real run's are, and
# every `_write_run` call in this module belongs to one frame, so recording it
# here keeps the helper signatures short.
_FRAME_DIR: Path | None = None


@pytest.fixture
def frame_dir(tmp_path: Path) -> Path:
    global _FRAME_DIR
    directory = tmp_path / "frame"
    directory.mkdir()
    df = _frame_rows()
    df.to_parquet(directory / "cricket_data_i7_validation.parquet",
                  index=False)
    _FRAME_DIR = directory
    return directory


def _current_frame_dir(frame_dir: Path | None = None) -> Path:
    directory = frame_dir or _FRAME_DIR
    if directory is None:
        raise AssertionError("this helper needs the frame_dir fixture")
    return directory


@pytest.fixture
def block_source(tmp_path: Path) -> Path:
    """A tiny cricsheet-shaped corpus: two events, four matches each."""
    directory = tmp_path / "t20s_json"
    directory.mkdir()
    for match in range(N_MATCHES):
        event = "Synthetic League A" if match < 4 else "Synthetic League B"
        day = 1 + match % 4
        payload = {"info": {"dates": [f"2025-0{1 if match < 4 else 7}-0{day}"],
                            "event": {"name": event},
                            "teams": [f"T{match}", f"U{match}"],
                            "gender": "male", "match_type": "T20"}}
        (directory / f"90000{match}.json").write_text(json.dumps(payload))
    return directory


CONFIG_IDS = ("mlp", "full", "fixed_decay", "fox")
# The five registered `same_entity` ids: note the unrestricted arm is
# `same_entity_unr`, NOT `same_entity_kunr` — nothing may build these by
# string template.
K_IDS = {"0": "same_entity_k0", "6": "same_entity_k6",
         "12": "same_entity_k12", "30": "same_entity_k30",
         "unr": "same_entity_unr"}
SIGNATURE = "a" * 64
# The pin's own facts, so a synthetic run's signature components can be built
# the way the driver builds them and checked the way the analysis checks them.
CACHE_MD5 = "cafe" * 8
CACHE_ROLE = "stats_cache_i7"
BASE_LOGITS_DIGEST = "b" * 32
TRAIN_SPLIT = {"path": "frame/cricket_data_i7_train.parquet",
               "md5": "dd" * 16, "n_rows": 1000,
               "match_date_min": "2005-02-17",
               "match_date_max": "2024-12-30"}


def _family(candidate: str, reference: str) -> dict:
    return {"candidate": candidate,
            "primary": {"contrast": f"{candidate} - {reference}",
                        "reference": reference, "slice": "all"},
            "death_gate": {"contrast": f"{candidate} - mlp",
                           "reference": "mlp", "slice": "death"},
            "chase_gate": {"contrast": f"{candidate} - mlp",
                           "reference": "mlp", "slice": "chase"},
            "holm_group": f"family_{candidate}"}


def _same_entity_entries() -> list[dict]:
    """The five registered `same_entity` configurations, real ids."""
    return [{"id": config_id, "arm": "same_entity",
             "params": {"k": (k if k == "unr" else int(k))},
             "access": {"features": True, "history": True, "identity": True,
                        "prod_logits": False},
             "history_input": "aligned", "wiring": "relay_free",
             "reference": ["mlp"], "tests": "k sweep", "queue_order": 9}
            for k, config_id in K_IDS.items()]


def _config_payload(config_ids=CONFIG_IDS, frame_dir: Path | None = None,
                    same_entity: bool = False) -> dict:
    payload = {
        "experiment": {"name": "seq_stage2_test",
                       "sign_convention": "candidate minus reference",
                       "evidence_status": "screening"},
        "data": {"directory": "frame", "n_features": 50,
                 "frame_role": "ball_frame_i7",
                 "stats_cache_role": "stats_cache_i7"},
        "training": {"seeds": list(SEEDS), "device": "mps", "dmodel": 128,
                     "layers": 2, "heads": 4, "batch": 128, "epochs": 30,
                     "learning_rate": 0.0003, "patience": 3, "aux": False,
                     "no_kit": True, "score_test": False},
        "configurations": [
            {"id": config_id, "arm": config_id, "params": {},
             "access": {"features": True, "history": config_id != "mlp",
                        "identity": False, "prod_logits": False},
             "history_input": "none" if config_id == "mlp" else "prev",
             "wiring": "token_mlp" if config_id == "mlp" else "standard",
             "reference": [] if config_id == "mlp" else ["mlp"],
             "tests": "synthetic", "queue_order": 1}
            for config_id in config_ids],
        "statistics": {
            "families": {"primary_slice": "all", "death_gate_slice": "death",
                         "chase_gate_slice": "chase",
                         "definition": "three members",
                         "gate_reference_rule": "against mlp",
                         "holm": {"scope": "within a family",
                                  "adjustment": "step-down",
                                  "ties": "registered order",
                                  "missing_member": "placeholder"},
                         "map": [_family("full", "mlp"),
                                 _family("fixed_decay", "mlp"),
                                 _family("fox", "fixed_decay")]},
            "bootstrap": {"reps": 50, "seed": 29, "weighting": "ball_weighted",
                          "resample": "tournament blocks", "min_blocks": 10,
                          "below_min_blocks": "descriptive",
                          "blocks_from": "load_competition_clusters",
                          "block_source_dir": "t20s_json"},
            "non_inferiority": {"margin_ll": 0.002, "sign": "strict",
                                "applies_to": ["death", "chase"],
                                "reference": "mlp",
                                "not_imported_from_stage_1": "no parity band"},
            "interval_labelling": {"rank_local": st.RANK_LOCAL_NOTE},
            "k_selection": {"tolerance_ll": 0.002,
                            "registered_sweep": [0, 6, 12, 30, "unr"],
                            "rule": "best mean unless within tolerance"},
        },
        "deviations": [{"id": "dev_one", "statement": "a registered deviation",
                        "reason": "because"}],
        "known_asymmetries": [{"id": "asym_one", "text": "an asymmetry"}],
        "known_limitations": ["a limitation that must survive the render"],
    }
    if same_entity:
        payload["configurations"].extend(_same_entity_entries())
        payload["statistics"]["families"]["map"].extend(
            _family(config_id, "mlp") for config_id in K_IDS.values())
    if frame_dir is not None:
        parquet = frame_dir / "cricket_data_i7_validation.parquet"
        payload["provenance"] = {
            "pinned_by": "test",
            "config_body_sha256": "0" * 64,
            "frame": {"dir": frame_dir.name, "version": "i7",
                      "feature_hash": {"hash": "c520a3ba08ae"},
                      "splits": {
                          "train": dict(TRAIN_SPLIT),
                          "validation": {
                              "path": str(parquet),
                              "md5": st.md5_file(parquet),
                              "n_rows": N_MATCHES * 2 * BALLS_PER_INNINGS,
                              "match_date_min": "2024-12-31",
                              "match_date_max": "2025-06-29"}}},
            "stats_cache": {"role": CACHE_ROLE, "md5": CACHE_MD5},
            "base_logits": {"train_validation_digest": BASE_LOGITS_DIGEST},
            "sources": {"source_sha256": _pinned_source_sha256()},
        }
    return payload


def _pinned_source_sha256() -> dict:
    """Fake but well-formed hashes for the five implementation sources."""
    driver = st.training_driver()
    return {name: st.sha256_text(name)
            for name in (driver.TRAINER_SOURCE, driver.FEATURE_CONTRACT_SOURCE,
                         driver.ARTIFACT_RESOLVER_SOURCE,
                         driver.FEATURE_REGISTRY_SOURCE,
                         driver.RECURRENT_SOURCE)}


def _entries(same_entity: bool = True) -> dict[str, dict]:
    """Every registered configuration entry, by id."""
    return {str(entry["id"]): entry for entry
            in _config_payload(same_entity=same_entity)["configurations"]}


def _anchor(frame_dir: Path) -> tuple[dict, st.Pin, dict]:
    config = _config_payload(frame_dir=frame_dir, same_entity=True)
    pin = st.load_pin(config)
    return config, pin, st.pinned_component_digests(config, pin)


def _registered_arm_params(config_id: str, entry: Mapping[str, Any],
                           pin: st.Pin) -> dict:
    """`arm_params_expected` in the driver's shape, for one configuration."""
    identity = st.expected_arm_identity(config_id, entry, pin)
    fields = identity["arm_params_fields"]
    return {"arm": fields["arm"], "k": fields["k"],
            "wiring": fields["wiring"],
            "history_input": fields["history_input"],
            "key_construction": identity["key_construction"],
            "bias": None, "residual_l2": fields["residual_l2"],
            "base_logits_md5": fields["base_logits_md5"]}


@pytest.fixture
def config_path(tmp_path: Path, frame_dir: Path) -> Path:
    path = tmp_path / "seq_stage2_test.yaml"
    path.write_text(yaml.safe_dump(_config_payload(frame_dir=frame_dir),
                                   sort_keys=False))
    return path


@pytest.fixture
def k_config(frame_dir: Path) -> dict:
    return _config_payload(frame_dir=frame_dir, same_entity=True)


def _components(config_id: str, frame_dir: Path | None = None,
                arm_params_expected: Mapping[str, Any] | None = None,
                **override) -> dict:
    """The signature components a real run of this configuration would carry.

    Every shared component is the digest the analysis recomputes FROM the pin,
    so a fixture run is admissible for the same reason a real one is. Keyword
    overrides simulate drift (a moved frame, a foreign cache, an arm's
    implementation set).
    """
    config, pin, anchored = _anchor(_current_frame_dir(frame_dir))
    entry = {str(e["id"]): e for e in config["configurations"]}[config_id]
    identity = st.expected_arm_identity(config_id, entry, pin)
    if arm_params_expected is None:
        arm_params_expected = _registered_arm_params(config_id, entry, pin)
    components = {
        "config_id": identity["config_id"],
        "arm": identity["arm"],
        "arm_params": st.component_digest(dict(arm_params_expected)),
        "training_block": anchored["training_block"],
        "frame": anchored["frame"],
        "stats_cache": anchored["stats_cache"],
        "base_logits": anchored["base_logits_by_reads_base"][
            identity["reads_base_logits"]],
        "implementation": anchored["implementation_by_recurrent"][
            identity["recurrent"]],
    }
    components.update(override)
    return components


def _write_run(runs_root: Path, config_id: str, seed: int, frame: pd.DataFrame,
               *, bias: float = 0.0, innings_override=None,
               y_override=None, n_params: int = 1000,
               extra_arm_params: dict | None = None,
               arm: str | None = None, k: Any = None,
               signature: str | None = None,
               components: dict | None = None,
               record_seed: int | None = None,
               record_config_id: str | None = None,
               break_manifest: bool = False,
               frame_dir: Path | None = None,
               arm_params_expected: dict | None = None,
               validation_ll: float = 1.5,
               drop_arm_params_expected: bool = False) -> Path:
    directory = runs_root / config_id / f"seed_{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(abs(hash((config_id, seed))) % (2 ** 31))
    logits = rng.normal(scale=0.5, size=(len(frame), 6))
    y = frame["ball_outcome"].map(st.CLASS_MAPPING).to_numpy(np.int64)
    # `bias` pushes probability mass onto the true class, so a larger bias is
    # a better (lower) log loss and the delta sign is controllable.
    logits[np.arange(len(y)), y] += bias
    probs = np.exp(logits)
    probs = (probs / probs.sum(axis=1, keepdims=True)).astype(np.float32)
    innings = (frame["innings_id"].astype(str).to_numpy()
               if innings_override is None else innings_override)
    np.savez_compressed(directory / "predictions_validation.npz",
                        probs=probs,
                        y=(y if y_override is None else y_override),
                        innings_id=np.asarray(innings, dtype=str))
    directory_frame = _current_frame_dir(frame_dir)
    config, pin, _ = _anchor(directory_frame)
    entry = {str(e["id"]): e
             for e in config["configurations"]}.get(config_id)
    if arm_params_expected is None and entry is not None:
        arm_params_expected = _registered_arm_params(config_id, entry, pin)
    arm_params_expected = dict(arm_params_expected or {})
    # The trainer's own block: the registered identity plus derived fields.
    arm_params = dict(arm_params_expected)
    arm_params.setdefault("arm", arm or config_id)
    arm_params.setdefault("k", k)
    arm_params.setdefault("wiring", "standard")
    arm_params.setdefault("history_input", "prev")
    arm_params.setdefault("key_construction", "shifted_history")
    arm_params.setdefault("bias", None)
    arm_params.setdefault("residual_l2", None)
    arm_params.setdefault("base_logits_md5", None)
    arm_params.update({"positional_embedding": True, "n_parameters": n_params})
    if arm is not None:
        arm_params["arm"] = arm
    if k is not None:
        arm_params["k"] = k
    arm_params.update(extra_arm_params or {})
    (directory / "metrics.json").write_text(json.dumps({
        "config": {"arm": arm or config_id, "seed": seed,
                   "out": str(directory)},
        "arm_params": arm_params,
        "validation_ll": float(validation_ll),
        "training_contract": {"mps_bit_reproducible": False,
                              "device": "mps"}}))
    (directory / "model.pt").write_bytes(b"not a real checkpoint")
    components = components or _components(
        config_id, directory_frame, arm_params_expected or None)
    # The signature is the driver's own function of its components, so a
    # fixture run passes the recomputation check for the same reason a real
    # run does, and an explicitly passed `signature` is a deliberate forgery.
    signature = signature or st.recompute_training_signature(components)
    (directory / "run_record.json").write_text(json.dumps({
        "config_id": record_config_id or config_id,
        "seed": record_seed if record_seed is not None else seed,
        "wall_seconds": 12.3,
        # The driver's own nested shape (D8.8), not flat keys.
        "machine_provenance": {
            "hostname": "host7" if seed == 7 else "host13",
            "machine": "laptop" if seed == 7 else "mini",
            "machine_label_source": "--machine",
            "chip": "Apple M-series", "chip_source": "sysctl",
            "os": "Darwin", "os_release": "25.4.0",
            "platform": "macOS-15", "torch_version": "2.4.0",
            "torch_mps_available": True, "torch_mps_built": True,
            "device_requested": "mps", "device_used": "mps",
            "thread_caps": {"OMP_NUM_THREADS": "2" if seed == 7 else "4",
                            "OPENBLAS_NUM_THREADS": "2",
                            "MKL_NUM_THREADS": "2",
                            "VECLIB_MAXIMUM_THREADS": "2"},
            "repo_is_worktree": seed == 7},
        "training_signature": signature,
        "training_signature_components": components,
        **({} if drop_arm_params_expected
           else {"arm_params_expected": arm_params_expected}),
    }))
    manifest = {}
    for name in st.RUN_ARTEFACTS:
        path = directory / name
        manifest[name] = {"bytes": path.stat().st_size,
                          "md5": st.md5_file(path)}
    if break_manifest:
        manifest["model.pt"]["md5"] = "0" * 32
    (directory / "COMPLETE.json").write_text(json.dumps({
        "config_id": record_config_id or config_id,
        "seed": record_seed if record_seed is not None else seed,
        "artifacts": manifest, "training_signature": signature,
        "training_signature_components": components}))
    return directory


def checkpoint_md5(runs_root: Path, config_id: str, seed: int) -> str:
    return st.md5_file(runs_root / config_id / f"seed_{seed}" / "model.pt")


def _restamp_metrics(directory: Path) -> None:
    """Re-record `metrics.json` in the completion manifest after an edit.

    Only that one entry: a test that deliberately corrupted another artefact's
    manifest entry must keep its corruption.
    """
    completion_path = directory / "COMPLETE.json"
    if not completion_path.exists():
        return
    completion = json.loads(completion_path.read_text())
    metrics_path = directory / "metrics.json"
    (completion.setdefault("artifacts", {}))["metrics.json"] = {
        "bytes": metrics_path.stat().st_size,
        "md5": st.md5_file(metrics_path)}
    completion_path.write_text(json.dumps(completion))


def _sync_metrics_ll(runs_root: Path, config_id: str, seed: int,
                     value: float) -> None:
    """Make the run's own `metrics.json` agree with the summary it feeds.

    The driver writes `summary.yaml`'s `ll` straight from
    `metrics.json[validation_ll]`, so a consistent fixture must too. A test
    that wants a hand-edited summary passes `tamper_ll=True` instead.
    """
    directory = runs_root / config_id / f"seed_{seed}"
    metrics_path = directory / "metrics.json"
    if not metrics_path.exists():
        return
    metrics = json.loads(metrics_path.read_text())
    metrics["validation_ll"] = float(value)
    metrics_path.write_text(json.dumps(metrics))
    _restamp_metrics(directory)


def _write_summary(runs_root: Path, config_id: str,
                   per_seed: dict[int, float | None],
                   signature: str | None = None,
                   tamper_ll: bool = False) -> Path:
    path = runs_root / config_id / "summary.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    if signature is None:
        record = (runs_root / config_id / f"seed_{sorted(per_seed)[0]}"
                  / "run_record.json")
        signature = (json.loads(record.read_text())["training_signature"]
                     if record.exists()
                     else st.sha256_text(f"signature::{config_id}"))
    rows = []
    for seed, value in per_seed.items():
        if value is None:
            continue
        if not tamper_ll:
            _sync_metrics_ll(runs_root, config_id, seed, value)
        model = runs_root / config_id / f"seed_{seed}" / "model.pt"
        rows.append({"seed": seed, "ll": value, "best_epoch": 4,
                     "wall_seconds": 12.3,
                     "checkpoint_md5": (st.md5_file(model)
                                        if model.exists() else None),
                     "checkpoint_dir": f"runs/{config_id}/seed_{seed}",
                     "training_signature": signature})
    path.write_text(yaml.safe_dump({
        "experiment": {"runs_recorded": len(rows), "complete": len(rows) == 2,
                       "training_signature": signature},
        "splits": {"validation": {"n_rows": 192, "n_matches": N_MATCHES,
                                 "per_seed": rows}}}, sort_keys=False))
    return path


@pytest.fixture
def runs_root(tmp_path: Path, frame_dir: Path) -> Path:
    root = tmp_path / "runs"
    frame = pd.read_parquet(frame_dir / "cricket_data_i7_validation.parquet")
    biases = {"mlp": 0.0, "full": 0.9, "fixed_decay": 0.5, "fox": 0.7}
    for config_id, bias in biases.items():
        for seed in SEEDS:
            _write_run(root, config_id, seed, frame, bias=bias)
        _write_summary(root, config_id, {7: 1.5 - bias / 10,
                                        13: 1.51 - bias / 10})
    return root


EXPECT = {"rows": N_MATCHES * 2 * BALLS_PER_INNINGS, "matches": N_MATCHES,
          "blocks": 2, "unmapped": 0}
# `resolve_blocks` takes the same four numbers as `expect_*` keywords.
EXPECT_KW = {f"expect_{name}": value for name, value in EXPECT.items()}


def _run_stats(config_path: Path, runs_root: Path, frame_dir: Path,
               block_source: Path, tmp_path: Path, **kwargs) -> dict:
    params = {"reps": 50, "seed": 29, "seeds": SEEDS,
              "expect": dict(EXPECT), "expected_families": 3}
    params.update(kwargs)
    return st.compute_statistics(
        config_path, runs_root, frame_dir, block_source,
        tmp_path / "missing_base_logits.npz", **params)


# ---------------------------------------------------------------------------
# registered constants
# ---------------------------------------------------------------------------

def test_class_mapping_matches_the_frame_contract():
    from embeddings_e1 import CLASS_MAPPING

    assert st.CLASS_MAPPING == CLASS_MAPPING


def test_registered_contract_constants():
    assert st.BOOTSTRAP_CONTRACT_VERSION == "tournament_time_block_v1"
    assert st.MAX_EVENT_GAP_DAYS == 120
    assert (st.REPS, st.RNG_SEED, st.MARGIN_LL, st.ALPHA) == (
        2000, 29, 0.002, 0.05)
    assert st.EXPECTED_ROWS, st.EXPECTED_MATCHES == (124292, 545)
    assert (st.EXPECTED_BLOCKS, st.EXPECTED_UNMAPPED) == (47, 0)
    assert st.ALLOWED_STATUSES == ("SCREEN_PASS", "SCREEN_NOT_PASS",
                                   "NOT_EVALUABLE")
    assert not any("advance" in status.lower()
                   for status in st.ALLOWED_STATUSES)


def test_no_status_may_contain_advance():
    with pytest.raises(st.RefusalError):
        st._check_status("SCREEN_ADVANCE")


def test_no_stage_one_margin_or_classification_is_imported():
    source = (REPO / "scripts" / "sequence_track"
              / "stage2_stats.py").read_text()
    assert "0.007" not in source
    assert "holm_stage1" not in source
    assert "PARITY_BAND" not in source


# ---------------------------------------------------------------------------
# D10.1 alignment
# ---------------------------------------------------------------------------

def test_alignment_refuses_reordered_innings_ids(tmp_path, frame_dir,
                                                 block_source, config_path):
    root = tmp_path / "runs"
    frame = pd.read_parquet(frame_dir / "cricket_data_i7_validation.parquet")
    shuffled = frame["innings_id"].astype(str).to_numpy()[::-1]
    for config_id in CONFIG_IDS:
        for seed in SEEDS:
            override = shuffled if config_id == "fox" else None
            _write_run(root, config_id, seed, frame,
                       innings_override=override)
        _write_summary(root, config_id, {7: 1.5, 13: 1.5})
    with pytest.raises(st.RefusalError, match="innings_id ordering"):
        _run_stats(config_path, root, frame_dir, block_source, tmp_path)


def test_alignment_refuses_a_relabelled_y_vector(tmp_path, frame_dir,
                                                 block_source, config_path):
    root = tmp_path / "runs"
    frame = pd.read_parquet(frame_dir / "cricket_data_i7_validation.parquet")
    y = frame["ball_outcome"].map(st.CLASS_MAPPING).to_numpy(np.int64)
    rolled = np.roll(y, 1)
    for config_id in CONFIG_IDS:
        for seed in SEEDS:
            _write_run(root, config_id, seed, frame,
                       y_override=rolled if config_id == "full" else None)
        _write_summary(root, config_id, {7: 1.5, 13: 1.5})
    with pytest.raises(st.RefusalError, match="label vector disagrees"):
        _run_stats(config_path, root, frame_dir, block_source, tmp_path)


def test_alignment_refuses_bad_probabilities(frame_dir, config_path):
    frame = st.load_frame(frame_dir, st.read_yaml(config_path))
    good = np.full((frame.n_rows, 6), 1 / 6, dtype=np.float32)
    st.assert_alignment("ok", good, frame.y, frame.innings_id, frame)
    with pytest.raises(st.RefusalError, match="do not sum to 1"):
        st.assert_alignment("bad", good * 0.5, frame.y, frame.innings_id,
                            frame)
    broken = good.copy()
    broken[0, 0] = np.nan
    with pytest.raises(st.RefusalError, match="finite"):
        st.assert_alignment("bad", broken, frame.y, frame.innings_id, frame)
    with pytest.raises(st.RefusalError, match="expected"):
        st.assert_alignment("bad", good[:, :5], frame.y, frame.innings_id,
                            frame)


def test_reconstructed_ll_is_labelled_and_never_replaces_summary(
        config_path, runs_root, frame_dir, block_source, tmp_path):
    payload = _run_stats(config_path, runs_root, frame_dir, block_source,
                         tmp_path)
    row = payload["runs"]["full"]["seeds"][0]
    assert row["reconstructed_validation_ll"] is not None
    assert row["summary_yaml_validation_ll"] == pytest.approx(1.5 - 0.09)
    assert row["reconstructed_validation_ll"] != row[
        "summary_yaml_validation_ll"]
    assert "reconstructed" in row["reconstructed_validation_ll_label"]
    assert row["summary_minus_reconstructed"] is not None


def test_every_delta_is_candidate_minus_reference(config_path, runs_root,
                                                  frame_dir, block_source,
                                                  tmp_path):
    payload = _run_stats(config_path, runs_root, frame_dir, block_source,
                         tmp_path)
    # `full` has the larger bias, so it has the lower log loss and
    # full - mlp must be negative under the registered sign convention.
    record = payload["contrasts"]["full-mlp@all"]
    assert record["available"]
    assert record["estimand_ii"]["point"] < 0
    reverse = payload["contrasts"]["fox-fixed_decay@all"]
    assert reverse["estimand_ii"]["point"] < 0


# ---------------------------------------------------------------------------
# D10.2 blocks
# ---------------------------------------------------------------------------

def test_block_totals_are_asserted(frame_dir, block_source, config_path):
    frame = st.load_frame(frame_dir, st.read_yaml(config_path))
    blocks = st.resolve_blocks(frame, block_source, **EXPECT_KW)
    assert blocks.n_blocks == 2 and blocks.unmapped == 0
    assert blocks.contract["bootstrap_contract_version"] == (
        "tournament_time_block_v1")
    assert blocks.contract["max_event_gap_days"] == 120
    assert blocks.contract["resampled_unit_passed_to_estimator"] == "block ids"
    with pytest.raises(st.RefusalError, match="registered"):
        st.resolve_blocks(frame, block_source,
                          **{**EXPECT_KW, "expect_blocks": 47})


def test_missing_block_mapping_is_refused_not_invented(frame_dir, tmp_path,
                                                       config_path,
                                                       block_source):
    frame = st.load_frame(frame_dir, st.read_yaml(config_path))
    thin = tmp_path / "thin_corpus"
    thin.mkdir()
    for path in sorted(block_source.glob("*.json"))[:2]:
        (thin / path.name).write_text(path.read_text())
    with pytest.raises(st.RefusalError, match="no tournament block"):
        st.resolve_blocks(frame, thin, **EXPECT_KW)


def test_fewer_than_ten_blocks_is_descriptive_and_cannot_pass(
        config_path, runs_root, frame_dir, block_source, tmp_path):
    payload = _run_stats(config_path, runs_root, frame_dir, block_source,
                         tmp_path)
    assert payload["blocks"]["n_blocks"] < st.MIN_BLOCKS
    for name, block in payload["slices"]["stats"].items():
        if block["available"]:
            assert block["descriptive"] is True, name
    record = payload["contrasts"]["full-mlp@all"]
    assert record["estimand_ii"]["descriptive_only"] is True
    for family in payload["families"]:
        for readout in st.READOUTS:
            assert family["screen"][readout]["status"] == "NOT_EVALUABLE"
            for member in family["holm"][readout]["members"]:
                assert member["status"] == "NOT_EVALUABLE"
                assert member["rejected"] is False


def test_block_ids_not_match_ids_reach_the_estimator(frame_dir, block_source,
                                                     config_path):
    frame = st.load_frame(frame_dir, st.read_yaml(config_path))
    blocks = st.resolve_blocks(frame, block_source, **EXPECT_KW)
    assert len(set(blocks.block_id.tolist())) == 2
    assert len(set(frame.match_id.tolist())) == N_MATCHES
    values = np.ones(frame.n_rows)
    # With two blocks the draws take only a few distinct values; with match
    # ids there would be many more. This pins that blocks are the unit.
    draws = st.bootstrap_draws(values, blocks.block_id, reps=32, seed=29)
    assert np.allclose(draws, 1.0)


# ---------------------------------------------------------------------------
# D10.3 estimands and the joint-resampling equivalence
# ---------------------------------------------------------------------------

def test_single_seed_draws_reproduce_match_cluster_ci():
    rng = np.random.default_rng(11)
    values = rng.normal(size=600)
    clusters = rng.integers(0, 17, size=600)
    draws = st.bootstrap_draws(values, clusters, reps=400, seed=29)
    got = [float(np.percentile(draws, 2.5)),
           float(np.percentile(draws, 97.5))]
    want = match_cluster_ci(values, clusters, 400, 29)
    assert got == pytest.approx(want, abs=1e-9)


def test_joint_draws_reproduce_seed_mean_match_cluster_ci():
    rng = np.random.default_rng(12)
    values = rng.normal(size=(2, 600))
    clusters = rng.integers(0, 19, size=600)
    draws = st.joint_seed_block_draws(list(values), clusters, reps=400,
                                      seed=29)
    got = [float(np.percentile(draws, 2.5)),
           float(np.percentile(draws, 97.5))]
    want = seed_mean_match_cluster_ci(values, clusters, 400, 29)
    assert got == pytest.approx(want, abs=1e-9)
    assert max(abs(a - b) for a, b in zip(got, want)) < 1e-9


def test_joint_estimand_is_not_the_ll_of_averaged_probabilities(
        config_path, runs_root, frame_dir, block_source, tmp_path):
    payload = _run_stats(config_path, runs_root, frame_dir, block_source,
                         tmp_path)
    record = payload["contrasts"]["full-mlp@all"]
    assert record["estimand_ii"][
        "is_log_loss_of_averaged_probabilities"] is False
    per_seed = record["per_seed_points"]
    assert record["estimand_ii"]["point"] == pytest.approx(
        (per_seed["7"] + per_seed["13"]) / 2)
    assert record["estimand_ii"]["label"].startswith("descriptive two-seed")
    assert set(record["estimand_i"]) == {"seed_7", "seed_13"}
    assert record["seed_spread"]["range"] >= 0


def test_joint_draws_need_two_seeds():
    with pytest.raises(st.RefusalError, match="at least two seeds"):
        st.joint_seed_block_draws([np.ones(4)], np.array([0, 0, 1, 1]),
                                  reps=8, seed=29)


# ---------------------------------------------------------------------------
# D10.4 families and Holm
# ---------------------------------------------------------------------------

def test_holm_arithmetic_on_hand_computed_p_values():
    # Three members, m = 3 so the multipliers are 3, 2, 1.
    assert st.holm_adjust([0.01, 0.02, 0.03]) == pytest.approx(
        [0.03, 0.04, 0.04])
    assert st.holm_step_down([0.01, 0.02, 0.03]) == [True, True, True]
    # Monotone: a small p behind a large one cannot come out smaller.
    assert st.holm_adjust([0.02, 0.021, 0.022]) == pytest.approx(
        [0.06, 0.06, 0.06])
    assert st.holm_step_down([0.02, 0.021, 0.022]) == [False, False, False]
    # Step-down stopping: rank 1 fails so nothing later rejects.
    assert st.holm_step_down([0.30, 0.001, 0.001]) == [False, True, True]
    assert st.holm_step_down([0.30, 0.40, 0.50]) == [False, False, False]
    # Stable ties keep the registered member order.
    assert st.holm_order([0.10, 0.10, 0.05]) == [2, 0, 1]
    assert st.holm_rank([0.10, 0.10, 0.05]) == [2, 3, 1]
    assert st.holm_level(1, 3) == pytest.approx(1 - 0.05 / 3)
    assert st.holm_level(3, 3) == pytest.approx(0.95)


def test_holm_placeholder_for_an_unavailable_member_cannot_pass(
        tmp_path, frame_dir, block_source, config_path):
    """An absent run cannot shrink a family or let it pass (D10.4)."""
    root = tmp_path / "runs"
    frame = pd.read_parquet(frame_dir / "cricket_data_i7_validation.parquet")
    for config_id in CONFIG_IDS:
        seeds = SEEDS if config_id != "fixed_decay" else (7,)
        for seed in seeds:
            _write_run(root, config_id, seed, frame, bias=0.5)
        _write_summary(root, config_id, {seed: 1.5 for seed in seeds})
    payload = _run_stats(config_path, root, frame_dir, block_source, tmp_path)
    assert payload["runs"]["fixed_decay"]["complete_paired_seeds"] is False
    by_candidate = {family["candidate"]: family
                    for family in payload["families"]}
    for candidate in ("fixed_decay", "fox"):
        table = by_candidate[candidate]["holm"]["seed_mean_joint"]
        assert len(table["members"]) == st.FAMILY_SIZE
        assert any(member["placeholder_non_rejecting"]
                   for member in table["members"])
        assert all(member["p_raw"] == 1.0 for member in table["members"]
                   if member["placeholder_non_rejecting"])
        assert all(member["p_holm"] == 1.0 for member in table["members"]
                   if member["placeholder_non_rejecting"])
        assert not any(member["rejected"] for member in table["members"])
        assert by_candidate[candidate]["screen"][
            "seed_mean_joint"]["status"] == "NOT_EVALUABLE"
    assert "fixed_decay-mlp@all" in payload["not_evaluable"]


def test_threshold_centred_p_and_the_resolution_floor():
    draws = np.linspace(-1.0, 1.0, 2000)
    zero = st.threshold_centred_p(draws, 0.0, reps=2000)
    assert zero["p_raw"] == pytest.approx(1.0, abs=1e-3)
    assert zero["p_resolution_floor"] is False
    # Every draw strictly below the threshold: one tail is empty, so the
    # p-value is the resolution bound with a flag, never exact zero.
    empty = st.threshold_centred_p(np.full(2000, -1.0), 0.0, reps=2000)
    assert empty["p_resolution_floor"] is True
    assert empty["p_raw"] == pytest.approx(0.001)
    assert empty["p_display"] == "< 0.001"
    assert empty["p_raw"] > 0.0
    # The non-inferiority threshold shifts the centring.
    shifted = st.threshold_centred_p(np.full(2000, 0.001), 0.002, reps=2000)
    assert shifted["threshold"] == 0.002
    assert shifted["p_resolution_floor"] is True


def test_fifteen_families_are_asserted():
    config = _config_payload()
    families = st.registered_families(config, expected=3)
    assert len(families) == 3  # the synthetic config registers three
    assert all(len(family["members"]) == st.FAMILY_SIZE
               for family in families)
    assert [member["member"] for member in families[0]["members"]] == list(
        st.FAMILY_MEMBER_ORDER)
    assert [member["threshold"] for member in families[0]["members"]] == [
        0.0, st.MARGIN_LL, st.MARGIN_LL]

    real = st.read_yaml(REPO / "experiments" / "configs"
                        / "seq_stage2_v1.yaml")
    real_families = st.registered_families(real)  # default 15
    assert len(real_families) == st.N_FAMILIES
    assert all(len(family["members"]) == st.FAMILY_SIZE
               for family in real_families)
    primaries = {family["candidate"]: family["members"][0]["reference"]
                 for family in real_families}
    assert primaries["fox"] == "fixed_decay"
    assert primaries["aligned_hist"] == "full"
    assert primaries["same_entity_k30"] == "recency_k30"
    assert primaries["same_entity_unr"] == "aligned_hist_rf"
    assert primaries["residual_t1"] == "residual_mlp"
    assert "mlp" not in primaries


def test_family_map_defects_are_refused():
    config = _config_payload()
    del config["statistics"]["families"]["map"][0]["death_gate"]
    with pytest.raises(st.RefusalError, match="death_gate"):
        st.registered_families(config, expected=3)

    config = _config_payload()
    config["statistics"]["families"]["map"][0]["primary"][
        "reference"] = "not_a_config"
    with pytest.raises(st.RefusalError, match="not a registered"):
        st.registered_families(config, expected=3)

    config = _config_payload()
    config["statistics"]["families"]["map"][0]["primary"]["slice"] = "death"
    with pytest.raises(st.RefusalError, match="registered on slice"):
        st.registered_families(config, expected=3)

    config = _config_payload()
    config["statistics"]["families"]["map"].append(_family("mlp", "full"))
    with pytest.raises(st.RefusalError):
        st.registered_families(config, expected=3)


def test_real_config_family_count_is_checked_against_fifteen():
    real = st.read_yaml(REPO / "experiments" / "configs"
                        / "seq_stage2_v1.yaml")
    real["statistics"]["families"]["map"] = real["statistics"][
        "families"]["map"][:14]
    with pytest.raises(st.RefusalError, match="not 15"):
        st.registered_families(real)


# ---------------------------------------------------------------------------
# D10.5 mechanism contrasts
# ---------------------------------------------------------------------------

def test_registered_mechanism_contrasts_are_exactly_the_five():
    assert st.MECHANISM_CONTRASTS == (
        ("fox", "fixed_decay", "learned forgetting beyond fixed decay"),
        ("same_entity_k30", "recency_k30",
         "ownership plus alignment beyond recency"),
        ("same_entity_unr", "aligned_hist_rf",
         "mask alone, given aligned inputs and matched relay-free keys"),
        ("aligned_hist_rf", "aligned_hist",
         "the relay-free wiring PLUS the key construction, not the wiring "
         "alone"),
        ("aligned_hist", "full", "the aligned history input"),
    )
    assert "not reportable" in st.MECHANISM_GUARD


def test_mechanism_contrasts_carry_their_labels_and_guard(
        config_path, runs_root, frame_dir, block_source, tmp_path):
    payload = _run_stats(config_path, runs_root, frame_dir, block_source,
                         tmp_path)
    entries = {f"{e['candidate']}-{e['reference']}": e
               for e in payload["mechanism_contrasts"]}
    assert len(entries) == len(st.MECHANISM_CONTRASTS)
    fox = entries["fox-fixed_decay"]
    assert fox["registered_label"] == (
        "learned forgetting beyond fixed decay")
    assert fox["inferential_in_a_registered_family"] is True
    assert fox["record"]["available"] is True
    missing = entries["same_entity_k30-recency_k30"]
    assert missing["record"]["available"] is False
    assert "incomplete paired seeds" in missing["record"]["reason"]


# ---------------------------------------------------------------------------
# D10.6 gates
# ---------------------------------------------------------------------------

def _gate_readout(u95: float, point: float = 0.0) -> dict:
    return {"point": point, "ci95": [point - 0.001, u95], "u95": u95,
            "l95": point - 0.001, "descriptive_only": False,
            "p_raw": 0.001, "p_display": "0.0010",
            "p_resolution_floor": False, "ci_clean_favourable": u95 < 0,
            "u95_below_zero": u95 < 0, "u95_below_margin": u95 < st.MARGIN_LL,
            "margin_ll": st.MARGIN_LL}


def test_u95_strictness_at_the_margin_boundary():
    member = {"kind": "non_inferiority", "threshold": st.MARGIN_LL}
    assert st.member_status(member, _gate_readout(0.0019), True) == (
        "SCREEN_PASS")
    # Exactly at the margin does NOT pass: the rule is strictly below.
    assert st.member_status(member, _gate_readout(0.002), True) == (
        "SCREEN_NOT_PASS")
    assert st.member_status(member, _gate_readout(0.0021), True) == (
        "SCREEN_NOT_PASS")
    # A numerical pass without Holm rejection is still not a pass.
    assert st.member_status(member, _gate_readout(0.0019), False) == (
        "SCREEN_NOT_PASS")
    # Descriptive slices and absent readouts are NOT_EVALUABLE.
    descriptive = _gate_readout(0.0019)
    descriptive["descriptive_only"] = True
    assert st.member_status(member, descriptive, True) == "NOT_EVALUABLE"
    assert st.member_status(member, None, True) == "NOT_EVALUABLE"


def test_primary_strictness_is_against_zero():
    member = {"kind": "superiority", "threshold": 0.0}
    assert st.member_status(member, _gate_readout(-1e-9), True) == (
        "SCREEN_PASS")
    assert st.member_status(member, _gate_readout(0.0), True) == (
        "SCREEN_NOT_PASS")
    assert st.member_status(member, _gate_readout(0.0019), True) == (
        "SCREEN_NOT_PASS")


def test_gate_rows_expose_every_registered_field(config_path, runs_root,
                                                 frame_dir, block_source,
                                                 tmp_path):
    payload = _run_stats(config_path, runs_root, frame_dir, block_source,
                         tmp_path)
    assert payload["gates"]
    for gate in payload["gates"]:
        assert gate["slice"] in ("death", "chase")
        assert gate["reference"] == "mlp"
        assert gate["margin_ll"] == st.MARGIN_LL
        assert gate["numerical_pass_rule"] == "strictly U95 < 0.002"
        for readout in st.READOUTS:
            row = gate["readouts"][readout]
            assert row["status"] in st.ALLOWED_STATUSES
            assert set(row) >= {"point", "ci95", "u95", "p_raw", "p_holm",
                                "u95_strictly_below_margin", "status"}


def test_statuses_are_only_the_three_allowed(config_path, runs_root, frame_dir,
                                             block_source, tmp_path):
    payload = _run_stats(config_path, runs_root, frame_dir, block_source,
                         tmp_path)
    statuses = set()
    for family in payload["families"]:
        for readout in st.READOUTS:
            statuses.add(family["screen"][readout]["status"])
            statuses.update(member["status"]
                            for member in family["holm"][readout]["members"])
    for gate in payload["gates"]:
        statuses.update(row["status"] for row in gate["readouts"].values())
    assert statuses <= set(st.ALLOWED_STATUSES)
    assert not any("advance" in status.lower() for status in statuses)
    for family in payload["families"]:
        for readout in st.READOUTS:
            assert family["screen"][readout]["status"] in st.ALLOWED_STATUSES


# ---------------------------------------------------------------------------
# D10.16 — five-seed readouts and the 4/5 extension qualification
# (Astra gate 2 round 2)
# ---------------------------------------------------------------------------

FIVE_SEEDS = (7, 13, 29, 42, 101)


def test_readouts_are_derived_from_the_seeds_not_hard_coded():
    assert st.readouts_for((7, 13)) == ("seed_7", "seed_13", "seed_mean_joint")
    assert st.READOUTS == st.readouts_for(st.REGISTERED_SEEDS)
    assert st.readouts_for(FIVE_SEEDS) == (
        "seed_7", "seed_13", "seed_29", "seed_42", "seed_101",
        "seed_mean_joint")
    # One estimand (i) readout per seed, and exactly one joint readout.
    for seeds in ((7,), (7, 13), FIVE_SEEDS):
        readouts = st.readouts_for(seeds)
        assert len(readouts) == len(seeds) + 1
        assert readouts[-1] == st.JOINT_READOUT
        assert readouts.count(st.JOINT_READOUT) == 1


def test_five_seed_statistics_carry_five_readouts_in_the_contract(
        config_path, runs_root, frame_dir, block_source, tmp_path):
    """Holm tables, gates and screens must not omit seeds 29, 42 and 101."""
    frame = pd.read_parquet(frame_dir / "cricket_data_i7_validation.parquet")
    biases = {"mlp": 0.0, "full": 0.9, "fixed_decay": 0.5, "fox": 0.7}
    for config_id, bias in biases.items():
        for seed in FIVE_SEEDS:
            _write_run(runs_root, config_id, seed, frame, bias=bias)
        _write_summary(runs_root, config_id,
                       {seed: 1.5 - bias / 10 for seed in FIVE_SEEDS})
    payload = _run_stats(config_path, runs_root, frame_dir, block_source,
                         tmp_path, seeds=FIVE_SEEDS)
    expected = list(st.readouts_for(FIVE_SEEDS))
    assert payload["contract"]["readouts"] == expected
    assert payload["contract"]["seeds"] == list(FIVE_SEEDS)
    assert payload["contract"]["n_seeds"] == 5
    assert payload["contract"][
        "five_seed_direction_requirement_applies"] is True
    assert "five-seed" not in payload["contract"]["estimand_ii_label"] or (
        "descriptive five-seed" in payload["contract"]["estimand_ii_label"])
    for family in payload["families"]:
        assert list(family["holm"]) == expected
        assert list(family["screen"]) == expected
    for gate in payload["gates"]:
        assert list(gate["readouts"]) == expected
    record = payload["contrasts"]["full-mlp@all"]
    assert sorted(record["estimand_i"]) == sorted(
        f"seed_{s}" for s in FIVE_SEEDS)
    assert record["n_seeds"] == 5
    # And the reported summary names the enforced requirement.
    text = "\n".join(st._summary_lines(payload))
    assert "seed_29" in text and "seed_101" in text
    assert "4/5 favourable-direction count REQUIRED" in text


def _screen_inputs(direction_count: int | None, n_seeds: int,
                   pass_everything: bool = True) -> tuple[dict, dict, dict]:
    """A family, its Holm tables and its contrasts, all passing but for 4/5."""
    family = {"candidate": "full",
              "members": [{"member": "primary", "candidate": "full",
                           "reference": "mlp", "slice": "all"},
                          {"member": "death_gate", "candidate": "full",
                           "reference": "mlp", "slice": "death"},
                          {"member": "chase_gate", "candidate": "full",
                           "reference": "mlp", "slice": "chase"}]}
    status = "SCREEN_PASS" if pass_everything else "SCREEN_NOT_PASS"
    readout = st.JOINT_READOUT
    tables = {readout: {"readout": readout, "members": [
        {"member": "primary", "status": status, "rejected": pass_everything,
         "u95": -0.01, "contrast_key": "full-mlp@all"},
        {"member": "death_gate", "status": status, "rejected": pass_everything,
         "u95": 0.0005, "contrast_key": "full-mlp@death"},
        {"member": "chase_gate", "status": status, "rejected": pass_everything,
         "u95": 0.0005, "contrast_key": "full-mlp@chase"}]}}
    primary = {"available": True, "n_seeds": n_seeds,
               "favourable_direction_count": direction_count,
               "estimand_ii": _gate_readout(-0.01, point=-0.02)}
    contrasts = {"full-mlp@all": primary}
    return family, tables, contrasts


def test_the_four_of_five_direction_count_is_enforced_at_five_seeds():
    """Astra gate 2 round 2: the count was calculated and never enforced."""
    for count, expected in ((5, "SCREEN_PASS"), (4, "SCREEN_PASS"),
                            (3, "SCREEN_NOT_PASS"), (0, "SCREEN_NOT_PASS")):
        family, tables, contrasts = _screen_inputs(count, 5)
        screen = st.family_screen(family, tables, contrasts, st.JOINT_READOUT)
        assert screen["status"] == expected, count
        assert screen["five_seed_direction_requirement_applies"] is True
        assert screen["required_favourable_directions"] == 4
        assert screen["favourable_direction_count"] == count
        assert screen["favourable_direction_requirement_met"] is (count >= 4)
        assert screen["five_seed_extension_qualified"] is (
            expected == "SCREEN_PASS")
        assert screen["five_seed_eligibility_rule"] == (
            st.FIVE_SEED_ELIGIBILITY_RULE)
        assert "of 5 per-seed primary directions" in screen[
            "five_seed_qualification_note"]


def test_a_missing_direction_count_at_five_seeds_cannot_qualify():
    family, tables, contrasts = _screen_inputs(None, 5)
    screen = st.family_screen(family, tables, contrasts, st.JOINT_READOUT)
    assert screen["status"] == "SCREEN_NOT_PASS"
    assert screen["favourable_direction_requirement_met"] is False


def test_a_two_seed_run_is_not_retroactively_failed_by_the_new_rule():
    """At two seeds the count cannot reach 4/5, so it must not apply."""
    for count in (0, 1, 2):
        family, tables, contrasts = _screen_inputs(count, 2)
        screen = st.family_screen(family, tables, contrasts, st.JOINT_READOUT)
        assert screen["status"] == "SCREEN_PASS", count
        assert screen["five_seed_direction_requirement_applies"] is False
        assert screen["required_favourable_directions"] is None
        assert screen["favourable_direction_requirement_met"] is None
        # A two-seed PASS is explicitly NOT the extension qualification.
        assert screen["five_seed_extension_qualified"] is False
        note = screen["five_seed_qualification_note"]
        assert "not applicable at 2 seed(s)" in note
        assert "NOT the five-seed extension qualification" in note
    # And a failure elsewhere still fails at two seeds.
    family, tables, contrasts = _screen_inputs(2, 2, pass_everything=False)
    assert st.family_screen(family, tables, contrasts,
                            st.JOINT_READOUT)["status"] == "SCREEN_NOT_PASS"


def test_a_two_seed_run_reports_that_the_requirement_does_not_apply(
        config_path, runs_root, frame_dir, block_source, tmp_path):
    payload = _run_stats(config_path, runs_root, frame_dir, block_source,
                         tmp_path)
    assert payload["contract"]["n_seeds"] == 2
    assert payload["contract"][
        "five_seed_direction_requirement_applies"] is False
    assert payload["contract"]["readouts"] == list(st.readouts_for(SEEDS))
    assert payload["contract"]["estimand_ii_label"] == st.ESTIMAND_II_LABEL
    text = "\n".join(st._summary_lines(payload))
    assert "does NOT apply at 2 seeds and fails nothing here" in text


def test_the_registered_five_seed_rule_is_stated_in_full():
    rule = st.FIVE_SEED_ELIGIBILITY_RULE
    assert "4 of 5 favourable per-seed directions" in rule
    assert "CI-clean under Holm" in rule
    assert "strictly U95 < 0.002" in rule
    assert "at least 10 blocks" in rule
    assert "complete paired seeds" in rule
    assert "no two-seed result is retroactively failed" in rule
    assert (st.FIVE_SEED_MINIMUM, st.FIVE_SEED_FAVOURABLE_DIRECTIONS) == (5, 4)


# ---------------------------------------------------------------------------
# the expected family count is derived from the config, not a constant
# ---------------------------------------------------------------------------

def test_expected_family_count_is_derived_from_the_config():
    seven = _config_payload(config_ids=("mlp", "full", "fixed_decay", "fox",
                                        "aligned_hist", "xlstm",
                                        "residual_mlp", "residual_t1"))
    assert st.expected_family_count(seven) == 7
    assert st.expected_family_count(
        _config_payload(config_ids=CONFIG_IDS)) == 3
    real = st.read_yaml(REPO / "experiments" / "configs"
                        / "seq_stage2_v1.yaml")
    assert st.expected_family_count(real) == st.N_FAMILIES == 15
    five_seed = st.read_yaml(REPO / "experiments" / "configs"
                             / "seq_stage2_5seed_v1.yaml")
    assert st.expected_family_count(five_seed) == 15


def test_registered_families_derives_the_count_when_none_is_given():
    """A seven-family and a fifteen-family config must both run with no flag."""
    payload = _config_payload(config_ids=CONFIG_IDS)
    assert len(st.registered_families(payload)) == 3
    real = st.read_yaml(REPO / "experiments" / "configs"
                        / "seq_stage2_v1.yaml")
    assert len(st.registered_families(real)) == 15
    five_seed = st.read_yaml(REPO / "experiments" / "configs"
                             / "seq_stage2_5seed_v1.yaml")
    assert len(st.registered_families(five_seed)) == 15
    # An explicit count still asserts.
    with pytest.raises(st.RefusalError, match="registers 3 families, not 15"):
        st.registered_families(payload, 15)
    # And a family map that does not cover the configurations still refuses.
    short = _config_payload(config_ids=CONFIG_IDS)
    short["statistics"]["families"]["map"] = short["statistics"][
        "families"]["map"][:-1]
    with pytest.raises(st.RefusalError, match="registers 2 families, not 3"):
        st.registered_families(short)


# ---------------------------------------------------------------------------
# night 1's evidence is never overwritten by a five-seed run
# ---------------------------------------------------------------------------

def test_the_five_seed_config_is_the_two_seed_one_with_only_the_seeds_changed():
    """Astra gate 2 round 2 MUST-FIX D.

    The earlier revision held eight configurations, so `ksweep` refused (no
    window arm survived) and families could lose the references their
    contrasts are defined against. The rebuild is `seq_stage2_v1.yaml` with
    only the seed list changed.
    """
    two = st.read_yaml(REPO / "experiments" / "configs"
                       / "seq_stage2_v1.yaml")
    five = st.read_yaml(REPO / "experiments" / "configs"
                        / "seq_stage2_5seed_v1.yaml")
    assert five["training"]["seeds"] == [7, 13, 29, 42, 101]
    assert two["training"]["seeds"] == [7, 13]

    # All sixteen configurations, contiguous orders 1..16, all fifteen
    # families -- including every gate reference and every window arm.
    assert len(five["configurations"]) == 16
    assert [e["queue_order"] for e in five["configurations"]] == list(
        range(1, 17))
    ids = [str(e["id"]) for e in five["configurations"]]
    assert ids == [str(e["id"]) for e in two["configurations"]]
    assert "mlp" in ids and "aligned_hist" in ids
    for k, config_id in K_IDS.items():
        assert config_id in ids, config_id
    assert "recency_k30" in ids and "aligned_hist_rf" in ids and "lstm" in ids
    assert len(five["statistics"]["families"]["map"]) == st.N_FAMILIES
    assert len(st.registered_families(five)) == st.N_FAMILIES

    # Nothing else differs: training settings, slices, families, deviations,
    # limitations and the k sweep are preserved verbatim.
    for key in ("data", "outputs", "configurations", "statistics", "cohort",
                "deviations", "known_asymmetries", "known_limitations",
                "forbidden_data", "queue_order_note"):
        assert five[key] == two[key], key
    training_five = {k: v for k, v in five["training"].items() if k != "seeds"}
    training_two = {k: v for k, v in two["training"].items() if k != "seeds"}
    assert training_five == training_two
    assert five["experiment"] == two["experiment"]

    # The k sweep can actually run: every registered k arm is present.
    sweep = five["statistics"]["k_selection"]["registered_sweep"]
    assert [str(k) for k in sweep] == list(st.REGISTERED_K_ORDER)

    # And the file says, in its own text, what it is and is not for.
    text = (REPO / "experiments" / "configs"
            / "seq_stage2_5seed_v1.yaml").read_text()
    assert "CONSOLIDATION AND ANALYSIS ONLY -- NEVER" in text
    assert "TRAINING" in text
    assert "SINGLE COMPLETE five-seed analysis registration" in text
    assert "4 of 5 favourable per-seed directions" in text
    assert "distinct from the two-seed ones" in text


def test_a_non_default_config_must_name_its_own_out_path(tmp_path):
    five_seed = REPO / "experiments" / "configs" / "seq_stage2_5seed_v1.yaml"
    with pytest.raises(st.RefusalError, match="Name a distinct path"):
        st.require_distinct_out(five_seed, st.DEFAULT_STATS_OUT,
                                st.DEFAULT_STATS_OUT)
    # An explicitly named path is accepted.
    mine = tmp_path / "five_seed_stats.json"
    assert st.require_distinct_out(five_seed, mine,
                                   st.DEFAULT_STATS_OUT) == mine
    # And the registered two-seed config keeps its default.
    assert st.require_distinct_out(st.DEFAULT_CONFIG, st.DEFAULT_STATS_OUT,
                                   st.DEFAULT_STATS_OUT) == (
        st.DEFAULT_STATS_OUT)


def test_the_cli_refuses_a_five_seed_run_at_the_two_seed_default(capsys):
    five_seed = REPO / "experiments" / "configs" / "seq_stage2_5seed_v1.yaml"
    for command, default in (("stats", st.DEFAULT_STATS_OUT),
                             ("ksweep", st.DEFAULT_KSWEEP_OUT)):
        assert st.main([command, "--config", str(five_seed),
                        "--out", str(default)]) == 2
        error = capsys.readouterr().err
        assert "REFUSED" in error
        assert "Name a distinct path explicitly" in error


# ---------------------------------------------------------------------------
# D10.7 slices
# ---------------------------------------------------------------------------

def test_slice_predicates_are_frozen_and_checked_against_the_frame(
        frame_dir, config_path):
    frame = st.load_frame(frame_dir, st.read_yaml(config_path))
    names = {predicate["slice"] for predicate in frame.predicates}
    assert names == {"all", "death", "chase", "powerplay", "middle",
                     "innings_1", "innings_2", "thin_pair"}
    by_name = {p["slice"]: p for p in frame.predicates}
    assert by_name["death"]["predicate"] == "is_death_overs == 1"
    assert by_name["chase"]["predicate"] == "chase_target > 0"
    assert by_name["all"]["n_rows"] == frame.n_rows
    assert by_name["death"]["n_rows"] == N_MATCHES * 2 * 3
    assert by_name["chase"]["n_rows"] == N_MATCHES * BALLS_PER_INNINGS
    assert frame.masks["innings_1"].sum() == N_MATCHES * BALLS_PER_INNINGS


def test_slice_stats_persist_a_row_mask_digest(frame_dir, config_path,
                                              block_source):
    """Astra gate 2 round 1 SHOULD 7.

    Slice identity must be comparable by row membership, so every slice
    persists the sha256 of its packed row mask. Two slices with identical row,
    match and block counts but different rows get different digests.
    """
    import numpy as np

    config = st.read_yaml(config_path)
    frame = st.load_frame(frame_dir, config)
    blocks = st.resolve_blocks(frame, block_source, **EXPECT_KW)
    digests = {name: st.slice_stats(frame, blocks, name)["mask_sha256"]
               for name in ("all", "death", "chase", "innings_1", "innings_2")}
    assert all(isinstance(value, str) and len(value) == 64
               for value in digests.values())
    assert digests["innings_2"] == st.slice_stats(frame, blocks,
                                                 "chase")["mask_sha256"]
    assert digests["death"] != digests["chase"]
    # Same counts, different membership: the digest separates them.
    left = np.zeros(8, dtype=bool)
    right = np.zeros(8, dtype=bool)
    left[:3] = True
    right[-3:] = True
    assert left.sum() == right.sum()
    assert st.mask_digest(left) != st.mask_digest(right)
    assert st.mask_digest(left) == st.mask_digest(left.copy())
    # Length is part of the digest, so a padded mask is not the same mask.
    assert st.mask_digest(left) != st.mask_digest(np.concatenate(
        [left, np.zeros(8, dtype=bool)]))


def test_thin_pair_is_reported_unavailable_not_invented(frame_dir,
                                                        config_path):
    frame = st.load_frame(frame_dir, st.read_yaml(config_path))
    thin = next(p for p in frame.predicates if p["slice"] == "thin_pair")
    assert thin["available"] is False
    assert "exposure" in thin["unavailable_reason"]
    assert "thin_pair" not in frame.masks


def test_thin_pair_turns_on_only_from_a_registered_spec(tmp_path, frame_dir):
    config = _config_payload(frame_dir=frame_dir)
    config["statistics"]["slice_predicates"] = {
        "thin_pair": {"exposure_columns": ["chase_target"],
                      "threshold": 1.0}}
    frame = st.load_frame(frame_dir, config)
    thin = next(p for p in frame.predicates if p["slice"] == "thin_pair")
    assert thin["available"] is True
    assert thin["threshold"] == 1.0
    assert frame.masks["thin_pair"].sum() == N_MATCHES * BALLS_PER_INNINGS

    config["statistics"]["slice_predicates"]["thin_pair"][
        "exposure_columns"] = ["no_such_column"]
    frame = st.load_frame(frame_dir, config)
    thin = next(p for p in frame.predicates if p["slice"] == "thin_pair")
    assert thin["available"] is False


def test_missing_predicate_column_is_refused(tmp_path):
    directory = tmp_path / "bad_frame"
    directory.mkdir()
    df = _frame_rows().drop(columns=["is_death_overs"])
    df.to_parquet(directory / "cricket_data_i7_validation.parquet",
                  index=False)
    config = _config_payload(frame_dir=directory)
    with pytest.raises(st.RefusalError, match="predicate column"):
        st.load_frame(directory, config)


def test_exploratory_slices_are_reported(config_path, runs_root, frame_dir,
                                         block_source, tmp_path):
    payload = _run_stats(config_path, runs_root, frame_dir, block_source,
                         tmp_path)
    exploratory = {key for key, record in payload["contrasts"].items()
                   if record["role"] == "exploratory"}
    assert any("@powerplay" in key for key in exploratory)
    assert any("@innings_2" in key for key in exploratory)
    assert payload["slices"]["frozen_before_computation"] is True
    assert "opens no test rows" in payload["slices"]["exploratory_note"]


# ---------------------------------------------------------------------------
# D10.11 residual
# ---------------------------------------------------------------------------

def test_residual_block_reports_base_only_absence_cleanly(
        config_path, runs_root, frame_dir, block_source, tmp_path):
    payload = _run_stats(config_path, runs_root, frame_dir, block_source,
                         tmp_path)
    residual = payload["residual"]
    assert residual["registered_primary_key"] == (
        "residual_t1-residual_mlp@all")
    assert "never a qualifying primary" in residual[
        "residual_t1_minus_mlp_cannot_qualify"]
    assert "production-prior control" in residual["residual_mlp_role"]
    assert residual["base_only"]["available"] is False
    assert all(row["available"] is False
               for row in residual["against_base_only"])


def test_base_only_readout_reads_logits_and_provenance(tmp_path, frame_dir,
                                                       block_source,
                                                       config_path):
    frame = st.load_frame(frame_dir, st.read_yaml(config_path))
    blocks = st.resolve_blocks(frame, block_source, **EXPECT_KW)
    npz = tmp_path / "validation.npz"
    logp = np.log(np.full((frame.n_rows, 6), 1 / 6, dtype=np.float32))
    np.savez_compressed(npz, logp=logp, n_rows=frame.n_rows)
    (tmp_path / "validation.json").write_text(json.dumps({
        "n_rows": frame.n_rows, "booster_md5": "7ee1e180",
        "model_dir": "models/xgb_i7_noweights_production",
        "model_role": "ball_model_prod", "parquet_md5": frame.md5,
        "floor": 0.0001, "renormalised": True}))
    pin = st.Pin(available=True, base_logits={"validation": {
        "npz_md5": st.md5_file(npz),
        "sidecar_sha256": st.sha256_file(tmp_path / "validation.json"),
        "parquet_md5": frame.md5, "booster_md5": "7ee1e180",
        "n_rows": frame.n_rows}})
    record = st.base_only_readout(npz, tmp_path / "validation.json", frame,
                                  blocks, pin)
    assert record["available"] is True
    assert record["row_ll_mean"] == pytest.approx(np.log(6))
    assert record["provenance"]["booster_md5"] == "7ee1e180"
    assert "row order" in record["provenance"]["row_alignment"]

    assert record["provenance"]["verified_against_pin"] is True
    assert all(check["agrees"] for check in record["hash_checks"].values())

    (tmp_path / "validation.json").write_text(json.dumps({"n_rows": 7}))
    bad = st.base_only_readout(npz, tmp_path / "validation.json", frame,
                               blocks, pin)
    assert bad["available"] is False and "rows" in bad["reason"]


def test_base_only_readout_rejects_a_stale_or_mislabelled_sidecar(
        tmp_path, frame_dir, block_source, config_path):
    """Provenance is CHECKED against the pin and the live parquet, not copied."""
    frame = st.load_frame(frame_dir, st.read_yaml(config_path))
    blocks = st.resolve_blocks(frame, block_source, **EXPECT_KW)
    npz = tmp_path / "validation.npz"
    logp = np.log(np.full((frame.n_rows, 6), 1 / 6, dtype=np.float32))
    np.savez_compressed(npz, logp=logp, n_rows=frame.n_rows)
    sidecar = tmp_path / "validation.json"

    def write_sidecar(**overrides):
        payload = {"n_rows": frame.n_rows, "booster_md5": "7ee1e180",
                   "parquet_md5": frame.md5, "floor": 0.0001,
                   "renormalised": True}
        payload.update(overrides)
        sidecar.write_text(json.dumps(payload))

    def pin_for(**overrides):
        block = {"npz_md5": st.md5_file(npz),
                 "sidecar_sha256": st.sha256_file(sidecar),
                 "parquet_md5": frame.md5, "booster_md5": "7ee1e180",
                 "n_rows": frame.n_rows}
        block.update(overrides)
        return st.Pin(available=True, base_logits={"validation": block})

    # The sidecar claims a parquet the live frame is not.
    write_sidecar(parquet_md5="0" * 32)
    record = st.base_only_readout(npz, sidecar, frame, blocks, pin_for())
    assert record["available"] is False
    assert "live validation parquet" in record["reason"]

    # A booster md5 the pin does not record.
    write_sidecar(booster_md5="deadbeef")
    record = st.base_only_readout(npz, sidecar, frame, blocks, pin_for())
    assert record["available"] is False
    assert "booster_md5" in record["reason"]

    # An npz whose bytes no longer match the pin.
    write_sidecar()
    record = st.base_only_readout(npz, sidecar, frame, blocks,
                                  pin_for(npz_md5="1" * 32))
    assert record["available"] is False
    assert "npz_md5" in record["reason"]

    # No pin entry at all: nothing can be verified, so nothing is reported.
    write_sidecar()
    record = st.base_only_readout(npz, sidecar, frame, blocks,
                                  st.Pin(available=True))
    assert record["available"] is False
    assert "cannot be verified" in record["reason"]

    # A missing sidecar carries no verifiable provenance.
    pin = pin_for()
    sidecar.unlink()
    record = st.base_only_readout(npz, sidecar, frame, blocks, pin)
    assert record["available"] is False
    assert "no verifiable provenance" in record["reason"]


# ---------------------------------------------------------------------------
# D9 k sweep
# ---------------------------------------------------------------------------

def _k_runs(tmp_path: Path, values: dict[str, dict[int, float | None]],
            frame_dir: Path | None = None, *, skip: tuple[str, ...] = (),
            components_by_k: Mapping[str, dict] | None = None) -> Path:
    """A runs tree for the k sweep, keyed by the REGISTERED config ids."""
    root = tmp_path / "kruns"
    frame = (pd.read_parquet(frame_dir / "cricket_data_i7_validation.parquet")
             if frame_dir is not None else _frame_rows())
    for k, per_seed in values.items():
        config_id = K_IDS[k]
        if config_id in skip:
            continue
        for seed, value in per_seed.items():
            if value is None:
                continue
            _write_run(root, config_id, seed, frame, arm="same_entity",
                       k=(k if k == "unr" else int(k)),
                       components=(components_by_k or {}).get(k))
        _write_summary(root, config_id, per_seed)
    return root


def test_k_rule_keeps_thirty_when_nothing_clears_the_tolerance(tmp_path, frame_dir, k_config):
    root = _k_runs(tmp_path, {
        "0": {7: 1.60, 13: 1.60}, "6": {7: 1.55, 13: 1.55},
        "12": {7: 1.50, 13: 1.50}, "30": {7: 1.50, 13: 1.50},
        "unr": {7: 1.52, 13: 1.52}})
    record = st.k_sweep(root, k_config, seeds=SEEDS)
    assert record["selection"] == "SELECTED"
    assert record["selected_k"] == "30"
    assert record["best_mean_k"] == "12"  # tie broken in registered order
    assert record["margin_vs_k30"] == pytest.approx(0.0)
    assert record["flags"] == []


def test_k_rule_equality_at_exactly_the_tolerance_keeps_thirty(tmp_path, frame_dir, k_config):
    root = _k_runs(tmp_path, {
        "0": {7: 0.60, 13: 0.60}, "6": {7: 0.60, 13: 0.60},
        "12": {7: 0.0, 13: 0.0}, "30": {7: 0.002, 13: 0.002},
        "unr": {7: 0.60, 13: 0.60}})
    record = st.k_sweep(root, k_config, seeds=SEEDS)
    # The two log losses are chosen so that the margin is EXACTLY the float
    # 0.002, which is the boundary the rule must not cross.
    assert record["margin_vs_k30"] == 0.002
    assert record["selected_k"] == "30"


def test_k_rule_selects_a_better_k_past_the_tolerance(tmp_path, frame_dir, k_config):
    root = _k_runs(tmp_path, {
        "0": {7: 1.60, 13: 1.60}, "6": {7: 1.49, 13: 1.49},
        "12": {7: 1.60, 13: 1.60}, "30": {7: 1.50, 13: 1.50},
        "unr": {7: 1.60, 13: 1.60}})
    record = st.k_sweep(root, k_config, seeds=SEEDS)
    assert record["selected_k"] == "6"
    assert record["selected_config_id"] == "same_entity_k6"
    assert "same_entity_k30" in record["not_selected"]
    assert record["paired_vs_k30"]["6"]["favourable_direction_count"] == 2


def test_k_rule_lets_unr_win(tmp_path, frame_dir, k_config):
    root = _k_runs(tmp_path, {
        "0": {7: 1.60, 13: 1.60}, "6": {7: 1.60, 13: 1.60},
        "12": {7: 1.60, 13: 1.60}, "30": {7: 1.50, 13: 1.50},
        "unr": {7: 1.48, 13: 1.48}})
    record = st.k_sweep(root, k_config, seeds=SEEDS)
    assert record["selected_k"] == "unr"
    assert record["selected_config_id"] == "same_entity_unr"


def test_k_rule_flags_a_non_default_selection_with_a_sign_change(tmp_path, frame_dir, k_config):
    root = _k_runs(tmp_path, {
        "0": {7: 1.60, 13: 1.60}, "6": {7: 1.60, 13: 1.60},
        "12": {7: 1.60, 13: 1.60}, "30": {7: 1.50, 13: 1.50},
        "unr": {7: 1.40, 13: 1.55}})
    record = st.k_sweep(root, k_config, seeds=SEEDS)
    assert record["selected_k"] == "unr"
    joined = " ".join(record["flags"])
    assert "change sign" in joined
    assert "tolerance" in joined
    assert "absolute mean difference" in joined


def test_k_sweep_blocks_on_an_incomplete_arm(tmp_path, frame_dir, k_config):
    root = _k_runs(tmp_path, {
        "0": {7: 1.60, 13: 1.60}, "6": {7: 1.55, 13: 1.55},
        "12": {7: 1.50, 13: 1.50}, "30": {7: 1.50, 13: 1.50},
        "unr": {7: 1.40, 13: None}})
    record = st.k_sweep(root, k_config, seeds=SEEDS)
    assert record["selection"] == "BLOCKED_INCOMPLETE"
    assert record["selected_k"] is None
    assert record["blocked_on"] == ["same_entity_unr"]
    assert "default-to-30" in record["note"]


def test_k_sweep_blocks_when_a_summary_is_missing_entirely(tmp_path, frame_dir, k_config):
    root = _k_runs(tmp_path, {
        "0": {7: 1.60, 13: 1.60}, "6": {7: 1.55, 13: 1.55},
        "12": {7: 1.50, 13: 1.50}, "30": {7: 1.50, 13: 1.50}})
    record = st.k_sweep(root, k_config, seeds=SEEDS)
    assert record["selection"] == "BLOCKED_INCOMPLETE"
    assert record["blocked_on"] == ["same_entity_unr"]


def test_k_sweep_records_every_source_and_a_hash(tmp_path, frame_dir, k_config):
    root = _k_runs(tmp_path, {
        "0": {7: 1.60, 13: 1.61}, "6": {7: 1.55, 13: 1.56},
        "12": {7: 1.50, 13: 1.51}, "30": {7: 1.50, 13: 1.52},
        "unr": {7: 1.52, 13: 1.53}})
    record = st.k_sweep(root, k_config, seeds=SEEDS)
    assert len(record["source_lls"]) == 5
    assert sum(len(v) for v in record["source_lls"].values()) == 10
    assert all(source["sha256"] for source in record["sources"])
    assert record["selection_record_sha256"]
    assert record["labelling"] == "two-seed directional screen"
    assert record["provisional"] is True
    assert record["registered_sweep"] == list(st.REGISTERED_K_ORDER)
    assert record["tolerance_ll"] == 0.002


def test_k_sweep_refuses_an_unregistered_seed(tmp_path, frame_dir, k_config):
    root = _k_runs(tmp_path, {
        "0": {7: 1.60, 13: 1.61}, "6": {7: 1.55, 13: 1.56},
        "12": {7: 1.50, 13: 1.51}, "30": {7: 1.50, 13: 1.52},
        "unr": {7: 1.52, 13: 1.53, 42: 1.40}})
    record = st.k_sweep(root, k_config, seeds=SEEDS)
    assert record["selection"] == "BLOCKED_INCOMPLETE"
    assert any("unregistered seed 42" in reason
               for reason in record["admission_rejections"])


def test_k_sweep_refuses_a_config_with_a_different_sweep_order(tmp_path, frame_dir, k_config):
    root = _k_runs(tmp_path, {str(k): {7: 1.5, 13: 1.5}
                              for k in ("0", "6", "12", "30", "unr")})
    config = _config_payload()
    config["statistics"]["k_selection"]["registered_sweep"] = [30, 12, 6, 0,
                                                               "unr"]
    with pytest.raises(st.RefusalError, match="sweep order"):
        st.k_sweep(root, config, seeds=SEEDS)


def test_real_config_registers_the_k_sweep_and_tolerance():
    real = st.read_yaml(REPO / "experiments" / "configs"
                        / "seq_stage2_v1.yaml")
    block = real["statistics"]["k_selection"]
    assert [str(k) for k in block["registered_sweep"]] == list(
        st.REGISTERED_K_ORDER)
    assert float(block["tolerance_ll"]) == st.K_TOLERANCE_LL


# ---------------------------------------------------------------------------
# access rules: no cohort, no smoke, no sealed holdout
# ---------------------------------------------------------------------------

FORBIDDEN_SOURCE_TOKENS = ("cohort", "smoke", "data/golden",
                           "forward_holdout")


def _string_constants(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    return [node.value for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)]


def test_no_source_path_constant_names_the_cohort_or_the_smoke_tree():
    """No string constant that looks like a path names a forbidden tree.

    The only literals allowed to carry a forbidden token AND a path separator
    are the refusal contract itself and the one CONTENT marker the report uses
    to label a dependency certificate taken from a smoke checkpoint (it is
    matched against a JSON field, never opened).
    """
    allowed = set(st.FORBIDDEN_FRAGMENTS) | {"/smoke/"}
    for name in ("stage2_stats.py", "render_stage2_report.py"):
        path = REPO / "scripts" / "sequence_track" / name
        for value in _string_constants(path):
            lowered = value.lower()
            if not any(token in lowered for token in FORBIDDEN_SOURCE_TOKENS):
                continue
            if value in allowed:
                continue
            # Anything else must be prose or a plain JSON key, never a path.
            assert "/" not in value or len(value) > 60, (name, value)


def test_guard_path_refuses_every_forbidden_tree():
    for bad in ("models/embeddings/seq_stage2/cohort/FROZEN.json",
                "models/embeddings/seq_stage2/smoke/mlp/seed_7/metrics.json",
                "data/golden/polymarket_test_v2/x.json",
                "data/forward_holdout/2026-06-01_2026-07-13/manifest.json"):
        with pytest.raises(st.RefusalError, match="refusing to open"):
            st.guard_path(REPO / bad)
    assert st.guard_path(REPO / "data" / "t20s_json").name == "t20s_json"


def test_a_whole_run_opens_no_forbidden_path(monkeypatch, config_path,
                                             runs_root, frame_dir,
                                             block_source, tmp_path):
    # pathlib routes through `io.open`, not `builtins.open`, so both are
    # instrumented: every real file this run touches is recorded.
    opened: list[str] = []
    real_builtin, real_io = builtins.open, io.open

    def spy(real):
        def wrapper(file, *args, **kwargs):
            opened.append(str(file))
            return real(file, *args, **kwargs)
        return wrapper

    monkeypatch.setattr(builtins, "open", spy(real_builtin))
    monkeypatch.setattr(io, "open", spy(real_io))
    _run_stats(config_path, runs_root, frame_dir, block_source, tmp_path)
    monkeypatch.setattr(builtins, "open", real_builtin)
    monkeypatch.setattr(io, "open", real_io)
    assert len(opened) > 10
    for path in opened:
        lowered = path.lower()
        for token in ("seq_stage2/cohort", "seq_stage2/smoke", "data/golden",
                      "forward_holdout"):
            assert token not in lowered, path


# ---------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------

def test_compute_statistics_degrades_cleanly_with_no_runs_at_all(
        config_path, frame_dir, block_source, tmp_path):
    payload = _run_stats(config_path, tmp_path / "empty_runs", frame_dir,
                         block_source, tmp_path)
    assert all(not block["complete_paired_seeds"]
               for block in payload["runs"].values())
    assert payload["not_evaluable"]
    assert len(payload["not_evaluable"]) == len(payload["contrasts"])
    for family in payload["families"]:
        for readout in st.READOUTS:
            assert family["screen"][readout]["status"] == "NOT_EVALUABLE"
    assert payload["cohort"]["cohort_status"] == "DEFERRED_UNOPENED"
    assert payload["cohort"]["advances"] == []


def test_cli_stats_and_ksweep_write_json(tmp_path, config_path, runs_root,
                                         frame_dir, block_source):
    out = tmp_path / "out" / "stats.json"
    code = st.main(["stats", "--config", str(config_path),
                    "--runs-root", str(runs_root),
                    "--frame-dir", str(frame_dir),
                    "--block-source-dir", str(block_source),
                    "--base-logits", str(tmp_path / "nope.npz"),
                    "--out", str(out), "--reps", "50",
                    "--expect-rows", str(EXPECT["rows"]),
                    "--expect-matches", str(EXPECT["matches"]),
                    "--expect-blocks", str(EXPECT["blocks"]),
                    "--expect-unmapped", "0",
                    "--expected-families", "3"])
    assert code == 0 and out.exists()
    payload = json.loads(out.read_text())
    assert payload["contract"]["reps"] == 50
    assert payload["contract"]["stage1_margin_imported"] is False

    k_root = _k_runs(tmp_path, {str(k): {7: 1.5, 13: 1.5}
                                for k in ("0", "6", "12", "30", "unr")})
    k_config_path = tmp_path / "k_config.yaml"
    k_config_path.write_text(yaml.safe_dump(
        _config_payload(frame_dir=frame_dir, same_entity=True),
        sort_keys=False))
    k_out = tmp_path / "out" / "k.json"
    assert st.main(["ksweep", "--config", str(k_config_path),
                    "--runs-root", str(k_root), "--out", str(k_out)]) == 0
    record = json.loads(k_out.read_text())
    assert record["selected_k"] == "30"
    assert record["k_to_config_id"]["unr"] == "same_entity_unr"


def test_cli_returns_two_on_a_refusal(tmp_path, config_path, runs_root,
                                     frame_dir, block_source, capsys):
    code = st.main(["stats", "--config", str(config_path),
                    "--runs-root", str(runs_root),
                    "--frame-dir", str(frame_dir),
                    "--block-source-dir", str(block_source),
                    "--out", str(tmp_path / "x.json"), "--reps", "50",
                    "--expected-families", "3", "--expect-blocks", "47"])
    assert code == 2
    assert "REFUSED" in capsys.readouterr().err


def test_recurrent_extra_arm_params_are_carried_not_asserted_away(
        tmp_path, frame_dir, block_source, config_path):
    root = tmp_path / "runs"
    frame = pd.read_parquet(frame_dir / "cricket_data_i7_validation.parquet")
    for config_id in CONFIG_IDS:
        for seed in SEEDS:
            extra = ({"cell": "sLSTM->mLSTM",
                      "simplifications": ["no conv1d", "single head"]}
                     if config_id == "fox" else None)
            _write_run(root, config_id, seed, frame, n_params=373174,
                       extra_arm_params=extra)
        _write_summary(root, config_id, {7: 1.5, 13: 1.5})
    payload = _run_stats(config_path, root, frame_dir, block_source, tmp_path)
    params = payload["runs"]["fox"]["seeds"][0]["arm_params"]
    assert params["n_parameters"] == 373174
    assert params["cell"] == "sLSTM->mLSTM"
    assert len(params["simplifications"]) == 2
    assert len(params) > 9


# ---------------------------------------------------------------------------
# The read-only admission verifier
# ---------------------------------------------------------------------------

def test_real_config_k_mapping_uses_the_registered_ids():
    """`same_entity_unr` has no `k` before `unr`; nothing may template it."""
    real = st.read_yaml(REPO / "experiments" / "configs"
                        / "seq_stage2_v1.yaml")
    mapping = st.k_configurations(real)
    assert [k for k, _ in mapping] == list(st.REGISTERED_K_ORDER)
    assert dict(mapping) == K_IDS
    assert dict(mapping)["unr"] == "same_entity_unr"
    assert "same_entity_kunr" not in dict(mapping).values()


def test_k_configurations_refuses_a_gap_or_a_duplicate():
    config = _config_payload(same_entity=True)
    config["configurations"] = [entry for entry in config["configurations"]
                               if entry["id"] != "same_entity_unr"]
    with pytest.raises(st.RefusalError, match="no `same_entity` configuration"):
        st.k_configurations(config)

    config = _config_payload(same_entity=True)
    config["configurations"].append(
        {"id": "same_entity_unr_again", "arm": "same_entity",
         "params": {"k": "unr"}})
    with pytest.raises(st.RefusalError, match="same k"):
        st.k_configurations(config)

    config = _config_payload(same_entity=True)
    config["configurations"].append(
        {"id": "same_entity_k99", "arm": "same_entity", "params": {"k": 99}})
    with pytest.raises(st.RefusalError, match="unregistered k"):
        st.k_configurations(config)

    config = _config_payload(same_entity=True)
    config["configurations"][-1].pop("params")
    with pytest.raises(st.RefusalError, match="no `params.k`"):
        st.k_configurations(config)


def test_k_sweep_finds_a_complete_real_id_sweep_and_lets_unr_win(
        tmp_path, frame_dir, k_config):
    """The regression for the templated-id defect: a complete sweep whose best
    k is the unrestricted arm selects it instead of reading BLOCKED."""
    root = _k_runs(tmp_path, {
        "0": {7: 1.60, 13: 1.60}, "6": {7: 1.58, 13: 1.58},
        "12": {7: 1.56, 13: 1.56}, "30": {7: 1.54, 13: 1.54},
        "unr": {7: 1.50, 13: 1.50}}, frame_dir)
    record = st.k_sweep(root, k_config, seeds=SEEDS)
    assert record["selection"] == "SELECTED"
    assert record["selected_config_id"] == "same_entity_unr"
    assert record["admission_rejections"] == []
    assert set(record["source_lls"]) == set(K_IDS.values())
    assert all(row["complete"] for row in record["rows"])


def test_admission_accepts_a_well_formed_run(tmp_path, frame_dir):
    frame = pd.read_parquet(frame_dir / "cricket_data_i7_validation.parquet")
    directory = _write_run(tmp_path / "runs", "full", 7, frame, arm="full")
    admission = st.verify_run_admission(directory, "full", 7,
                                        expected_arm="full", expected_k=None)
    assert admission.ok and admission.reason is None
    assert admission.training_signature
    assert admission.checkpoint_md5 == st.md5_file(directory / "model.pt")


@pytest.mark.parametrize("mutate,expected", [
    ("seed_in_run_record", "run_record.json names seed"),
    ("config_id_in_run_record", "run_record.json names config_id"),
    ("seed_in_metrics", "metrics.json trained seed"),
    ("wrong_arm", "arm_params.arm"),
    ("wrong_k", "arm_params.k"),
    ("no_signature", "no training_signature"),
    ("signature_disagreement", "training_signature"),
    ("broken_manifest", "md5 has changed"),
    ("missing_artefact", "incomplete run"),
    ("no_arm_params", "no arm_params"),
])
def test_admission_rejects_unverified_evidence(tmp_path, frame_dir, mutate,
                                               expected):
    frame = pd.read_parquet(frame_dir / "cricket_data_i7_validation.parquet")
    kwargs = {"arm": "full", "k": None}
    if mutate == "seed_in_run_record":
        kwargs["record_seed"] = 13
    if mutate == "config_id_in_run_record":
        kwargs["record_config_id"] = "fox"
    if mutate == "broken_manifest":
        kwargs["break_manifest"] = True
    directory = _write_run(tmp_path / "runs", "full", 7, frame, **kwargs)
    expected_arm, expected_k = "full", None
    if mutate == "seed_in_metrics":
        metrics = json.loads((directory / "metrics.json").read_text())
        metrics["config"]["seed"] = 99
        (directory / "metrics.json").write_text(json.dumps(metrics))
    if mutate == "wrong_arm":
        expected_arm = "mlp"
    if mutate == "wrong_k":
        expected_k = 30
    if mutate == "no_arm_params":
        metrics = json.loads((directory / "metrics.json").read_text())
        metrics.pop("arm_params")
        (directory / "metrics.json").write_text(json.dumps(metrics))
    if mutate == "no_signature":
        record = json.loads((directory / "run_record.json").read_text())
        record.pop("training_signature")
        (directory / "run_record.json").write_text(json.dumps(record))
    if mutate == "signature_disagreement":
        completion = json.loads((directory / "COMPLETE.json").read_text())
        completion["training_signature"] = "b" * 64
        (directory / "COMPLETE.json").write_text(json.dumps(completion))
    if mutate == "missing_artefact":
        (directory / "COMPLETE.json").unlink()

    admission = st.verify_run_admission(directory, "full", 7,
                                        expected_arm=expected_arm,
                                        expected_k=expected_k)
    assert admission.ok is False
    assert expected in (admission.reason or "")


def test_an_unverified_run_contributes_no_number(tmp_path, frame_dir,
                                                 block_source, config_path):
    root = tmp_path / "runs"
    frame = pd.read_parquet(frame_dir / "cricket_data_i7_validation.parquet")
    for config_id in CONFIG_IDS:
        for seed in SEEDS:
            _write_run(root, config_id, seed, frame, bias=0.5,
                       break_manifest=(config_id == "full" and seed == 13))
        _write_summary(root, config_id, {7: 1.5, 13: 1.5})
    payload = _run_stats(config_path, root, frame_dir, block_source, tmp_path)
    rows = {row["seed"]: row for row in payload["runs"]["full"]["seeds"]}
    assert rows[7]["admitted"] is True
    assert rows[13]["admitted"] is False
    assert "md5 has changed" in rows[13]["reason"]
    assert rows[13]["reconstructed_validation_ll"] is None
    assert payload["contrasts"]["full-mlp@all"]["available"] is False


def test_two_signatures_for_one_configuration_refuse(tmp_path, frame_dir,
                                                     block_source,
                                                     config_path):
    root = tmp_path / "runs"
    frame = pd.read_parquet(frame_dir / "cricket_data_i7_validation.parquet")
    for config_id in CONFIG_IDS:
        for seed in SEEDS:
            components = None
            if config_id == "fox" and seed == 13:
                # A self-consistent signature that is nonetheless a DIFFERENT
                # signature: its arm_params component moved, so it recomputes
                # correctly and still cannot sit in one table with seed 7.
                components = _components(
                    config_id, arm_params_expected={"arm": "fox", "k": None,
                                                    "wiring": "standard",
                                                    "history_input": "prev",
                                                    "key_construction": "x",
                                                    "bias": None,
                                                    "residual_l2": None,
                                                    "base_logits_md5": None})
            _write_run(root, config_id, seed, frame, components=components)
        _write_summary(root, config_id, {7: 1.5, 13: 1.5})
    with pytest.raises(st.RefusalError, match="different training signatures"):
        _run_stats(config_path, root, frame_dir, block_source, tmp_path)


def test_a_hand_edited_signature_string_refuses(tmp_path, frame_dir):
    """MUST-FIX A: the signature is recomputed from its own components, so an
    edited string cannot pass even when every component is well formed."""
    frame = pd.read_parquet(frame_dir / "cricket_data_i7_validation.parquet")
    directory = _write_run(tmp_path / "runs", "full", 7, frame,
                           signature="c" * 64)
    admission = st.verify_run_admission(directory, "full", 7)
    assert admission.ok is False
    assert "is not the signature its own components produce" in admission.reason


def test_a_moved_frame_component_refuses_across_arms(tmp_path, frame_dir,
                                                     block_source,
                                                     config_path):
    root = tmp_path / "runs"
    frame = pd.read_parquet(frame_dir / "cricket_data_i7_validation.parquet")
    for config_id in CONFIG_IDS:
        for seed in SEEDS:
            components = _components(
                config_id,
                **({"frame": st.sha256_text("moved")}
                   if config_id == "fox" else {}))
            _write_run(root, config_id, seed, frame, components=components)
        _write_summary(root, config_id, {7: 1.5, 13: 1.5})
    with pytest.raises(st.RefusalError, match="component 'frame' differs"):
        _run_stats(config_path, root, frame_dir, block_source, tmp_path)


def test_arm_specific_signature_components_may_differ(config_path, runs_root,
                                                      frame_dir, block_source,
                                                      tmp_path):
    """The driver's signature includes config_id/arm/arm_params, so requiring
    one signature across arms would reject every real night."""
    payload = _run_stats(config_path, runs_root, frame_dir, block_source,
                         tmp_path)
    signatures = payload["admission"]["training_signature_by_config"]
    assert len(set(signatures.values())) == len(CONFIG_IDS)
    assert set(payload["admission"]["shared_components"]) == set(
        st.SIGNATURE_SHARED_COMPONENTS)
    assert "config_id" in payload["admission"]["arm_specific_components"]


def test_frame_hash_drift_from_the_pin_refuses(tmp_path, frame_dir):
    config = _config_payload(frame_dir=frame_dir)
    config["provenance"]["frame"]["splits"]["validation"]["md5"] = "0" * 32
    with pytest.raises(st.RefusalError, match="the frame drifted"):
        st.load_frame(frame_dir, config)

    config = _config_payload(frame_dir=frame_dir)
    config["provenance"]["frame"]["splits"]["validation"]["n_rows"] = 999
    with pytest.raises(st.RefusalError, match="the pin records"):
        st.load_frame(frame_dir, config)

    config = _config_payload(frame_dir=frame_dir)
    config.pop("provenance")
    with pytest.raises(st.RefusalError, match="no pin to verify against"):
        st.load_frame(frame_dir, config)


def test_duplicate_seed_rows_in_a_summary_are_rejected(tmp_path, frame_dir):
    root = tmp_path / "runs"
    path = root / "full" / "summary.yaml"
    path.parent.mkdir(parents=True)
    rows = [{"seed": 7, "ll": 1.5}, {"seed": 7, "ll": 1.4}]
    path.write_text(yaml.safe_dump({
        "experiment": {"training_signature": SIGNATURE},
        "splits": {"validation": {"per_seed": rows}}}))
    with pytest.raises(st.RefusalError, match="more than once"):
        st.summary_lls(root, "full")


def test_k_sweep_verifies_summary_provenance_before_reading_the_lls(
        tmp_path, frame_dir, k_config):
    root = _k_runs(tmp_path, {
        "0": {7: 1.60, 13: 1.60}, "6": {7: 1.58, 13: 1.58},
        "12": {7: 1.56, 13: 1.56}, "30": {7: 1.54, 13: 1.54},
        "unr": {7: 1.20, 13: 1.20}}, frame_dir)
    # The unrestricted arm's summary claims a checkpoint md5 the manifest does
    # not have. Its (winning) log losses must not be read at all.
    path = root / "same_entity_unr" / "summary.yaml"
    payload = yaml.safe_load(path.read_text())
    payload["splits"]["validation"]["per_seed"][0]["checkpoint_md5"] = "0" * 32
    path.write_text(yaml.safe_dump(payload))
    record = st.k_sweep(root, k_config, seeds=SEEDS)
    assert record["selection"] == "BLOCKED_INCOMPLETE"
    assert record["selected_k"] is None
    assert any("checkpoint md5" in reason
               for reason in record["admission_rejections"])
    unr = next(row for row in record["rows"] if row["k"] == "unr")
    assert unr["per_seed"] == {"7": None, "13": None}
    assert unr["mean_ll"] is None


def test_k_sweep_rejects_a_summary_whose_signature_moved(tmp_path, frame_dir,
                                                        k_config):
    root = _k_runs(tmp_path, {
        "0": {7: 1.60, 13: 1.60}, "6": {7: 1.58, 13: 1.58},
        "12": {7: 1.56, 13: 1.56}, "30": {7: 1.54, 13: 1.54},
        "unr": {7: 1.20, 13: 1.20}}, frame_dir)
    path = root / "same_entity_unr" / "summary.yaml"
    payload = yaml.safe_load(path.read_text())
    payload["experiment"]["training_signature"] = "d" * 64
    path.write_text(yaml.safe_dump(payload))
    record = st.k_sweep(root, k_config, seeds=SEEDS)
    assert record["selection"] == "BLOCKED_INCOMPLETE"
    assert any("does not match the summary" in reason
               for reason in record["admission_rejections"])


def test_k_sweep_needs_the_registered_config():
    with pytest.raises(st.RefusalError, match="needs the registered config"):
        st.k_sweep(Path("."), {})


def test_machine_provenance_reads_the_drivers_nested_block(tmp_path,
                                                           frame_dir):
    frame = pd.read_parquet(frame_dir / "cricket_data_i7_validation.parquet")
    directory = _write_run(tmp_path / "runs", "full", 7, frame, arm="full")
    record = json.loads((directory / "run_record.json").read_text())
    metrics = json.loads((directory / "metrics.json").read_text())
    provenance = st.machine_provenance(record, metrics)
    assert provenance["machine"] == "laptop"
    assert provenance["hostname"] == "host7"
    assert provenance["chip"] == "Apple M-series"
    assert provenance["os"] == "Darwin 25.4.0"
    assert provenance["torch_version"] == "2.4.0"
    assert "mps built True" in provenance["mps_backend"]
    assert provenance["thread_caps"]["VECLIB_MAXIMUM_THREADS"] == "2"
    assert provenance["omp_num_threads"] == "2"
    assert provenance["mps_bit_reproducible"] is False
    assert provenance["machine_confounded_with_seed"] is True
    assert provenance["machine_term_fitted"] is False
    assert provenance["recorded"] is True


def test_machine_provenance_degrades_when_the_block_is_absent():
    provenance = st.machine_provenance({"wall_seconds": 1.0}, {})
    assert provenance["recorded"] is False
    assert provenance["machine"] is None
    assert provenance["chip"] is None
    assert provenance["thread_caps"] is None
    # The confound statement does not depend on any recorded field.
    assert provenance["machine_term_fitted"] is False


# ---------------------------------------------------------------------------
# Astra gate 1 round 3 — admission fails closed (MUST-FIX 2), summary values
# are authenticated and comparability reaches the k sweep (MUST-FIX 3), the
# four cross-arm requirements, and complete disclosure of inadmissible runs
# ---------------------------------------------------------------------------

def test_the_pin_translation_matches_the_drivers_own_construction(frame_dir):
    """The shared components are anchored to the pin, and the translation from
    the pin's shape to the driver's `signature_components` is exact.

    Without this lock, `pinned_component_digests` could drift from
    `retrain_stage2.signature_components` and every run would refuse (or,
    worse, agree with a wrong expectation).
    """
    config, pin, anchored = _anchor(frame_dir)
    driver = st.training_driver()
    provenance = config["provenance"]
    resolved = {
        "frame_dir_configured": provenance["frame"]["dir"],
        "frame_version": provenance["frame"]["version"],
        "feature_hash": provenance["frame"]["feature_hash"],
        "split_files": {split: provenance["frame"]["splits"][split]
                        for split in driver.CONTRACT_SPLITS},
        "stats_cache": {"role": CACHE_ROLE, "md5": CACHE_MD5},
    }
    training = config["training"]
    effective = {
        "config_id": "mlp",
        "arch": {"dmodel": training["dmodel"], "layers": training["layers"],
                 "heads": training["heads"]},
        "optimiser": {"lr": training["learning_rate"],
                      "batch": training["batch"],
                      "epochs": training["epochs"],
                      "patience": training["patience"],
                      "aux": training["aux"],
                      "aux_weight": driver.t1.AUX_WEIGHT_DEFAULT},
        "arm_params": {"arm": "mlp", "k": None, "wiring": "token_mlp",
                       "history_input": "none", "key_construction": "none",
                       "bias": None, "residual_l2": None,
                       "base_logits_md5": None},
        "overrides": {},
    }
    produced = driver.signature_components(effective, resolved)
    for name in st.SIGNATURE_SHARED_COMPONENTS:
        assert produced[name] == anchored[name], name
    assert produced["base_logits"] == anchored[
        "base_logits_by_reads_base"][False]
    # The implementation component: a pin carrying the real on-disk source
    # hashes must reproduce the driver's own `implementation_identity`, for a
    # non-recurrent and for a recurrent arm.
    real = _config_payload(frame_dir=frame_dir, same_entity=True)
    real["provenance"]["sources"]["source_sha256"] = {
        name: st.sha256_text((REPO / name).read_text())
        for name in (driver.TRAINER_SOURCE, driver.FEATURE_CONTRACT_SOURCE,
                     driver.ARTIFACT_RESOLVER_SOURCE,
                     driver.FEATURE_REGISTRY_SOURCE,
                     driver.RECURRENT_SOURCE)}
    real_anchored = st.pinned_component_digests(real, st.load_pin(real))
    # The pin anchors the three COMMON sources; the driver's set may be wider
    # (night 3 added feature_registry.py) and each run is anchored to the
    # hashes of its own recorded source list instead.
    common_identity = {name: digest for name, digest
                       in driver.implementation_identity("mlp").items()
                       if name in (driver.TRAINER_SOURCE,
                                   driver.FEATURE_CONTRACT_SOURCE,
                                   driver.ARTIFACT_RESOLVER_SOURCE)}
    assert real_anchored["implementation_by_recurrent"][False] == (
        st.component_digest(common_identity))
    recurrent_identity = dict(common_identity)
    recurrent_identity[driver.RECURRENT_SOURCE] = (
        driver.implementation_identity("lstm")[driver.RECURRENT_SOURCE])
    assert real_anchored["implementation_by_recurrent"][True] == (
        st.component_digest(recurrent_identity))


def test_a_manifest_missing_one_artefact_entry_refuses(tmp_path, frame_dir):
    """MUST-FIX A: an absent, empty or incomplete manifest is a refusal."""
    frame = pd.read_parquet(frame_dir / "cricket_data_i7_validation.parquet")
    directory = _write_run(tmp_path / "runs", "full", 7, frame)
    completion = json.loads((directory / "COMPLETE.json").read_text())
    dropped = dict(completion["artifacts"])
    dropped.pop("predictions_validation.npz")
    completion["artifacts"] = dropped
    (directory / "COMPLETE.json").write_text(json.dumps(completion))
    admission = st.verify_run_admission(directory, "full", 7)
    assert admission.ok is False
    assert "manifests no entry for predictions_validation.npz" in (
        admission.reason)


@pytest.mark.parametrize("mutate,expected", [
    ("empty_manifest", "carries no artefact manifest"),
    ("no_manifest_key", "carries no artefact manifest"),
    ("no_size", "declares no integer size"),
    ("no_md5", "declares no md5"),
    ("no_components", "carries no training_signature_components"),
    ("missing_component", "is missing ['stats_cache']"),
    ("mistyped_component", "are not sha256 hex digests"),
    ("no_validation_ll", "carries no finite validation_ll"),
])
def test_admission_fails_closed_on_every_missing_declaration(
        tmp_path, frame_dir, mutate, expected):
    frame = pd.read_parquet(frame_dir / "cricket_data_i7_validation.parquet")
    directory = _write_run(tmp_path / "runs", "full", 7, frame)
    completion = json.loads((directory / "COMPLETE.json").read_text())
    record = json.loads((directory / "run_record.json").read_text())
    if mutate == "empty_manifest":
        completion["artifacts"] = {}
    if mutate == "no_manifest_key":
        completion.pop("artifacts")
    if mutate == "no_size":
        completion["artifacts"]["model.pt"].pop("bytes")
    if mutate == "no_md5":
        completion["artifacts"]["model.pt"].pop("md5")
    if mutate == "no_components":
        record.pop("training_signature_components")
    if mutate == "missing_component":
        record["training_signature_components"].pop("stats_cache")
        completion["training_signature_components"].pop("stats_cache")
    if mutate == "mistyped_component":
        record["training_signature_components"]["frame"] = 17
        completion["training_signature_components"]["frame"] = 17
    (directory / "COMPLETE.json").write_text(json.dumps(completion))
    (directory / "run_record.json").write_text(json.dumps(record))
    if mutate == "no_validation_ll":
        metrics = json.loads((directory / "metrics.json").read_text())
        metrics.pop("validation_ll")
        (directory / "metrics.json").write_text(json.dumps(metrics))
        _restamp_metrics(directory)
    admission = st.verify_run_admission(directory, "full", 7)
    assert admission.ok is False
    assert expected in (admission.reason or "")


def test_a_hand_edited_summary_ll_refuses(tmp_path, frame_dir, block_source,
                                          config_path):
    """MUST-FIX B: changing only an `ll` in summary.yaml is refused, because it
    no longer equals the manifest-verified metrics.json validation LL."""
    root = tmp_path / "runs"
    frame = pd.read_parquet(frame_dir / "cricket_data_i7_validation.parquet")
    for config_id in CONFIG_IDS:
        for seed in SEEDS:
            _write_run(root, config_id, seed, frame)
        _write_summary(root, config_id, {7: 1.5, 13: 1.5})
    # The winner is edited to a better number and nothing else is touched.
    path = root / "fox" / "summary.yaml"
    payload = yaml.safe_load(path.read_text())
    payload["splits"]["validation"]["per_seed"][0]["ll"] = 0.9
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    with pytest.raises(st.RefusalError,
                       match="the summary was edited"):
        _run_stats(config_path, root, frame_dir, block_source, tmp_path)


def test_load_runs_calls_the_summary_verifier(tmp_path, frame_dir,
                                              block_source, config_path):
    """MUST-FIX B: `load_runs` used to read summaries without verifying them."""
    root = tmp_path / "runs"
    frame = pd.read_parquet(frame_dir / "cricket_data_i7_validation.parquet")
    for config_id in CONFIG_IDS:
        for seed in SEEDS:
            _write_run(root, config_id, seed, frame)
        _write_summary(root, config_id, {7: 1.5, 13: 1.5})
    # A checkpoint md5 the manifest does not have: verifiable provenance fails,
    # so no log loss from this summary is reported at all.
    path = root / "fox" / "summary.yaml"
    payload = yaml.safe_load(path.read_text())
    payload["splits"]["validation"]["per_seed"][0]["checkpoint_md5"] = "0" * 32
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    stats = _run_stats(config_path, root, frame_dir, block_source, tmp_path)
    block = stats["runs"]["fox"]
    assert block["summary_log_losses_dropped"] is True
    assert any("claims checkpoint md5" in problem
               for problem in block["summary_verification_problems"])
    assert all(row["summary_yaml_validation_ll"] is None
               for row in block["seeds"])
    # An untouched configuration still reports its summary number.
    assert stats["runs"]["full"]["seeds"][0][
        "summary_yaml_validation_ll"] == 1.5


def test_a_summary_ll_the_metrics_do_not_support_blocks_the_k_sweep(
        tmp_path, frame_dir, k_config):
    root = _k_runs(tmp_path, {
        "0": {7: 1.60, 13: 1.60}, "6": {7: 1.58, 13: 1.58},
        "12": {7: 1.56, 13: 1.56}, "30": {7: 1.54, 13: 1.54},
        "unr": {7: 1.53, 13: 1.53}}, frame_dir)
    path = root / "same_entity_unr" / "summary.yaml"
    payload = yaml.safe_load(path.read_text())
    for row in payload["splits"]["validation"]["per_seed"]:
        row["ll"] = 1.10
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    record = st.k_sweep(root, k_config, seeds=SEEDS)
    assert record["selection"] == "BLOCKED_INCOMPLETE"
    assert "same_entity_unr" in record["blocked_on"]
    assert any("the summary was edited" in reason
               for reason in record["admission_rejections"])


def test_k_sweep_refuses_incompatible_shared_components(tmp_path, frame_dir,
                                                        k_config):
    """MUST-FIX B: `k_sweep` never applied comparability, so five individually
    consistent summaries from incompatible runs could enter one selection."""
    moved = _components("same_entity_unr",
                        frame_dir, None, stats_cache=st.sha256_text("other"))
    root = _k_runs(tmp_path, {
        "0": {7: 1.60, 13: 1.60}, "6": {7: 1.58, 13: 1.58},
        "12": {7: 1.56, 13: 1.56}, "30": {7: 1.54, 13: 1.54},
        "unr": {7: 1.20, 13: 1.20}}, frame_dir,
        components_by_k={"unr": moved})
    with pytest.raises(st.RefusalError,
                       match="component 'stats_cache'"):
        st.k_sweep(root, k_config, seeds=SEEDS)


def test_k_sweep_records_that_it_applied_comparability(tmp_path, frame_dir,
                                                       k_config):
    root = _k_runs(tmp_path, {
        "0": {7: 1.60, 13: 1.60}, "6": {7: 1.58, 13: 1.58},
        "12": {7: 1.56, 13: 1.56}, "30": {7: 1.54, 13: 1.54},
        "unr": {7: 1.53, 13: 1.53}}, frame_dir)
    record = st.k_sweep(root, k_config, seeds=SEEDS)
    assert record["comparability"]["anchored_to_the_pin"] is True
    assert record["comparability"]["signature_recomputed_from_components"]
    assert len(record["comparability"]["training_signature_by_config"]) == 5


def _residual_config(frame_dir: Path) -> dict:
    """A config with the two residual arms, for the base-logits identity rule."""
    config = _config_payload(frame_dir=frame_dir)
    config["configurations"] = [
        {"id": "residual_mlp", "arm": "residual_mlp",
         "params": {"base_logits_dir": "models/embeddings/seq_stage2/"
                                       "base_logits", "residual_l2": 0.001},
         "access": {"features": True, "history": False, "identity": False,
                    "prod_logits": True},
         "history_input": "none", "wiring": "token_mlp",
         "reference": ["mlp"], "tests": "control", "queue_order": 1},
        {"id": "residual_t1", "arm": "residual_t1",
         "params": {"base_logits_dir": "models/embeddings/seq_stage2/"
                                       "base_logits", "residual_l2": 0.001},
         "access": {"features": True, "history": True, "identity": False,
                    "prod_logits": True},
         "history_input": "innings_previous", "wiring": "standard",
         "reference": ["residual_mlp"], "tests": "candidate",
         "queue_order": 2},
    ]
    return config


def _residual_admissions(frame_dir: Path, base_digests: Mapping[str, str]
                         ) -> dict:
    config = _residual_config(frame_dir)
    pin = st.load_pin(config)
    entries = {str(e["id"]): e for e in config["configurations"]}
    anchored = st.pinned_component_digests(config, pin)
    admissions: dict[str, dict[int, st.Admission]] = {}
    for config_id, entry in entries.items():
        identity = st.expected_arm_identity(config_id, entry, pin)
        arm_params = _registered_arm_params(config_id, entry, pin)
        components = {
            "config_id": identity["config_id"], "arm": identity["arm"],
            "arm_params": st.component_digest(dict(arm_params)),
            "training_block": anchored["training_block"],
            "frame": anchored["frame"],
            "stats_cache": anchored["stats_cache"],
            "base_logits": base_digests[config_id],
            "implementation": anchored["implementation_by_recurrent"][False],
        }
        admissions[config_id] = {7: st.Admission(
            True, None, st.recompute_training_signature(components),
            components, "md5",
            metrics_validation_ll=1.5,
            metrics_arm_params=dict(arm_params),
            arm_params_expected=dict(arm_params))}
    return config, pin, entries, admissions


def test_the_two_residual_arms_refuse_on_differing_base_logit_identity(
        frame_dir):
    """Astra round 3, cross-arm requirement 2."""
    config = _residual_config(frame_dir)
    pin = st.load_pin(config)
    right = st.component_digest(pin.base_logits_digest)
    wrong = st.component_digest("a different base-logits build")
    # Matched: one base-logits identity across both residual arms.
    config, pin, entries, admissions = _residual_admissions(
        frame_dir, {"residual_mlp": right, "residual_t1": right})
    report = st.assert_comparable(admissions, entries, config, pin)
    assert list(report["residual_base_logits_identities"]) == [right]

    config, pin, entries, admissions = _residual_admissions(
        frame_dir, {"residual_mlp": right, "residual_t1": wrong})
    with pytest.raises(st.RefusalError,
                       match="residual arms carry different `base_logits`"):
        st.assert_comparable(admissions, entries, config, pin)


def test_an_arms_components_must_match_its_registered_identity(tmp_path,
                                                               frame_dir):
    """Astra round 3, cross-arm requirement 3."""
    frame = pd.read_parquet(frame_dir / "cricket_data_i7_validation.parquet")
    root = tmp_path / "runs"
    # `full` is registered with history_input `prev`; this run recorded `none`.
    wrong = _registered_arm_params("full", _entries()["full"],
                                   st.load_pin(_anchor(frame_dir)[0]))
    wrong["history_input"] = "none"
    _write_run(root, "full", 7, frame, arm_params_expected=wrong)
    admissions = {"full": {7: st.verify_run_admission(
        root / "full" / "seed_7", "full", 7)}}
    config, pin, _ = _anchor(frame_dir)
    with pytest.raises(st.RefusalError, match="history_input"):
        st.assert_comparable(admissions, _entries(), config, pin)


def test_a_recurrent_arms_implementation_set_may_differ_but_not_the_common_one(
        frame_dir):
    """Astra round 3, cross-arm requirement 1."""
    config, pin, anchored = _anchor(frame_dir)
    non_recurrent = anchored["implementation_by_recurrent"][False]
    recurrent = anchored["implementation_by_recurrent"][True]
    assert non_recurrent != recurrent
    sources = anchored["implementation_sources_by_recurrent"]
    driver = st.training_driver()
    assert driver.RECURRENT_SOURCE in sources["True"]
    assert driver.RECURRENT_SOURCE not in sources["False"]
    for name in (driver.TRAINER_SOURCE, driver.FEATURE_CONTRACT_SOURCE,
                 driver.ARTIFACT_RESOLVER_SOURCE):
        assert name in sources["False"] and name in sources["True"]


def test_a_foreign_implementation_digest_refuses(tmp_path, frame_dir):
    frame = pd.read_parquet(frame_dir / "cricket_data_i7_validation.parquet")
    root = tmp_path / "runs"
    _write_run(root, "full", 7, frame, components=_components(
        "full", frame_dir, None, implementation=st.sha256_text("other build")))
    admissions = {"full": {7: st.verify_run_admission(
        root / "full" / "seed_7", "full", 7)}}
    config, pin, _ = _anchor(frame_dir)
    with pytest.raises(st.RefusalError, match="`implementation` component"):
        st.assert_comparable(admissions, _entries(), config, pin)


def test_every_inadmissible_run_is_reported_not_only_the_first(tmp_path,
                                                              frame_dir,
                                                              block_source,
                                                              config_path):
    """Astra round 3 minor item: report ALL of them."""
    root = tmp_path / "runs"
    frame = pd.read_parquet(frame_dir / "cricket_data_i7_validation.parquet")
    for config_id in CONFIG_IDS:
        for seed in SEEDS:
            _write_run(root, config_id, seed, frame,
                       break_manifest=(config_id in ("full", "fox")))
        _write_summary(root, config_id, {7: 1.5, 13: 1.5})
    stats = _run_stats(config_path, root, frame_dir, block_source, tmp_path)
    listed = {(row["config_id"], row["seed"])
              for row in stats["admission"]["inadmissible_runs"]}
    assert listed == {("full", 7), ("full", 13), ("fox", 7), ("fox", 13)}
    assert stats["admission"]["n_inadmissible_runs"] == 4
    assert all(row["reason"] for row in stats["admission"]["inadmissible_runs"])


def test_every_comparability_violation_is_reported_together(tmp_path,
                                                            frame_dir):
    frame = pd.read_parquet(frame_dir / "cricket_data_i7_validation.parquet")
    root = tmp_path / "runs"
    for config_id, moved in (("full", "frame"), ("fox", "stats_cache")):
        _write_run(root, config_id, 7, frame, components=_components(
            config_id, frame_dir, None,
            **{moved: st.sha256_text("moved " + moved)}))
    admissions = {config_id: {7: st.verify_run_admission(
        root / config_id / "seed_7", config_id, 7)}
        for config_id in ("full", "fox")}
    config, pin, _ = _anchor(frame_dir)
    with pytest.raises(st.RefusalError) as error:
        st.assert_comparable(admissions, _entries(), config, pin)
    message = str(error.value)
    assert "'frame'" in message and "'stats_cache'" in message
    assert "comparability violation(s)" in message


def test_a_provenance_block_without_a_validation_hash_is_no_pin(frame_dir):
    """MUST-FIX A, last clause."""
    config = _config_payload(frame_dir=frame_dir)
    config["provenance"]["frame"]["splits"]["validation"].pop("md5")
    pin = st.load_pin(config)
    assert pin.available is False
    assert "validation.md5" in pin.reason

    config = _config_payload(frame_dir=frame_dir)
    config["provenance"] = {"pinned_by": "someone"}
    pin = st.load_pin(config)
    assert pin.available is False
    assert "not a usable pin" in pin.reason


def test_no_run_is_admitted_without_a_pin_to_anchor_to(tmp_path, frame_dir):
    frame = pd.read_parquet(frame_dir / "cricket_data_i7_validation.parquet")
    root = tmp_path / "runs"
    _write_run(root, "full", 7, frame)
    admissions = {"full": {7: st.verify_run_admission(
        root / "full" / "seed_7", "full", 7)}}
    config = _config_payload(frame_dir=frame_dir)
    config["provenance"] = {"pinned_by": "someone"}
    with pytest.raises(st.RefusalError, match="no run may be admitted"):
        st.assert_comparable(admissions, _entries(), config,
                             st.load_pin(config))


def test_an_absent_component_is_never_skipped(frame_dir):
    """MUST-FIX A: `assert_comparable` used to `continue` past a missing one."""
    config, pin, anchored = _anchor(frame_dir)
    components = _components("full", frame_dir)
    components.pop("stats_cache")
    admissions = {"full": {7: st.Admission(
        True, None, "x" * 64, components, "md5",
        metrics_validation_ll=1.5)}}
    with pytest.raises(st.RefusalError, match=r"\['stats_cache'\] are absent"):
        st.assert_comparable(admissions, _entries(), config, pin)


def test_a_wrong_key_construction_refuses(tmp_path, frame_dir):
    """Requirement 3 covers `key_construction`, whose authority is the
    trainer's registered table — legitimately `None` for a non-attention arm."""
    frame = pd.read_parquet(frame_dir / "cricket_data_i7_validation.parquet")
    root = tmp_path / "runs"
    config, pin, _ = _anchor(frame_dir)
    identity = st.expected_arm_identity("mlp", _entries()["mlp"], pin)
    assert identity["key_construction"] is None
    forged = _registered_arm_params("full", _entries()["full"], pin)
    forged["key_construction"] = "own_outcome"
    _write_run(root, "full", 7, frame, arm_params_expected=forged)
    admissions = {"full": {7: st.verify_run_admission(
        root / "full" / "seed_7", "full", 7)}}
    with pytest.raises(st.RefusalError, match="key_construction"):
        st.assert_comparable(admissions, _entries(), config, pin)


# ---------------------------------------------------------------------------
# Astra gate 2 round 3 — MUST-FIX 4 (two direction counts, not one) and
# SHOULD 5 (no obsolete seed-status prose).
# ---------------------------------------------------------------------------

def _five_seed_statistics(tmp_path, frame_dir, block_source):
    config = _config_payload(frame_dir=frame_dir, same_entity=True)
    config["training"] = dict(config.get("training") or {})
    config["training"]["seeds"] = list(FIVE_SEEDS)
    config_path = tmp_path / "config_five.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    frame = pd.read_parquet(frame_dir / "cricket_data_i7_validation.parquet")
    runs = tmp_path / "runs"
    for config_id, bias in (("mlp", 0.0), ("full", 0.9),
                            ("fixed_decay", 0.5), ("fox", 0.7)):
        for seed in FIVE_SEEDS:
            _write_run(runs, config_id, seed, frame, bias=bias)
        _write_summary(runs, config_id,
                       {seed: 1.5 - bias / 10 + seed / 100000
                        for seed in FIVE_SEEDS})
    for k, config_id in K_IDS.items():
        for seed in FIVE_SEEDS:
            _write_run(runs, config_id, seed, frame, bias=0.4,
                       arm="same_entity", k=(k if k == "unr" else int(k)))
        _write_summary(runs, config_id,
                       {seed: 1.5 + seed / 100000 for seed in FIVE_SEEDS})
    stats = st.compute_statistics(
        config_path, runs, frame_dir, block_source,
        tmp_path / "no_base_logits.npz", reps=50, seed=29, seeds=FIVE_SEEDS,
        expect=dict(EXPECT), expected_families=8)
    return config, runs, stats


def test_both_direction_counts_are_recorded_with_their_threshold(
        tmp_path, frame_dir, block_source):
    """MUST-FIX 4: the registered count keeps its key and its arithmetic, and
    the strictly favourable count is published beside it."""
    _, _, stats = _five_seed_statistics(tmp_path, frame_dir, block_source)
    gates, primaries = 0, 0
    for record in stats["contrasts"].values():
        if not record.get("available") or not str(
                record.get("role", "")).startswith("family_"):
            continue
        points = list(record["per_seed_points"].values())
        threshold = record["direction_count_threshold"]
        assert threshold == record["threshold"]
        assert (record["favourable_direction_count"]
                == record["seeds_below_registered_threshold_count"]
                == sum(1 for value in points if value < threshold))
        assert record["seeds_below_zero_count"] == sum(
            1 for value in points if value < 0.0)
        assert "may still have an ADVERSE" in record["direction_count_note"]
        if record["role"] == "family_primary":
            primaries += 1
            assert threshold == 0.0
            assert (record["seeds_below_zero_count"]
                    == record["favourable_direction_count"])
        else:
            gates += 1
            assert threshold == st.MARGIN_LL
            assert (record["seeds_below_zero_count"]
                    <= record["favourable_direction_count"])
    assert primaries and gates


def test_the_family_evidence_status_names_the_actual_seed_count(
        tmp_path, frame_dir, block_source):
    """SHOULD 5: the family JSON said "screening: two seeds" at five seeds."""
    _, _, stats = _five_seed_statistics(tmp_path, frame_dir, block_source)
    for family in stats["families"]:
        for screen in family["screen"].values():
            assert screen["evidence_status"] == (
                "screening: five seeds, validation only, checkpoint selected "
                "on the same split")
    family, tables, contrasts = _screen_inputs(2, 2)
    screen = st.family_screen(family, tables, contrasts, st.JOINT_READOUT)
    assert screen["evidence_status"] == (
        "screening: two seeds, validation only, checkpoint selected on the "
        "same split")


def test_the_completed_five_seed_k_selection_awaits_disposition_not_another_extension(
        tmp_path, frame_dir, block_source):
    """SHOULD 5: the five-seed selection must not imply a further extension."""
    config, runs, _ = _five_seed_statistics(tmp_path, frame_dir, block_source)
    five = st.k_sweep(runs, config, seeds=FIVE_SEEDS)
    assert five["provisional"] is True
    assert "whole-family seed extension" not in five["provisional_note"]
    assert "awaiting final disposition" in five["provisional_note"]
    assert "validation-only" in five["provisional_note"]
    # Below the registered minimum the outstanding condition really is the
    # extension, and that wording is unchanged.
    two = st.k_sweep(runs, config, seeds=SEEDS)
    assert two["provisional_note"] == (
        "the two-seed selection remains explicitly provisional pending any "
        "registered whole-family seed extension")


# ---------------------------------------------------------------------------
# night 3 — the generalised family / role / slice machinery
#
# Every test below asserts BOTH halves of the contract: the new key does what
# it registers, and a config carrying none of the new keys behaves exactly as
# it did before.
# ---------------------------------------------------------------------------

def _general_family(candidate: str, members: list[dict], **extra) -> dict:
    entry = {"candidate": candidate, "members": members,
             "holm_group": f"family_{candidate}"}
    entry.update(extra)
    return entry


def _member(name: str, candidate: str, reference: str, slice_name: str,
            kind: str = "superiority", **extra) -> dict:
    member = {"name": name,
              "contrast": {"candidate": candidate, "reference": reference},
              "slice": slice_name, "kind": kind}
    member.update(extra)
    return member


def _gates(candidate: str, reference: str = "mlp") -> list[dict]:
    return [_member("death_gate", candidate, reference, "death",
                    "non_inferiority"),
            _member("chase_gate", candidate, reference, "chase",
                    "non_inferiority")]


def test_the_legacy_family_form_is_translated_into_the_general_form(
        config_path, runs_root, frame_dir, block_source, tmp_path):
    """The fixed {primary, death_gate, chase_gate} form is the general one."""
    config = _config_payload(frame_dir=frame_dir)
    families = st.registered_families(config)
    full = next(f for f in families if f["candidate"] == "full")
    assert [m["member"] for m in full["members"]] == list(
        st.FAMILY_MEMBER_ORDER)
    assert [m["kind"] for m in full["members"]] == [
        "superiority", "non_inferiority", "non_inferiority"]
    assert [m["threshold"] for m in full["members"]] == [
        0.0, st.MARGIN_LL, st.MARGIN_LL]
    assert [m["primary"] for m in full["members"]] == [True, False, False]
    assert [m["slice"] for m in full["members"]] == ["all", "death", "chase"]
    # A translated legacy family carries none of the generalised keys, so its
    # emitted payload is byte-for-byte what it always was.
    assert "screen_spec" not in full
    # Carried internally so the screen reads the registered control, and NOT
    # emitted for a legacy family (asserted on the payload below).
    assert full["shared_control"] == "mlp"

    # The same family written in the general form is the same family.
    general = dict(config)
    general["statistics"] = dict(config["statistics"])
    general["statistics"]["families"] = dict(config["statistics"]["families"])
    general["statistics"]["families"]["map"] = [
        _general_family("full", [
            _member("primary", "full", "mlp", "all", primary=True),
            *_gates("full")]),
        _family("fixed_decay", "mlp"), _family("fox", "fixed_decay")]
    payload = _run_stats(config_path, runs_root, frame_dir, block_source,
                         tmp_path)
    emitted = next(f for f in payload["families"] if f["candidate"] == "full")
    assert "shared_control" not in emitted
    assert "screen_spec" not in emitted
    assert all("primary" not in member for member in emitted["members"])
    screen = emitted["screen"][st.JOINT_READOUT]
    assert screen["death_gate_status"] in st.ALLOWED_STATUSES
    assert "screen_required_members" not in screen

    translated = next(f for f in st.registered_families(general)
                      if f["candidate"] == "full")
    for left, right in zip(full["members"], translated["members"]):
        assert {k: left[k] for k in ("member", "candidate", "reference",
                                     "slice", "kind", "threshold", "primary")
                } == {k: right[k] for k in ("member", "candidate", "reference",
                                            "slice", "kind", "threshold",
                                            "primary")}


def _four_member_config(frame_dir: Path, **family_extra) -> dict:
    config = _config_payload(frame_dir=frame_dir)
    config["statistics"]["families"]["map"] = [
        _general_family(
            "full",
            [_member("beats_fixed_decay", "full", "fixed_decay", "all",
                     primary=True),
             _member("beats_fox", "full", "fox", "all"),
             *_gates("full")],
            **family_extra),
        _family("fixed_decay", "mlp"),
        _family("fox", "fixed_decay")]
    return config


def _write_config(tmp_path: Path, config: dict, name: str = "general.yaml"):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def test_a_four_member_family_with_two_superiority_members_and_its_own_screen(
        tmp_path, frame_dir, runs_root, block_source):
    config = _four_member_config(
        frame_dir,
        screen={"require": ["beats_fixed_decay", "beats_fox", "death_gate",
                            "chase_gate"]})
    path = _write_config(tmp_path, config)
    payload = _run_stats(path, runs_root, frame_dir, block_source, tmp_path)

    family = next(f for f in payload["families"] if f["candidate"] == "full")
    assert len(family["members"]) == 4
    assert family["shared_control"] == "mlp"
    assert family["screen_spec"]["screen_is_registered_per_family"] is True

    # Holm's multiplier is the family's own size, not the legacy constant.
    for table in family["holm"].values():
        assert table["m"] == 4
        assert [row["member"] for row in table["members"]] == [
            "beats_fixed_decay", "beats_fox", "death_gate", "chase_gate"]
        # Tie order is the registered member order.
        assert [row["kind"] for row in table["members"]] == [
            "superiority", "superiority", "non_inferiority",
            "non_inferiority"]
        # A superiority member is strict against zero whatever its position.
        for row in table["members"][:2]:
            assert row["threshold"] == 0.0
            if row["u95"] is not None:
                assert row["strict_upper_bound_ok"] == (row["u95"] < 0.0)

    screen = family["screen"][st.JOINT_READOUT]
    assert screen["primary_member"] == "beats_fixed_decay"
    assert screen["screen_required_members"] == [
        "beats_fixed_decay", "beats_fox", "death_gate", "chase_gate"]
    assert set(screen["screen_required_member_statuses"]) == set(
        screen["screen_required_members"])
    assert screen["all_row_condition"] == "inherited"
    assert screen["status"] in st.ALLOWED_STATUSES
    # Both non-inferiority members, and only those, become gate rows.
    gate_keys = {row["contrast_key"] for row in payload["gates"]
                 if row["candidate"] == "full"}
    assert gate_keys == {"full-mlp@death", "full-mlp@chase"}
    # The untouched legacy families are still exactly three members.
    for candidate in ("fixed_decay", "fox"):
        other = next(f for f in payload["families"]
                     if f["candidate"] == candidate)
        assert len(other["members"]) == 3
        assert all(table["m"] == 3 for table in other["holm"].values())


def test_a_per_family_screen_requires_exactly_the_members_it_names(
        tmp_path, frame_dir, runs_root, block_source):
    """A narrower screen ignores the members it does not require."""
    config = _four_member_config(
        frame_dir, screen={"require": ["beats_fixed_decay"]},
        all_row_condition="none")
    path = _write_config(tmp_path, config)
    payload = _run_stats(path, runs_root, frame_dir, block_source, tmp_path)
    screen = next(f for f in payload["families"]
                  if f["candidate"] == "full")["screen"][st.JOINT_READOUT]
    assert screen["screen_required_members"] == ["beats_fixed_decay"]
    assert screen["gate_statuses"] == {}
    # `all_row_condition: none` retires the inherited extra condition, and the
    # report records that it did.
    assert screen["all_row_condition"] == "none"
    assert screen["all_row_condition_is_registered"] is True
    assert screen["all_row_candidate_minus_mlp_ci_clean_favourable"] is None
    # The screen is then exactly the required member's own status.
    if screen["status"] != st.STATUS_NOT_EVALUABLE:
        expected = (st.STATUS_PASS
                    if (screen["screen_required_member_statuses"][
                        "beats_fixed_decay"] == st.STATUS_PASS
                        and screen[
                            "favourable_direction_requirement_met"] is not
                        False)
                    else st.STATUS_NOT_PASS)
        assert screen["status"] == expected


def test_a_family_registers_two_to_six_members(frame_dir):
    config = _config_payload(frame_dir=frame_dir)
    base = [_member("primary", "full", "mlp", "all", primary=True),
            *_gates("full")]

    def with_members(members):
        payload = _config_payload(frame_dir=frame_dir)
        payload["statistics"]["families"]["map"] = [
            _general_family("full", members),
            _family("fixed_decay", "mlp"), _family("fox", "fixed_decay")]
        return payload

    two = st.registered_families(with_members(base[:2]))
    assert len(next(f for f in two if f["candidate"] == "full")["members"]) == 2

    # Six members: the 4b identity-residual shape.
    six = base + [
        _member("vs_fox", "full", "fox", "all"),
        _member("vs_fixed_decay", "full", "fixed_decay", "all"),
        _member("powerplay_gate", "full", "mlp", "powerplay",
                "non_inferiority")]
    assert len(six) == 6
    families = st.registered_families(with_members(six))
    full = next(f for f in families if f["candidate"] == "full")
    assert len(full["members"]) == 6
    assert st.FAMILY_MAX_MEMBERS == 6

    seven = six + [_member("middle_gate", "full", "mlp", "middle",
                           "non_inferiority")]
    with pytest.raises(st.RefusalError, match="2-6"):
        st.registered_families(with_members(seven))
    with pytest.raises(st.RefusalError, match="2-6"):
        st.registered_families(with_members(base[:1]))
    del config


def test_a_general_family_needs_exactly_one_primary_member(frame_dir):
    def with_members(members):
        payload = _config_payload(frame_dir=frame_dir)
        payload["statistics"]["families"]["map"] = [
            _general_family("full", members),
            _family("fixed_decay", "mlp"), _family("fox", "fixed_decay")]
        return payload

    none_flagged = [_member("a", "full", "mlp", "all"),
                    _member("b", "full", "fox", "all")]
    with pytest.raises(st.RefusalError, match="exactly one"):
        st.registered_families(with_members(none_flagged))
    two_flagged = [_member("a", "full", "mlp", "all", primary=True),
                   _member("b", "full", "fox", "all", primary=True)]
    with pytest.raises(st.RefusalError, match="exactly one"):
        st.registered_families(with_members(two_flagged))


def test_a_superiority_member_may_not_register_a_non_zero_threshold(frame_dir):
    payload = _config_payload(frame_dir=frame_dir)
    payload["statistics"]["families"]["map"] = [
        _general_family("full", [
            _member("primary", "full", "mlp", "all", primary=True,
                    threshold=0.002),
            *_gates("full")]),
        _family("fixed_decay", "mlp"), _family("fox", "fixed_decay")]
    with pytest.raises(st.RefusalError, match="threshold is 0"):
        st.registered_families(payload)


def test_a_control_role_configuration_is_a_candidate_in_no_family(
        tmp_path, frame_dir, runs_root, block_source):
    config = _config_payload(frame_dir=frame_dir)
    fox = next(entry for entry in config["configurations"]
               if entry["id"] == "fox")
    fox["role"] = "control"
    config["statistics"]["families"]["map"] = [
        _family("full", "mlp"), _family("fixed_decay", "mlp")]

    # Default is `candidate`, so the count is derived, not declared.
    assert st.config_roles(config)["full"] == "candidate"
    assert st.config_roles(config)["fox"] == "control"
    assert st.config_roles(config)["mlp"] == "control"
    assert st.candidate_ids(config) == {"full", "fixed_decay"}
    assert st.expected_family_count(config) == 2
    families = st.registered_families(config)
    assert {f["candidate"] for f in families} == {"full", "fixed_decay"}

    # A control that IS given a family is refused, and says why.
    with_family = _config_payload(frame_dir=frame_dir)
    next(entry for entry in with_family["configurations"]
         if entry["id"] == "fox")["role"] = "control"
    with pytest.raises(st.RefusalError, match="role: control"):
        st.registered_families(with_family, expected=3)

    # A candidate left out of the map is still refused, by name.
    missing = _config_payload(frame_dir=frame_dir)
    missing["statistics"]["families"]["map"] = [_family("full", "mlp"),
                                               _family("fox", "fixed_decay")]
    with pytest.raises(st.RefusalError, match="fixed_decay"):
        st.registered_families(missing, expected=2)

    # And an unknown role is refused rather than silently treated as one.
    bad = _config_payload(frame_dir=frame_dir)
    next(entry for entry in bad["configurations"]
         if entry["id"] == "fox")["role"] = "reference"
    with pytest.raises(st.RefusalError, match="role"):
        st.config_roles(bad)

    path = _write_config(tmp_path, config, "control_role.yaml")
    payload = _run_stats(path, runs_root, frame_dir, block_source, tmp_path,
                         expected_families=2)
    assert len(payload["families"]) == 2


def test_the_shared_control_key_replaces_the_mlp_literal(
        tmp_path, frame_dir, runs_root, block_source):
    config = _config_payload(frame_dir=frame_dir)
    assert st.shared_control(config) == "mlp"        # absent key -> mlp

    config["statistics"]["families"]["shared_control"] = "full"
    config["statistics"]["families"]["map"] = [
        _family("mlp", "full"), _family("fixed_decay", "full"),
        _family("fox", "full")]
    assert st.shared_control(config) == "full"
    assert st.candidate_ids(config) == {"mlp", "fixed_decay", "fox"}
    assert st.expected_family_count(config) == 3
    st.registered_families(config)

    # The shared control may not be a family candidate, whichever id it is.
    clash = yaml.safe_load(yaml.safe_dump(config))
    clash["statistics"]["families"]["map"].append(_family("full", "mlp"))
    with pytest.raises(st.RefusalError, match="full is the shared control"):
        st.registered_families(clash, expected=4)

    # The exploratory pair generator and the screen's all-row condition follow
    # the key, not the literal.
    path = _write_config(tmp_path, config, "shared_control.yaml")
    payload = _run_stats(path, runs_root, frame_dir, block_source, tmp_path)
    assert "mlp-full@all" in payload["contrasts"]
    assert "fox-full@death" in payload["contrasts"]
    assert payload["contrasts"]["fox-full@death"]["threshold"] == st.MARGIN_LL
    for family in payload["families"]:
        for screen in family["screen"].values():
            assert screen["all_row_candidate_minus_mlp_key"] == (
                f"{family['candidate']}-full@all")


# --- slices ----------------------------------------------------------------

def _runs_for(root: Path, directory: Path) -> Path:
    """The `runs_root` fixture's runs, written against another frame dir."""
    global _FRAME_DIR
    previous, _FRAME_DIR = _FRAME_DIR, directory
    try:
        frame = pd.read_parquet(
            directory / "cricket_data_i7_validation.parquet")
        for config_id, bias in {"mlp": 0.0, "full": 0.9, "fixed_decay": 0.5,
                                "fox": 0.7}.items():
            for seed in SEEDS:
                _write_run(root, config_id, seed, frame, bias=bias)
            _write_summary(root, config_id, {7: 1.5 - bias / 10,
                                             13: 1.51 - bias / 10})
    finally:
        _FRAME_DIR = previous
    return root


def _tier_frame(frame_dir: Path, tmp_path: Path, name: str = "tier_frame"):
    """The test frame plus a `competition_tier` column, tier by match index."""
    directory = tmp_path / name
    directory.mkdir()
    df = _frame_rows()
    match_index = df["innings_id"].str.split("_").str[-1].str[-1].astype(int)
    df["competition_tier"] = (match_index % 4) + 1
    df.to_parquet(directory / "cricket_data_i7_validation.parquet",
                  index=False)
    return directory


def test_tier_slices_turn_on_only_from_a_registered_spec(tmp_path, frame_dir):
    directory = _tier_frame(frame_dir, tmp_path)
    config = _config_payload(frame_dir=directory)

    # Absent key: the computed slice list is exactly the frozen one, even
    # though the frame carries the column.
    frame = st.load_frame(directory, config)
    assert {p["slice"] for p in frame.predicates} == {
        "all", "death", "chase", "powerplay", "middle", "innings_1",
        "innings_2", "thin_pair"}
    assert st.optional_slice_specs(config) == []

    config["statistics"]["slice_predicates"] = {
        "tier3": {"tier": 3}, "target_P": {"tier": 3}, "tier1": {"tier": 1},
        "death@target_P": {}, "chase@target_P": {"base": "chase",
                                                 "restrict": "target_P"}}
    frame = st.load_frame(directory, config)
    by_name = {p["slice"]: p for p in frame.predicates}
    assert by_name["tier3"]["predicate"] == "competition_tier == 3"
    # `target_P` is `tier3` under a second registered name: same rows.
    assert frame.masks["target_P"].tolist() == frame.masks["tier3"].tolist()
    assert frame.masks["tier3"].sum() == 2 * 2 * BALLS_PER_INNINGS
    assert frame.masks["tier1"].sum() == 2 * 2 * BALLS_PER_INNINGS
    assert not (frame.masks["tier1"] & frame.masks["tier3"]).any()

    # The composed gate slice is the base AND the restriction, and keeps the
    # base's gate role so it is still compared against the margin.
    composed = by_name["death@target_P"]
    assert composed["role"] == "gate"
    assert composed["restricts_slice"] == "death"
    assert composed["restricted_to"] == "target_P"
    assert composed["predicate"] == (
        "is_death_overs == 1 and competition_tier == 3")
    assert frame.masks["death@target_P"].tolist() == (
        frame.masks["death"] & frame.masks["target_P"]).tolist()
    assert frame.masks["chase@target_P"].tolist() == (
        frame.masks["chase"] & frame.masks["target_P"]).tolist()

    # An unregistered tier, an unknown base and a double composition refuse.
    for spec, match in (({"tier9": {"tier": 9}}, "tier 9"),
                        ({"nope@tier3": {}}, "restricts"),
                        ({"tier3": {"tier": 3}, "death@tier3": {},
                          "twice": {"base": "chase",
                                    "restrict": "death@tier3"}},
                         "compose once")):
        bad = _config_payload(frame_dir=directory)
        bad["statistics"]["slice_predicates"] = spec
        with pytest.raises(st.RefusalError, match=match):
            st.load_frame(directory, bad)


def test_a_tier_slice_is_reported_unavailable_when_the_column_is_absent(
        frame_dir):
    """Exactly the thin_pair discipline: unavailable with a reason, never
    invented and never breaking the read of the other slices."""
    config = _config_payload(frame_dir=frame_dir)
    config["statistics"]["slice_predicates"] = {
        "target_P": {"tier": 3}, "death@target_P": {}}
    frame = st.load_frame(frame_dir, config)          # no competition_tier
    by_name = {p["slice"]: p for p in frame.predicates}
    assert by_name["target_P"]["available"] is False
    assert "competition_tier" in by_name["target_P"]["unavailable_reason"]
    assert "target_P" not in frame.masks
    # The composition of an unavailable slice is itself unavailable, and says
    # which slice it was waiting on.
    assert by_name["death@target_P"]["available"] is False
    assert "target_P" in by_name["death@target_P"]["unavailable_reason"]
    assert "death@target_P" not in frame.masks
    # Every other slice is unaffected.
    assert by_name["death"]["available"] is True
    assert frame.masks["chase"].sum() == N_MATCHES * BALLS_PER_INNINGS


def test_match_list_slices_read_the_registered_file_and_check_its_sha256(
        tmp_path, frame_dir):
    listed = [f"90000{index}" for index in range(3)]
    target_e = tmp_path / "target_E_matches.json"
    target_e.write_text(json.dumps(listed))
    big3 = tmp_path / "big3_validation_matches.json"
    big3.write_text(json.dumps({"match_ids": listed[:1]}))

    config = _config_payload(frame_dir=frame_dir)
    config["statistics"]["slice_predicates"] = {
        "target_E": {"match_list": str(target_e),
                     "sha256": st.sha256_file(target_e)},
        "big3": {"match_list": str(big3),
                 "sha256": st.sha256_file(big3)},
        "death@target_E": {}}
    frame = st.load_frame(frame_dir, config)
    by_name = {p["slice"]: p for p in frame.predicates}
    assert by_name["target_E"]["available"] is True
    assert by_name["target_E"]["n_matches"] == 3
    assert by_name["target_E"]["match_list_sha256"] == st.sha256_file(target_e)
    # Membership is the innings_id SUFFIX, so both innings of a listed match
    # are in and nothing else is.
    assert frame.masks["target_E"].sum() == 3 * 2 * BALLS_PER_INNINGS
    assert frame.masks["big3"].sum() == 1 * 2 * BALLS_PER_INNINGS
    assert frame.masks["death@target_E"].tolist() == (
        frame.masks["death"] & frame.masks["target_E"]).tolist()

    # A drifted list computes no number.
    target_e.write_text(json.dumps(listed[:2]))
    frame = st.load_frame(frame_dir, config)
    drifted = next(p for p in frame.predicates if p["slice"] == "target_E")
    assert drifted["available"] is False
    assert "drifted" in drifted["unavailable_reason"]
    assert "target_E" not in frame.masks

    # An absent list is unavailable, not invented. (Its sha256 is still
    # required: the pin is what a later capture is checked against.)
    absent = tmp_path / "not_here.json"
    config["statistics"]["slice_predicates"] = {
        "target_E": {"match_list": str(absent), "sha256": "0" * 64}}
    frame = st.load_frame(frame_dir, config)
    missing = next(p for p in frame.predicates if p["slice"] == "target_E")
    assert missing["available"] is False
    assert "does not exist" in missing["unavailable_reason"]


def test_a_match_list_slice_without_a_sha256_is_refused(tmp_path, frame_dir):
    """No unpinned slices: the rows a match list selects must not be able to
    change under the analysis, so the digest is required at config load."""
    listed = tmp_path / "unpinned_matches.json"
    listed.write_text(json.dumps(["900000"]))
    config = _config_payload(frame_dir=frame_dir)
    config["statistics"]["slice_predicates"] = {
        "target_E": {"match_list": str(listed)}}
    with pytest.raises(st.RefusalError, match="no `sha256`"):
        st.optional_slice_specs(config)
    with pytest.raises(st.RefusalError, match="unpinned match list"):
        st.load_frame(frame_dir, config)
    # With the digest it loads.
    config["statistics"]["slice_predicates"]["target_E"]["sha256"] = (
        st.sha256_file(listed))
    frame = st.load_frame(frame_dir, config)
    assert frame.masks["target_E"].sum() == 2 * BALLS_PER_INNINGS


def test_guard_path_resolves_before_testing_the_forbidden_fragments(tmp_path):
    """A symlink must not carry a read into a sealed holdout (or the cohort).

    The guard tests the path as written AND its resolved form, and still returns
    the path as written so no reported path becomes absolute.
    """
    sealed = tmp_path / "data" / "golden"
    sealed.mkdir(parents=True)
    target = sealed / "polymarket_test_v2.json"
    target.write_text("{}")
    innocent = tmp_path / "innocuous_link.json"
    innocent.symlink_to(target)
    # Nothing in the name as written names a sealed holdout; the resolved form
    # does, and that is what the refusal reports.
    assert "data/golden" not in innocent.as_posix()
    with pytest.raises(st.RefusalError, match="resolves to"):
        st.guard_path(innocent)
    # The direct spelling is refused as before, naming the fragment.
    with pytest.raises(st.RefusalError, match="data/golden"):
        st.guard_path(target)
    # A `..` hop into the same tree is refused too, and it need not exist.
    hop = tmp_path / "data" / "sub" / ".." / "golden" / "x.json"
    assert "data/golden" not in hop.as_posix()
    with pytest.raises(st.RefusalError):
        st.guard_path(hop)
    # An ordinary path passes through UNCHANGED, not resolved.
    plain = Path("eval_out/seq_stage2_5seed/stats.json")
    assert st.guard_path(plain) == plain


def test_an_optional_slice_registers_a_predicate_or_is_refused(frame_dir):
    config = _config_payload(frame_dir=frame_dir)
    config["statistics"]["slice_predicates"] = {"mystery": {}}
    with pytest.raises(st.RefusalError, match="never invented"):
        st.optional_slice_specs(config)


def test_a_family_member_may_be_registered_on_an_optional_slice(
        tmp_path, frame_dir, block_source):
    """Block B's shape: a tier-restricted primary and tier-restricted gates."""
    directory = _tier_frame(frame_dir, tmp_path, "tier_frame_family")
    # The runs' signature is anchored to the pin of the frame they were trained
    # against, so this frame needs its own runs.
    runs_root = _runs_for(tmp_path / "tier_runs", directory)
    config = _config_payload(frame_dir=directory)
    config["statistics"]["slice_predicates"] = {
        "target_P": {"tier": 3}, "death@target_P": {},
        "chase@target_P": {}}
    config["statistics"]["slices"] = [
        "all", "death", "chase", "powerplay", "middle", "innings_1",
        "innings_2", "thin_pair", "target_P", "death@target_P",
        "chase@target_P"]
    config["statistics"]["families"]["map"] = [
        _general_family(
            "full",
            [_member("tier_primary", "full", "fixed_decay", "target_P",
                     primary=True),
             _member("tier_rowmatch", "full", "fox", "target_P"),
             _member("death_gate", "full", "mlp", "death@target_P",
                     "non_inferiority"),
             _member("chase_gate", "full", "mlp", "chase@target_P",
                     "non_inferiority")],
            screen={"require": ["tier_primary", "tier_rowmatch",
                                "death_gate", "chase_gate"]},
            all_row_condition="inherited"),
        _family("fixed_decay", "mlp"), _family("fox", "fixed_decay")]
    path = _write_config(tmp_path, config, "tier_family.yaml")
    payload = _run_stats(path, runs_root, frame_dir=directory,
                         block_source=block_source, tmp_path=tmp_path)
    assert "full-fixed_decay@target_P" in payload["contrasts"]
    assert "full-mlp@death@target_P" in payload["contrasts"]
    # A tier-restricted gate is still measured against the margin.
    assert payload["contrasts"]["full-mlp@death@target_P"]["threshold"] == (
        st.MARGIN_LL)
    gate_slices = {row["slice"] for row in payload["gates"]
                   if row["candidate"] == "full"}
    assert gate_slices == {"death@target_P", "chase@target_P"}
    assert payload["slices"]["stats"]["target_P"]["n_rows"] > 0


# --- mechanism contrasts ---------------------------------------------------

def test_mechanism_contrasts_fall_back_to_the_frozen_tuple(frame_dir):
    """The Stage 2 config describes its mechanism contrasts in prose and names
    pairs for only one of them, so it is NOT a complete registration and the
    frozen tuple is used — which is why Stage 2 reproduces."""
    config = _config_payload(frame_dir=frame_dir)
    config["statistics"]["contrasts"] = [
        {"name": "fox_minus_fixed_decay", "role": "mechanism",
         "tests": "learned forgetting beyond fixed decay"},
        {"name": "same_entity_minus_recency", "role": "mechanism",
         "tests": "ownership plus alignment beyond recency",
         "registered_pairs": [["same_entity_k30", "recency_k30"]]}]
    pairs, source = st.mechanism_contrasts(config)
    assert pairs == st.MECHANISM_CONTRASTS
    assert source == st.MECHANISM_SOURCE_BUILTIN

    # No mechanism entry at all also falls back.
    assert st.mechanism_contrasts(_config_payload(frame_dir=frame_dir)) == (
        st.MECHANISM_CONTRASTS, st.MECHANISM_SOURCE_BUILTIN)


def test_mechanism_contrasts_are_read_from_a_complete_registration(
        tmp_path, frame_dir, runs_root, block_source):
    config = _config_payload(frame_dir=frame_dir)
    config["statistics"]["contrasts"] = [
        {"name": "fox_minus_fixed_decay", "role": "mechanism",
         "tests": "learned forgetting beyond fixed decay",
         "registered_pairs": [["fox", "fixed_decay"]]},
        {"name": "full_minus_mlp", "role": "mechanism",
         "registered_pairs": [{"candidate": "full", "reference": "mlp",
                               "label": "the whole sequence model"}]},
        {"name": "arm_minus_mlp", "role": "primary_screen"}]
    pairs, source = st.mechanism_contrasts(config)
    assert pairs == (("fox", "fixed_decay",
                      "learned forgetting beyond fixed decay"),
                     ("full", "mlp", "the whole sequence model"))
    assert source == st.MECHANISM_SOURCE_CONFIG

    path = _write_config(tmp_path, config, "mechanism.yaml")
    payload = _run_stats(path, runs_root, frame_dir, block_source, tmp_path)
    assert [row["contrast_key"] for row in payload["mechanism_contrasts"]] == [
        "fox-fixed_decay@all", "full-mlp@all"]
    assert all(row["registered_in"] == st.MECHANISM_SOURCE_CONFIG
               for row in payload["mechanism_contrasts"])
    # `fox - fixed_decay` is also this config's `fox` family primary, and a
    # family role still wins over the mechanism role, exactly as before.
    assert payload["contrasts"]["fox-fixed_decay@all"]["role"] == (
        "family_primary")
    assert all(row["record"]["available"] is True
               for row in payload["mechanism_contrasts"])


# --- the reproduction gate -------------------------------------------------

REPRO_DIGESTS = Path(
    "research/reports/embeddings/stage2_five_seed_repro_digests.json")
LIVE_FIVE_SEED_DIR = Path("eval_out/seq_stage2_5seed")


def _canonical_digest(path: Path) -> str:
    """The document's sha256 with only the timestamp-derived fields removed.

    This is the recipe the committed digest file records, reimplemented here so
    the test does not depend on the tool it is checking.
    """
    payload = json.loads(path.read_text())
    payload.pop("generated_at_utc", None)
    # `selection_record_sha256` is the hash of the record INCLUDING its
    # timestamp, so it is timestamp-derived and removed with it.
    payload.pop("selection_record_sha256", None)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def test_the_committed_repro_digests_record_their_recipe_and_argv():
    committed = json.loads(REPRO_DIGESTS.read_text())
    assert set(committed["digests"]) == {"stats.json", "k_selection.json"}
    assert set(committed["removed_fields"]) == {"generated_at_utc",
                                               "selection_record_sha256"}
    assert all(len(value) == 64 for value in committed["digests"].values())
    # The argv is the pinned one, so the digest cannot silently describe some
    # other run.
    pin = json.loads(Path(
        "docs/sequence_track/stage2_five_seed_analysis_pin.json").read_text())
    for key in ("statistics", "ksweep"):
        assert committed["invocations"][key] == pin["invocations"][key]


@pytest.mark.parametrize("name", ["stats.json", "k_selection.json"])
def test_the_five_seed_analysis_reproduces_byte_identically(name):
    """The generalisation must not move one number of the landed analysis.

    The committed digest in `REPRO_DIGESTS` is the gate. It is checked against
    the LIVE files of record — the shipped
    `eval_out/seq_stage2_5seed/{stats,k_selection}.json` — and additionally
    against a fresh capture whenever `STAGE2_STATS_REPRO_DIR` points at one
    (a directory holding `stats_after/<name>`, optionally `stats_before/<name>`
    too). It never skips: an absent file of record is a failure, because then
    nothing is checking the reproduction.
    """
    committed = json.loads(REPRO_DIGESTS.read_text())
    expected = committed["digests"][name]

    live = Path(committed["files_of_record"][name])
    assert live == LIVE_FIVE_SEED_DIR / name
    assert live.exists(), (
        f"the five-seed file of record {live} is absent, so the reproduction "
        f"gate is unchecked. Reproduce it with the argv recorded in "
        f"{REPRO_DIGESTS} and re-run.")
    assert _canonical_digest(live) == expected, (
        f"{live} no longer reproduces the committed canonical digest "
        f"{expected}: a number of the landed five-seed analysis moved.")

    capture_dir = os.environ.get("STAGE2_STATS_REPRO_DIR")
    if capture_dir:
        checked = 0
        for sub in ("stats_before", "stats_after"):
            path = Path(capture_dir) / sub / name
            if not path.exists():
                continue
            checked += 1
            assert _canonical_digest(path) == expected, (
                f"the {sub} capture {path} does not reproduce the committed "
                f"canonical digest {expected}")
        assert checked, (
            f"STAGE2_STATS_REPRO_DIR={capture_dir} holds no "
            f"stats_before/{name} or stats_after/{name} to check")


def test_the_live_five_seed_stats_differs_from_a_capture_only_in_the_timestamp():
    """Whole-document equality, not just the digest, where a capture exists."""
    capture_dir = os.environ.get("STAGE2_STATS_REPRO_DIR")
    if not capture_dir:
        pytest.skip("no STAGE2_STATS_REPRO_DIR capture to compare against")
    capture = Path(capture_dir) / "stats_after" / "stats.json"
    if not capture.exists():
        pytest.skip(f"no capture at {capture}")
    left = json.loads((LIVE_FIVE_SEED_DIR / "stats.json").read_text())
    right = json.loads(capture.read_text())
    assert left.pop("generated_at_utc") != right.pop("generated_at_utc")
    assert left == right


# --- the direction rule on more than one member ----------------------------

def _two_superiority_screen_inputs(primary_count: int, second_count: int,
                                   direction_required=None,
                                   n_seeds: int = 5):
    """A 3a-shaped four-member family whose members all otherwise pass.

    The two superiority members differ ONLY in their favourable-direction
    counts, so the screen's verdict isolates the direction rule.
    """
    members = [
        _member("superiority_1", "full", "fixed_decay", "target_P",
                primary=True),
        _member("superiority_2", "full", "fox", "target_P"),
        _member("death_gate", "full", "mlp", "death@target_P",
                "non_inferiority"),
        _member("chase_gate", "full", "mlp", "chase@target_P",
                "non_inferiority")]
    entry = _general_family("full", members)
    if direction_required is not None:
        entry["direction_required"] = direction_required
    internal = [{"member": m["name"], "candidate": "full",
                 "reference": m["contrast"]["reference"],
                 "slice": m["slice"], "kind": m["kind"],
                 "contrast": m["name"],
                 "threshold": 0.0 if m["kind"] == "superiority" else 0.002,
                 "primary": bool(m.get("primary"))}
                for m in members]
    family = {"candidate": "full", "members": internal,
              "shared_control": "mlp",
              "screen_spec": st._family_screen_spec(entry, "full", internal,
                                                    legacy=False)}
    keys = {"superiority_1": "full-fixed_decay@target_P",
            "superiority_2": "full-fox@target_P",
            "death_gate": "full-mlp@death@target_P",
            "chase_gate": "full-mlp@chase@target_P"}
    readout = st.JOINT_READOUT
    tables = {readout: {"readout": readout, "members": [
        {"member": name, "status": "SCREEN_PASS", "rejected": True,
         "u95": (-0.01 if name.startswith("superiority") else 0.0005),
         "contrast_key": keys[name]} for name in keys]}}
    counts = {"superiority_1": primary_count, "superiority_2": second_count,
              "death_gate": 5, "chase_gate": 5}
    contrasts = {
        keys[name]: {"available": True, "n_seeds": n_seeds,
                     "favourable_direction_count": counts[name],
                     "estimand_ii": _gate_readout(-0.01, point=-0.02)}
        for name in keys}
    # The inherited extra all-row read.
    contrasts["full-mlp@all"] = {
        "available": True, "n_seeds": n_seeds,
        "favourable_direction_count": 5,
        "estimand_ii": _gate_readout(-0.01, point=-0.02)}
    return family, tables, contrasts


def test_the_direction_rule_covers_every_member_it_registers():
    """The 3a families enforce >=4/5 on BOTH superiority members."""
    required = ["superiority_1", "superiority_2"]

    # Both clear: the screen passes and reports both counts.
    family, tables, contrasts = _two_superiority_screen_inputs(
        5, 4, required)
    screen = st.family_screen(family, tables, contrasts, st.JOINT_READOUT)
    assert screen["status"] == st.STATUS_PASS
    assert screen["direction_required_members"] == required
    assert screen["direction_required_is_registered"] is True
    assert screen["all_direction_requirements_met"] is True
    counts = {name: row["favourable_direction_count"]
              for name, row in
              screen["direction_requirement_by_member"].items()}
    assert counts == {"superiority_1": 5, "superiority_2": 4}

    # The SECOND superiority member fails 4/5 while everything else passes:
    # the screen is NOT_PASS, and the per-member record says which member.
    family, tables, contrasts = _two_superiority_screen_inputs(
        5, 3, required)
    screen = st.family_screen(family, tables, contrasts, st.JOINT_READOUT)
    assert screen["status"] == st.STATUS_NOT_PASS
    assert screen["all_direction_requirements_met"] is False
    by_member = screen["direction_requirement_by_member"]
    assert by_member["superiority_1"][
        "favourable_direction_requirement_met"] is True
    assert by_member["superiority_2"][
        "favourable_direction_requirement_met"] is False
    assert by_member["superiority_2"]["favourable_direction_count"] == 3
    assert by_member["superiority_2"]["contrast_key"] == "full-fox@target_P"
    assert screen["five_seed_extension_qualified"] is False
    # Every required member still rejected; only the direction rule failed.
    assert set(screen["screen_required_member_statuses"].values()) == {
        st.STATUS_PASS}
    # And the headline keys still describe the primary, unchanged.
    assert screen["favourable_direction_count"] == 5
    assert screen["primary_status"] == st.STATUS_PASS


def test_the_direction_rule_defaults_to_the_primary_member_alone():
    """Default coverage keeps Stage 2 and the legacy form exactly as they were.

    The second superiority member fails 4/5 and the screen still passes,
    because the family did not register it as covered.
    """
    family, tables, contrasts = _two_superiority_screen_inputs(5, 0)
    screen = st.family_screen(family, tables, contrasts, st.JOINT_READOUT)
    assert screen["status"] == st.STATUS_PASS
    assert screen["direction_required_members"] == ["superiority_1"]
    assert screen["direction_required_is_registered"] is False
    assert list(screen["direction_requirement_by_member"]) == ["superiority_1"]
    # A legacy family carries none of these keys at all.
    legacy, legacy_tables, legacy_contrasts = _screen_inputs(4, 5)
    legacy_screen = st.family_screen(legacy, legacy_tables, legacy_contrasts,
                                     st.JOINT_READOUT)
    assert "direction_required_members" not in legacy_screen
    assert "direction_requirement_by_member" not in legacy_screen
    assert legacy_screen["status"] == st.STATUS_PASS


def test_direction_required_refuses_a_name_outside_the_family(frame_dir):
    members = [_member("primary", "full", "mlp", "all", primary=True),
               *_gates("full")]
    payload = _config_payload(frame_dir=frame_dir)
    payload["statistics"]["families"]["map"] = [
        _general_family("full", members,
                        direction_required=["primary", "no_such_member"]),
        _family("fixed_decay", "mlp"), _family("fox", "fixed_decay")]
    with pytest.raises(st.RefusalError, match="no_such_member"):
        st.registered_families(payload)

    # An empty list, and one that drops the primary, are refused too.
    payload["statistics"]["families"]["map"][0]["direction_required"] = []
    with pytest.raises(st.RefusalError, match="at least the primary"):
        st.registered_families(payload)
    payload["statistics"]["families"]["map"][0]["direction_required"] = [
        "death_gate"]
    with pytest.raises(st.RefusalError, match="omits its primary member"):
        st.registered_families(payload)


# --- night 3 Block B: the expectation is DERIVED from the driver -----------

NIGHT3_CONFIG_PATH = REPO / "experiments/configs/seq_stage3_night3_v1.yaml"


def test_the_night3_expected_digests_are_the_drivers_own(frame_dir):
    """MUST-FIX 1: statistics must expect what the driver actually produces.

    The stats-side expectation used to rebuild `training_block` in a fixed
    shape that had no `schedule` sub-block, so every night-3 run — whose
    configurations all set a step budget — carried a signature component no
    expectation could ever equal and was refused as incomparable. The expected
    `training_block` and `arm_params` digests are now read off the driver's
    own `effective_settings` / `signature_components` for the entry, which is
    what this locks, for all six registered configurations.
    """
    driver = st.training_driver()
    night3 = yaml.safe_load(NIGHT3_CONFIG_PATH.read_text())
    # The night-3 config is not pinned in this synthetic tree, so it borrows
    # the fixture's provenance block: the pin anchors the frame and the cache,
    # neither of which this test is about.
    fixture = _config_payload(frame_dir=frame_dir, same_entity=True)
    config = dict(night3, provenance=fixture["provenance"])
    pin = st.load_pin(config)
    assert pin.available
    anchored = st.pinned_component_digests(config, pin)
    resolved = {
        "frame_dir_configured": config["provenance"]["frame"]["dir"],
        "frame_version": config["provenance"]["frame"]["version"],
        "feature_hash": config["provenance"]["frame"]["feature_hash"],
        "split_files": {
            split: config["provenance"]["frame"]["splits"][split]
            for split in driver.CONTRACT_SPLITS},
        "stats_cache": {"role": CACHE_ROLE, "md5": CACHE_MD5},
    }
    entries = driver.configurations(config)
    assert len(entries) == 6
    for config_id, entry in entries.items():
        entry = dict(entry)
        entry["_params"] = driver._check_params(NIGHT3_CONFIG_PATH, entry)
        effective = driver.effective_settings(config, entry)
        # Every night-3 Block B arm runs the STEP schedule, which is exactly
        # the sub-block the old reconstruction dropped.
        assert effective["schedule"] == {"max_steps": 3840, "eval_every": 128}
        produced = driver.signature_components(effective, resolved)
        assert anchored["training_block_by_config"][config_id] == (
            produced["training_block"]), config_id
        assert anchored["arm_params_by_config"][config_id] == (
            produced["arm_params"]), config_id
    # One step budget across the block, so the shared digest is still one.
    assert len(set(anchored["training_block_by_config"].values())) == 1
    assert anchored["training_block"] == (
        anchored["training_block_by_config"]["mlp_pool"])


def test_a_stage2_configurations_expected_digests_are_the_drivers_own(
        frame_dir):
    """And the stage 2 shape is byte-identical to what it always produced."""
    config, pin, anchored = _anchor(frame_dir)
    driver = st.training_driver()
    for config_id, entry in driver.configurations(config).items():
        effective = st.driver_effective_settings(config, entry, pin)
        assert "schedule" not in effective, config_id
        block = {"arch": effective["arch"], "optimiser": effective["optimiser"]}
        assert anchored["training_block_by_config"][config_id] == (
            st.component_digest(block))
        assert anchored["training_block_by_config"][config_id] == (
            anchored["training_block"])


# --- MUST-FIX 2: a general family is a five-seed screen --------------------

def test_a_general_family_cannot_pass_below_five_complete_seeds():
    """A reduced-seed screen made the direction rule "not applicable" and a
    GENERAL-form family still reported SCREEN_PASS. Five complete paired
    seeds on every required member are now a precondition of a pass."""
    required = ["superiority_1", "superiority_2"]
    family, tables, contrasts = _two_superiority_screen_inputs(
        5, 4, required, n_seeds=3)
    screen = st.family_screen(family, tables, contrasts, st.JOINT_READOUT)
    assert screen["status"] == st.STATUS_NOT_PASS
    assert screen["status_reason"] == "fewer_than_five_seeds"
    assert screen["five_complete_seeds_required"] is True
    assert screen["required_members_below_five_seeds"] == sorted(
        screen["screen_required_members"])
    # The direction rule is still merely "not applicable" at three seeds; it
    # is the seed count itself that fails the family.
    assert screen["five_seed_direction_requirement_applies"] is False
    assert screen["favourable_direction_requirement_met"] is None
    assert screen["five_seed_extension_qualified"] is False

    # One short member is enough, and the payload names it.
    family, tables, contrasts = _two_superiority_screen_inputs(
        5, 4, required, n_seeds=5)
    contrasts["full-mlp@death@target_P"] = dict(
        contrasts["full-mlp@death@target_P"], n_seeds=4)
    screen = st.family_screen(family, tables, contrasts, st.JOINT_READOUT)
    assert screen["status"] == st.STATUS_NOT_PASS
    assert screen["status_reason"] == "fewer_than_five_seeds"
    assert screen["required_members_below_five_seeds"] == ["death_gate"]
    assert screen["required_member_n_seeds"]["superiority_1"] == 5

    # Five complete seeds everywhere: the screen passes exactly as before.
    family, tables, contrasts = _two_superiority_screen_inputs(
        5, 4, required, n_seeds=5)
    screen = st.family_screen(family, tables, contrasts, st.JOINT_READOUT)
    assert screen["status"] == st.STATUS_PASS
    assert screen["status_reason"] is None
    assert screen["required_members_below_five_seeds"] == []


def test_a_legacy_family_keeps_the_two_seed_screen():
    """Stage 2's two-seed screen must reproduce: the seed requirement is for
    the GENERAL form only."""
    family, tables, contrasts = _screen_inputs(None, 2)
    screen = st.family_screen(family, tables, contrasts, st.JOINT_READOUT)
    assert screen["status"] == st.STATUS_PASS
    # A legacy family's payload carries none of the five-complete-seeds keys
    # (byte-identity with the sealed stage 2 outputs).
    assert "status_reason" not in screen
    assert "five_complete_seeds_required" not in screen
    assert screen["five_seed_direction_requirement_applies"] is False
    # The legacy gates carry no contrast record at all, which would have been
    # "zero seeds" under the general rule and must not fail this family.
    assert "required_member_n_seeds" not in screen
