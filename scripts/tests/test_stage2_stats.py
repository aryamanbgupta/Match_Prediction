"""Tests for `scripts/sequence_track/stage2_stats.py` (D9, D10.1-D10.7, D10.11).

Synthetic fixtures only: a tiny fake frame, a tiny fake runs tree, a tiny
fake cricsheet corpus for the block lookup.  No real run, no real corpus, no
cohort, no smoke, no sealed holdout is touched.
"""
from __future__ import annotations

import ast
import builtins
import io
import json
import sys
from pathlib import Path

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


@pytest.fixture
def frame_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "frame"
    directory.mkdir()
    df = _frame_rows()
    df.to_parquet(directory / "cricket_data_i7_validation.parquet",
                  index=False)
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
                      "splits": {"validation": {
                          "path": str(parquet),
                          "md5": st.md5_file(parquet),
                          "n_rows": N_MATCHES * 2 * BALLS_PER_INNINGS}}},
            "stats_cache": {"md5": "cafe" * 8},
        }
    return payload


@pytest.fixture
def config_path(tmp_path: Path, frame_dir: Path) -> Path:
    path = tmp_path / "seq_stage2_test.yaml"
    path.write_text(yaml.safe_dump(_config_payload(frame_dir=frame_dir),
                                   sort_keys=False))
    return path


@pytest.fixture
def k_config(frame_dir: Path) -> dict:
    return _config_payload(frame_dir=frame_dir, same_entity=True)


def _components(config_id: str, frame: str = "fr", cache: str = "sc") -> dict:
    """Signature components in the driver's shape: arm-specific parts differ,
    the data and training-block parts are shared."""
    return {"config_id": st.sha256_text(config_id), "arm":
            st.sha256_text(config_id), "arm_params": st.sha256_text(config_id),
            "implementation": st.sha256_text("impl"),
            "training_block": st.sha256_text("tb"),
            "frame": st.sha256_text(frame), "stats_cache":
            st.sha256_text(cache), "base_logits": st.sha256_text("none")}


def _write_run(runs_root: Path, config_id: str, seed: int, frame: pd.DataFrame,
               *, bias: float = 0.0, innings_override=None,
               y_override=None, n_params: int = 1000,
               extra_arm_params: dict | None = None,
               arm: str | None = None, k: Any = None,
               signature: str | None = None,
               components: dict | None = None,
               record_seed: int | None = None,
               record_config_id: str | None = None,
               break_manifest: bool = False) -> Path:
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
    arm_params = {"arm": arm or config_id, "k": k, "wiring": "standard",
                  "history_input": "prev", "key_construction": "shifted",
                  "positional_embedding": True, "bias": None,
                  "residual_l2": None, "base_logits_md5": None,
                  "n_parameters": n_params}
    arm_params.update(extra_arm_params or {})
    (directory / "metrics.json").write_text(json.dumps({
        "config": {"arm": arm or config_id, "seed": seed,
                   "out": str(directory)},
        "arm_params": arm_params,
        "training_contract": {"mps_bit_reproducible": False,
                              "device": "mps"}}))
    (directory / "model.pt").write_bytes(b"not a real checkpoint")
    signature = signature or st.sha256_text(f"signature::{config_id}")
    components = components or _components(config_id)
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
        "training_signature_components": components}))
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


def _write_summary(runs_root: Path, config_id: str,
                   per_seed: dict[int, float | None],
                   signature: str | None = None) -> Path:
    path = runs_root / config_id / "summary.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    signature = signature or st.sha256_text(f"signature::{config_id}")
    rows = []
    for seed, value in per_seed.items():
        if value is None:
            continue
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
            frame_dir: Path | None = None, *, skip: tuple[str, ...] = ()
            ) -> Path:
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
                       k=(k if k == "unr" else int(k)))
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
            signature = ("c" * 64 if (config_id == "fox" and seed == 13)
                         else None)
            _write_run(root, config_id, seed, frame, signature=signature)
        _write_summary(root, config_id, {7: 1.5, 13: 1.5})
    with pytest.raises(st.RefusalError, match="different training signatures"):
        _run_stats(config_path, root, frame_dir, block_source, tmp_path)


def test_a_moved_frame_component_refuses_across_arms(tmp_path, frame_dir,
                                                     block_source,
                                                     config_path):
    root = tmp_path / "runs"
    frame = pd.read_parquet(frame_dir / "cricket_data_i7_validation.parquet")
    for config_id in CONFIG_IDS:
        for seed in SEEDS:
            components = _components(
                config_id, frame=("moved" if config_id == "fox" else "fr"))
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
