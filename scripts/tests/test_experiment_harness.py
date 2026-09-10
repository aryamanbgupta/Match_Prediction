"""H1--H12 contracts for the paired multi-seed experiment harness."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.preprocessing import LabelEncoder

import experiment_harness as harness
from calibrate_match_predictions import (
    _calibration_validation,
    _validate_calibration_contract,
)
from xgboost_match_v1 import (
    _apply_encoders,
    _early_stop_validation,
    _fit_encoders,
    predict_test,
)


def _frame(path: Path, n: int = 12) -> Path:
    path.mkdir(parents=True)
    rows = []
    for index in range(n):
        team1, team2 = f"A{index}", f"B{index}"
        left, right = float(index + 2), float(index)
        rows.append({
            "match_id": f"m{index}", "cricsheet_id": f"m{index}",
            "display_match_id": f"2025-01-{index + 1:02d}_{team1}_{team2}_Ground",
            "match_date": f"2025-01-{index + 1:02d}",
            "team1": team1, "team2": team2, "venue": f"V{index % 2}",
            "competition_tier": "1", "team1_wins": index % 2,
            "h2h_n_meetings": 0, "h2h_team1_win_rate_shrunk": 0.5,
            "team1_batting_elo": left, "team2_batting_elo": right,
            "elo_diff_batting": left - right,
        })
    frame = pd.DataFrame(rows)
    frame.iloc[:4].to_parquet(path / "train.parquet", index=False)
    frame.iloc[4:8].to_parquet(path / "validation.parquet", index=False)
    frame.to_parquet(path / "test.parquet", index=False)
    return path


def _config(tmp_path: Path, *, seeds=1, kind="trainer", folds=None,
            verify=None, expected=1, ceiling=240) -> Path:
    frame = _frame(tmp_path / "frame")
    payload = {
        "name": "H_TEST", "frame": str(frame),
        "baseline": {"trainer_args": []},
        "candidate": {"kind": kind, "trainer_args": []},
        "seeds": seeds, "slices": ["all", 50000, 100000],
        "cost": {"spread_bps": 0, "fee_bps": 0, "fee_basis": "winnings"},
        "expected_minutes_per_seed": expected, "ceiling_minutes": ceiling,
        "verify": verify or [], "out_dir": str(tmp_path / "out"),
    }
    if folds is not None:
        payload["folds"] = folds
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload))
    return path


def test_h1_schema_defaults_integer_seed_and_named_errors(tmp_path):
    config = harness.load_config(_config(tmp_path))
    assert config.seeds == [29]
    assert config.slices == ["all", 50000, 100000]
    bad = yaml.safe_load((tmp_path / "config.yaml").read_text())
    del bad["frame"]
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(bad))
    with pytest.raises(harness.ConfigError, match="frame: is required"):
        harness.load_config(path)


@pytest.mark.parametrize("token", [
    "--cmd", "--cm", "--seed=7", "--see", "--model-dir", "--model-d",
    "--data-dir=x", "--fit-encoders-o", "--config-json", "--config-j",
    "--early-stop-before=2025-01-01", "--early-stop-b",
])
def test_h1_rejects_exact_prefix_and_equals_harness_owned_args(tmp_path, token):
    path = _config(tmp_path)
    raw = yaml.safe_load(path.read_text())
    raw["candidate"]["trainer_args"] = [token]
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(harness.ConfigError, match="candidate.trainer_args: harness-owned"):
        harness.load_config(path)


def test_h9_integer_seed_is_count_from_default_ladder(tmp_path):
    assert harness.load_config(_config(tmp_path, seeds=3)).seeds == [29, 7, 13]


@pytest.mark.parametrize("fresh_prediction,model_present", [(False, True), (True, False)])
def test_i1_trainer_output_must_be_fresh_and_have_model(
    tmp_path, monkeypatch, fresh_prediction, model_present
):
    config = harness.load_config(_config(tmp_path))
    model_dir = tmp_path / "trainer"

    def fake_subprocess(*args, **kwargs):
        model_dir.mkdir(parents=True, exist_ok=True)
        harness._write_predictions(model_dir / "test_predictions.json", {}, {})
        if not fresh_prediction:
            os.utime(model_dir / "test_predictions.json", ns=(1, 1))
        if model_present:
            (model_dir / "model.pkl").write_bytes(b"model")

    monkeypatch.setattr(harness.subprocess, "run", fake_subprocess)
    with pytest.raises(RuntimeError, match=(
        "fresh test_predictions" if not fresh_prediction else "model.pkl"
    )):
        harness._run_trainer(config, "candidate", 29, config.frame, model_dir, False)


def test_h1_cli_run_entrypoint(tmp_path):
    path = _config(tmp_path)
    assert harness.main(["run", str(path), "--dry-run"]) == 0


def test_h2_h3_h5_h12_dry_run_resume_hash_and_gate(tmp_path, capsys):
    path = _config(tmp_path)
    config = harness.load_config(path)
    report = harness.run(config, dry_run=True)
    assert report["verdict"]["provisional"] is True
    assert report["verdict"]["decision_inputs"]["cluster_source_dir"] is not None
    assert (config.out_dir / "gate.json").exists()
    assert (config.out_dir / "harness.json").exists()
    assert "log_verdict.py verdict H_TEST" in capsys.readouterr().out
    artifact = config.out_dir / "candidate_seed29/test_predictions.json"
    sliced = config.out_dir / "candidate_seed29/sliced_50000.json"
    assert json.loads(artifact.read_text())["summary"]["config_hash"] == config.config_hash
    assert json.loads(sliced.read_text())["summary"]["model_seed"] == 29
    blended_summary = json.loads(
        (config.out_dir / "candidate_seed29/blended.json").read_text()
    )["summary"]
    completion = json.loads(
        (config.out_dir / "candidate_seed29/harness_run.json").read_text()
    )
    assert completion["arm"] == "candidate"
    assert completion["seed"] == 29
    assert completion["trainer_argv_sha256"] == harness._sha256_argv(
        completion["trainer_argv"]
    )
    assert completion["test_predictions_sha256"] == harness._sha256_file(artifact)
    assert completion["sliced_sha256"]["50000"] == harness._sha256_file(sliced)
    assert blended_summary["model_seed"] == 29
    assert blended_summary["arm"] == "candidate"
    assert blended_summary["config_hash"] == config.config_hash
    before = artifact.stat().st_mtime_ns
    harness.run(config, dry_run=True)
    assert artifact.stat().st_mtime_ns == before

    raw = yaml.safe_load(path.read_text())
    raw["candidate"]["trainer_args"] = ["--max-depth", "2"]
    path.write_text(yaml.safe_dump(raw))
    changed = harness.load_config(path)
    time.sleep(0.001)
    harness.run(changed, dry_run=True)
    assert artifact.stat().st_mtime_ns > before


def test_j1_swapped_arm_directories_and_edited_predictions_force_retrain(
    tmp_path, monkeypatch
):
    config = harness.load_config(_config(tmp_path))
    harness.run(config, dry_run=True)
    baseline = config.out_dir / "baseline_seed29"
    candidate = config.out_dir / "candidate_seed29"
    staging = config.out_dir / "swapped"
    baseline.rename(staging)
    candidate.rename(baseline)
    staging.rename(candidate)

    original = harness._stub_train
    calls = []

    def tracked(*args, **kwargs):
        calls.append((args[3], args[2]))
        return original(*args, **kwargs)

    monkeypatch.setattr(harness, "_stub_train", tracked)
    harness.run(config, dry_run=True)
    assert calls == [("baseline", 29), ("candidate", 29)]

    calls.clear()
    prediction = candidate / "test_predictions.json"
    prediction.write_text(prediction.read_text() + "\n")
    harness.run(config, dry_run=True)
    assert calls == [("candidate", 29)]


def test_j2_manifest_frame_content_mismatch_is_named(tmp_path, monkeypatch):
    frame = _frame(tmp_path / "role_frame")
    manifest = tmp_path / "MANIFEST.yaml"
    manifest.write_text(yaml.safe_dump({"artifacts": [{
        "role": "frame_role", "path": "role_frame", "kind": "dir",
        "hash": harness.md5_directory(frame), "producing_command": "test",
        "input_roles": [], "promoting_doc": "test",
    }]}))
    monkeypatch.setattr(harness, "MANIFEST_PATH", manifest)
    config_path = _config(tmp_path / "config_tree")
    raw = yaml.safe_load(config_path.read_text())
    raw["frame"] = "frame_role"
    config_path.write_text(yaml.safe_dump(raw))
    harness.load_config(config_path)

    train = pd.read_parquet(frame / "train.parquet")
    train.loc[0, "venue"] = "Tampered Ground"
    train.to_parquet(frame / "train.parquet", index=False)
    with pytest.raises(harness.ConfigError, match=(
        "frame: manifest role 'frame_role' verification MISMATCH"
    )):
        harness.load_config(config_path)


def test_i2_dry_run_then_real_run_does_not_resume_stub(tmp_path, monkeypatch):
    config = harness.load_config(_config(tmp_path))
    stale_model = config.out_dir / "candidate_seed29/model.pkl"
    stale_model.parent.mkdir(parents=True)
    stale_model.write_bytes(b"stale-real-model")
    harness.run(config, dry_run=True)
    stub_summary, _ = harness._prediction_parts(
        config.out_dir / "candidate_seed29/test_predictions.json"
    )
    assert stub_summary["stub_trainer"] is True
    assert not stale_model.exists()
    dry_registry = config.out_dir / "dry_registry.json"
    odds, clusters = harness._registry_entry(config.odds_role, dry_registry, config.out_dir)
    monkeypatch.setattr(harness, "_registry_entry", lambda *args: (odds, clusters))
    calls = []

    def fake_real(config, arm, seed, frame, model_dir, dry_run):
        assert dry_run is False
        calls.append((arm, seed))
        path = harness._stub_train(frame, model_dir, seed, arm, config.config_hash)
        summary, rows = harness._prediction_parts(path)
        summary.pop("stub_trainer")
        summary["dry_run"] = False
        harness._write_predictions(path, summary, rows)
        (model_dir / "model.pkl").write_bytes(b"model")
        return path

    monkeypatch.setattr(harness, "_run_trainer", fake_real)
    monkeypatch.setattr(
        harness.claim_gate,
        "decide",
        lambda *args, **kwargs: SimpleNamespace(
            verdict="TEST",
            as_dict=lambda: {"verdict": "TEST", "provisional": True},
        ),
    )
    harness.run(config, dry_run=False)
    assert calls == [("baseline", 29), ("candidate", 29)]


def test_i3_swapping_frame_contents_changes_resume_hash(tmp_path):
    path = _config(tmp_path)
    original = harness.load_config(path)
    harness.run(original, dry_run=True)
    artifact = original.out_dir / "candidate_seed29/test_predictions.json"
    before_mtime = artifact.stat().st_mtime_ns
    test_path = tmp_path / "frame" / "test.parquet"
    frame = pd.read_parquet(test_path)
    frame.loc[0, "venue"] = "Replacement Ground"
    frame.to_parquet(test_path, index=False)
    changed = harness.load_config(path)
    assert changed.config_hash != original.config_hash
    time.sleep(0.001)
    harness.run(changed, dry_run=True)
    assert artifact.stat().st_mtime_ns > before_mtime


def test_h4_budget_guard(tmp_path):
    config = harness.load_config(_config(tmp_path, seeds=[1, 2, 3], expected=2, ceiling=5))
    with pytest.raises(RuntimeError, match="budget guard"):
        harness.run(config, dry_run=True)
    harness.run(config, dry_run=True, allow_long=True)


def test_h5_three_seed_is_provisional(tmp_path):
    config = harness.load_config(_config(tmp_path, seeds=[1, 2, 3]))
    report = harness.run(config, dry_run=True)
    assert report["verdict"]["seed_count"] == 3
    assert report["verdict"]["provisional"] is True


def test_h6_pairing_refuses_before_gate(tmp_path):
    one = tmp_path / "one.json"
    two = tmp_path / "two.json"
    harness._write_predictions(one, {}, {"a": {}})
    harness._write_predictions(two, {}, {"b": {}})
    with pytest.raises(ValueError, match="pairing check failed"):
        harness._assert_pairing(one, two, "29")


def test_i6_evaluation_asserts_blended_ids_equal_predictions(tmp_path, monkeypatch):
    config = harness.load_config(_config(tmp_path))
    model_dir = tmp_path / "model"
    harness._write_predictions(model_dir / "test_predictions.json", {}, {
        "m1": {"match_id": "m1", "team1": "A", "team2": "B",
               "p_team1": 0.6, "p_team2": 0.4, "team1_wins": 1},
    })
    odds = tmp_path / "odds.json"
    odds.write_text(json.dumps({"matches": [{
        "match_id": "m1", "odds": {"winner": {"A": 2.0, "B": 2.0}},
        "actual_winner": "A",
    }]}))
    monkeypatch.setattr(
        harness.blend_eval_json, "blend",
        lambda *args, **kwargs: {"summary": {}, "matches": []},
    )
    with pytest.raises(AssertionError, match="blended match-id set"):
        harness._evaluate(config, model_dir, "candidate", 29, odds, tmp_path)


def test_h7_logit_ensemble_and_one_seed_estimand(tmp_path, monkeypatch):
    frame = _frame(tmp_path / "prod_frame")
    prod = tmp_path / "prod"
    harness._stub_train(frame, prod, 29, "baseline", "ignored")
    manifest = tmp_path / "MANIFEST.yaml"
    prod_hash = harness.md5_directory(prod)
    source_sha256 = harness._sha256_file(prod / "test_predictions.json")
    source_summary, _ = harness._prediction_parts(prod / "test_predictions.json")
    manifest.write_text(yaml.safe_dump({"artifacts": [{
        "role": "match_model_prod", "path": "prod", "kind": "dir",
        "hash": prod_hash, "producing_command": "never",
        "input_roles": [], "promoting_doc": "test",
    }]}))
    monkeypatch.setattr(harness, "MANIFEST_PATH", manifest)
    config_path = _config(tmp_path, seeds=[29, 7, 13, 42, 101], kind="ensemble")
    raw = yaml.safe_load(config_path.read_text())
    raw["baseline"] = {"role": "some_other_baseline"}
    config_path.write_text(yaml.safe_dump(raw))
    config = harness.load_config(config_path)
    report = harness.run(config, dry_run=True)
    assert report["verdict"]["seeds"] == ["prod_seed29"]
    assert report["verdict"]["provisional"] is True
    assert len(report["ensemble_diagnostics_only"]) == 4
    assert not list(config.out_dir.glob("baseline_seed*"))
    prod_summary, _ = harness._prediction_parts(
        config.out_dir / "baseline_prod_seed29/test_predictions.json"
    )
    assert prod_summary["model_seed"] == "prod_seed29"
    assert prod_summary["config_hash"] == prod_hash
    assert prod_summary["source_sha256"] == source_sha256
    assert prod_summary["source_summary"] == source_summary
    summary, rows = harness._prediction_parts(config.out_dir / "candidate_ensemble/test_predictions.json")
    assert summary["ensemble_method"] == "logit_mean"
    members = [harness._prediction_parts(
        config.out_dir / f"candidate_seed{seed}/test_predictions.json")[1]["m0"]["p_team1"]
        for seed in config.seeds]
    expected = 1 / (1 + np.exp(-np.mean(np.log(np.array(members) / (1 - np.array(members))))))
    assert rows["m0"]["p_team1"] == pytest.approx(expected)


def test_j3_tampered_production_predictions_are_refused(tmp_path, monkeypatch):
    frame = _frame(tmp_path / "prod_frame")
    prod = tmp_path / "prod"
    harness._stub_train(frame, prod, 29, "baseline", "ignored")
    manifest = tmp_path / "MANIFEST.yaml"
    manifest.write_text(yaml.safe_dump({"artifacts": [{
        "role": "match_model_prod", "path": "prod", "kind": "dir",
        "hash": harness.md5_directory(prod), "producing_command": "never",
        "input_roles": [], "promoting_doc": "test",
    }]}))
    monkeypatch.setattr(harness, "MANIFEST_PATH", manifest)
    config = harness.load_config(_config(tmp_path / "config_tree"))
    source = prod / "test_predictions.json"
    source.write_text(source.read_text() + "\n")
    with pytest.raises(harness.ConfigError, match=(
        "baseline.role: match_model_prod verification MISMATCH"
    )):
        harness._production_baseline(config, tmp_path / "odds", tmp_path / "clusters")


def test_h8_train_only_encoder_maps_unseen_to_minus_one():
    train = pd.DataFrame({"venue": ["Known"], "competition_tier": ["1"]})
    val = pd.DataFrame({"venue": ["New"], "competition_tier": ["2"]})
    encoders = _fit_encoders(train, val, val, fit_on="train")
    transformed = _apply_encoders(val, encoders)
    assert transformed["venue_id_encoded"].tolist() == [-1]
    assert transformed["competition_tier_encoded"].tolist() == [-1]
    assert transformed.attrs["degraded_unseen_categories"] == 2
    assert "New" not in encoders["venue"].classes_
    legacy = _fit_encoders(train, val, val, fit_on="all")
    assert "New" in legacy["venue"].classes_


def test_h8_predict_test_records_degraded_summary(tmp_path):
    frame = tmp_path / "frame"
    model_dir = tmp_path / "model"
    frame.mkdir()
    model_dir.mkdir()
    test = pd.DataFrame([{
        "match_id": "m1", "cricsheet_id": "m1", "display_match_id": "display1",
        "match_identity_version": "cricsheet_primary_v1",
        "elo_update_version": "match_elo_update_baseline_v1",
        "match_date": "2025-01-01", "team1": "A", "team2": "B",
        "venue": "Unseen", "competition_tier": "2", "team1_wins": 1,
    }])
    test.to_parquet(frame / "test.parquet", index=False)
    encoders = _fit_encoders(
        pd.DataFrame({"venue": ["Known"], "competition_tier": ["1"]}),
        fit_on="train",
    )

    class Model:
        seen = None

        def predict_proba(self, values):
            self.seen = values.copy()
            return np.array([[0.4, 0.6]])

    model = Model()
    args = SimpleNamespace(data_dir=str(frame), model_dir=str(model_dir),
                           fit_encoders_on="train", seed=29)
    path = predict_test(
        args, model, encoders,
        ["venue_id_encoded", "competition_tier_encoded"],
    )
    payload = json.loads(path.read_text())
    assert payload["summary"]["degraded_unseen_categories"] == 2
    assert model.seen.iloc[0].tolist() == [-1, -1]


def test_h9_calibration_and_early_stop_are_disjoint():
    frame = pd.DataFrame({
        "match_id": ["before", "boundary", "after"],
        "match_date": ["2025-01-01", "2025-02-01", "2025-03-01"],
    })
    early = _early_stop_validation(frame, "2025-02-01")
    calibration = _calibration_validation(frame, "2025-02-01")
    assert set(early.match_id).isdisjoint(calibration.match_id)
    assert calibration.match_id.tolist() == ["boundary", "after"]


def test_i7_calib_after_refuses_model_without_early_stop(tmp_path):
    validation = pd.DataFrame({
        "match_id": ["m1"], "match_date": ["2025-02-01"],
    })
    (tmp_path / "train_metrics.json").write_text(json.dumps({
        "early_stop_before": None, "early_stop_match_ids": [],
    }))
    with pytest.raises(ValueError, match="trained with --early-stop-before"):
        _validate_calibration_contract(tmp_path, validation, "2025-02-01")


def test_i7_calib_after_refuses_overlap_with_early_stop_ids(tmp_path):
    validation = pd.DataFrame({
        "match_id": ["m1"], "match_date": ["2025-02-01"],
    })
    (tmp_path / "train_metrics.json").write_text(json.dumps({
        "early_stop_before": "2025-02-01", "early_stop_match_ids": ["m1"],
    }))
    with pytest.raises(ValueError, match="overlap"):
        _validate_calibration_contract(tmp_path, validation, "2025-02-01")


def test_i7_calib_after_refuses_different_recorded_boundary(tmp_path):
    validation = pd.DataFrame({
        "match_id": ["m1"], "match_date": ["2025-02-01"],
    })
    (tmp_path / "train_metrics.json").write_text(json.dumps({
        "early_stop_before": "2025-01-15", "early_stop_match_ids": [],
    }))
    with pytest.raises(ValueError, match="must equal"):
        _validate_calibration_contract(tmp_path, validation, "2025-02-01")


def test_h10_fold_masks_and_means(tmp_path):
    folds = [{"train_until": "2025-01-04", "select_from": "2025-01-05",
              "select_until": "2025-01-08"}]
    config = harness.load_config(_config(tmp_path, folds=folds))
    report = harness.run(config, dry_run=True)
    assert len(report["folds"]) == 1
    assert set(report["fold_mean_log_loss"]) == {"baseline", "candidate"}
    fold_frame = config.out_dir / "fold_frames/fold0"
    assert set(pd.read_parquet(fold_frame / "train.parquet").venue) == {"V0", "V1"}
    assert pd.to_datetime(pd.read_parquet(fold_frame / "validation.parquet").match_date).min() >= pd.Timestamp("2025-01-05")


def test_h11_all_verify_hooks(tmp_path):
    config = harness.load_config(_config(
        tmp_path, verify=["diff_negation_identity", "h2h_prior_half", "swap_doubling"]
    ))
    harness.run_verify_hooks(config.frame, config.verify)


def test_encoder_helper_accepts_plain_label_encoder():
    encoder = LabelEncoder().fit(["a"])
    result = _apply_encoders(pd.DataFrame({"venue": ["b"]}), {"venue": encoder})
    assert result.venue_id_encoded.iloc[0] == -1
