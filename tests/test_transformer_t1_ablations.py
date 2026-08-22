"""Causal-contract tests for the registered T1 ablation arms."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import json  # noqa: E402

import pytest  # noqa: E402

from transformer_t1 import T1Model, calibration_metrics  # noqa: E402
from run_t1_ablation import (load_predictions, match_ids,  # noqa: E402
                             paired_cluster_ci, paired_seed_cluster_ci,
                             run_logistic, save_predictions)


def _inputs():
    torch.manual_seed(11)
    feats = torch.randn(2, 6, 50)
    prev = torch.randint(0, 7, (2, 6))
    pad = torch.zeros(2, 6, dtype=torch.bool)
    return feats, prev, pad


def _logits(arm: str, feats, prev, pad):
    torch.manual_seed(19)
    model = T1Model(50, dmodel=16, layers=2, heads=4, arm=arm).eval()
    with torch.no_grad():
        return model(feats, prev, pad)[0]


def test_mlp_ignores_outcome_history():
    feats, prev, pad = _inputs()
    changed = (prev + 1) % 7
    torch.testing.assert_close(
        _logits("mlp", feats, prev, pad),
        _logits("mlp", feats, changed, pad),
    )


def test_no_history_arm_ignores_outcome_history():
    feats, prev, pad = _inputs()
    changed = (prev + 1) % 7
    torch.testing.assert_close(
        _logits("no_history", feats, prev, pad),
        _logits("no_history", feats, changed, pad),
    )


def test_attention_disabled_does_not_mix_tokens():
    feats, prev, pad = _inputs()
    changed = prev.clone()
    changed[:, 2] = (changed[:, 2] + 1) % 7
    before = _logits("no_attention", feats, prev, pad)
    after = _logits("no_attention", feats, changed, pad)
    torch.testing.assert_close(before[:, :2], after[:, :2])
    torch.testing.assert_close(before[:, 3:], after[:, 3:])
    assert not torch.allclose(before[:, 2], after[:, 2])


def test_full_attention_is_causal():
    feats, prev, pad = _inputs()
    changed = prev.clone()
    changed[:, 3] = (changed[:, 3] + 1) % 7
    before = _logits("full", feats, prev, pad)
    after = _logits("full", feats, changed, pad)
    torch.testing.assert_close(before[:, :3], after[:, :3])
    assert not torch.allclose(before[:, 3:], after[:, 3:])


def test_every_arm_is_finite_with_padding():
    feats, prev, pad = _inputs()
    pad[1, 3:] = True
    for arm in ("mlp", "no_attention", "no_history", "full"):
        logits = _logits(arm, feats, prev, pad)
        assert torch.isfinite(logits[~pad]).all(), arm


def test_calibration_metrics_perfect_predictions():
    y = np.array([0, 1, 2, 3, 4, 5])
    probs = np.eye(6, dtype=np.float32)
    metrics = calibration_metrics(probs, y)
    assert metrics == {"brier": 0.0, "confidence_ece_15": 0.0}


def test_match_ids_strip_only_innings_prefix():
    values = np.array(["1_123", "2_123", "1_compound_id"])
    assert match_ids(values).tolist() == ["123", "123", "compound_id"]


def test_paired_cluster_ci_preserves_complete_clusters():
    # Every row has the same paired delta, so any whole-match resample must
    # return the same value even though match sizes differ.
    delta = np.full(7, -0.125)
    clusters = np.array(["a", "a", "b", "b", "b", "b", "b"])
    lo, hi = paired_cluster_ci(delta, clusters, reps=100, seed=3)
    assert lo == hi == -0.125


def test_seed_cluster_ci_propagates_seed_uncertainty():
    clusters = np.array(["a", "a", "b", "b"])
    deltas = np.stack([np.full(4, -1.0), np.full(4, 1.0)])
    lo, hi = paired_seed_cluster_ci(deltas, clusters, reps=1000, seed=5)
    assert lo == -1.0
    assert hi == 1.0


def _fabricate_logistic_dir(tmp_path, converged: bool):
    arm_dir = tmp_path / "logistic"
    arm_dir.mkdir()
    for split in ("validation", "test"):
        (arm_dir / f"predictions_{split}.npz").write_bytes(b"stub")
    (arm_dir / "metrics.json").write_text(json.dumps({"converged": converged}))
    return arm_dir


def test_non_converged_logistic_is_never_silently_reused(tmp_path):
    # File existence alone used to pass the "complete; reusing" check, so a
    # max_iter-hitting control could enter aggregation. The reuse path must
    # verify the recorded convergence flag.
    _fabricate_logistic_dir(tmp_path, converged=False)
    with pytest.raises(RuntimeError, match="non-converged"):
        run_logistic({}, tmp_path)


def test_converged_logistic_artifacts_are_reused(tmp_path, capsys):
    _fabricate_logistic_dir(tmp_path, converged=True)
    run_logistic({}, tmp_path)  # config untouched on the reuse path
    assert "reusing" in capsys.readouterr().out


def test_prediction_archive_is_pickle_free(tmp_path):
    path = tmp_path / "predictions.npz"
    probs = np.eye(6, dtype=np.float32)[:2]
    save_predictions(path, probs, np.array([0, 1]),
                     np.array(["1_match", "2_match"], dtype=object))
    loaded_probs, loaded_y, loaded_ids = load_predictions(path)
    np.testing.assert_array_equal(loaded_probs, probs)
    np.testing.assert_array_equal(loaded_y, np.array([0, 1]))
    assert loaded_ids.tolist() == ["1_match", "2_match"]
