"""Regression tests for the frozen-gated M7 forward prediction adapter."""

from __future__ import annotations

import builtins
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from forward_eval_contract import load_protocol  # noqa: E402
import score_forward_match_m7 as scorer_module  # noqa: E402
from score_forward_match_m7 import (  # noqa: E402
    FORBIDDEN_INPUT_COLUMNS,
    build_prediction_artifact,
    encode_and_predict,
    load_selected_feature_rows,
    score,
    write_locked_artifact,
)


PROTOCOL_PATH = (
    ROOT / "evaluation" / "forward_protocol_2026-06-01_2026-07-13.yaml"
)


class _MockModel:
    def predict_proba(self, frame):
        values = np.asarray(frame["signal"], dtype=float)
        return np.column_stack([1.0 - values, values])


def _encoder(values):
    encoder = LabelEncoder()
    encoder.fit(values)
    return encoder


def _synthetic_inputs():
    frame = pd.DataFrame([
        {
            "match_id": "2026-01-01_A_B_Ground",
            "cricsheet_id": "1",
            "match_date": "2026-01-01",
            "team1": "A",
            "team2": "B",
            "venue": "Known",
            "competition_tier": "2",
            "signal": 0.7,
            "top6_batting_elo_diff": 4.0,
        },
        {
            "match_id": "2026-01-01_D_C_Other",
            "cricsheet_id": "2",
            "match_date": "2026-01-01",
            "team1": "D",
            "team2": "C",
            "venue": "Unseen",
            "competition_tier": "3",
            "signal": 0.2,
            "top6_batting_elo_diff": -8.0,
        },
    ])
    ordered = [
        {
            "match_id": "2026-01-01_A_B_Ground",
            "cricsheet_id": "1",
            "date": "2026-01-01",
            "teams": ["A", "B"],
        },
        {
            "match_id": "2026-01-01_D_C_Other",
            "cricsheet_id": "2",
            "date": "2026-01-01",
            "teams": ["D", "C"],
        },
    ]
    return frame, ordered


def test_actual_input_selection_loads_no_outcomes_and_exact_holdout():
    protocol = load_protocol(PROTOCOL_PATH)
    feature_path = (
        ROOT / "models" / "xgb_match_v3_m7_production"
        / "feature_columns.txt"
    )
    feature_columns = [
        line for line in feature_path.read_text().splitlines() if line
    ]
    frame, ordered = load_selected_feature_rows(protocol, feature_columns)
    assert len(frame) == len(ordered) == 137
    assert not FORBIDDEN_INPUT_COLUMNS.intersection(frame.columns)
    assert list(frame["cricsheet_id"]) == [
        str(row["cricsheet_id"]) for row in ordered
    ]
    assert [
        (row["date"], row["match_id"]) for row in ordered
    ] == sorted((row["date"], row["match_id"]) for row in ordered)


def test_mock_prediction_preserves_team_order_and_handles_unseen_category():
    frame, _ = _synthetic_inputs()
    encoders = {
        "venue": _encoder(["Known", "Fallback"]),
        "competition_tier": _encoder(["2", "3"]),
    }
    probabilities, warnings = encode_and_predict(
        frame,
        ["signal", "venue_id_encoded", "competition_tier_encoded"],
        encoders,
        _MockModel(),
    )
    assert probabilities == [0.7, 0.2]
    assert warnings == {"venue": ["Unseen"]}


def test_artifact_is_outcome_free_and_team_probabilities_sum_to_one():
    protocol = load_protocol(PROTOCOL_PATH)
    frame, ordered = _synthetic_inputs()
    gate = {
        "protocol_sha256": "a" * 64,
        "holdout_fingerprint_sha256": "b" * 64,
        "state_fingerprint_sha256": "c" * 64,
    }
    artifact = build_prediction_artifact(
        protocol,
        gate,
        frame,
        ordered,
        [0.7, 0.2],
        {"venue": ["Unseen"]},
    )
    def nested_keys(value):
        if isinstance(value, dict):
            return set(value).union(*(
                nested_keys(child) for child in value.values()
            ))
        if isinstance(value, list):
            return set().union(*(nested_keys(child) for child in value))
        return set()

    keys = nested_keys(artifact)
    assert "team1_wins" not in keys
    assert "actual_winner" not in keys
    assert artifact["outcome_columns_loaded"] is False
    assert artifact["outcomes_joined"] is False
    for row in artifact["predictions"]:
        assert row["p_team1"] + row["p_team2"] == pytest.approx(1.0)
    assert artifact["predictions"][1]["team1"] == "D"
    assert artifact["predictions"][1]["team2"] == "C"


def test_locked_artifact_is_write_once(tmp_path: Path):
    artifact = {
        "artifact_type": "locked_outcome_free_predictions",
        "predictions": [{"match_id": "m", "p_team1": 0.6, "p_team2": 0.4}],
    }
    output = tmp_path / "predictions.json"
    digest = write_locked_artifact(output, artifact)
    assert len(digest) == 64
    assert output.is_file()
    assert output.with_suffix(".json.sha256").is_file()
    with pytest.raises(FileExistsError):
        write_locked_artifact(output, artifact)


def test_preflight_gate_blocks_before_joblib_import(
    tmp_path: Path,
    monkeypatch,
):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "joblib":
            raise AssertionError("joblib imported before frozen authorization")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    def _blocked_preflight(_path, *, require_frozen):
        assert require_frozen is True
        raise RuntimeError("synthetic preflight block")

    monkeypatch.setattr(
        scorer_module,
        "preflight",
        _blocked_preflight,
    )
    with pytest.raises(RuntimeError, match="synthetic preflight block"):
        score(PROTOCOL_PATH, tmp_path / "must_not_exist.json")
    assert not (tmp_path / "must_not_exist.json").exists()
