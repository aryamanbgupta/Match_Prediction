#!/usr/bin/env python3
"""Create a locked, outcome-free M7 prediction artifact.

The frozen-protocol preflight runs before importing joblib or loading a model.
On the current DRAFT protocol this command must fail without scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent.parent

from forward_eval_contract import (  # noqa: E402
    load_protocol,
    preflight,
    repo_path,
)


SCHEMA_VERSION = 1
FORBIDDEN_INPUT_COLUMNS = frozenset({
    "team1_wins",
    "actual_winner",
    "winner",
    "outcome",
    "result",
    "target",
    "label",
})
FORBIDDEN_OUTPUT_KEYS = FORBIDDEN_INPUT_COLUMNS | frozenset({
    "innings",
    "deliveries",
})
METADATA_COLUMNS = (
    "match_id",
    "cricsheet_id",
    "match_date",
    "team1",
    "team2",
    "venue",
    "competition_tier",
)


def _candidate_artifacts(
    protocol: dict[str, Any],
    candidate: str,
) -> dict[str, dict[str, str]]:
    rows = protocol["candidates"][candidate]["artifacts"]
    return {Path(row["path"]).name: row for row in rows}


def ordered_holdout_rows(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    holdout_dir = repo_path(protocol["holdout"]["directory"])
    manifest = json.loads((holdout_dir / "manifest.json").read_text())
    rows = list(manifest.get("matches") or [])
    rows.sort(key=lambda row: (str(row["date"]), str(row["match_id"])))
    if len(rows) != protocol["holdout"]["selected_matches"]:
        raise RuntimeError("ordered holdout row count differs from protocol")
    ids = [str(row["cricsheet_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate Cricsheet ID in ordered holdout rows")
    return rows


def load_selected_feature_rows(
    protocol: dict[str, Any],
    feature_columns: Iterable[str],
):
    """Load only prediction inputs—never the target—from the sidecar parquet."""
    import pandas as pd

    feature_columns = list(feature_columns)
    forbidden = FORBIDDEN_INPUT_COLUMNS.intersection(feature_columns)
    if forbidden:
        raise RuntimeError(
            f"model feature list contains forbidden outcome columns: "
            f"{sorted(forbidden)}"
        )
    raw_feature_columns = [
        column for column in feature_columns
        if not column.endswith("_encoded")
    ]
    read_columns = list(dict.fromkeys(
        list(METADATA_COLUMNS) + raw_feature_columns
    ))
    if FORBIDDEN_INPUT_COLUMNS.intersection(read_columns):
        raise RuntimeError("forward feature read requested an outcome column")

    state_dir = repo_path(protocol["state"]["directory"])
    parquet_path = state_dir / "match_features" / "golden_test.parquet"
    frame = pd.read_parquet(parquet_path, columns=read_columns)
    if FORBIDDEN_INPUT_COLUMNS.intersection(frame.columns):
        raise RuntimeError("outcome column entered the forward prediction frame")
    frame = frame.copy()
    frame["cricsheet_id"] = frame["cricsheet_id"].astype(str)
    if frame["cricsheet_id"].duplicated().any():
        raise RuntimeError("duplicate Cricsheet ID in sidecar feature rows")

    ordered = ordered_holdout_rows(protocol)
    by_id = frame.set_index("cricsheet_id", drop=False)
    selected_ids = [str(row["cricsheet_id"]) for row in ordered]
    missing = [match_id for match_id in selected_ids if match_id not in by_id.index]
    if missing:
        raise RuntimeError(f"holdout feature rows missing: {missing[:20]}")
    selected = by_id.loc[selected_ids].reset_index(drop=True)
    if len(selected) != len(ordered):
        raise RuntimeError("selected feature row count differs from holdout")

    for index, (manifest_row, feature_row) in enumerate(
        zip(ordered, selected.to_dict("records"))
    ):
        expected = (
            str(manifest_row["match_id"]),
            tuple(map(str, manifest_row["teams"])),
        )
        actual = (
            str(feature_row["match_id"]),
            (str(feature_row["team1"]), str(feature_row["team2"])),
        )
        if actual != expected:
            raise RuntimeError(
                f"team/order alignment failed at row {index}: "
                f"{actual!r} != {expected!r}"
            )
    return selected, ordered


def encode_and_predict(
    frame,
    feature_columns: list[str],
    encoders: dict[str, Any],
    model: Any,
) -> tuple[list[float], dict[str, list[str]]]:
    """Apply frozen encoders and return P(team1), with no metric computation."""
    import numpy as np

    encoded = frame.copy()
    warnings: dict[str, list[str]] = {}
    for column, encoder in encoders.items():
        if column not in encoded.columns:
            raise RuntimeError(f"encoder input column is missing: {column}")
        encoded_column = (
            f"{column}_id_encoded"
            if column == "venue"
            else f"{column}_encoded"
        )
        known = set(map(str, encoder.classes_))
        values = encoded[column].astype(str)
        unseen = sorted(set(values) - known)
        if unseen:
            warnings[column] = unseen
            fallback = str(encoder.classes_[0])
            values = values.map(lambda value: value if value in known else fallback)
        encoded[encoded_column] = encoder.transform(values)

    missing_features = [
        column for column in feature_columns if column not in encoded.columns
    ]
    if missing_features:
        raise RuntimeError(f"encoded feature columns missing: {missing_features}")
    probabilities = np.asarray(
        model.predict_proba(encoded[feature_columns]),
        dtype=float,
    )
    if probabilities.shape != (len(encoded), 2):
        raise RuntimeError(
            f"predict_proba shape {probabilities.shape} is not "
            f"({len(encoded)}, 2)"
        )
    p_team1 = probabilities[:, 1]
    if not np.isfinite(p_team1).all():
        raise RuntimeError("non-finite M7 probability")
    if ((p_team1 < 0.0) | (p_team1 > 1.0)).any():
        raise RuntimeError("M7 probability outside [0, 1]")
    return p_team1.tolist(), warnings


def _assert_outcome_free(value: Any, path: str = "$") -> None:
    if isinstance(value, dict):
        forbidden = FORBIDDEN_OUTPUT_KEYS.intersection(value)
        if forbidden:
            raise RuntimeError(
                f"outcome-bearing keys in prediction artifact at {path}: "
                f"{sorted(forbidden)}"
            )
        for key, child in value.items():
            _assert_outcome_free(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_outcome_free(child, f"{path}[{index}]")


def build_prediction_artifact(
    protocol: dict[str, Any],
    preflight_report: dict[str, Any],
    frame,
    ordered_rows: list[dict[str, Any]],
    probabilities: list[float],
    encoder_warnings: dict[str, list[str]],
) -> dict[str, Any]:
    if len(frame) != len(ordered_rows) or len(frame) != len(probabilities):
        raise RuntimeError("prediction artifact inputs have different lengths")
    predictions = []
    records = frame.to_dict("records")
    for manifest_row, feature_row, p_team1 in zip(
        ordered_rows,
        records,
        probabilities,
    ):
        p_team1 = float(p_team1)
        p_team2 = 1.0 - p_team1
        predictions.append({
            "match_id": str(manifest_row["match_id"]),
            "cricsheet_id": str(manifest_row["cricsheet_id"]),
            **({
                "display_match_id": str(manifest_row["display_match_id"]),
                "match_identity_version": str(
                    manifest_row["match_identity_version"]
                ),
            } if manifest_row.get("display_match_id") else {}),
            "date": str(manifest_row["date"]),
            "team1": str(feature_row["team1"]),
            "team2": str(feature_row["team2"]),
            "p_team1": p_team1,
            "p_team2": p_team2,
            "top6_batting_elo_diff": float(
                feature_row["top6_batting_elo_diff"]
            ),
        })
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "locked_outcome_free_predictions",
        "model_id": "match_m7",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": preflight_report["protocol_sha256"],
        "holdout_fingerprint_sha256": preflight_report[
            "holdout_fingerprint_sha256"
        ],
        "state_fingerprint_sha256": preflight_report[
            "state_fingerprint_sha256"
        ],
        "prediction_count": len(predictions),
        "probability_semantics": "p_team1_wins",
        "calibration": "none",
        "outcome_columns_loaded": False,
        "outcomes_joined": False,
        "model_artifacts": protocol["candidates"]["match_m7"]["artifacts"],
        "encoder_warnings": encoder_warnings,
        "predictions": predictions,
    }
    _assert_outcome_free(artifact)
    return artifact


def write_locked_artifact(path: Path, artifact: dict[str, Any]) -> str:
    """Write JSON plus a SHA-256 sidecar exactly once."""
    _assert_outcome_free(artifact)
    path = path.resolve()
    digest_path = path.with_suffix(path.suffix + ".sha256")
    if path.exists() or digest_path.exists():
        raise FileExistsError(
            f"locked prediction artifact already exists: {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    digest = hashlib.sha256(payload).hexdigest()
    with path.open("xb") as handle:
        handle.write(payload)
    try:
        with digest_path.open("x") as handle:
            handle.write(f"{digest}  {path.name}\n")
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return digest


def score(protocol_path: Path, output_path: Path) -> dict[str, Any]:
    """Score M7 only after frozen authorization; never compute outcomes."""
    gate = preflight(protocol_path, require_frozen=True)
    protocol = load_protocol(protocol_path)

    # The fail-closed gate above intentionally precedes model-library imports.
    import joblib

    artifacts = _candidate_artifacts(protocol, "match_m7")
    feature_path = repo_path(artifacts["feature_columns.txt"]["path"])
    feature_columns = [
        line.strip() for line in feature_path.read_text().splitlines()
        if line.strip()
    ]
    frame, ordered_rows = load_selected_feature_rows(protocol, feature_columns)
    model = joblib.load(repo_path(artifacts["model.pkl"]["path"]))
    encoders = joblib.load(repo_path(artifacts["encoders.pkl"]["path"]))
    probabilities, warnings = encode_and_predict(
        frame,
        feature_columns,
        encoders,
        model,
    )
    artifact = build_prediction_artifact(
        protocol,
        gate,
        frame,
        ordered_rows,
        probabilities,
        warnings,
    )
    digest = write_locked_artifact(output_path, artifact)
    return {
        "status": "LOCKED",
        "model_id": "match_m7",
        "prediction_count": len(probabilities),
        "output": str(output_path.resolve()),
        "sha256": digest,
        "outcomes_joined": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(score(args.protocol, args.out), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
