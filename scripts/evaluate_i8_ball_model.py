#!/usr/bin/env python3
"""Paired ball-level comparison for the frozen I7 and candidate I8 models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from calibration import _apply_encoders_to_df


ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = ("dot", "one", "two", "four", "six", "wicket")
TARGET_MAP = {0: 0, 1: 1, 2: 2, 4: 3, 6: 4, 7: 5, -1: 5}
IDENTITY_COLUMNS = (
    "innings_id",
    "over_idx",
    "ball_idx",
    "batter_id",
    "bowler_id",
    "ball_outcome",
)


def _artifact_paths(version: str) -> dict[str, Path]:
    model_dir = ROOT / "models" / f"xgb_{version}"
    data_dir = ROOT / "data" / f"xgb_data_{version}"
    return {
        "model_dir": model_dir,
        "data_dir": data_dir,
        "model": model_dir / f"xgboost_model_{version}.pkl",
        "features": model_dir / f"feature_columns_{version}.txt",
    }


def _load_predictions(version: str, split: str) -> dict:
    paths = _artifact_paths(version)
    parquet = (
        paths["data_dir"] / f"cricket_data_{version}_{split}.parquet"
    )
    print(f"Loading {version}/{split}: {parquet}", flush=True)
    df = pd.read_parquet(parquet)
    feature_columns = [
        line.strip()
        for line in paths["features"].read_text().splitlines()
        if line.strip()
    ]

    identity_hash = pd.util.hash_pandas_object(
        df[list(IDENTITY_COLUMNS)],
        index=False,
    ).to_numpy(dtype=np.uint64)
    match_ids = (
        df["innings_id"].astype(str).str.split("_", n=1).str[-1]
        .to_numpy()
    )
    labels = df["ball_outcome"].map(TARGET_MAP)
    if labels.isna().any():
        unknown = sorted(df.loc[labels.isna(), "ball_outcome"].unique())
        raise RuntimeError(f"{version}/{split} has unknown targets: {unknown}")
    labels_array = labels.to_numpy(dtype=np.int8)

    _apply_encoders_to_df(
        df,
        feature_columns,
        encoder_dir=str(paths["model_dir"]),
    )
    missing = sorted(set(feature_columns) - set(df.columns))
    if missing:
        raise RuntimeError(
            f"{version}/{split} is missing model features: {missing}"
        )

    model = joblib.load(paths["model"])
    probabilities = model.predict_proba(df[feature_columns])
    if probabilities.shape != (len(df), len(CLASS_NAMES)):
        raise RuntimeError(
            f"{version}/{split} probability shape is "
            f"{probabilities.shape}, expected {(len(df), len(CLASS_NAMES))}"
        )

    return {
        "labels": labels_array,
        "probabilities": probabilities,
        "identity_hash": identity_hash,
        "match_ids": match_ids,
        "n_features": len(feature_columns),
    }


def _metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    one_hot = np.eye(len(CLASS_NAMES), dtype=float)[labels]
    predicted = probabilities.argmax(axis=1)
    calibration = {}
    for index, name in enumerate(CLASS_NAMES):
        actual_rate = float(np.mean(labels == index))
        mean_probability = float(probabilities[:, index].mean())
        calibration[name] = {
            "actual_rate": actual_rate,
            "mean_probability": mean_probability,
            "gap": mean_probability - actual_rate,
            "one_vs_rest_brier": float(np.mean(
                (probabilities[:, index] - (labels == index)) ** 2
            )),
        }
    return {
        "n_balls": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predicted)),
        "log_loss": float(log_loss(
            labels,
            probabilities,
            labels=np.arange(len(CLASS_NAMES)),
        )),
        # Unscaled multiclass Brier: range [0, 2], lower is better.
        "multiclass_brier": float(np.mean(np.sum(
            (probabilities - one_hot) ** 2,
            axis=1,
        ))),
        "calibration": calibration,
    }


def _paired_deltas(
    labels: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    match_ids: np.ndarray,
    *,
    n_resamples: int,
    seed: int,
) -> dict:
    row_index = np.arange(len(labels))
    base_log = -np.log(np.clip(baseline[row_index, labels], 1e-15, 1.0))
    cand_log = -np.log(np.clip(candidate[row_index, labels], 1e-15, 1.0))
    one_hot = np.eye(len(CLASS_NAMES), dtype=float)[labels]
    base_brier = np.sum((baseline - one_hot) ** 2, axis=1)
    cand_brier = np.sum((candidate - one_hot) ** 2, axis=1)

    codes, unique_matches = pd.factorize(match_ids, sort=True)
    n_matches = len(unique_matches)

    def summarize(delta: np.ndarray) -> dict:
        sums = np.bincount(codes, weights=delta, minlength=n_matches)
        counts = np.bincount(codes, minlength=n_matches)
        rng = np.random.default_rng(seed)
        boot = np.empty(n_resamples, dtype=float)
        for index in range(n_resamples):
            sampled = rng.integers(0, n_matches, size=n_matches)
            boot[index] = sums[sampled].sum() / counts[sampled].sum()
        return {
            "candidate_minus_baseline": float(delta.mean()),
            "match_cluster_bootstrap_95_ci": [
                float(np.quantile(boot, 0.025)),
                float(np.quantile(boot, 0.975)),
            ],
        }

    return {
        "n_matches": int(n_matches),
        "n_resamples": int(n_resamples),
        "seed": int(seed),
        "log_loss": summarize(cand_log - base_log),
        "multiclass_brier": summarize(cand_brier - base_brier),
    }


def evaluate_split(
    baseline_version: str,
    candidate_version: str,
    split: str,
    *,
    n_resamples: int,
    seed: int,
) -> dict:
    baseline = _load_predictions(baseline_version, split)
    candidate = _load_predictions(candidate_version, split)

    if not np.array_equal(
        baseline["identity_hash"],
        candidate["identity_hash"],
    ):
        raise RuntimeError(
            f"{baseline_version}/{candidate_version} {split} rows are not "
            "identical and ordered; paired comparison is invalid"
        )
    if not np.array_equal(baseline["labels"], candidate["labels"]):
        raise RuntimeError(f"{split} label arrays differ")

    return {
        "baseline": {
            "version": baseline_version,
            "n_features": baseline["n_features"],
            **_metrics(baseline["labels"], baseline["probabilities"]),
        },
        "candidate": {
            "version": candidate_version,
            "n_features": candidate["n_features"],
            **_metrics(candidate["labels"], candidate["probabilities"]),
        },
        "paired_delta": _paired_deltas(
            baseline["labels"],
            baseline["probabilities"],
            candidate["probabilities"],
            baseline["match_ids"],
            n_resamples=n_resamples,
            seed=seed,
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare candidate I8 and baseline I7 ball models"
    )
    parser.add_argument("--baseline", default="i7")
    parser.add_argument("--candidate", default="i8")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=("validation", "test"),
        choices=("validation", "test"),
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "reports" / "i8_ball_evaluation.json",
    )
    args = parser.parse_args()

    report = {
        "contract": {
            "baseline": args.baseline,
            "candidate": args.candidate,
            "splits": list(args.splits),
            "paired_rows_required": True,
            "bootstrap_unit": "match",
            "multiclass_brier_definition": (
                "mean sum_c (p_c - y_c)^2; unscaled range [0,2]"
            ),
        },
        "results": {},
    }
    for split in args.splits:
        report["results"][split] = evaluate_split(
            args.baseline,
            args.candidate,
            split,
            n_resamples=args.bootstrap_resamples,
            seed=args.seed,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
