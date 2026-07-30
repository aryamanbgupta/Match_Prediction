#!/usr/bin/env python3
"""Evaluate the frozen I9 provisional-ELO development gate.

The inputs are validation-prediction parquet files emitted by xgboost_v2.py.
Rows must be paired exactly; the candidate is always subtracted from the
baseline so negative deltas are improvements.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


EXPOSURE_LIMIT = 120
GUARDRAIL_TOLERANCE = 0.001
BOOTSTRAP_SEED = 29
BOOTSTRAP_REPLICATES = 10_000

IDENTITY_COLUMNS = (
    "split_row",
    "innings_id",
    "ball_idx",
    "batter_elo_exposure",
    "bowler_elo_exposure",
    "target",
)


def _probability_columns(frame: pd.DataFrame) -> list[str]:
    columns = [
        column
        for column in frame.columns
        if column.startswith("prob_class_")
    ]
    return sorted(columns, key=lambda column: int(column.rsplit("_", 1)[1]))


def _validate_pair(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
) -> list[str]:
    missing = [
        column
        for column in IDENTITY_COLUMNS
        if column not in baseline.columns or column not in candidate.columns
    ]
    if missing:
        raise ValueError(
            f"I9 prediction artifacts are missing columns: {missing}"
        )
    if len(baseline) != len(candidate):
        raise ValueError(
            "I9 validation row-count mismatch: "
            f"baseline={len(baseline)}, candidate={len(candidate)}"
        )
    for column in IDENTITY_COLUMNS:
        left = baseline[column].to_numpy()
        right = candidate[column].to_numpy()
        if not np.array_equal(left, right):
            raise ValueError(
                f"I9 validation pairing mismatch in {column!r}"
            )

    baseline_probabilities = _probability_columns(baseline)
    candidate_probabilities = _probability_columns(candidate)
    if (
        not baseline_probabilities
        or baseline_probabilities != candidate_probabilities
    ):
        raise ValueError(
            "I9 probability-column mismatch: "
            f"baseline={baseline_probabilities}, "
            f"candidate={candidate_probabilities}"
        )
    return baseline_probabilities


def _row_log_losses(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2:
        raise ValueError("probabilities must be a two-dimensional array")
    if np.any(labels < 0) or np.any(labels >= probabilities.shape[1]):
        raise ValueError("labels are outside the probability class range")
    selected = probabilities[np.arange(len(labels)), labels]
    return -np.log(np.clip(selected, 1e-15, 1.0))


def _row_brier_scores(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    one_hot = np.zeros_like(probabilities)
    one_hot[np.arange(len(labels)), labels] = 1.0
    return np.sum((probabilities - one_hot) ** 2, axis=1)


def _match_blocks(innings_ids: Iterable[str]) -> np.ndarray:
    blocks = []
    for innings_id in innings_ids:
        text = str(innings_id)
        if "_" not in text:
            raise ValueError(f"invalid innings_id for pairing: {text!r}")
        blocks.append(text.split("_", 1)[1])
    return np.asarray(blocks, dtype=object)


def _paired_match_block_interval(
    row_deltas: np.ndarray,
    match_blocks: np.ndarray,
    *,
    seed: int = BOOTSTRAP_SEED,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> tuple[float, float]:
    row_deltas = np.asarray(row_deltas, dtype=np.float64)
    match_blocks = np.asarray(match_blocks, dtype=object)
    unique_blocks, inverse = np.unique(match_blocks, return_inverse=True)
    if len(unique_blocks) < 2:
        raise ValueError(
            "paired match-block bootstrap requires at least two matches"
        )
    block_sums = np.bincount(inverse, weights=row_deltas)
    block_counts = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, 1_000):
        stop = min(start + 1_000, replicates)
        sampled = rng.integers(
            0,
            len(unique_blocks),
            size=(stop - start, len(unique_blocks)),
        )
        estimates[start:stop] = (
            block_sums[sampled].sum(axis=1)
            / block_counts[sampled].sum(axis=1)
        )
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def evaluate_ball_gate(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    seed: int = BOOTSTRAP_SEED,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> dict:
    probability_columns = _validate_pair(baseline, candidate)
    labels = baseline["target"].to_numpy(dtype=np.int64)
    baseline_probabilities = baseline[probability_columns].to_numpy()
    candidate_probabilities = candidate[probability_columns].to_numpy()
    baseline_ll = _row_log_losses(labels, baseline_probabilities)
    candidate_ll = _row_log_losses(labels, candidate_probabilities)
    baseline_brier = _row_brier_scores(labels, baseline_probabilities)
    candidate_brier = _row_brier_scores(labels, candidate_probabilities)

    batter_exposure = baseline["batter_elo_exposure"].to_numpy()
    bowler_exposure = baseline["bowler_elo_exposure"].to_numpy()
    provisional = (
        (batter_exposure < EXPOSURE_LIMIT)
        | (bowler_exposure < EXPOSURE_LIMIT)
    )
    established = (
        (batter_exposure >= EXPOSURE_LIMIT)
        & (bowler_exposure >= EXPOSURE_LIMIT)
    )
    if not np.any(provisional):
        raise ValueError("I9 validation set contains no provisional events")
    if not np.any(established):
        raise ValueError(
            "I9 validation set contains no established-vs-established events"
        )

    provisional_delta = candidate_ll[provisional] - baseline_ll[provisional]
    interval = _paired_match_block_interval(
        provisional_delta,
        _match_blocks(
            baseline.loc[provisional, "innings_id"].astype(str)
        ),
        seed=seed,
        replicates=replicates,
    )
    overall_ll_delta = float(np.mean(candidate_ll - baseline_ll))
    overall_brier_delta = float(
        np.mean(candidate_brier - baseline_brier)
    )
    established_ll_delta = float(
        np.mean(candidate_ll[established] - baseline_ll[established])
    )

    return {
        "contract": {
            "exposure_limit": EXPOSURE_LIMIT,
            "candidate_minus_baseline": True,
            "bootstrap": {
                "unit": "match",
                "seed": seed,
                "replicates": replicates,
                "interval": 0.95,
            },
            "guardrail_tolerance": GUARDRAIL_TOLERANCE,
        },
        "counts": {
            "validation_balls": int(len(baseline)),
            "provisional_balls": int(np.sum(provisional)),
            "established_balls": int(np.sum(established)),
            "provisional_matches": int(
                len(np.unique(_match_blocks(
                    baseline.loc[provisional, "innings_id"].astype(str)
                )))
            ),
        },
        "primary": {
            "provisional_log_loss": {
                "baseline": float(np.mean(baseline_ll[provisional])),
                "candidate": float(np.mean(candidate_ll[provisional])),
                "delta": float(np.mean(provisional_delta)),
                "paired_match_block_ci95": list(interval),
                "passed": bool(interval[1] < 0.0),
            }
        },
        "guardrails": {
            "overall_log_loss": {
                "baseline": float(np.mean(baseline_ll)),
                "candidate": float(np.mean(candidate_ll)),
                "delta": overall_ll_delta,
                "passed": bool(
                    overall_ll_delta <= GUARDRAIL_TOLERANCE
                ),
            },
            "overall_brier": {
                "baseline": float(np.mean(baseline_brier)),
                "candidate": float(np.mean(candidate_brier)),
                "delta": overall_brier_delta,
                "passed": bool(
                    overall_brier_delta <= GUARDRAIL_TOLERANCE
                ),
            },
            "established_log_loss": {
                "baseline": float(np.mean(baseline_ll[established])),
                "candidate": float(np.mean(candidate_ll[established])),
                "delta": established_ll_delta,
                "passed": bool(
                    established_ll_delta <= GUARDRAIL_TOLERANCE
                ),
            },
        },
    }


def _direct_gate(pairs: list[list[str]]) -> dict:
    if not pairs:
        return {
            "status": "not_evaluated",
            "passed": False,
            "required_pairs": 5,
        }
    if len(pairs) != 5:
        raise ValueError(
            "I9 direct-model guardrail requires exactly five paired seeds"
        )
    records = []
    for baseline_path, candidate_path in pairs:
        baseline_metrics = json.loads(Path(baseline_path).read_text())
        candidate_metrics = json.loads(Path(candidate_path).read_text())
        baseline_seed = int(baseline_metrics["seed"])
        candidate_seed = int(candidate_metrics["seed"])
        if baseline_seed != candidate_seed:
            raise ValueError(
                "I9 direct-model seed mismatch: "
                f"baseline={baseline_seed}, candidate={candidate_seed}"
            )
        records.append({
            "seed": baseline_seed,
            "baseline": float(baseline_metrics["val_log_loss"]),
            "candidate": float(candidate_metrics["val_log_loss"]),
        })
    records.sort(key=lambda record: record["seed"])
    required_seeds = [7, 13, 29, 42, 101]
    actual_seeds = [record["seed"] for record in records]
    if actual_seeds != required_seeds:
        raise ValueError(
            "I9 direct-model seeds must be exactly "
            f"{required_seeds}; got {actual_seeds}"
        )
    baseline_losses = [record["baseline"] for record in records]
    candidate_losses = [record["candidate"] for record in records]
    baseline_mean = float(np.mean(baseline_losses))
    candidate_mean = float(np.mean(candidate_losses))
    return {
        "status": "evaluated",
        "seeds": actual_seeds,
        "baseline_losses": baseline_losses,
        "candidate_losses": candidate_losses,
        "baseline_mean": baseline_mean,
        "candidate_mean": candidate_mean,
        "delta": candidate_mean - baseline_mean,
        "passed": bool(candidate_mean <= baseline_mean),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the frozen I9 development gate"
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--direct-pair",
        action="append",
        nargs=2,
        metavar=("BASELINE_METRICS", "CANDIDATE_METRICS"),
        default=[],
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=BOOTSTRAP_REPLICATES,
    )
    args = parser.parse_args()

    report = evaluate_ball_gate(
        pd.read_parquet(args.baseline),
        pd.read_parquet(args.candidate),
        replicates=args.bootstrap_replicates,
    )
    report["direct_match_guardrail"] = _direct_gate(args.direct_pair)
    ball_passed = (
        report["primary"]["provisional_log_loss"]["passed"]
        and all(
            guardrail["passed"]
            for guardrail in report["guardrails"].values()
        )
    )
    direct_status = report["direct_match_guardrail"]["status"]
    report["decision"] = (
        "ELIGIBLE"
        if ball_passed and report["direct_match_guardrail"]["passed"]
        else "INCOMPLETE"
        if ball_passed and direct_status == "not_evaluated"
        else "FAILED"
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
