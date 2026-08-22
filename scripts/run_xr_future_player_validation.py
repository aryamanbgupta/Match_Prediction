#!/usr/bin/env python3
"""Future player validation for staged xR process metrics.

For each batter and fixed N, the first N eligible deliveries are the prior
window and the next H deliveries from strictly later matches are the target.
The validation split tunes only shrinkage constants; test results are reported
once using those frozen constants. Golden and forward holdouts are refused.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from registered_experiment import match_ids, reject_sealed, sha256  # noqa: E402

DEFAULT_CONFIG = Path(
    "experiments/configs/xr_future_player_validation_v1.yaml")
METHODS = ("raw", "eb", "raa", "delta", "xr")


def load_rows(split: str, config: dict) -> pd.DataFrame:
    data_path = Path(config["data"]["directory"]) / (
        f"cricket_data_v3_{split}.parquet")
    process_path = Path(config["data"]["process_directory"]) / (
        f"process_metrics_{split}.npz")
    reject_sealed(data_path)
    reject_sealed(process_path)
    # Early same-cohort artifacts saved innings_id as NumPy object strings.
    # They are local, self-generated files; every loaded identifier and target
    # is exhaustively checked against parquet immediately below.
    process = np.load(process_path, allow_pickle=True)
    row_idx = process["row_idx"].astype(np.int64)
    source = pd.read_parquet(
        data_path,
        columns=["innings_id", "match_date", "over_idx", "ball_idx",
                 "batter_id", "team_runs"],
    )
    frame = source.iloc[row_idx].copy().reset_index(drop=True)
    if not np.array_equal(frame["team_runs"].to_numpy(),
                          process["actual_team_runs"]):
        raise ValueError(f"team-run alignment failed for {split}")
    if not np.array_equal(frame["innings_id"].astype(str).to_numpy(),
                          process["innings_id"].astype(str)):
        raise ValueError(f"innings alignment failed for {split}")
    frame["match_id"] = match_ids(frame["innings_id"].to_numpy())
    frame["actual"] = process["actual_team_runs"].astype(float)
    frame["context"] = process["xr_context"].astype(float)
    frame["shot"] = process["xr_shot"].astype(float)
    frame["context_residual"] = frame["actual"] - frame["context"]
    frame["execution_residual"] = frame["actual"] - frame["shot"]
    frame["match_date"] = frame["match_date"].astype(str)
    frame["_row_order"] = np.arange(len(frame), dtype=np.int64)
    return frame.sort_values(
        ["match_date", "match_id", "innings_id", "over_idx", "ball_idx",
         "_row_order"], kind="stable").reset_index(drop=True)


def build_examples(frame: pd.DataFrame, sample_sizes: list[int],
                   horizon: int) -> dict[int, pd.DataFrame]:
    output = {n: [] for n in sample_sizes}
    for player, player_rows in frame.groupby("batter_id", sort=False):
        matches = [group for _, group in player_rows.groupby(
            ["match_date", "match_id"], sort=False)]
        # Accumulate prior matches as parts and materialize only when a
        # window fires — the old per-match pd.concat was quadratic per player.
        prior_parts: list[pd.DataFrame] = []
        prior_len = 0
        crossed = set()
        for match_index, match in enumerate(matches):
            if prior_len:
                for n in sample_sizes:
                    if n in crossed or prior_len < n:
                        continue
                    future_parts, future_count = [], 0
                    for later in matches[match_index:]:
                        future_parts.append(later)
                        future_count += len(later)
                        if future_count >= horizon:
                            break
                    if future_count < horizon:
                        continue
                    history = pd.concat(
                        prior_parts, ignore_index=True).iloc[:n]
                    future = pd.concat(future_parts, ignore_index=True).iloc[
                        :horizon]
                    output[n].append({
                        "player_id": str(player), "n_prior": n,
                        "cutoff_date": str(match.iloc[0]["match_date"]),
                        "future_n": int(len(future)),
                        "future_actual": float(future["actual"].mean()),
                        "future_context_residual": float(
                            future["context_residual"].mean()),
                        "future_execution_residual": float(
                            future["execution_residual"].mean()),
                        "prior_actual": float(history["actual"].mean()),
                        "prior_context": float(history["context"].mean()),
                        "prior_shot": float(history["shot"].mean()),
                        "prior_context_residual": float(
                            history["context_residual"].mean()),
                        "prior_execution_residual": float(
                            history["execution_residual"].mean()),
                    })
                    crossed.add(n)
            prior_parts.append(match)
            prior_len += len(match)
    return {n: pd.DataFrame(rows) for n, rows in output.items()}


def estimates(frame: pd.DataFrame, k: float, mean_actual: float) -> dict:
    weight = frame["n_prior"].to_numpy(float) / (
        frame["n_prior"].to_numpy(float) + k)
    return {
        "raw": frame["prior_actual"].to_numpy(float),
        "eb": mean_actual + weight * (
            frame["prior_actual"].to_numpy(float) - mean_actual),
        "raa": frame["prior_context"].to_numpy(float) + weight * (
            frame["prior_context_residual"].to_numpy(float)),
        "delta": frame["prior_shot"].to_numpy(float) + weight * (
            frame["prior_execution_residual"].to_numpy(float)),
        "xr": frame["prior_shot"].to_numpy(float),
    }


def tune_k(examples: dict[int, pd.DataFrame], grid: list[float],
           mean_actual: float) -> dict[str, float]:
    tuned = {}
    for method in ("eb", "raa", "delta"):
        scores = []
        for k in grid:
            errors = []
            for frame in examples.values():
                if len(frame):
                    pred = estimates(frame, k, mean_actual)[method]
                    errors.extend(np.abs(
                        pred - frame["future_actual"].to_numpy(float)))
            scores.append((float(np.mean(errors)), float(k)))
        tuned[method] = min(scores)[1]
    return tuned


def bootstrap_delta(delta: np.ndarray, reps: int, seed: int) -> list[float]:
    if not len(delta):
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    stats = np.empty(reps)
    for index in range(reps):
        sampled = rng.integers(0, len(delta), size=len(delta))
        stats[index] = delta[sampled].mean()
    return [float(np.percentile(stats, 2.5)),
            float(np.percentile(stats, 97.5))]


def spearman(predicted: np.ndarray, actual: np.ndarray) -> float:
    if len(predicted) < 3:
        return float("nan")
    return float(pd.Series(predicted).rank().corr(pd.Series(actual).rank()))


def evaluate(examples: dict[int, pd.DataFrame], tuned: dict[str, float],
             mean_actual: float, config: dict) -> dict:
    reps = int(config["metrics"]["bootstrap_repetitions"])
    seed = int(config["metrics"]["bootstrap_seed"])
    output = {}
    for n, frame in examples.items():
        if not len(frame):
            output[str(n)] = {"n_players": 0}
            continue
        prediction = estimates(frame, tuned["delta"], mean_actual)
        prediction["eb"] = estimates(frame, tuned["eb"], mean_actual)["eb"]
        prediction["raa"] = estimates(frame, tuned["raa"], mean_actual)["raa"]
        actual = frame["future_actual"].to_numpy(float)
        absolute = {name: np.abs(value - actual)
                    for name, value in prediction.items()}
        result = {
            "n_players": int(len(frame)),
            "future_deliveries_per_player": int(frame["future_n"].iloc[0]),
            "methods": {
                name: {
                    "player_equal_weight_mae": float(error.mean()),
                    "delivery_weighted_mae": float(np.average(
                        error, weights=frame["future_n"].to_numpy(float))),
                    "player_spearman": spearman(prediction[name], actual),
                    "bias": float((prediction[name] - actual).mean()),
                }
                for name, error in absolute.items()
            },
            "paired_deltas": {},
            "residual_persistence": {
                "context_residual_spearman": spearman(
                    frame["prior_context_residual"].to_numpy(float),
                    frame["future_context_residual"].to_numpy(float)),
                "execution_residual_spearman": spearman(
                    frame["prior_execution_residual"].to_numpy(float),
                    frame["future_execution_residual"].to_numpy(float)),
            },
        }
        for name in ("raw", "raa", "delta", "xr"):
            delta = absolute[name] - absolute["eb"]
            result["paired_deltas"][f"{name}_minus_eb_mae"] = {
                "mean": float(delta.mean()),
                "player_bootstrap_ci95": bootstrap_delta(
                    delta, reps, seed + n),
            }
        output[str(n)] = result
    return output


def markdown(summary: dict) -> str:
    lines = ["# Future player xR validation v1", "",
             "Prior windows use the first N eligible deliveries for a batter; "
             "targets use the next 100 deliveries from strictly later matches. "
             "Shrinkage constants were selected on validation and frozen for "
             "test. No golden or forward holdout was read.", "",
             f"**Promotion gate: {summary['claim_gate']['status']}.** "
             + summary["claim_gate"]["interpretation"], "",
             "## Test results", "",
             "| N prior | players | raw MAE | EB MAE | RAA MAE | Delta MAE | "
             "shot xR MAE | Delta−EB [95% CI] |", 
             "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for n, result in summary["test"].items():
        if result["n_players"] == 0:
            lines.append(f"| {n} | 0 | — | — | — | — | — | — |")
            continue
        m = result["methods"]
        d = result["paired_deltas"]["delta_minus_eb_mae"]
        lo, hi = d["player_bootstrap_ci95"]
        lines.append(
            f"| {n} | {result['n_players']} | "
            f"{m['raw']['player_equal_weight_mae']:.5f} | "
            f"{m['eb']['player_equal_weight_mae']:.5f} | "
            f"{m['raa']['player_equal_weight_mae']:.5f} | "
            f"{m['delta']['player_equal_weight_mae']:.5f} | "
            f"{m['xr']['player_equal_weight_mae']:.5f} | "
            f"{d['mean']:+.5f} [{lo:+.5f}, {hi:+.5f}] |")
    lines.extend(["", "## Decision payload", "",
                  "The locked payload is an internal luck-adjusted auction "
                  "shortlist: Delta expected runs per 100 deliveries above "
                  "the test-pool mean, restricted to batters with at least "
                  "250 prior and 100 strictly future labeled deliveries. "
                  "It remains internal because the process labels derive "
                  "from DeepCrease.", "", "## Interpretation", "",
                  summary["interpretation"], ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    reject_sealed(args.config)
    config = yaml.safe_load(args.config.read_text())
    for key in ("directory", "process_directory"):
        reject_sealed(config["data"][key])
    sample_sizes = [int(value) for value in config["data"]["sample_sizes"]]
    horizon = int(config["data"]["prediction_horizon"])
    rows = {split: load_rows(split, config) for split in ("validation", "test")}
    examples = {split: build_examples(rows[split], sample_sizes, horizon)
                for split in rows}
    mean_actual = float(rows["validation"]["actual"].mean())
    test_mean_actual = float(rows["test"]["actual"].mean())
    grid = [float(value) for value in config["estimators"]["eb"]["k_grid"]]
    tuned = tune_k(examples["validation"], grid, mean_actual)
    validation = evaluate(examples["validation"], tuned, mean_actual, config)
    test = evaluate(examples["test"], tuned, mean_actual, config)
    checks = []
    for n in (100, 250):
        result = test[str(n)]
        if result["n_players"]:
            checks.append(result["paired_deltas"]["delta_minus_eb_mae"])
    clean = [result for result in checks
             if result["player_bootstrap_ci95"][1] < 0]
    point_nonworse = bool(checks and all(result["mean"] <= 0
                                         for result in checks))
    passed = bool(clean and point_nonworse)
    interpretation = (
        "The luck-adjusted Delta metric clears the registered future-value "
        "gate and may advance to a larger open-data/decision validation."
        if passed else
        "The process metric does not clear the registered future-value gate; "
        "retain it as an internal descriptive measurement and do not optimize "
        "auction decisions against it yet.")
    summary = {
        "contract": config["experiment"]["name"],
        "sealed_data_used": False, "future_match_boundary_enforced": True,
        "horizon": horizon, "validation_pool_mean": mean_actual,
        "test_pool_mean": test_mean_actual,
        "tuned_shrinkage_k": tuned, "validation": validation, "test": test,
        "claim_gate": {"status": "PASS" if passed else "FAIL",
                       "interpretation": interpretation},
        "decision_payload": config["decision_payload"],
        "interpretation": interpretation,
        "provenance": {
            "config_sha256": sha256(args.config),
            "script_sha256": sha256(Path(__file__)),
            "input_sha256": {
                str(path): sha256(path)
                for path in [
                    *(Path(config["data"]["directory"]) /
                      f"cricket_data_v3_{split}.parquet"
                      for split in ("validation", "test")),
                    *(Path(config["data"]["process_directory"]) /
                      f"process_metrics_{split}.npz"
                      for split in ("validation", "test")),
                ]
            },
        },
    }
    summary_path = Path(config["outputs"]["summary"])
    report_path = Path(config["outputs"]["report"])
    payload_path = Path(config["outputs"]["player_payload"])
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    report_path.write_text(markdown(summary))
    payload = examples["test"].get(250, pd.DataFrame()).copy()
    if len(payload):
        preds = estimates(payload, tuned["delta"], mean_actual)
        payload["delta_estimate"] = preds["delta"]
        # Registered unit (config + report): "expected team runs per 100
        # deliveries above the TEST-pool mean". The estimator prior stays
        # the validation mean (a frozen modeling choice); only the payload's
        # reference point is the test pool. Pre-fix this subtracted the
        # validation mean — a constant offset, ranking unaffected.
        payload["auction_runs_above_mean_per_100"] = 100.0 * (
            payload["delta_estimate"] - test_mean_actual)
        payload["future_absolute_error"] = np.abs(
            payload["delta_estimate"] - payload["future_actual"])
        payload.sort_values("auction_runs_above_mean_per_100", ascending=False
                            ).to_csv(payload_path, index=False)
    print(json.dumps({"gate": summary["claim_gate"],
                      "tuned_shrinkage_k": tuned,
                      "test_player_counts": {
                          n: value["n_players"] for n, value in test.items()},
                      "report": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()
