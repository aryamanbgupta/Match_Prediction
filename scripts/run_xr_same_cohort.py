#!/usr/bin/env python3
"""Registered same-cohort staged xR/xW experiment.

All arms use one complete DeepCrease-labeled cohort, a fixed-width input, and
the same residual MLP. Later arms reveal realized line/length, shot, and the
DeepCrease control field (clean-contact proxy). Those arms are measurement
models, not pre-ball forecasts.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, TensorDataset

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from registered_experiment import (environment_block, git_provenance,  # noqa: E402
                                   match_cluster_ci, match_ids,
                                   reject_sealed, row_log_loss,
                                   seed_draw_match_cluster_ci, sha256)
from transformer_t1 import build_features, load_split  # noqa: E402

DEFAULT_CONFIG = Path("experiments/configs/xr_same_cohort_v1.yaml")
LABEL_COLS = ("line", "length", "shot", "control")
ARMS = ("context", "delivery", "shot", "contact")
RUN_CLASSES = 6

# This runner's comparison CI draws ONE fitted seed per replicate (wider,
# includes single-seed variance); reports label it "seed-draw+match 95% CI".
# Contrast with run_t1_ablation's seed-mean estimator. Aliases keep the
# names the focused suite pins.
cluster_bootstrap = match_cluster_ci
seed_cluster_bootstrap = seed_draw_match_cluster_ci


def _normalise_labels(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    complete = np.ones(len(frame), dtype=bool)
    for column in LABEL_COLS:
        complete &= frame[column].notna().to_numpy()
        if column == "shot":
            complete &= frame[column].astype(str).ne("-").to_numpy()
    frame = frame.loc[complete].copy()
    for column in LABEL_COLS:
        frame[column] = frame[column].astype(str)
    return frame


def build_vocabs(label_path: Path) -> dict[str, dict[str, int]]:
    labels = _normalise_labels(pd.read_parquet(label_path))
    return {
        column: {value: index for index, value in enumerate(
            sorted(labels[column].unique().tolist()))}
        for column in LABEL_COLS
    }


def load_cohort(split: str, data_dir: Path, labels_dir: Path,
                vocabs: dict[str, dict[str, int]]) -> dict:
    data_path = data_dir / f"cricket_data_v3_{split}.parquet"
    label_path = labels_dir / f"{split}.parquet"
    reject_sealed(data_path)
    reject_sealed(label_path)

    frame = load_split(split, data_dir)
    base = build_features(frame)
    meta = pd.read_parquet(
        data_path,
        columns=["innings_id", "team_runs", "is_wicket", "match_date",
                 "inning_idx", "over_idx", "batter_id", "bowler_id"],
    )
    labels_raw = pd.read_parquet(label_path)
    labels = _normalise_labels(labels_raw)
    known = np.ones(len(labels), dtype=bool)
    for column in LABEL_COLS:
        known &= labels[column].isin(vocabs[column]).to_numpy()
    labels = labels.loc[known].sort_values("row_idx").reset_index(drop=True)
    row_idx = labels["row_idx"].to_numpy(np.int64)
    if len(np.unique(row_idx)) != len(row_idx):
        raise ValueError(f"duplicate label row_idx in {label_path}")
    if len(row_idx) == 0 or row_idx.min() < 0 or row_idx.max() >= len(frame):
        raise ValueError(f"invalid label row_idx in {label_path}")

    offsets, width = {}, 0
    for column in LABEL_COLS:
        offsets[column] = (width, width + len(vocabs[column]))
        width += len(vocabs[column])
    staged = np.zeros((len(labels), width), dtype=np.float32)
    encoded = {}
    for column in LABEL_COLS:
        values = labels[column].map(vocabs[column]).to_numpy(np.int64)
        encoded[column] = values
        start, _ = offsets[column]
        staged[np.arange(len(labels)), start + values] = 1.0

    x = np.hstack([base[row_idx], staged]).astype(np.float32, copy=False)
    innings = meta.iloc[row_idx]["innings_id"].astype(str).to_numpy()
    return {
        "x": x,
        "y": frame.iloc[row_idx]["y"].to_numpy(np.int64),
        "team_runs": meta.iloc[row_idx]["team_runs"].to_numpy(np.float32),
        "is_wicket": meta.iloc[row_idx]["is_wicket"].to_numpy(np.int8),
        "innings_id": innings,
        "match_id": match_ids(innings),
        "row_idx": row_idx,
        "match_date": meta.iloc[row_idx]["match_date"].astype(str).to_numpy(),
        "inning_idx": meta.iloc[row_idx]["inning_idx"].to_numpy(np.int8),
        "over_idx": meta.iloc[row_idx]["over_idx"].to_numpy(np.int16),
        "batter_id": meta.iloc[row_idx]["batter_id"].astype(str).to_numpy(),
        "bowler_id": meta.iloc[row_idx]["bowler_id"].astype(str).to_numpy(),
        "encoded": encoded,
        "offsets": offsets,
        "n_base_features": int(base.shape[1]),
        "coverage": {
            "all_split_rows": int(len(frame)),
            "joined_rows": int(len(labels_raw)),
            "complete_known_rows": int(len(labels)),
            "complete_known_fraction_all": float(len(labels) / len(frame)),
            "date_min": str(meta.iloc[row_idx]["match_date"].min()),
            "date_max": str(meta.iloc[row_idx]["match_date"].max()),
            "matches": int(len(np.unique(match_ids(innings)))),
        },
    }


def arm_mask(cohort: dict, arm: str) -> np.ndarray:
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    mask = np.ones(cohort["x"].shape[1], dtype=np.float32)
    base = cohort["n_base_features"]
    reveal = {
        "context": (),
        "delivery": ("line", "length"),
        "shot": ("line", "length", "shot"),
        "contact": LABEL_COLS,
    }[arm]
    mask[base:] = 0.0
    for column in reveal:
        start, end = cohort["offsets"][column]
        mask[base + start:base + end] = 1.0
    return mask


class ResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.linear1 = nn.Linear(width, 2 * width)
        self.linear2 = nn.Linear(2 * width, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value):
        hidden = nn.functional.gelu(self.linear1(self.norm(value)))
        return value + self.dropout(self.linear2(self.dropout(hidden)))


class SameCohortHead(nn.Module):
    def __init__(self, n_features: int, width: int, blocks: int,
                 dropout: float):
        super().__init__()
        self.input = nn.Linear(n_features, width)
        self.blocks = nn.Sequential(*[
            ResidualBlock(width, dropout) for _ in range(blocks)])
        self.output = nn.Linear(width, RUN_CLASSES)

    def forward(self, value):
        return self.output(self.blocks(self.input(value)))


def select_device(value: str) -> str:
    if value == "auto":
        return ("mps" if torch.backends.mps.is_available()
                else "cuda" if torch.cuda.is_available() else "cpu")
    if value == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return value


def predict(model, cohort: dict, mask: torch.Tensor, device: str,
            batch_size: int) -> np.ndarray:
    model.eval()
    output = np.empty((len(cohort["y"]), RUN_CLASSES), dtype=np.float32)
    x = torch.from_numpy(cohort["x"])
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = x[start:start + batch_size].to(device)
            logits = model(xb * mask)
            output[start:start + len(xb)] = torch.softmax(
                logits, dim=1).cpu().numpy()
    return output


def fit_one(train: dict, validation: dict, test: dict, arm: str, seed: int,
            config: dict, output: Path) -> None:
    training = config["training"]
    device = select_device(str(training.get("device", "auto")))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    model = SameCohortHead(
        train["x"].shape[1], int(training["hidden_width"]),
        int(training["residual_blocks"]), float(training["dropout"]),
    ).to(device)
    n_params = sum(parameter.numel() for parameter in model.parameters())
    mask = torch.from_numpy(arm_mask(train, arm)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    loss_fn = nn.CrossEntropyLoss()
    dataset = TensorDataset(torch.from_numpy(train["x"]),
                            torch.from_numpy(train["y"]))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset, batch_size=int(training["batch"]), shuffle=True,
        generator=generator, num_workers=0,
    )

    best_ll, best_epoch, best_state, bad = math.inf, -1, None, 0
    history = []
    started = time.time()
    for epoch in range(int(training["epochs"])):
        model.train()
        running, seen = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb * mask), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += float(loss.detach()) * len(yb)
            seen += len(yb)
        val_probs = predict(model, validation, mask, device,
                            int(training["batch"]) * 2)
        val_ll = float(row_log_loss(val_probs, validation["y"]).mean())
        history.append({"epoch": epoch, "train_ll": running / seen,
                        "validation_ll": val_ll})
        print(f"arm={arm} seed={seed} epoch={epoch} "
              f"train={running/seen:.6f} val={val_ll:.6f}", flush=True)
        if val_ll < best_ll - 1e-5:
            best_ll, best_epoch, bad = val_ll, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= int(training["patience"]):
                break
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)

    arm_dir = output / arm / f"seed_{seed}"
    arm_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), arm_dir / "model.pt")
    split_metrics = {}
    for split, cohort in (("validation", validation), ("test", test)):
        probs = predict(model, cohort, mask, device,
                        int(training["batch"]) * 2)
        np.savez_compressed(
            arm_dir / f"predictions_{split}.npz", probs=probs,
            y=cohort["y"], row_idx=cohort["row_idx"],
            innings_id=np.asarray(cohort["innings_id"], dtype=str),
            team_runs=cohort["team_runs"],
            is_wicket=cohort["is_wicket"],
        )
        split_metrics[f"{split}_ll"] = float(
            row_log_loss(probs, cohort["y"]).mean())
    metrics = {
        "arm": arm, "seed": seed, "device": device,
        "n_parameters": n_params, "best_epoch": best_epoch,
        "best_validation_ll": best_ll, "history": history,
        "elapsed_seconds": time.time() - started, **split_metrics,
    }
    (arm_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))


def multiclass_brier(probs: np.ndarray, y: np.ndarray) -> float:
    target = np.eye(probs.shape[1], dtype=np.float32)[y]
    return float(np.square(probs - target).sum(axis=1).mean())


def binary_log_loss(prob: np.ndarray, target: np.ndarray) -> float:
    prob = np.clip(np.asarray(prob, dtype=np.float64), 1e-15, 1 - 1e-15)
    return float(-(target * np.log(prob) + (1 - target) * np.log(1 - prob)).mean())


def calibration_bins(predicted: np.ndarray, actual: np.ndarray,
                     n_bins: int = 10) -> list[dict]:
    order = np.argsort(predicted)
    output = []
    for indices in np.array_split(order, n_bins):
        if len(indices):
            output.append({
                "n": int(len(indices)),
                "mean_predicted": float(predicted[indices].mean()),
                "mean_actual": float(actual[indices].mean()),
            })
    return output


def calibration_line(predicted: np.ndarray, actual: np.ndarray) -> dict:
    design = np.column_stack([np.ones(len(predicted)), predicted])
    intercept, slope = np.linalg.lstsq(design, actual, rcond=None)[0]
    return {"intercept": float(intercept), "slope": float(slope)}


def prediction_metrics(probs: np.ndarray, cohort: dict,
                       run_mapping: np.ndarray) -> dict:
    y, runs = cohort["y"], cohort["team_runs"].astype(float)
    wicket = cohort["is_wicket"].astype(float)
    xr = probs @ run_mapping
    xw = probs[:, 5]
    return {
        "six_class_log_loss": float(row_log_loss(probs, y).mean()),
        "multiclass_brier": multiclass_brier(probs, y),
        "expected_runs": {
            "mean_predicted": float(xr.mean()),
            "mean_actual": float(runs.mean()),
            "bias": float((xr - runs).mean()),
            "mae": float(np.abs(xr - runs).mean()),
            "rmse": float(np.sqrt(np.square(xr - runs).mean())),
            "calibration_line": calibration_line(xr, runs),
            "calibration_bins": calibration_bins(xr, runs),
        },
        "wicket": {
            "mean_predicted": float(xw.mean()),
            "mean_actual": float(wicket.mean()),
            "log_loss": binary_log_loss(xw, wicket),
            "brier": float(np.square(xw - wicket).mean()),
            "calibration_line": calibration_line(xw, wicket),
            "calibration_bins": calibration_bins(xw, wicket),
        },
    }


def _load_predictions(path: Path, cohort: dict) -> np.ndarray:
    # Runs produced before the fixed-width unicode save used NumPy object
    # strings for innings_id. These are local, self-generated artifacts and
    # are still exhaustively row-alignment checked below.
    payload = np.load(path, allow_pickle=True)
    for name in ("y", "row_idx", "innings_id", "team_runs", "is_wicket"):
        if not np.array_equal(payload[name], cohort[name]):
            raise ValueError(f"row alignment failed for {name}: {path}")
    return payload["probs"]


def summarize(config: dict, config_path: Path, cohorts: dict,
              vocabs: dict, output: Path) -> dict:
    seeds = [int(value) for value in config["training"]["seeds"]]
    reps = int(config["statistics"]["bootstrap_repetitions"])
    boot_seed = int(config["statistics"]["bootstrap_seed"])
    train = cohorts["train"]
    run_mapping = np.asarray([
        train["team_runs"][train["y"] == target].mean()
        for target in range(RUN_CLASSES)
    ], dtype=np.float64)
    summary = {
        "contract": config["experiment"]["name"],
        "n_seeds": len(seeds),
        "calibration_applied": False,
        "realized_information_not_preball_prediction": True,
        "vocabs": {name: list(values) for name, values in vocabs.items()},
        "run_mapping_team_runs_by_class": run_mapping.tolist(),
        "coverage": {split: cohort["coverage"]
                     for split, cohort in cohorts.items()},
        "splits": {},
    }
    all_probs = {}
    for split in ("validation", "test"):
        cohort = cohorts[split]
        split_result = {"arms": {}, "comparisons": {}}
        all_probs[split] = {}
        for arm in ARMS:
            probs_by_seed, per_seed = [], []
            for seed in seeds:
                path = output / arm / f"seed_{seed}" / f"predictions_{split}.npz"
                probs = _load_predictions(path, cohort)
                probs_by_seed.append(probs)
                per_seed.append({
                    "seed": seed,
                    **prediction_metrics(probs, cohort, run_mapping),
                })
            stacked = np.stack(probs_by_seed)
            all_probs[split][arm] = stacked
            mean_probs = stacked.mean(axis=0)
            seed_ll = np.asarray([entry["six_class_log_loss"]
                                  for entry in per_seed])
            split_result["arms"][arm] = {
                "seed_mean_log_loss": float(seed_ll.mean()),
                "seed_sd_log_loss": float(seed_ll.std(ddof=1)),
                "ensemble_mean_probability_metrics": prediction_metrics(
                    mean_probs, cohort, run_mapping),
                "per_seed": per_seed,
            }

        for left, right in zip(ARMS[:-1], ARMS[1:]):
            left_probs, right_probs = all_probs[split][left], all_probs[split][right]
            left_ll = np.stack([row_log_loss(value, cohort["y"])
                                for value in left_probs])
            right_ll = np.stack([row_log_loss(value, cohort["y"])
                                 for value in right_probs])
            delta = right_ll - left_ll
            left_xr = left_probs @ run_mapping
            right_xr = right_probs @ run_mapping
            left_abs = np.abs(left_xr - cohort["team_runs"][None, :])
            right_abs = np.abs(right_xr - cohort["team_runs"][None, :])
            name = f"{right}_vs_{left}"
            split_result["comparisons"][name] = {
                "delta_log_loss": float(delta.mean()),
                "delta_log_loss_seed_match_ci95": seed_cluster_bootstrap(
                    delta, cohort["match_id"], reps, boot_seed),
                "direction_better_seeds": int(sum(
                    delta[index].mean() < 0 for index in range(len(seeds)))),
                "delta_expected_runs_mae": float((right_abs - left_abs).mean()),
                "delta_expected_runs_mae_seed_match_ci95": seed_cluster_bootstrap(
                    right_abs - left_abs, cohort["match_id"], reps, boot_seed),
                "mean_xr_increment": float((right_xr - left_xr).mean()),
                "mean_xr_increment_seed_match_ci95": seed_cluster_bootstrap(
                    right_xr - left_xr, cohort["match_id"], reps, boot_seed),
            }

        mean_xr = {arm: all_probs[split][arm].mean(axis=0) @ run_mapping
                   for arm in ARMS}
        process = {
            "row_idx": cohort["row_idx"],
            "innings_id": cohort["innings_id"],
            "actual_team_runs": cohort["team_runs"],
            "actual_wicket": cohort["is_wicket"],
            **{f"xr_{arm}": values for arm, values in mean_xr.items()},
            "delivery_value": mean_xr["delivery"] - mean_xr["context"],
            "shot_selection_value": mean_xr["shot"] - mean_xr["delivery"],
            "contact_value": mean_xr["contact"] - mean_xr["shot"],
            "execution_residual": cohort["team_runs"] - mean_xr["shot"],
        }
        np.savez_compressed(output / f"process_metrics_{split}.npz", **process)
        split_result["process_metrics"] = {
            name: {
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)),
                "quantiles": {
                    str(q): float(np.quantile(values, q))
                    for q in (0.05, 0.25, 0.5, 0.75, 0.95)
                },
                "match_cluster_ci95": cluster_bootstrap(
                    values, cohort["match_id"], reps, boot_seed),
            }
            for name, values in process.items()
            if name in {"delivery_value", "shot_selection_value",
                        "contact_value", "execution_residual"}
        }
        summary["splits"][split] = split_result

    delivery_gate = {}
    for split in ("validation", "test"):
        comparison = summary["splits"][split]["comparisons"][
            "delivery_vs_context"]
        delivery_gate[split] = bool(
            comparison["delta_log_loss_seed_match_ci95"][1] < 0
            and comparison["direction_better_seeds"] >= 4)
    summary["claim_gate"] = {
        "delivery_information": delivery_gate,
        "passes": bool(all(delivery_gate.values())),
        "predictive_value_established": False,
    }

    input_paths = [config_path, Path("scripts/run_xr_same_cohort.py"),
                   Path("scripts/registered_experiment.py"),
                   Path("scripts/transformer_t1.py")]
    input_paths.extend(Path(config["data"]["directory"])
                       / f"cricket_data_v3_{split}.parquet"
                       for split in ("train", "validation", "test"))
    input_paths.extend(Path(config["data"]["labels"]) / f"{split}.parquet"
                       for split in ("train", "validation", "test"))
    summary["provenance"] = {
        "input_sha256": {str(path): sha256(path) for path in input_paths},
        "environment": environment_block(np, pd, torch),
        **git_provenance(),
    }
    return summary


def markdown(summary: dict) -> str:
    n_seeds = int(summary.get("n_seeds", 5))
    lines = ["# Same-cohort staged xR/xW v1", "",
             "All arms use identical rows, input width, residual-head capacity, "
             "training recipe, and seeds. Delivery, shot, and contact are "
             "realized post-ball information, not pre-ball forecasts. The "
             "seed-draw+match bootstrap draws one fitted seed per replicate, "
             "so its CIs include single-seed variance (wider than a "
             "seed-mean estimator).", "",
             "## Coverage", "",
             "| split | selected rows | all rows | coverage | matches | dates |",
             "|---|---:|---:|---:|---:|---|"]
    for split, values in summary["coverage"].items():
        lines.append(
            f"| {split} | {values['complete_known_rows']:,} | "
            f"{values['all_split_rows']:,} | "
            f"{values['complete_known_fraction_all']:.1%} | "
            f"{values['matches']:,} | {values['date_min']} to "
            f"{values['date_max']} |")
    mapping = summary["run_mapping_team_runs_by_class"]
    lines.extend(["", "Train-only run mapping for classes "
                  "[0, 1, 2, 4, 6, wicket]: "
                  + ", ".join(f"{value:.4f}" for value in mapping) + ".", ""])
    for split, result in summary["splits"].items():
        lines.extend([f"## {split.title()}", "",
                      "| arm | seed mean LL ± SD | ensemble xR MAE | "
                      "xR bias | xW Brier |",
                      "|---|---:|---:|---:|---:|"])
        for arm, values in result["arms"].items():
            metrics = values["ensemble_mean_probability_metrics"]
            lines.append(
                f"| {arm} | {values['seed_mean_log_loss']:.6f} ± "
                f"{values['seed_sd_log_loss']:.6f} | "
                f"{metrics['expected_runs']['mae']:.6f} | "
                f"{metrics['expected_runs']['bias']:+.6f} | "
                f"{metrics['wicket']['brier']:.6f} |")
        lines.extend(["", "### Calibration", "",
                      "| arm | xR intercept | xR slope | xW intercept | "
                      "xW slope | xW log loss |",
                      "|---|---:|---:|---:|---:|---:|"])
        for arm, values in result["arms"].items():
            metrics = values["ensemble_mean_probability_metrics"]
            xr_line = metrics["expected_runs"]["calibration_line"]
            xw_line = metrics["wicket"]["calibration_line"]
            lines.append(
                f"| {arm} | {xr_line['intercept']:+.6f} | "
                f"{xr_line['slope']:.6f} | {xw_line['intercept']:+.6f} | "
                f"{xw_line['slope']:.6f} | "
                f"{metrics['wicket']['log_loss']:.6f} |")
        lines.extend(["", "| comparison | ΔLL | seed-draw+match 95% CI | "
                      "better seeds | ΔxR MAE |",
                      "|---|---:|---:|---:|---:|"])
        for name, values in result["comparisons"].items():
            lo, hi = values["delta_log_loss_seed_match_ci95"]
            lines.append(
                f"| {name} | {values['delta_log_loss']:+.6f} | "
                f"[{lo:+.6f}, {hi:+.6f}] | "
                f"{values['direction_better_seeds']}/{n_seeds} | "
                f"{values['delta_expected_runs_mae']:+.6f} |")
        lines.extend(["", "### Process metrics (runs per delivery)", "",
                      "| metric | mean | match-bootstrap 95% CI | SD |",
                      "|---|---:|---:|---:|"])
        for name, values in result["process_metrics"].items():
            lo, hi = values["match_cluster_ci95"]
            lines.append(f"| {name} | {values['mean']:+.6f} | "
                         f"[{lo:+.6f}, {hi:+.6f}] | {values['sd']:.6f} |")
        lines.extend(["", "The mean staged increment can be near zero even "
                      "when the information is valuable: positive and "
                      "negative delivery/shot/contact effects cancel across "
                      "the cohort. Use paired LL/MAE for information value "
                      "and row/player aggregates for process value; do not "
                      "interpret the cohort mean alone.", ""])
    verdict = "PASS" if summary["claim_gate"]["passes"] else "FAIL"
    lines.extend(["## Claim gate", "",
                  f"**Realized delivery-information gate: {verdict}.** "
                  "This gate does not establish future predictive value. "
                  "The contact arm uses DeepCrease control as the available "
                  "clean-contact proxy; no independent contact field exists.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--only-arm", choices=ARMS)
    parser.add_argument("--only-seed", type=int)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    reject_sealed(args.config)
    config = yaml.safe_load(args.config.read_text())
    data_dir = Path(config["data"]["directory"])
    labels_dir = Path(config["data"]["labels"])
    output = Path(config["outputs"]["directory"])
    report = Path(config["outputs"]["report"])
    for path in (data_dir, labels_dir, output, report):
        reject_sealed(path)

    vocabs = build_vocabs(labels_dir / "train.parquet")
    cohorts = {split: load_cohort(split, data_dir, labels_dir, vocabs)
               for split in ("train", "validation", "test")}
    widths = {cohort["x"].shape[1] for cohort in cohorts.values()}
    if len(widths) != 1:
        raise ValueError(f"feature width mismatch: {widths}")
    print("cohorts: " + ", ".join(
        f"{split}={len(cohort['y']):,}" for split, cohort in cohorts.items())
          + f"; width={next(iter(widths))}; vocabs="
          + str({name: len(values) for name, values in vocabs.items()}),
          flush=True)

    seeds = [int(value) for value in config["training"]["seeds"]]
    if args.only_seed is not None:
        if args.only_seed not in seeds:
            raise SystemExit(f"seed {args.only_seed} is not registered")
        seeds = [args.only_seed]
    arms = [args.only_arm] if args.only_arm else list(ARMS)
    if args.dry_run:
        for arm in arms:
            for seed in seeds:
                print(f"would train arm={arm} seed={seed}")
        return

    output.mkdir(parents=True, exist_ok=True)
    if not args.aggregate_only:
        for arm in arms:
            for seed in seeds:
                fit_one(cohorts["train"], cohorts["validation"],
                        cohorts["test"], arm, seed, config, output)
    if args.only_arm or args.only_seed is not None:
        print("partial run complete; aggregate after every registered arm/seed",
              flush=True)
        return

    result = summarize(config, args.config, cohorts, vocabs, output)
    (output / "summary.yaml").write_text(yaml.safe_dump(
        result, sort_keys=False, allow_unicode=True))
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(markdown(result))
    print(f"wrote {output / 'summary.yaml'} and {report}", flush=True)
    if not result["claim_gate"]["passes"]:
        raise SystemExit("registered delivery-information gate failed")


if __name__ == "__main__":
    main()
