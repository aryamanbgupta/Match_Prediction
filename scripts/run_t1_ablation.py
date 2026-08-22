"""Run and summarize the registered exact-information T1 ablation.

The raw checkpoints and row-aligned probabilities live under ``models/`` and
are intentionally not versioned. The registered YAML and a copied, reviewed
Markdown result are the durable repository record.

Examples:
    .venv/bin/python scripts/run_t1_ablation.py --dry-run
    .venv/bin/python scripts/run_t1_ablation.py
    .venv/bin/python scripts/run_t1_ablation.py --aggregate-only
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from registered_experiment import (environment_block, git_provenance,  # noqa: E402
                                   match_cluster_ci, match_ids,
                                   reject_sealed, row_log_loss,
                                   seed_mean_match_cluster_ci, sha256)
from transformer_t1 import (build_features, calibration_metrics, load_split)  # noqa: E402

DEFAULT_CONFIG = Path("experiments/configs/t1_ablation_v1.yaml")

# This runner's CI targets the ACROSS-SEED MEAN of each arm; the reports
# label it "seed-mean+match bootstrap 95% CI" (contrast with the xR runner's
# wider seed-draw estimator). Aliases keep the registered names importable.
row_ll = row_log_loss
paired_cluster_ci = match_cluster_ci
paired_seed_cluster_ci = seed_mean_match_cluster_ci


def load_predictions(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path) as payload:
        return (payload["probs"], payload["y"],
                payload["innings_id"].astype(str))


def save_predictions(path: Path, probs: np.ndarray, y: np.ndarray,
                     innings_id: np.ndarray) -> None:
    np.savez_compressed(path, probs=probs.astype(np.float32), y=y,
                        innings_id=innings_id.astype(str))


def logistic_metrics(probs: np.ndarray, y: np.ndarray) -> dict:
    losses = row_ll(probs, y)
    return {"ll": round(float(losses.mean()), 6),
            "calibration": calibration_metrics(probs, y)}


def run_logistic(config: dict, output: Path) -> None:
    arm_dir = output / "logistic"
    arm_dir.mkdir(parents=True, exist_ok=True)
    complete = all((arm_dir / f"predictions_{split}.npz").exists()
                   for split in ("validation", "test"))
    metrics_path = arm_dir / "metrics.json"
    if complete and metrics_path.exists():
        # File existence is not enough: a run that hit max_iter must never
        # be silently reused as the registered control.
        if json.loads(metrics_path.read_text()).get("converged"):
            print("logistic: complete; reusing", flush=True)
            return
        raise RuntimeError(
            f"existing logistic artifacts in {arm_dir} are non-converged; "
            "delete the directory (or raise max_iter) and refit")

    data_dir = Path(config["data"]["directory"])
    print("logistic: loading exact 50-feature splits", flush=True)
    frames = {name: load_split(name, data_dir)
              for name in ("train", "validation", "test")}
    features = {name: build_features(frame) for name, frame in frames.items()}
    labels = {name: frame["y"].to_numpy() for name, frame in frames.items()}

    spec = config["training"]["logistic"]
    scaler = StandardScaler(copy=False)
    x_train = scaler.fit_transform(features["train"])
    model = LogisticRegression(
        solver=spec["solver"], max_iter=int(spec["max_iter"]),
        tol=float(spec["tolerance"]), C=float(spec["C"]),
    )
    started = time.time()
    model.fit(x_train, labels["train"])
    converged = bool(np.max(model.n_iter_) < int(spec["max_iter"]))
    metrics = {"converged": converged,
               "iterations": [int(value) for value in model.n_iter_],
               "seconds": round(time.time() - started, 1)}
    if not converged:
        # Persist only the diagnostic record — never predictions or the model
        # pickle, which the reuse path could otherwise mistake for a valid
        # registered control.
        metrics_path.write_text(json.dumps(metrics, indent=2))
        raise RuntimeError(
            "logistic control hit max_iter; increase it before interpretation")
    for split in ("validation", "test"):
        transformed = scaler.transform(features[split])
        probs = model.predict_proba(transformed)
        save_predictions(
            arm_dir / f"predictions_{split}.npz", probs, labels[split],
            frames[split]["innings_id"].astype(str).to_numpy())
        metrics[split] = logistic_metrics(probs, labels[split])
    joblib.dump({"scaler": scaler, "model": model}, arm_dir / "model.joblib")
    metrics_path.write_text(json.dumps(metrics, indent=2))


def neural_command(config: dict, arm: str, seed: int, output: Path) -> list[str]:
    train = config["training"]
    return [
        sys.executable, str(SCRIPTS / "transformer_t1.py"),
        "--arm", arm, "--seed", str(seed),
        "--dmodel", str(train["dmodel"]), "--layers", str(train["layers"]),
        "--heads", str(train["heads"]), "--batch", str(train["batch"]),
        "--epochs", str(train["epochs"]), "--lr", str(train["learning_rate"]),
        "--patience", str(train["patience"]),
        "--device", str(train.get("device", "auto")),
        "--data-dir", str(config["data"]["directory"]),
        "--kit-dir", str(config["data"]["eval_kit"]),
        "--out", str(output / arm / f"seed_{seed}"),
        "--save-predictions",
    ]


def run_neural(config: dict, output: Path, only_arm: str | None,
               only_seed: int | None, dry_run: bool) -> None:
    arms = config["arms"]["neural"]
    if only_arm:
        if only_arm not in arms:
            raise ValueError(f"{only_arm} is not a configured neural arm")
        arms = [only_arm]
    seeds = config["training"]["seeds"]
    if only_seed is not None:
        if only_seed not in seeds:
            raise ValueError(f"seed {only_seed} is not registered")
        seeds = [only_seed]
    for arm in arms:
        for seed in seeds:
            arm_dir = output / arm / f"seed_{seed}"
            command = neural_command(config, arm, int(seed), output)
            if dry_run:
                print(" ".join(command))
                continue
            if ((arm_dir / "metrics.json").exists()
                    and (arm_dir / "predictions_validation.npz").exists()
                    and (arm_dir / "predictions_test.npz").exists()):
                print(f"{arm} seed {seed}: complete; reusing", flush=True)
                continue
            print(f"{arm} seed {seed}: training", flush=True)
            subprocess.run(command, check=True)


def slice_masks(frame: pd.DataFrame, probes: pd.DataFrame) -> dict[str, np.ndarray]:
    innings = frame["innings_id"].astype(str).str.split("_", n=1).str[0]
    batter_count = frame["batter_id"].map(
        probes["train_balls_batting"]).fillna(0).to_numpy()
    bowler_count = frame["bowler_id"].map(
        probes["train_balls_bowling"]).fillna(0).to_numpy()
    middle = frame["is_middle_overs"].to_numpy(bool)
    death = frame["is_death_overs"].to_numpy(bool)
    return {
        "all": np.ones(len(frame), dtype=bool),
        "thin_pair_under_200_train_balls": np.minimum(
            batter_count, bowler_count) < 200,
        "powerplay": ~(middle | death),
        "middle": middle,
        "death": death,
        "innings_1": innings.to_numpy() == "1",
        "innings_2": innings.to_numpy() == "2",
    }


def summarize(config: dict, output: Path, config_path: Path,
              allow_partial: bool = False) -> dict:
    baseline = output / "logistic"
    if not baseline.exists():
        raise FileNotFoundError("logistic outputs are missing")
    data_dir = Path(config["data"]["directory"])
    kit_dir = Path(config["data"]["eval_kit"])
    probes = pd.read_parquet(kit_dir / "probe_labels.parquet")
    reps = int(config["statistics"]["bootstrap_repetitions"])
    boot_seed = int(config["statistics"]["bootstrap_seed"])

    summary: dict = {"experiment": config["experiment"],
                     "n_seeds": len(config["training"]["seeds"]),
                     "splits": {}}
    for split in ("validation", "test"):
        base_probs, y, innings_id = load_predictions(
            baseline / f"predictions_{split}.npz")
        base_loss = row_ll(base_probs, y)
        clusters = match_ids(innings_id)
        frame = load_split(split, data_dir)
        if not np.array_equal(
                frame["innings_id"].astype(str).to_numpy(), innings_id):
            raise ValueError(f"data/prediction row alignment failed: {split}")
        masks = slice_masks(frame, probes)
        split_result = {
            "n_rows": int(len(y)),
            "n_matches": int(len(np.unique(clusters))),
            "logistic": logistic_metrics(base_probs, y), "arms": {}}
        arm_seed_losses: dict[str, np.ndarray] = {}
        for arm in config["arms"]["neural"]:
            expected = [output / arm / f"seed_{seed}"
                        / f"predictions_{split}.npz"
                        for seed in config["training"]["seeds"]]
            if not all(path.exists() for path in expected):
                if allow_partial:
                    continue
                missing = next(path for path in expected if not path.exists())
                raise FileNotFoundError(f"registered output is missing: {missing}")
            seed_losses = []
            seed_calibration = []
            per_seed = []
            for seed in config["training"]["seeds"]:
                path = output / arm / f"seed_{seed}" / f"predictions_{split}.npz"
                probs, candidate_y, candidate_innings = load_predictions(path)
                if not (np.array_equal(y, candidate_y)
                        and np.array_equal(innings_id, candidate_innings)):
                    raise ValueError(f"row alignment failed: {path}")
                loss = row_ll(probs, y)
                delta = loss - base_loss
                ci = paired_cluster_ci(delta, clusters, reps, boot_seed)
                seed_losses.append(loss)
                seed_calibration.append(calibration_metrics(probs, y))
                per_seed.append({
                    "seed": int(seed), "ll": round(float(loss.mean()), 6),
                    "delta_ll_vs_logistic": round(float(delta.mean()), 6),
                    "delta_ll_ci95": [round(ci[0], 6), round(ci[1], 6)],
                })
            stacked = np.stack(seed_losses)
            arm_seed_losses[arm] = stacked
            mean_loss = stacked.mean(axis=0)
            mean_delta = mean_loss - base_loss
            aggregate_match_ci = paired_cluster_ci(
                mean_delta, clusters, reps, boot_seed)
            aggregate_joint_ci = paired_seed_cluster_ci(
                stacked - base_loss[None, :], clusters, reps, boot_seed)
            slices = {}
            for name, mask in masks.items():
                match_ci = paired_cluster_ci(
                    mean_delta[mask], clusters[mask], reps, boot_seed)
                joint_ci = paired_seed_cluster_ci(
                    stacked[:, mask] - base_loss[None, mask],
                    clusters[mask], reps, boot_seed)
                slices[name] = {
                    "n": int(mask.sum()),
                    "delta_ll_vs_logistic": round(float(mean_delta[mask].mean()), 6),
                    "delta_ll_match_ci95": [round(match_ci[0], 6),
                                             round(match_ci[1], 6)],
                    "delta_ll_seed_match_ci95": [round(joint_ci[0], 6),
                                                  round(joint_ci[1], 6)],
                }
            run_means = stacked.mean(axis=1)
            split_result["arms"][arm] = {
                "seed_mean_ll": round(float(run_means.mean()), 6),
                "seed_sd_ll": round(float(run_means.std(ddof=1)), 6),
                "mean_delta_ll_vs_logistic": round(float(mean_delta.mean()), 6),
                "mean_delta_ll_match_ci95": [round(aggregate_match_ci[0], 6),
                                              round(aggregate_match_ci[1], 6)],
                "mean_delta_ll_seed_match_ci95": [
                    round(aggregate_joint_ci[0], 6),
                    round(aggregate_joint_ci[1], 6)],
                "direction_better_seeds": int(sum(
                    entry["delta_ll_vs_logistic"] < 0 for entry in per_seed)),
                "mean_calibration": {
                    key: round(float(np.mean([value[key]
                                             for value in seed_calibration])), 6)
                    for key in seed_calibration[0]
                },
                "per_seed": per_seed,
                "slices": slices,
            }
        comparisons = {}
        if "full" in arm_seed_losses:
            full = arm_seed_losses["full"]
            for reference in ("mlp", "no_attention", "no_history"):
                if reference not in arm_seed_losses:
                    continue
                other = arm_seed_losses[reference]
                seed_delta = full - other
                delta = seed_delta.mean(axis=0)
                match_ci = paired_cluster_ci(delta, clusters, reps, boot_seed)
                joint_ci = paired_seed_cluster_ci(
                    seed_delta, clusters, reps, boot_seed)
                comparison_slices = {}
                for slice_name, mask in masks.items():
                    slice_delta = seed_delta[:, mask]
                    slice_mean = slice_delta.mean(axis=0)
                    slice_match_ci = paired_cluster_ci(
                        slice_mean, clusters[mask], reps, boot_seed)
                    slice_joint_ci = paired_seed_cluster_ci(
                        slice_delta, clusters[mask], reps, boot_seed)
                    comparison_slices[slice_name] = {
                        "n": int(mask.sum()),
                        "delta_ll": round(float(slice_mean.mean()), 6),
                        "delta_ll_match_ci95": [round(slice_match_ci[0], 6),
                                                 round(slice_match_ci[1], 6)],
                        "delta_ll_seed_match_ci95": [
                            round(slice_joint_ci[0], 6),
                            round(slice_joint_ci[1], 6)],
                        "direction_better_seeds": int(sum(
                            slice_delta[index].mean() < 0
                            for index in range(slice_delta.shape[0]))),
                    }
                comparisons[f"full_vs_{reference}"] = {
                    "delta_ll": round(float(delta.mean()), 6),
                    "delta_ll_match_ci95": [round(match_ci[0], 6),
                                             round(match_ci[1], 6)],
                    "delta_ll_seed_match_ci95": [round(joint_ci[0], 6),
                                                  round(joint_ci[1], 6)],
                    "direction_better_seeds": int(sum(
                        full[index].mean() < other[index].mean()
                        for index in range(full.shape[0]))),
                    "slices": comparison_slices,
                }
        split_result["sequence_comparisons"] = comparisons
        full_result = split_result["arms"].get("full")
        gate_checks = {}
        # Registered gate estimator (decision 2026-08-22): the seed-mean+match
        # CI — a comparison must clear zero net of BOTH match resampling and
        # seed noise. The registered YAML's shorthand "match-clustered" is
        # read as this joint interval; the v1 verdict is estimator-robust
        # (full_vs_mlp straddles zero on the match-only CI too, both splits).
        if full_result:
            gate_checks["full_vs_logistic"] = bool(
                full_result["mean_delta_ll_seed_match_ci95"][1] < 0
                and full_result["direction_better_seeds"] >= 4)
            for name, values in comparisons.items():
                gate_checks[name] = bool(
                    values["delta_ll_seed_match_ci95"][1] < 0
                    and values["direction_better_seeds"] >= 4)
        split_result["claim_gate"] = {
            "estimator": "seed_mean_match_ci95",
            "checks": gate_checks,
            "passes": bool(gate_checks and all(gate_checks.values())),
        }
        summary["splits"][split] = split_result

    summary["claim_gate"] = {
        "split_passes": {
            split: bool(result["claim_gate"]["passes"])
            for split, result in summary["splits"].items()
        }
    }
    summary["claim_gate"]["passes"] = bool(
        all(summary["claim_gate"]["split_passes"].values()))

    input_paths = [config_path, Path("scripts/transformer_t1.py"),
                   Path("scripts/run_t1_ablation.py"),
                   Path("scripts/registered_experiment.py"),
                   Path("scripts/embeddings_e1.py")]
    input_paths.extend(data_dir / f"cricket_data_v3_{split}.parquet"
                       for split in ("train", "validation", "test"))
    input_paths.extend([kit_dir / "unseen_pair_masks.npz",
                        kit_dir / "probe_labels.parquet"])
    artifact_paths = sorted(
        path for path in output.rglob("*")
        if path.is_file() and path.name not in {
            "summary.yaml", "summary.md", "summary_partial.yaml",
            "summary_partial.md"})
    summary["provenance"] = {
        "input_sha256": {str(path): sha256(path) for path in input_paths},
        "artifact_sha256": {
            str(path.relative_to(output)): sha256(path)
            for path in artifact_paths
        },
        "environment": environment_block(np, pd, sklearn, torch),
        **git_provenance(),
    }
    return summary


def markdown(summary: dict) -> str:
    n_seeds = int(summary.get("n_seeds", 5))
    lines = ["# T1 exact-information ablation v1", "",
             "Generated from the registered five-arm protocol. Negative ΔLL "
             "beats the scaled logistic control. The seed-mean+match "
             "bootstrap targets the across-seed mean (seeds resampled with "
             "replacement, then averaged) — not single-seed variance.", ""]
    for split, result in summary["splits"].items():
        lines.extend([f"## {split.title()}", "",
                      "| arm | seed mean LL | seed SD | ΔLL vs logistic | "
                      "seed-mean+match bootstrap 95% CI | better seeds |",
                      "|---|---:|---:|---:|---:|---:|"])
        base = result["logistic"]["ll"]
        lines.append(f"| logistic | {base:.6f} | — | — | — | — |")
        for arm, values in result["arms"].items():
            lo, hi = values["mean_delta_ll_seed_match_ci95"]
            lines.append(
                f"| {arm} | {values['seed_mean_ll']:.6f} | "
                f"{values['seed_sd_ll']:.6f} | "
                f"{values['mean_delta_ll_vs_logistic']:+.6f} | "
                f"[{lo:+.6f}, {hi:+.6f}] | "
                f"{values['direction_better_seeds']}/{n_seeds} |")
        lines.extend(["", "### Calibration", "",
                      "| arm | multiclass Brier | confidence ECE (15 bins) |",
                      "|---|---:|---:|"])
        base_cal = result["logistic"]["calibration"]
        lines.append(
            f"| logistic | {base_cal['brier']:.6f} | "
            f"{base_cal['confidence_ece_15']:.6f} |")
        for arm, values in result["arms"].items():
            cal = values["mean_calibration"]
            lines.append(
                f"| {arm} | {cal['brier']:.6f} | "
                f"{cal['confidence_ece_15']:.6f} |")
        if result["sequence_comparisons"]:
            lines.extend(["", "### Full T1 attribution", "",
                          "| comparison | ΔLL (full − reference) | "
                          "seed-mean+match bootstrap 95% CI | better seeds |",
                          "|---|---:|---:|---:|"])
            for name, values in result["sequence_comparisons"].items():
                lo, hi = values["delta_ll_seed_match_ci95"]
                lines.append(
                    f"| {name} | {values['delta_ll']:+.6f} | "
                    f"[{lo:+.6f}, {hi:+.6f}] | "
                    f"{values['direction_better_seeds']}/{n_seeds} |")
            primary = result["sequence_comparisons"].get("full_vs_mlp")
            if primary:
                lines.extend(["", "### Full versus MLP by slice", "",
                              "| slice | n | ΔLL | seed-mean+match 95% CI | "
                              "better seeds |",
                              "|---|---:|---:|---:|---:|"])
                for name, values in primary["slices"].items():
                    lo, hi = values["delta_ll_seed_match_ci95"]
                    lines.append(
                        f"| {name} | {values['n']} | "
                        f"{values['delta_ll']:+.6f} | "
                        f"[{lo:+.6f}, {hi:+.6f}] | "
                        f"{values['direction_better_seeds']}/{n_seeds} |")
        lines.append("")
        checks = result["claim_gate"]["checks"]
        if checks:
            label = "PASS" if result["claim_gate"]["passes"] else "FAIL"
            detail = ", ".join(
                f"{name}={'pass' if passed else 'fail'}"
                for name, passed in checks.items())
            lines.extend([f"**Registered sequence gate: {label}.** {detail}.", ""])
    lines.extend(["## Interpretation gate", "",
                  "Do not claim sequence gain unless full T1 clears every "
                  "preregistered comparison and is directionally stable in "
                  "at least four of five seeds.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--only-arm", choices=["mlp", "no_attention",
                                               "no_history", "full"])
    parser.add_argument("--only-logistic", action="store_true")
    parser.add_argument("--only-seed", type=int)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--aggregate-partial", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    reject_sealed(args.config)
    config = yaml.safe_load(args.config.read_text())
    # The config declares forbidden_data; enforce it on the actual paths.
    for value in (config["data"]["directory"], config["data"]["eval_kit"],
                  config["outputs"]["directory"]):
        reject_sealed(value)
    output = Path(config["outputs"]["directory"])

    if args.dry_run:
        if args.only_logistic:
            print(f"fit converged logistic -> {output / 'logistic'}")
            return
        run_neural(config, output, args.only_arm, args.only_seed, dry_run=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    if args.only_logistic:
        run_logistic(config, output)
        return
    if not args.aggregate_only:
        run_logistic(config, output)
        run_neural(config, output, args.only_arm, args.only_seed, dry_run=False)
    if args.only_arm or args.only_seed is not None:
        if args.only_seed is None:
            result = summarize(config, output, args.config, allow_partial=True)
            (output / "summary_partial.yaml").write_text(
                yaml.safe_dump(result, sort_keys=False))
            (output / "summary_partial.md").write_text(markdown(result))
            print(f"wrote {output / 'summary_partial.md'}", flush=True)
        else:
            print("partial seed run complete; arm needs all registered seeds")
        return
    result = summarize(config, output, args.config,
                       allow_partial=args.aggregate_partial)
    (output / "summary.yaml").write_text(yaml.safe_dump(result, sort_keys=False))
    (output / "summary.md").write_text(markdown(result))
    print(f"wrote {output / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
