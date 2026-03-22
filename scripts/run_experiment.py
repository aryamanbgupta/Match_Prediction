"""
Pipeline Runner — Run a complete experiment from a YAML config.

Usage:
    uv run python scripts/run_experiment.py experiments/configs/xgb_v3_baseline.yaml
    uv run python scripts/run_experiment.py experiments/configs/xgb_v3_baseline.yaml --skip-parsing
    uv run python scripts/run_experiment.py experiments/configs/xgb_v3_baseline.yaml --only-eval
    uv run python scripts/run_experiment.py experiments/configs/xgb_v3_baseline.yaml --dry-run
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from feature_registry import resolve_feature_list, get_feature_hash
from experiment_tracker import ExperimentTracker


def load_config(config_path: str) -> dict:
    """Load and validate a YAML experiment config."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path) as f:
        config = yaml.safe_load(f)

    # Validate required fields
    required = ["experiment", "data", "features", "model"]
    for field in required:
        if field not in config:
            raise ValueError(f"Missing required config field: '{field}'")

    if "name" not in config["experiment"]:
        raise ValueError("Config must have experiment.name")
    if "type" not in config["model"]:
        raise ValueError("Config must have model.type")
    if "groups" not in config["features"]:
        raise ValueError("Config must have features.groups")

    return config


def check_smart_cache(config: dict, feature_list: list) -> bool:
    """Check if parsing can be skipped (parquet exists + feature hash matches)."""
    version = config["data"].get("version", "v3")
    data_dir = Path(f"data/xgb_data_{version}") if version != "v2" else Path("data/xgb_data")

    # Check if parquet files exist
    train_file = data_dir / f"cricket_data_{version}_train.parquet"
    if not train_file.exists():
        return False

    # Check feature hash
    hash_file = data_dir / ".feature_hash"
    if not hash_file.exists():
        return False

    try:
        with open(hash_file) as f:
            cached = json.load(f)
        current_hash = get_feature_hash(feature_list)
        return cached.get("hash") == current_hash
    except (json.JSONDecodeError, KeyError):
        return False


def run_step(cmd: list, step_name: str, tracker: ExperimentTracker,
             capture: bool = True) -> subprocess.CompletedProcess:
    """Run a pipeline step, timing it and logging output."""
    print(f"\n{'='*60}")
    print(f"  {step_name}")
    print(f"{'='*60}")
    print(f"  Command: {' '.join(cmd)}")
    print()

    start = time.time()
    result = subprocess.run(
        cmd,
        capture_output=capture,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    duration = time.time() - start
    tracker.log_step_duration(step_name, round(duration, 1))

    if capture:
        if result.stdout:
            print(result.stdout)
            tracker.log_console_output(f"\n--- {step_name} stdout ---\n{result.stdout}")
        if result.stderr:
            # Only print stderr if non-empty and not just warnings
            stderr_lines = [l for l in result.stderr.split('\n') if l.strip()]
            if stderr_lines:
                print(f"[stderr] {result.stderr}", file=sys.stderr)
                tracker.log_console_output(f"\n--- {step_name} stderr ---\n{result.stderr}")

    if result.returncode != 0:
        print(f"\n  FAILED (exit code {result.returncode}) after {duration:.1f}s")
        raise RuntimeError(f"{step_name} failed with exit code {result.returncode}")

    print(f"  Completed in {duration:.1f}s")
    return result


def build_training_cmd(config: dict, feature_list: list) -> list:
    """Build the training command based on model type."""
    model_type = config["model"]["type"]
    hyperparams = config["model"].get("hyperparameters", {})

    # Config JSON passed to the training script
    config_json = json.dumps({
        "features": {
            "groups": config["features"]["groups"],
            "exclude": config["features"].get("exclude", []),
            "include_extra": config["features"].get("include_extra", []),
        },
        "model": {
            "hyperparameters": hyperparams,
        },
    })

    script_map = {
        "xgboost": "scripts/xgboost_v2.py",
        "lstm": "scripts/lstm_v1.py",
        "transformer": "scripts/transformer_v1.py",
        "mlp": "scripts/mlp_v1.py",
    }

    script = script_map.get(model_type)
    if not script:
        raise ValueError(f"Unknown model type: {model_type}")

    cmd = [sys.executable, script, "--config-json", config_json]

    # Add tune flag for xgboost
    if model_type == "xgboost" and config["model"].get("tune"):
        cmd.append("--tune")
        n_trials = config["model"].get("tune_trials", 50)
        cmd.extend(["--n-trials", str(n_trials)])

    # Add model-specific CLI args from hyperparameters
    if model_type in ("lstm", "transformer", "mlp"):
        cli_map = {
            "epochs": "--epochs",
            "batch_size": "--batch-size",
            "lr": "--lr",
            "hidden_size": "--hidden-size",
            "num_layers": "--num-layers",
            "dropout": "--dropout",
            "weight_decay": "--weight-decay",
            "warmup_epochs": "--warmup-epochs",
            "window_size": "--window-size",
            "focal_gamma": "--focal-gamma",
            "label_smoothing": "--label-smoothing",
            "patience": "--patience",
            "nhead": "--nhead",
            "dim_feedforward": "--dim-feedforward",
            "max_seq_len": "--max-seq-len",
        }
        for key, flag in cli_map.items():
            if key in hyperparams:
                cmd.extend([flag, str(hyperparams[key])])

    return cmd


def build_eval_cmd(config: dict) -> list:
    """Build the evaluation command."""
    model_type = config["model"]["type"]
    version = config["data"].get("version", "v3")
    test_dir = config["data"].get("test_dir", "data/betting_test")
    odds_file = config["data"].get("odds_file", "betting_odds_v3.json")
    n_sims = config.get("evaluation", {}).get("n_sims", 1000)
    parallel = config.get("evaluation", {}).get("parallel", False)

    cmd = [
        sys.executable, "scripts/sim_eval/run_sim_eval.py",
        "--model-type", model_type,
        "--model-version", version,
        "--test-dir", test_dir,
        "--odds", odds_file,
        "--n-sims", str(n_sims),
    ]

    if parallel:
        cmd.append("--parallel")

    # Calibration flags
    eval_config = config.get("evaluation", {})
    if eval_config.get("calibrate"):
        cmd.append("--calibrate")
    if eval_config.get("calibration_method"):
        cmd.extend(["--calibration-method", eval_config["calibration_method"]])
    if eval_config.get("ball_calibrate"):
        cmd.append("--ball-calibrate")
    if eval_config.get("ball_calibrate_data"):
        cmd.extend(["--ball-calibrate-data", eval_config["ball_calibrate_data"]])
    if eval_config.get("ball_diagnostics"):
        cmd.append("--ball-diagnostics")
    if eval_config.get("save_calibrator"):
        cmd.extend(["--save-calibrator", eval_config["save_calibrator"]])
    if eval_config.get("load_calibrator"):
        cmd.extend(["--load-calibrator", eval_config["load_calibrator"]])

    return cmd


def main():
    parser = argparse.ArgumentParser(description="Run a complete experiment from YAML config")
    parser.add_argument("config", type=str, help="Path to YAML experiment config")
    parser.add_argument("--skip-parsing", action="store_true",
                        help="Skip the parsing step")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip the training step")
    parser.add_argument("--only-eval", action="store_true",
                        help="Only run evaluation (skip parsing and training)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    exp_name = config["experiment"]["name"]
    model_type = config["model"]["type"]

    # Resolve features
    feature_list = resolve_feature_list(
        config["features"]["groups"],
        config["features"].get("exclude"),
        config["features"].get("include_extra"),
    )
    feature_hash = get_feature_hash(feature_list)

    print(f"\n{'#'*60}")
    print(f"  Experiment: {exp_name}")
    print(f"  Model: {model_type}")
    print(f"  Features: {len(feature_list)} (hash: {feature_hash})")
    print(f"  Config: {args.config}")
    print(f"{'#'*60}")

    # Determine which steps to run
    pipeline = config.get("pipeline", {})
    skip_parsing = args.skip_parsing or args.only_eval or pipeline.get("skip_parsing", False)
    skip_training = args.skip_training or args.only_eval or pipeline.get("skip_training", False)
    skip_eval = pipeline.get("skip_evaluation", False)

    # Smart cache check
    if not skip_parsing and check_smart_cache(config, feature_list):
        print("\n  Smart cache hit — parsed data matches feature hash. Skipping parsing.")
        skip_parsing = True

    # Build commands
    parse_cmd = [sys.executable, "scripts/parsing_v2.py"]
    train_cmd = build_training_cmd(config, feature_list)
    eval_cmd = build_eval_cmd(config)

    if args.dry_run:
        print("\n--- DRY RUN ---")
        if not skip_parsing:
            print(f"[1/3] PARSE: {' '.join(parse_cmd)}")
        else:
            print("[1/3] PARSE: SKIPPED")
        if not skip_training:
            print(f"[2/3] TRAIN: {' '.join(train_cmd)}")
        else:
            print("[2/3] TRAIN: SKIPPED")
        if not skip_eval:
            print(f"[3/3] EVAL:  {' '.join(eval_cmd)}")
        else:
            print("[3/3] EVAL:  SKIPPED")
        print(f"\nFeatures ({len(feature_list)}):")
        for i, feat in enumerate(feature_list, 1):
            print(f"  {i:2d}. {feat}")
        return

    # Initialize tracker
    tracker = ExperimentTracker()
    exp_id = tracker.start_experiment(config)
    print(f"\n  Experiment ID: {exp_id}")
    print(f"  Results dir: {tracker.experiment_dir}")

    try:
        # Step 1: Parsing
        if not skip_parsing:
            run_step(parse_cmd, "Parsing (feature engineering)", tracker, capture=False)
        else:
            print("\n  Skipping parsing step.")

        # Step 2: Training
        if not skip_training:
            run_step(train_cmd, f"Training ({model_type})", tracker, capture=False)
        else:
            print("\n  Skipping training step.")

        # Step 3: Evaluation
        if not skip_eval:
            result = run_step(eval_cmd, "Evaluation", tracker, capture=True)

            # Try to extract metrics from output
            if result.stdout:
                metrics = _extract_metrics(result.stdout)
                if metrics:
                    tracker.log_evaluation_results(metrics)
                    print("\n  Extracted metrics:")
                    for k, v in metrics.items():
                        print(f"    {k}: {v}")
        else:
            print("\n  Skipping evaluation step.")

        tracker.finish_experiment("completed")
        print(f"\n{'='*60}")
        print(f"  Experiment completed: {exp_id}")
        print(f"  Results: {tracker.experiment_dir}")
        print(f"{'='*60}")

    except (RuntimeError, Exception) as e:
        tracker.finish_experiment("failed")
        print(f"\n  Experiment FAILED: {e}")
        sys.exit(1)


def _extract_metrics(output: str) -> dict:
    """Extract key metrics from evaluation console output.

    Parses the structured summary block from run_sim_eval.py, tracking which
    betting-strategy section we're in so that identically-named fields (ROI,
    Win Rate, Total P&L) are attributed to the correct strategy.
    """
    metrics = {}
    section = None  # tracks current betting-strategy section

    for line in output.split("\n"):
        stripped = line.strip()

        # --- section headers ---
        if stripped.startswith("Flat Staking (1 unit"):
            section = "flat"
        elif stripped.startswith("Full Kelly Criterion"):
            section = "full_kelly"
        elif stripped.startswith("Fractional Kelly"):
            section = "frac_kelly"
        elif stripped.startswith("---") or stripped == "":
            # Reset section on dividers / blanks so per-match-type blocks
            # (Favorites, Underdogs) don't overwrite strategy metrics
            if stripped.startswith("---"):
                section = None

        # --- top-level metrics (outside any strategy block) ---
        if "Average Log Loss:" in stripped:
            try:
                metrics["avg_log_loss"] = float(stripped.split(":")[-1].strip())
            except ValueError:
                pass
        elif "Average Brier Score:" in stripped:
            try:
                metrics["avg_brier_score"] = float(stripped.split(":")[-1].strip())
            except ValueError:
                pass
        elif "Matches evaluated:" in stripped:
            try:
                metrics["matches_evaluated"] = int(stripped.split(":")[-1].strip())
            except ValueError:
                pass
        elif "Average Edge (magnitude):" in stripped:
            try:
                metrics["avg_edge_pct"] = float(stripped.split(":")[-1].strip().rstrip("%"))
            except ValueError:
                pass
        elif "Average Signed Edge:" in stripped:
            try:
                val = stripped.split(":")[-1].strip().split("(")[0].strip().rstrip("%")
                metrics["avg_signed_edge_pct"] = float(val)
            except ValueError:
                pass

        # --- per-strategy metrics ---
        elif section and "Total P&L:" in stripped:
            try:
                val = stripped.split(":")[-1].strip().replace("units", "").strip()
                metrics[f"{section}_pnl"] = float(val)
            except ValueError:
                pass
        elif section and "ROI:" in stripped and "Flat ROI" not in stripped:
            try:
                val = stripped.split(":")[-1].strip().rstrip("%").strip()
                metrics[f"{section}_roi_pct"] = float(val)
            except ValueError:
                pass
        elif section and "Win Rate:" in stripped:
            try:
                val = stripped.split(":")[-1].strip().rstrip("%").strip()
                metrics[f"{section}_win_rate_pct"] = float(val)
            except ValueError:
                pass
        elif section and "Sharpe Ratio:" in stripped:
            try:
                val = stripped.split(":")[-1].strip().split()[0]
                metrics[f"{section}_sharpe"] = float(val)
            except ValueError:
                pass

    # Backward compat: also store top-level total_pnl / roi_pct from frac kelly
    if "frac_kelly_pnl" in metrics:
        metrics["total_pnl"] = metrics["frac_kelly_pnl"]
    if "frac_kelly_roi_pct" in metrics:
        metrics["roi_pct"] = metrics["frac_kelly_roi_pct"]

    return metrics


if __name__ == "__main__":
    main()
