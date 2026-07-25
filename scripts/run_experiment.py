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
from parsing_v2 import (
    I5_DELIVERY_SEMANTICS,
    LEGACY_DELIVERY_SEMANTICS,
)


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


def _check_parquet_cache(config: dict, feature_list: list) -> bool:
    """Check if the parquet split files are current.

    Returns True iff ALL of:
      1. train parquet file exists
      2. `.feature_hash` JSON exists and parses
      3. cached `hash` matches the feature list
      4. cached `splits` matches YAML's effective splits (merge of
         data.splits over DEFAULT_SPLITS)
      5. cached `gender_filter` matches YAML's (None → 'all')
      6. parquet mtime >= SQLite mtime (if SQLite exists)

    Missing `splits`/`gender_filter` in the cached payload → legacy
    format → cache miss (forces one rematerialize; subsequent runs hit).
    """
    version = config["data"].get("version", "v3")
    data_dir = Path(f"data/xgb_data_{version}") if version != "v2" else Path("data/xgb_data")

    train_file = data_dir / f"cricket_data_{version}_train.parquet"
    if not train_file.exists():
        return False

    hash_file = data_dir / ".feature_hash"
    if not hash_file.exists():
        return False

    try:
        with open(hash_file) as f:
            cached = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False

    # Feature hash.
    if cached.get("hash") != get_feature_hash(feature_list):
        return False

    # Splits (must be present in cached payload and match effective splits).
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from loaders_common import effective_splits
    except ImportError:
        return False
    yaml_splits = config.get("data", {}).get("splits") or {}
    want_splits = effective_splits(yaml_splits)
    cached_splits = cached.get("splits")
    if cached_splits != want_splits:
        return False

    # Gender filter (canonicalized the same way materialize_features.py writes it).
    yaml_gender = config.get("data", {}).get("gender_filter", "male")
    want_gender = yaml_gender if yaml_gender else "all"
    cached_gender = cached.get("gender_filter")
    if cached_gender != want_gender:
        return False

    want_semantics = config.get("data", {}).get(
        "delivery_semantics", LEGACY_DELIVERY_SEMANTICS)
    cached_semantics = cached.get(
        "delivery_semantics", LEGACY_DELIVERY_SEMANTICS)
    if cached_semantics != want_semantics:
        return False

    # Phase 6: outcome_dist k_player / k_venue. parquet content depends
    # on these (shrinkage strength), so a k-sweep config must invalidate
    # the cache. None on either side defaults to 30/200 (matches the
    # materialize_features default).
    od_cfg = config.get("outcome_dist", {}) or {}
    want_k_player = float(od_cfg.get("k_player", 30.0))
    want_k_venue  = float(od_cfg.get("k_venue", 200.0))
    cached_k_player = cached.get("k_player")
    cached_k_venue  = cached.get("k_venue")
    if cached_k_player is not None and float(cached_k_player) != want_k_player:
        return False
    if cached_k_venue  is not None and float(cached_k_venue)  != want_k_venue:
        return False
    # Cached payloads from before Phase 6 lack these fields. Treat them
    # as 30/200 (the default at the time). If the YAML asks for the
    # default, it's a hit; otherwise a miss.
    if cached_k_player is None and want_k_player != 30.0:
        return False
    if cached_k_venue is None and want_k_venue != 200.0:
        return False

    # Parquet mtime >= SQLite mtime. A rebuild of SQLite invalidates the
    # parquet even if features/splits/gender are unchanged — the parquet
    # may contain snapshots derived from older SQLite state.
    sqlite_path = PROJECT_ROOT / "models" / f"player_stats_cache_{version}.sqlite"
    if sqlite_path.exists():
        if train_file.stat().st_mtime < sqlite_path.stat().st_mtime - 1:
            return False

    return True


def _check_sqlite_cache(config: dict) -> bool:
    """SQLite cache is current iff `_meta.schema_version` matches the
    live `stats_sqlite_backend.SCHEMA_VERSION` (currently 4) AND
    the deterministic same-day ordering contract matches AND the source
    membership/mtime matches the live JSON corpus. Returns False on any
    read error (missing file, bad schema, etc.)."""
    import sqlite3
    version = config["data"].get("version", "v3")
    sqlite_path = Path(f"models/player_stats_cache_{version}.sqlite")
    if not sqlite_path.exists():
        return False

    try:
        conn = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
        meta = dict(conn.execute("SELECT key, value FROM _meta"))
        conn.close()
    except sqlite3.DatabaseError:
        return False

    # Schema check — import lazily to avoid hard dep if Phase B backend missing
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from loaders_common import SAME_DAY_ORDER_VERSION
        from stats_sqlite_backend import SCHEMA_VERSION
    except ImportError:
        return False
    try:
        if int(meta.get("schema_version", -1)) != SCHEMA_VERSION:
            return False
    except (TypeError, ValueError):
        return False
    if meta.get("same_day_order_version") != SAME_DAY_ORDER_VERSION:
        return False
    want_semantics = config.get("data", {}).get(
        "delivery_semantics", LEGACY_DELIVERY_SEMANTICS)
    cached_semantics = meta.get(
        "delivery_semantics", LEGACY_DELIVERY_SEMANTICS)
    if cached_semantics != want_semantics:
        return False

    # JSON membership + mtime check — SQLite is stale if a file was
    # added/removed or any live JSON is newer.
    json_dir = PROJECT_ROOT / "data" / "t20s_json"
    if not json_dir.exists():
        # No JSONs to check against; trust the SQLite.
        return True
    json_files = list(json_dir.glob("*.json"))
    if not json_files:
        return True
    expected_sources = json.dumps(
        [str(json_dir.resolve())],
        separators=(",", ":"),
    )
    if meta.get("source_dirs_json") != expected_sources:
        return False
    live_mtime = max(p.stat().st_mtime for p in json_files)
    try:
        cached_mtime = float(meta.get("source_json_mtime_max", 0))
        cached_count = int(meta.get("source_json_file_count", -1))
    except (TypeError, ValueError):
        return False
    return (
        cached_mtime + 1 >= live_mtime
        and cached_count == len(json_files)
    )


def check_smart_cache(config: dict, feature_list: list) -> tuple[bool, bool]:
    """Return (sqlite_valid, parquet_valid). Phase B introduces two
    independent artifacts; the caller decides which steps to skip."""
    sqlite_valid = _check_sqlite_cache(config)
    # Parquet is downstream of SQLite. If the cache contract or source
    # membership invalidates SQLite, the replacement file will necessarily
    # be newer and semantically different; force materialization in the same
    # run instead of pairing new state with legacy-order feature rows.
    parquet_valid = (
        _check_parquet_cache(config, feature_list)
        if sqlite_valid
        else False
    )
    return sqlite_valid, parquet_valid


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
        # Phase 6: forward outcome_dist YAML block so xgboost_v2.py can
        # write the sidecar (k_player / k_venue) consumed by the sim
        # wrappers at inference time.
        "outcome_dist": config.get("outcome_dist", {}),
        "data": {
            "version": config["data"].get("version", "v3"),
            "delivery_semantics": config["data"].get(
                "delivery_semantics", LEGACY_DELIVERY_SEMANTICS),
            "source_dir": config["data"].get(
                "source_dir", "data/t20s_json"),
            "gender_filter": config["data"].get(
                "gender_filter", "male"),
            "splits": config["data"].get("splits", {}),
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

    # Liquidity slice (Phase 1 of outcome-dist follow-ups)
    if eval_config.get("min_volume") is not None:
        cmd.extend(["--min-volume", str(eval_config["min_volume"])])
    if eval_config.get("bootstrap_resamples") is not None:
        cmd.extend(["--bootstrap-resamples", str(eval_config["bootstrap_resamples"])])

    if eval_config.get("calibrate"):
        cmd.append("--calibrate")
    if eval_config.get("calibration_method"):
        cmd.extend(["--calibration-method", eval_config["calibration_method"]])
    if eval_config.get("ball_calibrate"):
        cmd.append("--ball-calibrate")
    if eval_config.get("ball_calibrate_data"):
        cmd.extend(["--ball-calibrate-data", eval_config["ball_calibrate_data"]])
    if eval_config.get("ball_calibrator_path"):
        cmd.extend([
            "--ball-calibrator-path",
            eval_config["ball_calibrator_path"],
        ])
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
    data_cfg = config.get("data", {})
    version = data_cfg.get("version", "v3")
    delivery_semantics = data_cfg.get(
        "delivery_semantics", LEGACY_DELIVERY_SEMANTICS)
    if (delivery_semantics == I5_DELIVERY_SEMANTICS
            and version in {"v2", "v3"}):
        raise ValueError(
            "I5 delivery semantics require an isolated data.version "
            "(for example 'i5')"
        )

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

    # Phase B pipeline: two independent artifacts (SQLite cache + parquet).
    sqlite_valid = parquet_valid = False
    if not skip_parsing:
        sqlite_valid, parquet_valid = check_smart_cache(config, feature_list)
        if sqlite_valid and parquet_valid:
            print("\n  Smart cache hit — SQLite + parquet both current. "
                  "Skipping parsing.")
            skip_parsing = True
        else:
            print(f"\n  Cache state: sqlite_valid={sqlite_valid}, "
                  f"parquet_valid={parquet_valid}")

    cache_cmd = [
        sys.executable,
        "scripts/build_stats_cache.py",
        "--out",
        f"models/player_stats_cache_{version}.sqlite",
        "--delivery-semantics",
        delivery_semantics,
    ]
    mat_cmd = [sys.executable, "scripts/materialize_features.py",
               "--config", args.config]
    train_cmd = build_training_cmd(config, feature_list)
    eval_cmd = build_eval_cmd(config)

    if args.dry_run:
        print("\n--- DRY RUN ---")
        if skip_parsing:
            print("[1/3] PARSE: SKIPPED (smart cache hit)")
        else:
            if sqlite_valid:
                print("[1a/3] BUILD_STATS_CACHE: SKIPPED (sqlite current)")
            else:
                print(f"[1a/3] BUILD_STATS_CACHE: {' '.join(cache_cmd)}")
            if parquet_valid:
                print("[1b/3] MATERIALIZE_FEATURES: SKIPPED (parquet current)")
            else:
                print(f"[1b/3] MATERIALIZE_FEATURES: {' '.join(mat_cmd)}")
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
        # Step 1: Parsing — two-artifact pipeline.
        if not skip_parsing:
            if not sqlite_valid:
                run_step(cache_cmd, "Build stats cache (JSON → SQLite)",
                         tracker, capture=False)
            else:
                print("\n  Stats cache current; skipping "
                      "build_stats_cache.py.")
            if not parquet_valid:
                run_step(mat_cmd,
                         "Materialize features (SQLite + JSON → parquet)",
                         tracker, capture=False)
            else:
                print("\n  Parquet current; skipping "
                      "materialize_features.py.")
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
