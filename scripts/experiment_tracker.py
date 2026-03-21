"""
Experiment Tracker — Lightweight structured result storage using plain files.

Usage:
    tracker = ExperimentTracker()
    exp_id = tracker.start_experiment(config_dict)
    tracker.log_step_duration("training", 123.4)
    tracker.log_training_metrics({"accuracy": 0.558, "loss": 1.23})
    tracker.log_evaluation_results({"avg_log_loss": 0.65, "avg_brier": 0.22})
    tracker.finish_experiment()
"""

import json
import os
import platform
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


class ExperimentTracker:
    def __init__(self, results_dir: str = "experiments/results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._experiment_dir: Optional[Path] = None
        self._start_time: Optional[float] = None
        self._step_durations: Dict[str, float] = {}
        self._experiment_id: Optional[str] = None

    @staticmethod
    def _get_git_info() -> Dict[str, Any]:
        """Capture current git state."""
        info = {"hash": "unknown", "branch": "unknown", "dirty": False}
        try:
            info["hash"] = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            ).decode().strip()
            info["branch"] = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.DEVNULL,
            ).decode().strip()
            status = subprocess.check_output(
                ["git", "status", "--porcelain"],
                stderr=subprocess.DEVNULL,
            ).decode().strip()
            info["dirty"] = len(status) > 0
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        return info

    def start_experiment(self, config: Dict[str, Any]) -> str:
        """Start a new experiment. Returns the experiment ID."""
        name = config.get("experiment", {}).get("name", "unnamed")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        git_info = self._get_git_info()
        git_hash = git_info["hash"]

        self._experiment_id = f"{name}_{timestamp}_{git_hash}"
        self._experiment_dir = self.results_dir / self._experiment_id
        self._experiment_dir.mkdir(parents=True, exist_ok=True)
        self._start_time = time.time()
        self._step_durations = {}

        # Save config
        config_path = self._experiment_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        # Save initial metadata
        metadata = {
            "experiment_id": self._experiment_id,
            "name": name,
            "status": "running",
            "started_at": datetime.now().isoformat(),
            "git": git_info,
            "platform": {
                "system": platform.system(),
                "machine": platform.machine(),
                "python": platform.python_version(),
            },
            "tags": config.get("experiment", {}).get("tags", []),
        }
        self._write_json("metadata.json", metadata)

        return self._experiment_id

    def log_step_duration(self, step_name: str, seconds: float):
        """Record how long a pipeline step took."""
        self._step_durations[step_name] = seconds

    def log_training_metrics(self, metrics: Dict[str, Any]):
        """Save training metrics (accuracy, loss, etc.)."""
        self._write_json("training_metrics.json", metrics)

    def log_evaluation_results(self, results: Dict[str, Any]):
        """Save evaluation results."""
        self._write_json("metrics.json", results)

    def log_console_output(self, output: str):
        """Append console output."""
        if self._experiment_dir:
            log_path = self._experiment_dir / "console_output.log"
            with open(log_path, "a") as f:
                f.write(output)

    def finish_experiment(self, status: str = "completed"):
        """Finalize the experiment with timing and status."""
        if not self._experiment_dir:
            return

        # Update metadata
        metadata = self._read_json("metadata.json")
        metadata["status"] = status
        metadata["finished_at"] = datetime.now().isoformat()
        if self._start_time:
            metadata["total_duration_seconds"] = round(time.time() - self._start_time, 1)
        metadata["step_durations"] = self._step_durations
        self._write_json("metadata.json", metadata)

    @property
    def experiment_dir(self) -> Optional[Path]:
        return self._experiment_dir

    @property
    def experiment_id(self) -> Optional[str]:
        return self._experiment_id

    # ── Static methods for browsing experiments ──

    @staticmethod
    def list_experiments(results_dir: str = "experiments/results") -> List[Dict[str, Any]]:
        """List all experiments with summary info."""
        results_path = Path(results_dir)
        if not results_path.exists():
            return []

        experiments = []
        for exp_dir in sorted(results_path.iterdir()):
            if not exp_dir.is_dir():
                continue
            metadata_path = exp_dir / "metadata.json"
            if not metadata_path.exists():
                continue
            try:
                with open(metadata_path) as f:
                    metadata = json.load(f)

                summary = {
                    "id": exp_dir.name,
                    "name": metadata.get("name", "unknown"),
                    "status": metadata.get("status", "unknown"),
                    "started_at": metadata.get("started_at", ""),
                    "tags": metadata.get("tags", []),
                    "duration": metadata.get("total_duration_seconds"),
                    "git_hash": metadata.get("git", {}).get("hash", ""),
                }

                # Add key metrics if available
                metrics_path = exp_dir / "metrics.json"
                if metrics_path.exists():
                    with open(metrics_path) as f:
                        metrics = json.load(f)
                    summary["avg_log_loss"] = metrics.get("avg_log_loss")
                    summary["avg_brier_score"] = metrics.get("avg_brier_score")

                experiments.append(summary)
            except (json.JSONDecodeError, KeyError):
                continue

        return experiments

    @staticmethod
    def load_experiment(experiment_id: str,
                        results_dir: str = "experiments/results") -> Dict[str, Any]:
        """Load full experiment data."""
        exp_dir = Path(results_dir) / experiment_id
        if not exp_dir.exists():
            raise FileNotFoundError(f"Experiment not found: {experiment_id}")

        data: Dict[str, Any] = {"id": experiment_id}
        for filename in ["metadata.json", "metrics.json", "training_metrics.json", "config.yaml"]:
            filepath = exp_dir / filename
            if not filepath.exists():
                continue
            if filename.endswith(".json"):
                with open(filepath) as f:
                    data[filename.replace(".json", "")] = json.load(f)
            elif filename.endswith(".yaml"):
                with open(filepath) as f:
                    data["config"] = yaml.safe_load(f)

        return data

    # ── Internal helpers ──

    def _write_json(self, filename: str, data: Dict[str, Any]):
        if self._experiment_dir:
            path = self._experiment_dir / filename
            with open(path, "w") as f:
                json.dump(data, f, indent=2, default=str)

    def _read_json(self, filename: str) -> Dict[str, Any]:
        if self._experiment_dir:
            path = self._experiment_dir / filename
            if path.exists():
                with open(path) as f:
                    return json.load(f)
        return {}
