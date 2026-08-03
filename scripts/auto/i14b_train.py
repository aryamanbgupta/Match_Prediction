#!/usr/bin/env python
"""I14b: train the venue-physical arm on the i14b frame.

Same recipe as the promoted production ball model (D16 no-weights, i7
config extracted programmatically — zero hand-transcription) with exactly
two deltas: `data.version = i14b` (the additive vphys_* frame) and
`features.include_extra += the 10 vphys_* columns`. Hyperparameters stay at
the swept config per D18 (re-tuning does not transfer).

Run: uv run python scripts/auto/i14b_train.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from feature_registry import resolve_feature_list  # noqa: E402
from run_experiment import build_training_cmd  # noqa: E402

CONFIG_YAML = REPO / "experiments/configs/xgb_i7_venue_identity.yaml"
MODEL_DIR = REPO / "models/auto/i14b/venuephys"
RAW_DIR = REPO / "research/handoff/I14B/raw"

VPHYS_COLS = [
    "vphys_straight_mid_m", "vphys_square_mid_m", "vphys_boundary_known",
    "vphys_altitude_m", "vphys_altitude_known",
    "vphys_temp_c", "vphys_precip_mm", "vphys_windmax_kmh", "vphys_rh_pct",
    "vphys_climate_known",
]


def main() -> int:
    config = yaml.safe_load(CONFIG_YAML.read_text())
    feature_list = resolve_feature_list(
        config["features"]["groups"],
        config["features"].get("exclude"),
        config["features"].get("include_extra"),
    )
    train_cmd = build_training_cmd(config, feature_list)
    config_json = json.loads(train_cmd[train_cmd.index("--config-json") + 1])

    extra = list(config_json["features"].get("include_extra") or [])
    config_json["features"]["include_extra"] = extra + VPHYS_COLS
    config_json["data"]["version"] = "i14b"

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    log = RAW_DIR / "train_venuephys.log"
    cmd = [
        sys.executable, str(REPO / "scripts/xgboost_v2.py"),
        "--config-json", json.dumps(config_json),
        "--no-class-weights",
        "--model-dir", str(MODEL_DIR),
    ]
    print(f"training venuephys -> {MODEL_DIR} (log: {log})")
    with log.open("w") as fh:
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                            cwd=REPO).returncode
    print(f"exit {rc}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
