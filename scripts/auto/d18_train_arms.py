#!/usr/bin/env python
"""D18: no-weights-adapted hyperparameter arms on the i7 frame.

D16 showed uniform-weight training cuts the i7 config (lr 0.2404,
n_estimators 444, swept under balanced weights) to best_iteration 24 —
the loss surface changed and the config is over-aggressive. This launcher
trains the D18 grid arms with `--no-class-weights` at gentler learning
rates, with n_estimators raised so early stopping (not the cap) chooses
the tree count. Selection is VAL-LL-ONLY (D8/E4 discipline): the winner
must beat D16's no-weights val mlogloss 1.4334 or D18 stops FAILED with
no sim eval.

Config extraction mirrors D16 exactly: `build_training_cmd` on the loaded
`xgb_i7_venue_identity.yaml` — zero hand-transcription; only
`learning_rate` / `n_estimators` are overridden per arm.

Usage:
    uv run python scripts/auto/d18_train_arms.py            # both arms
    uv run python scripts/auto/d18_train_arms.py --arm lr005
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from feature_registry import resolve_feature_list  # noqa: E402
from run_experiment import build_training_cmd  # noqa: E402

CONFIG_YAML = REPO / "experiments/configs/xgb_i7_venue_identity.yaml"
OUT_ROOT = REPO / "models/auto/d18"
RAW_DIR = REPO / "research/handoff/D18/raw"

# lr 0.2404 is NOT retrained: that arm is exactly D16's noweights model
# (models/auto/d16/noweights, val mlogloss 1.4334 at best_iteration 24).
ARMS = {
    "lr0025": {"learning_rate": 0.025, "n_estimators": 4000},
    "lr005": {"learning_rate": 0.05, "n_estimators": 2000},
    "lr010": {"learning_rate": 0.10, "n_estimators": 1000},
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=sorted(ARMS), default=None)
    args = ap.parse_args()

    config = yaml.safe_load(CONFIG_YAML.read_text())
    feature_list = resolve_feature_list(
        config["features"]["groups"],
        config["features"].get("exclude"),
        config["features"].get("include_extra"),
    )
    train_cmd = build_training_cmd(config, feature_list)
    config_json = json.loads(train_cmd[train_cmd.index("--config-json") + 1])
    base_hp = dict(config_json["model"]["hyperparameters"])
    print(f"base hyperparameters (from YAML, untouched elsewhere): {base_hp}")

    arms = [args.arm] if args.arm else sorted(ARMS)
    for arm in arms:
        arm_json = json.loads(json.dumps(config_json))
        arm_json["model"]["hyperparameters"].update(ARMS[arm])
        model_dir = OUT_ROOT / arm
        model_dir.mkdir(parents=True, exist_ok=True)
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        log = RAW_DIR / f"train_{arm}.log"
        cmd = [
            sys.executable,
            str(REPO / "scripts/xgboost_v2.py"),
            "--config-json",
            json.dumps(arm_json),
            "--no-class-weights",
            "--model-dir",
            str(model_dir),
        ]
        print(f"[{arm}] overrides={ARMS[arm]} -> {model_dir}  (log: {log})")
        with log.open("w") as fh:
            rc = subprocess.run(
                cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=REPO
            ).returncode
        print(f"[{arm}] exit {rc}")
        if rc != 0:
            return rc
    return 0


if __name__ == "__main__":
    sys.exit(main())
