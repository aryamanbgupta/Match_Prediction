#!/usr/bin/env python3
"""Train the frozen five-seed direct-model I9 control and candidate."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SEEDS = (29, 7, 13, 42, 101)
ARMS = {
    "baseline": (
        Path("data/xgb_match_data_i9_baseline"),
        "xgb_match_i9_baseline",
    ),
    "candidate": (
        Path("data/xgb_match_data_i9"),
        "xgb_match_i9",
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arm",
        choices=("baseline", "candidate", "both"),
        default="both",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    arms = ARMS if args.arm == "both" else {args.arm: ARMS[args.arm]}
    for arm, (data_dir, model_prefix) in arms.items():
        for seed in SEEDS:
            model_dir = Path("models") / f"{model_prefix}_seed{seed}"
            command = [
                sys.executable,
                "scripts/xgboost_match_v1.py",
                "--cmd",
                "train",
                "--data-dir",
                str(data_dir),
                "--model-dir",
                str(model_dir),
                "--seed",
                str(seed),
                "--monotone",
            ]
            print(
                f"\n[I9 direct] arm={arm} seed={seed} "
                f"model={model_dir}",
                flush=True,
            )
            print("  " + " ".join(command), flush=True)
            if not args.dry_run:
                subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
