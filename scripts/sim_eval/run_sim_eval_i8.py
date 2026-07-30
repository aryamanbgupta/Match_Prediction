#!/usr/bin/env python3
"""Run match simulation with the fail-closed, schema-v5 I8 adapter."""

from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_sim_eval as runner  # noqa: E402
from sim_i8 import FailClosedXGBoostModelI8  # noqa: E402


def _arg_value(flag: str, default: str) -> str:
    try:
        return sys.argv[sys.argv.index(flag) + 1]
    except (ValueError, IndexError):
        return default


def main() -> None:
    model_type = _arg_value("--model-type", "xgboost")
    model_version = _arg_value("--model-version", "v3")
    if model_type != "xgboost" or model_version != "i8":
        raise SystemExit(
            "run_sim_eval_i8.py requires "
            "--model-type xgboost --model-version i8"
        )

    # The base runner catches ordinary model-load errors and swaps in a dummy
    # model. Its I8 replacement raises SystemExit instead, so an incompatible
    # cache/model/feature contract stops the evaluation.
    runner.XGBoostModelV2 = FailClosedXGBoostModelI8
    runner.main()


if __name__ == "__main__":
    main()
