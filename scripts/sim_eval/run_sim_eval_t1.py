#!/usr/bin/env python3
"""Run match simulation with the T1 innings transformer (candidate runner).

i8 pattern: never edits the frozen runner — imports it and swaps the
TransformerModelV1 class for TransformerT1SimModel, then delegates.
Invoke with --model-type transformer; all other run_sim_eval flags pass
through. Checkpoint dir override: T1_SIM_MODEL_DIR env var.
"""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_sim_eval as runner  # noqa: E402
from sim_t1 import TransformerT1SimModel  # noqa: E402


def _arg_value(flag: str, default: str) -> str:
    try:
        return sys.argv[sys.argv.index(flag) + 1]
    except (ValueError, IndexError):
        return default


def main() -> None:
    if _arg_value("--model-type", "xgboost") != "transformer":
        raise SystemExit("run_sim_eval_t1.py requires --model-type transformer")
    runner.TransformerModelV1 = TransformerT1SimModel
    runner.main()


if __name__ == "__main__":
    main()
