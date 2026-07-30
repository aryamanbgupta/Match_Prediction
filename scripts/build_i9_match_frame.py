#!/usr/bin/env python3
"""Build the exact M7 direct-model frame for the I9 candidate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_i7_match_frame import build_frame
from elo_update import (
    PROVISIONAL_ELO_UPDATE_VERSION,
    assert_elo_update_version,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/xgb_match_data_i9_full"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/xgb_match_data_i9"),
    )
    parser.add_argument(
        "--feature-columns",
        type=Path,
        default=Path(
            "models/xgb_match_v3_m7_production/feature_columns.txt"
        ),
    )
    args = parser.parse_args()

    elo_path = args.source_dir / "elo_update.json"
    if not elo_path.exists():
        raise RuntimeError(
            f"{elo_path} is missing; materialize I9 match features first"
        )
    assert_elo_update_version(
        json.loads(elo_path.read_text()),
        expected=PROVISIONAL_ELO_UPDATE_VERSION,
        context="full I9 match materialization",
    )
    build_frame(args.source_dir, args.out_dir, args.feature_columns)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
