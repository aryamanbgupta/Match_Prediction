#!/usr/bin/env python3
"""Build the exact production M7 feature frame from I7 materialization.

The full match materializer contains later experimental feature families.
This script selects the current production model's 46 numeric columns in
their exact order, preserves metadata, and carries the I7 venue contract.
That makes the retrain a one-variable comparison: venue identities changed;
the M7 architecture and feature set did not.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from identity_maps import assert_venue_alias_contract  # noqa: E402
from match_identity import identity_contract  # noqa: E402


METADATA = [
    "match_id",
    "cricsheet_id",
    "match_date",
    "team1",
    "team2",
    "venue",
    "competition_tier",
    "team1_wins",
]
OPTIONAL_IDENTITY_METADATA = [
    "display_match_id",
    "match_identity_version",
    "elo_update_version",
]
ENCODED = {"venue_id_encoded", "competition_tier_encoded"}
SPLITS = ("train", "validation", "test", "golden_test")


def build_frame(
    source_dir: Path,
    out_dir: Path,
    feature_columns_path: Path,
) -> dict[str, int]:
    identity_path = source_dir / "venue_identity.json"
    if not identity_path.exists():
        raise RuntimeError(
            f"{identity_path} is missing; materialize I7 match features first"
        )
    identity = json.loads(identity_path.read_text())
    assert_venue_alias_contract(
        identity,
        context="full I7 match materialization",
    )

    production_features = [
        value for value in feature_columns_path.read_text().splitlines()
        if value
    ]
    numeric_features = [
        value for value in production_features if value not in ENCODED
    ]
    if len(production_features) != 48 or len(numeric_features) != 46:
        raise RuntimeError(
            "unexpected M7 production feature contract: "
            f"{len(production_features)} total / "
            f"{len(numeric_features)} numeric"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for split in SPLITS:
        source_path = source_dir / f"{split}.parquet"
        if not source_path.exists():
            continue
        frame = pd.read_parquet(source_path)
        optional_identity = [
            column for column in OPTIONAL_IDENTITY_METADATA
            if column in frame
        ]
        columns = METADATA + optional_identity + numeric_features
        missing = [column for column in columns if column not in frame]
        if missing:
            raise RuntimeError(f"{split} is missing columns: {missing}")
        selected = frame[columns]
        selected.to_parquet(out_dir / f"{split}.parquet", index=False)
        counts[split] = len(selected)
        print(
            f"  {split}: {len(frame.columns)} -> {len(selected.columns)} "
            f"columns, {len(selected):,} rows"
        )

    (out_dir / "venue_identity.json").write_text(
        json.dumps(identity, indent=2) + "\n"
    )
    match_identity_path = source_dir / "match_identity.json"
    if match_identity_path.exists():
        match_identity = json.loads(match_identity_path.read_text())
        if match_identity != identity_contract():
            raise RuntimeError(
                f"{match_identity_path} has an unsupported contract"
            )
        (out_dir / "match_identity.json").write_text(
            json.dumps(match_identity, indent=2) + "\n"
        )
    elo_update_path = source_dir / "elo_update.json"
    if elo_update_path.exists():
        (out_dir / "elo_update.json").write_text(
            elo_update_path.read_text()
        )
    print(
        f"wrote {out_dir}: {len(METADATA)} metadata + "
        f"{len(numeric_features)} numeric columns"
    )
    return counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/xgb_match_data_i7_full"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/xgb_match_data_i7"),
    )
    parser.add_argument(
        "--feature-columns",
        type=Path,
        default=Path(
            "models/xgb_match_v3_m7_production/feature_columns.txt"
        ),
    )
    args = parser.parse_args()
    build_frame(args.source_dir, args.out_dir, args.feature_columns)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
