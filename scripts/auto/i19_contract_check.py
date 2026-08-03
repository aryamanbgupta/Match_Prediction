#!/usr/bin/env python3
"""I19 gate 3: prove the coherent-contract I7 twin frame carries the full
I15/I16 match-identity contract on every split.

``data/xgb_match_data_i7_v2`` is a byte-copy of I18's parity-verified
production-subset frame (``data/auto/i18/frame``). This script asserts that
the copy is a *cricsheet-primary* frame end to end:

  * ``match_id == cricsheet_id`` on 100% of rows in every split;
  * ``display_match_id`` present, non-null (uniqueness NOT required — the
    display string collides on same-day doubleheaders by design);
  * ``match_identity_version`` constant ``cricsheet_primary_v1``;
  * ``elo_update_version`` constant ``fixed_competition_k_v1``;
  * row counts train 7972 / validation 528 / test 798 / golden_test 227;
  * ``match_id`` unique per row in every split;

and that the three sidecars agree with the code contract:

  * ``venue_identity.json`` byte-equal to the frozen I7 frame's sibling;
  * ``match_identity.json`` == ``scripts/match_identity.identity_contract()``;
  * ``elo_update.json`` == ``{"elo_update_version": "fixed_competition_k_v1"}``.

Fails closed: any violation exits non-zero.

Usage:
    uv run python scripts/auto/i19_contract_check.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from match_identity import identity_contract  # noqa: E402

EXPECTED_ROWS = {
    "train": 7972,
    "validation": 528,
    "test": 798,
    "golden_test": 227,
}
EXPECTED_MATCH_IDENTITY_VERSION = "cricsheet_primary_v1"
EXPECTED_ELO_UPDATE_VERSION = "fixed_competition_k_v1"
EXPECTED_ELO_SIDECAR = {"elo_update_version": EXPECTED_ELO_UPDATE_VERSION}


def check_split(frame_dir: Path, split: str) -> bool:
    path = frame_dir / f"{split}.parquet"
    print(f"  [{split}]")
    if not path.exists():
        print(f"    FAIL: missing {path}")
        return False
    df = pd.read_parquet(path).reset_index(drop=True)
    n = len(df)
    ok = True

    expected_n = EXPECTED_ROWS[split]
    print(f"    rows={n} (expected {expected_n})  columns={len(df.columns)}")
    if n != expected_n:
        print(f"    FAIL: row count {n} != expected {expected_n}")
        ok = False

    required = [
        "match_id",
        "cricsheet_id",
        "display_match_id",
        "match_identity_version",
        "elo_update_version",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"    FAIL: missing identity columns {missing}")
        return False

    n_primary = int(
        (df["match_id"].astype(str) == df["cricsheet_id"].astype(str)).sum()
    )
    print(f"    match_id == cricsheet_id on {n_primary}/{n} rows")
    if n_primary != n:
        print("    FAIL: frame is not cricsheet-primary")
        ok = False

    n_null_display = int(df["display_match_id"].isna().sum())
    n_blank_display = int(
        (df["display_match_id"].astype(str).str.strip() == "").sum()
    )
    n_unique_display = int(df["display_match_id"].astype(str).nunique())
    print(
        f"    display_match_id: {n_unique_display} unique, "
        f"{n_null_display} null, {n_blank_display} blank"
    )
    if n_null_display or n_blank_display:
        print("    FAIL: display_match_id has null/blank values")
        ok = False
    # Orchestrator ruling 2026-07-31: display_match_id uniqueness is NOT part
    # of the identity contract — match_identity.py documents the display
    # string as non-unique for same-day doubleheaders ("must never be used as
    # a new primary key"). The primary-key property is match_id uniqueness,
    # asserted below. Collision count stays informational.
    if n_unique_display != n:
        print(
            f"    note: display_match_id has {n - n_unique_display} "
            "doubleheader collision rows (informational, contract-consistent)"
        )

    mi_values = sorted(set(df["match_identity_version"].astype(str)))
    print(f"    match_identity_version values: {mi_values}")
    if mi_values != [EXPECTED_MATCH_IDENTITY_VERSION]:
        print(
            f"    FAIL: expected constant "
            f"[{EXPECTED_MATCH_IDENTITY_VERSION!r}]"
        )
        ok = False

    elo_values = sorted(set(df["elo_update_version"].astype(str)))
    print(f"    elo_update_version values: {elo_values}")
    if elo_values != [EXPECTED_ELO_UPDATE_VERSION]:
        print(
            f"    FAIL: expected constant [{EXPECTED_ELO_UPDATE_VERSION!r}]"
        )
        ok = False

    n_unique_match_id = int(df["match_id"].astype(str).nunique())
    n_null_match_id = int(df["match_id"].isna().sum())
    print(
        f"    match_id: {n_unique_match_id} unique, "
        f"{n_null_match_id} null"
    )
    if n_unique_match_id != n or n_null_match_id:
        print("    FAIL: match_id is not unique/non-null per row")
        ok = False

    print(f"    {split}: {'PASS' if ok else 'FAIL'}")
    return ok


def check_venue_identity(frame_dir: Path, reference_dir: Path) -> bool:
    print("  [venue_identity.json]")
    cand = frame_dir / "venue_identity.json"
    ref = reference_dir / "venue_identity.json"
    if not cand.exists() or not ref.exists():
        print(f"    FAIL: missing ({cand.exists()=}, {ref.exists()=})")
        return False
    cand_bytes = cand.read_bytes()
    ref_bytes = ref.read_bytes()
    equal = cand_bytes == ref_bytes
    print(f"    candidate bytes={len(cand_bytes)}  reference bytes={len(ref_bytes)}")
    print(f"    byte-equal to {ref}: {equal}")
    print(f"    content: {json.dumps(json.loads(cand_bytes), sort_keys=True)}")
    if not equal:
        print(f"    reference content: "
              f"{json.dumps(json.loads(ref_bytes), sort_keys=True)}")
        print("    FAIL: venue identity sidecars differ")
        return False
    print("    PASS")
    return True


def check_match_identity(frame_dir: Path) -> bool:
    print("  [match_identity.json]")
    path = frame_dir / "match_identity.json"
    if not path.exists():
        print(f"    FAIL: missing {path}")
        return False
    observed = json.loads(path.read_text())
    expected = identity_contract()
    print(f"    observed: {json.dumps(observed, sort_keys=True)}")
    print(f"    expected: {json.dumps(expected, sort_keys=True)}")
    if observed != expected:
        print("    FAIL: does not equal match_identity.identity_contract()")
        return False
    print("    PASS")
    return True


def check_elo_update(frame_dir: Path) -> bool:
    print("  [elo_update.json]")
    path = frame_dir / "elo_update.json"
    if not path.exists():
        print(f"    FAIL: missing {path}")
        return False
    observed = json.loads(path.read_text())
    print(f"    observed: {json.dumps(observed, sort_keys=True)}")
    print(f"    expected: {json.dumps(EXPECTED_ELO_SIDECAR, sort_keys=True)}")
    if observed != EXPECTED_ELO_SIDECAR:
        print("    FAIL: elo update sidecar mismatch")
        return False
    print("    PASS")
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frame-dir", type=Path, default=Path("data/xgb_match_data_i7_v2")
    )
    parser.add_argument(
        "--reference-dir", type=Path, default=Path("data/xgb_match_data_i7")
    )
    args = parser.parse_args()

    print(f"frame:     {args.frame_dir}")
    print(f"reference: {args.reference_dir}")
    print()
    print("=" * 70)
    print("I19 CONTRACT CHECK — cricsheet-primary identity on every split")
    print("=" * 70)

    results = {
        split: check_split(args.frame_dir, split) for split in EXPECTED_ROWS
    }
    results["venue_identity"] = check_venue_identity(
        args.frame_dir, args.reference_dir
    )
    results["match_identity"] = check_match_identity(args.frame_dir)
    results["elo_update"] = check_elo_update(args.frame_dir)

    print()
    for name, ok in results.items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    all_ok = all(results.values())
    print()
    print(f"I19 CONTRACT GATE: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
