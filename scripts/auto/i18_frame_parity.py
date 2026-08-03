#!/usr/bin/env python3
"""I18 Gate A: parity check between a freshly rebuilt I7 match frame and the
frozen production I7 frame.

I18 re-materializes the I7 identity frame with the golden cricsheet pool
merged in (``--extra-source-dir data/golden/t20s_json``) so that a
``golden_test`` split appears. Golden matches (2026-04-17 onward) all
post-date the test split, so the train/validation/test rows must carry
identical FEATURES to ``data/xgb_match_data_i7/``.

Two gates are evaluated, in order:

STRICT PASS (informational)
    Full ``DataFrame.equals`` parity. This FAILS against the frozen frame
    built 2026-07-25, because ``materialize_match_features.py`` has since
    gained the I15/I16 match-identity columns. The frozen frame keys
    ``match_id`` by the legacy display string and has no
    ``display_match_id`` / ``match_identity_version`` / ``elo_update_version``.

RELAXED CONTRACT (the binding gate, per orchestrator ruling 1a)
    Holds iff ALL of:
      1. every shared column except ``match_id`` is bit-identical on
         train/validation/test, with identical row order;
      2. the candidate-only columns are EXACTLY the identity set
         {display_match_id, match_identity_version, elo_update_version}
         and there are no reference-only columns;
      3. the ``match_id`` semantic swap is verified both ways:
         reference ``match_id`` == candidate ``display_match_id`` on 100%
         of rows, and candidate ``match_id`` == candidate ``cricsheet_id``
         on 100% of rows;
      4. ``venue_identity.json`` is content-identical;
      5. ``golden_test.parquet`` carries ``cricsheet_id`` and
         ``match_identity_version``, with unique non-null cricsheet ids.

Passing the relaxed contract does NOT authorize copying
``golden_test.parquet`` into ``data/xgb_match_data_i7/``. The frozen
siblings predate I15, so a copy would create a mixed-contract directory —
the exact silent-join hazard I15 exists to prevent. Score directly from
``data/auto/i18/frame/golden_test.parquet`` instead (ruling 1b/1c).

This script fails closed: a relaxed-contract violation exits non-zero.

Usage:
    uv run python scripts/auto/i18_frame_parity.py \
        --candidate-dir data/auto/i18/frame \
        --reference-dir data/xgb_match_data_i7
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SPLITS = ("train", "validation", "test")
REQUIRED_GOLDEN_COLUMNS = ("cricsheet_id", "match_identity_version")
# The exact set of columns the I15/I16 match-identity contract adds on top of
# the pre-I15 frozen frame. Anything else appearing is a contract violation.
IDENTITY_COLUMNS = frozenset(
    {"display_match_id", "match_identity_version", "elo_update_version"}
)


def _describe_diff(cand: pd.DataFrame, ref: pd.DataFrame) -> list[str]:
    """Return human-readable lines describing where two frames differ."""
    lines: list[str] = []
    shared = [c for c in ref.columns if c in cand.columns]
    for column in shared:
        left = cand[column]
        right = ref[column]
        if left.equals(right):
            continue
        if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(
            right
        ):
            delta = np.abs(
                left.to_numpy(dtype="float64", na_value=np.nan)
                - right.to_numpy(dtype="float64", na_value=np.nan)
            )
            finite = delta[np.isfinite(delta)]
            max_delta = float(finite.max()) if finite.size else float("nan")
            n_diff = int((finite > 0).sum())
            lines.append(
                f"      column {column!r}: numeric mismatch, "
                f"max_abs_delta={max_delta!r}, n_rows_differing={n_diff}"
            )
        else:
            mask = left.astype(str) != right.astype(str)
            n_diff = int(mask.sum())
            examples = [
                (int(i), left.iloc[i], right.iloc[i])
                for i in np.flatnonzero(mask.to_numpy())[:3]
            ]
            lines.append(
                f"      column {column!r}: non-numeric mismatch, "
                f"n_rows_differing={n_diff}, examples(cand,ref)={examples}"
            )
    return lines


def check_split(candidate_dir: Path, reference_dir: Path, split: str) -> bool:
    cand_path = candidate_dir / f"{split}.parquet"
    ref_path = reference_dir / f"{split}.parquet"
    print(f"  [{split}]")
    if not cand_path.exists():
        print(f"    FAIL: candidate missing {cand_path}")
        return False
    if not ref_path.exists():
        print(f"    FAIL: reference missing {ref_path}")
        return False
    cand = pd.read_parquet(cand_path).reset_index(drop=True)
    ref = pd.read_parquet(ref_path).reset_index(drop=True)
    print(f"    candidate shape {cand.shape}  reference shape {ref.shape}")
    ok = True
    if cand.shape != ref.shape:
        print("    FAIL: shape mismatch")
        ok = False
    if list(cand.columns) != list(ref.columns):
        ok = False
        only_cand = [c for c in cand.columns if c not in set(ref.columns)]
        only_ref = [c for c in ref.columns if c not in set(cand.columns)]
        print("    FAIL: column list mismatch")
        print(f"      candidate-only: {only_cand}")
        print(f"      reference-only: {only_ref}")
        if not only_cand and not only_ref:
            print("      (same set, different order)")
    if ok and cand.equals(ref):
        print("    OK: content-identical (DataFrame.equals)")
        return True
    if ok:
        print("    FAIL: content differs")
        for line in _describe_diff(cand, ref):
            print(line)
        return False
    # shape/column mismatch already reported; still try a column-wise probe
    for line in _describe_diff(cand, ref):
        print(line)
    return False


def check_split_relaxed(
    candidate_dir: Path, reference_dir: Path, split: str
) -> bool:
    """The binding gate: features bit-identical, identity columns added."""
    cand_path = candidate_dir / f"{split}.parquet"
    ref_path = reference_dir / f"{split}.parquet"
    print(f"  [{split}]")
    if not cand_path.exists() or not ref_path.exists():
        print("    FAIL: missing parquet on one side")
        return False
    cand = pd.read_parquet(cand_path).reset_index(drop=True)
    ref = pd.read_parquet(ref_path).reset_index(drop=True)
    ok = True

    if len(cand) != len(ref):
        print(f"    FAIL: row count {len(cand)} != {len(ref)}")
        return False

    # (2) added columns are exactly the identity set, nothing removed
    only_cand = set(cand.columns) - set(ref.columns)
    only_ref = set(ref.columns) - set(cand.columns)
    if only_ref:
        print(f"    FAIL: reference-only columns present: {sorted(only_ref)}")
        ok = False
    if only_cand != IDENTITY_COLUMNS:
        print(
            f"    FAIL: candidate-only columns {sorted(only_cand)} != "
            f"identity set {sorted(IDENTITY_COLUMNS)}"
        )
        ok = False
    else:
        print(f"    OK: added columns are exactly {sorted(IDENTITY_COLUMNS)}")

    # (3) match_id semantic swap, verified both ways
    if "display_match_id" in cand:
        same = (
            cand["display_match_id"].astype(str) == ref["match_id"].astype(str)
        )
        n = int(same.sum())
        print(
            f"    reference.match_id == candidate.display_match_id on "
            f"{n}/{len(cand)} rows"
        )
        if n != len(cand):
            print("    FAIL: display-id swap not exact")
            ok = False
    else:
        print("    FAIL: candidate has no display_match_id")
        ok = False
    same_cid = cand["match_id"].astype(str) == cand["cricsheet_id"].astype(str)
    n_cid = int(same_cid.sum())
    print(
        f"    candidate.match_id == candidate.cricsheet_id on "
        f"{n_cid}/{len(cand)} rows"
    )
    if n_cid != len(cand):
        print("    FAIL: candidate match_id is not the cricsheet id")
        ok = False

    # row alignment
    cid_c = cand["cricsheet_id"].astype(str)
    cid_r = ref["cricsheet_id"].astype(str)
    order_equal = bool((cid_c == cid_r).all())
    print(
        f"    cricsheet_id set equal: {set(cid_c) == set(cid_r)}  "
        f"order equal: {order_equal}"
    )
    if not order_equal:
        print("    FAIL: row order differs")
        ok = False

    # (1) every shared column except match_id is bit-identical
    shared = [c for c in ref.columns if c in set(cand.columns)
              and c != "match_id"]
    diffs = _describe_diff(cand[shared], ref[shared])
    print(f"    shared columns compared (excl. match_id): {len(shared)}")
    if diffs:
        print("    FAIL: differing columns:")
        for line in diffs:
            print(line)
        ok = False
    else:
        print("    OK: all shared columns (excl. match_id) bit-identical")
    return ok


def check_venue_identity(candidate_dir: Path, reference_dir: Path) -> bool:
    print("  [venue_identity.json]")
    cand_path = candidate_dir / "venue_identity.json"
    ref_path = reference_dir / "venue_identity.json"
    if not cand_path.exists() or not ref_path.exists():
        print(f"    FAIL: missing ({cand_path.exists()=}, {ref_path.exists()=})")
        return False
    cand = json.loads(cand_path.read_text())
    ref = json.loads(ref_path.read_text())
    if cand != ref:
        print("    FAIL: venue identity contracts differ")
        print(f"      candidate: {json.dumps(cand, sort_keys=True)}")
        print(f"      reference: {json.dumps(ref, sort_keys=True)}")
        return False
    print(f"    OK: {json.dumps(cand, sort_keys=True)}")
    return True


def check_golden(candidate_dir: Path) -> bool:
    print("  [golden_test.parquet]")
    path = candidate_dir / "golden_test.parquet"
    if not path.exists():
        print(f"    FAIL: {path} does not exist")
        return False
    frame = pd.read_parquet(path)
    print(f"    rows={len(frame)}  columns={len(frame.columns)}")
    missing = [c for c in REQUIRED_GOLDEN_COLUMNS if c not in frame.columns]
    if missing:
        print(f"    FAIL: missing required identity columns: {missing}")
        return False
    versions = sorted(set(frame["match_identity_version"].astype(str)))
    n_null_cricsheet = int(frame["cricsheet_id"].isna().sum())
    n_unique_cricsheet = int(frame["cricsheet_id"].nunique())
    print(f"    match_identity_version values: {versions}")
    print(
        f"    cricsheet_id: {n_unique_cricsheet} unique, "
        f"{n_null_cricsheet} null"
    )
    if n_null_cricsheet:
        print("    FAIL: null cricsheet_id present")
        return False
    if n_unique_cricsheet != len(frame):
        print("    FAIL: cricsheet_id is not unique per row")
        return False
    if frame.empty:
        print("    FAIL: golden_test is empty")
        return False
    dates = frame["match_date"].astype(str)
    print(f"    match_date range: {dates.min()} -> {dates.max()}")
    print("    OK")
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate-dir", type=Path, default=Path("data/auto/i18/frame")
    )
    parser.add_argument(
        "--reference-dir", type=Path, default=Path("data/xgb_match_data_i7")
    )
    args = parser.parse_args()

    print(f"candidate: {args.candidate_dir}")
    print(f"reference: {args.reference_dir}")
    print()
    print("=" * 70)
    print("PASS 1 — STRICT parity (informational; expected to FAIL against a")
    print("         pre-I15 frozen frame that lacks the identity columns)")
    print("=" * 70)
    strict = {
        split: check_split(args.candidate_dir, args.reference_dir, split)
        for split in SPLITS
    }
    strict_ok = all(strict.values())
    print()
    print(f"STRICT PARITY: {'PASS' if strict_ok else 'FAIL'}")

    print()
    print("=" * 70)
    print("PASS 2 — RELAXED identity contract (the BINDING gate)")
    print("=" * 70)
    results = {
        split: check_split_relaxed(args.candidate_dir, args.reference_dir, split)
        for split in SPLITS
    }
    results["venue_identity"] = check_venue_identity(
        args.candidate_dir, args.reference_dir
    )
    results["golden_test"] = check_golden(args.candidate_dir)

    print()
    for name, ok in results.items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    all_ok = all(results.values())
    print()
    print(f"I18 RELAXED PARITY GATE: {'PASS' if all_ok else 'FAIL'}")
    print(
        "NOTE: passing does NOT authorize copying golden_test.parquet into "
        "the frozen I7 frame."
    )
    print(
        "      Score directly from "
        f"{args.candidate_dir / 'golden_test.parquet'} (ruling 1b/1c)."
    )
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
