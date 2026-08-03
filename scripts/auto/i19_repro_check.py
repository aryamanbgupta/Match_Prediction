#!/usr/bin/env python3
"""I19 gate 2: prove retrains on the coherent-contract twin frame reproduce
the archived I17 successor arms EXACTLY.

For each arm in {base, swap} compare
``models/auto/i19/<arm>_seed29`` against ``models/auto/i17/<arm>_seed29``:

  * ``test_predictions.json`` key sets identical (798 cricsheet-primary IDs);
  * ``max |Δ p_team1|`` printed in scientific notation — the gate needs
    exactly ``0.000e+00``;
  * ``feature_columns.txt`` byte-equal (48 lines);
  * val/test log loss printed side by side from both ``train_metrics.json``.

Fails closed: any nonzero delta, key-set mismatch, or feature-column
difference exits non-zero.

Usage:
    uv run python scripts/auto/i19_repro_check.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ARMS = ("base", "swap")
EXPECTED_N_KEYS = 798
EXPECTED_N_FEATURES = 48


def _load_predictions(path: Path) -> dict:
    with open(path) as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"{path} is not a dict-keyed prediction artifact")
    return data


def check_arm(arm: str, candidate_root: Path, reference_root: Path) -> bool:
    cand_dir = candidate_root / f"{arm}_seed29"
    ref_dir = reference_root / f"{arm}_seed29"
    print(f"  [{arm}_seed29]")
    print(f"    candidate: {cand_dir}")
    print(f"    reference: {ref_dir}")
    ok = True

    cand_pred_path = cand_dir / "test_predictions.json"
    ref_pred_path = ref_dir / "test_predictions.json"
    if not cand_pred_path.exists() or not ref_pred_path.exists():
        print(
            f"    FAIL: missing predictions "
            f"({cand_pred_path.exists()=}, {ref_pred_path.exists()=})"
        )
        return False

    cand = _load_predictions(cand_pred_path)
    ref = _load_predictions(ref_pred_path)
    print(f"    n_keys candidate={len(cand)}  reference={len(ref)}")
    if len(cand) != EXPECTED_N_KEYS or len(ref) != EXPECTED_N_KEYS:
        print(f"    FAIL: expected {EXPECTED_N_KEYS} keys on both sides")
        ok = False
    if set(cand) != set(ref):
        only_cand = sorted(set(cand) - set(ref))[:5]
        only_ref = sorted(set(ref) - set(cand))[:5]
        print("    FAIL: key sets differ")
        print(f"      candidate-only (first 5): {only_cand}")
        print(f"      reference-only (first 5): {only_ref}")
        return False
    print("    OK: key sets identical")

    max_delta = 0.0
    arg_max_key = None
    n_differing = 0
    for key in cand:
        delta = abs(
            float(cand[key]["p_team1"]) - float(ref[key]["p_team1"])
        )
        if delta > 0.0:
            n_differing += 1
        if delta > max_delta:
            max_delta = delta
            arg_max_key = key
    print(f"    n_keys={len(cand)}")
    print(f"    max |delta p_team1| = {max_delta:.3e}")
    print(f"    rows with nonzero delta: {n_differing}")
    if arg_max_key is not None:
        print(
            f"    argmax key={arg_max_key}  "
            f"candidate={cand[arg_max_key]['p_team1']!r}  "
            f"reference={ref[arg_max_key]['p_team1']!r}"
        )
    if max_delta != 0.0:
        print("    FAIL: predictions are not bit-identical")
        ok = False
    else:
        print("    OK: predictions reproduce exactly (max |delta p| = 0)")

    cand_fc = cand_dir / "feature_columns.txt"
    ref_fc = ref_dir / "feature_columns.txt"
    if not cand_fc.exists() or not ref_fc.exists():
        print("    FAIL: missing feature_columns.txt on one side")
        ok = False
    else:
        cand_bytes = cand_fc.read_bytes()
        ref_bytes = ref_fc.read_bytes()
        n_cand_lines = len(cand_bytes.decode().splitlines())
        n_ref_lines = len(ref_bytes.decode().splitlines())
        equal = cand_bytes == ref_bytes
        print(
            f"    feature_columns.txt: candidate {n_cand_lines} lines, "
            f"reference {n_ref_lines} lines, byte-equal={equal}"
        )
        if not equal:
            print("    FAIL: feature column lists differ")
            ok = False
        if n_cand_lines != EXPECTED_N_FEATURES:
            print(f"    FAIL: expected {EXPECTED_N_FEATURES} feature lines")
            ok = False

    cand_metrics = json.loads((cand_dir / "train_metrics.json").read_text())
    ref_metrics = json.loads((ref_dir / "train_metrics.json").read_text())
    print("    train_metrics.json (candidate vs reference):")
    for field in (
        "val_log_loss",
        "test_log_loss",
        "val_brier",
        "test_brier",
        "n_train",
        "n_val",
        "n_test",
        "seed",
    ):
        print(
            f"      {field}: {cand_metrics.get(field)!r}  |  "
            f"{ref_metrics.get(field)!r}"
        )
    for field in ("val_log_loss", "test_log_loss"):
        if cand_metrics.get(field) != ref_metrics.get(field):
            print(f"    NOTE: {field} differs between candidate and reference")

    print(f"    {arm}_seed29: {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate-root", type=Path, default=Path("models/auto/i19")
    )
    parser.add_argument(
        "--reference-root", type=Path, default=Path("models/auto/i17")
    )
    args = parser.parse_args()

    print("=" * 70)
    print("I19 REPRODUCTION CHECK — i7_v2 retrains vs archived I17 arms")
    print("=" * 70)
    results = {
        arm: check_arm(arm, args.candidate_root, args.reference_root)
        for arm in ARMS
    }
    print()
    for name, ok in results.items():
        print(f"  {name}_seed29: {'PASS' if ok else 'FAIL'}")
    all_ok = all(results.values())
    print()
    print(f"I19 REPRODUCTION GATE: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
