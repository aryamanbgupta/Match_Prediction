"""Tests for the frozen I9 paired development gate."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_i9_provisional_elo import evaluate_ball_gate  # noqa: E402


def _frame(true_probability: float) -> pd.DataFrame:
    rows = []
    for index, (match_id, exposure) in enumerate(
        [("m1", 0), ("m1", 140), ("m2", 20), ("m2", 180)]
    ):
        rows.append({
            "split_row": index,
            "innings_id": f"1_{match_id}",
            "ball_idx": index,
            "batter_elo_exposure": exposure,
            "bowler_elo_exposure": exposure,
            "target": 0,
            "prob_class_0": true_probability,
            "prob_class_1": 1.0 - true_probability,
        })
    return pd.DataFrame(rows)


def test_i9_gate_uses_frozen_slices_and_candidate_minus_baseline():
    baseline = _frame(0.7)
    candidate = _frame(0.7)
    provisional = candidate["batter_elo_exposure"] < 120
    candidate.loc[provisional, ["prob_class_0", "prob_class_1"]] = [
        0.8,
        0.2,
    ]

    report = evaluate_ball_gate(
        baseline,
        candidate,
        replicates=500,
    )

    assert report["counts"]["provisional_balls"] == 2
    assert report["counts"]["established_balls"] == 2
    assert (
        report["primary"]["provisional_log_loss"]["delta"] < 0
    )
    assert (
        report["primary"]["provisional_log_loss"]
        ["paired_match_block_ci95"][1] < 0
    )
    assert report["guardrails"]["established_log_loss"]["delta"] == 0
    assert report["primary"]["provisional_log_loss"]["passed"] is True


def test_i9_gate_rejects_pairing_drift():
    baseline = _frame(0.7)
    candidate = _frame(0.8)
    candidate.loc[0, "innings_id"] = "1_wrong"

    try:
        evaluate_ball_gate(baseline, candidate, replicates=100)
    except ValueError as exc:
        assert "pairing mismatch" in str(exc)
    else:
        raise AssertionError("pairing drift should fail closed")
