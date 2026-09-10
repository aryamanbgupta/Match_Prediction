"""Contracts for future player xR validation."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"

from run_xr_future_player_validation import (  # noqa: E402
    build_examples, estimates, reject_sealed)


def _rows() -> pd.DataFrame:
    rows = []
    for match, date in (("m1", "2025-01-01"), ("m2", "2025-01-02"),
                        ("m3", "2025-01-03")):
        for ball in range(4):
            actual = float(ball % 2)
            rows.append({
                "batter_id": "p1", "match_id": match,
                "match_date": date, "actual": actual,
                "context": 0.4, "shot": 0.5,
                "context_residual": actual - 0.4,
                "execution_residual": actual - 0.5,
            })
    return pd.DataFrame(rows)


def test_future_window_starts_at_strictly_later_match():
    examples = build_examples(_rows(), [4], horizon=4)[4]
    assert len(examples) == 1
    assert examples.iloc[0]["n_prior"] == 4
    assert examples.iloc[0]["cutoff_date"] == "2025-01-02"
    assert examples.iloc[0]["future_n"] == 4
    assert examples.iloc[0]["prior_actual"] == pytest.approx(0.5)
    assert examples.iloc[0]["future_actual"] == pytest.approx(0.5)


def test_delta_shrinks_execution_residual_around_shot_xr():
    frame = pd.DataFrame({
        "n_prior": [100], "prior_actual": [1.4],
        "prior_context": [1.2], "prior_context_residual": [0.2],
        "prior_shot": [1.5], "prior_execution_residual": [-0.1],
    })
    result = estimates(frame, k=100, mean_actual=1.3)
    assert result["eb"][0] == pytest.approx(1.35)
    assert result["raa"][0] == pytest.approx(1.3)
    assert result["delta"][0] == pytest.approx(1.45)
    assert result["xr"][0] == pytest.approx(1.5)


def test_sealed_paths_fail_closed():
    with pytest.raises(SystemExit, match="refusing sealed path"):
        reject_sealed("data/golden/player_rows.parquet")
