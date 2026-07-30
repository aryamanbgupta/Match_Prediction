"""Focused tests for the reproducible I8 paired evaluators."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from compare_i8_match_eval import (  # noqa: E402
    _cluster_bootstrap,
    _flat_pnl,
    _longshot_wins,
    _pair,
    _point_metrics,
)
from evaluate_i8_ball_model import _metrics, _paired_deltas  # noqa: E402


def _match(
    match_id: str,
    *,
    simulated_prob: float,
    bet_team: str | None,
    actual_winner: str,
    cluster: str,
    market_a: float = 0.25,
) -> dict:
    market = {"A": market_a, "B": 1.0 - market_a}
    odds = {"A": 1.0 / market_a, "B": 1.0 / (1.0 - market_a)}
    actual_probability = (
        simulated_prob if actual_winner == "A" else 1.0 - simulated_prob
    )
    return {
        "match_id": match_id,
        "actual_winner": actual_winner,
        "market_prob": market,
        "market_odds": odds,
        "simulated_prob": {
            "A": simulated_prob,
            "B": 1.0 - simulated_prob,
        },
        "log_loss": -np.log(actual_probability),
        "brier_score": 2.0 * (1.0 - actual_probability) ** 2,
        "bet_placed": bet_team is not None,
        "bet_team": bet_team,
        "realized_pnl": None if bet_team is None else 999.0,
        "competition_cluster_id": cluster,
    }


def test_flat_pnl_is_derived_from_market_and_outcome() -> None:
    winner = _match(
        "win",
        simulated_prob=0.7,
        bet_team="A",
        actual_winner="A",
        cluster="c1",
    )
    loser = {**winner, "match_id": "loss", "actual_winner": "B"}
    no_bet = {**winner, "match_id": "none", "bet_placed": False,
              "bet_team": None, "realized_pnl": None}

    assert _flat_pnl(winner) == pytest.approx(3.0)
    assert _flat_pnl(loser) == -1.0
    assert _flat_pnl(no_bet) == 0.0


def test_match_comparison_is_paired_and_cluster_bootstrapped() -> None:
    baseline = {
        "m1": _match(
            "m1",
            simulated_prob=0.55,
            bet_team="A",
            actual_winner="A",
            cluster="c1",
        ),
        "m2": _match(
            "m2",
            simulated_prob=0.45,
            bet_team="A",
            actual_winner="B",
            cluster="c2",
        ),
    }
    candidate = {
        "m1": _match(
            "m1",
            simulated_prob=0.65,
            bet_team="A",
            actual_winner="A",
            cluster="c1",
        ),
        "m2": _match(
            "m2",
            simulated_prob=0.35,
            bet_team="A",
            actual_winner="B",
            cluster="c2",
        ),
    }

    paired = _pair(baseline, candidate)
    points = _point_metrics(paired)
    bootstrap = _cluster_bootstrap(paired, n_resamples=100, seed=7)

    assert points["candidate_minus_baseline"]["log_loss"] < 0.0
    assert points["candidate_minus_baseline"]["brier"] < 0.0
    assert bootstrap["n_clusters"] == 2
    assert bootstrap["n_resamples"] == 100


def test_longshot_sensitivity_uses_candidate_bet_and_market_probability() -> None:
    rows = {
        "longshot": _match(
            "longshot",
            simulated_prob=0.6,
            bet_team="A",
            actual_winner="A",
            cluster="c1",
            market_a=0.05,
        ),
        "favorite": _match(
            "favorite",
            simulated_prob=0.8,
            bet_team="A",
            actual_winner="A",
            cluster="c2",
            market_a=0.75,
        ),
    }
    assert _longshot_wins(rows) == ["longshot"]


def test_ball_metrics_and_match_cluster_delta_use_same_rows() -> None:
    labels = np.array([0, 1, 0, 1], dtype=np.int8)
    baseline = np.array([
        [0.55, 0.45, 0, 0, 0, 0],
        [0.45, 0.55, 0, 0, 0, 0],
        [0.60, 0.40, 0, 0, 0, 0],
        [0.40, 0.60, 0, 0, 0, 0],
    ])
    candidate = np.array([
        [0.65, 0.35, 0, 0, 0, 0],
        [0.35, 0.65, 0, 0, 0, 0],
        [0.70, 0.30, 0, 0, 0, 0],
        [0.30, 0.70, 0, 0, 0, 0],
    ])
    match_ids = np.array(["m1", "m1", "m2", "m2"])

    metrics = _metrics(labels, candidate)
    delta = _paired_deltas(
        labels,
        baseline,
        candidate,
        match_ids,
        n_resamples=100,
        seed=9,
    )

    assert metrics["n_balls"] == 4
    assert metrics["accuracy"] == 1.0
    assert delta["n_matches"] == 2
    assert delta["log_loss"]["candidate_minus_baseline"] < 0.0
    assert (
        delta["multiclass_brier"]["candidate_minus_baseline"] < 0.0
    )
