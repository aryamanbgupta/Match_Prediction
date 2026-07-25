"""Contracts for the I13 usage-aware prop fair baseline."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim_eval"))

from prop_fair_baselines import AsOf, baseline_rows, poisson_at_least


def _logs(usage, bowler=None):
    return {
        "batter": {},
        "bowler": bowler or {},
        "bowling_usage": usage,
        "venue_inn": {},
        "venue_match": {},
        "pos_top": [],
    }


def test_poisson_tail_matches_closed_form_and_is_monotone():
    probabilities = [poisson_at_least(1.0, threshold)
                     for threshold in (1, 2, 3)]
    assert probabilities == pytest.approx([
        0.6321205588,
        0.2642411177,
        0.0803013971,
    ])
    assert probabilities[0] > probabilities[1] > probabilities[2]
    assert poisson_at_least(1.0, 0) == 1.0
    assert poisson_at_least(-1.0, 1) == 0.0


def test_usage_history_is_strictly_before_match_date():
    asof = AsOf(_logs({
        "Player": [
            ("2024-01-01", 12, 1),
            ("2024-01-02", 24, 2),
        ],
    }))
    assert asof.player_bowling_usage("Player", "2024-01-02") == (1, 12, 1)
    assert asof.player_bowling_usage("Player", "2024-01-03") == (2, 36, 3)


def test_zero_ball_appearances_reduce_expected_usage():
    non_bowler = [("2024-01-%02d" % day, 0, 0) for day in range(1, 11)]
    frontline = [("2024-01-%02d" % day, 12, 1) for day in range(1, 11)]
    asof = AsOf(_logs({"Non Bowler": non_bowler, "Frontline": frontline}))

    non_bowler_balls, _ = asof.bowling_expectation(
        "Non Bowler", "2024-02-01")
    debutant_balls, _ = asof.bowling_expectation(
        "Debutant", "2024-02-01")
    frontline_balls, _ = asof.bowling_expectation(
        "Frontline", "2024-02-01")

    assert non_bowler_balls < debutant_balls < frontline_balls


def test_top_bowler_prices_normalize_usage_within_team():
    non_bowler = [("2024-01-%02d" % day, 0, 0) for day in range(1, 11)]
    frontline = [("2024-01-%02d" % day, 12, 1) for day in range(1, 11)]
    asof = AsOf(_logs({"Non Bowler": non_bowler, "Frontline": frontline}))
    detail = [{
        "match_id": "2024-02-01_A_B_Unknown",
        "obs": {
            "top_bowler": [
                {"team": "A", "name": "Non Bowler", "p": 0.5, "y": 0},
                {"team": "A", "name": "Frontline", "p": 0.5, "y": 1},
            ],
        },
    }]

    rows = baseline_rows(detail, asof)["top_bowler"]
    prices = {row["y"]: row["p_base"] for row in rows}
    assert sum(row["p_base"] for row in rows) == pytest.approx(1.0)
    assert prices[1] > prices[0]


def test_wicket_count_keeps_stronger_threshold_rate_as_primary_baseline():
    usage = [
        ("2024-01-%02d" % day, 12, int(day == 1))
        for day in range(1, 11)
    ]
    bowler = [
        ("2024-01-%02d" % day, int(day == 1))
        for day in range(1, 11)
    ]
    asof = AsOf(_logs({"Player": usage}, {"Player": bowler}))
    detail = [{
        "match_id": "2024-02-01_A_B_Unknown",
        "obs": {
            "bowler_wkts_1plus": [
                {"team": "A", "name": "Player", "p": 0.5, "y": 0},
            ],
        },
    }]

    row = baseline_rows(detail, asof)["bowler_wkts_1plus"][0]
    assert row["p_base"] == pytest.approx(0.1)
    assert row["p_usage_count"] == pytest.approx(
        poisson_at_least(0.1, 1))
    assert row["p_usage_count"] < row["p_base"]
