"""Generate the frozen pre-refactor betting-math parity corpus.

This script intentionally calls the private arithmetic methods on the
unrefactored MatchEvaluator.  It is retained so the provenance of the fixture
is reproducible; do not regenerate it after changing those methods.
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS_DIR))

from sim_eval.loaders import BettingOddsLoader
from sim_eval.match_evaluator import BET_EDGE_THRESHOLD, MatchLevelEvaluator


SEED = 20260910
OUTPUT = Path(__file__).parent / "fixtures" / "market_math_parity.json"


def _record(evaluator: MatchLevelEvaluator, index: int, case: dict) -> dict:
    teams = ("Alpha", "Bravo")
    p = float(case["p"])
    simulated_prob = {teams[0]: p, teams[1]: 1.0 - p}
    market_odds = {
        teams[0]: case["odds_a"],
        teams[1]: case["odds_b"],
    }
    market_prob = BettingOddsLoader.get_implied_probabilities(market_odds)
    edge = evaluator._calculate_edge(simulated_prob, market_prob)
    winner = case["winner"]
    realized_pnl = evaluator._calculate_realized_pnl(
        edge, market_odds, winner
    )

    best_team = None
    best_edge = 0.0
    for team, team_edge in edge.items():
        if team_edge > best_edge:
            best_edge = team_edge
            best_team = team

    if (
        best_team
        and best_edge > BET_EDGE_THRESHOLD
        and best_team in market_odds
    ):
        odds = market_odds[best_team]
        win_prob = simulated_prob[best_team]
        expected_value = evaluator._calculate_expected_value(win_prob, odds)
        full_kelly_fraction = evaluator._calculate_kelly_fraction(
            win_prob, odds
        )
        full_kelly_pnl = evaluator._calculate_kelly_pnl(
            full_kelly_fraction, odds, best_team, winner
        )
        fractional_kelly_fraction = full_kelly_fraction * 0.25
        fractional_kelly_pnl = evaluator._calculate_kelly_pnl(
            fractional_kelly_fraction, odds, best_team, winner
        )
    else:
        expected_value = 0.0
        full_kelly_fraction = 0.0
        full_kelly_pnl = None
        fractional_kelly_fraction = 0.0
        fractional_kelly_pnl = None

    return {
        "id": f"parity-{index:03d}",
        "inputs": {
            "market_odds": market_odds,
            "simulated_prob": simulated_prob,
            "winner": winner,
        },
        "outputs": {
            "bet_team": best_team,
            "edge": edge,
            "expected_value": expected_value,
            "fractional_kelly_fraction": fractional_kelly_fraction,
            "fractional_kelly_pnl": fractional_kelly_pnl,
            "full_kelly_fraction": full_kelly_fraction,
            "full_kelly_pnl": full_kelly_pnl,
            "market_prob": market_prob,
            "realized_pnl": realized_pnl,
        },
        "scenario": case["scenario"],
    }


def build_cases() -> list[dict]:
    rng = random.Random(SEED)
    scenarios = [
        ("win", 0.72, 2.05, 1.85, "Alpha"),
        ("loss", 0.72, 2.05, 1.85, "Bravo"),
        ("no_bet_edge", 0.5, 2.0, 2.0, "Alpha"),
        ("odds_below_one", 0.9, 0.95, 4.0, "Alpha"),
        ("non_finite_odds", 0.9, math.inf, 2.0, "Alpha"),
        ("missing_winner", 0.72, 2.05, 1.85, None),
        ("p_zero", 0.0, 1.8, 2.2, "Bravo"),
        ("p_one", 1.0, 2.2, 1.8, "Alpha"),
        ("long_shot", 0.18, 12.5, 1.08, "Alpha"),
        ("near_one", 0.99, 1.000001, 30.0, "Bravo"),
    ]
    cases = []
    for repetition in range(6):
        for scenario, p, odds_a, odds_b, winner in scenarios:
            jitter = 0.0 if scenario in {
                "no_bet_edge", "odds_below_one", "non_finite_odds",
                "p_zero", "p_one", "near_one",
            } else rng.uniform(-0.015, 0.015)
            cases.append({
                "scenario": scenario,
                "p": min(max(p + jitter, 0.0), 1.0),
                "odds_a": odds_a,
                "odds_b": odds_b,
                "winner": winner,
                "repetition": repetition,
            })
    return cases


def main() -> None:
    evaluator = MatchLevelEvaluator.__new__(MatchLevelEvaluator)
    payload = {
        "generator": "scripts/tests/gen_market_math_parity.py",
        "records": [
            _record(evaluator, index, case)
            for index, case in enumerate(build_cases())
        ],
        "seed": SEED,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n"
    )


if __name__ == "__main__":
    main()
