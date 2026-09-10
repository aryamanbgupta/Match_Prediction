from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from sim_eval.market_math import (
    DEFAULT_SCENARIOS,
    CostModel,
    InvalidMarketPriceError,
    edge,
    expected_value,
    implied_probs,
    kelly_fraction,
    settle_flat,
    settle_kelly,
)


NONE = CostModel.none()
FIXTURE = Path(__file__).parent / "fixtures" / "market_math_parity.json"


def test_cost_units_defaults_and_scenarios():
    cost = CostModel(100, 250, "stake")
    assert cost.spread == 0.01
    assert cost.fee == 0.025
    assert NONE == CostModel(0, 0, "winnings")
    assert [c.as_dict() for c in DEFAULT_SCENARIOS] == [
        {"spread_bps": 0, "fee_bps": 0, "fee_basis": "winnings"},
        {"spread_bps": 100, "fee_bps": 0, "fee_basis": "winnings"},
        {"spread_bps": 200, "fee_bps": 0, "fee_basis": "winnings"},
        {"spread_bps": 100, "fee_bps": 100, "fee_basis": "winnings"},
        {"spread_bps": 100, "fee_bps": 100, "fee_basis": "stake"},
    ]


@pytest.mark.parametrize("odds", [1.0, 0.99, 0.0, -2.0, math.inf, math.nan])
def test_price_must_imply_q_strictly_between_zero_and_one(odds):
    with pytest.raises(InvalidMarketPriceError):
        expected_value(0.6, odds, NONE)


def test_implied_probs_raw_normalized_edge_and_invalid_price():
    assert implied_probs({"A": 2.0, "B": 4.0}, False) == {"A": 0.5, "B": 0.25}
    assert implied_probs({"A": 2.0, "B": 4.0}, True) == {
        "A": 2.0 / 3.0, "B": 1.0 / 3.0,
    }
    assert edge(0.7, 0.55) == 0.7 - 0.55
    with pytest.raises(InvalidMarketPriceError):
        implied_probs({"A": 0.9, "B": 2.0}, True)


def test_effective_price_cap_and_returns_at_both_fee_bases():
    zero_spread_cap = CostModel.none()
    near_boundary_odds = 1.0 / (1.0 - 5e-10)
    capped_return = settle_flat(
        "A", near_boundary_odds, "A", zero_spread_cap
    )
    assert capped_return == 1.0 / (1.0 - 1e-9) - 1.0
    assert capped_return != near_boundary_odds - 1.0
    capped = CostModel(9999, 0, "winnings")
    assert settle_flat("A", 1.0000001, "A", capped) == pytest.approx(
        1.0 / (1.0 - 1e-9) - 1.0
    )
    exactly_at_cap = CostModel(9999.99998, 0, "winnings")
    assert settle_flat("A", 2.0, "A", exactly_at_cap) == pytest.approx(
        1.0 / (1.0 - 1e-9) - 1.0
    )
    winnings = CostModel(100, 100, "winnings")
    q_eff = 0.5 + 0.01 / 2.0
    assert settle_flat("A", 2.0, "A", winnings) == pytest.approx(
        (1.0 / q_eff - 1.0) * 0.99
    )
    assert settle_flat("A", 2.0, "B", winnings) == -1.0
    stake = CostModel(100, 100, "stake")
    assert settle_flat("A", 2.0, "A", stake) == pytest.approx(
        (1.0 / q_eff - 1.0) - 0.01
    )
    assert settle_flat("A", 2.0, "B", stake) == -1.01


def test_ev_and_kelly_share_effective_returns_and_general_formula():
    cost = CostModel(100, 100, "stake")
    p = 0.6
    b = (1.0 / 0.505 - 1.0) - 0.01
    loss = 1.01
    assert expected_value(p, 2.0, cost) == pytest.approx(p * b - (1 - p) * loss)
    expected_kelly = (p * b - (1 - p) * loss) / (b * loss)
    assert kelly_fraction(p, 2.0, cost) == pytest.approx(expected_kelly)
    assert settle_kelly(expected_kelly, 2.0, "A", "B", cost) == pytest.approx(
        -expected_kelly * loss
    )

    winnings = CostModel(100, 200, "winnings")
    winnings_return = (1.0 / 0.505 - 1.0) * 0.98
    winnings_ev = p * winnings_return - (1.0 - p)
    assert expected_value(p, 2.0, winnings) == pytest.approx(winnings_ev)
    assert kelly_fraction(p, 2.0, winnings) == pytest.approx(
        winnings_ev / winnings_return
    )


def test_kelly_zero_conditions_and_stake_bankroll_bound():
    assert kelly_fraction(0.5, 2.0, NONE) == 0.0
    assert kelly_fraction(0.6, 1.01, CostModel(0, 500, "stake")) == 0.0
    assert kelly_fraction(0.0, 2.0, NONE) == 0.0
    assert kelly_fraction(1.0, 2.0, NONE) == 0.0
    assert kelly_fraction(0.9, 1.5, CostModel(0, 5000, "stake")) == 0.0
    cost = CostModel(0, 5000, "stake")
    fraction = kelly_fraction(0.999999, 100.0, cost)
    assert fraction * 1.5 <= 1.0


@pytest.mark.parametrize(
    "args",
    [(-1, 0, "winnings"), (10000, 0, "winnings"),
     (0, -1, "winnings"), (0, 10000, "winnings"),
     (math.nan, 0, "winnings"), (math.inf, 0, "winnings"),
     (0, math.nan, "winnings"), (0, math.inf, "winnings"),
     (0, 0, "turnover")],
)
def test_cost_constructor_rejects_invalid_values(args):
    with pytest.raises(ValueError):
        CostModel(*args)


def test_frozen_legacy_parity_bit_for_bit():
    records = json.loads(FIXTURE.read_text())["records"]
    assert len(records) >= 50
    for record in records:
        inputs = record["inputs"]
        outputs = record["outputs"]
        bet_team = outputs["bet_team"]
        winner = inputs["winner"]
        if bet_team is None:
            assert outputs["realized_pnl"] in (None, 0.0)
            continue
        odds = inputs["market_odds"][bet_team]
        if not math.isfinite(odds) or odds <= 1.0:
            # Legacy policy rejection happens before the strict primitive.
            assert outputs["realized_pnl"] == 0.0
            continue
        p = inputs["simulated_prob"][bet_team]
        assert settle_flat(bet_team, odds, winner, NONE) == outputs["realized_pnl"]
        assert expected_value(p, odds, NONE) == outputs["expected_value"]
        full = kelly_fraction(p, odds, NONE)
        assert full == outputs["full_kelly_fraction"]
        assert settle_kelly(full, odds, bet_team, winner, NONE) == outputs["full_kelly_pnl"]
        fractional = full * 0.25
        assert fractional == outputs["fractional_kelly_fraction"]
        assert settle_kelly(fractional, odds, bet_team, winner, NONE) == outputs["fractional_kelly_pnl"]
