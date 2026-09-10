"""Pure numeric primitives for two-outcome betting markets.

Placement, suppression, liquidity, and sizing policies deliberately live in
their callers.  This module owns only price conversion and return arithmetic.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping


class InvalidMarketPriceError(ValueError):
    """Raised when decimal odds do not encode a price strictly in (0, 1)."""


@dataclass(frozen=True)
class CostModel:
    spread_bps: float
    fee_bps: float
    fee_basis: str

    def __post_init__(self) -> None:
        spread = self.spread_bps / 10_000.0
        fee = self.fee_bps / 10_000.0
        if not math.isfinite(spread) or not 0.0 <= spread < 1.0:
            raise ValueError("spread_bps must encode a spread in [0, 1)")
        if not math.isfinite(fee) or not 0.0 <= fee < 1.0:
            raise ValueError("fee_bps must encode a fee in [0, 1)")
        if self.fee_basis not in {"winnings", "stake"}:
            raise ValueError("fee_basis must be 'winnings' or 'stake'")

    @classmethod
    def none(cls) -> "CostModel":
        return cls(0.0, 0.0, "winnings")

    @property
    def spread(self) -> float:
        return self.spread_bps / 10_000.0

    @property
    def fee(self) -> float:
        return self.fee_bps / 10_000.0

    def as_dict(self) -> dict:
        return {
            "spread_bps": self.spread_bps,
            "fee_bps": self.fee_bps,
            "fee_basis": self.fee_basis,
        }


DEFAULT_SCENARIOS = [
    CostModel(0, 0, "winnings"),
    CostModel(100, 0, "winnings"),
    CostModel(200, 0, "winnings"),
    CostModel(100, 100, "winnings"),
    CostModel(100, 100, "stake"),
]


def _returns(odds: float, cost: CostModel) -> tuple[float, float]:
    try:
        decimal_odds = float(odds)
        q = 1.0 / decimal_odds
    except (TypeError, ValueError, ZeroDivisionError, OverflowError) as exc:
        raise InvalidMarketPriceError(f"invalid decimal odds: {odds!r}") from exc
    if not math.isfinite(q) or not 0.0 < q < 1.0:
        raise InvalidMarketPriceError(
            f"decimal odds must imply a price strictly in (0, 1): {odds!r}"
        )
    q_eff = min(q + cost.spread / 2.0, 1.0 - 1e-9)
    # Avoid a reciprocal round-trip at zero spread: this is what makes the
    # no-cost model reproduce the legacy ``odds - 1`` arithmetic bit-for-bit.
    o_eff = decimal_odds if (cost.spread == 0.0 and q_eff == q) else 1.0 / q_eff
    if cost.fee_basis == "winnings":
        return (o_eff - 1.0) * (1.0 - cost.fee), -1.0
    return (o_eff - 1.0) - cost.fee, -(1.0 + cost.fee)


def implied_probs(
    odds: Mapping[str, float], remove_margin: bool = True
) -> dict[str, float]:
    """Convert a complete odds mapping to raw or margin-normalized prices."""
    if not odds:
        return {}
    raw = {}
    for team, decimal_odds in odds.items():
        try:
            value = float(decimal_odds)
            q = 1.0 / value
        except (TypeError, ValueError, ZeroDivisionError, OverflowError) as exc:
            raise InvalidMarketPriceError(
                f"invalid decimal odds for {team!r}: {decimal_odds!r}"
            ) from exc
        if not math.isfinite(q) or not 0.0 < q < 1.0:
            raise InvalidMarketPriceError(
                f"decimal odds for {team!r} must imply q in (0, 1)"
            )
        raw[team] = q
    if not remove_margin:
        return raw
    total = sum(raw.values())
    return {team: probability / total for team, probability in raw.items()}


def edge(model_p: float, market_p: float) -> float:
    return model_p - market_p


def settle_flat(
    bet_team: str | None,
    odds: float,
    winner: str | None,
    cost: CostModel,
) -> float | None:
    if winner is None:
        return None
    if bet_team is None:
        return 0.0
    win_return, loss_return = _returns(odds, cost)
    return win_return if bet_team == winner else loss_return


def kelly_fraction(p: float, odds: float, cost: CostModel) -> float:
    if not math.isfinite(p) or not 0.0 < p < 1.0:
        return 0.0
    win_return, loss_return = _returns(odds, cost)
    loss = -loss_return
    numerator = p * win_return - (1.0 - p) * loss
    if win_return <= 0.0 or numerator <= 0.0:
        return 0.0
    fraction = numerator / (win_return * loss)
    return min(fraction, 1.0 / loss)


def settle_kelly(
    fraction: float,
    odds: float,
    bet_team: str,
    winner: str | None,
    cost: CostModel,
) -> float | None:
    if fraction <= 0.0 or winner is None:
        return None
    win_return, loss_return = _returns(odds, cost)
    return fraction * (win_return if bet_team == winner else loss_return)


def expected_value(p: float, odds: float, cost: CostModel) -> float:
    win_return, loss_return = _returns(odds, cost)
    return p * win_return + (1.0 - p) * loss_return
