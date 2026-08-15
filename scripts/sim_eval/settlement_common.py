"""Shared cricsheet settlement conventions (2026-08-14 review, Phase 4).

Three parsers (prop_backtest.compute_actuals, prop_fair_baselines'
corpus builder, b9's usage corpus) each hand-rolled these conventions and
had already drifted: super-over handling, bowler-credited dismissal kinds,
and extras charging diverged between the settlement side and the
fair-baseline side of the SAME comparison. The conventions live here once;
parsers import them.
"""

from __future__ import annotations

from typing import Iterable, List, Mapping

# Dismissals credited to the bowler. "run out", "retired hurt",
# "retired out", "retired not out", "obstructing the field", "handled the
# ball", "hit the ball twice", "timed out" are NOT bowler wickets.
BOWLER_KINDS = {
    "bowled", "caught", "lbw", "stumped", "caught and bowled", "hit wicket",
}

# Not dismissals at all: the batter may resume; neither the partnership nor
# runs-before-first-wicket settlement ends on these.
NOT_OUT_KINDS = {"retired hurt", "retired not out"}


def regulation_innings(data: Mapping) -> List[Mapping]:
    """The (at most two) regulation innings of a T20 match.

    Cricsheet appends super overs as extra innings objects restarting at
    over 0 (flagged `super_over: true`); counting them corrupts per-innings
    settlement and tie detection. Both the flag filter and the [:2] cap are
    applied so either representation is handled.
    """
    return [
        inn for inn in data.get("innings", []) if not inn.get("super_over")
    ][:2]


def is_legal_delivery(extras: Mapping | None) -> bool:
    """Legal delivery = neither a wide nor a no-ball."""
    extras = extras or {}
    return "wides" not in extras and "noballs" not in extras


def counts_as_ball_faced(extras: Mapping | None) -> bool:
    """Balls faced exclude wides only (DK + cricsheet convention)."""
    return "wides" not in (extras or {})


def bowler_conceded(delivery: Mapping) -> int:
    """Runs charged to the bowler for one delivery.

    Batter runs + wides + no-balls are charged; byes and leg-byes are not.
    """
    extras = delivery.get("extras", {}) or {}
    non_bowler = (extras.get("byes", 0) or 0) + (extras.get("legbyes", 0) or 0)
    return int(delivery["runs"]["total"]) - int(non_bowler)


def bowler_credited_wickets(wickets: Iterable[Mapping]) -> int:
    return sum(
        1 for w in wickets
        if str(w.get("kind") or "").lower() in BOWLER_KINDS
    )
