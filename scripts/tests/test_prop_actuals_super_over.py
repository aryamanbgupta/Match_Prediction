"""Regression test for review finding PROP3 (2026-08-14).

Cricsheet appends super overs as extra innings objects that restart at
over 0. `compute_actuals` used to iterate them as regular innings:
per-innings assignments (powerplay, first over, first wicket, max over)
were OVERWRITTEN by super-over values, totals accumulated super-over runs,
and `is_tie` compared totals including the super over — so every genuinely
tied match settled as not-a-tie. Settlement must cover regulation play
only, matching the sim (exactly two innings) and the fair-baseline corpus
(`innings[:2]`).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sim_eval.prop_backtest import compute_actuals  # noqa: E402


def _delivery(batter, bowler, batter_runs, extras=None):
    total = batter_runs + sum((extras or {}).values())
    d = {
        "batter": batter,
        "bowler": bowler,
        "non_striker": "NS",
        "runs": {"batter": batter_runs, "extras": total - batter_runs,
                 "total": total},
    }
    if extras:
        d["extras"] = extras
    return d


def _innings(team, batter, bowler, runs_per_over, super_over=False):
    inn = {
        "team": team,
        "overs": [
            {"over": i, "deliveries": [
                _delivery(batter, bowler, r) for r in over_runs
            ]}
            for i, over_runs in enumerate(runs_per_over)
        ],
    }
    if super_over:
        inn["super_over"] = True
    return inn


def _tied_match_with_super_over():
    # Regulation: both teams score 12 (6+6 across two overs).
    # Super over: A scores 8, B scores 3 — the pre-fix code broke the tie
    # and rewrote A's powerplay/first-over actuals with the super over's.
    return {
        "innings": [
            _innings("A", "A1", "B1", [[1] * 6, [1] * 6]),
            _innings("B", "B2", "A2", [[1] * 6, [1] * 6]),
            _innings("A", "A1", "B1", [[4, 4, 0, 0, 0, 0]],
                     super_over=True),
            _innings("B", "B2", "A2", [[1, 1, 1, 0, 0, 0]],
                     super_over=True),
        ],
    }


def test_super_over_is_excluded_from_settlement():
    actuals = compute_actuals(_tied_match_with_super_over())
    assert actuals["is_tie"] == 1, \
        "a regulation tie must settle as a tie regardless of the super over"
    assert actuals["team_runs"] == {"A": 12, "B": 12}
    # Per-innings props keep the REGULATION values, not the super over's.
    assert actuals["team_first_over_runs"] == {"A": 6, "B": 6}
    assert actuals["team_pp_runs"] == {"A": 12, "B": 12}
    assert actuals["team_max_over_runs"] == {"A": 6, "B": 6}
    # Batter/bowler tallies exclude super-over deliveries too.
    assert actuals["batter_runs"]["A1"] == 12
    assert actuals["batter_fours"]["A1"] == 0


def test_regulation_only_match_is_unchanged():
    match = {"innings": _tied_match_with_super_over()["innings"][:2]}
    actuals = compute_actuals(match)
    assert actuals["is_tie"] == 1
    assert actuals["team_runs"] == {"A": 12, "B": 12}


def test_dismissal_kinds_follow_settlement_conventions():
    """Bowler credit uses BOWLER_KINDS (retired hurt / obstructing the
    field are NOT bowler wickets); runs-before-first-wicket settles on any
    actual dismissal (run out included) but not on a retirement."""
    def _wicket_delivery(batter, bowler, runs, kind):
        d = _delivery(batter, bowler, runs)
        d["wickets"] = [{"kind": kind, "player_out": batter}]
        return d

    match = {"innings": [{
        "team": "A",
        "overs": [{"over": 0, "deliveries": [
            _delivery("A1", "B1", 1),           # 1 run
            _wicket_delivery("A1", "B1", 1, "retired hurt"),   # 2 runs
            _wicket_delivery("A2", "B1", 0, "run out"),        # first wicket
            _wicket_delivery("A3", "B1", 0, "bowled"),
            _delivery("A4", "B1", 4),
        ]}],
    }]}
    actuals = compute_actuals(match)
    assert actuals["bowler_wkts"]["B1"] == 1, \
        "only the bowled dismissal is bowler-credited"
    assert actuals["team_first_wicket_runs"]["A"] == 2, \
        "first wicket = the run out at 2 team runs; the retirement at 2 " \
        "runs is not out and must not settle the prop"
