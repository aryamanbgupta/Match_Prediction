"""
Characterization and I3 regression tests for the eval math.

Pins the evaluation behavior that research-loop verdicts rest on. Covered:

  - match_evaluator._bootstrap_ci        (reproducibility, strata, degenerate)
  - reslice_eval_json._bootstrap_ci      (parity with the evaluator's copy)
  - match_evaluator._calculate_kelly_fraction / _calculate_kelly_pnl
  - match_evaluator._calculate_realized_pnl
  - match_evaluator._aggregate_results   (flat-ROI/PnL bookkeeping, CI scaling)
  - reslice_eval_json.reslice            (min-volume boundaries, summary math)
  - blend_eval_json                      (w=0/w=1 identities, team alignment,
                                          pnl/kelly parity with the evaluator)

I3 replaces the P&L sentinel with an explicit edge/odds bet decision and
uses tournament/tour-season block bootstrap intervals for headline metrics.

Run standalone (repo convention) or under pytest:
    uv run python scripts/tests/test_eval_math.py
    uv run python -m pytest scripts/tests/test_eval_math.py -q
"""

import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

from sim_eval.match_evaluator import (  # noqa: E402
    BET_EDGE_THRESHOLD,
    MatchEvaluationResult,
    MatchLevelEvaluator,
)
from sim_eval import blend_eval_json as bl  # noqa: E402
from sim_eval import reslice_eval_json as rs  # noqa: E402
from sim_eval import sizing_rules as sizing  # noqa: E402
from auto import a7_conditional_threshold as a7  # noqa: E402
from sim_eval.eval_statistics import (  # noqa: E402
    BOOTSTRAP_CONTRACT_VERSION,
    bootstrap_mean_ci,
    cluster_id_for_record,
    competition_cluster_from_info,
    flat_bet_team,
    load_competition_clusters,
)


# --------------------------------------------------------------------------
# Fixtures / helpers
# --------------------------------------------------------------------------

def _evaluator(resamples: int = 1000) -> MatchLevelEvaluator:
    # model / engine are never touched by the math under test.
    return MatchLevelEvaluator(
        model=None, simulation_engine=None, n_simulations=0,
        parallel=False, bootstrap_resamples=resamples)


def _mk_result(match_id="m", team1="A", team2="B", p1=0.5,
               market_prob=None, market_odds=None, actual_winner="A",
               log_loss=0.5, brier=0.25, edge=None, realized_pnl=0.0,
               **kw) -> MatchEvaluationResult:
    sim = {team1: p1, team2: 1.0 - p1}
    market_prob = market_prob if market_prob is not None else {team1: 0.5, team2: 0.5}
    market_odds = market_odds if market_odds is not None else {team1: 2.0, team2: 2.0}
    if edge is None:
        edge = {t: sim[t] - market_prob.get(t, 0.5) for t in sim}
    return MatchEvaluationResult(
        match_id=match_id, team1=team1, team2=team2,
        simulated_win_prob=sim, simulated_scores={},
        market_win_prob=market_prob, market_odds=market_odds,
        actual_winner=actual_winner, log_loss=log_loss, brier_score=brier,
        edge=edge, realized_pnl=realized_pnl, **kw)


_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
_STRATA = ["a"] * 5 + ["b"] * 5


def _isnan_pair(pair):
    return math.isnan(pair[0]) and math.isnan(pair[1])


# --------------------------------------------------------------------------
# A. _bootstrap_ci — evaluator + reslice copies
# --------------------------------------------------------------------------

def test_bootstrap_empty_and_zero_resamples_return_nan():
    ev = _evaluator()
    assert _isnan_pair(ev._bootstrap_ci([])), "empty values must give (nan, nan)"
    assert _isnan_pair(ev._bootstrap_ci([1.0, 2.0], n_resamples=0)), \
        "n_resamples=0 must give (nan, nan)"
    assert _isnan_pair(rs._bootstrap_ci([])), "reslice: empty must give (nan, nan)"
    # Both wrappers delegate to eval_statistics.bootstrap_mean_ci, which
    # guards n_resamples <= 0 — assert the reslice side stays guarded too.
    assert _isnan_pair(rs._bootstrap_ci([1.0, 2.0], n=0)), \
        "reslice: n=0 must give (nan, nan), not np.quantile of empty"


def test_bootstrap_reproducible_at_seed_42():
    ev = _evaluator()
    a = ev._bootstrap_ci(_VALUES)
    b = ev._bootstrap_ci(_VALUES)
    assert a == b, f"same input, default seed 42 must be bit-identical: {a} vs {b}"
    c = ev._bootstrap_ci(_VALUES, seed=7)
    assert a != c, "a different seed must produce a different resample stream"


def test_bootstrap_constant_input_degenerate_ci():
    ev = _evaluator()
    lo, hi = ev._bootstrap_ci([0.5] * 20)
    assert lo == 0.5 and hi == 0.5, f"constant input must pin CI to the value: ({lo}, {hi})"


def test_bootstrap_sanity_bounds():
    ev = _evaluator()
    lo, hi = ev._bootstrap_ci(_VALUES)
    mean = sum(_VALUES) / len(_VALUES)
    assert lo <= hi, f"lo must be <= hi: ({lo}, {hi})"
    assert min(_VALUES) <= lo and hi <= max(_VALUES), \
        "bootstrap means cannot leave the data range"
    assert lo <= mean <= hi, \
        f"95% CI should cover the sample mean here: {lo} <= {mean} <= {hi}"


def test_bootstrap_evaluator_reslice_parity():
    """The loop reads CIs from BOTH implementations interchangeably
    (run_sim_eval summaries vs reslice slices) — they must agree exactly
    at the shared defaults (seed 42, 1000 resamples, ci 0.95)."""
    ev = _evaluator(resamples=1000)
    assert ev._bootstrap_ci(_VALUES) == rs._bootstrap_ci(_VALUES, n=1000), \
        "unstratified: evaluator and reslice copies must be bit-identical"
    a = ev._bootstrap_ci(_VALUES, strata=_STRATA)
    b = rs._bootstrap_ci(_VALUES, n=1000, strata=_STRATA)
    assert a == b, f"stratified: evaluator vs reslice diverge: {a} vs {b}"


def test_bootstrap_single_stratum_equals_unstratified():
    """One stratum consumes the RNG identically to the unstratified path,
    so the outputs must match exactly — a useful structural invariant."""
    ev = _evaluator()
    assert ev._bootstrap_ci(_VALUES) == ev._bootstrap_ci(_VALUES, strata=["x"] * 10)
    assert rs._bootstrap_ci(_VALUES) == rs._bootstrap_ci(_VALUES, strata=["x"] * 10)


def test_bootstrap_strata_length_mismatch_raises():
    ev = _evaluator()
    for fn in (lambda: ev._bootstrap_ci(_VALUES, strata=["a"] * 3),
               lambda: rs._bootstrap_ci(_VALUES, strata=["a"] * 3)):
        try:
            fn()
        except ValueError:
            pass
        else:
            raise AssertionError("strata length mismatch must raise ValueError")


def test_bootstrap_pinned_values():
    """Characterization pin, computed on the repo venv (numpy 1.24.3,
    PCG64 via np.random.default_rng(42)). If numpy's Generator stream
    ever changes across an upgrade, re-pin these — the OTHER tests
    (reproducibility/parity) are the behavior contract; this one detects
    silent environment drift between research-loop sessions."""
    ev = _evaluator()
    lo, hi = ev._bootstrap_ci(_VALUES)
    assert abs(lo - 0.37) < 1e-12 and abs(hi - 0.73) < 1e-12, \
        f"unstratified pin moved: ({lo}, {hi}) != (0.37, 0.73)"
    slo, shi = ev._bootstrap_ci(_VALUES, strata=_STRATA)
    assert abs(slo - 0.4600000000000001) < 1e-12 and abs(shi - 0.6302499999999999) < 1e-12, \
        f"stratified pin moved: ({slo}, {shi})"


def test_cluster_bootstrap_resamples_whole_competitions():
    values = [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]
    clusters = ["tour-a"] * 3 + ["league-b"] * 3
    iid_lo, iid_hi = bootstrap_mean_ci(values, n_resamples=4000, seed=42)
    block_lo, block_hi = bootstrap_mean_ci(
        values,
        n_resamples=4000,
        seed=42,
        clusters=clusters,
    )
    assert (block_hi - block_lo) > (iid_hi - iid_lo)
    assert block_lo == -1.0 and block_hi == 1.0


def test_competition_cluster_uses_event_time_blocks(tmp_path):
    info = {
        "dates": ["2026-01-03"],
        "teams": ["A", "B"],
        "venue": "Ground",
        "event": {"name": "Example League"},
    }
    payload = {"info": info}
    (tmp_path / "1.json").write_text(json.dumps(payload))
    lookup = load_competition_clusters(tmp_path)
    expected = "event:Example League|block_start:2026-01-03"
    assert competition_cluster_from_info(info).startswith(
        "event:Example League|season:"
    )
    assert lookup["2026-01-03_A_B_Ground"] == expected
    assert BOOTSTRAP_CONTRACT_VERSION == "tournament_time_block_v1"


def test_event_block_crosses_calendar_boundary_but_splits_after_gap(tmp_path):
    rows = [
        ("1", "2025-12-20", ["A", "B"]),
        ("2", "2026-01-10", ["C", "D"]),
        ("3", "2026-08-01", ["A", "C"]),
    ]
    for stem, date, teams in rows:
        (tmp_path / f"{stem}.json").write_text(json.dumps({
            "info": {
                "dates": [date],
                "teams": teams,
                "venue": "Ground",
                "event": {"name": "Cross-Year League"},
            }
        }))
    lookup = load_competition_clusters(tmp_path)
    first = lookup["2025-12-20_A_B_Ground"]
    assert lookup["2026-01-10_C_D_Ground"] == first
    assert lookup["2026-08-01_A_C_Ground"] != first


# --------------------------------------------------------------------------
# B. Kelly fraction / Kelly PnL
# --------------------------------------------------------------------------

def test_kelly_zero_on_invalid_or_no_edge():
    ev = _evaluator()
    assert ev._calculate_kelly_fraction(0.5, 1.0) == 0.0, "odds <= 1.0 -> 0"
    assert ev._calculate_kelly_fraction(0.5, 0.5) == 0.0
    assert ev._calculate_kelly_fraction(0.0, 2.5) == 0.0, "p <= 0 -> 0"
    assert ev._calculate_kelly_fraction(1.0, 2.5) == 0.0, "p >= 1 -> 0"
    assert ev._calculate_kelly_fraction(0.5, 2.0) == 0.0, \
        "fair line (kelly exactly 0) -> no bet"
    assert ev._calculate_kelly_fraction(0.4, 1.5) == 0.0, "negative kelly -> 0"


def test_kelly_formula_grid():
    ev = _evaluator()
    for p, odds, expect in [
        (0.60, 2.0, 0.2),          # (1*0.6 - 0.4) / 1
        (0.55, 2.2, 0.175),        # (1.2*0.55 - 0.45) / 1.2
        (0.30, 5.0, 0.125),        # (4*0.3 - 0.7) / 4
    ]:
        got = ev._calculate_kelly_fraction(p, odds)
        assert abs(got - expect) < 1e-12, f"kelly({p},{odds}) = {got}, want {expect}"


def test_kelly_is_uncapped():
    """Deliberate current behavior ("no cap as requested"): full Kelly can
    approach the whole bankroll. Downstream risk control is only the 25%
    fractional-Kelly variant. If a cap is ever added, this pin flips."""
    ev = _evaluator()
    got = ev._calculate_kelly_fraction(0.99, 100.0)
    assert abs(got - 98.0 / 99.0) < 1e-12, f"kelly(0.99, 100) = {got}, want 98/99"
    assert ev._calculate_kelly_fraction(0.9, 10.0) - 8.0 / 9.0 < 1e-12
    assert got > 0.98, "no cap: fraction may exceed any prudent stake bound"


def test_kelly_pnl_arithmetic():
    ev = _evaluator()
    assert abs(ev._calculate_kelly_pnl(0.2, 2.5, "A", "A") - 0.3) < 1e-12, \
        "win: stake * (odds - 1)"
    assert abs(ev._calculate_kelly_pnl(0.2, 2.5, "A", "B") - (-0.2)) < 1e-12, \
        "loss: -stake"
    assert ev._calculate_kelly_pnl(0.0, 2.5, "A", "A") is None, "no stake -> None"
    assert ev._calculate_kelly_pnl(0.2, 2.5, "A", None) is None, "no winner -> None"


# --------------------------------------------------------------------------
# C. _calculate_realized_pnl (flat 1-unit staking)
# --------------------------------------------------------------------------

def test_realized_pnl_none_paths():
    ev = _evaluator()
    odds = {"A": 2.0, "B": 2.0}
    assert ev._calculate_realized_pnl({"A": 0.1}, odds, None) is None, "no winner"
    assert ev._calculate_realized_pnl({}, odds, "A") is None, "empty edge"
    assert ev._calculate_realized_pnl({"A": 0.1}, {}, "A") is None, "empty odds"


def test_realized_pnl_edge_threshold_is_strict():
    ev = _evaluator()
    odds = {"A": 2.5, "B": 1.6}
    assert BET_EDGE_THRESHOLD == 0.0, "loop semantics assume threshold 0"
    assert ev._calculate_realized_pnl({"A": 0.0, "B": -0.1}, odds, "A") == 0.0, \
        "edge exactly 0 -> no bet (strict >)"
    assert ev._calculate_realized_pnl({"A": -0.2, "B": -0.1}, odds, "A") == 0.0, \
        "all-negative edge -> no bet"
    got = ev._calculate_realized_pnl({"A": 1e-9, "B": -0.1}, odds, "A")
    assert abs(got - 1.5) < 1e-12, f"any positive edge places the bet: {got}"


def test_realized_pnl_win_loss_and_team_choice():
    ev = _evaluator()
    odds = {"A": 2.5, "B": 3.0}
    assert abs(ev._calculate_realized_pnl({"A": 0.08, "B": -0.02}, odds, "A") - 1.5) < 1e-12
    assert ev._calculate_realized_pnl({"A": 0.08, "B": -0.02}, odds, "B") == -1.0
    got = ev._calculate_realized_pnl({"A": 0.05, "B": 0.12}, odds, "B")
    assert abs(got - 2.0) < 1e-12, "must bet the HIGHEST positive edge (B), not any"


def test_realized_pnl_zero_return_win_equals_no_bet_sentinel():
    """The scalar P&L function has an unavoidable zero-value collision.

    Aggregation must therefore use the explicit bet decision rather than
    this scalar as a placement sentinel.
    """
    ev = _evaluator()
    no_bet = ev._calculate_realized_pnl({"A": -0.1, "B": -0.1}, {"A": 2.0}, "A")
    zero_win = ev._calculate_realized_pnl({"A": 0.1, "B": -0.1}, {"A": 1.0}, "A")
    assert no_bet == 0.0 and zero_win == 0.0 and no_bet == zero_win, \
        "the two outcomes are indistinguishable downstream"


# --------------------------------------------------------------------------
# D. _aggregate_results — PnL bookkeeping + CI scaling
# --------------------------------------------------------------------------

def test_aggregate_empty_input():
    res = _evaluator()._aggregate_results([], 0.0)
    assert res.n_matches == 0
    assert math.isnan(res.avg_log_loss)
    assert res.bets_placed == 0 and res.total_pnl == 0.0 and res.roi == 0.0


def test_aggregate_flat_roi_arithmetic_and_ci_scaling():
    ev = _evaluator()
    results = [
        # placed bet, won at 2.5: pnl +1.5
        _mk_result("m1", p1=0.5, market_prob={"A": 0.4, "B": 0.6},
                   market_odds={"A": 2.5, "B": 1.6}, actual_winner="A",
                   log_loss=0.4, brier=0.16, realized_pnl=1.5),
        # placed bet on B, lost: pnl -1.0
        _mk_result("m2", p1=0.4, market_prob={"A": 0.55, "B": 0.45},
                   market_odds={"A": 1.8, "B": 2.2}, actual_winner="A",
                   log_loss=0.9, brier=0.25, realized_pnl=-1.0),
        # no bet (no positive edge): pnl 0.0
        _mk_result("m3", p1=0.5, market_prob={"A": 0.55, "B": 0.5},
                   market_odds={"A": 1.8, "B": 2.0}, actual_winner="B",
                   log_loss=0.6, brier=0.2, realized_pnl=0.0),
        # no result: pnl None, nan log loss
        _mk_result("m4", actual_winner=None, log_loss=float("nan"),
                   brier=float("nan"), realized_pnl=None),
    ]
    res = ev._aggregate_results(results, 0.0)
    assert res.n_matches == 4
    assert res.bets_placed == 2, f"pnl in {{+1.5, -1.0}} = 2 bets, got {res.bets_placed}"
    assert abs(res.total_pnl - 0.5) < 1e-12
    assert abs(res.roi - 25.0) < 1e-9, f"ROI = 0.5/2*100 = 25.0, got {res.roi}"
    assert abs(res.win_rate - 0.5) < 1e-12
    assert abs(res.avg_log_loss - (0.4 + 0.9 + 0.6) / 3) < 1e-12, \
        "nan log losses are excluded from the mean"
    # ROI CI is the per-bet PnL-mean CI x100, over ONLY the placed bets.
    bet_clusters = [
        cluster_id_for_record(results[0]),
        cluster_id_for_record(results[1]),
    ]
    lo, hi = ev._bootstrap_ci(
        [1.5, -1.0],
        clusters=bet_clusters,
    )
    assert abs(res.flat_roi_ci_low - lo * 100) < 1e-9
    assert abs(res.flat_roi_ci_high - hi * 100) < 1e-9
    # LL CI over the three valid log losses.
    llo, lhi = ev._bootstrap_ci(
        [0.4, 0.9, 0.6],
        clusters=[
            cluster_id_for_record(result)
            for result in results[:3]
        ],
    )
    assert res.avg_log_loss_ci_low == llo and res.avg_log_loss_ci_high == lhi


def test_aggregate_counts_zero_pnl_win_as_bet():
    """A bet genuinely placed (positive edge) and WON at decimal odds 1.0
    has realized_pnl exactly 0.0. Correct behavior: it IS a placed, winning
    bet -> bets_placed == 2, win_rate == 1.0."""
    ev = _evaluator()
    results = [
        _mk_result("m1", market_prob={"A": 0.4, "B": 0.6},
                   market_odds={"A": 2.5, "B": 1.6}, actual_winner="A",
                   edge={"A": 0.1, "B": -0.1}, realized_pnl=1.5),
        # placed win at odds 1.0 -> pnl 0.0 (see sentinel test above)
        _mk_result("m2", market_prob={"A": 0.9, "B": 0.1},
                   market_odds={"A": 1.0, "B": 9.0}, actual_winner="A",
                   edge={"A": 0.05, "B": -0.05}, realized_pnl=0.0),
    ]
    res = ev._aggregate_results(results, 0.0)
    assert res.bets_placed == 2, \
        f"zero-return win must count as a placed bet; got {res.bets_placed}"
    assert res.win_rate == 1.0
    assert flat_bet_team(results[1]) == "A"


# --------------------------------------------------------------------------
# D2. 2026-08-14 review fixes: fail-closed joins + corrected diagnostics
# --------------------------------------------------------------------------

def test_unknown_winner_fails_closed_not_phantom_half():
    """A winner spelled differently from the simulated teams used to score
    a silent 0.5 in LL, None in blend, and a LOST STAKE in P&L — three
    answers to one join defect. All three must now raise."""
    ev = _evaluator()
    sim = {"Royal Challengers Bengaluru": 0.7, "Mumbai Indians": 0.3}
    with np.testing.assert_raises(RuntimeError):
        ev._calculate_log_loss(sim, "Royal Challengers Bangalore",
                               "Royal Challengers Bengaluru",
                               "Mumbai Indians")
    with np.testing.assert_raises(RuntimeError):
        ev._calculate_realized_pnl(
            {"Royal Challengers Bengaluru": 0.2, "Mumbai Indians": -0.2},
            {"Royal Challengers Bengaluru": 2.0, "Mumbai Indians": 2.0},
            "Royal Challengers Bangalore")
    # Null winner (no result) keeps the NaN / None paths.
    assert math.isnan(ev._calculate_log_loss(sim, None, "x", "y"))
    assert ev._calculate_realized_pnl({}, {}, None) is None


def test_sharpe_is_per_bet_not_a_t_statistic():
    """Same per-bet mean/std at different n must give the same Sharpe —
    the old * sqrt(n) factor grew with sample size."""
    ev = _evaluator()
    short = [1.0, -1.0] * 5
    long = [1.0, -1.0] * 50
    assert abs(ev._calculate_sharpe_ratio(short)
               - ev._calculate_sharpe_ratio(long)) < 1e-9


def test_kelly_roi_is_return_on_amount_staked():
    """P&L divided by summed stake fractions, not by bet count: a 0.5-stake
    win at evens (+0.5) and a 0.1-stake loss (-0.1) is +0.4 on 0.6 staked
    = +66.67%, not +20%."""
    ev = _evaluator()
    results = [
        _mk_result("k1", realized_pnl=1.0,
                   full_kelly_fraction=0.5, full_kelly_pnl=0.5,
                   fractional_kelly_pnl=0.125),
        _mk_result("k2", actual_winner="B", realized_pnl=-1.0,
                   full_kelly_fraction=0.1, full_kelly_pnl=-0.1,
                   fractional_kelly_pnl=-0.025),
    ]
    agg = ev._aggregate_results(results, 0.0)
    assert abs(agg.full_kelly_roi - (0.4 / 0.6 * 100)) < 1e-9
    assert abs(agg.fractional_kelly_roi - (0.1 / 0.15 * 100)) < 1e-9


# --------------------------------------------------------------------------
# E. reslice path — min-volume boundaries + summary math
# --------------------------------------------------------------------------

def _write_reslice_fixture(tmp: Path):
    eval_json = tmp / "eval.json"
    odds_json = tmp / "odds.json"
    matches = [
        {"match_id": "m_a", "log_loss": 0.4, "brier_score": 0.16,
         "realized_pnl": 1.5, "bet_placed": True, "bet_team": "A",
         "actual_winner": "A"},
        {"match_id": "m_b", "log_loss": 0.9, "brier_score": 0.25,
         "realized_pnl": -1.0, "bet_placed": True, "bet_team": "A",
         "actual_winner": "B"},
        {"match_id": "m_c", "log_loss": 0.6, "brier_score": 0.20,
         "realized_pnl": 0.0, "bet_placed": True, "bet_team": "A",
         "actual_winner": "A"},
        {"match_id": "m_d", "log_loss": 0.5, "brier_score": 0.20,
         "realized_pnl": None, "bet_placed": False},
        {"match_id": "m_e", "log_loss": 0.7, "brier_score": 0.21,
         "realized_pnl": 0.5, "bet_placed": True, "bet_team": "A",
         "actual_winner": "A"},
    ]
    eval_json.write_text(json.dumps({"matches": matches}))
    odds_json.write_text(json.dumps({"matches": [
        {"match_id": "m_a", "polymarket_volume_usd": 250_000},
        {"match_id": "m_b", "polymarket_volume_usd": 50_000},   # exact boundary
        {"match_id": "m_c", "polymarket_volume_usd": 49_999},
        {"match_id": "m_d"},                                    # no volume field
        # m_e deliberately absent from the odds file entirely
    ]}))
    return str(eval_json), str(odds_json)


def test_reslice_no_filter_keeps_matches_missing_from_odds():
    with tempfile.TemporaryDirectory() as tmp:
        ep, op = _write_reslice_fixture(Path(tmp))
        out = rs.reslice(ep, op, min_volume=None)
        s = out["summary"]
        assert s["n_matches_evaluated"] == 5, \
            "min_volume=None applies NO volume predicate — even matches " \
            "absent from the odds file stay in the slice"
        assert abs(s["avg_log_loss"] - (0.4 + 0.9 + 0.6 + 0.5 + 0.7) / 5) < 1e-12


def test_reslice_min_volume_boundary_is_inclusive():
    with tempfile.TemporaryDirectory() as tmp:
        ep, op = _write_reslice_fixture(Path(tmp))
        out = rs.reslice(ep, op, min_volume=50_000)
        kept = {m["match_id"] for m in out["matches"]}
        assert kept == {"m_a", "m_b"}, \
            f"vol >= threshold kept (inclusive); missing/absent vol dropped: {kept}"
        out100 = rs.reslice(ep, op, min_volume=100_000)
        assert {m["match_id"] for m in out100["matches"]} == {"m_a"}


def test_reslice_summary_math_and_ci_consistency():
    with tempfile.TemporaryDirectory() as tmp:
        ep, op = _write_reslice_fixture(Path(tmp))
        out = rs.reslice(ep, op, min_volume=None)
        s = out["summary"]
        # Explicit placement retains the zero-return win.
        assert s["flat_betting_bets_placed"] == 4
        assert abs(s["flat_betting_total_pnl"] - 1.0) < 1e-12
        assert abs(s["flat_betting_roi_pct"] - 25.0) < 1e-9
        assert abs(s["flat_betting_win_rate"] - 3.0 / 4) < 1e-12
        # Self-consistency: summary CIs == module's own bootstrap on the
        # same inputs (order preserved from the eval JSON).
        llo, lhi = rs._bootstrap_ci(
            [0.4, 0.9, 0.6, 0.5, 0.7],
            n=10_000,
        )
        assert s["avg_log_loss_ci_low"] == llo and s["avg_log_loss_ci_high"] == lhi
        # Synthetic IDs have no source metadata, so each is a singleton block
        # and the block bootstrap reduces exactly to i.i.d. resampling.
        plo, phi = rs._bootstrap_ci(
            [1.5, -1.0, 0.0, 0.5],
            n=10_000,
        )
        assert abs(s["flat_betting_roi_ci_low"] - plo * 100) < 1e-9
        assert abs(s["flat_betting_roi_ci_high"] - phi * 100) < 1e-9


def test_reslice_counts_zero_pnl_win_as_bet():
    """m_c's explicit zero-return win remains in the denominator."""
    with tempfile.TemporaryDirectory() as tmp:
        ep, op = _write_reslice_fixture(Path(tmp))
        out = rs.reslice(ep, op, min_volume=None)
        assert out["summary"]["flat_betting_bets_placed"] == 4, \
            "zero-return placed bet must stay in the ROI denominator"
        rows = {row["match_id"]: row for row in out["matches"]}
        assert rows["m_c"]["bet_placed"] is True
        assert rows["m_c"]["bet_team"] == "A"
        assert rows["m_c"]["competition_cluster_id"]


# --------------------------------------------------------------------------
# E2. Invariant #7 — pair-block fallback must demote the bootstrap contract
# --------------------------------------------------------------------------

def _write_clustered_fixture(tmp: Path, n_clusters: int):
    """Eval JSON whose rows carry their own competition_cluster_id (one row
    per cluster), plus an odds file granting each row >= $50k volume."""
    matches = []
    odds = []
    for i in range(n_clusters):
        mid = f"cl_{i}"
        matches.append({
            "match_id": mid, "log_loss": 0.6, "brier_score": 0.2,
            "realized_pnl": 0.5 if i % 2 == 0 else -1.0,
            "bet_placed": True, "bet_team": "A", "actual_winner": "A",
            "competition_cluster_id": f"event:League{i}|block_start:2026-01-01",
        })
        odds.append({"match_id": mid, "polymarket_volume_usd": 60_000})
    eval_json = tmp / "eval.json"
    odds_json = tmp / "odds.json"
    eval_json.write_text(json.dumps({"matches": matches}))
    odds_json.write_text(json.dumps({"matches": odds}))
    return str(eval_json), str(odds_json)


def test_reslice_fallback_blocks_demote_contract_and_reliability():
    """Rows with no cluster metadata land in team-pair fallback blocks; the
    summary must say so loudly instead of stamping the real contract (the
    pre-2026-08-14 behavior silently reported tournament_time_block_v1 with
    bootstrap_reliable computed from the inflated pair-block count)."""
    with tempfile.TemporaryDirectory() as tmp:
        ep, op = _write_reslice_fixture(Path(tmp))
        out = rs.reslice(ep, op, min_volume=None)
        s = out["summary"]
        assert s["cluster_resolution"] == {
            "stamped": 0, "lookup": 0, "fallback": 5,
        }
        assert s["bootstrap_contract"] == \
            "tournament_time_block_v1_fallback_pair_blocks"
        assert s["bootstrap_reliable"] is False
        assert s["bootstrap_unreliable_reason"] == "fallback_pair_blocks"
        assert s["cluster_metadata_coverage"] == 0


def test_reslice_stamped_clusters_keep_contract_when_enough_blocks():
    with tempfile.TemporaryDirectory() as tmp:
        ep, op = _write_clustered_fixture(Path(tmp), n_clusters=12)
        out = rs.reslice(ep, op, min_volume=50_000)
        s = out["summary"]
        assert s["cluster_resolution"]["stamped"] == 12
        assert s["cluster_resolution"]["fallback"] == 0
        assert s["bootstrap_contract"] == "tournament_time_block_v1"
        assert s["n_bootstrap_clusters"] == 12
        assert s["bootstrap_reliable"] is True
        assert s["bootstrap_unreliable_reason"] is None
        assert s["cluster_metadata_coverage"] == 12


def test_reslice_few_stamped_clusters_stay_descriptive():
    """<10 real blocks keeps the honest contract stamp but must not read
    as reliable."""
    with tempfile.TemporaryDirectory() as tmp:
        ep, op = _write_clustered_fixture(Path(tmp), n_clusters=4)
        out = rs.reslice(ep, op, min_volume=50_000)
        s = out["summary"]
        assert s["bootstrap_contract"] == "tournament_time_block_v1"
        assert s["bootstrap_reliable"] is False
        assert s["bootstrap_unreliable_reason"] == "fewer_than_10_clusters"


def test_gate_requires_reliable_bootstrap_for_roi_leg():
    """Invariant #7 at the gate: a positive ROI CI on an unreliable
    bootstrap must not clear the gate's ROI condition."""
    from sim_eval import blend_report as br

    def _summary(ll, ci_low, reliable):
        return {
            "avg_log_loss": ll,
            "flat_betting_roi_ci_low": ci_low,
            "bootstrap_reliable": reliable,
        }

    grid = {"min_volume_50000": {
        0.0: _summary(0.55, 2.0, True),    # clears both, reliably
        0.5: _summary(0.55, 2.0, False),   # positive CI but descriptive
        1.0: _summary(0.70, -1.0, True),   # clears neither
    }}
    market = {"min_volume_50000": {"market_ll": 0.60, "n_priced": 3}}
    g = br._gate_check(grid, market)
    assert g["roi_ci_excludes_zero"] == [0.0]
    assert g["roi_ci_positive_but_descriptive_only"] == [0.5]
    assert g["both_clear"] == [0.0]

    # Legacy sliced JSONs without the flag fail closed on the ROI leg.
    legacy = {"min_volume_50000": {
        0.0: {"avg_log_loss": 0.55, "flat_betting_roi_ci_low": 2.0},
    }}
    g2 = br._gate_check(legacy, market)
    assert g2["roi_ci_excludes_zero"] == []
    assert g2["roi_ci_positive_but_descriptive_only"] == [0.0]
    assert g2["both_clear"] == []


def test_a7_summary_counts_zero_return_win_without_pnl_sentinel():
    summary = a7._summarize(
        pnls=[0.0, -1.0],
        clusters=["event:a", "event:b"],
        wins=[True, False],
        n_resamples=100,
    )
    assert summary["n_bets"] == 2
    assert summary["win_rate"] == 0.5


def test_sizing_summary_counts_zero_return_flat_win():
    row = {
        "match_id": "m_zero",
        "teams": ["A", "B"],
        "edge": {"A": 0.1, "B": -0.1},
        "market_odds": {"A": 1.0, "B": 9.0},
        "actual_winner": "A",
    }
    summary = sizing.evaluate(
        [row],
        vol_by_id={},
        feat_lookup={},
        threshold=0.0,
        sizing="flat",
        n_resamples=100,
    )
    assert summary["n_bets"] == 1
    assert summary["total_pnl"] == 0.0
    assert summary["win_rate"] == 1.0


def test_sizing_kelly_zero_fraction_is_not_a_placed_bet():
    row = {
        "match_id": "m_no_kelly",
        "teams": ["A", "B"],
        "edge": {"A": 0.1, "B": -0.1},
        "market_odds": {"A": 2.0, "B": 2.0},
        "actual_winner": "A",
        "full_kelly_fraction": 0.0,
    }
    summary = sizing.evaluate(
        [row],
        vol_by_id={},
        feat_lookup={},
        threshold=0.0,
        sizing="kelly",
        n_resamples=100,
    )
    assert summary["n_bets"] == 0


def test_invalid_or_nonfinite_odds_do_not_place_bets():
    base = {
        "match_id": "m_invalid",
        "teams": ["A", "B"],
        "edge": {"A": 0.1, "B": -0.1},
        "actual_winner": "A",
        "realized_pnl": 0.0,
    }
    for invalid in (None, "bad", float("nan"), 0.9):
        row = dict(base, market_odds={"A": invalid, "B": 2.0})
        assert flat_bet_team(row) is None
        assert sizing._compute_pnl(row, "flat", 0.25, 0.02) is None


# --------------------------------------------------------------------------
# F. blend path — identities, alignment, parity with the evaluator
# --------------------------------------------------------------------------

def _blend_fixture():
    sim_json = {"matches": [
        {"match_id": "m1", "teams": ["A", "B"],
         "simulated_prob": {"A": 0.7, "B": 0.3},
         "market_prob": {"A": 0.55, "B": 0.45},
         "market_odds": {"A": 1.82, "B": 2.22},
         "actual_winner": "A", "log_loss": -math.log(0.7),
         "brier_score": 0.09, "realized_pnl": 0.82},
        {"match_id": "m2", "teams": ["A", "B"],
         "simulated_prob": {"A": 0.45, "B": 0.55},
         "market_prob": {"A": 0.5, "B": 0.5},
         "market_odds": {"A": 2.0, "B": 2.0},
         "actual_winner": "B", "log_loss": -math.log(0.55),
         "brier_score": 0.2025, "realized_pnl": 1.0},
        {"match_id": "m3", "teams": ["A", "B"],
         "simulated_prob": {"A": 0.6, "B": 0.4},
         "market_prob": {"A": 0.6, "B": 0.4},
         "market_odds": {"A": 1.67, "B": 2.5},
         "actual_winner": "B", "log_loss": -math.log(0.4),
         "brier_score": 0.36, "realized_pnl": 0.0},
    ]}
    direct = {
        "m1": {"team1": "A", "team2": "B", "p_team1": 0.6, "p_team2": 0.4},
        # m2: direct file orders the teams the OTHER way round
        "m2": {"team1": "B", "team2": "A", "p_team1": 0.35, "p_team2": 0.65},
        # m3 deliberately missing -> passthrough
    }
    return sim_json, direct


def test_blend_w0_is_direct_w1_is_sim():
    sim_json, direct = _blend_fixture()
    out0 = bl.blend(sim_json, direct, w=0.0)
    m1 = out0["matches"][0]
    assert abs(m1["simulated_prob"]["A"] - 0.6) < 1e-9, \
        "w=0 must reproduce the direct model probability"
    assert abs(m1["log_loss"] - (-math.log(0.6))) < 1e-9
    out1 = bl.blend(sim_json, direct, w=1.0)
    assert abs(out1["matches"][0]["simulated_prob"]["A"] - 0.7) < 1e-9, \
        "w=1 must reproduce the sim probability"
    assert out0["summary"]["n_matches_blended"] == 2
    assert out0["summary"]["n_matches_passthrough"] == 1


def test_blend_midpoint_is_logit_space():
    sim_json, direct = _blend_fixture()
    out = bl.blend(sim_json, direct, w=0.5)
    got = out["matches"][0]["simulated_prob"]["A"]
    logit = lambda p: math.log(p / (1 - p))
    expect = 1.0 / (1.0 + math.exp(-(0.5 * logit(0.7) + 0.5 * logit(0.6))))
    assert abs(got - expect) < 1e-9, f"blend is in logit space: {got} vs {expect}"


def test_blend_aligns_reversed_direct_team_order():
    sim_json, direct = _blend_fixture()
    out = bl.blend(sim_json, direct, w=0.0)
    m2 = out["matches"][1]
    # direct entry lists team1=B (p 0.35), team2=A (p 0.65); eval team1 is A.
    assert abs(m2["simulated_prob"]["A"] - 0.65) < 1e-9, \
        "direct predictions must be re-aligned to the eval JSON's team1"


def test_blend_passthrough_missing_direct_preserves_metrics_and_adds_contract():
    sim_json, direct = _blend_fixture()
    out = bl.blend(sim_json, direct, w=0.0)
    passthrough = out["matches"][2]
    for key, value in sim_json["matches"][2].items():
        assert passthrough[key] == value, \
            "a missing direct prediction must preserve source metrics"
    assert passthrough["bet_placed"] is False
    assert passthrough["bet_team"] is None
    # 2026-07-30: blend must NOT stamp a fallback cluster id — a stamped
    # team-pair fallback overrides reslice's event-time block lookup and
    # silently degrades the I3 bootstrap (observed: 134 clusters vs 19).
    assert "competition_cluster_id" not in passthrough


def test_blend_envelope_passthrough_fails_closed():
    """SRV4 (2026-08-14 review): a synthetic envelope's sim_prob is a 50/50
    placeholder — a join miss must raise, not silently score the placeholder
    as the model in the w=0 golden flow."""
    sim_json, direct = _blend_fixture()
    sim_json.setdefault("summary", {})["envelope_for"] = \
        "golden direct-only eval (w=0 blend)"
    with np.testing.assert_raises(RuntimeError):
        bl.blend(sim_json, direct, w=0.0)
    # Non-envelope sim JSONs keep the documented passthrough behavior.
    del sim_json["summary"]["envelope_for"]
    out = bl.blend(sim_json, direct, w=0.0)
    assert out["summary"]["n_matches_passthrough"] >= 1


def test_blend_recomputed_metrics_match_evaluator_formulas():
    ev = _evaluator()
    sim_json, direct = _blend_fixture()
    out = bl.blend(sim_json, direct, w=0.0)
    m1 = out["matches"][0]
    p = m1["simulated_prob"]
    # Edge = blended prob - market prob.
    assert abs(m1["edge"]["A"] - (0.6 - 0.55)) < 1e-9
    # Brier: blend's two-class sum/2 equals the evaluator's one-sided form.
    assert abs(m1["brier_score"] -
               ev._calculate_brier_score(p, "A", "A", "B")) < 1e-9
    # Realized pnl: bet A (edge +0.05), won at 1.82 -> +0.82.
    assert abs(m1["realized_pnl"] - 0.82) < 1e-9
    assert abs(m1["realized_pnl"] -
               ev._calculate_realized_pnl(m1["edge"], m1["market_odds"], "A")) < 1e-12
    # Kelly parity on the chosen side.
    assert abs(m1["full_kelly_fraction"] -
               ev._calculate_kelly_fraction(p["A"], 1.82)) < 1e-12
    assert abs(m1["full_kelly_pnl"] - m1["full_kelly_fraction"] * 0.82) < 1e-12
    assert abs(m1["fractional_kelly_pnl"] - m1["full_kelly_pnl"] * 0.25) < 1e-12


def test_blend_realized_pnl_mirror_grid():
    """bl._recompute_realized_pnl claims to 'mirror match_evaluator.
    _calculate_realized_pnl exactly' — hold it to that over the branch
    space (the loop's recipe-A envelope depends on this mirroring)."""
    ev = _evaluator()
    odds_full = {"A": 2.5, "B": 3.0}
    cases = [
        ({"A": 0.1, "B": -0.1}, odds_full, "A"),      # win
        ({"A": 0.1, "B": -0.1}, odds_full, "B"),      # loss
        ({"A": 0.05, "B": 0.12}, odds_full, "B"),     # max-edge choice, win
        ({"A": 0.0, "B": -0.1}, odds_full, "A"),      # threshold: no bet
        ({"A": -0.2, "B": -0.1}, odds_full, "A"),     # no positive edge
        ({"A": 0.1, "B": -0.1}, odds_full, None),     # no winner
        ({}, odds_full, "A"),                         # empty edge
        ({"A": 0.1, "B": -0.1}, {}, "A"),             # empty odds
        ({"A": 0.1, "B": -0.1}, {"B": 3.0}, "A"),     # best team missing odds
    ]
    for edge, odds, winner in cases:
        a = bl._recompute_realized_pnl(edge, odds, winner)
        b = ev._calculate_realized_pnl(edge, odds, winner)
        assert a == b, f"mirror broken for {(edge, odds, winner)}: {a} vs {b}"
    assert bl._recompute_realized_pnl(
        {"A": 0.1, "B": -0.1},
        {"B": 3.0},
        "A",
    ) == 0.0, "missing odds for the selected team means no placed bet"
    assert bl.BET_EDGE_THRESHOLD == BET_EDGE_THRESHOLD, \
        "blend duplicates the threshold constant — must stay in lockstep"


def test_blend_persists_recomputed_bet_contract():
    sim_json, direct = _blend_fixture()
    out = bl.blend(sim_json, direct, w=0.0)
    blended = out["matches"][0]
    assert blended["bet_placed"] is True
    assert blended["bet_team"] == "A"
    assert "competition_cluster_id" not in blended
    passthrough = out["matches"][2]
    assert passthrough["bet_placed"] is False
    assert passthrough["bet_team"] is None
    assert "competition_cluster_id" not in passthrough


# --------------------------------------------------------------------------
# Standalone runner (repo convention) — reports xfails explicitly.
# --------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    n_pass = n_xfail = 0
    failures = []
    for name, fn in tests:
        reason = getattr(fn, "__xfail_reason__", None)
        try:
            fn()
        except AssertionError as e:
            if reason:
                print(f"XFAIL {name}\n      ({reason})")
                n_xfail += 1
            else:
                print(f"FAIL  {name}: {e}")
                failures.append(name)
        else:
            if reason:
                print(f"ERROR {name}: XPASS — the known bug appears fixed; "
                      f"remove the xfail marker")
                failures.append(name)
            else:
                print(f"PASS  {name}")
                n_pass += 1
    print(f"\n{n_pass} passed, {n_xfail} xfailed (known bugs), "
          f"{len(failures)} failed")
    sys.exit(1 if failures else 0)
