"""Pins the T1 candidate runner's same-day replay lifecycle.

`run_sim_eval_t1._T1ReplayEvaluator.evaluate_all` must drive the replay
provider exactly like the certified PPC runners: every same-day fixture
advances the trackers in chronological order, evaluated fixtures lock their
prediction before their own deliveries replay, and odds-less fixtures still
advance as context.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]

from sim_eval.run_sim_eval_t1 import _T1ReplayEvaluator  # noqa: E402


class _RecordingProvider:
    def __init__(self):
        self.events = []

    def begin_date(self, date_str, docs):
        self.events.append(("begin_date", date_str, len(docs)))

    def begin_match(self, match_id, _doc, *, prediction_required):
        self.events.append(("begin_match", match_id, prediction_required))

    def lock_prediction(self, match_id):
        self.events.append(("lock", match_id))

    def advance_match(self, match_id, _doc):
        self.events.append(("advance", match_id))


def _doc(date):
    return {"info": {"dates": [date]}}


def _evaluator(provider, context_by_date):
    ev = _T1ReplayEvaluator(
        model=None, simulation_engine=None, n_simulations=1, parallel=False)
    ev.provider = provider
    ev.context_by_date = context_by_date
    ev._resolve_odds_row = (
        lambda match_id, state, odds_lookup, claimed_rows=None:
        odds_lookup.get(match_id))
    ev._evaluate_single_match = lambda match_id, state, odds: SimpleNamespace(
        match_id=match_id, simulation_time=0.0,
        simulated_win_prob={"A": 0.6, "B": 0.4},
        log_loss=0.5, actual_winner="A")
    ev._aggregate_results = lambda results, total_time: results
    return ev


def test_lifecycle_orders_context_prediction_and_replay():
    provider = _RecordingProvider()
    context = {
        "2025-01-01": [("ctx1", _doc("2025-01-01")),
                       ("m1", _doc("2025-01-01"))],
        "2025-01-02": [("m2", _doc("2025-01-02")),
                       ("ctx2", _doc("2025-01-02"))],
    }
    ev = _evaluator(provider, context)
    matches = [("m1", SimpleNamespace(team1="A", team2="B")),
               ("m2", SimpleNamespace(team1="A", team2="B"))]
    results = ev.evaluate_all(matches, {"m1": {"odds": {}}})  # m2 has no odds

    assert [r.match_id for r in results] == ["m1"]
    assert provider.events == [
        ("begin_date", "2025-01-01", 2),
        ("begin_match", "ctx1", False),
        ("advance", "ctx1"),
        ("begin_match", "m1", True),
        ("lock", "m1"),
        ("advance", "m1"),
        ("begin_date", "2025-01-02", 2),
        # No odds -> context-only advance, never locked.
        ("begin_match", "m2", False),
        ("advance", "m2"),
        ("begin_match", "ctx2", False),
        ("advance", "ctx2"),
    ]


def test_dates_without_requested_matches_are_skipped_whole():
    provider = _RecordingProvider()
    context = {
        "2025-01-01": [("m1", _doc("2025-01-01"))],
        "2025-01-02": [("other", _doc("2025-01-02"))],
        "2025-01-03": [("m3", _doc("2025-01-03"))],
    }
    ev = _evaluator(provider, context)
    matches = [("m1", SimpleNamespace(team1="A", team2="B")),
               ("m3", SimpleNamespace(team1="A", team2="B"))]
    ev.evaluate_all(matches, {"m1": {"odds": {}}, "m3": {"odds": {}}})
    dates = [e[1] for e in provider.events if e[0] == "begin_date"]
    assert dates == ["2025-01-01", "2025-01-03"]


def test_simulation_error_still_locks_and_advances():
    provider = _RecordingProvider()
    context = {"2025-01-01": [("m1", _doc("2025-01-01"))]}
    ev = _evaluator(provider, context)

    def _boom(match_id, state, odds):
        raise RuntimeError("sim exploded")

    ev._evaluate_single_match = _boom
    results = ev.evaluate_all(
        [("m1", SimpleNamespace(team1="A", team2="B"))], {"m1": {"odds": {}}})
    assert results == []
    assert provider.events[-2:] == [("lock", "m1"), ("advance", "m1")]
