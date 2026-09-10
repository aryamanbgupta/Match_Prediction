"""Training/serving and state-machine contracts for the T1 simulator."""
from __future__ import annotations

import json
import random
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"

import sim_t1  # noqa: E402
from embeddings_e1 import (EB_BAT_COLS, EB_BOWL_COLS, VENUE_COLS,  # noqa: E402
                           make_ctx)
from sim_t1 import TransformerT1SimModel  # noqa: E402
from run_t1_sim_ppc import (_ordinary_test_order_reference,  # noqa: E402
                            _result_signature, _swap_storage)
from run_t1_sim_extras_ppc import _summary as extras_summary  # noqa: E402
from sim_v1_2 import (BowlerSelector, EmpiricalBowlerSelector, MatchState,  # noqa: E402
                      Outcome, Player, RandomBowlerSelector,
                      RosterEmpiricalBowlerSelector, SimulationEngine,
                      T20Rules, TeamLineup)
from transformer_t1 import BOS, build_features  # noqa: E402


def _lineup(name: str) -> TeamLineup:
    return TeamLineup(
        name,
        [Player(f"{name}-{i}", f"{name} player {i}", name)
         for i in range(11)],
    )


def _state(*, innings: int = 1, balls: int = 0, score: float = 0.0,
           wickets: int = 0) -> MatchState:
    state = MatchState(
        team1_lineup=_lineup("A"),
        team2_lineup=_lineup("B"),
        batting_first="A",
        venue="Test Ground",
        match_date=datetime(2025, 2, 3),
    )
    state.innings = innings
    state.balls = balls
    team = state.current_team_idx
    state.runs[team] = score
    state.wickets[team] = wickets
    return state


def _bare_wrapper(*, max_seq_len: int = 200) -> TransformerT1SimModel:
    model = TransformerT1SimModel.__new__(TransformerT1SimModel)
    model.max_seq_len = max_seq_len
    model._cache = model._empty_cache()
    model.extras_graft = None
    return model


class _RecordingSelector(BowlerSelector):
    def __init__(self):
        self.calls = []

    def select_bowler(self, state, available):
        choice = 4 if 4 in available else available[0]
        self.calls.append((state.balls, choice))
        return choice


class _AlwaysWicketModel:
    delivery_semantics = "inclusive_total_runs_v1"
    extras_graft = None

    def extract_features(self, state):
        return None

    def predict_next_ball(self, features):
        return {"wicket": 1.0}


def test_sim_feature_vector_is_build_features_exact(monkeypatch):
    state = _state(innings=2, balls=48, score=50.0, wickets=3)
    state.runs[0] = 160.0
    dist_cols = EB_BAT_COLS + EB_BOWL_COLS + VENUE_COLS
    dist = {name: (i + 1) / 100.0 for i, name in enumerate(dist_cols)}

    def fill(features, *_args, **_kwargs):
        features.update(dist)

    monkeypatch.setattr(sim_t1, "_fill_outcome_dists", fill)
    wrapper = _bare_wrapper()
    wrapper.stats_provider = object()
    actual = wrapper._ball_features(state)

    balls_remaining = 120 - state.balls
    rrr = (state.target - state.runs[state.current_team_idx]) * 6 \
        / balls_remaining
    row = {
        **dist,
        "is_middle_overs": 1.0,
        "is_death_overs": 0.0,
        "wickets_in_hand": 7.0,
        "chase_target": float(state.target),
        "balls_remaining": float(balls_remaining),
        "score": 50.0,
        "run_rate": 50.0 / (48 / 6),
        "run_rate_required": float(rrr),
    }
    expected = build_features(pd.DataFrame([row]))[0]
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(
        actual[42:46], make_ctx(pd.DataFrame([row]))[0])


def test_run_rate_matches_training_after_opening_extra(monkeypatch):
    state = _state(balls=0, score=1.0)
    monkeypatch.setattr(
        sim_t1, "_fill_outcome_dists",
        lambda features, *_args, **_kwargs: None,
    )
    wrapper = _bare_wrapper()
    wrapper.stats_provider = object()
    vec = wrapper._ball_features(state)
    # score / max(0 overs, 0.1) = 10; then T1 divides by 12.
    assert vec[-2] == pytest.approx(10.0 / 12.0)


def test_cache_tracks_exact_state_and_shifted_outcomes():
    wrapper = _bare_wrapper()
    state = _state()
    first, prev = wrapper._sequence_inputs(
        state, np.zeros(sim_t1.N_FEATS, dtype=np.float32))
    assert len(first) == 1
    assert prev == [BOS]

    state.update(Outcome.ONE, 1)
    feats, prev = wrapper._sequence_inputs(
        state, np.ones(sim_t1.N_FEATS, dtype=np.float32))
    assert len(feats) == 2
    assert prev == [BOS, 1]

    state.start_new_innings()
    feats, prev = wrapper._sequence_inputs(
        state, np.full(sim_t1.N_FEATS, 2.0, dtype=np.float32))
    assert len(feats) == 1
    assert prev == [BOS]


def test_cache_fails_closed_on_repeat_call_or_historical_copy():
    wrapper = _bare_wrapper()
    state = _state()
    token = np.zeros(sim_t1.N_FEATS, dtype=np.float32)
    wrapper._sequence_inputs(state, token)
    with pytest.raises(RuntimeError, match="desynchronization"):
        wrapper._sequence_inputs(state, token)

    state.update(Outcome.FOUR, 4)
    copied = state.copy()
    fresh_wrapper = _bare_wrapper()
    with pytest.raises(RuntimeError, match="existing current-innings history"):
        fresh_wrapper._sequence_inputs(copied, token)


def test_cache_refuses_position_aliasing_past_checkpoint_capacity():
    wrapper = _bare_wrapper(max_seq_len=1)
    state = _state()
    token = np.zeros(sim_t1.N_FEATS, dtype=np.float32)
    wrapper._sequence_inputs(state, token)
    state.update(Outcome.DOT, 0)
    with pytest.raises(RuntimeError, match="position capacity"):
        wrapper._sequence_inputs(state, token)


def test_flat_and_empirical_extras_composition_contracts():
    wrapper = _bare_wrapper()
    probs = np.array([0.2, 0.3, 0.1, 0.15, 0.05, 0.2])
    flat = wrapper._compose_delivery_probs(probs)
    assert sum(flat.values()) == pytest.approx(1.0)
    assert flat["wide"] == flat["no_ball"] == 0.01
    assert flat["one"] / flat["dot"] == pytest.approx(1.5)

    wrapper.extras_graft = SimpleNamespace(p_wide=0.04, p_no_ball=0.006)
    empirical = wrapper._compose_delivery_probs(probs)
    assert sum(empirical.values()) == pytest.approx(1.0)
    assert empirical["wide"] == pytest.approx(0.04)
    assert empirical["no_ball"] == pytest.approx(0.006)
    assert empirical["one"] / empirical["dot"] == pytest.approx(1.5)


def test_extras_aggregate_accepts_corrected_baseline_schema():
    innings = {
        "total_runs": 100, "wickets": 5, "legal_balls": 120,
        "deliveries": 122, "extras_events": 4, "unique_bowlers": 6,
        "phase_runs": {"powerplay": 30, "middle": 50, "death": 20},
    }
    simulation = {"winner": "A", "innings": [innings, innings]}
    match = {
        "match_id": "m1", "first_batting_team": "A",
        "actual_winner": "A", "actual_innings": [innings, innings],
        "simulations": [simulation],
    }
    candidate = {"engine_contract": "first_over_selector_v1",
                 "n_sims": 1, "matches": [match]}
    baseline = {"engine_contract": "first_over_selector_v1",
                "n_sims": 1, "matches": [match]}
    config = {"statistics": {"bootstrap_repetitions": 10,
                              "bootstrap_seed": 1}}
    result = extras_summary(candidate, baseline, config)
    assert result["baseline_comparison_valid"] is True
    assert result["extras_events_per_innings"]["flat_baseline"] == 4


def test_state_copy_preserves_transient_delivery_channels():
    state = _state()
    state.last_dismissal = "runout_nonstriker"
    state.last_batter_runs = 4
    state.last_bowler_runs = 5
    state.last_is_legal = False
    state.active_bowler_rosters = {1: (2, 3, 4, 5, 6)}
    state.active_bowler_quotas = {1: (0, 0, 4, 4, 4, 4, 4, 0, 0, 0, 0)}
    copied = state.copy()
    assert copied.last_dismissal == "runout_nonstriker"
    assert copied.last_batter_runs == 4
    assert copied.last_bowler_runs == 5
    assert copied.last_is_legal is False
    assert copied.active_bowler_rosters == {1: (2, 3, 4, 5, 6)}
    assert copied.active_bowler_quotas == {
        1: (0, 0, 4, 4, 4, 4, 4, 0, 0, 0, 0)}


def test_extra_is_legal_on_nominal_last_ball():
    state = _state(balls=119)
    rules = T20Rules(RandomBowlerSelector())
    assert rules.is_legal_outcome(state, Outcome.WIDE)
    assert rules.is_legal_outcome(state, Outcome.NO_BALL)


def test_first_over_uses_configured_bowler_selector():
    selector = _RecordingSelector()
    engine = SimulationEngine(_AlwaysWicketModel(), T20Rules(selector))
    result = engine._simulate_innings(_state())
    assert selector.calls[0] == (0, 4)
    assert result.balls[0].bowler_idx == 4
    assert {ball.bowler_idx for ball in result.balls[:6]} == {4}


def test_team_storage_swap_is_exact_for_deterministic_rollout():
    selector = _RecordingSelector()
    engine = SimulationEngine(_AlwaysWicketModel(), T20Rules(selector))
    random.seed(29)
    original = engine.simulate_match(_state(), match_id="original")
    random.seed(29)
    swapped = engine.simulate_match(
        _swap_storage(_state()), match_id="swapped")
    assert _result_signature(original) == _result_signature(swapped)


def test_bowler_phase_prior_excludes_current_and_future_years(monkeypatch):
    selector = EmpiricalBowlerSelector("unused.json")
    payload = {
        "by_year_league": {
            "2023": {"pp_share": 0.2, "mid_share": 0.5,
                     "death_share": 0.3, "total_balls": 100},
            "2024": {"pp_share": 0.4, "mid_share": 0.5,
                     "death_share": 0.1, "total_balls": 300},
            # Must not influence a 2025 fixture.
            "2025": {"pp_share": 0.9, "mid_share": 0.05,
                     "death_share": 0.05, "total_balls": 10000},
            "2026": {"pp_share": 0.0, "mid_share": 0.0,
                     "death_share": 1.0, "total_balls": 10000},
        },
    }
    monkeypatch.setattr(selector, "_load", lambda: payload)
    share = selector._league_share(2025)
    assert share == pytest.approx({"pp": 0.35, "mid": 0.5,
                                   "death": 0.15})


def test_order_flip_is_compared_to_behavior_not_forced_to_zero(tmp_path):
    winners = ["Chasing", "Chasing", "First"]
    for index, winner in enumerate(winners):
        document = {
            "info": {"outcome": {"winner": winner}},
            "innings": [{"team": "First"}, {"team": "Chasing"}],
        }
        (tmp_path / f"{index}.json").write_text(json.dumps(document))
    result = _ordinary_test_order_reference(tmp_path, reps=200, seed=7)
    assert result["n_decided_matches"] == 3
    assert result["first_batting_win_rate"] == pytest.approx(1 / 3)
    assert result["chasing_win_rate"] == pytest.approx(2 / 3)
    assert result["chasing_minus_first_batting_win_rate"] == pytest.approx(
        1 / 3)


def test_roster_selector_is_causal_and_keeps_bowling_inside_unit(tmp_path):
    usage_path = tmp_path / "usage.json"
    usage_path.write_text(json.dumps({
        "by_player": {},
        "by_year_league": {
            "2023": {"pp_share": 0.3, "mid_share": 0.5,
                     "death_share": 0.2, "total_balls": 100},
        },
    }))
    roster_path = tmp_path / "roster.json"
    roster_path.write_text(json.dumps({
        "policy": "causal_intended_bowling_roster_size_v1",
        "cold_start_counts": {"6": 1},
        "by_year": {
            "2023": {"5": 1},
            # A 2025 fixture may not use current or future years.
            "2025": {"9": 100},
            "2026": {"9": 100},
        },
    }))
    selector = RosterEmpiricalBowlerSelector(
        str(usage_path), str(roster_path))
    assert selector._roster_counts(2025) == {5: 1}
    state = _state()
    draws = [selector.select_bowler(state, list(range(11)))
             for _ in range(50)]
    roster = state.active_bowler_rosters[state.bowling_team_idx]
    assert len(roster) == 5
    assert set(draws) <= set(roster)

    # A five-player unit must remain schedulable for all 20 overs; unbounded
    # random sampling can otherwise exhaust four bowlers too early.
    state = _state()
    selector = RosterEmpiricalBowlerSelector(
        str(usage_path), str(roster_path))
    used = set()
    for _ in range(20):
        chosen = selector.select_bowler(state, state.get_available_bowlers())
        used.add(chosen)
        state.bowler_idx = chosen
        for _ in range(6):
            state.update(Outcome.DOT, 0)
    assert state.balls == 120
    assert used == set(state.active_bowler_rosters[state.bowling_team_idx])

    # Mid-innings first entry is unsupported: the 20-over quota schedule
    # assumes a full regulation innings, so the selector must fail closed
    # instead of raising an obscure infeasibility error later.
    late_state = _state(balls=36)
    late_selector = RosterEmpiricalBowlerSelector(
        str(usage_path), str(roster_path))
    with pytest.raises(ValueError, match="first over"):
        late_selector.select_bowler(late_state, list(range(11)))
