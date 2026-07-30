"""Tests for the transient same-day forward statistics overlay."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from parsing_v2 import (  # noqa: E402
    PlayerEloTracker,
    PlayerStatsTracker,
    VenueStatsTracker,
    parse_match_data_v2,
)
from elo_update import PROVISIONAL_ELO_UPDATE_VERSION  # noqa: E402
from loaders_common import extract_match_metadata  # noqa: E402
from sim_eval import same_day_stats as module  # noqa: E402
from sim_eval.same_day_stats import SameDayReplayStatsProvider  # noqa: E402
from stats_provider import wrap_with_cache  # noqa: E402


class _FakeBackend:
    def __init__(self):
        self._prior = (0.30, 0.30, 0.15, 0.12, 0.05, 0.08)

    def _ensure_conn(self):
        return object()


class _FakeBaseProvider:
    def __init__(self):
        self._backend = _FakeBackend()

    def get_phase_outcome_dist(self, balls_bowled):
        return {"phase_p0": float(balls_bowled)}


class _Metadata:
    def get_player_metadata(self, player_id):
        return {
            "batter_hand": "right",
            "bowler_arm": "right",
            "is_pace": True,
            "bowling_type": "fast",
        }

    def get_player_age(self, player_id, match_date):
        return 30.0

    def get_matchup_type(self, batter_id, bowler_id):
        return "RHB_vs_fast"

    def get_spin_matchup_advantage(self, batter_id, bowler_id):
        return 0

    def get_same_arm_matchup(self, batter_id, bowler_id):
        return True


def _empty_rehydration(monkeypatch):
    monkeypatch.setattr(
        module,
        "rehydrate_stats_tracker",
        lambda provider, date, players: PlayerStatsTracker(),
    )
    monkeypatch.setattr(
        module,
        "rehydrate_elo_tracker",
        lambda provider, date, players: PlayerEloTracker(),
    )
    monkeypatch.setattr(
        module,
        "rehydrate_venue_tracker",
        lambda provider, date, venues: VenueStatsTracker(),
    )


def _match(date="2026-01-01", a_runs=4, b_runs=1):
    return {
        "meta": {"data_version": "1.1.0"},
        "info": {
            "dates": [date],
            "teams": ["A", "B"],
            "venue": "Ground",
            "team_type": "international",
            "event": {"name": "Synthetic"},
            "toss": {"winner": "A", "decision": "bat"},
            "registry": {
                "people": {
                    "A One": "a1",
                    "A Two": "a2",
                    "B One": "b1",
                    "B Two": "b2",
                }
            },
            "players": {
                "A": ["A One", "A Two"],
                "B": ["B One", "B Two"],
            },
            "outcome": {"winner": "A"},
        },
        "innings": [
            {
                "team": "A",
                "overs": [
                    {
                        "over": 0,
                        "deliveries": [
                            {
                                "batter": "A One",
                                "non_striker": "A Two",
                                "bowler": "B One",
                                "runs": {
                                    "batter": a_runs,
                                    "extras": 0,
                                    "total": a_runs,
                                },
                            }
                        ],
                    }
                ],
            },
            {
                "team": "B",
                "overs": [
                    {
                        "over": 0,
                        "deliveries": [
                            {
                                "batter": "B One",
                                "non_striker": "B Two",
                                "bowler": "A One",
                                "runs": {
                                    "batter": b_runs,
                                    "extras": 0,
                                    "total": b_runs,
                                },
                            }
                        ],
                    }
                ],
            },
        ],
    }


def _provider(monkeypatch):
    _empty_rehydration(monkeypatch)
    return SameDayReplayStatsProvider(_FakeBaseProvider(), _Metadata())


def test_prediction_must_lock_before_replay_and_cache_is_invalidated(
    monkeypatch,
):
    provider = _provider(monkeypatch)
    first = _match()
    second = _match(a_runs=2, b_runs=0)
    provider.begin_date("2026-01-01", [first, second])
    provider.begin_match(
        "001", first, prediction_required=True
    )

    # This value is memoized before replay. A successful advance must clear it.
    assert provider.get_batting_stats("a1", "2026-01-01") == {
        "avg": 0,
        "sr": 0,
    }
    with pytest.raises(RuntimeError, match="before its prediction is locked"):
        provider.advance_match("001", first)
    assert provider.get_batting_stats("a1", "2026-01-01")["avg"] == 0

    provider.lock_prediction("001")
    receipt = provider.advance_match("001", first)
    assert receipt == {
        "match_id": "001",
        "deliveries_replayed": 2,
        "innings_replayed": 2,
    }
    assert provider.matches_advanced == 1
    assert provider.get_batting_stats("a1", "2026-01-01") == {
        "avg": 4.0,
        "sr": 400.0,
    }
    assert provider.get_batting_recent("a1", "2026-01-01") == {
        "avg": 4.0,
        "sr": 400.0,
    }
    assert provider.get_h2h_stats("a1", "b1", "2026-01-01") == {
        "avg": 4.0,
        "sr": 400.0,
    }
    assert provider.get_batting_vs_type_stats(
        "a1", "2026-01-01"
    )["avg_vs_pace"] == 4.0
    assert provider.get_bowling_vs_hand_stats(
        "b1", "2026-01-01"
    )["avg_vs_rhb"] == 4.0
    assert provider.get_venue_avg_score("Ground", "2026-01-01") == 2.5
    venue = provider.get_venue_profile("Ground", "2026-01-01")
    assert venue["venue_boundary_pct"] == 0.5
    assert venue["venue_first_innings_avg"] == 4.0
    assert venue["venue_chase_win_pct"] == 0.0
    assert provider.get_batting_elo("a1", "2026-01-01") != 1500.0

    # The next same-day fixture sees the advanced state.
    provider.begin_match(
        "002", second, prediction_required=True
    )
    assert provider.get_batting_stats("a1", "2026-01-01")["avg"] == 4.0


def test_context_only_match_can_advance_without_prediction(monkeypatch):
    provider = _provider(monkeypatch)
    match = _match()
    provider.begin_date("2026-01-01", [match])
    provider.begin_match("001", match, prediction_required=False)
    provider.advance_match("001", match)
    assert provider.matches_advanced == 1


def test_same_day_order_and_date_guards(monkeypatch):
    provider = _provider(monkeypatch)
    match = _match()
    provider.begin_date("2026-01-01", [match])
    provider.begin_match("002", match, prediction_required=False)
    provider.advance_match("002", match)

    with pytest.raises(ValueError, match="strictly increasing by match_id"):
        provider.begin_match("001", match, prediction_required=False)
    with pytest.raises(ValueError, match="strictly increasing"):
        provider.begin_date("2026-01-01", [match])
    with pytest.raises(ValueError, match="not 2026-01-02"):
        provider.get_batting_stats("a1", "2026-01-02")


def test_i9_same_day_replay_matches_one_chronological_pass(monkeypatch):
    monkeypatch.setattr(
        module,
        "rehydrate_stats_tracker",
        lambda provider, date, players: PlayerStatsTracker(),
    )
    monkeypatch.setattr(
        module,
        "rehydrate_elo_tracker",
        lambda provider, date, players: PlayerEloTracker(
            PROVISIONAL_ELO_UPDATE_VERSION
        ),
    )
    monkeypatch.setattr(
        module,
        "rehydrate_venue_tracker",
        lambda provider, date, venues: VenueStatsTracker(),
    )
    provider = SameDayReplayStatsProvider(
        _FakeBaseProvider(),
        _Metadata(),
    )
    matches = [_match(), _match(a_runs=2, b_runs=0)]
    provider.begin_date("2026-01-01", matches)
    for match_id, match in zip(("001", "002"), matches):
        provider.begin_match(
            match_id,
            match,
            prediction_required=False,
        )
        provider.advance_match(match_id, match)

    reference_stats = PlayerStatsTracker()
    reference_elo = PlayerEloTracker(PROVISIONAL_ELO_UPDATE_VERSION)
    reference_venue = VenueStatsTracker()
    for match_id, match in zip(("001", "002"), matches):
        metadata = extract_match_metadata(match)
        _, _, venue, details, chase_won = parse_match_data_v2(
            json.dumps(match),
            reference_stats,
            reference_venue,
            _Metadata(),
            elo_tracker=reference_elo,
            match_k_factor=metadata["k_factor"],
            match_ref=match_id,
        )
        for detail in details:
            reference_venue.update_venue_stats_detailed(venue, detail)
        if chase_won is not None:
            reference_venue.update_venue_match_result(venue, chase_won)

    assert provider._elo.batting_elo == reference_elo.batting_elo
    assert provider._elo.bowling_elo == reference_elo.bowling_elo
    assert provider._elo.batting_exposure == reference_elo.batting_exposure
    assert provider._elo.bowling_exposure == reference_elo.bowling_exposure


def test_model_cache_wrapper_is_idempotent(monkeypatch):
    provider = _provider(monkeypatch)
    assert wrap_with_cache(provider) is provider


def test_pre_prediction_setup_reads_only_info(monkeypatch):
    provider = _provider(monkeypatch)
    match = _match()

    class _ForbiddenInnings:
        def __iter__(self):
            raise AssertionError("innings were read before prediction")

        def __len__(self):
            raise AssertionError("innings were read before prediction")

        def __getitem__(self, key):
            raise AssertionError("innings were read before prediction")

    guarded = {"info": match["info"], "innings": _ForbiddenInnings()}
    provider.begin_date("2026-01-01", [guarded])
    provider.begin_match("001", guarded, prediction_required=True)
    provider.lock_prediction("001")
