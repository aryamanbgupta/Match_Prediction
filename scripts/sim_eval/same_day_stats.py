"""Transient, chronology-safe statistics for forward ball-model scoring.

The sealed SQLite cache stores one aggregate snapshot per date: the state
immediately before the first fixture on that date.  That is sufficient for
the first match, but a later same-day match must also see completed earlier
fixtures.  This module rehydrates the existing in-memory trackers once per
date and advances them only after the current match's prediction is locked.

Nothing is written to SQLite or to the production model directory.
"""
from __future__ import annotations

import copy
import json
from datetime import datetime
from typing import Iterable, Mapping, Optional

from loaders_common import extract_match_metadata
from identity_maps import canonicalize_venue
from parsing_v2 import parse_match_data_v2
from stats_provider import StatsProviderCache
from tracker_rehydration import (
    rehydrate_elo_tracker,
    rehydrate_stats_tracker,
    rehydrate_venue_tracker,
)


def _norm_date(value) -> str:
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d")
    return str(value)


def _match_date(match_data: Mapping) -> str:
    dates = match_data["info"].get("dates", [])
    if not dates:
        raise ValueError("match is missing info.dates[0]")
    return _norm_date(dates[0])


def _safe_player_ids(match_data: Mapping) -> set[str]:
    """Return player IDs using pre-match ``info`` fields only."""
    info = match_data["info"]
    registry = info.get("registry", {}).get("people", {})
    player_ids = {str(value) for value in registry.values()}
    for names in (info.get("players", {}) or {}).values():
        for name in names:
            player_ids.add(str(registry.get(name, name)))
    return player_ids


class _TrackerStatsView:
    """StatsProvider-compatible view over rehydrated mutable trackers."""

    def __init__(self, base_provider, player_metadata):
        self._base_provider = base_provider
        self._player_metadata = player_metadata
        self._stats = None
        self._elo = None
        self._venue = None
        self._prior = None
        self._current_date: Optional[str] = None
        self._last_date: Optional[str] = None
        self._last_match_id: Optional[str] = None
        self._active_match: Optional[dict] = None
        self.matches_advanced = 0

    @property
    def current_date(self) -> Optional[str]:
        return self._current_date

    def begin_date(
        self,
        as_of_date,
        match_documents: Iterable[Mapping],
    ) -> dict:
        """Rehydrate first-of-date state from pre-match ``info`` only."""
        if self._active_match is not None:
            raise RuntimeError(
                "cannot change date while a match is awaiting replay"
            )

        date_str = _norm_date(as_of_date)
        if self._last_date is not None and date_str <= self._last_date:
            raise ValueError(
                "forward dates must be strictly increasing: "
                f"previous={self._last_date}, requested={date_str}"
            )

        documents = list(match_documents)
        if not documents:
            raise ValueError(f"no match documents supplied for {date_str}")

        player_ids: set[str] = set()
        venues: set[str] = set()
        for match_data in documents:
            actual_date = _match_date(match_data)
            if actual_date != date_str:
                raise ValueError(
                    f"date batch mismatch: expected {date_str}, "
                    f"found {actual_date}"
                )
            player_ids.update(_safe_player_ids(match_data))
            venues.add(match_data["info"].get("venue", "unknown"))

        self._stats = rehydrate_stats_tracker(
            self._base_provider, date_str, sorted(player_ids)
        )
        self._elo = rehydrate_elo_tracker(
            self._base_provider, date_str, sorted(player_ids)
        )
        self._venue = rehydrate_venue_tracker(
            self._base_provider, date_str, sorted(venues)
        )

        backend = self._base_provider._backend
        backend._ensure_conn()
        self._prior = backend._prior
        self._current_date = date_str
        self._last_date = date_str
        self._last_match_id = None
        return {
            "date": date_str,
            "players_rehydrated": len(player_ids),
            "venues_rehydrated": len(venues),
        }

    def begin_match(
        self,
        match_id: str,
        match_data: Mapping,
        *,
        prediction_required: bool,
    ) -> None:
        """Open one fixture without reading its innings or outcome."""
        self._require_ready()
        if self._active_match is not None:
            raise RuntimeError(
                f"match {self._active_match['match_id']} is still active"
            )
        if _match_date(match_data) != self._current_date:
            raise ValueError(
                f"{match_id} is not in active date {self._current_date}"
            )
        if self._last_match_id is not None and match_id <= self._last_match_id:
            raise ValueError(
                "same-day matches must be strictly increasing by match_id: "
                f"previous={self._last_match_id}, requested={match_id}"
            )

        info = match_data["info"]
        self._active_match = {
            "match_id": str(match_id),
            "prediction_required": bool(prediction_required),
            "prediction_locked": False,
            "date": self._current_date,
            "teams": tuple(info.get("teams", [])),
            "venue": canonicalize_venue(info.get("venue")),
        }

    def lock_prediction(self, match_id: str) -> None:
        """Mark the current evaluated prediction immutable before replay."""
        active = self._require_active(match_id)
        if not active["prediction_required"]:
            raise RuntimeError(
                f"{match_id} is context-only and has no prediction to lock"
            )
        if active["prediction_locked"]:
            raise RuntimeError(f"prediction already locked for {match_id}")
        active["prediction_locked"] = True

    def advance_match(self, match_id: str, match_data: Mapping) -> dict:
        """Apply actual deliveries after the prediction-order guard passes."""
        active = self._require_active(match_id)
        if active["prediction_required"] and not active["prediction_locked"]:
            raise RuntimeError(
                f"cannot replay {match_id} before its prediction is locked"
            )

        info = match_data["info"]
        identity = (
            _match_date(match_data),
            tuple(info.get("teams", [])),
            canonicalize_venue(info.get("venue")),
        )
        expected = (
            active["date"],
            active["teams"],
            active["venue"],
        )
        if identity != expected:
            raise ValueError(
                f"replay document identity mismatch for {match_id}: "
                f"expected={expected!r}, found={identity!r}"
            )

        # Replay is transactional. A malformed JSON must not leave partially
        # advanced trackers that could contaminate a later prediction.
        snapshot = (
            copy.deepcopy(self._stats),
            copy.deepcopy(self._elo),
            copy.deepcopy(self._venue),
        )
        try:
            metadata = extract_match_metadata(dict(match_data))
            rows, innings_totals, venue, details, chase_won = (
                parse_match_data_v2(
                    json.dumps(match_data),
                    self._stats,
                    self._venue,
                    self._player_metadata,
                    elo_tracker=self._elo,
                    match_k_factor=metadata["k_factor"],
                    match_ref=str(match_id),
                )
            )
            for detail in details:
                self._venue.update_venue_stats_detailed(venue, detail)
            if chase_won is not None:
                self._venue.update_venue_match_result(venue, chase_won)
        except Exception:
            self._stats, self._elo, self._venue = snapshot
            raise

        self._active_match = None
        self._last_match_id = str(match_id)
        self.matches_advanced += 1
        return {
            "match_id": str(match_id),
            "deliveries_replayed": len(rows),
            "innings_replayed": len(innings_totals),
        }

    def _require_ready(self) -> None:
        if self._current_date is None or self._stats is None:
            raise RuntimeError("begin_date must be called before stats access")

    def _require_date(self, as_of_date) -> None:
        self._require_ready()
        requested = _norm_date(as_of_date)
        if requested != self._current_date:
            raise ValueError(
                f"stats view is active for {self._current_date}, "
                f"not {requested}"
            )

    def _require_active(self, match_id: str) -> dict:
        if self._active_match is None:
            raise RuntimeError("no active match")
        if self._active_match["match_id"] != str(match_id):
            raise ValueError(
                f"active match is {self._active_match['match_id']}, "
                f"not {match_id}"
            )
        return self._active_match

    # Player and matchup statistics.

    def get_batting_stats(self, player_id, as_of_date):
        self._require_date(as_of_date)
        row = self._stats.get_batting_features(player_id)
        return {"avg": row["batsman_avg"], "sr": row["batsman_sr"]}

    def get_batting_recent(self, player_id, as_of_date):
        self._require_date(as_of_date)
        row = self._stats.get_batting_features(player_id)
        return {
            "avg": row["batsman_recent_avg"],
            "sr": row["batsman_recent_sr"],
        }

    def get_bowling_stats(self, player_id, as_of_date):
        self._require_date(as_of_date)
        row = self._stats.get_bowling_features(player_id)
        return {"avg": row["bowler_avg"], "econ": row["bowler_econ"]}

    def get_bowling_recent(self, player_id, as_of_date):
        self._require_date(as_of_date)
        row = self._stats.get_bowling_features(player_id)
        return {
            "avg": row["bowler_recent_avg"],
            "econ": row["bowler_recent_econ"],
        }

    def get_h2h_stats(self, batter_id, bowler_id, as_of_date):
        self._require_date(as_of_date)
        row = self._stats.get_h2h_features(batter_id, bowler_id)
        return {"avg": row["h2h_avg"], "sr": row["h2h_sr"]}

    def get_batting_vs_type_stats(self, player_id, as_of_date):
        self._require_date(as_of_date)
        row = self._stats.get_batting_vs_type_features(player_id)
        return {
            "avg_vs_pace": row["batter_avg_vs_pace"],
            "sr_vs_pace": row["batter_sr_vs_pace"],
            "avg_vs_spin": row["batter_avg_vs_spin"],
            "sr_vs_spin": row["batter_sr_vs_spin"],
        }

    def get_bowling_vs_hand_stats(self, player_id, as_of_date):
        self._require_date(as_of_date)
        row = self._stats.get_bowling_vs_hand_features(player_id)
        return {
            "avg_vs_lhb": row["bowler_avg_vs_lhb"],
            "econ_vs_lhb": row["bowler_econ_vs_lhb"],
            "avg_vs_rhb": row["bowler_avg_vs_rhb"],
            "econ_vs_rhb": row["bowler_econ_vs_rhb"],
        }

    # ELO and team aggregates.

    def get_batting_elo(self, player_id, as_of_date):
        self._require_date(as_of_date)
        return self._elo.get_batting_elo(player_id)

    def get_bowling_elo(self, player_id, as_of_date):
        self._require_date(as_of_date)
        return self._elo.get_bowling_elo(player_id)

    def get_team_batting_elo(self, player_ids, as_of_date):
        self._require_date(as_of_date)
        return self._elo.get_team_batting_elo(player_ids)

    def get_team_bowling_elo(self, player_ids, as_of_date):
        self._require_date(as_of_date)
        return self._elo.get_team_bowling_elo(player_ids)

    def get_team_batting_strength(self, player_ids, as_of_date):
        self._require_date(as_of_date)
        avgs, strike_rates = [], []
        for player_id in player_ids:
            row = self.get_batting_stats(player_id, as_of_date)
            if row["avg"] > 0:
                avgs.append(row["avg"])
                strike_rates.append(row["sr"])
        return {
            "team_batting_avg": sum(avgs) / len(avgs) if avgs else 0.0,
            "team_batting_sr": (
                sum(strike_rates) / len(strike_rates)
                if strike_rates else 0.0
            ),
        }

    def get_team_bowling_strength(self, player_ids, as_of_date):
        self._require_date(as_of_date)
        avgs, economies = [], []
        for player_id in player_ids:
            row = self.get_bowling_stats(player_id, as_of_date)
            if row["avg"] > 0:
                avgs.append(row["avg"])
                economies.append(row["econ"])
        return {
            "team_bowling_avg": sum(avgs) / len(avgs) if avgs else 0.0,
            "team_bowling_econ": (
                sum(economies) / len(economies) if economies else 0.0
            ),
        }

    # Venue and empirical outcome distributions.

    def get_venue_avg_score(self, venue, as_of_date):
        self._require_date(as_of_date)
        return self._venue.get_venue_avg_score(venue)

    def get_venue_profile(self, venue, as_of_date):
        self._require_date(as_of_date)
        return self._venue.get_venue_profile(venue)

    def get_batter_outcome_dist(
        self, player_id, as_of_date, k: float = 30.0
    ):
        self._require_date(as_of_date)
        return self._stats.get_batter_outcome_dist(
            player_id, self._prior, k=k
        )

    def get_bowler_outcome_dist(
        self, player_id, as_of_date, k: float = 30.0
    ):
        self._require_date(as_of_date)
        return self._stats.get_bowler_outcome_dist(
            player_id, self._prior, k=k
        )

    def get_batter_vs_type_outcome_dist(
        self,
        player_id,
        as_of_date,
        k: float = 30.0,
        hierarchical: bool = True,
    ):
        self._require_date(as_of_date)
        return self._stats.get_batter_vs_type_outcome_dist(
            player_id,
            self._prior,
            k=k,
            hierarchical=hierarchical,
        )

    def get_bowler_vs_hand_outcome_dist(
        self,
        player_id,
        as_of_date,
        k: float = 30.0,
        hierarchical: bool = True,
    ):
        self._require_date(as_of_date)
        return self._stats.get_bowler_vs_hand_outcome_dist(
            player_id,
            self._prior,
            k=k,
            hierarchical=hierarchical,
        )

    def get_venue_outcome_dist(
        self, venue, as_of_date, k: float = 200.0
    ):
        self._require_date(as_of_date)
        return self._venue.get_venue_outcome_dist(
            venue, self._prior, k=k
        )

    def get_phase_outcome_dist(self, balls_bowled: int):
        self._require_ready()
        return self._base_provider.get_phase_outcome_dist(balls_bowled)

    def get_all_stats(self, batter_id, bowler_id, as_of_date):
        batting = self.get_batting_stats(batter_id, as_of_date)
        bowling = self.get_bowling_stats(bowler_id, as_of_date)
        h2h = self.get_h2h_stats(batter_id, bowler_id, as_of_date)
        return {
            "batsman_avg": batting["avg"],
            "batsman_sr": batting["sr"],
            "bowler_avg": bowling["avg"],
            "bowler_econ": bowling["econ"],
            "h2h_avg": h2h["avg"],
            "h2h_sr": h2h["sr"],
        }


class SameDayReplayStatsProvider(StatsProviderCache):
    """Mutable replay provider with automatic cache invalidation.

    It subclasses ``StatsProviderCache`` deliberately: the simulation model's
    existing ``wrap_with_cache`` helper therefore treats this object as already
    wrapped.  Every successful state mutation clears all memoized values.
    """

    def __init__(self, base_provider, player_metadata):
        self._live_view = _TrackerStatsView(
            base_provider=base_provider,
            player_metadata=player_metadata,
        )
        super().__init__(self._live_view)

    @property
    def current_date(self) -> Optional[str]:
        return self._live_view.current_date

    @property
    def matches_advanced(self) -> int:
        return self._live_view.matches_advanced

    def begin_date(self, as_of_date, match_documents):
        result = self._live_view.begin_date(as_of_date, match_documents)
        self.clear_memo()
        return result

    def begin_match(
        self,
        match_id: str,
        match_data: Mapping,
        *,
        prediction_required: bool,
    ) -> None:
        self._live_view.begin_match(
            match_id,
            match_data,
            prediction_required=prediction_required,
        )

    def lock_prediction(self, match_id: str) -> None:
        self._live_view.lock_prediction(match_id)

    def advance_match(self, match_id: str, match_data: Mapping):
        result = self._live_view.advance_match(match_id, match_data)
        self.clear_memo()
        return result
