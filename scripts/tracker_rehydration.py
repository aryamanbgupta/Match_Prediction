"""Rehydrate PlayerStatsTracker / PlayerEloTracker / VenueStatsTracker
from a SQLite snapshot.

Phase A parity harness uses these to prove that a per-match, SQLite-seeded
materializer reproduces the monolith's output bit-for-bit. Phase B's
`materialize_features.py` will reuse them.

Design:
* Only rehydrate state needed by one match: players in the two lineups plus
  the venue. This bounds each per-match rehydration to ~250 SQLite reads.
* Raw counters, not computed averages — the tracker re-derives averages
  from counters, so we match its internal representation exactly.
* `first_innings_totals` is reconstructed as a length-`count` integer list
  whose sum equals the stored `fi_sum`; that preserves `sum(list) /
  len(list)` exactly (down to the last ULP) even though the individual
  innings totals are lost in SQLite.
"""
from __future__ import annotations

import sys
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent))

from parsing_v2 import PlayerEloTracker, PlayerStatsTracker, VenueStatsTracker
from stats_sqlite_backend import (
    _Q_BATTING_VS_TYPE,
    _Q_BOWLING_VS_HAND,
    _Q_VENUE,
)


def _norm_date(as_of_date) -> str:
    if isinstance(as_of_date, datetime):
        return as_of_date.strftime("%Y-%m-%d")
    return as_of_date


def _get_raw_batting_vs_type(backend, player_id, bowl_type: int, as_of_date):
    """Raw counters (runs, balls, dismissals) or None — not exposed as a
    private method on _SQLiteBackend today, so inline the same query."""
    conn = backend._ensure_conn()
    pid = backend._player_id_map.get(str(player_id))
    if pid is None:
        return None
    did = backend._resolve_date_id(as_of_date)
    if did < 0:
        return None
    return conn.execute(_Q_BATTING_VS_TYPE, (pid, bowl_type, did)).fetchone()


def _get_raw_bowling_vs_hand(backend, player_id, bat_hand: int, as_of_date):
    conn = backend._ensure_conn()
    pid = backend._player_id_map.get(str(player_id))
    if pid is None:
        return None
    did = backend._resolve_date_id(as_of_date)
    if did < 0:
        return None
    return conn.execute(_Q_BOWLING_VS_HAND, (pid, bat_hand, did)).fetchone()


def _get_raw_venue(backend, venue: str, as_of_date):
    conn = backend._ensure_conn()
    vid = backend._venue_id_map.get(venue)
    if vid is None:
        return None
    did = backend._resolve_date_id(as_of_date)
    if did < 0:
        return None
    return conn.execute(_Q_VENUE, (vid, did)).fetchone()


def _reconstruct_fi_list(fi_sum: int, fi_count: int) -> list[int]:
    """Return a length-fi_count integer list summing to fi_sum.

    The monolith keeps `first_innings_totals` as a list of individual
    innings scores; SQLite stores only (sum, count). `get_venue_profile`
    uses `sum(list) / len(list)` — so any reconstruction with the same
    sum and length is observation-equivalent. We spread the remainder
    across the first `r` entries so all values stay integers; this keeps
    the final float division bit-identical to the monolith's path.
    """
    if fi_count == 0:
        return []
    q, r = divmod(fi_sum, fi_count)
    return [q + 1] * r + [q] * (fi_count - r)


def _backend(provider):
    """StatsProvider facade → underlying _SQLiteBackend."""
    backend = provider._backend
    if type(backend).__name__ != "_SQLiteBackend":
        raise RuntimeError(
            "tracker rehydration requires SQLite backend; got "
            f"{type(backend).__name__}"
        )
    return backend


def rehydrate_stats_tracker(
    provider,
    as_of_date,
    player_ids: Iterable[str],
) -> PlayerStatsTracker:
    """Seed a PlayerStatsTracker from the SQLite snapshot at `as_of_date`.

    Populates batting_stats, bowling_stats, batting_vs_type,
    bowling_vs_hand, and the `recent_*` single-entry deques for every
    pid in `player_ids`. h2h is populated for every (batter, bowler)
    cross-product pair of those pids (both directions).
    """
    backend = _backend(provider)
    as_of = _norm_date(as_of_date)

    tracker = PlayerStatsTracker()
    pids = list(player_ids)

    for pid in pids:
        raw = backend._get_raw_batting(pid, as_of)
        if raw is not None:
            tracker.batting_stats[pid] = {
                "runs": int(raw["runs"]),
                "balls": int(raw["balls"]),
                "dismissals": int(raw["dismissals"]),
            }
            # Seed recent deque with a single aggregated entry — sum of a
            # 1-entry deque == pre-summed total, so get_batting_features'
            # sum() call returns the same value as the monolith.
            tracker.recent_batting[pid].append({
                "runs": int(raw["recent_runs"]),
                "balls": int(raw["recent_balls"]),
                "dismissals": int(raw["recent_dismissals"]),
            })

        raw = backend._get_raw_bowling(pid, as_of)
        if raw is not None:
            tracker.bowling_stats[pid] = {
                "runs_given": int(raw["runs_given"]),
                "balls_bowled": int(raw["balls_bowled"]),
                "wickets": int(raw["wickets"]),
            }
            tracker.recent_bowling[pid].append({
                "runs_given": int(raw["recent_runs_given"]),
                "balls_bowled": int(raw["recent_balls_bowled"]),
                "wickets": int(raw["recent_wickets"]),
            })

        # pace=0, spin=1
        pace = _get_raw_batting_vs_type(backend, pid, 0, as_of)
        spin = _get_raw_batting_vs_type(backend, pid, 1, as_of)
        if pace is not None or spin is not None:
            entry = {
                "pace": {"runs": 0, "balls": 0, "dismissals": 0},
                "spin": {"runs": 0, "balls": 0, "dismissals": 0},
            }
            if pace is not None:
                entry["pace"] = {
                    "runs": int(pace[0]), "balls": int(pace[1]),
                    "dismissals": int(pace[2]),
                }
            if spin is not None:
                entry["spin"] = {
                    "runs": int(spin[0]), "balls": int(spin[1]),
                    "dismissals": int(spin[2]),
                }
            tracker.batting_vs_type[pid] = entry

        # left=0, right=1
        lhb = _get_raw_bowling_vs_hand(backend, pid, 0, as_of)
        rhb = _get_raw_bowling_vs_hand(backend, pid, 1, as_of)
        if lhb is not None or rhb is not None:
            entry = {
                "left":  {"runs_given": 0, "balls_bowled": 0, "wickets": 0},
                "right": {"runs_given": 0, "balls_bowled": 0, "wickets": 0},
            }
            if lhb is not None:
                entry["left"] = {
                    "runs_given": int(lhb[0]), "balls_bowled": int(lhb[1]),
                    "wickets": int(lhb[2]),
                }
            if rhb is not None:
                entry["right"] = {
                    "runs_given": int(rhb[0]), "balls_bowled": int(rhb[1]),
                    "wickets": int(rhb[2]),
                }
            tracker.bowling_vs_hand[pid] = entry

    # h2h: cross-product of all pids in both directions. Most pairs have
    # no history and return None; the defaultdict auto-returns zeros for
    # unseeded keys, matching the monolith's behavior.
    for bat in pids:
        for bowl in pids:
            if bat == bowl:
                continue
            raw = backend._get_raw_h2h(bat, bowl, as_of)
            if raw is not None:
                tracker.h2h_stats[(bat, bowl)] = {
                    "runs": int(raw["runs"]),
                    "balls": int(raw["balls"]),
                    "dismissals": int(raw["dismissals"]),
                }

    return tracker


def rehydrate_elo_tracker(
    provider,
    as_of_date,
    player_ids: Iterable[str],
) -> PlayerEloTracker:
    """Seed a PlayerEloTracker with batting/bowling ELO for each pid."""
    backend = _backend(provider)
    as_of = _norm_date(as_of_date)

    tracker = PlayerEloTracker()
    for pid in player_ids:
        bat = backend.get_batting_elo(pid, as_of)
        bowl = backend.get_bowling_elo(pid, as_of)
        # Only record non-default ratings — default (1500.0) is what the
        # monolith returns for players with no history, via dict.get().
        if bat != PlayerEloTracker.DEFAULT_ELO:
            tracker.batting_elo[pid] = bat
        if bowl != PlayerEloTracker.DEFAULT_ELO:
            tracker.bowling_elo[pid] = bowl
    return tracker


def rehydrate_venue_tracker(
    provider,
    as_of_date,
    venues,
) -> VenueStatsTracker:
    """Seed a VenueStatsTracker with one-or-more venues' counters.

    Accepts either a single venue string or an iterable of venues —
    needed for per-date batching where multiple matches may be at
    different grounds on the same date.
    """
    backend = _backend(provider)
    as_of = _norm_date(as_of_date)

    if isinstance(venues, str):
        venue_list = [venues]
    else:
        venue_list = list(venues)

    tracker = VenueStatsTracker()
    for venue in venue_list:
        row = _get_raw_venue(backend, venue, as_of)
        if row is None:
            continue
        (total_runs, innings_count, total_balls, total_boundaries,
         total_dots, total_wickets, pp_runs, pp_balls,
         death_runs, death_balls, fi_sum, fi_count,
         matches_total, chase_wins) = row

        tracker.venue_stats[venue] = {
            "total_runs": int(total_runs),
            "innings_count": int(innings_count),
            "total_balls": int(total_balls),
            "total_boundaries": int(total_boundaries),
            "total_dots": int(total_dots),
            "total_wickets": int(total_wickets),
            "powerplay_runs": int(pp_runs),
            "powerplay_balls": int(pp_balls),
            "death_runs": int(death_runs),
            "death_balls": int(death_balls),
            "first_innings_totals": _reconstruct_fi_list(
                int(fi_sum), int(fi_count)),
            "matches_total": int(matches_total),
            "chase_wins": int(chase_wins),
        }
    return tracker


def extract_match_player_ids(match_json_data: dict) -> list[str]:
    """All player IDs from both teams' lineups in a cricsheet match dict."""
    registry = match_json_data["info"]["registry"]["people"]
    pids: set[str] = set()
    for team, names in match_json_data["info"].get("players", {}).items():
        for name in names:
            pid = registry.get(name)
            if pid is not None:
                pids.add(pid)
    return sorted(pids)
