"""Rehydrate PlayerStatsTracker / PlayerEloTracker / VenueStatsTracker
from a SQLite snapshot.

Phase A parity harness uses these to prove that a SQLite-seeded
materializer reproduces the monolith's output bit-for-bit. Phase B's
`materialize_features.py` reuses them for the real pipeline.

Design:
* Rehydrate state needed for the date batch: the union of players across
  all same-day matches, plus the set of same-day venues. Per-date batching
  means that on a solo-date match, we touch ~250 SQLite reads (22 pids × 6
  tables + ~100 h2h pairs + 2 ELO × 22 + 1 venue); a 4-fixture ICC day
  with ~80 unique pids scales h2h cross-product to ~6 400 queries. Absolute
  cost is tiny on mmap SQLite (p50 ~3 µs → ~20 ms/date), but watch for
  regressions if h2h schema changes its PK shape.
* Raw counters, not computed averages — the tracker re-derives averages
  from counters, so we match its internal representation exactly.
* `first_innings_totals` is reconstructed losslessly w.r.t. sum AND len
  (see `_reconstruct_fi_list`). Contract: **no consumer may iterate
  individual entries of that list** — only `sum(…) / len(…)` reads are
  supported. Currently honored by `VenueStatsTracker.get_venue_profile` /
  `get_venue_avg_score` (parsing_v2.py). Adding a variance/trajectory
  feature that reads individual entries would silently diverge from the
  monolith; if such a feature is needed, extend SQLite to store the full
  list (schema bump) before reading it here.

Architectural note (TODO): this module reaches into
`_SQLiteBackend._get_raw_batting` / `_player_id_map` / `_resolve_date_id` —
private accessors on the backend. Promoting them to public API (or adding a
batched `get_rehydration_snapshot` method) is tracked in TODO.md as a
follow-up to Phase B's facade narrowing.
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

    **Contract (observation-equivalence, NOT equality)**: this returns a
    list that is indistinguishable from the monolith's original list
    under `sum(list) / len(list)` — i.e. only aggregate consumers are
    supported. Individual entries are NOT reconstructible from SQLite
    because only (fi_sum, fi_count) is stored.

    Current consumers in `parse_match_data_v2` are all aggregate:
    - `VenueStatsTracker.get_venue_avg_score` — `sum / len`
    - `VenueStatsTracker.get_venue_profile` — `sum / len` (as
      `venue_first_innings_avg`)

    If you add a feature that reads individual entries (e.g. a
    first-innings-variance or trajectory feature), this reconstruction
    will silently diverge from the monolith. In that case, extend the
    SQLite `venue` schema to store the full list (e.g. a new
    `venue_first_innings_totals(venue_id, date_id, seq_idx, runs)` table)
    and rewrite this function to read it. At ~18k rows full-corpus, the
    storage cost is negligible.

    We spread the remainder across the first `r` entries so all values
    stay integers; the final `sum / len` division is then bit-identical
    to the monolith's path.
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
            # Schema v3: seed the deque with up to 5 individual match
            # aggregates from `batting_match_log`. This reproduces the
            # monolith's eviction behavior on same-day `end_match` pushes,
            # which a single summed seed cannot. Log is newest-first;
            # deque append order is oldest-first.
            log = backend.get_batting_match_log_recent(pid, as_of, limit=5)
            for entry in reversed(log):
                tracker.recent_batting[pid].append({
                    "runs": int(entry["runs"]),
                    "balls": int(entry["balls"]),
                    "dismissals": int(entry["dismissals"]),
                })

        raw = backend._get_raw_bowling(pid, as_of)
        if raw is not None:
            tracker.bowling_stats[pid] = {
                "runs_given": int(raw["runs_given"]),
                "balls_bowled": int(raw["balls_bowled"]),
                "wickets": int(raw["wickets"]),
            }
            # Schema v3: same treatment for bowling recent-form.
            log = backend.get_bowling_match_log_recent(pid, as_of, limit=5)
            for entry in reversed(log):
                tracker.recent_bowling[pid].append({
                    "runs_given": int(entry["runs_given"]),
                    "balls_bowled": int(entry["balls_bowled"]),
                    "wickets": int(entry["wickets"]),
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
