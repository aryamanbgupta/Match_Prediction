"""SQLite-backed StatsProvider.

Replaces the per-date pickle-chunk cache with a single mmap-backed SQLite
file. All reader workers share OS page cache on the same file, which is
the structural fix for the parallel-eval RSS blowup (see
feedback_no_parallel_sim_eval).

Layout (all tables use integer IDs internally to cut ~3x vs TEXT keys):

  players (id PK, player_id TEXT UNIQUE)
  dates   (id PK, date TEXT UNIQUE)       -- id is assigned in sorted order
  venues  (id PK, venue TEXT UNIQUE)

  batting / bowling / batting_elo / bowling_elo
    : PRIMARY KEY (player_id, date_id) WITHOUT ROWID, delta-compressed
  h2h   : rowid + UNIQUE INDEX (batter_id, bowler_id, date_id)
          (big table; WITHOUT ROWID would embed the composite PK into
          every internal b-tree page and bloat the file)
  batting_vs_type (player_id, date_id, bowl_type) / bowling_vs_hand (bat_hand)
    : WITHOUT ROWID on the composite PK, delta-compressed
  venue : (venue_id, date_id) WITHOUT ROWID, counters only — callers
          derive pct/avg in Python
  _meta : key/value text pairs (schema_version, build_timestamp, ...)

Every getter runs the same query shape:

    SELECT ... FROM <table>
    WHERE key = ? AND date_id <= ?
    ORDER BY date_id DESC LIMIT 1

so the planner walks the primary/unique index backwards and stops at the
first hit. EXPLAIN QUERY PLAN must report `SEARCH TABLE ... USING INDEX`
for every getter — the benchmark enforces this.

Process model:
  * __init__ stores the db_path only. No connection.
  * _ensure_conn() opens the connection lazily and re-opens it if it
    observes a PID change (fork safety: a sqlite3.Connection object can't
    legally be shared across processes).
  * __getstate__ strips (_conn, _conn_pid). multiprocessing.Pool.starmap
    pickles the model, including this backend, which would otherwise
    carry a PID-bound Connection into the worker.

Date resolution: the caller passes an arbitrary date string (possibly
not aligned with a snapshot). We bisect into self._date_strs (loaded
once at first query) to find the largest snapshot date ≤ target; its
date_id is the array index because dates.id is assigned in sorted order
at build time.
"""
from __future__ import annotations

import bisect
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


SCHEMA_VERSION = 4


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS players (
    id INTEGER PRIMARY KEY,
    player_id TEXT UNIQUE NOT NULL
);
CREATE TABLE IF NOT EXISTS dates (
    id INTEGER PRIMARY KEY,
    date TEXT UNIQUE NOT NULL
);
CREATE TABLE IF NOT EXISTS venues (
    id INTEGER PRIMARY KEY,
    venue TEXT UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS batting (
    player_id INTEGER NOT NULL,
    date_id INTEGER NOT NULL,
    runs INTEGER NOT NULL,
    balls INTEGER NOT NULL,
    dismissals INTEGER NOT NULL,
    recent_runs INTEGER NOT NULL DEFAULT 0,
    recent_balls INTEGER NOT NULL DEFAULT 0,
    recent_dismissals INTEGER NOT NULL DEFAULT 0,
    c0 INTEGER NOT NULL DEFAULT 0,
    c1 INTEGER NOT NULL DEFAULT 0,
    c2 INTEGER NOT NULL DEFAULT 0,
    c4 INTEGER NOT NULL DEFAULT 0,
    c6 INTEGER NOT NULL DEFAULT 0,
    cw INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (player_id, date_id)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS bowling (
    player_id INTEGER NOT NULL,
    date_id INTEGER NOT NULL,
    runs_given INTEGER NOT NULL,
    balls_bowled INTEGER NOT NULL,
    wickets INTEGER NOT NULL,
    recent_runs_given INTEGER NOT NULL DEFAULT 0,
    recent_balls_bowled INTEGER NOT NULL DEFAULT 0,
    recent_wickets INTEGER NOT NULL DEFAULT 0,
    c0 INTEGER NOT NULL DEFAULT 0,
    c1 INTEGER NOT NULL DEFAULT 0,
    c2 INTEGER NOT NULL DEFAULT 0,
    c4 INTEGER NOT NULL DEFAULT 0,
    c6 INTEGER NOT NULL DEFAULT 0,
    cw INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (player_id, date_id)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS h2h (
    rowid INTEGER PRIMARY KEY,
    batter_id INTEGER NOT NULL,
    bowler_id INTEGER NOT NULL,
    date_id INTEGER NOT NULL,
    runs INTEGER NOT NULL,
    balls INTEGER NOT NULL,
    dismissals INTEGER NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS h2h_lookup
    ON h2h (batter_id, bowler_id, date_id);

CREATE TABLE IF NOT EXISTS batting_vs_type (
    player_id INTEGER NOT NULL,
    date_id INTEGER NOT NULL,
    bowl_type INTEGER NOT NULL,  -- 0=pace, 1=spin
    runs INTEGER NOT NULL,
    balls INTEGER NOT NULL,
    dismissals INTEGER NOT NULL,
    c0 INTEGER NOT NULL DEFAULT 0,
    c1 INTEGER NOT NULL DEFAULT 0,
    c2 INTEGER NOT NULL DEFAULT 0,
    c4 INTEGER NOT NULL DEFAULT 0,
    c6 INTEGER NOT NULL DEFAULT 0,
    cw INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (player_id, bowl_type, date_id)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS bowling_vs_hand (
    player_id INTEGER NOT NULL,
    date_id INTEGER NOT NULL,
    bat_hand INTEGER NOT NULL,  -- 0=left, 1=right
    runs_given INTEGER NOT NULL,
    balls_bowled INTEGER NOT NULL,
    wickets INTEGER NOT NULL,
    c0 INTEGER NOT NULL DEFAULT 0,
    c1 INTEGER NOT NULL DEFAULT 0,
    c2 INTEGER NOT NULL DEFAULT 0,
    c4 INTEGER NOT NULL DEFAULT 0,
    c6 INTEGER NOT NULL DEFAULT 0,
    cw INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (player_id, bat_hand, date_id)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS venue (
    venue_id INTEGER NOT NULL,
    date_id INTEGER NOT NULL,
    total_runs INTEGER NOT NULL,
    innings_count INTEGER NOT NULL,
    total_balls INTEGER NOT NULL,
    total_boundaries INTEGER NOT NULL,
    total_dots INTEGER NOT NULL,
    total_wickets INTEGER NOT NULL,
    powerplay_runs INTEGER NOT NULL,
    powerplay_balls INTEGER NOT NULL,
    death_runs INTEGER NOT NULL,
    death_balls INTEGER NOT NULL,
    fi_totals_sum INTEGER NOT NULL,
    fi_totals_count INTEGER NOT NULL,
    matches_total INTEGER NOT NULL,
    chase_wins INTEGER NOT NULL,
    c0 INTEGER NOT NULL DEFAULT 0,
    c1 INTEGER NOT NULL DEFAULT 0,
    c2 INTEGER NOT NULL DEFAULT 0,
    c4 INTEGER NOT NULL DEFAULT 0,
    c6 INTEGER NOT NULL DEFAULT 0,
    cw INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (venue_id, date_id)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS batting_elo (
    player_id INTEGER NOT NULL,
    date_id INTEGER NOT NULL,
    elo REAL NOT NULL,
    PRIMARY KEY (player_id, date_id)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS bowling_elo (
    player_id INTEGER NOT NULL,
    date_id INTEGER NOT NULL,
    elo REAL NOT NULL,
    PRIMARY KEY (player_id, date_id)
) WITHOUT ROWID;

-- Schema v3: per-match aggregate logs. One row per (player, match) for
-- each role (batting / bowling). The deque-reconstruction path reads the
-- 5 most recent rows strictly before an as-of-date, allowing bit-exact
-- reproduction of monolith's recent-form evictions on same-day secondary
-- matches. Pre-summed recent_* columns on batting/bowling remain as a
-- denormalized cache for the simulation hot path (E1 decision).
CREATE TABLE IF NOT EXISTS batting_match_log (
    player_id INTEGER NOT NULL,
    date_id INTEGER NOT NULL,
    intra_date_idx INTEGER NOT NULL,   -- 0,1,2... versioned match-ID order
    runs INTEGER NOT NULL,
    balls INTEGER NOT NULL,
    dismissals INTEGER NOT NULL,
    PRIMARY KEY (player_id, date_id, intra_date_idx)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS bowling_match_log (
    player_id INTEGER NOT NULL,
    date_id INTEGER NOT NULL,
    intra_date_idx INTEGER NOT NULL,
    runs_given INTEGER NOT NULL,
    balls_bowled INTEGER NOT NULL,
    wickets INTEGER NOT NULL,
    PRIMARY KEY (player_id, date_id, intra_date_idx)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS _meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""


# Putting the composite-PK column for the branch discriminator (bowl_type,
# bat_hand) immediately after player_id in the PK order lets
# (player_id, bowl_type, date_id <= ?) walk the PK index without any
# extra secondary index. Same for bowling_vs_hand.

_Q_BATTING = """
SELECT runs, balls, dismissals,
       recent_runs, recent_balls, recent_dismissals,
       c0, c1, c2, c4, c6, cw
FROM batting
WHERE player_id = ? AND date_id <= ?
ORDER BY date_id DESC LIMIT 1
"""
_Q_BOWLING = """
SELECT runs_given, balls_bowled, wickets,
       recent_runs_given, recent_balls_bowled, recent_wickets,
       c0, c1, c2, c4, c6, cw
FROM bowling
WHERE player_id = ? AND date_id <= ?
ORDER BY date_id DESC LIMIT 1
"""
_Q_H2H = """
SELECT runs, balls, dismissals FROM h2h
WHERE batter_id = ? AND bowler_id = ? AND date_id <= ?
ORDER BY date_id DESC LIMIT 1
"""
_Q_BATTING_VS_TYPE = """
SELECT runs, balls, dismissals,
       c0, c1, c2, c4, c6, cw
FROM batting_vs_type
WHERE player_id = ? AND bowl_type = ? AND date_id <= ?
ORDER BY date_id DESC LIMIT 1
"""
_Q_BOWLING_VS_HAND = """
SELECT runs_given, balls_bowled, wickets,
       c0, c1, c2, c4, c6, cw
FROM bowling_vs_hand
WHERE player_id = ? AND bat_hand = ? AND date_id <= ?
ORDER BY date_id DESC LIMIT 1
"""
_Q_VENUE = """
SELECT total_runs, innings_count, total_balls, total_boundaries,
       total_dots, total_wickets, powerplay_runs, powerplay_balls,
       death_runs, death_balls, fi_totals_sum, fi_totals_count,
       matches_total, chase_wins,
       c0, c1, c2, c4, c6, cw
FROM venue WHERE venue_id = ? AND date_id <= ?
ORDER BY date_id DESC LIMIT 1
"""
_Q_BATTING_ELO = """
SELECT elo FROM batting_elo
WHERE player_id = ? AND date_id <= ?
ORDER BY date_id DESC LIMIT 1
"""
_Q_BOWLING_ELO = """
SELECT elo FROM bowling_elo
WHERE player_id = ? AND date_id <= ?
ORDER BY date_id DESC LIMIT 1
"""

# Schema v3 match-log recency queries. Strict `date_id < ?` (not `<=`) —
# we read at the pre-date boundary, so same-date matches must be excluded.
# PK (player_id, date_id, intra_date_idx) lets the planner walk the index
# backwards and LIMIT terminate early.
_Q_BATTING_LOG_RECENT = """
SELECT runs, balls, dismissals FROM batting_match_log
WHERE player_id = ? AND date_id < ?
ORDER BY date_id DESC, intra_date_idx DESC
LIMIT ?
"""
_Q_BOWLING_LOG_RECENT = """
SELECT runs_given, balls_bowled, wickets FROM bowling_match_log
WHERE player_id = ? AND date_id < ?
ORDER BY date_id DESC, intra_date_idx DESC
LIMIT ?
"""


# Every getter is one of these shapes. Exported so benchmarks can
# iterate the full set for EXPLAIN QUERY PLAN assertions.
QUERY_PLAN_CASES = [
    ("batting",             _Q_BATTING,             (1, 1)),
    ("bowling",             _Q_BOWLING,             (1, 1)),
    ("h2h",                 _Q_H2H,                 (1, 1, 1)),
    ("batting_vs_type",     _Q_BATTING_VS_TYPE,     (1, 0, 1)),
    ("bowling_vs_hand",     _Q_BOWLING_VS_HAND,     (1, 0, 1)),
    ("venue",               _Q_VENUE,               (1, 1)),
    ("batting_elo",         _Q_BATTING_ELO,         (1, 1)),
    ("bowling_elo",         _Q_BOWLING_ELO,         (1, 1)),
    ("batting_match_log",   _Q_BATTING_LOG_RECENT,  (1, 1, 5)),
    ("bowling_match_log",   _Q_BOWLING_LOG_RECENT,  (1, 1, 5)),
]


class _SQLiteBackend:
    """Read-only SQLite backend. Public API mirrors StatsProvider."""

    def __init__(self, db_path):
        self.db_path = str(db_path)
        self._conn: Optional[sqlite3.Connection] = None
        self._conn_pid: Optional[int] = None
        # Lookup tables loaded on first open.
        self._player_id_map: Optional[Dict[str, int]] = None
        self._venue_id_map: Optional[Dict[str, int]] = None
        self._date_strs: Optional[list] = None  # sorted; index = date_id
        # Schema v4: global empirical outcome prior (p0, p1, p2, p4, p6, pw),
        # loaded from _meta at first connection. Used for empirical-Bayes
        # shrinkage in get_*_outcome_dist getters.
        self._prior: Optional[tuple] = None
        # Phase 3: per-phase priors. Keyed by 'powerplay'/'middle'/'death'.
        # Loaded lazily from _meta; falls back to the global prior π if the
        # phase rows are absent (pre-Phase-3 cache). The per-phase getter
        # is the only consumer; outcome-dist features unchanged.
        self._phase_priors: Optional[Dict[str, tuple]] = None
        # Memo for `_norm_date`. The hot path calls `_norm_date(state.match_date)`
        # via `_resolve_date_id` on every getter — ~1.4 M times in a typical
        # 261×100 eval. The underlying `datetime.strftime` is ~1 µs each;
        # caching collapses the post-warmup cost to a dict lookup. Capped at
        # _DATE_NORM_CACHE_MAX so a misuse (e.g. dynamic dates) can't grow
        # the dict unboundedly. Stripped on pickle so workers start clean.
        self._date_norm_cache: Dict = {}

    _DATE_NORM_CACHE_MAX = 16

    # --- pickle: strip PID-bound connection so workers re-open ----------

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_conn'] = None
        state['_conn_pid'] = None
        # Don't drag the date-norm memo across fork; tiny but principled.
        state['_date_norm_cache'] = {}
        return state

    # --- connection & lookups -------------------------------------------

    def _ensure_conn(self) -> sqlite3.Connection:
        pid = os.getpid()
        conn = self._conn
        if conn is not None and self._conn_pid == pid:
            return conn

        conn = sqlite3.connect(
            f"file:{self.db_path}?mode=ro",
            uri=True,
            check_same_thread=False,
            cached_statements=64,
        )
        # Read-side pragmas. locking_mode=EXCLUSIVE is safe here because
        # we opened read-only; it skips per-query lock acquisition and is
        # a known latency win for hot read paths.
        conn.execute("PRAGMA mmap_size = 536870912")  # 512 MB
        conn.execute("PRAGMA query_only = 1")
        conn.execute("PRAGMA temp_store = MEMORY")
        conn.execute("PRAGMA cache_size = -4000")     # 4 MB per conn
        conn.execute("PRAGMA locking_mode = EXCLUSIVE")

        self._conn = conn
        self._conn_pid = pid

        if self._player_id_map is None:
            self._load_lookups(conn)
        return conn

    def _load_lookups(self, conn: sqlite3.Connection) -> None:
        self._player_id_map = {
            row[0]: row[1]
            for row in conn.execute("SELECT player_id, id FROM players")
        }
        self._venue_id_map = {
            row[0]: row[1]
            for row in conn.execute("SELECT venue, id FROM venues")
        }
        # dates.id is assigned in sorted order at build time; ORDER BY id
        # gives the same order as ORDER BY date.
        self._date_strs = [
            row[0]
            for row in conn.execute("SELECT date FROM dates ORDER BY id ASC")
        ]
        meta = dict(conn.execute("SELECT key, value FROM _meta"))
        # Fall back to a flat uniform prior if the DB predates v4; keeps
        # the backend usable in migration windows. Real rebuilds always
        # write the six prior_p* rows from build_stats_cache.
        try:
            self._prior = (
                float(meta['prior_p0']), float(meta['prior_p1']),
                float(meta['prior_p2']), float(meta['prior_p4']),
                float(meta['prior_p6']), float(meta['prior_pw']),
            )
        except (KeyError, ValueError, TypeError):
            self._prior = (1/6,) * 6

        # Phase 3: read 18 phase-prior _meta rows (prior_{pp,mid,death}_p*).
        # Missing rows → fall back to global π for that phase, so the
        # backend stays usable on pre-Phase-3 caches.
        _phase_short = {'powerplay': 'pp', 'middle': 'mid', 'death': 'death'}
        phase_priors: Dict[str, tuple] = {}
        for phase, short in _phase_short.items():
            try:
                phase_priors[phase] = (
                    float(meta[f'prior_{short}_p0']),
                    float(meta[f'prior_{short}_p1']),
                    float(meta[f'prior_{short}_p2']),
                    float(meta[f'prior_{short}_p4']),
                    float(meta[f'prior_{short}_p6']),
                    float(meta[f'prior_{short}_pw']),
                )
            except (KeyError, ValueError, TypeError):
                phase_priors[phase] = self._prior
        self._phase_priors = phase_priors

    def _norm_date(self, as_of_date) -> str:
        # Fast path: caller already passed a string (common in rehydration
        # and test-fixture code). Skip cache + strftime.
        if not isinstance(as_of_date, datetime):
            return as_of_date
        cache = self._date_norm_cache
        cached = cache.get(as_of_date)
        if cached is not None:
            return cached
        s = as_of_date.strftime('%Y-%m-%d')
        if len(cache) < self._DATE_NORM_CACHE_MAX:
            cache[as_of_date] = s
        return s

    def _resolve_date_id(self, as_of_date) -> int:
        """Largest date_id whose date ≤ as_of_date, or -1 if none."""
        target = self._norm_date(as_of_date)
        idx = bisect.bisect_right(self._date_strs, target)
        return idx - 1

    # --- core getters ----------------------------------------------------

    def get_batting_stats(self, player_id, as_of_date) -> Dict[str, float]:
        conn = self._ensure_conn()
        pid = self._player_id_map.get(str(player_id))
        if pid is None:
            return {'avg': 0.0, 'sr': 0.0}
        did = self._resolve_date_id(as_of_date)
        if did < 0:
            return {'avg': 0.0, 'sr': 0.0}
        row = conn.execute(_Q_BATTING, (pid, did)).fetchone()
        if row is None:
            return {'avg': 0.0, 'sr': 0.0}
        runs, balls, dismissals = row[0], row[1], row[2]
        if balls == 0:
            return {'avg': 0.0, 'sr': 0.0}
        return {
            'avg': runs / max(dismissals, 1),
            'sr': (runs / balls) * 100,
        }

    def get_batting_recent(self, player_id, as_of_date) -> Dict[str, float]:
        conn = self._ensure_conn()
        pid = self._player_id_map.get(str(player_id))
        if pid is None:
            return {'avg': 0.0, 'sr': 0.0}
        did = self._resolve_date_id(as_of_date)
        if did < 0:
            return {'avg': 0.0, 'sr': 0.0}
        row = conn.execute(_Q_BATTING, (pid, did)).fetchone()
        if row is None:
            return {'avg': 0.0, 'sr': 0.0}
        recent_runs, recent_balls, recent_dismissals = row[3], row[4], row[5]
        if recent_balls == 0:
            return {'avg': 0.0, 'sr': 0.0}
        return {
            'avg': recent_runs / max(recent_dismissals, 1),
            'sr': (recent_runs / recent_balls) * 100,
        }

    def get_bowling_stats(self, player_id, as_of_date) -> Dict[str, float]:
        conn = self._ensure_conn()
        pid = self._player_id_map.get(str(player_id))
        if pid is None:
            return {'avg': 0.0, 'econ': 0.0}
        did = self._resolve_date_id(as_of_date)
        if did < 0:
            return {'avg': 0.0, 'econ': 0.0}
        row = conn.execute(_Q_BOWLING, (pid, did)).fetchone()
        if row is None:
            return {'avg': 0.0, 'econ': 0.0}
        runs_given, balls_bowled, wickets = row[0], row[1], row[2]
        if balls_bowled == 0:
            return {'avg': 0.0, 'econ': 0.0}
        return {
            'avg': runs_given / max(wickets, 1),
            'econ': (runs_given / balls_bowled) * 6,
        }

    def get_bowling_recent(self, player_id, as_of_date) -> Dict[str, float]:
        conn = self._ensure_conn()
        pid = self._player_id_map.get(str(player_id))
        if pid is None:
            return {'avg': 0.0, 'econ': 0.0}
        did = self._resolve_date_id(as_of_date)
        if did < 0:
            return {'avg': 0.0, 'econ': 0.0}
        row = conn.execute(_Q_BOWLING, (pid, did)).fetchone()
        if row is None:
            return {'avg': 0.0, 'econ': 0.0}
        recent_runs_given, recent_balls_bowled, recent_wickets = row[3], row[4], row[5]
        if recent_balls_bowled == 0:
            return {'avg': 0.0, 'econ': 0.0}
        return {
            'avg': recent_runs_given / max(recent_wickets, 1),
            'econ': (recent_runs_given / recent_balls_bowled) * 6,
        }

    def get_h2h_stats(self, batter_id, bowler_id, as_of_date) -> Dict[str, float]:
        conn = self._ensure_conn()
        bat = self._player_id_map.get(str(batter_id))
        bowl = self._player_id_map.get(str(bowler_id))
        if bat is None or bowl is None:
            return {'avg': 0.0, 'sr': 0.0}
        did = self._resolve_date_id(as_of_date)
        if did < 0:
            return {'avg': 0.0, 'sr': 0.0}
        row = conn.execute(_Q_H2H, (bat, bowl, did)).fetchone()
        if row is None:
            return {'avg': 0.0, 'sr': 0.0}
        runs, balls, dismissals = row
        if balls == 0:
            return {'avg': 0.0, 'sr': 0.0}
        return {
            'avg': runs / max(dismissals, 1),
            'sr': (runs / balls) * 100,
        }

    def _venue_row(self, venue: str, as_of_date):
        conn = self._ensure_conn()
        vid = self._venue_id_map.get(venue)
        if vid is None:
            return None
        did = self._resolve_date_id(as_of_date)
        if did < 0:
            return None
        return conn.execute(_Q_VENUE, (vid, did)).fetchone()

    def get_venue_avg_score(self, venue: str, as_of_date) -> float:
        row = self._venue_row(venue, as_of_date)
        if row is None:
            return 0.0
        total_runs, innings_count, *_ = row
        if innings_count == 0:
            return 0.0
        return total_runs / innings_count

    def get_venue_profile(self, venue: str, as_of_date) -> Dict[str, float]:
        default = {
            'venue_boundary_pct': 0.0, 'venue_dot_pct': 0.0,
            'venue_wicket_rate': 0.0,
            'venue_powerplay_avg': 0.0, 'venue_death_avg': 0.0,
            'venue_first_innings_avg': 0.0, 'venue_chase_win_pct': 0.5,
        }
        row = self._venue_row(venue, as_of_date)
        if row is None:
            return default
        (total_runs, innings_count, total_balls, total_boundaries,
         total_dots, total_wickets, pp_runs, pp_balls,
         death_runs, death_balls, fi_sum, fi_count,
         matches_total, chase_wins) = row[:14]

        if total_balls == 0:
            # Legacy/empty venue — fall back to innings avg like
            # StatsProvider.get_venue_profile does at line 321-325.
            avg_score = total_runs / innings_count if innings_count > 0 else 0.0
            return {**default, 'venue_first_innings_avg': avg_score}

        return {
            'venue_boundary_pct': total_boundaries / total_balls,
            'venue_dot_pct': total_dots / total_balls,
            'venue_wicket_rate': total_wickets / total_balls,
            'venue_powerplay_avg': (pp_runs / pp_balls * 36) if pp_balls > 0 else 0.0,
            'venue_death_avg': (death_runs / death_balls * 30) if death_balls > 0 else 0.0,
            'venue_first_innings_avg': (fi_sum / fi_count) if fi_count > 0 else 0.0,
            'venue_chase_win_pct': (chase_wins / matches_total) if matches_total > 0 else 0.5,
        }

    def _typed_batting(self, player_id, bowl_type: int, as_of_date):
        conn = self._ensure_conn()
        pid = self._player_id_map.get(str(player_id))
        if pid is None:
            return None
        did = self._resolve_date_id(as_of_date)
        if did < 0:
            return None
        return conn.execute(_Q_BATTING_VS_TYPE, (pid, bowl_type, did)).fetchone()

    def get_batting_vs_type_stats(self, batter_id, as_of_date) -> Dict[str, float]:
        pace = self._typed_batting(batter_id, 0, as_of_date)
        spin = self._typed_batting(batter_id, 1, as_of_date)

        def _pair(row):
            if row is None:
                return 0.0, 0.0
            runs, balls, dismissals = row[0], row[1], row[2]
            if balls == 0:
                return 0.0, 0.0
            return runs / max(dismissals, 1), (runs / balls) * 100

        avg_p, sr_p = _pair(pace)
        avg_s, sr_s = _pair(spin)
        return {
            'avg_vs_pace': avg_p, 'sr_vs_pace': sr_p,
            'avg_vs_spin': avg_s, 'sr_vs_spin': sr_s,
        }

    def _hand_bowling(self, player_id, bat_hand: int, as_of_date):
        conn = self._ensure_conn()
        pid = self._player_id_map.get(str(player_id))
        if pid is None:
            return None
        did = self._resolve_date_id(as_of_date)
        if did < 0:
            return None
        return conn.execute(_Q_BOWLING_VS_HAND, (pid, bat_hand, did)).fetchone()

    def get_bowling_vs_hand_stats(self, bowler_id, as_of_date) -> Dict[str, float]:
        lhb = self._hand_bowling(bowler_id, 0, as_of_date)
        rhb = self._hand_bowling(bowler_id, 1, as_of_date)

        def _pair(row):
            if row is None:
                return 0.0, 0.0
            runs_given, balls_bowled, wickets = row[0], row[1], row[2]
            if balls_bowled == 0:
                return 0.0, 0.0
            return runs_given / max(wickets, 1), (runs_given / balls_bowled) * 6

        avg_l, econ_l = _pair(lhb)
        avg_r, econ_r = _pair(rhb)
        return {
            'avg_vs_lhb': avg_l, 'econ_vs_lhb': econ_l,
            'avg_vs_rhb': avg_r, 'econ_vs_rhb': econ_r,
        }

    def get_batting_elo(self, player_id, as_of_date) -> float:
        conn = self._ensure_conn()
        pid = self._player_id_map.get(str(player_id))
        if pid is None:
            return 1500.0
        did = self._resolve_date_id(as_of_date)
        if did < 0:
            return 1500.0
        row = conn.execute(_Q_BATTING_ELO, (pid, did)).fetchone()
        return row[0] if row is not None else 1500.0

    def get_bowling_elo(self, player_id, as_of_date) -> float:
        conn = self._ensure_conn()
        pid = self._player_id_map.get(str(player_id))
        if pid is None:
            return 1500.0
        did = self._resolve_date_id(as_of_date)
        if did < 0:
            return 1500.0
        row = conn.execute(_Q_BOWLING_ELO, (pid, did)).fetchone()
        return row[0] if row is not None else 1500.0

    # --- team aggregates: delegate. StatsProviderCache memoizes above. --

    def get_team_batting_elo(self, player_ids, as_of_date) -> float:
        return sum(self.get_batting_elo(pid, as_of_date) for pid in player_ids)

    def get_team_bowling_elo(self, player_ids, as_of_date) -> float:
        return sum(self.get_bowling_elo(pid, as_of_date) for pid in player_ids)

    def get_team_batting_strength(self, player_ids, as_of_date) -> Dict[str, float]:
        avgs, srs = [], []
        for pid in player_ids:
            s = self.get_batting_stats(pid, as_of_date)
            if s['avg'] > 0:
                avgs.append(s['avg'])
                srs.append(s['sr'])
        return {
            'team_batting_avg': sum(avgs) / len(avgs) if avgs else 0.0,
            'team_batting_sr': sum(srs) / len(srs) if srs else 0.0,
        }

    def get_team_bowling_strength(self, player_ids, as_of_date) -> Dict[str, float]:
        avgs, econs = [], []
        for pid in player_ids:
            s = self.get_bowling_stats(pid, as_of_date)
            if s['avg'] > 0:
                avgs.append(s['avg'])
                econs.append(s['econ'])
        return {
            'team_bowling_avg': sum(avgs) / len(avgs) if avgs else 0.0,
            'team_bowling_econ': sum(econs) / len(econs) if econs else 0.0,
        }

    # --- raw counter access (validator parity, per Phase 2 plan) --------

    def _get_raw_batting(self, player_id, as_of_date) -> Optional[Dict[str, int]]:
        conn = self._ensure_conn()
        pid = self._player_id_map.get(str(player_id))
        if pid is None:
            return None
        did = self._resolve_date_id(as_of_date)
        if did < 0:
            return None
        row = conn.execute(_Q_BATTING, (pid, did)).fetchone()
        if row is None:
            return None
        return {
            'runs': row[0], 'balls': row[1], 'dismissals': row[2],
            'recent_runs': row[3], 'recent_balls': row[4],
            'recent_dismissals': row[5],
            'c0': row[6], 'c1': row[7], 'c2': row[8],
            'c4': row[9], 'c6': row[10], 'cw': row[11],
        }

    def _get_raw_bowling(self, player_id, as_of_date) -> Optional[Dict[str, int]]:
        conn = self._ensure_conn()
        pid = self._player_id_map.get(str(player_id))
        if pid is None:
            return None
        did = self._resolve_date_id(as_of_date)
        if did < 0:
            return None
        row = conn.execute(_Q_BOWLING, (pid, did)).fetchone()
        if row is None:
            return None
        return {
            'runs_given': row[0], 'balls_bowled': row[1], 'wickets': row[2],
            'recent_runs_given': row[3], 'recent_balls_bowled': row[4],
            'recent_wickets': row[5],
            'c0': row[6], 'c1': row[7], 'c2': row[8],
            'c4': row[9], 'c6': row[10], 'cw': row[11],
        }

    def _get_raw_h2h(self, batter_id, bowler_id, as_of_date) -> Optional[Dict[str, int]]:
        conn = self._ensure_conn()
        bat = self._player_id_map.get(str(batter_id))
        bowl = self._player_id_map.get(str(bowler_id))
        if bat is None or bowl is None:
            return None
        did = self._resolve_date_id(as_of_date)
        if did < 0:
            return None
        row = conn.execute(_Q_H2H, (bat, bowl, did)).fetchone()
        if row is None:
            return None
        return {'runs': row[0], 'balls': row[1], 'dismissals': row[2]}

    # --- schema v4 outcome-distribution getters -------------------------
    # Empirical Bayes shrinkage toward the global corpus prior π:
    #     p̂_c = (n_c + k · π_c) / (N + k),  N = Σ n_c
    # N → 0  ⇒ p̂ → π   (new player / unseen venue falls back to prior)
    # N → ∞  ⇒ p̂ → n/N (rich history dominates)
    # k is the "prior sample size"; larger k = more shrinkage. Per-
    # hierarchy defaults are on the caller side.

    @staticmethod
    def _shrink(counts, prior, k: float) -> tuple:
        """Dirichlet-posterior-mean shrinkage of a 6-count vector toward a
        6-prob prior. Returns a 6-tuple summing to 1.0 (within fp eps)."""
        n = counts[0] + counts[1] + counts[2] + counts[3] + counts[4] + counts[5]
        denom = n + k
        # denom > 0 always, since k > 0 is required by the caller.
        return (
            (counts[0] + k * prior[0]) / denom,
            (counts[1] + k * prior[1]) / denom,
            (counts[2] + k * prior[2]) / denom,
            (counts[3] + k * prior[3]) / denom,
            (counts[4] + k * prior[4]) / denom,
            (counts[5] + k * prior[5]) / denom,
        )

    def _batting_counts(self, player_id, as_of_date) -> tuple:
        """Return the 6-tuple of outcome counts for a batter at as-of-date
        (or (0,0,0,0,0,0) if the player has no history). Zero tuple drives
        full fallback to the prior in _shrink."""
        conn = self._ensure_conn()
        pid = self._player_id_map.get(str(player_id))
        if pid is None:
            return (0, 0, 0, 0, 0, 0)
        did = self._resolve_date_id(as_of_date)
        if did < 0:
            return (0, 0, 0, 0, 0, 0)
        row = conn.execute(_Q_BATTING, (pid, did)).fetchone()
        if row is None:
            return (0, 0, 0, 0, 0, 0)
        return (row[6], row[7], row[8], row[9], row[10], row[11])

    def _bowling_counts(self, player_id, as_of_date) -> tuple:
        conn = self._ensure_conn()
        pid = self._player_id_map.get(str(player_id))
        if pid is None:
            return (0, 0, 0, 0, 0, 0)
        did = self._resolve_date_id(as_of_date)
        if did < 0:
            return (0, 0, 0, 0, 0, 0)
        row = conn.execute(_Q_BOWLING, (pid, did)).fetchone()
        if row is None:
            return (0, 0, 0, 0, 0, 0)
        return (row[6], row[7], row[8], row[9], row[10], row[11])

    def _batting_vs_type_counts(self, player_id, bowl_type: int, as_of_date) -> tuple:
        row = self._typed_batting(player_id, bowl_type, as_of_date)
        if row is None:
            return (0, 0, 0, 0, 0, 0)
        # row = (runs, balls, dismissals, c0, c1, c2, c4, c6, cw)
        return (row[3], row[4], row[5], row[6], row[7], row[8])

    def _bowling_vs_hand_counts(self, player_id, bat_hand: int, as_of_date) -> tuple:
        row = self._hand_bowling(player_id, bat_hand, as_of_date)
        if row is None:
            return (0, 0, 0, 0, 0, 0)
        return (row[3], row[4], row[5], row[6], row[7], row[8])

    def _venue_counts(self, venue: str, as_of_date) -> tuple:
        row = self._venue_row(venue, as_of_date)
        if row is None:
            return (0, 0, 0, 0, 0, 0)
        # row indices 14..19 are c0..cw after the venue row widening.
        return (row[14], row[15], row[16], row[17], row[18], row[19])

    def get_batter_outcome_dist(
        self, player_id, as_of_date, k: float = 30.0,
    ) -> Dict[str, float]:
        p = self._shrink(self._batting_counts(player_id, as_of_date),
                         self._prior, k)
        return {
            'batter_p0': p[0], 'batter_p1': p[1], 'batter_p2': p[2],
            'batter_p4': p[3], 'batter_p6': p[4], 'batter_pw': p[5],
        }

    def get_bowler_outcome_dist(
        self, player_id, as_of_date, k: float = 30.0,
    ) -> Dict[str, float]:
        p = self._shrink(self._bowling_counts(player_id, as_of_date),
                         self._prior, k)
        return {
            'bowler_p0': p[0], 'bowler_p1': p[1], 'bowler_p2': p[2],
            'bowler_p4': p[3], 'bowler_p6': p[4], 'bowler_pw': p[5],
        }

    def get_batter_vs_type_outcome_dist(
        self, player_id, as_of_date, k: float = 30.0,
        hierarchical: bool = True,
    ) -> Dict[str, float]:
        """Phase 5: when `hierarchical=True` (default), the vs-pace and
        vs-spin cells shrink toward the batter's overall distribution
        (which itself is shrunk toward π) instead of directly toward π.
        For batters with sparse vs-type data but rich overall data, this
        falls back to the player's own profile rather than the global
        prior. `hierarchical=False` reproduces the legacy flat-shrink
        behavior."""
        if hierarchical:
            parent = self._shrink(self._batting_counts(player_id, as_of_date),
                                  self._prior, k)
        else:
            parent = self._prior
        pp = self._shrink(self._batting_vs_type_counts(player_id, 0, as_of_date),
                          parent, k)
        ps = self._shrink(self._batting_vs_type_counts(player_id, 1, as_of_date),
                          parent, k)
        return {
            'batter_p0_vs_pace': pp[0], 'batter_p1_vs_pace': pp[1],
            'batter_p2_vs_pace': pp[2], 'batter_p4_vs_pace': pp[3],
            'batter_p6_vs_pace': pp[4], 'batter_pw_vs_pace': pp[5],
            'batter_p0_vs_spin': ps[0], 'batter_p1_vs_spin': ps[1],
            'batter_p2_vs_spin': ps[2], 'batter_p4_vs_spin': ps[3],
            'batter_p6_vs_spin': ps[4], 'batter_pw_vs_spin': ps[5],
        }

    def get_bowler_vs_hand_outcome_dist(
        self, player_id, as_of_date, k: float = 30.0,
        hierarchical: bool = True,
    ) -> Dict[str, float]:
        """Phase 5: when `hierarchical=True` (default), the vs-LHB and
        vs-RHB cells shrink toward the bowler's overall distribution.
        See `get_batter_vs_type_outcome_dist` for rationale."""
        if hierarchical:
            parent = self._shrink(self._bowling_counts(player_id, as_of_date),
                                  self._prior, k)
        else:
            parent = self._prior
        pl = self._shrink(self._bowling_vs_hand_counts(player_id, 0, as_of_date),
                          parent, k)
        pr = self._shrink(self._bowling_vs_hand_counts(player_id, 1, as_of_date),
                          parent, k)
        return {
            'bowler_p0_vs_lhb': pl[0], 'bowler_p1_vs_lhb': pl[1],
            'bowler_p2_vs_lhb': pl[2], 'bowler_p4_vs_lhb': pl[3],
            'bowler_p6_vs_lhb': pl[4], 'bowler_pw_vs_lhb': pl[5],
            'bowler_p0_vs_rhb': pr[0], 'bowler_p1_vs_rhb': pr[1],
            'bowler_p2_vs_rhb': pr[2], 'bowler_p4_vs_rhb': pr[3],
            'bowler_p6_vs_rhb': pr[4], 'bowler_pw_vs_rhb': pr[5],
        }

    def get_venue_outcome_dist(
        self, venue: str, as_of_date, k: float = 200.0,
    ) -> Dict[str, float]:
        p = self._shrink(self._venue_counts(venue, as_of_date),
                         self._prior, k)
        return {
            'venue_p0': p[0], 'venue_p1': p[1], 'venue_p2': p[2],
            'venue_p4': p[3], 'venue_p6': p[4], 'venue_pw': p[5],
        }

    def get_phase_outcome_dist(self, balls_bowled: int) -> Dict[str, float]:
        """Phase 3 phase prior. Returns 6 phase_p{0,1,2,4,6,w} features
        based on the pre-ball phase (PP / mid / death). No shrinkage —
        these are global constants over millions of balls. The
        as_of_date arg is intentionally absent: phase priors are
        match-date-independent."""
        if self._phase_priors is None:
            self._ensure_conn()
        if balls_bowled < 36:
            phase = 'powerplay'
        elif balls_bowled < 96:
            phase = 'middle'
        else:
            phase = 'death'
        p = self._phase_priors.get(phase, self._prior)
        return {
            'phase_p0': p[0], 'phase_p1': p[1], 'phase_p2': p[2],
            'phase_p4': p[3], 'phase_p6': p[4], 'phase_pw': p[5],
        }

    # --- schema v3 match-log getters ------------------------------------
    # Result ordered newest-first (date_id DESC, intra_date_idx DESC).
    # Callers that need deque append order (oldest-first) should reverse.
    # Used by tracker_rehydration.py to rebuild PlayerStatsTracker's
    # recent_batting / recent_bowling deques bit-exactly on same-day
    # secondaries.

    def _strict_before_bound(self, as_of_date) -> int:
        """Return the smallest date_id whose date is >= as_of_date.
        Rows with date_id strictly less than this bound are matches
        played BEFORE as_of_date. Used for strict-pre-date queries.

        Returns 0 when as_of_date is before all snapshots (→ no rows).
        Returns len(_date_strs) when as_of_date is after all snapshots
        (→ all rows). `_resolve_date_id` is not reusable here because
        it returns the LARGEST did <= target, which for an exact-date
        match gives target's own date_id — we need the one AFTER.
        """
        target = self._norm_date(as_of_date)
        return bisect.bisect_left(self._date_strs, target)

    def get_batting_match_log_recent(
        self, player_id, as_of_date, limit: int = 5,
    ) -> list:
        conn = self._ensure_conn()
        pid = self._player_id_map.get(str(player_id))
        if pid is None:
            return []
        bound = self._strict_before_bound(as_of_date)
        if bound <= 0:
            return []
        rows = conn.execute(
            _Q_BATTING_LOG_RECENT, (pid, bound, limit)
        ).fetchall()
        return [
            {"runs": r[0], "balls": r[1], "dismissals": r[2]}
            for r in rows
        ]

    def get_bowling_match_log_recent(
        self, player_id, as_of_date, limit: int = 5,
    ) -> list:
        conn = self._ensure_conn()
        pid = self._player_id_map.get(str(player_id))
        if pid is None:
            return []
        bound = self._strict_before_bound(as_of_date)
        if bound <= 0:
            return []
        rows = conn.execute(
            _Q_BOWLING_LOG_RECENT, (pid, bound, limit)
        ).fetchall()
        return [
            {"runs_given": r[0], "balls_bowled": r[1], "wickets": r[2]}
            for r in rows
        ]

    def get_all_stats(self, batter_id, bowler_id, as_of_date) -> Dict[str, float]:
        b = self.get_batting_stats(batter_id, as_of_date)
        bw = self.get_bowling_stats(bowler_id, as_of_date)
        h = self.get_h2h_stats(batter_id, bowler_id, as_of_date)
        return {
            'batsman_avg': b['avg'], 'batsman_sr': b['sr'],
            'bowler_avg': bw['avg'], 'bowler_econ': bw['econ'],
            'h2h_avg': h['avg'], 'h2h_sr': h['sr'],
        }

    # --- meta ------------------------------------------------------------

    def get_meta(self) -> Dict[str, str]:
        conn = self._ensure_conn()
        return {k: v for k, v in conn.execute("SELECT key, value FROM _meta")}


def open_backend(db_path) -> _SQLiteBackend:
    """Convenience constructor that also asserts the file exists."""
    p = Path(db_path)
    if not p.exists():
        raise FileNotFoundError(f"SQLite stats cache not found: {p}")
    return _SQLiteBackend(p)
