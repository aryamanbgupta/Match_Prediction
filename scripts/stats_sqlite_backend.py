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


SCHEMA_VERSION = 2


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
    PRIMARY KEY (player_id, bowl_type, date_id)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS bowling_vs_hand (
    player_id INTEGER NOT NULL,
    date_id INTEGER NOT NULL,
    bat_hand INTEGER NOT NULL,  -- 0=left, 1=right
    runs_given INTEGER NOT NULL,
    balls_bowled INTEGER NOT NULL,
    wickets INTEGER NOT NULL,
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
       recent_runs, recent_balls, recent_dismissals
FROM batting
WHERE player_id = ? AND date_id <= ?
ORDER BY date_id DESC LIMIT 1
"""
_Q_BOWLING = """
SELECT runs_given, balls_bowled, wickets,
       recent_runs_given, recent_balls_bowled, recent_wickets
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
SELECT runs, balls, dismissals FROM batting_vs_type
WHERE player_id = ? AND bowl_type = ? AND date_id <= ?
ORDER BY date_id DESC LIMIT 1
"""
_Q_BOWLING_VS_HAND = """
SELECT runs_given, balls_bowled, wickets FROM bowling_vs_hand
WHERE player_id = ? AND bat_hand = ? AND date_id <= ?
ORDER BY date_id DESC LIMIT 1
"""
_Q_VENUE = """
SELECT total_runs, innings_count, total_balls, total_boundaries,
       total_dots, total_wickets, powerplay_runs, powerplay_balls,
       death_runs, death_balls, fi_totals_sum, fi_totals_count,
       matches_total, chase_wins
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


# Every getter is one of these shapes. Exported so benchmarks can
# iterate the full set for EXPLAIN QUERY PLAN assertions.
QUERY_PLAN_CASES = [
    ("batting",         _Q_BATTING,         (1, 1)),
    ("bowling",         _Q_BOWLING,         (1, 1)),
    ("h2h",             _Q_H2H,             (1, 1, 1)),
    ("batting_vs_type", _Q_BATTING_VS_TYPE, (1, 0, 1)),
    ("bowling_vs_hand", _Q_BOWLING_VS_HAND, (1, 0, 1)),
    ("venue",           _Q_VENUE,           (1, 1)),
    ("batting_elo",     _Q_BATTING_ELO,     (1, 1)),
    ("bowling_elo",     _Q_BOWLING_ELO,     (1, 1)),
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

    # --- pickle: strip PID-bound connection so workers re-open ----------

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_conn'] = None
        state['_conn_pid'] = None
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

    @staticmethod
    def _norm_date(as_of_date) -> str:
        if isinstance(as_of_date, datetime):
            return as_of_date.strftime('%Y-%m-%d')
        return as_of_date

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
         matches_total, chase_wins) = row

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
            runs, balls, dismissals = row
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
            runs_given, balls_bowled, wickets = row
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
