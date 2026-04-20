"""
Player Stats Provider for Match Simulations (Chunked Format)

This module provides temporal access to player statistics using a chunked,
lazy-loading cache system that avoids loading 7.8GB into memory.

Architecture:
- Chunked storage: 69 files (~110MB each) instead of single 7.8GB file
- Lazy loading: Load chunks on-demand with LRU eviction
- Binary search: O(log n) temporal lookups across 3,442 date snapshots
- Memory efficient: ~300-550MB in memory (5 chunks) vs 7.8GB full cache
- Returns same format as training to ensure consistency

Performance:
- Initialization: ~1-2 seconds (metadata only)
- Query speed: <0.01ms (after chunk cached)
- Cache hit rate: ~95%+ for sequential dates

Usage:
    provider = StatsProvider('models')  # Pass directory, not file
    stats = provider.get_batting_stats('player_123', '2024-06-01')
    # Returns: {'avg': 31.4, 'sr': 140.2}
"""

import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple
import bisect


class _ChunkedBackend:
    """Chunked pickle backend (legacy).

    Kept behind the `StatsProvider` facade below as the fallback when no
    SQLite cache file exists. API is identical to `_SQLiteBackend` so the
    facade can dispatch to either transparently.

    DESIGN DECISION: Chunked lazy loading with LRU cache
    REASONING: 7.8GB full cache too large for memory, chunking allows
               on-demand loading while maintaining fast O(log n) lookups.
               LRU cache keeps frequently-used chunks in memory.
    """

    def __init__(self, cache_dir: str = 'models', max_cached_chunks: int = 5, version: str = 'v3'):
        """
        Load the player stats cache from chunked format with lazy loading.

        Args:
            cache_dir: Directory containing cache chunk files
            max_cached_chunks: Maximum number of chunks to keep in memory (LRU cache)
            version: Cache version to load ('v2' or 'v3'). v3 includes type-based stats.
        """
        from pathlib import Path
        from collections import OrderedDict

        self.cache_dir = Path(cache_dir)
        self.version = version

        # Select metadata file based on version
        if version == 'v3':
            metadata_path = self.cache_dir / 'player_stats_cache_v3_metadata.pkl'
        else:
            metadata_path = self.cache_dir / 'player_stats_cache_metadata.pkl'

        print(f"Loading player stats cache ({version}) from {cache_dir}...")

        # Load metadata
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)

        print(f"  Found {self.metadata['num_chunks']} cache chunks")

        # Build date-to-chunk-index mapping for fast lookup
        self.date_to_chunk_idx = {}
        all_dates = []

        for chunk_idx, chunk_info in enumerate(self.metadata['chunks']):
            for date in chunk_info['dates']:
                self.date_to_chunk_idx[date] = chunk_idx
                all_dates.append(date)

        # Pre-sort all dates for binary search
        self.dates = sorted(all_dates)

        # Initialize LRU cache for chunks
        self.max_cached_chunks = max_cached_chunks
        self.chunk_cache = OrderedDict()  # {chunk_idx: chunk_data}

        print(f"  ✓ Initialized lazy loading for {len(self.dates):,} date snapshots")
        print(f"  Date range: {self.dates[0]} to {self.dates[-1]}")
        print(f"  Players: {self.metadata['num_players_batting']:,} batters, "
              f"{self.metadata['num_players_bowling']:,} bowlers")
        print(f"  Cache size: {max_cached_chunks} chunks (~{max_cached_chunks * 110}MB max)")

    def _load_chunk(self, chunk_idx: int) -> Dict:
        """
        Load a chunk from disk and add to LRU cache.

        Args:
            chunk_idx: Index of chunk to load

        Returns:
            Chunk data dict
        """
        # Check if already in cache
        if chunk_idx in self.chunk_cache:
            # Move to end (mark as recently used)
            self.chunk_cache.move_to_end(chunk_idx)
            return self.chunk_cache[chunk_idx]

        # Load from disk
        chunk_info = self.metadata['chunks'][chunk_idx]
        chunk_path = self.cache_dir / chunk_info['file']

        with open(chunk_path, 'rb') as f:
            chunk_data = pickle.load(f)

        # Add to cache
        self.chunk_cache[chunk_idx] = chunk_data

        # Evict oldest if cache is full (LRU)
        if len(self.chunk_cache) > self.max_cached_chunks:
            # Remove least recently used (first item in OrderedDict)
            evicted_idx, _ = self.chunk_cache.popitem(last=False)

        return chunk_data

    def _get_snapshot_for_date(self, target_date: str) -> Optional[Dict]:
        """
        Find the most recent snapshot before or on target_date.

        Uses binary search for O(log n) performance and lazy loading.

        Args:
            target_date: Date string in 'YYYY-MM-DD' format

        Returns:
            Snapshot dict or None if no history exists
        """
        # Binary search to find rightmost date <= target_date
        idx = bisect.bisect_right(self.dates, target_date)

        if idx == 0:
            # No snapshot exists before this date
            return None

        # Get the snapshot date
        snapshot_date = self.dates[idx - 1]

        # Find which chunk contains this date
        chunk_idx = self.date_to_chunk_idx[snapshot_date]

        # Load chunk (from cache or disk)
        chunk_data = self._load_chunk(chunk_idx)

        return chunk_data[snapshot_date]

    # -- raw counter access (for validators / parity tests) ---------------
    # These return the same int dicts the derivation formulas above start
    # from. Exposed so validators don't need to reach into the snapshot
    # dict via `_get_snapshot_for_date`, which isn't portable across
    # backends (the SQLite backend has no materialised snapshots).

    def _get_raw_batting(self, player_id: str, as_of_date) -> Optional[Dict[str, int]]:
        if isinstance(as_of_date, datetime):
            as_of_date = as_of_date.strftime('%Y-%m-%d')
        snap = self._get_snapshot_for_date(as_of_date)
        if snap is None:
            return None
        return snap['batting'].get(player_id)

    def _get_raw_bowling(self, player_id: str, as_of_date) -> Optional[Dict[str, int]]:
        if isinstance(as_of_date, datetime):
            as_of_date = as_of_date.strftime('%Y-%m-%d')
        snap = self._get_snapshot_for_date(as_of_date)
        if snap is None:
            return None
        return snap['bowling'].get(player_id)

    def _get_raw_h2h(self, batter_id: str, bowler_id: str, as_of_date) -> Optional[Dict[str, int]]:
        if isinstance(as_of_date, datetime):
            as_of_date = as_of_date.strftime('%Y-%m-%d')
        snap = self._get_snapshot_for_date(as_of_date)
        if snap is None:
            return None
        return snap.get('h2h', {}).get((batter_id, bowler_id))

    def get_batting_stats(self, player_id: str, as_of_date: str) -> Dict[str, float]:
        """
        Get batting statistics for a player as of a specific date.

        Matches the interface of PlayerStatsTracker.get_batting_features()

        Args:
            player_id: Player identifier
            as_of_date: Date in 'YYYY-MM-DD' format or datetime object

        Returns:
            Dict with keys: 'avg', 'sr'
            Returns zeros if player unknown or no history
        """
        # Handle datetime objects
        if isinstance(as_of_date, datetime):
            as_of_date = as_of_date.strftime('%Y-%m-%d')

        snapshot = self._get_snapshot_for_date(as_of_date)

        if snapshot is None:
            # No history exists yet
            return {'avg': 0.0, 'sr': 0.0}

        batting_stats = snapshot['batting'].get(player_id, {
            'runs': 0, 'balls': 0, 'dismissals': 0
        })

        # Replicate exact formula from PlayerStatsTracker.get_batting_features()
        if batting_stats['balls'] == 0:
            return {'avg': 0.0, 'sr': 0.0}

        avg = batting_stats['runs'] / max(batting_stats['dismissals'], 1)
        sr = (batting_stats['runs'] / batting_stats['balls']) * 100

        return {'avg': avg, 'sr': sr}

    def get_bowling_stats(self, player_id: str, as_of_date: str) -> Dict[str, float]:
        """
        Get bowling statistics for a player as of a specific date.

        Matches the interface of PlayerStatsTracker.get_bowling_features()

        Args:
            player_id: Player identifier
            as_of_date: Date in 'YYYY-MM-DD' format or datetime object

        Returns:
            Dict with keys: 'avg', 'econ'
            Returns zeros if player unknown or no history
        """
        # Handle datetime objects
        if isinstance(as_of_date, datetime):
            as_of_date = as_of_date.strftime('%Y-%m-%d')

        snapshot = self._get_snapshot_for_date(as_of_date)

        if snapshot is None:
            return {'avg': 0.0, 'econ': 0.0}

        bowling_stats = snapshot['bowling'].get(player_id, {
            'runs_given': 0, 'balls_bowled': 0, 'wickets': 0
        })

        # Replicate exact formula from PlayerStatsTracker.get_bowling_features()
        if bowling_stats['balls_bowled'] == 0:
            return {'avg': 0.0, 'econ': 0.0}

        avg = bowling_stats['runs_given'] / max(bowling_stats['wickets'], 1)
        econ = (bowling_stats['runs_given'] / bowling_stats['balls_bowled']) * 6

        return {'avg': avg, 'econ': econ}

    def get_h2h_stats(self, batter_id: str, bowler_id: str, as_of_date: str) -> Dict[str, float]:
        """
        Get head-to-head matchup statistics as of a specific date.

        Matches the interface of PlayerStatsTracker.get_h2h_features()

        Args:
            batter_id: Batter player identifier
            bowler_id: Bowler player identifier
            as_of_date: Date in 'YYYY-MM-DD' format or datetime object

        Returns:
            Dict with keys: 'avg', 'sr'
            Returns zeros if matchup unknown or no history
        """
        # Handle datetime objects
        if isinstance(as_of_date, datetime):
            as_of_date = as_of_date.strftime('%Y-%m-%d')

        snapshot = self._get_snapshot_for_date(as_of_date)

        if snapshot is None:
            return {'avg': 0.0, 'sr': 0.0}

        h2h_stats = snapshot['h2h'].get((batter_id, bowler_id), {
            'runs': 0, 'balls': 0, 'dismissals': 0
        })

        # Replicate exact formula from PlayerStatsTracker.get_h2h_features()
        if h2h_stats['balls'] == 0:
            return {'avg': 0.0, 'sr': 0.0}

        avg = h2h_stats['runs'] / max(h2h_stats['dismissals'], 1)
        sr = (h2h_stats['runs'] / h2h_stats['balls']) * 100

        return {'avg': avg, 'sr': sr}

    def get_venue_avg_score(self, venue: str, as_of_date: str) -> float:
        """
        Get historical average score at venue as of a specific date.

        Args:
            venue: Venue name
            as_of_date: Date in 'YYYY-MM-DD' format or datetime object

        Returns:
            Average score at venue, or 0 if no history
        """
        # Handle datetime objects
        if isinstance(as_of_date, datetime):
            as_of_date = as_of_date.strftime('%Y-%m-%d')

        snapshot = self._get_snapshot_for_date(as_of_date)

        if snapshot is None:
            return 0.0

        # Check if venue stats exist in snapshot
        venue_stats = snapshot.get('venue', {}).get(venue, {
            'total_runs': 0, 'innings_count': 0
        })

        if venue_stats['innings_count'] == 0:
            return 0.0

        return venue_stats['total_runs'] / venue_stats['innings_count']

    def get_venue_profile(self, venue: str, as_of_date: str) -> Dict[str, float]:
        """
        Get rich venue profile as of a specific date.
        Gracefully handles old cache format (returns defaults for missing fields).

        Returns:
            Dict with keys matching venue_profile feature group.
        """
        if isinstance(as_of_date, datetime):
            as_of_date = as_of_date.strftime('%Y-%m-%d')

        default = {
            'venue_boundary_pct': 0.0, 'venue_dot_pct': 0.0, 'venue_wicket_rate': 0.0,
            'venue_powerplay_avg': 0.0, 'venue_death_avg': 0.0,
            'venue_first_innings_avg': 0.0, 'venue_chase_win_pct': 0.5,
        }

        snapshot = self._get_snapshot_for_date(as_of_date)
        if snapshot is None:
            return default

        venue_stats = snapshot.get('venue', {}).get(venue, {})
        total_balls = venue_stats.get('total_balls', 0)

        if total_balls == 0:
            # Old cache format or no data — fall back to basic avg
            innings_count = venue_stats.get('innings_count', 0)
            total_runs = venue_stats.get('total_runs', 0)
            avg_score = total_runs / innings_count if innings_count > 0 else 0.0
            return {**default, 'venue_first_innings_avg': avg_score}

        pp_balls = venue_stats.get('powerplay_balls', 0)
        death_balls = venue_stats.get('death_balls', 0)
        fi_totals = venue_stats.get('first_innings_totals', [])
        matches_total = venue_stats.get('matches_total', 0)

        return {
            'venue_boundary_pct': venue_stats.get('total_boundaries', 0) / total_balls,
            'venue_dot_pct': venue_stats.get('total_dots', 0) / total_balls,
            'venue_wicket_rate': venue_stats.get('total_wickets', 0) / total_balls,
            'venue_powerplay_avg': (venue_stats.get('powerplay_runs', 0) / pp_balls * 36) if pp_balls > 0 else 0.0,
            'venue_death_avg': (venue_stats.get('death_runs', 0) / death_balls * 30) if death_balls > 0 else 0.0,
            'venue_first_innings_avg': sum(fi_totals) / len(fi_totals) if fi_totals else 0.0,
            'venue_chase_win_pct': venue_stats.get('chase_wins', 0) / matches_total if matches_total > 0 else 0.5,
        }

    def get_batting_vs_type_stats(self, batter_id: str, as_of_date: str) -> Dict[str, float]:
        """
        Get batter's stats against pace and spin bowlers as of a specific date.

        NEW: Type-based batting stats for Tier 3 features.

        Args:
            batter_id: Batter player identifier
            as_of_date: Date in 'YYYY-MM-DD' format or datetime object

        Returns:
            Dict with keys: 'avg_vs_pace', 'sr_vs_pace', 'avg_vs_spin', 'sr_vs_spin'
            Returns zeros if no history exists
        """
        # Handle datetime objects
        if isinstance(as_of_date, datetime):
            as_of_date = as_of_date.strftime('%Y-%m-%d')

        snapshot = self._get_snapshot_for_date(as_of_date)

        if snapshot is None:
            return {
                'avg_vs_pace': 0.0, 'sr_vs_pace': 0.0,
                'avg_vs_spin': 0.0, 'sr_vs_spin': 0.0,
            }

        # Check if type-based stats exist in snapshot
        batting_vs_type = snapshot.get('batting_vs_type', {}).get(batter_id, {
            'pace': {'runs': 0, 'balls': 0, 'dismissals': 0},
            'spin': {'runs': 0, 'balls': 0, 'dismissals': 0},
        })

        # vs Pace
        pace_stats = batting_vs_type.get('pace', {'runs': 0, 'balls': 0, 'dismissals': 0})
        if pace_stats['balls'] == 0:
            avg_vs_pace, sr_vs_pace = 0.0, 0.0
        else:
            avg_vs_pace = pace_stats['runs'] / max(pace_stats['dismissals'], 1)
            sr_vs_pace = (pace_stats['runs'] / pace_stats['balls']) * 100

        # vs Spin
        spin_stats = batting_vs_type.get('spin', {'runs': 0, 'balls': 0, 'dismissals': 0})
        if spin_stats['balls'] == 0:
            avg_vs_spin, sr_vs_spin = 0.0, 0.0
        else:
            avg_vs_spin = spin_stats['runs'] / max(spin_stats['dismissals'], 1)
            sr_vs_spin = (spin_stats['runs'] / spin_stats['balls']) * 100

        return {
            'avg_vs_pace': avg_vs_pace,
            'sr_vs_pace': sr_vs_pace,
            'avg_vs_spin': avg_vs_spin,
            'sr_vs_spin': sr_vs_spin,
        }

    def get_bowling_vs_hand_stats(self, bowler_id: str, as_of_date: str) -> Dict[str, float]:
        """
        Get bowler's stats against left and right hand batters as of a specific date.

        NEW: Hand-based bowling stats for Tier 3 features.

        Args:
            bowler_id: Bowler player identifier
            as_of_date: Date in 'YYYY-MM-DD' format or datetime object

        Returns:
            Dict with keys: 'avg_vs_lhb', 'econ_vs_lhb', 'avg_vs_rhb', 'econ_vs_rhb'
            Returns zeros if no history exists
        """
        # Handle datetime objects
        if isinstance(as_of_date, datetime):
            as_of_date = as_of_date.strftime('%Y-%m-%d')

        snapshot = self._get_snapshot_for_date(as_of_date)

        if snapshot is None:
            return {
                'avg_vs_lhb': 0.0, 'econ_vs_lhb': 0.0,
                'avg_vs_rhb': 0.0, 'econ_vs_rhb': 0.0,
            }

        # Check if hand-based stats exist in snapshot
        bowling_vs_hand = snapshot.get('bowling_vs_hand', {}).get(bowler_id, {
            'left': {'runs_given': 0, 'balls_bowled': 0, 'wickets': 0},
            'right': {'runs_given': 0, 'balls_bowled': 0, 'wickets': 0},
        })

        # vs LHB
        lhb_stats = bowling_vs_hand.get('left', {'runs_given': 0, 'balls_bowled': 0, 'wickets': 0})
        if lhb_stats['balls_bowled'] == 0:
            avg_vs_lhb, econ_vs_lhb = 0.0, 0.0
        else:
            avg_vs_lhb = lhb_stats['runs_given'] / max(lhb_stats['wickets'], 1)
            econ_vs_lhb = (lhb_stats['runs_given'] / lhb_stats['balls_bowled']) * 6

        # vs RHB
        rhb_stats = bowling_vs_hand.get('right', {'runs_given': 0, 'balls_bowled': 0, 'wickets': 0})
        if rhb_stats['balls_bowled'] == 0:
            avg_vs_rhb, econ_vs_rhb = 0.0, 0.0
        else:
            avg_vs_rhb = rhb_stats['runs_given'] / max(rhb_stats['wickets'], 1)
            econ_vs_rhb = (rhb_stats['runs_given'] / rhb_stats['balls_bowled']) * 6

        return {
            'avg_vs_lhb': avg_vs_lhb,
            'econ_vs_lhb': econ_vs_lhb,
            'avg_vs_rhb': avg_vs_rhb,
            'econ_vs_rhb': econ_vs_rhb,
        }

    def get_batting_elo(self, player_id: str, as_of_date) -> float:
        """Get a player's batting ELO as of a specific date."""
        if isinstance(as_of_date, datetime):
            as_of_date = as_of_date.strftime('%Y-%m-%d')
        snapshot = self._get_snapshot_for_date(as_of_date)
        if snapshot is None:
            return 1500.0
        return snapshot.get('batting_elo', {}).get(player_id, 1500.0)

    def get_bowling_elo(self, player_id: str, as_of_date) -> float:
        """Get a player's bowling ELO as of a specific date."""
        if isinstance(as_of_date, datetime):
            as_of_date = as_of_date.strftime('%Y-%m-%d')
        snapshot = self._get_snapshot_for_date(as_of_date)
        if snapshot is None:
            return 1500.0
        return snapshot.get('bowling_elo', {}).get(player_id, 1500.0)

    def get_team_batting_elo(self, player_ids: list, as_of_date) -> float:
        """Sum of batting ELOs for a team lineup."""
        return sum(self.get_batting_elo(pid, as_of_date) for pid in player_ids)

    def get_team_bowling_elo(self, player_ids: list, as_of_date) -> float:
        """Sum of bowling ELOs for a team lineup."""
        return sum(self.get_bowling_elo(pid, as_of_date) for pid in player_ids)

    def get_team_batting_strength(self, player_ids: list, as_of_date) -> Dict[str, float]:
        """Aggregated batting stats for a team lineup."""
        avgs, srs = [], []
        for pid in player_ids:
            stats = self.get_batting_stats(pid, as_of_date)
            if stats['avg'] > 0:
                avgs.append(stats['avg'])
                srs.append(stats['sr'])
        return {
            'team_batting_avg': sum(avgs) / len(avgs) if avgs else 0.0,
            'team_batting_sr': sum(srs) / len(srs) if srs else 0.0,
        }

    def get_team_bowling_strength(self, player_ids: list, as_of_date) -> Dict[str, float]:
        """Aggregated bowling stats for a team lineup."""
        avgs, econs = [], []
        for pid in player_ids:
            stats = self.get_bowling_stats(pid, as_of_date)
            if stats['avg'] > 0:
                avgs.append(stats['avg'])
                econs.append(stats['econ'])
        return {
            'team_bowling_avg': sum(avgs) / len(avgs) if avgs else 0.0,
            'team_bowling_econ': sum(econs) / len(econs) if econs else 0.0,
        }

    def get_all_stats(self, batter_id: str, bowler_id: str, as_of_date: str) -> Dict[str, float]:
        """
        Convenience method to get all stats at once.

        Args:
            batter_id: Batter player identifier
            bowler_id: Bowler player identifier
            as_of_date: Date in 'YYYY-MM-DD' format or datetime object

        Returns:
            Combined dict with all stats
        """
        batting = self.get_batting_stats(batter_id, as_of_date)
        bowling = self.get_bowling_stats(bowler_id, as_of_date)
        h2h = self.get_h2h_stats(batter_id, bowler_id, as_of_date)

        return {
            'batsman_avg': batting['avg'],
            'batsman_sr': batting['sr'],
            'bowler_avg': bowling['avg'],
            'bowler_econ': bowling['econ'],
            'h2h_avg': h2h['avg'],
            'h2h_sr': h2h['sr']
        }


class StatsProvider:
    """Public stats-provider facade — dispatches to SQLite or chunks.

    Auto-detect:
      * If `models/player_stats_cache_{version}.sqlite` exists, use the
        mmap-backed `_SQLiteBackend`. Validate its schema_version, and
        fail loudly if the file is older than the raw chunks (staleness
        is a real source of hard-to-debug parity drift — don't mask it).
      * Otherwise fall back to `_ChunkedBackend`.

    The backend choice is logged at init time so anyone running eval can
    see which cache is in use without grepping RSS.

    Method calls are delegated to the underlying backend via __getattr__.
    For back-compat, the facade exposes `.dates` and `.version` directly
    (callers read these). The chunked-only internal `_get_snapshot_for_date`
    is intentionally NOT implemented on SQLite — reconstructing a full
    snapshot would be ~300K queries, a trap. Use the `_get_raw_*`
    helpers for per-entity raw-counter access instead.
    """

    def __init__(self, cache_dir: str = 'models', max_cached_chunks: int = 5,
                 version: str = 'v3'):
        from pathlib import Path as _Path

        cache_root = _Path(cache_dir)
        sqlite_path = cache_root / f'player_stats_cache_{version}.sqlite'

        if sqlite_path.exists():
            self._backend = self._open_sqlite(sqlite_path, cache_root, version)
            self.dates = list(self._backend._date_strs)
            self.backend_name = 'sqlite'
        else:
            print(
                f"StatsProvider: {sqlite_path.name} not found — "
                "falling back to chunked backend",
                flush=True,
            )
            self._backend = _ChunkedBackend(
                cache_dir=str(cache_root),
                max_cached_chunks=max_cached_chunks,
                version=version,
            )
            self.dates = self._backend.dates
            self.backend_name = 'chunks'

        self.version = version
        self.cache_dir = cache_root

    # --- backend selection ----------------------------------------------

    @staticmethod
    def _open_sqlite(sqlite_path, cache_root, version):
        # Local import keeps the chunked backend usable when sqlite3 is
        # unavailable or the module is imported for its class symbols
        # only. stats_sqlite_backend itself pulls sqlite3.
        from stats_sqlite_backend import _SQLiteBackend, SCHEMA_VERSION

        backend = _SQLiteBackend(str(sqlite_path))
        backend._ensure_conn()
        meta = backend.get_meta()

        file_schema = int(meta.get('schema_version', -1))
        if file_schema != SCHEMA_VERSION:
            raise RuntimeError(
                f"SQLite schema mismatch: {sqlite_path} has "
                f"schema_version={file_schema}, code expects "
                f"{SCHEMA_VERSION}. Delete the file and rebuild with "
                f"`uv run python scripts/build_stats_sqlite.py`."
            )

        # Staleness: fail loudly rather than silently serving old stats.
        chunks_dir = cache_root / f'cache_chunks_{version}'
        if chunks_dir.exists():
            chunk_files = list(chunks_dir.glob('*.pkl'))
            if chunk_files:
                chunks_mtime = max(p.stat().st_mtime for p in chunk_files)
                sqlite_src_mtime = float(meta.get('source_chunks_mtime_max', 0))
                # 1s tolerance absorbs filesystem-level rounding.
                if sqlite_src_mtime + 1 < chunks_mtime:
                    raise RuntimeError(
                        f"SQLite cache is stale:\n"
                        f"  {sqlite_path} built from chunks @ "
                        f"{sqlite_src_mtime}\n"
                        f"  {chunks_dir} current max mtime = {chunks_mtime}\n"
                        f"Rebuild with: "
                        f"uv run python scripts/build_stats_sqlite.py"
                    )

        print(
            f"StatsProvider: using SQLite backend "
            f"{sqlite_path.name} ({sqlite_path.stat().st_size / 1e6:.1f} MB)",
            flush=True,
        )
        return backend

    # --- delegation ------------------------------------------------------

    def __getattr__(self, name):
        # Only hit when the attribute isn't set on `self` directly. Defer
        # to the underlying backend (either chunks or sqlite).
        try:
            backend = self.__dict__['_backend']
        except KeyError:
            raise AttributeError(name)
        return getattr(backend, name)


class StatsProviderCache:
    """Per-instance memo layer over a StatsProvider for team/venue lookups.

    The 5 memoized methods are pure functions of (lineup_ids, date) or
    (venue, date) — both constant across every sim of a single match.
    Without this wrapper, each of the ~240 balls × 100 sims re-runs
    11-player loops inside the provider; with it, each match computes
    them once and the remaining ~24,000 calls are dict hits.

    Thin by design: non-cached methods are forwarded via __getattr__,
    so callers use the wrapped instance exactly like a StatsProvider.
    Wrap once at model construction; the wrapper is picklable (memos
    hold only immutable keys and scalar/dict outputs) and survives the
    multiprocessing.Pool.starmap hand-off to workers.
    """

    def __init__(self, provider):
        self._provider = provider
        self._team_batting_elo: Dict = {}
        self._team_bowling_elo: Dict = {}
        self._team_batting_strength: Dict = {}
        self._team_bowling_strength: Dict = {}
        self._venue_profile: Dict = {}

    def __getattr__(self, name):
        # __getattr__ runs only when normal lookup misses. During pickle
        # restore, `_provider` itself hasn't been set yet — calling
        # `self._provider` here would recurse. Use __dict__ directly and
        # surface the miss as AttributeError so pickle can proceed.
        try:
            provider = self.__dict__['_provider']
        except KeyError:
            raise AttributeError(name)
        return getattr(provider, name)

    @staticmethod
    def _norm_date(as_of_date) -> str:
        if isinstance(as_of_date, datetime):
            return as_of_date.strftime('%Y-%m-%d')
        return as_of_date

    def _team_key(self, player_ids, as_of_date):
        return (tuple(player_ids), self._norm_date(as_of_date))

    def get_team_batting_elo(self, player_ids, as_of_date) -> float:
        key = self._team_key(player_ids, as_of_date)
        cached = self._team_batting_elo.get(key)
        if cached is None:
            cached = self._provider.get_team_batting_elo(player_ids, as_of_date)
            self._team_batting_elo[key] = cached
        return cached

    def get_team_bowling_elo(self, player_ids, as_of_date) -> float:
        key = self._team_key(player_ids, as_of_date)
        cached = self._team_bowling_elo.get(key)
        if cached is None:
            cached = self._provider.get_team_bowling_elo(player_ids, as_of_date)
            self._team_bowling_elo[key] = cached
        return cached

    def get_team_batting_strength(self, player_ids, as_of_date) -> Dict[str, float]:
        key = self._team_key(player_ids, as_of_date)
        cached = self._team_batting_strength.get(key)
        if cached is None:
            cached = self._provider.get_team_batting_strength(player_ids, as_of_date)
            self._team_batting_strength[key] = cached
        return cached

    def get_team_bowling_strength(self, player_ids, as_of_date) -> Dict[str, float]:
        key = self._team_key(player_ids, as_of_date)
        cached = self._team_bowling_strength.get(key)
        if cached is None:
            cached = self._provider.get_team_bowling_strength(player_ids, as_of_date)
            self._team_bowling_strength[key] = cached
        return cached

    def get_venue_profile(self, venue: str, as_of_date) -> Dict[str, float]:
        key = (venue, self._norm_date(as_of_date))
        cached = self._venue_profile.get(key)
        if cached is None:
            cached = self._provider.get_venue_profile(venue, as_of_date)
            self._venue_profile[key] = cached
        return cached

    def clear_memo(self) -> None:
        self._team_batting_elo.clear()
        self._team_bowling_elo.clear()
        self._team_batting_strength.clear()
        self._team_bowling_strength.clear()
        self._venue_profile.clear()


def wrap_with_cache(provider):
    """Return a StatsProviderCache around `provider`; idempotent."""
    if provider is None or isinstance(provider, StatsProviderCache):
        return provider
    return StatsProviderCache(provider)


# Example usage and testing
if __name__ == "__main__":
    # Example: Load and query stats
    provider = StatsProvider()

    # Test with some sample dates and players
    test_date = '2024-06-01'
    test_player = 'dummy_player'  # Replace with actual player ID from cache

    batting_stats = provider.get_batting_stats(test_player, test_date)
    print(f"\nBatting stats for {test_player} as of {test_date}:")
    print(f"  Average: {batting_stats['avg']:.2f}")
    print(f"  Strike Rate: {batting_stats['sr']:.2f}")

    print(f"\nCache metadata:")
    for key, value in provider.metadata.items():
        print(f"  {key}: {value}")
