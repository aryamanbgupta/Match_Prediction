"""
Player Stats Provider for Match Simulations (SQLite-only, post-Phase-B).

Single-backend facade over the schema-v3 SQLite cache at
`models/player_stats_cache_v3.sqlite` (built by
`scripts/build_stats_cache.py`). The legacy pickle-chunk backend was
removed in Phase B once the full-corpus Phase A parity harness hit
63/63 bit-exact on all 9519 matches.

Usage:
    provider = StatsProvider('models')
    stats = provider.get_batting_stats('player_123', '2024-06-01')
    # Returns: {'avg': 31.4, 'sr': 140.2}
"""

from pathlib import Path
from datetime import datetime
from typing import Dict


class StatsProvider:
    """Public stats-provider facade over `_SQLiteBackend`.

    Method calls are delegated to the backend via `__getattr__`. The
    facade exposes `.dates`, `.version`, `.backend_name`, `.cache_dir`
    directly for callers that read them. The backend's raw-counter
    helpers (`_get_raw_batting`, `_get_raw_bowling`, `_get_raw_h2h`,
    plus `get_batting_match_log_recent` / `get_bowling_match_log_recent`)
    are used by `tracker_rehydration.py` to seed the monolith's
    trackers bit-exactly for parity.
    """

    def __init__(self, cache_dir: str = 'models', max_cached_chunks: int = 5,
                 version: str = 'v3'):
        # `max_cached_chunks` is retained as a no-op kwarg for callers
        # that still pass it; the chunked backend it once configured is
        # gone.
        del max_cached_chunks

        cache_root = Path(cache_dir)
        sqlite_path = cache_root / f'player_stats_cache_{version}.sqlite'

        if not sqlite_path.exists():
            raise FileNotFoundError(
                f"{sqlite_path} not found. Build it with "
                f"`uv run python scripts/build_stats_cache.py`."
            )

        self._backend = self._open_sqlite(sqlite_path, cache_root, version)
        self.dates = list(self._backend._date_strs)
        self.backend_name = 'sqlite'
        self.version = version
        self.cache_dir = cache_root

    # --- backend selection ----------------------------------------------

    @staticmethod
    def _open_sqlite(sqlite_path, cache_root, version):
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
                f"`uv run python scripts/build_stats_cache.py`."
            )

        # Staleness: fail loudly rather than silently serving old stats.
        # `source_json_mtime_max` is written by build_stats_cache.py;
        # compare against live JSON corpus mtime.
        json_dir = cache_root.parent / 'data' / 't20s_json'
        if 'source_json_mtime_max' in meta and json_dir.exists():
            json_files = list(json_dir.glob('*.json'))
            if json_files:
                live_mtime = max(p.stat().st_mtime for p in json_files)
                sqlite_src_mtime = float(
                    meta.get('source_json_mtime_max', 0))
                if sqlite_src_mtime + 1 < live_mtime:
                    raise RuntimeError(
                        f"SQLite cache is stale:\n"
                        f"  {sqlite_path} built from JSONs @ "
                        f"{sqlite_src_mtime}\n"
                        f"  {json_dir} current max mtime = {live_mtime}\n"
                        f"Rebuild with: "
                        f"uv run python scripts/build_stats_cache.py"
                    )

        print(
            f"StatsProvider: using SQLite backend "
            f"{sqlite_path.name} ({sqlite_path.stat().st_size / 1e6:.1f} MB)",
            flush=True,
        )
        return backend

    # --- delegation ------------------------------------------------------

    def __getattr__(self, name):
        # Only hit when the attribute isn't set on `self` directly.
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
