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
from typing import Dict, Optional


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
                 version: str = 'v3',
                 require_order_contract: bool = False,
                 required_schema_version: Optional[int] = None):
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

        self._backend = self._open_sqlite(
            sqlite_path,
            cache_root,
            version,
            require_order_contract=require_order_contract,
            required_schema_version=required_schema_version,
        )
        self.dates = list(self._backend._date_strs)
        self.backend_name = 'sqlite'
        self.version = version
        self.cache_dir = cache_root

    # --- backend selection ----------------------------------------------

    @staticmethod
    def _open_sqlite(
        sqlite_path,
        cache_root,
        version,
        require_order_contract=False,
        required_schema_version=None,
    ):
        from loaders_common import SAME_DAY_ORDER_VERSION
        from stats_sqlite_backend import (
            _SQLiteBackend,
            SUPPORTED_SCHEMA_VERSIONS,
        )

        backend = _SQLiteBackend(str(sqlite_path))
        backend._ensure_conn()
        meta = backend.get_meta()

        file_schema = int(meta.get('schema_version', -1))
        if file_schema not in SUPPORTED_SCHEMA_VERSIONS:
            raise RuntimeError(
                f"SQLite schema mismatch: {sqlite_path} has "
                f"schema_version={file_schema}, code expects "
                f"one of {SUPPORTED_SCHEMA_VERSIONS}. Delete the file and "
                "rebuild with "
                f"`uv run python scripts/build_stats_cache.py`."
            )
        if (
            required_schema_version is not None
            and file_schema != int(required_schema_version)
        ):
            raise RuntimeError(
                f"SQLite schema mismatch: {sqlite_path} has "
                f"schema_version={file_schema}, this caller requires "
                f"{required_schema_version}."
            )

        cache_order = meta.get('same_day_order_version')
        if cache_order != SAME_DAY_ORDER_VERSION:
            message = (
                f"SQLite same-day ordering mismatch: {sqlite_path} has "
                f"{cache_order!r}, code expects {SAME_DAY_ORDER_VERSION!r}. "
                "Rebuild the cache before deterministic materialization."
            )
            if require_order_contract:
                raise RuntimeError(message)
            print(f"WARNING: {message}", flush=True)

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
        # Short-circuit dunders so pickle doesn't pull __setstate__ / etc.
        # from the wrapped backend during deserialisation (which would
        # restore the wrong shape onto this instance).
        if name.startswith('__'):
            raise AttributeError(name)
        try:
            backend = self.__dict__['_backend']
        except KeyError:
            raise AttributeError(name)
        return getattr(backend, name)

    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)


class StatsProviderCache:
    """Per-instance memo layer over a StatsProvider.

    Two tiers of memoization:

      1. Team-level methods (5 originals): keyed on `(tuple(lineup_ids), date)`
         or `(venue, date)`. Constant across every ball of a single match.
      2. Per-player getters (12 added 2026-05-08): keyed on `(player_id, date)`
         or `(p1, p2, date)` for h2h. Within an innings the striker stays
         many balls; the bowler stays for 6. Cache locality is enormous.

    The cProfile of v7's 41-min eval showed 1.4 M SQLite queries — most of
    them per-player getters re-fetching the same `(pid, date)` row. With
    tier 2 in place the post-warmup hot path is dict hits.

    Thin by design: non-cached methods are forwarded via __getattr__,
    so callers use the wrapped instance exactly like a StatsProvider.
    Wrap once at model construction; the wrapper is picklable (memos
    hold only immutable keys and scalar/dict outputs) and survives the
    multiprocessing.Pool.starmap hand-off to workers.

    Memory: across a 261-match eval each memo holds ≤ 261 × ~30 = ~8K
    entries (one per unique `(player, date)` pair). Twelve memos × ~16 KB
    each ≈ 200 KB. No invalidation needed at production scale.
    """

    def __init__(self, provider):
        self._provider = provider
        # Tier 1: team-level (originals).
        self._team_batting_elo: Dict = {}
        self._team_bowling_elo: Dict = {}
        self._team_batting_strength: Dict = {}
        self._team_bowling_strength: Dict = {}
        self._venue_profile: Dict = {}
        # Tier 2: per-player getters keyed on (pid, date) or
        # (bid, bowler_id, date) for h2h.
        self._batting_stats: Dict = {}
        self._bowling_stats: Dict = {}
        self._h2h_stats: Dict = {}
        self._batting_recent: Dict = {}
        self._bowling_recent: Dict = {}
        self._batting_vs_type_stats: Dict = {}
        self._bowling_vs_hand_stats: Dict = {}
        # Outcome-dist getters take a `k` (and `hierarchical` for vs-cells)
        # kwarg; in production the values are constant per run, but include
        # them in the key so a non-default call can't return a stale entry.
        self._batter_outcome_dist: Dict = {}
        self._bowler_outcome_dist: Dict = {}
        self._batter_vs_type_outcome_dist: Dict = {}
        self._bowler_vs_hand_outcome_dist: Dict = {}
        self._venue_outcome_dist: Dict = {}
        self._batter_phase_outcome_dist: Dict = {}
        self._bowler_phase_outcome_dist: Dict = {}
        self._h2h_outcome_dist: Dict = {}

    def __getattr__(self, name):
        # __getattr__ runs only when normal lookup misses. During pickle
        # restore, `_provider` itself hasn't been set yet — calling
        # `self._provider` here would recurse. Use __dict__ directly and
        # surface the miss as AttributeError so pickle can proceed.
        # Dunder names short-circuit straight to AttributeError: pickle
        # probes for `__setstate__` / `__reduce_ex__` etc. during restore,
        # and forwarding those to the (not-yet-restored) provider would
        # return inappropriate methods that corrupt the deserialised state.
        if name.startswith('__'):
            raise AttributeError(name)
        try:
            provider = self.__dict__['_provider']
        except KeyError:
            raise AttributeError(name)
        return getattr(provider, name)

    def __getstate__(self):
        # Explicit so pickle uses __dict__ verbatim instead of probing
        # through __getattr__ (which forwards to the wrapped provider and
        # confuses the deserialiser).
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)

    @staticmethod
    def _norm_date(as_of_date) -> str:
        if isinstance(as_of_date, datetime):
            # `strftime('%Y-%m-%d')` is unusually expensive on the simulator
            # hot path (~17 calls per delivery). `date().isoformat()` has the
            # same YYYY-MM-DD contract and is hundreds of times faster.
            return as_of_date.date().isoformat()
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

    # --- Tier 2: per-player getters --------------------------------------

    def _player_key(self, player_id, as_of_date):
        return (player_id, self._norm_date(as_of_date))

    def get_batting_stats(self, player_id, as_of_date) -> Dict[str, float]:
        key = self._player_key(player_id, as_of_date)
        cached = self._batting_stats.get(key)
        if cached is None:
            cached = self._provider.get_batting_stats(player_id, as_of_date)
            self._batting_stats[key] = cached
        return cached

    def get_bowling_stats(self, player_id, as_of_date) -> Dict[str, float]:
        key = self._player_key(player_id, as_of_date)
        cached = self._bowling_stats.get(key)
        if cached is None:
            cached = self._provider.get_bowling_stats(player_id, as_of_date)
            self._bowling_stats[key] = cached
        return cached

    def get_h2h_stats(self, batter_id, bowler_id, as_of_date) -> Dict[str, float]:
        key = (batter_id, bowler_id, self._norm_date(as_of_date))
        cached = self._h2h_stats.get(key)
        if cached is None:
            cached = self._provider.get_h2h_stats(batter_id, bowler_id, as_of_date)
            self._h2h_stats[key] = cached
        return cached

    def get_batting_recent(self, player_id, as_of_date) -> Dict[str, float]:
        key = self._player_key(player_id, as_of_date)
        cached = self._batting_recent.get(key)
        if cached is None:
            cached = self._provider.get_batting_recent(player_id, as_of_date)
            self._batting_recent[key] = cached
        return cached

    def get_bowling_recent(self, player_id, as_of_date) -> Dict[str, float]:
        key = self._player_key(player_id, as_of_date)
        cached = self._bowling_recent.get(key)
        if cached is None:
            cached = self._provider.get_bowling_recent(player_id, as_of_date)
            self._bowling_recent[key] = cached
        return cached

    def get_batting_vs_type_stats(self, player_id, as_of_date) -> Dict[str, float]:
        key = self._player_key(player_id, as_of_date)
        cached = self._batting_vs_type_stats.get(key)
        if cached is None:
            cached = self._provider.get_batting_vs_type_stats(player_id, as_of_date)
            self._batting_vs_type_stats[key] = cached
        return cached

    def get_bowling_vs_hand_stats(self, player_id, as_of_date) -> Dict[str, float]:
        key = self._player_key(player_id, as_of_date)
        cached = self._bowling_vs_hand_stats.get(key)
        if cached is None:
            cached = self._provider.get_bowling_vs_hand_stats(player_id, as_of_date)
            self._bowling_vs_hand_stats[key] = cached
        return cached

    # Outcome-dist getters (Phase 5/6). `k` and `hierarchical` are part of
    # the key so a non-default call doesn't poison the default-arg cache.

    def get_batter_outcome_dist(self, player_id, as_of_date,
                                k: float = 30.0) -> Dict[str, float]:
        key = (player_id, self._norm_date(as_of_date), k)
        cached = self._batter_outcome_dist.get(key)
        if cached is None:
            cached = self._provider.get_batter_outcome_dist(player_id, as_of_date, k=k)
            self._batter_outcome_dist[key] = cached
        return cached

    def get_bowler_outcome_dist(self, player_id, as_of_date,
                                k: float = 30.0) -> Dict[str, float]:
        key = (player_id, self._norm_date(as_of_date), k)
        cached = self._bowler_outcome_dist.get(key)
        if cached is None:
            cached = self._provider.get_bowler_outcome_dist(player_id, as_of_date, k=k)
            self._bowler_outcome_dist[key] = cached
        return cached

    def get_batter_vs_type_outcome_dist(self, player_id, as_of_date,
                                        k: float = 30.0,
                                        hierarchical: bool = True) -> Dict[str, float]:
        key = (player_id, self._norm_date(as_of_date), k, hierarchical)
        cached = self._batter_vs_type_outcome_dist.get(key)
        if cached is None:
            cached = self._provider.get_batter_vs_type_outcome_dist(
                player_id, as_of_date, k=k, hierarchical=hierarchical)
            self._batter_vs_type_outcome_dist[key] = cached
        return cached

    def get_bowler_vs_hand_outcome_dist(self, player_id, as_of_date,
                                        k: float = 30.0,
                                        hierarchical: bool = True) -> Dict[str, float]:
        key = (player_id, self._norm_date(as_of_date), k, hierarchical)
        cached = self._bowler_vs_hand_outcome_dist.get(key)
        if cached is None:
            cached = self._provider.get_bowler_vs_hand_outcome_dist(
                player_id, as_of_date, k=k, hierarchical=hierarchical)
            self._bowler_vs_hand_outcome_dist[key] = cached
        return cached

    def get_venue_outcome_dist(self, venue: str, as_of_date,
                               k: float = 200.0) -> Dict[str, float]:
        key = (venue, self._norm_date(as_of_date), k)
        cached = self._venue_outcome_dist.get(key)
        if cached is None:
            cached = self._provider.get_venue_outcome_dist(venue, as_of_date, k=k)
            self._venue_outcome_dist[key] = cached
        return cached

    def get_batter_phase_outcome_dist(
        self,
        player_id,
        as_of_date,
        balls_bowled: int,
        k_player: float = 30.0,
        k_phase: float = 30.0,
    ) -> Dict[str, float]:
        key = (
            player_id, self._norm_date(as_of_date), int(balls_bowled),
            k_player, k_phase,
        )
        cached = self._batter_phase_outcome_dist.get(key)
        if cached is None:
            cached = self._provider.get_batter_phase_outcome_dist(
                player_id,
                as_of_date,
                balls_bowled,
                k_player=k_player,
                k_phase=k_phase,
            )
            self._batter_phase_outcome_dist[key] = cached
        return cached

    def get_bowler_phase_outcome_dist(
        self,
        player_id,
        as_of_date,
        balls_bowled: int,
        k_player: float = 30.0,
        k_phase: float = 30.0,
    ) -> Dict[str, float]:
        key = (
            player_id, self._norm_date(as_of_date), int(balls_bowled),
            k_player, k_phase,
        )
        cached = self._bowler_phase_outcome_dist.get(key)
        if cached is None:
            cached = self._provider.get_bowler_phase_outcome_dist(
                player_id,
                as_of_date,
                balls_bowled,
                k_player=k_player,
                k_phase=k_phase,
            )
            self._bowler_phase_outcome_dist[key] = cached
        return cached

    def get_h2h_outcome_dist(
        self,
        batter_id,
        bowler_id,
        as_of_date,
        k_player: float = 30.0,
        k_h2h: float = 60.0,
    ) -> Dict[str, float]:
        key = (
            batter_id, bowler_id, self._norm_date(as_of_date),
            k_player, k_h2h,
        )
        cached = self._h2h_outcome_dist.get(key)
        if cached is None:
            cached = self._provider.get_h2h_outcome_dist(
                batter_id,
                bowler_id,
                as_of_date,
                k_player=k_player,
                k_h2h=k_h2h,
            )
            self._h2h_outcome_dist[key] = cached
        return cached

    def clear_memo(self) -> None:
        # Tier 1
        self._team_batting_elo.clear()
        self._team_bowling_elo.clear()
        self._team_batting_strength.clear()
        self._team_bowling_strength.clear()
        self._venue_profile.clear()
        # Tier 2
        self._batting_stats.clear()
        self._bowling_stats.clear()
        self._h2h_stats.clear()
        self._batting_recent.clear()
        self._bowling_recent.clear()
        self._batting_vs_type_stats.clear()
        self._bowling_vs_hand_stats.clear()
        self._batter_outcome_dist.clear()
        self._bowler_outcome_dist.clear()
        self._batter_vs_type_outcome_dist.clear()
        self._bowler_vs_hand_outcome_dist.clear()
        self._venue_outcome_dist.clear()
        self._batter_phase_outcome_dist.clear()
        self._bowler_phase_outcome_dist.clear()
        self._h2h_outcome_dist.clear()


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
