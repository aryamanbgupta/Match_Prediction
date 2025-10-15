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


class StatsProvider:
    """
    Provides temporal access to player statistics for simulations.

    DESIGN DECISION: Chunked lazy loading with LRU cache
    REASONING: 7.8GB full cache too large for memory, chunking allows
               on-demand loading while maintaining fast O(log n) lookups.
               LRU cache keeps frequently-used chunks in memory.
    """

    def __init__(self, cache_dir: str = 'models', max_cached_chunks: int = 5):
        """
        Load the player stats cache from chunked format with lazy loading.

        Args:
            cache_dir: Directory containing cache chunk files
            max_cached_chunks: Maximum number of chunks to keep in memory (LRU cache)
        """
        from pathlib import Path
        from collections import OrderedDict

        self.cache_dir = Path(cache_dir)
        metadata_path = self.cache_dir / 'player_stats_cache_metadata.pkl'

        print(f"Loading player stats cache from {cache_dir}...")

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
