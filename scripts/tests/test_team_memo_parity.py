"""Bit-exact parity + caching guards for StatsProviderCache.

The wrapper exists to cut repeated team-level / venue-profile lookups that
every ball of every sim re-executes. Correctness is the only acceptable
outcome: the cached response must equal what the underlying provider
would have produced. These tests pin that contract.
"""
import pickle
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stats_provider import StatsProviderCache, wrap_with_cache


class _CountingFake:
    """Minimal stats-provider-shaped fake that records call counts."""

    def __init__(self):
        self.calls = {
            'get_team_batting_elo': 0,
            'get_team_bowling_elo': 0,
            'get_team_batting_strength': 0,
            'get_team_bowling_strength': 0,
            'get_venue_profile': 0,
            'get_batting_vs_type_stats': 0,
        }

    def get_team_batting_elo(self, player_ids, as_of_date):
        self.calls['get_team_batting_elo'] += 1
        return float(sum(hash(pid) % 500 + 1500 for pid in player_ids))

    def get_team_bowling_elo(self, player_ids, as_of_date):
        self.calls['get_team_bowling_elo'] += 1
        return float(sum(hash(pid) % 400 + 1400 for pid in player_ids))

    def get_team_batting_strength(self, player_ids, as_of_date):
        self.calls['get_team_batting_strength'] += 1
        return {'team_batting_avg': float(len(player_ids)) * 2.5,
                'team_batting_sr': float(len(player_ids)) * 120.0}

    def get_team_bowling_strength(self, player_ids, as_of_date):
        self.calls['get_team_bowling_strength'] += 1
        return {'team_bowling_avg': float(len(player_ids)) * 25.0,
                'team_bowling_econ': float(len(player_ids)) * 7.5}

    def get_venue_profile(self, venue, as_of_date):
        self.calls['get_venue_profile'] += 1
        return {'venue_boundary_pct': 0.15, 'venue_dot_pct': 0.38,
                'venue_wicket_rate': 0.05, 'venue_powerplay_avg': 45.0,
                'venue_death_avg': 55.0, 'venue_first_innings_avg': 160.0,
                'venue_chase_win_pct': 0.52}

    # A method the cache does NOT override — must still work via __getattr__.
    def get_batting_vs_type_stats(self, batter_id, as_of_date):
        self.calls['get_batting_vs_type_stats'] += 1
        return {'avg_vs_pace': 30.0, 'sr_vs_pace': 140.0,
                'avg_vs_spin': 28.0, 'sr_vs_spin': 130.0}


LINEUP = ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10', 'p11']
OTHER_LINEUP = ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9', 'q10', 'q11']


def test_first_call_matches_provider_bit_exact():
    provider = _CountingFake()
    cache = StatsProviderCache(provider)
    got = cache.get_team_batting_elo(LINEUP, '2024-06-15')
    expected = _CountingFake().get_team_batting_elo(LINEUP, '2024-06-15')
    assert got == expected


def test_repeated_same_args_hits_cache():
    provider = _CountingFake()
    cache = StatsProviderCache(provider)
    for _ in range(50):
        cache.get_team_batting_elo(LINEUP, '2024-06-15')
        cache.get_team_bowling_elo(LINEUP, '2024-06-15')
        cache.get_team_batting_strength(LINEUP, '2024-06-15')
        cache.get_team_bowling_strength(LINEUP, '2024-06-15')
        cache.get_venue_profile('MCG', '2024-06-15')
    assert provider.calls['get_team_batting_elo'] == 1
    assert provider.calls['get_team_bowling_elo'] == 1
    assert provider.calls['get_team_batting_strength'] == 1
    assert provider.calls['get_team_bowling_strength'] == 1
    assert provider.calls['get_venue_profile'] == 1


def test_distinct_args_miss_cache():
    provider = _CountingFake()
    cache = StatsProviderCache(provider)
    cache.get_team_batting_elo(LINEUP, '2024-06-15')
    cache.get_team_batting_elo(LINEUP, '2024-06-16')       # different date
    cache.get_team_batting_elo(OTHER_LINEUP, '2024-06-15')  # different lineup
    assert provider.calls['get_team_batting_elo'] == 3


def test_datetime_and_string_dates_are_same_key():
    provider = _CountingFake()
    cache = StatsProviderCache(provider)
    cache.get_team_batting_elo(LINEUP, '2024-06-15')
    cache.get_team_batting_elo(LINEUP, datetime(2024, 6, 15))
    cache.get_venue_profile('MCG', '2024-06-15')
    cache.get_venue_profile('MCG', datetime(2024, 6, 15))
    assert provider.calls['get_team_batting_elo'] == 1
    assert provider.calls['get_venue_profile'] == 1


def test_passthrough_for_uncached_methods():
    provider = _CountingFake()
    cache = StatsProviderCache(provider)
    assert hasattr(cache, 'get_batting_vs_type_stats')
    got = cache.get_batting_vs_type_stats('p1', '2024-06-15')
    assert got['avg_vs_pace'] == 30.0
    # Uncached — every call hits the underlying provider.
    cache.get_batting_vs_type_stats('p1', '2024-06-15')
    assert provider.calls['get_batting_vs_type_stats'] == 2


def test_pickle_roundtrip_preserves_memo():
    provider = _CountingFake()
    cache = StatsProviderCache(provider)
    pre = cache.get_team_batting_elo(LINEUP, '2024-06-15')
    restored = pickle.loads(pickle.dumps(cache))
    assert restored.get_team_batting_elo(LINEUP, '2024-06-15') == pre
    # The underlying provider on the restored instance saw zero new calls
    # (restored._provider is a pickled copy, isolated from `provider`).
    assert restored._provider.calls['get_team_batting_elo'] == 1


def test_wrap_with_cache_is_idempotent():
    provider = _CountingFake()
    once = wrap_with_cache(provider)
    twice = wrap_with_cache(once)
    assert once is twice
    assert wrap_with_cache(None) is None


def test_clear_memo_forces_refresh():
    provider = _CountingFake()
    cache = StatsProviderCache(provider)
    cache.get_team_batting_elo(LINEUP, '2024-06-15')
    cache.get_venue_profile('MCG', '2024-06-15')
    cache.clear_memo()
    cache.get_team_batting_elo(LINEUP, '2024-06-15')
    cache.get_venue_profile('MCG', '2024-06-15')
    assert provider.calls['get_team_batting_elo'] == 2
    assert provider.calls['get_venue_profile'] == 2
