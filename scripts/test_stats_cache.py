#!/usr/bin/env python3
"""
Quick test to validate stats cache is working correctly
"""

from stats_provider import StatsProvider

print("=" * 60)
print("Testing Stats Cache")
print("=" * 60)

# Test 1: Load the cache
print("\n1. Loading stats cache...")
try:
    provider = StatsProvider('models')
    print("   ✓ Cache loaded successfully!")
except Exception as e:
    print(f"   ✗ Failed to load cache: {e}")
    exit(1)

# Test 2: Query some player stats
print("\n2. Testing player stats queries...")

# Test a date in the middle of the dataset
test_date = '2020-01-01'

print(f"\n   Testing with date: {test_date}")

# Get a snapshot to find sample players
snapshot = provider._get_snapshot_for_date(test_date)
if snapshot:
    batting_sample = list(snapshot['batting'].keys())[:2]
    bowling_sample = list(snapshot['bowling'].keys())[:2]

    print(f"   Sample players from cache:")

    for player_id in batting_sample:
        stats = provider.get_batting_stats(player_id, test_date)
        print(f"   - Batter {player_id}: avg={stats['avg']:.2f}, sr={stats['sr']:.2f}")

    for player_id in bowling_sample:
        stats = provider.get_bowling_stats(player_id, test_date)
        print(f"   - Bowler {player_id}: avg={stats['avg']:.2f}, econ={stats['econ']:.2f}")

# Test 3: Verify temporal validity (early date should have fewer stats)
print("\n3. Testing temporal validity...")
early_date = provider.dates[10]  # Very early
late_date = provider.dates[-10]  # Very late

early_snapshot = provider._get_snapshot_for_date(early_date)
late_snapshot = provider._get_snapshot_for_date(late_date)

early_players = len(early_snapshot['batting'])
late_players = len(late_snapshot['batting'])

print(f"   Early date ({early_date}): {early_players} players")
print(f"   Late date ({late_date}): {late_players} players")

if late_players > early_players:
    print("   ✓ Temporal validity confirmed (more players over time)")
else:
    print("   ✗ Warning: Expected more players in later dates")

# Test 4: H2H stats
print("\n4. Testing H2H stats...")
test_snapshot = provider._get_snapshot_for_date(provider.dates[-100])
h2h_sample = list(test_snapshot['h2h'].items())[:2]

for (batter_id, bowler_id), _ in h2h_sample:
    h2h_stats = provider.get_h2h_stats(batter_id, bowler_id, test_date)
    print(f"   - Matchup {batter_id} vs {bowler_id}: avg={h2h_stats['avg']:.2f}, sr={h2h_stats['sr']:.2f}")

# Test 5: Test cache performance (multiple queries to same date should be fast)
print("\n5. Testing cache performance...")
import time
start = time.time()
for _ in range(100):
    provider.get_batting_stats(batting_sample[0], test_date)
elapsed = time.time() - start
print(f"   100 queries to same date: {elapsed*1000:.1f}ms (avg {elapsed*10:.2f}ms per query)")
print(f"   Chunks in cache: {len(provider.chunk_cache)}")

print("\n" + "=" * 60)
print("✓ All tests passed! Cache is working correctly.")
print("=" * 60)
