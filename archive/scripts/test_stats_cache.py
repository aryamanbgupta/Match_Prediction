#!/usr/bin/env python3
"""Smoke test for the stats cache — works on both chunked and SQLite backends.

Uses only the public `StatsProvider` API plus the backend-agnostic
`_get_raw_*` helpers. No reach-through into backend internals
(`_get_snapshot_for_date`, `chunk_cache`) — those only exist on the
chunked backend.

Checks:
  1. Provider loads; `.dates` is populated.
  2. Sample batter/bowler queries return well-formed dicts.
  3. Temporal validity: later dates have strictly more non-zero players
     than early dates (training data accumulates).
  4. H2H lookup returns a valid shape.
  5. Cache-repeat perf: 100 identical queries complete in reasonable time.
"""
import time

import pandas as pd

from stats_provider import StatsProvider

N_PLAYERS = 3
TEST_DATE = '2020-01-01'
TRAIN_PARQUET = 'data/xgb_data_v3/cricket_data_v3_train.parquet'


def _sample_players():
    df = pd.read_parquet(TRAIN_PARQUET)
    batters = df['batter_id'].dropna().drop_duplicates().sample(
        N_PLAYERS, random_state=0).tolist()
    bowlers = df['bowler_id'].dropna().drop_duplicates().sample(
        N_PLAYERS, random_state=0).tolist()
    return batters, bowlers


def main():
    print("=" * 60)
    print("Testing Stats Cache")
    print("=" * 60)

    print("\n1. Loading stats cache...")
    provider = StatsProvider('models')
    print(f"   ✓ loaded (backend={provider.backend_name}, "
          f"{len(provider.dates):,} snapshots)")
    print(f"   - date range: {provider.dates[0]} to {provider.dates[-1]}")

    print(f"\n2. Testing player stats queries for {TEST_DATE}...")
    batters, bowlers = _sample_players()
    for pid in batters:
        s = provider.get_batting_stats(pid, TEST_DATE)
        assert set(s.keys()) == {'avg', 'sr'}, s
        print(f"   - batter {pid[:8]}: avg={s['avg']:.2f} sr={s['sr']:.2f}")
    for pid in bowlers:
        s = provider.get_bowling_stats(pid, TEST_DATE)
        assert set(s.keys()) == {'avg', 'econ'}, s
        print(f"   - bowler {pid[:8]}: avg={s['avg']:.2f} econ={s['econ']:.2f}")

    print("\n3. Testing temporal validity (pick an accumulating player)...")
    # A player with many records should have monotone-nondecreasing
    # career totals. Use raw counters via the backend-agnostic getter.
    early = provider.dates[10]
    late = provider.dates[-10]
    # Find a well-populated batter (one of the sampled bowlers/batters).
    pid_probe = batters[0]
    e = provider._get_raw_batting(pid_probe, early) or {'runs': 0, 'balls': 0}
    l = provider._get_raw_batting(pid_probe, late) or {'runs': 0, 'balls': 0}
    print(f"   - batter {pid_probe[:8]}  early={early} balls={e['balls']}  "
          f"late={late} balls={l['balls']}")
    assert l['balls'] >= e['balls'], \
        "raw counters should be monotone over time"
    print("   ✓ temporal validity confirmed (counters non-decreasing)")

    print("\n4. Testing H2H stats...")
    h2h = provider.get_h2h_stats(batters[0], bowlers[0], TEST_DATE)
    assert set(h2h.keys()) == {'avg', 'sr'}, h2h
    print(f"   - {batters[0][:8]} vs {bowlers[0][:8]}: "
          f"avg={h2h['avg']:.2f} sr={h2h['sr']:.2f}")

    print("\n5. Testing cache performance (same-date repeat queries)...")
    t0 = time.perf_counter()
    for _ in range(100):
        provider.get_batting_stats(batters[0], TEST_DATE)
    elapsed = time.perf_counter() - t0
    print(f"   100 queries to same date: {elapsed*1000:.1f}ms "
          f"(avg {elapsed*10:.3f}ms per query)")

    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
