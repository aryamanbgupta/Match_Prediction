#!/usr/bin/env python3
"""Validate cache stats derivation against raw counters, backend-agnostic.

Asserts `get_batting_stats` / `get_bowling_stats` compute the same values
from the underlying integer counters that `_get_raw_batting` /
`_get_raw_bowling` return. Works for both the chunked StatsProvider and
the SQLite backend (Phase 3), because it doesn't touch the in-memory
snapshot format — it uses the public raw getters instead.

Sample players come from the training parquet (real batter/bowler IDs
from v3 training data). Test dates are spread across the cache range.
"""
from pathlib import Path

import pandas as pd

from stats_provider import StatsProvider


N_SAMPLE_PLAYERS = 5
TEST_DATES = ['2010-01-01', '2015-01-01', '2020-01-01', '2022-01-01']
TRAIN_PARQUET = Path('data/xgb_data_v3/cricket_data_v3_train.parquet')
TOL = 0.01


def _expected_batting(raw):
    if not raw or raw['balls'] == 0:
        return 0.0, 0.0
    avg = raw['runs'] / max(raw['dismissals'], 1)
    sr = (raw['runs'] / raw['balls']) * 100
    return avg, sr


def _expected_bowling(raw):
    if not raw or raw['balls_bowled'] == 0:
        return 0.0, 0.0
    avg = raw['runs_given'] / max(raw['wickets'], 1)
    econ = (raw['runs_given'] / raw['balls_bowled']) * 6
    return avg, econ


def _pick_sample_players():
    df = pd.read_parquet(TRAIN_PARQUET)
    batters = df['batter_id'].dropna().drop_duplicates()
    bowlers = df['bowler_id'].dropna().drop_duplicates()
    return (
        batters.sample(N_SAMPLE_PLAYERS, random_state=0).tolist(),
        bowlers.sample(N_SAMPLE_PLAYERS, random_state=0).tolist(),
    )


def validate_stats_match() -> bool:
    print("=" * 60)
    print("Validating Training vs Cache Stats Match")
    print("=" * 60)

    if not TRAIN_PARQUET.exists():
        print(f"✗ Training data not found at {TRAIN_PARQUET}")
        print("  Run parsing_v2.py first to generate training data")
        return False

    print(f"\n1. Sampling {N_SAMPLE_PLAYERS} batters + bowlers from "
          f"{TRAIN_PARQUET.name}")
    batters, bowlers = _pick_sample_players()

    print("\n2. Loading stats cache ...")
    provider = StatsProvider('models', version='v3')
    print("   ✓ loaded")

    print("\n3. Validating derivation formula against raw counters")
    mismatches = 0
    checked = 0
    for date in TEST_DATES:
        print(f"\n   date: {date}")
        for pid in batters:
            raw = provider._get_raw_batting(pid, date)
            exp_avg, exp_sr = _expected_batting(raw)
            got = provider.get_batting_stats(pid, date)
            if abs(got['avg'] - exp_avg) > TOL or abs(got['sr'] - exp_sr) > TOL:
                print(f"   ✗ batter {pid[:8]}  "
                      f"expected avg={exp_avg:.2f} sr={exp_sr:.2f}  "
                      f"got avg={got['avg']:.2f} sr={got['sr']:.2f}")
                mismatches += 1
            checked += 1

        for pid in bowlers:
            raw = provider._get_raw_bowling(pid, date)
            exp_avg, exp_econ = _expected_bowling(raw)
            got = provider.get_bowling_stats(pid, date)
            if abs(got['avg'] - exp_avg) > TOL or abs(got['econ'] - exp_econ) > TOL:
                print(f"   ✗ bowler {pid[:8]}  "
                      f"expected avg={exp_avg:.2f} econ={exp_econ:.2f}  "
                      f"got avg={got['avg']:.2f} econ={got['econ']:.2f}")
                mismatches += 1
            checked += 1

        print(f"   ✓ checked {2 * N_SAMPLE_PLAYERS} players")

    print("\n" + "=" * 60)
    if mismatches == 0:
        print(f"✓ SUCCESS — {checked} checks passed")
        print("=" * 60)
        return True
    print(f"✗ FAILED — {mismatches}/{checked} checks mismatched")
    print("=" * 60)
    return False


if __name__ == "__main__":
    import sys
    sys.exit(0 if validate_stats_match() else 1)
