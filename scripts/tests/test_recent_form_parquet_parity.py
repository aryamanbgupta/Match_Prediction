"""Parquet ↔ SQLite parity for the 4 recent-form features.

The SQLite cache was rebuilt from the same snapshots that fed the parquet
training data, so `get_batting_recent(batter_id, match_date)` at inference
time must return the same numbers that are baked into the parquet row for
that (batter, match_date) pair.

Snapshot semantics subtlety: snapshots are keyed per calendar date and taken
BEFORE the first match of that date. Within a single match on that date,
`batsman_recent_*` values are constant across all balls. If the same batter
plays in a *second* match on the same date (rare but possible in doubleheaders
/ women's + men's same-day bills), the parquet values for that second match
reflect the post-match-1 tracker state, while SQLite still returns the
pre-match-1 snapshot. To avoid conflating that with a real bug, we restrict
parity to (batter_id, match_date) pairs that appear in exactly one innings
per that date in the parquet — the common case.

Run:
    uv run python scripts/tests/test_recent_form_parquet_parity.py
"""
from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))

from stats_provider import StatsProvider  # noqa: E402


PARQUET_PATH = ROOT / 'data' / 'xgb_data_v3' / 'cricket_data_v3_test.parquet'
SAMPLE_N = 2_000
RANDOM_SEED = 0xBEEF


def _load_parquet_slim() -> pd.DataFrame:
    print(f"loading parquet: {PARQUET_PATH} ...", flush=True)
    t0 = time.time()
    cols = [
        'batter_id', 'bowler_id', 'match_date', 'innings_id',
        'batsman_recent_avg', 'batsman_recent_sr',
        'bowler_recent_avg', 'bowler_recent_econ',
    ]
    df = pd.read_parquet(PARQUET_PATH, columns=cols)
    print(f"  loaded {len(df):,} rows in {time.time()-t0:.1f}s", flush=True)
    if 'match_date' not in df.columns:
        raise SystemExit(
            "match_date column missing — re-run parsing_v2.py to regenerate "
            "the parquet with the new column."
        )
    if df['match_date'].isna().any():
        missing = int(df['match_date'].isna().sum())
        raise SystemExit(f"match_date has {missing} NaNs — aborting.")
    return df


def _pick_single_match_rows(df: pd.DataFrame, col_id: str) -> pd.DataFrame:
    """Keep rows where (col_id, match_date) has exactly one innings_id — i.e.
    the player isn't involved in two matches that day, so snapshot parity is
    guaranteed."""
    grouped = (
        df.groupby([col_id, 'match_date'])['innings_id']
        .nunique()
        .reset_index(name='n_innings')
    )
    single = grouped[grouped['n_innings'] == 1][[col_id, 'match_date']]
    print(f"  {col_id}: {len(single):,} single-match "
          f"(player, date) pairs", flush=True)
    merged = df.merge(single, on=[col_id, 'match_date'], how='inner')
    return merged


def _sample(df: pd.DataFrame, rng: random.Random) -> pd.DataFrame:
    n = min(SAMPLE_N, len(df))
    idxs = rng.sample(range(len(df)), n)
    return df.iloc[idxs]


def _assert_batting(provider: StatsProvider, df: pd.DataFrame, rng: random.Random):
    print("\nbatting parity ...", flush=True)
    eligible = _pick_single_match_rows(
        df[['batter_id', 'match_date', 'innings_id',
            'batsman_recent_avg', 'batsman_recent_sr']],
        col_id='batter_id',
    )
    sample = _sample(eligible, rng)
    print(f"  checking {len(sample):,} batting rows", flush=True)

    failures = []
    for _, row in sample.iterrows():
        got = provider.get_batting_recent(str(row['batter_id']), row['match_date'])
        want = (float(row['batsman_recent_avg']), float(row['batsman_recent_sr']))
        have = (float(got['avg']), float(got['sr']))
        if want != have:
            failures.append((row['batter_id'], row['match_date'], want, have))
            if len(failures) >= 5:
                break

    if failures:
        print(f"FAIL: {len(failures)}+ batting mismatches (first 5):")
        for pid, d, want, have in failures:
            print(f"  batter={pid} date={d}")
            print(f"    parquet: avg={want[0]!r} sr={want[1]!r}")
            print(f"    sqlite:  avg={have[0]!r} sr={have[1]!r}")
        raise AssertionError(f"{len(failures)}+ batting mismatches")
    print(f"  PASS {len(sample):,} batting rows")


def _assert_bowling(provider: StatsProvider, df: pd.DataFrame, rng: random.Random):
    print("\nbowling parity ...", flush=True)
    eligible = _pick_single_match_rows(
        df[['bowler_id', 'match_date', 'innings_id',
            'bowler_recent_avg', 'bowler_recent_econ']],
        col_id='bowler_id',
    )
    sample = _sample(eligible, rng)
    print(f"  checking {len(sample):,} bowling rows", flush=True)

    failures = []
    for _, row in sample.iterrows():
        got = provider.get_bowling_recent(str(row['bowler_id']), row['match_date'])
        want = (float(row['bowler_recent_avg']), float(row['bowler_recent_econ']))
        have = (float(got['avg']), float(got['econ']))
        if want != have:
            failures.append((row['bowler_id'], row['match_date'], want, have))
            if len(failures) >= 5:
                break

    if failures:
        print(f"FAIL: {len(failures)}+ bowling mismatches (first 5):")
        for pid, d, want, have in failures:
            print(f"  bowler={pid} date={d}")
            print(f"    parquet: avg={want[0]!r} econ={want[1]!r}")
            print(f"    sqlite:  avg={have[0]!r} econ={have[1]!r}")
        raise AssertionError(f"{len(failures)}+ bowling mismatches")
    print(f"  PASS {len(sample):,} bowling rows")


def test_recent_form_parquet_parity():
    rng = random.Random(RANDOM_SEED)
    df = _load_parquet_slim()
    provider = StatsProvider(str(ROOT / 'models'), version='v3')
    _assert_batting(provider, df, rng)
    _assert_bowling(provider, df, rng)


if __name__ == '__main__':
    test_recent_form_parquet_parity()
    print("\nALL RECENT-FORM PARQUET PARITY CHECKS PASSED")
