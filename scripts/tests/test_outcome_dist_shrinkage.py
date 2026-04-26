"""Unit tests for the Dirichlet-posterior-mean shrinkage used by the
schema-v4 outcome-distribution feature path.

The shrinkage formula, duplicated in `parsing_v2._shrink_counts` (live-
tracker path) and `stats_sqlite_backend._SQLiteBackend._shrink` (SQLite
read path), is:

    p̂_c = (n_c + k · π_c) / (N + k),    N = Σ n_c

These tests lock the two limits and the invariant Σ p̂ = 1.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from parsing_v2 import PlayerStatsTracker, VenueStatsTracker, _shrink_counts
from stats_sqlite_backend import _SQLiteBackend


# A non-uniform prior closer to the true T20 distribution so we can
# distinguish prior-fallback from uniform-fallback regressions.
PRIOR = (0.35, 0.40, 0.10, 0.08, 0.04, 0.03)


def _approx_equal(a, b, tol=1e-9):
    return all(abs(x - y) <= tol for x, y in zip(a, b))


def test_shrink_empty_counts_returns_prior():
    """N → 0 ⇒ p̂ → π (full shrinkage)."""
    p = _shrink_counts((0, 0, 0, 0, 0, 0), PRIOR, k=30.0)
    assert _approx_equal(p, PRIOR), f"empty → prior: got {p}, expected {PRIOR}"


def test_shrink_heavy_data_approaches_mle():
    """N → ∞ ⇒ p̂ → n/N (data dominates)."""
    counts = (10_000, 20_000, 5_000, 3_000, 1_000, 1_000)
    N = sum(counts)
    p = _shrink_counts(counts, PRIOR, k=30.0)
    mle = tuple(c / N for c in counts)
    # At N=40k and k=30, shrinkage pull < 0.001 per component.
    assert _approx_equal(p, mle, tol=1e-3), (
        f"heavy-data: p={p}, mle={mle}"
    )


def test_shrink_output_sums_to_one():
    """Σ p̂ = 1 always, for any n and any k > 0."""
    for counts, k in [
        ((0, 0, 0, 0, 0, 0), 30.0),
        ((1, 2, 3, 4, 5, 6), 30.0),
        ((100, 0, 0, 0, 0, 0), 200.0),
        ((7, 7, 7, 7, 7, 7), 1.0),
    ]:
        p = _shrink_counts(counts, PRIOR, k=k)
        s = sum(p)
        assert abs(s - 1.0) < 1e-9, f"sum {s} != 1 for counts={counts} k={k}"


def test_shrink_half_weight_point():
    """At N = k: p̂ is the equal-weight average of MLE and prior. This
    is the most interpretable regression lock on the formula."""
    counts = (30, 0, 0, 0, 0, 0)  # N = 30 = k
    p = _shrink_counts(counts, PRIOR, k=30.0)
    # MLE is (1,0,0,0,0,0); half-and-half with PRIOR.
    expected = (0.5 * 1 + 0.5 * PRIOR[0],
                0.5 * 0 + 0.5 * PRIOR[1],
                0.5 * 0 + 0.5 * PRIOR[2],
                0.5 * 0 + 0.5 * PRIOR[3],
                0.5 * 0 + 0.5 * PRIOR[4],
                0.5 * 0 + 0.5 * PRIOR[5])
    assert _approx_equal(p, expected), f"half-weight: got {p}, expected {expected}"


def test_backend_shrink_matches_parsing_shrink():
    """Both shrinkage implementations (tracker + SQLite backend) must
    produce bit-identical output for the same input."""
    for counts, k in [
        ((0, 0, 0, 0, 0, 0), 30.0),
        ((3, 10, 2, 5, 1, 1), 30.0),
        ((500, 800, 200, 150, 50, 100), 200.0),
    ]:
        a = _shrink_counts(counts, PRIOR, k=k)
        b = _SQLiteBackend._shrink(counts, PRIOR, k=k)
        assert _approx_equal(a, b, tol=0.0), (
            f"tracker vs backend mismatch: counts={counts} k={k} "
            f"tracker={a} backend={b}"
        )


def test_tracker_getter_dist_sums_to_one():
    """PlayerStatsTracker.get_batter_outcome_dist must return a valid
    probability vector regardless of how much history is present."""
    t = PlayerStatsTracker()
    # Empty tracker — must fall back to prior.
    d = t.get_batter_outcome_dist('alice', PRIOR, k=30.0)
    assert abs(sum(d.values()) - 1.0) < 1e-9
    assert abs(d['batter_p0'] - PRIOR[0]) < 1e-9

    # After updates, still sums to 1.
    t.update_stats('alice', 'bob', 0, False)
    t.update_stats('alice', 'bob', 4, False)
    t.update_stats('alice', 'bob', 6, False)
    t.update_stats('alice', 'bob', 0, True)
    d = t.get_batter_outcome_dist('alice', PRIOR, k=30.0)
    assert abs(sum(d.values()) - 1.0) < 1e-9
    assert set(d.keys()) == {
        'batter_p0', 'batter_p1', 'batter_p2',
        'batter_p4', 'batter_p6', 'batter_pw',
    }


def test_venue_tracker_dist_sums_to_one():
    v = VenueStatsTracker()
    # Empty.
    d = v.get_venue_outcome_dist('MCG', PRIOR, k=200.0)
    assert abs(sum(d.values()) - 1.0) < 1e-9

    # Populate via update_venue_stats_detailed.
    v.update_venue_stats_detailed('MCG', {
        'total_runs': 180, 'total_balls': 120,
        'boundaries': 12, 'dots': 30, 'wickets': 5,
        'powerplay_runs': 55, 'powerplay_balls': 36,
        'death_runs': 45, 'death_balls': 30,
        'is_first_innings': True,
        'c0': 30, 'c1': 50, 'c2': 20, 'c4': 10, 'c6': 5, 'cw': 5,
    })
    d = v.get_venue_outcome_dist('MCG', PRIOR, k=200.0)
    assert abs(sum(d.values()) - 1.0) < 1e-9


if __name__ == '__main__':
    test_shrink_empty_counts_returns_prior()
    test_shrink_heavy_data_approaches_mle()
    test_shrink_output_sums_to_one()
    test_shrink_half_weight_point()
    test_backend_shrink_matches_parsing_shrink()
    test_tracker_getter_dist_sums_to_one()
    test_venue_tracker_dist_sums_to_one()
    print("all shrinkage tests passed")
