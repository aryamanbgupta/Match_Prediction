"""Pins the cache contract that keeps same-day replay serving honest (BR1,
2026-08-14 review — investigated and found SAFE BY DESIGN; these tests keep
it that way).

Three interlocking facts prevent stale venue features across same-day
sibling matches:
  1. SameDayReplayStatsProvider subclasses StatsProviderCache, so
     wrap_with_cache (called by TransformerT1SimModel.__init__) is the
     IDENTITY for it — no second, never-invalidated memo layer appears.
  2. Its begin_date/advance_match call clear_memo() after every state
     mutation.
  3. clear_memo() actually clears the venue-dist memo.
If any of these breaks, the second fixture at the same ground on the same
day would be served the first fixture's pre-match venue distribution.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from sim_eval.same_day_stats import SameDayReplayStatsProvider  # noqa: E402
from stats_provider import StatsProviderCache, wrap_with_cache  # noqa: E402


class _StubProvider:
    """Counts venue-dist fetches so memo behavior is observable."""

    def __init__(self):
        self.calls = 0

    def get_venue_outcome_dist(self, venue, as_of_date, k=200.0):
        self.calls += 1
        return {"venue_p0": 0.3, "fetch_serial": self.calls}


def test_replay_provider_is_a_cache_so_wrap_is_identity():
    assert issubclass(SameDayReplayStatsProvider, StatsProviderCache)
    cache = StatsProviderCache(_StubProvider())
    assert wrap_with_cache(cache) is cache, \
        "wrap_with_cache must not re-wrap an existing cache subclass"


def test_clear_memo_invalidates_the_venue_dist():
    stub = _StubProvider()
    cache = StatsProviderCache(stub)
    first = cache.get_venue_outcome_dist("Eden Gardens", "2026-05-01")
    again = cache.get_venue_outcome_dist("Eden Gardens", "2026-05-01")
    assert stub.calls == 1 and first == again, "same-key call must memoize"
    cache.clear_memo()
    refreshed = cache.get_venue_outcome_dist("Eden Gardens", "2026-05-01")
    assert stub.calls == 2, \
        "clear_memo must invalidate the venue-dist memo — a same-day " \
        "sibling would otherwise see the previous match's venue state"
    assert refreshed["fetch_serial"] == 2


def test_replay_provider_mutations_clear_the_memo():
    """begin_date and advance_match must route through clear_memo (source
    contract; checked structurally so this test needs no SQLite cache)."""
    import inspect

    for method in ("begin_date", "advance_match"):
        source = inspect.getsource(
            getattr(SameDayReplayStatsProvider, method))
        assert "clear_memo" in source, \
            f"SameDayReplayStatsProvider.{method} no longer invalidates " \
            "the memo after mutating state"
