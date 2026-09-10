"""`get_t1_outcome_counts` must be a pure read on replay tracker state.

The accessor used to index the tracker's defaultdicts directly, so looking up
an unseen player INSERTED a zero row into state that `advance_match`
deep-copies for transactional rollback — a read that writes. Numerically
harmless, but the replay provider's state must only move through its
begin/advance lifecycle.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

from parsing_v2 import PlayerStatsTracker  # noqa: E402
from sim_eval.same_day_stats import _TrackerStatsView  # noqa: E402


def _view() -> _TrackerStatsView:
    view = _TrackerStatsView(base_provider=None, player_metadata=None)
    view._stats = PlayerStatsTracker()
    view._current_date = "2026-01-01"
    return view


def test_unknown_player_reads_zero_without_inserting():
    view = _view()
    for kind, cell in (("batter", None), ("bowler", None),
                       ("batter_type", 0), ("bowler_hand", 1)):
        counts = view.get_t1_outcome_counts(
            kind, "ghost", "2026-01-01", cell=cell)
        assert counts == (0, 0, 0, 0, 0, 0)
    assert "ghost" not in view._stats.batting_stats
    assert "ghost" not in view._stats.bowling_stats
    assert "ghost" not in view._stats.batting_vs_type
    assert "ghost" not in view._stats.bowling_vs_hand


def test_vs_cell_kinds_require_a_cell():
    view = _view()
    with pytest.raises(ValueError, match="requires a cell"):
        view.get_t1_outcome_counts("batter_type", "ghost", "2026-01-01")


def test_unknown_kind_raises():
    view = _view()
    with pytest.raises(ValueError, match="unknown T1 count kind"):
        view.get_t1_outcome_counts("venue", "ghost", "2026-01-01")
