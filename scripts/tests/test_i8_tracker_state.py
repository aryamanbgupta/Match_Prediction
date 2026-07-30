"""I8 in-memory tracker counting and snapshot contract tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from parsing_v2 import PlayerStatsTracker, deep_copy_stats  # noqa: E402


PRIOR = (0.4, 0.3, 0.1, 0.1, 0.05, 0.05)


def test_i8_tracker_counts_phase_and_h2h_after_delivery_only():
    tracker = PlayerStatsTracker(enable_i8=True)

    before = tracker.get_batter_phase_outcome_dist(
        "bat",
        PRIOR,
        balls_bowled=0,
    )
    tracker.update_stats(
        "bat",
        "bowl",
        4,
        False,
        phase="powerplay",
    )

    assert before == {
        "batter_phase_p0": 0.4,
        "batter_phase_p1": 0.3,
        "batter_phase_p2": 0.1,
        "batter_phase_p4": 0.1,
        "batter_phase_p6": 0.05,
        "batter_phase_pw": 0.05,
    }
    assert tracker.batting_phase["bat"]["powerplay"]["c4"] == 1
    assert tracker.bowling_phase["bowl"]["powerplay"]["c4"] == 1
    assert tracker.h2h_stats[("bat", "bowl")]["c4"] == 1
    assert sum(
        tracker.h2h_stats[("bat", "bowl")][key]
        for key in ("c0", "c1", "c2", "c4", "c6", "cw")
    ) == tracker.h2h_stats[("bat", "bowl")]["balls"]


def test_i8_illegal_delivery_does_not_enter_outcome_cells():
    tracker = PlayerStatsTracker(enable_i8=True)
    tracker.update_stats(
        "bat",
        "bowl",
        1,
        False,
        is_legal=False,
        phase="powerplay",
    )

    assert tracker.h2h_stats[("bat", "bowl")]["runs"] == 1
    assert tracker.h2h_stats[("bat", "bowl")]["balls"] == 0
    assert tracker.batting_phase == {}
    assert tracker.bowling_phase == {}


def test_i8_legal_delivery_requires_pre_ball_phase():
    tracker = PlayerStatsTracker(enable_i8=True)
    with pytest.raises(ValueError, match="valid pre-ball phase"):
        tracker.update_stats("bat", "bowl", 0, False)
    assert tracker.batting_stats == {}
    assert tracker.bowling_stats == {}
    assert tracker.h2h_stats == {}


def test_i8_snapshot_carries_independent_phase_and_h2h_counts():
    tracker = PlayerStatsTracker(enable_i8=True)
    tracker.update_stats(
        "bat",
        "bowl",
        0,
        True,
        is_bowler_wicket=False,
        phase="death",
    )

    snapshot = deep_copy_stats(tracker)
    assert snapshot["batting_phase"]["bat"]["death"]["cw"] == 1
    assert snapshot["bowling_phase"]["bowl"]["death"]["cw"] == 1
    assert snapshot["h2h"][("bat", "bowl")]["cw"] == 1
    # Scalar dismissals deliberately retain bowler-wicket semantics.
    assert snapshot["h2h"][("bat", "bowl")]["dismissals"] == 0

    tracker.batting_phase["bat"]["death"]["cw"] += 1
    assert snapshot["batting_phase"]["bat"]["death"]["cw"] == 1


def test_v4_tracker_has_no_i8_sparse_state_overhead():
    tracker = PlayerStatsTracker()
    tracker.update_stats("bat", "bowl", 1, False)

    assert tracker.enable_i8 is False
    assert tracker.batting_phase is None
    assert tracker.bowling_phase is None
    assert set(tracker.h2h_stats[("bat", "bowl")]) == {
        "runs",
        "balls",
        "dismissals",
    }
    with pytest.raises(RuntimeError, match="enable_i8=True"):
        tracker.get_h2h_outcome_dist("bat", "bowl", PRIOR)
