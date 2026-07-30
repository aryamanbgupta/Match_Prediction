"""I9 provisional-ELO schedule, state, and provenance regression tests."""
from __future__ import annotations

import json
import math
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from build_stats_cache import build  # noqa: E402
from elo_update import (  # noqa: E402
    BASELINE_ELO_UPDATE_VERSION,
    PROVISIONAL_ELO_UPDATE_VERSION,
)
from parsing_v2 import (  # noqa: E402
    PlayerEloTracker,
    PlayerStatsTracker,
    deep_copy_stats,
)
from stats_provider import StatsProvider  # noqa: E402
from tracker_rehydration import rehydrate_elo_tracker  # noqa: E402


def _legacy_update(ratings, batter, bowler, runs, wicket, k):
    bat_elo = ratings[0].get(batter, 1500.0)
    bowl_elo = ratings[1].get(bowler, 1500.0)
    expected = 1.0 / (1.0 + 10 ** ((bowl_elo - bat_elo) / 400.0))
    actual = 0.0 if wicket else min(0.4 + runs * 0.1, 1.0)
    ratings[0][batter] = bat_elo + k * (actual - expected)
    ratings[1][bowler] = (
        bowl_elo + k * ((1 - actual) - (1 - expected))
    )


def _match(date: str, runs: int = 4) -> dict:
    return {
        "info": {
            "venue": "Test Ground",
            "dates": [date],
            "gender": "male",
            "teams": ["A", "B"],
            "registry": {
                "people": {
                    "Alice": "p_alice",
                    "Beth": "p_beth",
                    "Cara": "p_cara",
                    "Dana": "p_dana",
                }
            },
            "players": {
                "A": ["Alice", "Beth"],
                "B": ["Cara", "Dana"],
            },
            "toss": {"winner": "A", "decision": "bat"},
            "team_type": "club",
            "outcome": {"winner": "A"},
        },
        "innings": [
            {
                "team": "A",
                "overs": [
                    {
                        "over": 0,
                        "deliveries": [
                            {
                                "batter": "Alice",
                                "non_striker": "Beth",
                                "bowler": "Cara",
                                "runs": {
                                    "batter": runs,
                                    "extras": 0,
                                    "total": runs,
                                },
                            }
                        ],
                    }
                ],
            }
        ],
    }


def test_precommitted_k_schedule():
    tracker = PlayerEloTracker(PROVISIONAL_ELO_UPDATE_VERSION)
    assert tracker.effective_k_factor(1.0, 0) == 4.0
    assert tracker.effective_k_factor(1.0, 60) == 2.5
    assert tracker.effective_k_factor(1.0, 120) == 1.0
    assert tracker.effective_k_factor(1.0, 121) == 1.0


def test_role_exposures_are_independent():
    tracker = PlayerEloTracker(PROVISIONAL_ELO_UPDATE_VERSION)
    tracker.batting_exposure["debutant"] = 0
    tracker.bowling_exposure["established"] = 120
    tracker.update(
        "debutant",
        "established",
        runs=6,
        is_wicket=False,
        k_factor=1.0,
    )

    assert tracker.get_batting_elo("debutant") == 1502.0
    assert tracker.get_bowling_elo("established") == 1499.5
    assert tracker.get_batting_exposure("debutant") == 1
    assert tracker.get_bowling_exposure("established") == 121


def test_baseline_update_remains_bit_exact_and_has_no_exposure_state():
    tracker = PlayerEloTracker()
    expected = ({}, {})
    sequence = [
        ("a", "x", 4, False, 1.0),
        ("a", "x", 0, True, 2.0),
        ("b", "x", 6, False, 4.0),
    ]
    for batter, bowler, runs, wicket, k in sequence:
        tracker.update(batter, bowler, runs, wicket, k_factor=k)
        _legacy_update(
            expected,
            batter,
            bowler,
            runs,
            wicket,
            k,
        )

    assert tracker.batting_elo == expected[0]
    assert tracker.bowling_elo == expected[1]
    assert tracker.batting_exposure == {}
    assert tracker.bowling_exposure == {}


def test_provisional_snapshot_is_independent():
    tracker = PlayerEloTracker(PROVISIONAL_ELO_UPDATE_VERSION)
    tracker.update("a", "x", 4, False, k_factor=1.0)
    snapshot = deep_copy_stats(
        # deep_copy_stats requires a player-stats tracker as its first arg;
        # a minimal empty tracker keeps this test focused on ELO state.
        PlayerStatsTracker(),
        elo_tracker=tracker,
    )
    tracker.batting_exposure["a"] += 1

    assert snapshot["elo_update_version"] == PROVISIONAL_ELO_UPDATE_VERSION
    assert snapshot["batting_elo_exposure"]["a"] == 1
    assert snapshot["bowling_elo_exposure"]["x"] == 1


def test_cache_rehydration_matches_uninterrupted_same_day_state(
    tmp_path: Path,
):
    source = tmp_path / "json"
    source.mkdir()
    (source / "100.json").write_text(
        json.dumps(_match("2026-01-01", runs=4))
    )
    (source / "200.json").write_text(
        json.dumps(_match("2026-01-01", runs=4))
    )
    (source / "300.json").write_text(
        json.dumps(_match("2026-01-02", runs=1))
    )
    output = tmp_path / "player_stats_cache_i9.sqlite"

    build(
        source,
        output,
        gender="male",
        metadata_csv=ROOT / "data" / "all_players_enriched.csv",
        elo_update_version=PROVISIONAL_ELO_UPDATE_VERSION,
    )

    with sqlite3.connect(output) as conn:
        metadata = dict(conn.execute("SELECT key, value FROM _meta"))
    assert (
        metadata["elo_update_version"]
        == PROVISIONAL_ELO_UPDATE_VERSION
    )

    provider = StatsProvider(
        str(tmp_path),
        version="i9",
        required_elo_update_version=PROVISIONAL_ELO_UPDATE_VERSION,
    )
    rehydrated = rehydrate_elo_tracker(
        provider,
        "2026-01-02",
        {"p_alice", "p_cara"},
        elo_update_version=PROVISIONAL_ELO_UPDATE_VERSION,
    )
    uninterrupted = PlayerEloTracker(PROVISIONAL_ELO_UPDATE_VERSION)
    uninterrupted.update(
        "p_alice", "p_cara", 4, False, k_factor=1.0
    )
    uninterrupted.update(
        "p_alice", "p_cara", 4, False, k_factor=1.0
    )

    assert rehydrated.batting_exposure["p_alice"] == 2
    assert rehydrated.bowling_exposure["p_cara"] == 2
    assert math.isclose(
        rehydrated.batting_elo["p_alice"],
        uninterrupted.batting_elo["p_alice"],
        rel_tol=0.0,
        abs_tol=0.0,
    )
    assert math.isclose(
        rehydrated.bowling_elo["p_cara"],
        uninterrupted.bowling_elo["p_cara"],
        rel_tol=0.0,
        abs_tol=0.0,
    )

    with pytest.raises(RuntimeError, match="ELO update mismatch"):
        StatsProvider(
            str(tmp_path),
            version="i9",
            required_elo_update_version=BASELINE_ELO_UPDATE_VERSION,
        )


def test_exposure_parity_holds_on_extras_bearing_matches(tmp_path: Path):
    """Extras-bearing parity pin (2026-07-30): under legacy semantics BOTH
    the live update (fires on every delivery) and the rehydration seed (the
    ``balls`` counters, which the legacy stats path counts inclusive of
    extras) include wides — so exposure is consistent at 2 here, not 1.
    A 2026-07-30 review note claimed these two paths diverged on extras;
    writing this test refuted that claim, and it now pins the verified
    contract so the question stays settled."""
    match = _match("2026-01-01", runs=4)
    deliveries = match["innings"][0]["overs"][0]["deliveries"]
    deliveries.insert(0, {
        "batter": "Alice",
        "non_striker": "Beth",
        "bowler": "Cara",
        "runs": {"batter": 0, "extras": 1, "total": 1},
        "extras": {"wides": 1},
    })
    source = tmp_path / "json"
    source.mkdir()
    (source / "100.json").write_text(json.dumps(match))
    # A next-day match creates the 2026-01-02 snapshot (which reflects only
    # matches strictly before it, i.e. the wide-bearing match above).
    (source / "300.json").write_text(
        json.dumps(_match("2026-01-02", runs=1))
    )
    output = tmp_path / "player_stats_cache_i9.sqlite"
    build(
        source,
        output,
        gender="male",
        metadata_csv=ROOT / "data" / "all_players_enriched.csv",
        elo_update_version=PROVISIONAL_ELO_UPDATE_VERSION,
    )

    provider = StatsProvider(
        str(tmp_path),
        version="i9",
        required_elo_update_version=PROVISIONAL_ELO_UPDATE_VERSION,
    )
    rehydrated = rehydrate_elo_tracker(
        provider,
        "2026-01-02",
        {"p_alice", "p_cara"},
        elo_update_version=PROVISIONAL_ELO_UPDATE_VERSION,
    )

    # Live chronological pass under legacy semantics: the wide (team run 1)
    # updates ratings AND exposure, exactly like the rehydration counters.
    uninterrupted = PlayerEloTracker(PROVISIONAL_ELO_UPDATE_VERSION)
    uninterrupted.update("p_alice", "p_cara", 1, False, k_factor=1.0)
    uninterrupted.update("p_alice", "p_cara", 4, False, k_factor=1.0)

    assert uninterrupted.batting_exposure["p_alice"] == 2
    assert uninterrupted.bowling_exposure["p_cara"] == 2
    assert rehydrated.batting_exposure["p_alice"] == 2
    assert rehydrated.bowling_exposure["p_cara"] == 2
    assert math.isclose(
        rehydrated.batting_elo["p_alice"],
        uninterrupted.batting_elo["p_alice"],
        rel_tol=0.0,
        abs_tol=0.0,
    )
    assert math.isclose(
        rehydrated.bowling_elo["p_cara"],
        uninterrupted.bowling_elo["p_cara"],
        rel_tol=0.0,
        abs_tol=0.0,
    )
