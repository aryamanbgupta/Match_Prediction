"""Tiny end-to-end schema-v5 cache build smoke test."""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from build_stats_cache import build  # noqa: E402
from stats_provider import StatsProvider  # noqa: E402
from stats_sqlite_backend import I8_SCHEMA_VERSION  # noqa: E402
from tracker_rehydration import rehydrate_stats_tracker  # noqa: E402


def _match(date: str) -> dict:
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
                                    "batter": 1,
                                    "extras": 0,
                                    "total": 1,
                                },
                            }
                        ],
                    }
                ],
            }
        ],
    }


def test_schema_v5_builder_persists_phase_and_h2h_counts(tmp_path: Path):
    source = tmp_path / "json"
    source.mkdir()
    (source / "100.json").write_text(json.dumps(_match("2026-01-01")))
    (source / "200.json").write_text(json.dumps(_match("2026-01-02")))
    output = tmp_path / "player_stats_cache_i8.sqlite"

    build(
        source,
        output,
        gender="male",
        metadata_csv=ROOT / "data" / "all_players_enriched.csv",
        schema_version=I8_SCHEMA_VERSION,
    )

    with sqlite3.connect(output) as conn:
        meta = dict(conn.execute("SELECT key, value FROM _meta"))
        h2h = conn.execute(
            "SELECT balls,c0,c1,c2,c4,c6,cw FROM h2h "
            "ORDER BY date_id DESC LIMIT 1"
        ).fetchone()
        batter_phase = conn.execute(
            "SELECT phase,c0,c1,c2,c4,c6,cw FROM batting_phase "
            "ORDER BY date_id DESC LIMIT 1"
        ).fetchone()
        bowler_phase = conn.execute(
            "SELECT phase,c0,c1,c2,c4,c6,cw FROM bowling_phase "
            "ORDER BY date_id DESC LIMIT 1"
        ).fetchone()

    assert meta["schema_version"] == "5"
    assert meta["features"] == "v5"
    assert h2h == (1, 0, 1, 0, 0, 0, 0)
    assert batter_phase == (0, 0, 1, 0, 0, 0, 0)
    assert bowler_phase == (0, 0, 1, 0, 0, 0, 0)

    provider = StatsProvider(
        str(tmp_path),
        version="i8",
        required_schema_version=I8_SCHEMA_VERSION,
    )
    tracker = rehydrate_stats_tracker(
        provider,
        "2026-01-02",
        {"p_alice", "p_cara"},
    )
    assert tracker.enable_i8 is True
    assert tracker.batting_phase["p_alice"]["powerplay"]["c1"] == 1
    assert tracker.bowling_phase["p_cara"]["powerplay"]["c1"] == 1
    assert tracker.h2h_stats[("p_alice", "p_cara")]["c1"] == 1
