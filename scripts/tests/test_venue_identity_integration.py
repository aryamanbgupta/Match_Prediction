"""Cross-pipeline checks for the active I7 venue identity contract."""
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from build_polymarket_odds import build_match_id  # noqa: E402
from identity_maps import load_venue_aliases  # noqa: E402
from loaders_common import extract_match_metadata  # noqa: E402
from score_forward_ball_v7 import pre_match_spec  # noqa: E402
from sim_eval.eval_statistics import match_id_from_info  # noqa: E402


ALIAS = "Bay Oval"
CANONICAL = "Bay Oval, Mount Maunganui"
CANONICAL_ID = (
    "2026-01-01_New_Zealand_India_Bay_Oval,_Mount_Maunganui"
)


def _info() -> dict:
    teams = ["New Zealand", "India"]
    names = {
        team: [f"{team} Player {index}" for index in range(11)]
        for team in teams
    }
    registry = {
        name: f"{team}_{index}"
        for team, roster in names.items()
        for index, name in enumerate(roster)
    }
    return {
        "dates": ["2026-01-01"],
        "teams": teams,
        "venue": ALIAS,
        "team_type": "international",
        "event": {"name": "Synthetic"},
        "toss": {"winner": teams[0], "decision": "bat"},
        "registry": {"people": registry},
        "players": names,
    }


def test_production_map_contains_reviewed_rows() -> None:
    aliases = load_venue_aliases()
    assert len(aliases) == 94
    assert aliases[ALIAS] == CANONICAL


def test_cache_and_ball_metadata_use_same_canonical_venue() -> None:
    info = _info()
    assert extract_match_metadata({"info": info})["venue"] == CANONICAL
    assert pre_match_spec({"info": info})["venue"] == CANONICAL


def test_polymarket_and_eval_build_the_same_canonical_id() -> None:
    info = _info()
    assert match_id_from_info(info) == CANONICAL_ID
    assert build_match_id(
        "2026-01-01",
        "New Zealand",
        "India",
        ALIAS,
    ) == CANONICAL_ID
