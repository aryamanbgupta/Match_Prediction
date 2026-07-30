"""Regression tests for I15 Cricsheet-primary match identity."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from match_identity import (  # noqa: E402
    MATCH_IDENTITY_VERSION,
    build_compatibility_alias_lookup,
    build_primary_lookup,
    build_unambiguous_alias_lookup,
    new_match_identity,
    resolve_match_identity,
)
from sim_eval.loaders import (  # noqa: E402
    BettingOddsLoader,
    TestMatchLoader as MatchLoader,
)
from sim_eval.reslice_eval_json import (  # noqa: E402
    _build_identity_lookup,
    _lookup_for_match,
)


def _row(cricsheet_id: str) -> dict[str, str]:
    return new_match_identity(
        cricsheet_id,
        date_text="2026-01-01",
        team1="Alpha",
        team2="Beta",
        venue="Test Ground",
    ).as_fields()


def test_doubleheaders_have_unique_primary_ids_and_shared_display_id():
    first = _row("1001")
    second = _row("1002")
    assert first["match_id"] == "1001"
    assert second["match_id"] == "1002"
    assert first["display_match_id"] == second["display_match_id"]
    assert first["match_identity_version"] == MATCH_IDENTITY_VERSION
    assert set(build_primary_lookup(
        [first, second],
        context="doubleheader",
    )) == {"1001", "1002"}


def test_legacy_alias_join_fails_closed_on_doubleheader():
    with pytest.raises(ValueError, match="ambiguous match alias"):
        build_unambiguous_alias_lookup(
            [_row("1001"), _row("1002")],
            context="doubleheader",
        )


def test_compatibility_index_allows_primary_join_with_unrelated_collision():
    lookup = build_compatibility_alias_lookup(
        [_row("1001"), _row("1002")],
        context="large prediction artifact",
    )
    assert lookup["1001"]["cricsheet_id"] == "1001"
    with pytest.raises(ValueError, match="ambiguous match alias"):
        _ = lookup[_row("1001")["display_match_id"]]


def test_new_contract_requires_primary_equal_cricsheet_id():
    row = _row("1001")
    row["match_id"] = "synthetic"
    with pytest.raises(ValueError, match="match_id == cricsheet_id"):
        resolve_match_identity(row)


def test_odds_loader_rejects_duplicate_primary_id(tmp_path: Path):
    first = _row("1001")
    second = {**_row("1001"), "display_match_id": "another-display"}
    path = tmp_path / "odds.json"
    path.write_text(json.dumps({"matches": [first, second]}))
    with pytest.raises(RuntimeError, match="duplicate odds primary"):
        BettingOddsLoader.load_odds(path)


def test_match_loader_uses_filename_stem_as_primary_id(tmp_path: Path):
    teams = ["Alpha", "Beta"]
    players = {
        team: [f"{team} {index}" for index in range(11)]
        for team in teams
    }
    registry = {
        name: f"{team}-{index}"
        for team, lineup in players.items()
        for index, name in enumerate(lineup)
    }
    payload = {
        "info": {
            "dates": ["2026-01-01"],
            "teams": teams,
            "venue": "Test Ground",
            "toss": {"winner": "Alpha", "decision": "bat"},
            "registry": {"people": registry},
            "players": players,
            "event": {"name": "Synthetic"},
            "team_type": "club",
        },
    }
    (tmp_path / "1001.json").write_text(json.dumps(payload))
    matches = MatchLoader().load_matches(tmp_path)
    assert len(matches) == 1
    match_id, state = matches[0]
    assert match_id == "1001"
    assert state.cricsheet_id == "1001"
    assert state.display_match_id.startswith("2026-01-01_Alpha_Beta_")


def test_compatibility_lookup_joins_new_eval_to_frozen_odds():
    frozen_odds = {
        "match_id": _row("1001")["display_match_id"],
        "polymarket_volume_usd": 50_000,
    }
    lookup = _build_identity_lookup(
        [frozen_odds],
        value_fn=lambda row: row["polymarket_volume_usd"],
        context="test odds",
    )
    assert _lookup_for_match(lookup, _row("1001")) == 50_000


def test_compatibility_lookup_refuses_ambiguous_display_alias():
    lookup = _build_identity_lookup(
        [_row("1001"), _row("1002")],
        value_fn=lambda row: row["cricsheet_id"],
        context="test features",
    )
    assert _lookup_for_match(lookup, {"match_id": "1001"}) == "1001"
    with pytest.raises(ValueError, match="ambiguous match alias"):
        _lookup_for_match(
            lookup,
            {"match_id": _row("1001")["display_match_id"]},
        )


def test_evaluator_refuses_one_odds_row_for_two_doubleheader_matches():
    from types import SimpleNamespace

    from sim_eval.match_evaluator import MatchLevelEvaluator

    display = _row("1001")["display_match_id"]
    odds_row = {"match_id": display, "odds": {"winner": {}}}
    lookup = {display: odds_row}
    claimed: dict[int, str] = {}
    first_state = SimpleNamespace(display_match_id=display)
    assert MatchLevelEvaluator._resolve_odds_row(
        "1001", first_state, lookup, claimed_rows=claimed) is odds_row
    # Re-resolving the same match is idempotent.
    assert MatchLevelEvaluator._resolve_odds_row(
        "1001", first_state, lookup, claimed_rows=claimed) is odds_row
    with pytest.raises(RuntimeError, match="maps to the same row"):
        MatchLevelEvaluator._resolve_odds_row(
            "1002",
            SimpleNamespace(display_match_id=display),
            lookup,
            claimed_rows=claimed,
        )


def _sim_row(cricsheet_id: str, display: str) -> dict:
    return {
        "match_id": cricsheet_id,
        "cricsheet_id": cricsheet_id,
        "display_match_id": display,
        "teams": ["Alpha", "Beta"],
        "match_date": "2026-01-01",
        "simulated_prob": {"Alpha": 0.6, "Beta": 0.4},
        "market_prob": {"Alpha": 0.5, "Beta": 0.5},
        "market_odds": {"Alpha": 2.0, "Beta": 2.0},
        "actual_winner": "Alpha",
    }


def test_blend_joins_frozen_direct_predictions_via_display_alias():
    from sim_eval.blend_eval_json import blend

    display = _row("1001")["display_match_id"]
    sim_json = {"matches": [_sim_row("1001", display)]}
    direct = {display: {"team1": "Alpha", "team2": "Beta",
                        "p_team1": 0.7, "p_team2": 0.3}}
    out = blend(sim_json, direct, w=0.0)
    assert out["summary"]["n_matches_blended"] == 1


def test_blend_fails_closed_when_doubleheader_shares_direct_row():
    from sim_eval.blend_eval_json import blend

    display = _row("1001")["display_match_id"]
    sim_json = {
        "matches": [_sim_row("1001", display), _sim_row("1002", display)],
    }
    direct = {display: {"team1": "Alpha", "team2": "Beta",
                        "p_team1": 0.7, "p_team2": 0.3}}
    with pytest.raises(RuntimeError, match="shared legacy display alias"):
        blend(sim_json, direct, w=0.0)


def test_cluster_lookup_fails_loudly_on_doubleheader_display_alias(
    tmp_path: Path,
):
    from sim_eval.eval_statistics import (
        AMBIGUOUS_CLUSTER_ALIAS,
        cluster_id_for_record,
        load_competition_clusters,
        match_id_from_info,
    )

    info = {
        "dates": ["2026-01-01"],
        "teams": ["Alpha", "Beta"],
        "venue": "Test Ground",
        "event": {"name": "Synthetic"},
    }
    for stem in ("1001", "1002"):
        (tmp_path / f"{stem}.json").write_text(json.dumps({"info": info}))
    lookup = load_competition_clusters(tmp_path)
    display = match_id_from_info(info)
    assert lookup["1001"] == lookup["1002"]
    assert lookup[display] == AMBIGUOUS_CLUSTER_ALIAS
    assert cluster_id_for_record({"match_id": "1001"}, lookup) == lookup["1001"]
    with pytest.raises(RuntimeError, match="doubleheader"):
        cluster_id_for_record({"match_id": display}, lookup)
