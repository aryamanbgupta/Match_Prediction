"""Operational-contract tests for live fixture prediction."""

from __future__ import annotations

import json
import pickle
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from predict_fixture import (  # noqa: E402
    _resolve_player_ids,
    clear_name_lookup_cache,
    assess_state_freshness,
    build_tracker_snapshot,
    compute_bet,
    load_trackers,
    read_sqlite_state_metadata,
    read_tracker_state_metadata,
)
from identity_maps import venue_alias_contract  # noqa: E402


def _write_sqlite(path: Path, as_of: str, source_count: int) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE dates (id INTEGER PRIMARY KEY, date TEXT)")
        conn.execute("INSERT INTO dates(date) VALUES (?)", (as_of,))
        conn.execute("CREATE TABLE _meta (key TEXT PRIMARY KEY, value TEXT)")
        rows = [
            ("source_match_count", str(source_count)),
            ("build_timestamp", "2026-07-20T00:00:00Z"),
            (
                "same_day_order_version",
                "date_then_match_id_lexicographic_v1",
            ),
        ]
        rows.extend(
            (key, str(value)) for key, value in venue_alias_contract().items()
        )
        conn.executemany(
            "INSERT INTO _meta(key, value) VALUES (?, ?)",
            rows,
        )


def _write_snapshot(path: Path, as_of: str, source_count: int) -> None:
    with path.open("wb") as handle:
        pickle.dump(
            {
                "as_of": as_of,
                "n_matches_walked": source_count,
                "built_at": "2026-07-20T00:00:00Z",
                "source_dirs": ["/sealed/source"],
                "same_day_order_version": (
                    "date_then_match_id_lexicographic_v1"
                ),
                **venue_alias_contract(),
            },
            handle,
        )


def test_state_metadata_and_freshness_use_older_component(tmp_path: Path):
    sqlite_path = tmp_path / "player_stats_cache_v3.sqlite"
    snapshot_path = tmp_path / "tracker.pkl"
    _write_sqlite(sqlite_path, "2026-07-20", 100)
    _write_snapshot(snapshot_path, "2026-07-18", 100)

    sqlite_state = read_sqlite_state_metadata(tmp_path)
    tracker_state = read_tracker_state_metadata(snapshot_path)
    assessment = assess_state_freshness(
        "2026-07-24",
        sqlite_state,
        tracker_state,
        max_state_age_days=7,
    )

    assert assessment["status"] == "fresh"
    assert assessment["state_available_through"] == "2026-07-18"
    assert assessment["query_as_of"] == "2026-07-24"
    assert assessment["age_days"] == 6
    assert assessment["sqlite"]["age_days"] == 4
    assert assessment["tracker"]["age_days"] == 6


def test_state_older_than_budget_is_stale():
    assessment = assess_state_freshness(
        "2026-07-24",
        {"as_of": "2026-07-01", "source_match_count": 10},
        {"as_of": "2026-07-01", "source_match_count": 10},
        max_state_age_days=14,
    )
    assert assessment["status"] == "stale"
    assert assessment["age_days"] == 23


def test_state_newer_than_fixture_is_filtered_not_stale():
    assessment = assess_state_freshness(
        "2026-07-10",
        {"as_of": "2026-07-20", "source_match_count": 10},
        {"as_of": "2026-07-20", "source_match_count": 10},
        max_state_age_days=0,
    )
    assert assessment["status"] == "fresh"
    assert assessment["age_days"] == 0


def test_sqlite_tracker_source_count_mismatch_fails_closed():
    with pytest.raises(RuntimeError, match="source-count mismatch"):
        assess_state_freshness(
            "2026-07-24",
            {"as_of": "2026-07-20", "source_match_count": 100},
            {"as_of": "2026-07-20", "source_match_count": 99},
        )


def test_negative_age_budget_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        assess_state_freshness(
            "2026-07-24",
            {"as_of": "2026-07-20", "source_match_count": 100},
            {"as_of": "2026-07-20", "source_match_count": 100},
            max_state_age_days=-1,
        )


def test_a7_close_fixture_uses_strictly_positive_edge_and_normalized_market():
    decision = compute_bet(
        "A",
        "B",
        0.55,
        {"A": 2.10, "B": 1.80},
        top6_batting_elo_diff=5.0,
        polymarket_volume_usd=50_000,
    )
    inverse_sum = 1 / 2.10 + 1 / 1.80
    assert decision["market_implied_prob"]["A"] == pytest.approx(
        (1 / 2.10) / inverse_sum
    )
    assert decision["elo_regime"] == "close"
    assert decision["edge_threshold_pp"] == 0.0
    assert decision["shadow_bet_placed"] is True
    assert decision["shadow_bet_team"] == "A"
    assert decision["execution_authorized"] is False
    assert decision["bet_team"] is None


@pytest.mark.parametrize(
    ("model_probability", "expected_placed"),
    [
        (0.60, False),
        (0.600001, True),
    ],
)
def test_a7_mismatch_requires_edge_strictly_above_ten_points(
    model_probability: float,
    expected_placed: bool,
):
    decision = compute_bet(
        "A",
        "B",
        model_probability,
        {"A": 2.0, "B": 2.0},
        top6_batting_elo_diff=5.000001,
        polymarket_volume_usd=50_000,
    )
    assert decision["elo_regime"] == "mismatch"
    assert decision["edge_threshold_pp"] == 10.0
    assert decision["shadow_bet_placed"] is expected_placed


def test_a7_requires_primary_liquidity_and_fresh_state():
    low_volume = compute_bet(
        "A",
        "B",
        0.65,
        {"A": 2.0, "B": 2.0},
        top6_batting_elo_diff=2.0,
        polymarket_volume_usd=49_999.99,
    )
    assert low_volume["shadow_bet_placed"] is False
    assert low_volume["suppression_reasons"] == [
        "below_minimum_liquidity"
    ]

    stale = compute_bet(
        "A",
        "B",
        0.65,
        {"A": 2.0, "B": 2.0},
        top6_batting_elo_diff=2.0,
        polymarket_volume_usd=50_000,
        state_eligible=False,
    )
    assert stale["shadow_bet_placed"] is False
    assert stale["suppression_reasons"] == ["state_not_fresh"]


def test_a7_missing_volume_or_elo_suppresses_shadow_candidate():
    decision = compute_bet(
        "A",
        "B",
        0.65,
        {"A": 2.0, "B": 2.0},
    )
    assert decision["shadow_bet_placed"] is False
    assert decision["suppression_reasons"] == [
        "missing_top6_batting_elo_diff",
        "missing_liquidity",
    ]


def _match_json(match_id: str, date: str, teams: list[str], winner: str,
                venue: str = "Kennington Oval, London") -> str:
    return json.dumps({
        "info": {
            "dates": [date],
            "gender": "male",
            "teams": teams,
            "venue": venue,
            "outcome": {"winner": winner},
        }
    })


def test_auxiliary_pool_feeds_trackers_without_inflating_state_count(
    tmp_path: Path,
):
    """A non-T20 competition can populate the trackers, but the SQLite /
    tracker source-count agreement check must still describe the primary
    pool only — otherwise stale T20 state hides behind fresh Hundred results.
    """
    primary = tmp_path / "primary"
    aux = tmp_path / "aux"
    primary.mkdir()
    aux.mkdir()
    (primary / "1.json").write_text(
        _match_json("1", "2026-06-01", ["Alpha", "Beta"], "Alpha"))
    (primary / "2.json").write_text(
        _match_json("2", "2026-06-02", ["Alpha", "Beta"], "Beta"))
    (aux / "9001.json").write_text(
        _match_json("9001", "2026-07-20", ["Gamma", "Delta"], "Gamma"))

    snapshot_path = tmp_path / "snapshot.pkl"
    snapshot = build_tracker_snapshot(primary, snapshot_path, aux)

    assert snapshot["n_matches_walked"] == 2
    assert snapshot["n_aux_matches_walked"] == 1
    assert snapshot["as_of"] == "2026-06-02"       # primary coverage
    assert snapshot["aux_as_of"] == "2026-07-20"   # aux does not mask it

    state = read_tracker_state_metadata(snapshot_path)
    assert state["source_match_count"] == 2
    assert state["aux_source_match_count"] == 1

    # The auxiliary result is still in the trackers.
    form, _, _ = load_trackers(snapshot_path, primary, aux)
    rate, n = form.get_last_n_win_rate("Gamma", datetime(2026, 7, 25))
    assert (rate, n) == (1.0, 1)


def test_team_aliases_fold_renamed_franchise_history(tmp_path: Path):
    source = tmp_path / "pool"
    source.mkdir()
    (source / "1.json").write_text(
        _match_json("1", "2026-06-01", ["Old Name", "Rival"], "Old Name"))
    (source / "2.json").write_text(
        _match_json("2", "2026-06-02", ["Old Name", "Rival"], "Rival"))
    snapshot_path = tmp_path / "snapshot.pkl"
    build_tracker_snapshot(source, snapshot_path)

    aliases = {"Old Name": "New Name"}
    form, h2h, home = load_trackers(snapshot_path, source,
                                    team_aliases=aliases)
    as_of = datetime(2026, 6, 10)
    assert form.get_last_n_win_rate("New Name", as_of) == (0.5, 2)
    assert form.get_last_n_win_rate("Old Name", as_of) == (0.5, 0)
    rate, meetings = h2h.get_h2h("New Name", "Rival", as_of)
    assert meetings == 2 and rate == pytest.approx(0.5)
    assert home.records[("New Name", "Kennington Oval, London")]


@pytest.fixture(autouse=True)
def _reset_name_lookup():
    clear_name_lookup_cache()
    yield
    clear_name_lookup_cache()


def test_unresolvable_lineup_fails_closed():
    """An unresolved player silently defaults their ELO and career stats.
    One or two is a warning; a systematically mis-spelled XI must raise.
    """
    metadata = SimpleNamespace(players={"aaaaaaaa": {"name": "Real Player"}})
    lineup = [f"Not A Player {i}" for i in range(5)]
    with pytest.raises(ValueError, match="did not resolve"):
        _resolve_player_ids(lineup, metadata, side="team1 lineup")

    tolerated = _resolve_player_ids(
        ["aaaaaaaa", "Unknown Debutant"], metadata, side="team1 lineup")
    assert tolerated == ["aaaaaaaa", "Unknown Debutant"]


def test_name_lookup_accepts_cricsheet_initials_form():
    metadata = SimpleNamespace(players={
        "9caf69a1": {"name": "Will Jacks", "unique_name": "WG Jacks",
                     "full_name": "William George Jacks"},
    })
    for spelling in ("Will Jacks", "WG Jacks", "William George Jacks",
                     "wg jacks"):
        assert _resolve_player_ids([spelling], metadata) == ["9caf69a1"]


def test_aux_snapshot_without_declared_mode_fails_closed(tmp_path: Path):
    """Aux pools are new-competition state: undeclared provenance must not
    load (the pre-restamp Hundred snapshot shape)."""
    snapshot_path = tmp_path / "snapshot.pkl"
    with snapshot_path.open("wb") as handle:
        pickle.dump(
            {
                "as_of": "2026-07-20",
                "n_matches_walked": 2,
                "n_aux_matches_walked": 1,
                "aux_source_dirs": ["/aux/hundred"],
                "same_day_order_version": (
                    "date_then_match_id_lexicographic_v1"
                ),
                **venue_alias_contract(),
            },
            handle,
        )
    with pytest.raises(RuntimeError, match="declares no venue identity mode"):
        read_tracker_state_metadata(snapshot_path)


def test_legacy_mode_refuses_aux_snapshot(tmp_path: Path):
    snapshot_path = tmp_path / "snapshot.pkl"
    with snapshot_path.open("wb") as handle:
        pickle.dump(
            {
                "as_of": "2026-07-20",
                "n_matches_walked": 2,
                "n_aux_matches_walked": 1,
                "aux_source_dirs": ["/aux/hundred"],
                "venue_identity_mode": "i7",
                "same_day_order_version": (
                    "date_then_match_id_lexicographic_v1"
                ),
                **venue_alias_contract(),
            },
            handle,
        )
    with pytest.raises(RuntimeError,
                       match="must not acquire new-competition state"):
        read_tracker_state_metadata(snapshot_path, identity_mode="legacy")


def test_legacy_mode_refuses_identity_declaring_model(tmp_path: Path):
    from predict_fixture import _load_model_artifacts

    (tmp_path / "venue_identity.json").write_text("{}")
    with pytest.raises(RuntimeError, match="legacy mode"):
        _load_model_artifacts(str(tmp_path), "legacy")


def test_a7_out_of_scope_competition_is_suppressed():
    """A qualified fixture served outside the predeclared A7 universe (aux
    pools / aliases / non-T20 format) must never record a shadow bet."""
    out_of_scope = compute_bet(
        "Alpha", "Beta", 0.75,
        {"Alpha": 2.0, "Beta": 2.0},
        top6_batting_elo_diff=0.0,
        polymarket_volume_usd=100_000.0,
        state_eligible=True,
        policy_scope_eligible=False,
    )
    assert out_of_scope["shadow_bet_placed"] is False
    assert out_of_scope["policy_scope_eligible"] is False
    assert ("competition_out_of_policy_scope"
            in out_of_scope["suppression_reasons"])

    in_scope = compute_bet(
        "Alpha", "Beta", 0.75,
        {"Alpha": 2.0, "Beta": 2.0},
        top6_batting_elo_diff=0.0,
        polymarket_volume_usd=100_000.0,
        state_eligible=True,
    )
    assert in_scope["shadow_bet_placed"] is True
