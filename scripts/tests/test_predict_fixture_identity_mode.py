"""Serving-contract tests for I7 venue identity compatibility modes."""

from __future__ import annotations

import pickle
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import predict_fixture  # noqa: E402
from identity_maps import venue_alias_contract  # noqa: E402
from predict_fixture import (  # noqa: E402
    VENUE_IDENTITY_I7,
    VENUE_IDENTITY_LEGACY,
    _load_model_artifacts,
    read_sqlite_state_metadata,
    read_tracker_state_metadata,
    resolve_venue_identity,
)


def _write_sqlite(path: Path, metadata: dict[str, object]) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE dates (id INTEGER PRIMARY KEY, date TEXT)")
        conn.execute("INSERT INTO dates(date) VALUES ('2026-07-20')")
        conn.execute("CREATE TABLE _meta (key TEXT PRIMARY KEY, value TEXT)")
        conn.executemany(
            "INSERT INTO _meta(key, value) VALUES (?, ?)",
            [(key, str(value)) for key, value in metadata.items()],
        )


def _write_snapshot(path: Path, metadata: dict[str, object]) -> None:
    with path.open("wb") as handle:
        pickle.dump(
            {
                "as_of": "2026-07-20",
                "n_matches_walked": 10,
                **metadata,
            },
            handle,
        )


def test_legacy_preserves_alias_while_i7_canonicalizes_it():
    assert (
        resolve_venue_identity("  Kennington Oval  ", VENUE_IDENTITY_LEGACY)
        == "Kennington Oval"
    )
    assert (
        resolve_venue_identity("Kennington Oval", VENUE_IDENTITY_I7)
        == "Kennington Oval, London"
    )


def test_legacy_accepts_frozen_state_without_i7_metadata(tmp_path: Path):
    sqlite_path = tmp_path / "player_stats_cache_v3.sqlite"
    snapshot_path = tmp_path / "tracker.pkl"
    _write_sqlite(sqlite_path, {"source_match_count": 10})
    _write_snapshot(snapshot_path, {})

    sqlite_state = read_sqlite_state_metadata(
        tmp_path,
        identity_mode=VENUE_IDENTITY_LEGACY,
    )
    tracker_state = read_tracker_state_metadata(
        snapshot_path,
        identity_mode=VENUE_IDENTITY_LEGACY,
    )

    assert sqlite_state["venue_identity_mode"] == VENUE_IDENTITY_LEGACY
    assert tracker_state["venue_identity_mode"] == VENUE_IDENTITY_LEGACY


def test_i7_rejects_state_without_current_identity_contract(tmp_path: Path):
    sqlite_path = tmp_path / "player_stats_cache_v3.sqlite"
    snapshot_path = tmp_path / "tracker.pkl"
    _write_sqlite(sqlite_path, {"source_match_count": 10})
    _write_snapshot(snapshot_path, {})

    with pytest.raises(RuntimeError, match="venue-alias contract mismatch"):
        read_sqlite_state_metadata(
            tmp_path,
            identity_mode=VENUE_IDENTITY_I7,
        )
    with pytest.raises(RuntimeError, match="venue-alias contract mismatch"):
        read_tracker_state_metadata(
            snapshot_path,
            identity_mode=VENUE_IDENTITY_I7,
        )


def test_declared_snapshot_mode_cannot_be_mixed(tmp_path: Path):
    snapshot_path = tmp_path / "tracker.pkl"
    _write_snapshot(
        snapshot_path,
        {
            "venue_identity_mode": VENUE_IDENTITY_I7,
            **venue_alias_contract(),
        },
    )

    with pytest.raises(RuntimeError, match="declares venue identity mode"):
        read_tracker_state_metadata(
            snapshot_path,
            identity_mode=VENUE_IDENTITY_LEGACY,
        )


def test_legacy_model_loader_allows_missing_identity_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    (tmp_path / "feature_columns.txt").write_text("feature_a\n")
    monkeypatch.setattr(
        predict_fixture.joblib,
        "load",
        lambda path: Path(path).name,
    )
    _load_model_artifacts.cache_clear()

    model, encoders, columns = _load_model_artifacts(
        str(tmp_path),
        VENUE_IDENTITY_LEGACY,
    )

    assert model == "model.pkl"
    assert encoders == "encoders.pkl"
    assert columns == ["feature_a"]


def test_i7_model_loader_requires_identity_file(tmp_path: Path):
    _load_model_artifacts.cache_clear()
    with pytest.raises(RuntimeError, match="venue_identity.json is missing"):
        _load_model_artifacts(str(tmp_path), VENUE_IDENTITY_I7)


def test_unknown_mode_fails_closed():
    with pytest.raises(ValueError, match="unsupported venue identity mode"):
        resolve_venue_identity("Kennington Oval", "experimental")
