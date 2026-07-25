"""I6 regression tests for deterministic same-day match ordering."""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import build_stats_cache as cache_builder  # noqa: E402
import run_experiment  # noqa: E402
from loaders_common import (  # noqa: E402
    SAME_DAY_ORDER_VERSION,
    iter_matches_chronological,
    iter_matches_chronological_multi,
)
from stats_sqlite_backend import SCHEMA_VERSION  # noqa: E402
from materialize_match_features import resolved_match_winner  # noqa: E402


def _write_match(
    directory: Path,
    match_id: str,
    date: str,
    gender: str = "male",
) -> None:
    payload = {
        "info": {
            "dates": [date],
            "gender": gender,
        }
    }
    (directory / f"{match_id}.json").write_text(json.dumps(payload))


def _ids(iterator) -> list[str]:
    return [match_id for match_id, _text, _date in iterator]


def test_single_source_order_does_not_depend_on_creation_order(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()

    # Same logical corpus, deliberately created in opposite orders.
    rows = [
        ("300", "2026-06-02"),
        ("100", "2026-06-01"),
        ("200", "2026-06-01"),
    ]
    for match_id, date in rows:
        _write_match(left, match_id, date)
    for match_id, date in reversed(rows):
        _write_match(right, match_id, date)

    expected = ["100", "200", "300"]
    assert _ids(iter_matches_chronological(left)) == expected
    assert _ids(iter_matches_chronological(right)) == expected


def test_multi_source_order_is_global_date_then_match_id(tmp_path):
    base = tmp_path / "base"
    extra = tmp_path / "extra"
    base.mkdir()
    extra.mkdir()
    _write_match(base, "300", "2026-06-02")
    _write_match(base, "200", "2026-06-01")
    _write_match(extra, "100", "2026-06-01")
    _write_match(extra, "400", "2026-06-03")

    assert _ids(iter_matches_chronological_multi([base, extra])) == [
        "100",
        "200",
        "300",
        "400",
    ]
    assert SAME_DAY_ORDER_VERSION == "date_then_match_id_lexicographic_v1"


def test_multi_source_duplicate_match_id_fails_closed(tmp_path):
    base = tmp_path / "base"
    extra = tmp_path / "extra"
    base.mkdir()
    extra.mkdir()
    _write_match(base, "100", "2026-06-01")
    _write_match(extra, "100", "2026-06-02")

    try:
        list(iter_matches_chronological_multi([base, extra]))
    except RuntimeError as exc:
        assert "duplicate match ID" in str(exc)
    else:
        raise AssertionError("duplicate match ID was silently replayed")


def test_gender_filter_precedes_global_ordering(tmp_path):
    _write_match(tmp_path, "100", "2026-06-01", gender="female")
    _write_match(tmp_path, "200", "2026-06-01", gender="male")
    assert _ids(iter_matches_chronological(tmp_path, gender="male")) == ["200"]


def _write_meta_cache(
    db_path: Path,
    source_dirs: list[Path],
    source_mtime: float,
    source_count: int,
    order_version: str | None = SAME_DAY_ORDER_VERSION,
) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE _meta (key TEXT PRIMARY KEY, value TEXT)")
    rows = [
        ("schema_version", str(SCHEMA_VERSION)),
        ("source_dirs_json", cache_builder._source_paths_json(source_dirs)),
        ("source_json_mtime_max", str(source_mtime)),
        ("source_json_file_count", str(source_count)),
    ]
    rows.extend(
        (key, str(value))
        for key, value in cache_builder.venue_alias_contract().items()
    )
    if order_version is not None:
        rows.append(("same_day_order_version", order_version))
    conn.executemany("INSERT INTO _meta VALUES (?, ?)", rows)
    conn.commit()
    conn.close()


def test_cache_staleness_requires_order_contract_and_all_sources(tmp_path):
    base = tmp_path / "base"
    extra = tmp_path / "extra"
    base.mkdir()
    extra.mkdir()
    _write_match(base, "100", "2026-06-01")
    _write_match(extra, "200", "2026-06-02")
    source_files = [base / "100.json", extra / "200.json"]
    max_mtime = max(path.stat().st_mtime for path in source_files)

    current_db = tmp_path / "current.sqlite"
    _write_meta_cache(current_db, [base, extra], max_mtime, 2)
    assert cache_builder.sqlite_up_to_date(current_db, [base, extra])
    assert not cache_builder.sqlite_up_to_date(current_db, [base])

    legacy_db = tmp_path / "legacy.sqlite"
    _write_meta_cache(
        legacy_db,
        [base, extra],
        max_mtime,
        2,
        order_version=None,
    )
    assert not cache_builder.sqlite_up_to_date(legacy_db, [base, extra])

    os.utime(extra / "200.json", (max_mtime + 10, max_mtime + 10))
    assert not cache_builder.sqlite_up_to_date(current_db, [base, extra])


def test_invalid_sqlite_always_invalidates_downstream_parquet(monkeypatch):
    monkeypatch.setattr(
        run_experiment,
        "_check_sqlite_cache",
        lambda _config: False,
    )
    monkeypatch.setattr(
        run_experiment,
        "_check_parquet_cache",
        lambda _config, _features: True,
    )
    assert run_experiment.check_smart_cache({}, []) == (False, False)


def test_match_winner_uses_eliminator_for_resolved_tie():
    info = {
        "teams": ["Sweden", "Portugal"],
        "outcome": {"result": "tie", "eliminator": "Portugal"},
    }
    assert resolved_match_winner(info) == "Portugal"
    assert resolved_match_winner(
        {
            "teams": ["A", "B"],
            "outcome": {"winner": "A"},
        }
    ) == "A"
    assert resolved_match_winner(
        {
            "teams": ["A", "B"],
            "outcome": {"result": "no result"},
        }
    ) is None


def test_forward_cache_priors_are_frozen_from_pre_holdout_source(tmp_path):
    source = tmp_path / "source.sqlite"
    target = tmp_path / "target.sqlite"
    for path, offset in ((source, 100), (target, 200)):
        conn = sqlite3.connect(path)
        conn.execute("CREATE TABLE _meta (key TEXT PRIMARY KEY, value TEXT)")
        rows = [("schema_version", str(SCHEMA_VERSION))]
        rows.extend(
            (key, str(offset + index))
            for index, key in enumerate(cache_builder.PRIOR_META_KEYS)
        )
        conn.executemany("INSERT INTO _meta VALUES (?, ?)", rows)
        conn.commit()
        conn.close()

    provenance = cache_builder.freeze_priors_from_sqlite(target, source)
    conn = sqlite3.connect(target)
    meta = dict(conn.execute("SELECT key, value FROM _meta"))
    conn.close()

    for index, key in enumerate(cache_builder.PRIOR_META_KEYS):
        assert meta[key] == str(100 + index)
    assert meta["prior_contract"] == "frozen_external_sqlite_v1"
    assert meta["prior_source_sha256"] == provenance["prior_source_sha256"]
