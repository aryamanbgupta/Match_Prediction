from __future__ import annotations

import json
import fcntl
import pickle
import sqlite3
from pathlib import Path

import pytest

from cricsheet_fixtures import match_json, write_metadata_csv
from daily.refresh_state import refresh_state


def _setup(tmp_path):
    data = tmp_path / "data"
    base = data / "t20s_json"
    base.mkdir(parents=True)
    (base / "1.json").write_text(match_json("2026-09-08", serialize=True))
    contexts = tmp_path / "daily" / "context"
    metadata = data / "players.csv"
    write_metadata_csv(metadata)
    prior = tmp_path / "prior.sqlite"
    prior.touch()
    return data, base, contexts, metadata, prior


def test_refresh_accumulates_contexts_filters_duplicates_and_promotes(tmp_path):
    data, base, contexts, metadata, prior = _setup(tmp_path)
    calls = []
    fetch_number = 0

    def runner(command, **kwargs):
        nonlocal fetch_number
        calls.append(command)
        if isinstance(command, str):
            fetch_number += 1
            incoming = Path(kwargs["env"]["CRICML_CONTEXT_DIR"])
            if fetch_number == 1:
                (incoming / "1.json").write_text(match_json("2026-09-08", serialize=True))
                (incoming / "2.json").write_text(match_json("2026-09-09", serialize=True))
            else:
                (incoming / "2.json").write_text(match_json("2026-09-09", serialize=True))
        elif "scripts/build_stats_cache.py" in command:
            sources = [Path(command[command.index("--source-dir") + 1])]
            sources.extend(Path(command[index + 1]) for index, value in enumerate(command)
                           if value == "--extra-source-dir")
            count = sum(len(list(source.glob("*.json"))) for source in sources)
            output = Path(command[command.index("--out") + 1])
            with sqlite3.connect(output) as conn:
                conn.execute("CREATE TABLE _meta (key TEXT PRIMARY KEY, value TEXT)")
                conn.execute("INSERT INTO _meta VALUES ('source_match_count', ?)", (str(count),))
        elif "scripts/predict_fixture.py" in command:
            sources = [Path(command[index + 1]) for index, value in enumerate(command)
                       if value == "--tracker-source-dir"]
            count = sum(len(list(source.glob("*.json"))) for source in sources)
            snapshot = Path(command[command.index("--tracker-snapshot") + 1])
            with snapshot.open("wb") as handle:
                pickle.dump({"n_matches_walked": count, "as_of": "2026-09-09"}, handle)

    leftover = contexts / ".tmp-2026-09-01-999"
    leftover.mkdir(parents=True)
    (leftover / "partial.json").write_text("{}")
    first = refresh_state(
        base_source=base, context_root=contexts, state_parent=data,
        stable_link=data / "live_state_i7", metadata_csv=metadata,
        prior_sqlite=prior, fetch_cmd="stub-fetch", context_date="2026-09-10",
        runner=runner,
    )
    second = refresh_state(
        base_source=base, context_root=contexts, state_parent=data,
        stable_link=data / "live_state_i7", metadata_csv=metadata,
        prior_sqlite=prior, fetch_cmd="stub-fetch", context_date="2026-09-11",
        runner=runner,
    )
    assert first != second
    assert not leftover.exists()
    assert sorted(path.name for path in (contexts / "2026-09-10").glob("*.json")) == ["2.json"]
    assert list((contexts / "2026-09-11").glob("*.json")) == []
    assert (second / "BUILT").read_text() == "sealed\n"
    assert (data / "live_state_i7").resolve() == second.resolve()
    state = json.loads((second / "state.json").read_text())
    assert state["state_as_of"] == "2026-09-09"
    assert state["context_dirs"] == [
        str(contexts / "2026-09-10"), str(contexts / "2026-09-11")
    ]
    second_cache = [call for call in calls if isinstance(call, list)
                    and "scripts/build_stats_cache.py" in call][-1]
    assert [second_cache[index + 1] for index, value in enumerate(second_cache)
            if value == "--extra-source-dir"] == state["context_dirs"]

    # Out-of-order publication dedupes against newer contexts too.
    third = refresh_state(
        base_source=base, context_root=contexts, state_parent=data,
        stable_link=data / "live_state_i7", metadata_csv=metadata,
        prior_sqlite=prior, fetch_cmd="stub-fetch", context_date="2026-09-09",
        runner=runner,
    )
    assert third != second
    assert list((contexts / "2026-09-09").glob("*.json")) == []


def test_refresh_path_guards_run_before_mutation(tmp_path):
    data, base, contexts, metadata, prior = _setup(tmp_path)
    common = dict(base_source=base, metadata_csv=metadata, prior_sqlite=prior,
                  fetch_cmd="stub", context_date="2026-09-10",
                  runner=lambda *args, **kwargs: pytest.fail("runner called"))
    with pytest.raises(ValueError, match="context root"):
        refresh_state(context_root=base / "context", state_parent=data,
                      stable_link=data / "live_state_i7", **common)
    with pytest.raises(ValueError, match="state parent must not"):
        refresh_state(context_root=contexts, state_parent=base / "states",
                      stable_link=base / "states" / "live_state_i7", **common)
    with pytest.raises(ValueError, match="state parent must be data"):
        refresh_state(context_root=contexts, state_parent=tmp_path / "elsewhere",
                      stable_link=tmp_path / "elsewhere" / "live_state_i7", **common)
    sealed = tmp_path / "sealed"
    sealed.mkdir()
    (sealed / "BUILT").write_text("sealed\n")
    with pytest.raises(RuntimeError, match="sealed build"):
        refresh_state(context_root=sealed / "context", state_parent=data,
                      stable_link=data / "live_state_i7", **common)
    for invalid in ("../../data/t20s_json/injected", "/tmp/escaped", "2026-9-10"):
        with pytest.raises(ValueError, match="context_date"):
            refresh_state(context_root=contexts, state_parent=data,
                          stable_link=data / "live_state_i7",
                          **{**common, "context_date": invalid})


def test_refresh_refuses_held_global_lock(tmp_path):
    data, base, contexts, metadata, prior = _setup(tmp_path)
    contexts.parent.mkdir(parents=True, exist_ok=True)
    lock = contexts.parent / "refresh.lock"
    with lock.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="already held"):
            refresh_state(base_source=base, context_root=contexts, state_parent=data,
                          stable_link=data / "live_state_i7", metadata_csv=metadata,
                          prior_sqlite=prior, fetch_cmd="stub", context_date="2026-09-10",
                          runner=lambda *args, **kwargs: pytest.fail("runner called"))


def test_interrupted_fetch_never_publishes_partial_context(tmp_path):
    data, base, contexts, metadata, prior = _setup(tmp_path)
    def fail_after_write(command, **kwargs):
        incoming = Path(kwargs["env"]["CRICML_CONTEXT_DIR"])
        (incoming / "partial.json").write_text("{}")
        raise RuntimeError("fetch interrupted")
    with pytest.raises(RuntimeError, match="fetch interrupted"):
        refresh_state(base_source=base, context_root=contexts, state_parent=data,
                      stable_link=data / "live_state_i7", metadata_csv=metadata,
                      prior_sqlite=prior, fetch_cmd="stub", context_date="2026-09-10",
                      runner=fail_after_write)
    assert not (contexts / "2026-09-10").exists()
    assert list(contexts.glob(".tmp-*")) == []


def test_stale_build_is_sealed_but_not_promoted(tmp_path, capsys):
    data, base, contexts, metadata, prior = _setup(tmp_path)
    current = data / "live_state_i7_current"
    current.mkdir()
    (current / "BUILT").write_text("sealed\n")
    (current / "state.json").write_text(json.dumps({
        "state_as_of": "2026-09-10", "source_match_count": 99,
    }))
    stable = data / "live_state_i7"
    stable.symlink_to(current.name, target_is_directory=True)

    def runner(command, **kwargs):
        if isinstance(command, str):
            incoming = Path(kwargs["env"]["CRICML_CONTEXT_DIR"])
            (incoming / "2.json").write_text(match_json("2026-09-09", serialize=True))
        elif "scripts/build_stats_cache.py" in command:
            output = Path(command[command.index("--out") + 1])
            with sqlite3.connect(output) as conn:
                conn.execute("CREATE TABLE _meta (key TEXT PRIMARY KEY, value TEXT)")
                conn.execute("INSERT INTO _meta VALUES ('source_match_count', '2')")
        else:
            snapshot = Path(command[command.index("--tracker-snapshot") + 1])
            with snapshot.open("wb") as handle:
                pickle.dump({"n_matches_walked": 2}, handle)

    build = refresh_state(
        base_source=base, context_root=contexts, state_parent=data,
        stable_link=stable, metadata_csv=metadata, prior_sqlite=prior,
        fetch_cmd="stub", context_date="2026-09-10", runner=runner,
    )
    assert stable.resolve() == current.resolve()
    assert (build / "BUILT").is_file()
    assert "stale build not promoted" in capsys.readouterr().err


def _runner_with_count(count: int, match_date: str = "2026-09-09"):
    def runner(command, **kwargs):
        if isinstance(command, str):
            incoming = Path(kwargs["env"]["CRICML_CONTEXT_DIR"])
            (incoming / "2.json").write_text(match_json(match_date, serialize=True))
        elif "scripts/build_stats_cache.py" in command:
            output = Path(command[command.index("--out") + 1])
            with sqlite3.connect(output) as conn:
                conn.execute("CREATE TABLE _meta (key TEXT PRIMARY KEY, value TEXT)")
                conn.execute(f"INSERT INTO _meta VALUES ('source_match_count', '{count}')")
        else:
            snapshot = Path(command[command.index("--tracker-snapshot") + 1])
            with snapshot.open("wb") as handle:
                pickle.dump({"n_matches_walked": count}, handle)
    return runner


def test_migrated_production_state_without_state_json_is_not_regressed(tmp_path, capsys):
    """The Mac mini dry day (2026-09-10): the current target carries only a bare
    BUILT and its cache _meta; a smaller, older build must NOT be promoted."""
    data, base, contexts, metadata, prior = _setup(tmp_path)
    current = data / "live_state_i7_2026-07-30_20260801T050221Z"
    current.mkdir()
    (current / "BUILT").write_text("sealed\n")
    with sqlite3.connect(current / "player_stats_cache_i7.sqlite") as conn:
        conn.execute("CREATE TABLE _meta (key TEXT PRIMARY KEY, value TEXT)")
        conn.execute("INSERT INTO _meta VALUES ('source_match_count', '9948')")
    stable = data / "live_state_i7"
    stable.symlink_to(current.name, target_is_directory=True)
    build = refresh_state(
        base_source=base, context_root=contexts, state_parent=data,
        stable_link=stable, metadata_csv=metadata, prior_sqlite=prior,
        fetch_cmd="stub", context_date="2026-09-10", runner=_runner_with_count(2),
    )
    assert stable.resolve() == current.resolve()
    assert (build / "BUILT").is_file()
    assert "not promoted" in capsys.readouterr().err


def test_unreadable_current_state_metadata_refuses_promotion(tmp_path, capsys):
    data, base, contexts, metadata, prior = _setup(tmp_path)
    current = data / "live_state_i7_current"   # no as_of in the name, no _meta
    current.mkdir()
    (current / "BUILT").write_text("sealed\n")
    stable = data / "live_state_i7"
    stable.symlink_to(current.name, target_is_directory=True)
    build = refresh_state(
        base_source=base, context_root=contexts, state_parent=data,
        stable_link=stable, metadata_csv=metadata, prior_sqlite=prior,
        fetch_cmd="stub", context_date="2026-09-10", runner=_runner_with_count(10_000),
    )
    assert stable.resolve() == current.resolve()
    assert "unreadable" in capsys.readouterr().err
