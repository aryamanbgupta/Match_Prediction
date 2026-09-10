#!/usr/bin/env python3
"""Build and atomically promote a fresh matched SQLite/tracker state pair."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import pickle
import shlex
import shutil
import sqlite3
import subprocess
import sys
import re
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from artifacts import promote_live_state, refuse_if_built  # noqa: E402

# Operation 6's fetcher, with its module-level destinations redirected so it
# cannot merge into the frozen/base corpus. The fetch cache remains beside
# the dated contexts; only new immutable match JSONs land in this run's dir.
DEFAULT_FETCH_CMD = (
    "uv run --no-sync python -c '"
    "import os,sys; from pathlib import Path; "
    "sys.path.insert(0,\"scripts\"); import fetch_cricsheet as f; "
    "c=Path(os.environ[\"CRICML_CONTEXT_DIR\"]); w=c.parent/\".fetch_cache\"; "
    "f.DATA_DIR=w; f.ZIP_DIR=w/\".cricsheet_zips\"; "
    "f.STAGING_DIR=f.ZIP_DIR/\".staging\"; f.EXTRACT_DIR=f.ZIP_DIR/\".extract\"; "
    "f.MANIFEST_PATH=f.ZIP_DIR/\"manifest.json\"; "
    "f.LOG_FILE=f.ZIP_DIR/\".refresh.log\"; f.MATCH_DIR=c; "
    "f.PEOPLE_PATH=w/\"cricsheet_people.csv\"; "
    "f.ENRICHED_PATH=Path(\"data/all_players_enriched.csv\"); f.main()'"
)


def corpus_as_of(paths: list[Path]) -> str:
    dates = []
    for source in paths:
        for path in source.glob("*.json"):
            try:
                values = json.loads(path.read_text()).get("info", {}).get("dates") or []
                if values:
                    dates.append(str(values[0]))
            except (OSError, json.JSONDecodeError):
                continue
    if not dates:
        raise RuntimeError("state sources contain no dated Cricsheet matches")
    return max(dates)


def _sqlite_count(path: Path) -> int | None:
    with sqlite3.connect(path) as conn:
        row = conn.execute("SELECT value FROM _meta WHERE key='source_match_count'").fetchone()
    return int(row[0]) if row else None


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(parent.resolve(strict=False))
        return True
    except ValueError:
        return False


def _refuse_built_ancestor(path: Path) -> None:
    resolved = path.resolve(strict=False)
    for candidate in (resolved, *resolved.parents):
        if (candidate / "BUILT").exists():
            raise RuntimeError(f"destination is beneath sealed build: {candidate}")


def _validate_paths(*, base_source: Path, context_root: Path,
                    state_parent: Path, stable_link: Path) -> None:
    base = base_source.resolve(strict=False)
    context = context_root.resolve(strict=False)
    state = state_parent.resolve(strict=False)
    if _is_within(context, base):
        raise ValueError("context root must not be inside the base corpus")
    if _is_within(state, base):
        raise ValueError("state parent must not be inside the base corpus")
    if state != base.parent or state.name != "data" or base.name != "t20s_json":
        raise ValueError("state parent must be data/ (the parent of the base corpus)")
    # Repointing the stable symlink writes in its parent; following the link
    # would incorrectly classify every healthy, sealed current target as a
    # destination ancestor.
    for destination in (context, state, stable_link.parent):
        _refuse_built_ancestor(Path(destination))


def dated_context_dirs(context_root: Path) -> list[Path]:
    """Return immutable YYYY-MM-DD contexts in deterministic date order."""
    return sorted(
        path for path in context_root.iterdir()
        if path.is_dir()
        and len(path.name) == 10
        and path.name[4:5] == "-"
        and path.name[7:8] == "-"
        and path.name.replace("-", "").isdigit()
    ) if context_root.exists() else []


def _known_ids(base_source: Path, contexts: list[Path]) -> set[str]:
    return {
        path.stem
        for source in (base_source, *contexts)
        for path in source.glob("*.json")
    }


def _fetch_new_context(*, context: Path, base_source: Path,
                       published_contexts: list[Path], fetch_cmd: str,
                       runner) -> None:
    context.parent.mkdir(parents=True, exist_ok=True)
    incoming = context.parent / f".tmp-{context.name}-{os.getpid()}"
    shutil.rmtree(incoming, ignore_errors=True)
    incoming.mkdir()
    try:
        env = dict(os.environ, CRICML_CONTEXT_DIR=str(incoming))
        command = fetch_cmd.format(context_dir=shlex.quote(str(incoming)))
        runner(command, cwd=REPO, shell=True, check=True, env=env)
        known = _known_ids(base_source, published_contexts)
        for path in sorted(incoming.glob("*.json"), key=lambda item: item.stem):
            if path.stem in known:
                path.unlink()
            else:
                known.add(path.stem)
        os.rename(incoming, context)
    except BaseException:
        shutil.rmtree(incoming, ignore_errors=True)
        raise


def _refresh_state_unlocked(*, base_source: Path, context_root: Path, state_parent: Path,
                  stable_link: Path, metadata_csv: Path, prior_sqlite: Path,
                  fetch_cmd: str = DEFAULT_FETCH_CMD,
                  context_date: str | None = None,
                  runner=subprocess.run) -> Path:
    _validate_paths(base_source=base_source, context_root=context_root,
                    state_parent=state_parent, stable_link=stable_link)
    day = context_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day):
        raise ValueError("context_date must match YYYY-MM-DD")
    context = context_root / day
    resolved_context = context.resolve(strict=False)
    if not _is_within(resolved_context, context_root):
        raise ValueError("context destination escapes context root")
    if _is_within(resolved_context, base_source) or (resolved_context / "BUILT").exists():
        raise ValueError("context destination is protected")
    _refuse_built_ancestor(resolved_context)
    if context_root.exists():
        for leftover in context_root.glob(".tmp-*"):
            if leftover.is_dir():
                shutil.rmtree(leftover)
    if context.exists():
        if stable_link.exists() and (stable_link / "BUILT").exists():
            state_record = stable_link / "state.json"
            if state_record.exists() and json.loads(state_record.read_text()).get(
                "context_dir"
            ) == str(context):
                return stable_link.resolve()
        # A failed prior attempt may have completed the immutable fetch but
        # not the state build. Reuse it read-only; never fetch into it again.
    else:
        published = dated_context_dirs(context_root)
        _fetch_new_context(context=context, base_source=base_source,
                           published_contexts=published, fetch_cmd=fetch_cmd,
                           runner=runner)
    contexts = dated_context_dirs(context_root)
    sources = [base_source, *contexts]
    as_of = corpus_as_of(sources)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    build = state_parent / f"live_state_i7_{as_of}_{stamp}"
    build.mkdir(parents=True)
    refuse_if_built(build)
    sqlite_path = build / "player_stats_cache_i7.sqlite"
    cache_command = [
        "uv", "run", "--no-sync", "python", "scripts/build_stats_cache.py",
        "--source-dir", str(base_source),
        "--out", str(sqlite_path), "--metadata-csv", str(metadata_csv),
        "--prior-source-sqlite", str(prior_sqlite), "--force-rebuild",
    ]
    for source in contexts:
        cache_command.extend(("--extra-source-dir", str(source)))
    runner(cache_command, cwd=REPO, check=True)
    latest_path = max(
        [path for source in sources for path in source.glob("*.json")],
        key=lambda path: (
            str((json.loads(path.read_text()).get("info", {}).get("dates") or [""])[0]),
            path.stem,
        ),
    )
    latest_info = json.loads(latest_path.read_text())["info"]
    latest_teams = latest_info["teams"]
    latest_players = latest_info.get("players") or {}
    registry = (latest_info.get("registry") or {}).get("people") or {}
    placeholder = build / "snapshot_fixture.json"
    placeholder.write_text(json.dumps({
        "date": as_of, "team1": latest_teams[0], "team2": latest_teams[1],
        "venue": latest_info.get("venue") or "Unknown venue",
        "team1_lineup": [registry.get(p, p) for p in latest_players[latest_teams[0]]],
        "team2_lineup": [registry.get(p, p) for p in latest_players[latest_teams[1]]],
    }))
    smoke_output = build / "snapshot_smoke.json"
    tracker_command = [
        "uv", "run", "--no-sync", "python", "scripts/predict_fixture.py",
        "--fixture", str(placeholder), "--state-dir", str(build),
        "--tracker-snapshot", str(build / "tracker_snapshot.pkl"),
        "--tracker-source-dir", str(base_source),
        "--rebuild-snapshot",
        "--out", str(smoke_output),
    ]
    for source in contexts:
        insertion = tracker_command.index("--rebuild-snapshot")
        tracker_command[insertion:insertion] = ["--tracker-source-dir", str(source)]
    runner(tracker_command, cwd=REPO, check=True)
    placeholder.unlink()
    smoke_output.unlink(missing_ok=True)
    with (build / "tracker_snapshot.pkl").open("rb") as handle:
        snapshot = pickle.load(handle)
    source_count = _sqlite_count(sqlite_path)
    if source_count != snapshot.get("n_matches_walked"):
        raise RuntimeError("SQLite/tracker source-match counts disagree")
    (build / "state.json").write_text(json.dumps({
        "state_as_of": as_of, "context_dir": str(context),
        "context_dirs": [str(path) for path in contexts],
        "source_match_count": source_count,
        "built_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2) + "\n")
    promote = True
    if stable_link.exists() and (stable_link / "state.json").exists():
        current = json.loads((stable_link / "state.json").read_text())
        current_count = current.get("source_match_count")
        if current_count is None:
            current_count = _sqlite_count(stable_link / "player_stats_cache_i7.sqlite")
        promote = (as_of >= str(current.get("state_as_of", ""))
                   and source_count >= int(current_count or 0))
    if promote:
        promote_live_state(build, link_path=stable_link)
    else:
        (build / "BUILT").write_text("sealed\n")
        print("stale build not promoted", file=sys.stderr)
    return build


def refresh_state(**kwargs) -> Path:
    """Hold one non-blocking global refresh lock for the complete run."""
    context_root = Path(kwargs["context_root"])
    _validate_paths(base_source=Path(kwargs["base_source"]), context_root=context_root,
                    state_parent=Path(kwargs["state_parent"]),
                    stable_link=Path(kwargs["stable_link"]))
    day = kwargs.get("context_date") or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day):
        raise ValueError("context_date must match YYYY-MM-DD")
    destination = (context_root / day).resolve(strict=False)
    if not _is_within(destination, context_root):
        raise ValueError("context destination escapes context root")
    _refuse_built_ancestor(destination)
    lock_dir = context_root.parent
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / "refresh.lock"
    with lock_path.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            print(f"refresh refused: lock held: {lock_path}", file=sys.stderr)
            raise RuntimeError("refresh lock is already held") from exc
        return _refresh_state_unlocked(**kwargs)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-source", type=Path, required=True)
    p.add_argument("--context-root", type=Path, required=True)
    p.add_argument("--state-parent", type=Path, default=REPO / "data")
    p.add_argument("--stable-link", type=Path, default=REPO / "data" / "live_state_i7")
    p.add_argument("--metadata-csv", type=Path, default=REPO / "data" / "all_players_enriched.csv")
    p.add_argument("--prior-sqlite", type=Path, default=REPO / "models" / "player_stats_cache_i7.sqlite")
    p.add_argument("--fetch-cmd", default=DEFAULT_FETCH_CMD)
    p.add_argument("--context-date")
    a = p.parse_args(argv)
    print(refresh_state(base_source=a.base_source, context_root=a.context_root,
                        state_parent=a.state_parent, stable_link=a.stable_link,
                        metadata_csv=a.metadata_csv, prior_sqlite=a.prior_sqlite,
                        fetch_cmd=a.fetch_cmd, context_date=a.context_date))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
