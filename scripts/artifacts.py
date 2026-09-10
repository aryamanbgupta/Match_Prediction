#!/usr/bin/env python3
"""Resolve, verify, rebuild, pull, and promote artifacts of record.

The checked-in manifest is the single owner of production/evaluation artifact
paths.  Import :func:`artifact_path` from loaders instead of spelling those
paths a second time.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Iterable

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = REPO_ROOT / "models" / "MANIFEST.yaml"
REQUIRED_FIELDS = {
    "role",
    "path",
    "kind",
    "hash",
    "producing_command",
    "input_roles",
    "promoting_doc",
}


class ArtifactError(RuntimeError):
    """A manifest contract or artifact operation failed."""


def _manifest_root(manifest_path: Path) -> Path:
    """Infer the repository root for the standard and temporary layouts."""
    manifest_path = manifest_path.resolve()
    if manifest_path.parent.name == "models":
        return manifest_path.parent.parent
    return manifest_path.parent


def load_manifest(manifest_path: Path | str = DEFAULT_MANIFEST) -> dict[str, dict[str, Any]]:
    path = Path(manifest_path)
    try:
        payload = yaml.safe_load(path.read_text())
    except FileNotFoundError as exc:
        raise ArtifactError(f"manifest not found: {path}") from exc
    rows = payload.get("artifacts") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ArtifactError(f"{path}: top-level 'artifacts' must be a list")
    entries: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ArtifactError(f"{path}: artifacts[{index}] is not a mapping")
        missing = REQUIRED_FIELDS - row.keys()
        if missing:
            raise ArtifactError(
                f"{path}: artifacts[{index}] missing {', '.join(sorted(missing))}"
            )
        role = row["role"]
        if not isinstance(role, str) or not role:
            raise ArtifactError(f"{path}: artifacts[{index}] has invalid role")
        if role in entries:
            raise ArtifactError(f"{path}: duplicate role {role!r}")
        if row["kind"] not in {"file", "dir"}:
            raise ArtifactError(f"{path}: {role}: kind must be file or dir")
        if not isinstance(row["input_roles"], list):
            raise ArtifactError(f"{path}: {role}: input_roles must be a list")
        entries[role] = row
    unknown = {
        dependency
        for row in entries.values()
        for dependency in row["input_roles"]
        if dependency not in entries
    }
    if unknown:
        raise ArtifactError(f"unknown input roles: {', '.join(sorted(unknown))}")
    return entries


def _safe_repo_path(raw_path: str, root: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        raise ArtifactError(f"manifest paths must be repository-relative: {raw_path}")
    resolved = (root / path).resolve(strict=False)
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ArtifactError(f"artifact path escapes repository: {raw_path}") from exc
    return resolved


def artifact_path(
    role: str,
    explicit: Path | str | None = None,
    *,
    manifest_path: Path | str = DEFAULT_MANIFEST,
) -> Path:
    """Resolve explicit CLI path > selected role > that loader's default role.

    A loader supplies its default role as ``role`` when no ``--role`` was
    given, or the selected role when it was.  Passing an explicit path always
    wins.  Manifest paths remain repository-relative so existing CLI output
    and no-flag behavior stay stable.
    """
    if explicit is not None:
        return Path(explicit)
    entries = load_manifest(manifest_path)
    try:
        return Path(entries[role]["path"])
    except KeyError as exc:
        raise ArtifactError(f"unknown artifact role: {role}") from exc


def md5_file(path: Path | str, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.md5()  # noqa: S324 - artifact identity, not security
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def md5_directory(path: Path | str) -> str:
    """Hash sorted ``relative_path md5`` lines, joined with LF (no final LF).

    The ``BUILT`` marker is excluded: it records promotion time and differs
    per machine, and it is the immutability seal, not content (2026-09-10).
    """
    root = Path(path)
    lines = [
        f"{child.relative_to(root).as_posix()} {md5_file(child)}"
        for child in root.rglob("*")
        if child.is_file() and child.name != "BUILT"
    ]
    lines.sort()
    digest = hashlib.md5()  # noqa: S324 - artifact identity, not security
    digest.update("\n".join(lines).encode("utf-8"))
    return digest.hexdigest()


def computed_hash(path: Path, kind: str) -> str:
    return md5_file(path) if kind == "file" else md5_directory(path)


def verify_role(
    role: str,
    *,
    manifest_path: Path | str = DEFAULT_MANIFEST,
    root: Path | None = None,
) -> tuple[str, str | None]:
    entries = load_manifest(manifest_path)
    if role not in entries:
        raise ArtifactError(f"unknown artifact role: {role}")
    row = entries[role]
    repo = root.resolve() if root else _manifest_root(Path(manifest_path))
    path = _safe_repo_path(row["path"], repo)
    if not path.exists():
        return "MISSING", None
    if row["kind"] == "file" and not path.is_file():
        return "MISMATCH", None
    if row["kind"] == "dir" and not path.is_dir():
        return "MISMATCH", None
    actual = computed_hash(path, row["kind"])
    return ("OK" if actual == row["hash"] else "MISMATCH"), actual


def verify(
    roles: Iterable[str] | None = None,
    *,
    manifest_path: Path | str = DEFAULT_MANIFEST,
    root: Path | None = None,
) -> bool:
    entries = load_manifest(manifest_path)
    selected = list(roles) if roles else list(entries)
    ok = True
    for role in selected:
        status, actual = verify_role(
            role, manifest_path=manifest_path, root=root
        )
        suffix = f" (actual {actual})" if actual and status != "OK" else ""
        print(f"{role}: {status}{suffix}")
        ok &= status == "OK"
    return ok


def topological_roles(role: str, entries: dict[str, dict[str, Any]]) -> list[str]:
    if role not in entries:
        raise ArtifactError(f"unknown artifact role: {role}")
    order: list[str] = []
    active: set[str] = set()
    complete: set[str] = set()

    def visit(current: str) -> None:
        if current in complete:
            return
        if current in active:
            raise ArtifactError(f"dependency cycle includes {current}")
        active.add(current)
        for dependency in entries[current]["input_roles"]:
            visit(dependency)
        active.remove(current)
        complete.add(current)
        order.append(current)

    visit(role)
    return order


def rebuild(
    role: str,
    *,
    dry_run: bool = False,
    manifest_path: Path | str = DEFAULT_MANIFEST,
    root: Path | None = None,
    runner: Callable[..., Any] = subprocess.run,
) -> list[str]:
    entries = load_manifest(manifest_path)
    repo = root.resolve() if root else _manifest_root(Path(manifest_path))
    order = topological_roles(role, entries)
    print("topological order: " + " -> ".join(order))
    for dependency in order[:-1]:
        status, _ = verify_role(
            dependency, manifest_path=manifest_path, root=repo
        )
        if status != "OK":
            raise ArtifactError(
                f"{role}: input {dependency} is {status}; refusing rebuild"
            )
    for current in order:
        command = entries[current]["producing_command"]
        print(f"{current}: {command}")
        if dry_run:
            continue
        if command.startswith("unknown ("):
            raise ArtifactError(f"{current}: no producing command is recorded")
        runner(command, cwd=repo, shell=True, check=True)
    return order


def pull(
    host: str,
    role: str,
    *,
    force: bool = False,
    manifest_path: Path | str = DEFAULT_MANIFEST,
    root: Path | None = None,
    runner: Callable[..., Any] = subprocess.run,
) -> bool:
    if not host or any(char.isspace() for char in host):
        raise ArtifactError("--from must be a non-empty host without whitespace")
    entries = load_manifest(manifest_path)
    if role not in entries:
        raise ArtifactError(f"unknown artifact role: {role}")
    row = entries[role]
    repo = root.resolve() if root else _manifest_root(Path(manifest_path))
    local = _safe_repo_path(row["path"], repo)
    status, _ = verify_role(role, manifest_path=manifest_path, root=repo)
    if status == "OK" and not force:
        raise ArtifactError(
            f"{role}: local artifact already matches; pass --force to overwrite"
        )
    local.parent.mkdir(parents=True, exist_ok=True)
    remote = f"{host}:{repo.as_posix()}/{row['path']}"
    if row["kind"] == "dir":
        local.mkdir(parents=True, exist_ok=True)
        remote += "/"
        destination = f"{local.as_posix()}/"
        command = ["rsync", "-a", "--delete", remote, destination]
    else:
        command = ["rsync", "-a", remote, local.as_posix()]
    runner(command, check=True)
    final_status, actual = verify_role(
        role, manifest_path=manifest_path, root=repo
    )
    print(f"{role}: {final_status}" + (f" ({actual})" if actual else ""))
    if final_status != "OK":
        raise ArtifactError(f"{role}: verification failed after rsync")
    return True


def refuse_if_built(directory: Path | str) -> None:
    marker = Path(directory) / "BUILT"
    if marker.exists():
        raise ArtifactError(f"immutable build directory already sealed: {marker}")


def promote_live_state(
    build_dir: Path | str,
    *,
    link_path: Path | str = REPO_ROOT / "data" / "live_state_i7",
) -> Path:
    """Seal ``build_dir`` and atomically repoint the stable live-state link."""
    build = Path(build_dir).resolve()
    link = Path(link_path)
    if not build.is_dir():
        raise ArtifactError(f"live-state build directory not found: {build}")
    refuse_if_built(build)
    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        build.relative_to(link.parent.resolve())
    except ValueError as exc:
        raise ArtifactError("live-state build must be beside the stable link") from exc
    # The marker is deliberately the final write inside the immutable build.
    (build / "BUILT").write_text("sealed\n")
    temporary = link.with_name(f".{link.name}.new-{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        temporary.unlink()
    temporary.symlink_to(build.name, target_is_directory=True)
    os.replace(temporary, link)
    return build


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=DEFAULT_MANIFEST, help=argparse.SUPPRESS
    )
    sub = parser.add_subparsers(dest="command", required=True)
    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument("role", nargs="?")
    path_parser = sub.add_parser("path")
    path_parser.add_argument("role")
    rebuild_parser = sub.add_parser("rebuild")
    rebuild_parser.add_argument("role")
    rebuild_parser.add_argument("--dry-run", action="store_true")
    pull_parser = sub.add_parser("pull")
    pull_parser.add_argument("--from", dest="host", required=True)
    pull_parser.add_argument("role", nargs="?")
    pull_parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "verify":
            roles = [args.role] if args.role else None
            return 0 if verify(roles, manifest_path=args.manifest) else 1
        if args.command == "path":
            print(artifact_path(args.role, manifest_path=args.manifest))
            return 0
        if args.command == "rebuild":
            rebuild(
                args.role, dry_run=args.dry_run, manifest_path=args.manifest
            )
            return 0
        if args.command == "pull":
            roles = [args.role] if args.role else list(load_manifest(args.manifest))
            for role in roles:
                pull(
                    args.host,
                    role,
                    force=args.force,
                    manifest_path=args.manifest,
                )
            return 0
    except (ArtifactError, subprocess.CalledProcessError) as exc:
        print(f"artifacts: ERROR: {exc}", file=sys.stderr)
        return 2
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
