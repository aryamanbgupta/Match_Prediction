from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from artifacts import (
    ArtifactError,
    artifact_path,
    md5_directory,
    md5_file,
    promote_live_state,
    pull,
    rebuild,
    refuse_if_built,
    verify,
    verify_role,
)


FIELDS = {
    "producing_command": "true",
    "input_roles": [],
    "promoting_doc": "test",
}


def _row(role: str, path: str, kind: str, digest: str, **updates):
    row = {"role": role, "path": path, "kind": kind, "hash": digest, **FIELDS}
    row.update(updates)
    return row


def _manifest(root: Path, rows: list[dict]) -> Path:
    path = root / "MANIFEST.yaml"
    path.write_text(yaml.safe_dump({"artifacts": rows}, sort_keys=False))
    return path


def test_file_and_directory_hash_contract(tmp_path):
    one = tmp_path / "one.bin"
    one.write_bytes(b"one")
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "b").write_bytes(b"two")
    (tree / "a").write_bytes(b"one")

    assert md5_file(one) == hashlib.md5(b"one").hexdigest()
    lines = "\n".join(
        [
            f"a {hashlib.md5(b'one').hexdigest()}",
            f"b {hashlib.md5(b'two').hexdigest()}",
        ]
    )
    assert md5_directory(tree) == hashlib.md5(lines.encode()).hexdigest()


def test_verify_ok_missing_and_mismatch(tmp_path, capsys):
    good = tmp_path / "good"
    good.write_text("right")
    bad = tmp_path / "bad"
    bad.write_text("wrong")
    manifest = _manifest(
        tmp_path,
        [
            _row("good_role", "good", "file", md5_file(good)),
            _row("missing_role", "missing", "file", "0" * 32),
            _row("bad_role", "bad", "file", hashlib.md5(b"right").hexdigest()),
        ],
    )
    assert verify_role("good_role", manifest_path=manifest) == (
        "OK",
        md5_file(good),
    )
    assert verify_role("missing_role", manifest_path=manifest) == ("MISSING", None)
    assert verify_role("bad_role", manifest_path=manifest)[0] == "MISMATCH"
    assert not verify(manifest_path=manifest)
    output = capsys.readouterr().out
    assert "missing_role: MISSING" in output
    assert "bad_role: MISMATCH" in output


def test_artifact_path_precedence(tmp_path):
    manifest = _manifest(
        tmp_path,
        [
            _row("default", "default/path", "dir", "0" * 32),
            _row("selected", "selected/path", "dir", "0" * 32),
        ],
    )
    assert artifact_path("default", manifest_path=manifest) == Path("default/path")
    assert artifact_path("selected", manifest_path=manifest) == Path("selected/path")
    assert artifact_path(
        "selected", "explicit/path", manifest_path=manifest
    ) == Path("explicit/path")


def test_rebuild_diamond_order_and_dry_run(tmp_path, capsys):
    rows = []
    for role, inputs in [
        ("base", []),
        ("left", ["base"]),
        ("right", ["base"]),
        ("top", ["left", "right"]),
    ]:
        artifact = tmp_path / role
        artifact.write_text(role)
        rows.append(
            _row(
                role,
                role,
                "file",
                md5_file(artifact),
                input_roles=inputs,
                producing_command=f"build {role}",
            )
        )
    manifest = _manifest(tmp_path, rows)
    calls = []
    order = rebuild(
        "top",
        dry_run=True,
        manifest_path=manifest,
        runner=lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    assert order == ["base", "left", "right", "top"]
    assert calls == []
    assert "base -> left -> right -> top" in capsys.readouterr().out


def test_rebuild_refuses_input_hash_mismatch(tmp_path):
    dependency = tmp_path / "dependency"
    dependency.write_text("corrupt")
    target = tmp_path / "target"
    target.write_text("target")
    manifest = _manifest(
        tmp_path,
        [
            _row("dependency", "dependency", "file", "0" * 32),
            _row(
                "target",
                "target",
                "file",
                md5_file(target),
                input_roles=["dependency"],
            ),
        ],
    )
    with pytest.raises(ArtifactError, match="dependency is MISMATCH"):
        rebuild("target", manifest_path=manifest)


def test_pull_refuses_matching_without_force_and_scopes_rsync(tmp_path):
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"same")
    manifest = _manifest(
        tmp_path,
        [_row("thing", "artifact.bin", "file", md5_file(artifact))],
    )
    with pytest.raises(ArtifactError, match="already matches"):
        pull("mac-mini", "thing", manifest_path=manifest)

    calls = []
    pull(
        "mac-mini",
        "thing",
        force=True,
        manifest_path=manifest,
        runner=lambda command, **kwargs: calls.append(command),
    )
    assert len(calls) == 1
    assert calls[0][:2] == ["rsync", "-a"]
    assert calls[0][-1] == str(artifact)
    assert calls[0][-2].endswith(f":{tmp_path}/artifact.bin")


def test_live_state_promote_refuse_and_repromote(tmp_path):
    link = tmp_path / "live_state_i7"
    first = tmp_path / "live_state_i7_2026-07-30_20260801T050221Z"
    first.mkdir()
    (first / "cache").write_text("first")
    promote_live_state(first, link_path=link)
    assert link.is_symlink()
    assert link.resolve() == first.resolve()
    assert (first / "BUILT").read_text() == "sealed\n"
    with pytest.raises(ArtifactError, match="already sealed"):
        refuse_if_built(first)
    with pytest.raises(ArtifactError, match="already sealed"):
        promote_live_state(first, link_path=link)

    second = tmp_path / "live_state_i7_2026-07-30_20260801T060000Z"
    second.mkdir()
    (second / "cache").write_text("second")
    promote_live_state(second, link_path=link)
    assert link.resolve() == second.resolve()
    assert (first / "cache").read_text() == "first"
    assert (second / "BUILT").exists()
