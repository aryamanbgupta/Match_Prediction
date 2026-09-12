# manifest-exempt: embeddings-ladder experimental artifact namespace
"""Stage 4 input guard — the only inputs the Stage 4 scripts may open.

Every Stage 4 helper calls `assert_permitted_inputs(...)` BEFORE it opens
anything. The guard is deliberately an allowlist of *resolved* paths with
content pins, not a blocklist of names, because a blocklist is defeated
by a symlink: a file called `cricket_data_i7_train.parquet` can point at
the sealed test parquet, and a name check would wave it through.

Accepted, and nothing else:

  * data/xgb_data_i7/cricket_data_i7_train.parquet
  * data/xgb_data_i7/cricket_data_i7_validation.parquet
        — accepted only when os.path.realpath lands exactly on that
          repo path AND the file's size and the sha256 of its first
          64 KiB match the values pinned below. The test parquet, a
          symlink wearing a permitted name, and anything under
          data/golden or data/forward_holdout therefore all fail on
          the resolved-path check before a read happens.
  * models/player_stats_cache_i7.sqlite
        — accepted only when its md5 matches the `stats_cache_i7` role
          hash recorded in models/MANIFEST.yaml (read through
          scripts/artifacts.py, so the manifest stays the single
          source of truth and this module pins no second copy).

Preflight:
    uv run --no-sync python scripts/sequence_track/stage4_guard.py --preflight
records one positive and four negative checks to
models/embeddings/stage4/preflight.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import artifacts as artifacts_mod  # noqa: E402

HEAD_BYTES = 64 * 1024

# Pinned parquet identity: repo-relative path -> (size, sha256 of head).
PINNED_PARQUETS: dict[str, tuple[int, str]] = {
    "data/xgb_data_i7/cricket_data_i7_train.parquet": (
        687804221,
        "52091ac96ff7cfb0a2b090f16b58135f529d1e4014caf65ba01f6a91a1c9df69",
    ),
    "data/xgb_data_i7/cricket_data_i7_validation.parquet": (
        54044281,
        "4ee5843009e51cb22ddb5f6b7579eed00c8ed3f2f60c1854b1878bbe09b00509",
    ),
}

CACHE_ROLE = "stats_cache_i7"
PINNED_CACHE = "models/player_stats_cache_i7.sqlite"


class GuardError(RuntimeError):
    """An input is not on the Stage 4 allowlist."""


def _realpath(path: Path | str) -> Path:
    """Resolve symlinks and '..' without touching the file."""
    return Path(os.path.realpath(str(path)))


def _rel(path: Path) -> str | None:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return None


def _head_sha256(path: Path) -> str:
    with path.open("rb") as fh:
        return hashlib.sha256(fh.read(HEAD_BYTES)).hexdigest()


def _manifest_cache_hash() -> str:
    entries = artifacts_mod.load_manifest()
    entry = entries.get(CACHE_ROLE)
    if entry is None:
        raise GuardError(f"models/MANIFEST.yaml declares no {CACHE_ROLE!r} role")
    return str(entry["hash"])


def _check_one(path: Path | str) -> str:
    """Return the accepted repo-relative path, or raise GuardError."""
    resolved = _realpath(path)
    rel = _rel(resolved)
    if rel is None:
        raise GuardError(f"{path}: resolves outside the repo ({resolved})")

    if rel in PINNED_PARQUETS:
        size, head = PINNED_PARQUETS[rel]
        actual_size = resolved.stat().st_size
        if actual_size != size:
            raise GuardError(
                f"{rel}: size {actual_size} != pinned {size}; the frame of "
                "record changed, re-pin deliberately")
        actual_head = _head_sha256(resolved)
        if actual_head != head:
            raise GuardError(
                f"{rel}: head sha256 {actual_head} != pinned {head}")
        return rel

    if rel == PINNED_CACHE:
        expected = _manifest_cache_hash()
        actual = artifacts_mod.md5_file(resolved)
        if actual != expected:
            raise GuardError(
                f"{rel}: md5 {actual} != the {CACHE_ROLE} hash {expected} "
                "recorded in models/MANIFEST.yaml")
        return rel

    raise GuardError(
        f"{path}: resolves to {rel}, which is not a permitted Stage 4 input. "
        f"Permitted: {', '.join(sorted(PINNED_PARQUETS))}, {PINNED_CACHE}. "
        "The test split and everything under data/golden and "
        "data/forward_holdout are sealed for Stage 4.")


def assert_permitted_inputs(paths) -> list[str]:
    """Validate every input BEFORE any of them is opened for data.

    Raises GuardError on the first path that is not on the allowlist.
    Returns the accepted repo-relative paths, in the order given.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    return [_check_one(p) for p in paths]


# --- preflight ------------------------------------------------------------

def _expect_pass(label: str, path) -> dict:
    try:
        rel = assert_permitted_inputs(path)[0]
        return {"check": label, "path": str(path), "expected": "accept",
                "result": "accepted", "resolved": rel, "ok": True}
    except GuardError as exc:
        return {"check": label, "path": str(path), "expected": "accept",
                "result": "refused", "error": str(exc), "ok": False}


def _expect_refuse(label: str, path) -> dict:
    try:
        assert_permitted_inputs(path)
    except GuardError as exc:
        return {"check": label, "path": str(path), "expected": "refuse",
                "result": "refused", "error": str(exc), "ok": True}
    return {"check": label, "path": str(path), "expected": "refuse",
            "result": "accepted", "ok": False,
            "error": "GUARD BREACH: this input was accepted"}


def preflight() -> dict:
    checks: list[dict] = []
    for rel in sorted(PINNED_PARQUETS):
        checks.append(_expect_pass(f"pinned parquet {Path(rel).name}",
                                   ROOT / rel))
    checks.append(_expect_pass("stats cache (manifest md5)",
                               ROOT / PINNED_CACHE))

    checks.append(_expect_refuse(
        "sealed test parquet",
        ROOT / "data/xgb_data_i7/cricket_data_i7_test.parquet"))

    # A symlink wearing a permitted NAME but pointing at the sealed test
    # parquet — the case a name-based check would wave through.
    with tempfile.TemporaryDirectory() as tmp:
        link = Path(tmp) / "cricket_data_i7_train.parquet"
        link.symlink_to(ROOT / "data/xgb_data_i7/cricket_data_i7_test.parquet")
        checks.append(_expect_refuse(
            "symlink named cricket_data_i7_train.parquet -> test parquet",
            link))

    checks.append(_expect_refuse(
        "data/golden path",
        ROOT / "data/golden/polymarket_test_v2"))
    checks.append(_expect_refuse(
        "data/forward_holdout path",
        ROOT / "data/forward_holdout/2026-06-01_2026-07-13"))

    result = {
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "guard": "scripts/sequence_track/stage4_guard.py",
        "manifest_role": CACHE_ROLE,
        "manifest_hash": _manifest_cache_hash(),
        "head_bytes": HEAD_BYTES,
        "checks": checks,
        "n_checks": len(checks),
        "n_failed": sum(1 for c in checks if not c["ok"]),
    }
    result["passed"] = result["n_failed"] == 0
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preflight", action="store_true",
                    help="run the positive/negative guard checks and record "
                         "them to models/embeddings/stage4/preflight.json")
    args = ap.parse_args()
    if not args.preflight:
        ap.error("nothing to do; pass --preflight")

    result = preflight()
    out = ROOT / "models/embeddings/stage4/preflight.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    for c in result["checks"]:
        mark = "ok  " if c["ok"] else "FAIL"
        print(f"  [{mark}] {c['check']}: expected {c['expected']}, "
              f"got {c['result']}")
    print(f"\n{result['n_checks'] - result['n_failed']}/{result['n_checks']} "
          f"checks passed -> {out}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
