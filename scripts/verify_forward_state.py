#!/usr/bin/env python3
"""Verify a forward-state sidecar without importing or scoring a model."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

from loaders_common import SAME_DAY_ORDER_VERSION  # noqa: E402
from verify_forward_holdout import verify as verify_holdout  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(128 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_fingerprint(directory: Path) -> str:
    """Hash relative paths and bytes for every file in a deterministic order."""
    digest = hashlib.sha256()
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        relative = path.relative_to(directory).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def verify(state_dir: Path) -> dict:
    state_dir = state_dir.resolve()
    manifest_path = state_dir / "manifest.json"
    marker_path = state_dir / "NO_MODEL_SCORING"
    if not manifest_path.is_file() or not marker_path.is_file():
        raise RuntimeError("state must contain manifest.json and NO_MODEL_SCORING")

    manifest = json.loads(manifest_path.read_text())
    if manifest.get("model_scoring_performed") is not False:
        raise RuntimeError("manifest does not assert model_scoring_performed=false")
    if manifest.get("same_day_order_version") != SAME_DAY_ORDER_VERSION:
        raise RuntimeError("manifest same-day ordering contract mismatch")

    listed_paths: set[str] = set()
    for artifact in manifest.get("artifacts", []):
        relative = artifact["path"]
        if relative in listed_paths:
            raise RuntimeError(f"duplicate artifact in manifest: {relative}")
        listed_paths.add(relative)
        path = state_dir / relative
        if not path.is_file():
            raise RuntimeError(f"manifest artifact is missing: {relative}")
        if path.stat().st_size != artifact["size"]:
            raise RuntimeError(f"artifact size mismatch: {relative}")
        if sha256_file(path) != artifact["sha256"]:
            raise RuntimeError(f"artifact hash mismatch: {relative}")

    live_artifacts = {
        path.relative_to(state_dir).as_posix()
        for path in state_dir.rglob("*")
        if path.is_file()
        and path.name not in {"manifest.json", "NO_MODEL_SCORING"}
    }
    if live_artifacts != listed_paths:
        raise RuntimeError(
            "manifest artifact inventory mismatch: "
            f"missing={sorted(live_artifacts - listed_paths)}, "
            f"extra={sorted(listed_paths - live_artifacts)}"
        )

    sqlite_path = state_dir / "player_stats_cache_v3.sqlite"
    conn = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    try:
        sqlite_meta = dict(conn.execute("SELECT key, value FROM _meta"))
    finally:
        conn.close()
    if sqlite_meta != manifest.get("sqlite_meta"):
        raise RuntimeError("live SQLite metadata differs from the manifest")
    if sqlite_meta.get("same_day_order_version") != SAME_DAY_ORDER_VERSION:
        raise RuntimeError("SQLite same-day ordering contract mismatch")
    if sqlite_meta.get("prior_contract") != "frozen_external_sqlite_v1":
        raise RuntimeError("SQLite does not use frozen pre-holdout priors")
    prior_contract = manifest.get("prior_contract", {})
    if (
        sqlite_meta.get("prior_source_sha256")
        != prior_contract.get("prior_source_sha256")
    ):
        raise RuntimeError("SQLite prior source hash differs from the manifest")

    holdout_dir = Path(manifest["holdout_directory"])
    holdout_report = verify_holdout(holdout_dir)
    if (
        holdout_report["dataset_fingerprint_sha256"]
        != manifest["holdout_dataset_fingerprint_sha256"]
    ):
        raise RuntimeError("sealed holdout fingerprint differs from state manifest")
    holdout_manifest = json.loads((holdout_dir / "manifest.json").read_text())
    selected_ids = {
        str(row["cricsheet_id"]) for row in holdout_manifest["matches"]
    }

    golden_path = state_dir / "match_features" / "golden_test.parquet"
    golden = pd.read_parquet(golden_path, columns=["cricsheet_id"])
    golden_ids = set(golden["cricsheet_id"].astype(str))
    missing = sorted(selected_ids - golden_ids)
    if missing:
        raise RuntimeError(
            f"selected holdout IDs missing from golden features: {missing[:20]}"
        )
    if len(selected_ids) != manifest["selected_holdout_feature_rows_verified"]:
        raise RuntimeError("selected holdout feature count differs from manifest")

    return {
        "status": "VERIFIED",
        "model_scoring_performed": False,
        "state_directory": str(state_dir),
        "directory_fingerprint_sha256": directory_fingerprint(state_dir),
        "same_day_order_version": SAME_DAY_ORDER_VERSION,
        "prior_contract": sqlite_meta["prior_contract"],
        "holdout_dataset_fingerprint_sha256": holdout_report[
            "dataset_fingerprint_sha256"
        ],
        "artifact_count": len(listed_paths),
        "selected_holdout_feature_rows": len(selected_ids),
        "golden_context_rows": len(golden),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state_dir", type=Path)
    report = verify(parser.parse_args().state_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
