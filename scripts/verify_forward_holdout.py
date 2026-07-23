#!/usr/bin/env python3
"""Read-only integrity verification for a sealed forward holdout.

This verifier never imports model code and never scores predictions. It
checks the sealed dataset against its manifest, raw strict-market source,
selected Cricsheet payload hashes, and existing evaluated pools.

Example:

    uv run python scripts/verify_forward_holdout.py \
      data/forward_holdout/2026-06-01_2026-07-13
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXISTING_EVALUATED_DIRS = (
    ROOT / "data" / "t20s_json",
    ROOT / "data" / "polymarket_test",
    ROOT / "data" / "golden" / "polymarket_test",
    ROOT / "data" / "golden_blast" / "polymarket_test",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(128 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_fingerprint(dataset_dir: Path) -> str:
    """Hash relative paths and content hashes for all sealed artifacts."""
    digest = hashlib.sha256()
    for path in sorted(p for p in dataset_dir.rglob("*") if p.is_file()):
        relative = path.relative_to(dataset_dir).as_posix()
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(sha256_file(path).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read valid JSON from {path}: {exc}") from exc


def verify(dataset_dir: Path) -> dict:
    required = (
        "SEALED",
        "manifest.json",
        "integrity_report.json",
        "betting_odds.json",
        "diagnostics.json",
    )
    missing = [name for name in required if not (dataset_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"missing sealed artifacts: {missing}")

    manifest = _load_json(dataset_dir / "manifest.json")
    integrity = _load_json(dataset_dir / "integrity_report.json")
    odds = _load_json(dataset_dir / "betting_odds.json")

    if manifest.get("model_scoring_performed") is not False:
        raise RuntimeError("manifest does not attest model_scoring_performed=false")
    if integrity.get("model_scoring_performed") is not False:
        raise RuntimeError("integrity report does not attest no model scoring")
    if integrity.get("status") != "PASS":
        raise RuntimeError("integrity report status is not PASS")

    raw_spec = manifest.get("source_market_file") or {}
    raw_dir = dataset_dir / "raw"
    raw_files = sorted(raw_dir.glob("*.json"))
    if len(raw_files) != 1:
        raise RuntimeError(f"expected exactly one raw market JSON, got {len(raw_files)}")
    if sha256_file(raw_files[0]) != raw_spec.get("sha256"):
        raise RuntimeError("raw strict-market source hash differs from manifest")
    market_blob = _load_json(raw_files[0])
    raw_markets = {
        str(row["market_id"]): row for row in market_blob.get("matches") or []
    }

    manifest_rows = manifest.get("matches") or []
    expected_count = int(manifest.get("selected_match_count", -1))
    if expected_count != len(manifest_rows):
        raise RuntimeError("manifest selected_match_count does not match rows")
    cricsheet_ids = [str(row["cricsheet_id"]) for row in manifest_rows]
    market_ids = [str(row["polymarket_market_id"]) for row in manifest_rows]
    if len(set(cricsheet_ids)) != len(cricsheet_ids):
        raise RuntimeError("duplicate Cricsheet ID in manifest")
    if len(set(market_ids)) != len(market_ids):
        raise RuntimeError("duplicate Polymarket market ID in manifest")

    selected_dir = dataset_dir / "polymarket_test"
    selected_files = {path.stem: path for path in selected_dir.glob("*.json")}
    if set(selected_files) != set(cricsheet_ids):
        raise RuntimeError("selected Cricsheet files do not match manifest IDs")
    context_ids = {
        path.stem for path in (dataset_dir / "context_t20s_json").glob("*.json")
    }
    if not set(cricsheet_ids) <= context_ids:
        raise RuntimeError("selected Cricsheet files are missing from context corpus")

    odds_rows = odds.get("matches") or []
    if odds.get("total_matches") != expected_count or len(odds_rows) != expected_count:
        raise RuntimeError("betting_odds count does not match manifest")
    for row in manifest_rows:
        cricsheet_id = str(row["cricsheet_id"])
        selected_path = selected_files[cricsheet_id]
        if sha256_file(selected_path) != row.get("cricsheet_sha256"):
            raise RuntimeError(f"Cricsheet hash mismatch: {cricsheet_id}")
        raw_market = raw_markets.get(str(row["polymarket_market_id"]))
        if raw_market is None:
            raise RuntimeError(
                f"market missing from raw strict source: {row['polymarket_market_id']}"
            )
        if raw_market.get("event_id") != row.get("polymarket_event_id"):
            raise RuntimeError(f"event ID mismatch for {cricsheet_id}")
        if raw_market.get("price_timestamp") != row.get("price_timestamp"):
            raise RuntimeError(f"price timestamp mismatch for {cricsheet_id}")
        if not (
            str(row["price_timestamp"])
            < str(row["scheduled_start_timestamp"])
        ):
            raise RuntimeError(f"non-prematch timestamp for {cricsheet_id}")

    odds_market_ids = {
        str(row.get("polymarket_market_id")) for row in odds_rows
    }
    if odds_market_ids != set(market_ids):
        raise RuntimeError("betting_odds market IDs do not match manifest")

    existing_ids: set[str] = set()
    for directory in EXISTING_EVALUATED_DIRS:
        existing_ids.update(path.stem for path in directory.glob("*.json"))
    overlap = sorted(set(cricsheet_ids) & existing_ids)
    if overlap:
        raise RuntimeError(f"sealed holdout now overlaps existing pools: {overlap[:20]}")

    return {
        "status": "PASS",
        "model_scoring_performed": False,
        "selected_matches": expected_count,
        "context_matches": len(context_ids),
        "strict_timestamp_checks": expected_count,
        "cricsheet_hash_checks": expected_count,
        "overlap_with_existing_pools": 0,
        "dataset_fingerprint_sha256": dataset_fingerprint(dataset_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.dataset_dir.resolve()), indent=2))


if __name__ == "__main__":
    main()
