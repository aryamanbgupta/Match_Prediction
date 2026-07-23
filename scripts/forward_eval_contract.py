#!/usr/bin/env python3
"""Model-free validation for the one-time forward-evaluation contract.

This module deliberately imports no model, simulation, or training code.
It verifies the sealed holdout, the sidecar state, all declared candidate
artifacts, liquidity counts, and the DRAFT/FROZEN opening conditions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent

from verify_forward_holdout import verify as verify_holdout  # noqa: E402
from verify_forward_state import verify as verify_state  # noqa: E402


ALLOWED_PROTOCOL_STATUSES = frozenset({"DRAFT", "FROZEN"})
REQUIRED_OPENING_CONDITIONS = (
    "i3_complete",
    "i6_complete",
    "scorer_tests_complete",
    "ball_same_day_replay_complete",
    "scoring_code_hashes_recorded",
    "user_approved",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(128 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_path(value: str) -> Path:
    """Resolve a protocol path and reject paths outside this repository."""
    path = (ROOT / value).resolve()
    try:
        path.relative_to(ROOT)
    except ValueError as exc:
        raise RuntimeError(f"protocol path escapes repository: {value}") from exc
    return path


def load_protocol(path: Path) -> dict[str, Any]:
    try:
        document = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise RuntimeError(f"cannot load protocol {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise RuntimeError("forward protocol must be a mapping")
    if document.get("schema_version") != 1:
        raise RuntimeError("unsupported forward protocol schema")
    status = document.get("status")
    if status not in ALLOWED_PROTOCOL_STATUSES:
        raise RuntimeError(f"invalid protocol status: {status!r}")
    return document


def liquidity_slice_counts(odds_document: dict[str, Any]) -> dict[str, int]:
    rows = odds_document.get("matches") or []
    volumes = [float(row.get("polymarket_volume_usd") or 0.0) for row in rows]
    return {
        "all": len(rows),
        "min_volume_50000": sum(volume >= 50_000 for volume in volumes),
        "min_volume_100000": sum(volume >= 100_000 for volume in volumes),
    }


def _verify_artifacts(protocol: dict[str, Any]) -> list[dict[str, str]]:
    verified: list[dict[str, str]] = []
    candidates = protocol.get("candidates") or {}
    if set(candidates) != {"match_m7", "ball_v7"}:
        raise RuntimeError("protocol must declare exactly match_m7 and ball_v7")
    for candidate_name, candidate in candidates.items():
        artifacts = candidate.get("artifacts") or []
        if not artifacts:
            raise RuntimeError(f"{candidate_name} has no pinned artifacts")
        seen: set[str] = set()
        for artifact in artifacts:
            relative = str(artifact.get("path") or "")
            expected_hash = str(artifact.get("sha256") or "")
            if relative in seen:
                raise RuntimeError(
                    f"duplicate artifact for {candidate_name}: {relative}"
                )
            seen.add(relative)
            path = repo_path(relative)
            if not path.is_file():
                raise RuntimeError(f"missing candidate artifact: {relative}")
            actual_hash = sha256_file(path)
            if actual_hash != expected_hash:
                raise RuntimeError(
                    f"candidate artifact hash mismatch: {relative}"
                )
            verified.append({
                "candidate": candidate_name,
                "path": relative,
                "sha256": actual_hash,
            })
    return verified


def preflight(protocol_path: Path, require_frozen: bool = False) -> dict[str, Any]:
    """Verify the protocol and return a model-free preflight report."""
    protocol_path = protocol_path.resolve()
    protocol = load_protocol(protocol_path)
    holdout_spec = protocol.get("holdout") or {}
    state_spec = protocol.get("state") or {}
    holdout_dir = repo_path(str(holdout_spec.get("directory") or ""))
    state_dir = repo_path(str(state_spec.get("directory") or ""))

    holdout_report = verify_holdout(holdout_dir)
    if (
        holdout_report["dataset_fingerprint_sha256"]
        != holdout_spec.get("dataset_fingerprint_sha256")
    ):
        raise RuntimeError("holdout fingerprint differs from protocol")
    if holdout_report["selected_matches"] != holdout_spec.get("selected_matches"):
        raise RuntimeError("holdout match count differs from protocol")

    odds = json.loads((holdout_dir / "betting_odds.json").read_text())
    live_slices = liquidity_slice_counts(odds)
    expected_slices = holdout_spec.get("liquidity_slices") or {}
    if live_slices != expected_slices:
        raise RuntimeError(
            f"liquidity slices differ from protocol: "
            f"{live_slices} != {expected_slices}"
        )

    state_report = verify_state(state_dir)
    if (
        state_report["directory_fingerprint_sha256"]
        != state_spec.get("directory_fingerprint_sha256")
    ):
        raise RuntimeError("forward-state fingerprint differs from protocol")
    if (
        state_report["holdout_dataset_fingerprint_sha256"]
        != holdout_report["dataset_fingerprint_sha256"]
    ):
        raise RuntimeError("forward state is bound to a different holdout")
    if (
        state_report["same_day_order_version"]
        != state_spec.get("same_day_order_version")
    ):
        raise RuntimeError("same-day ordering differs from protocol")
    if state_report["prior_contract"] != state_spec.get("prior_contract"):
        raise RuntimeError("state prior contract differs from protocol")
    sqlite_path = state_dir / "player_stats_cache_v3.sqlite"
    if sha256_file(sqlite_path) != state_spec.get("sqlite_sha256"):
        raise RuntimeError("state SQLite hash differs from protocol")

    verified_artifacts = _verify_artifacts(protocol)
    opening = protocol.get("opening_conditions") or {}
    missing_conditions = [
        name for name in REQUIRED_OPENING_CONDITIONS
        if opening.get(name) is not True
    ]
    if set(opening) != set(REQUIRED_OPENING_CONDITIONS):
        raise RuntimeError("opening-condition keys differ from contract schema")

    status = protocol["status"]
    if status == "FROZEN" and missing_conditions:
        raise RuntimeError(
            "FROZEN protocol has unmet opening conditions: "
            + ", ".join(missing_conditions)
        )
    scoring_allowed = status == "FROZEN" and not missing_conditions
    if require_frozen and not scoring_allowed:
        raise RuntimeError(
            "model scoring is blocked until the protocol is FROZEN and every "
            "opening condition is true"
        )

    return {
        "status": "PASS",
        "protocol_id": protocol["protocol_id"],
        "protocol_status": status,
        "protocol_sha256": sha256_file(protocol_path),
        "holdout_fingerprint_sha256": holdout_report[
            "dataset_fingerprint_sha256"
        ],
        "state_fingerprint_sha256": state_report[
            "directory_fingerprint_sha256"
        ],
        "selected_matches": holdout_report["selected_matches"],
        "liquidity_slices": live_slices,
        "candidate_artifacts_verified": len(verified_artifacts),
        "opening_condition_blockers": missing_conditions,
        "scoring_allowed": scoring_allowed,
        "model_imports_performed": False,
        "model_scoring_performed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", type=Path)
    parser.add_argument(
        "--require-frozen",
        action="store_true",
        help="Fail unless the protocol is FROZEN and all conditions are met.",
    )
    args = parser.parse_args()
    report = preflight(args.protocol, require_frozen=args.require_frozen)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
