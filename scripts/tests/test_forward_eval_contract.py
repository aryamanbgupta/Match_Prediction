"""Tests for the model-free forward-evaluation preflight contract."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import forward_eval_contract as contract_module  # noqa: E402
from forward_eval_contract import (  # noqa: E402
    _verify_scoring_code,
    liquidity_slice_counts,
    load_protocol,
    preflight,
    repo_path,
    sha256_file,
)


PROTOCOL = (
    ROOT / "evaluation" / "forward_protocol_2026-06-01_2026-07-13.yaml"
)


def test_consumed_preflight_fails_closed_after_source_drift():
    assert load_protocol(PROTOCOL)["status"] == "FROZEN"
    with pytest.raises(RuntimeError, match="artifact hash mismatch"):
        preflight(PROTOCOL)


def test_require_frozen_fails_closed_on_draft(tmp_path, monkeypatch):
    protocol = load_protocol(PROTOCOL)
    protocol["status"] = "DRAFT"
    draft = tmp_path / "protocol.yaml"
    draft.write_text(yaml.safe_dump(protocol, sort_keys=False))
    monkeypatch.setattr(
        contract_module,
        "_verify_artifacts",
        lambda _protocol: [],
    )
    monkeypatch.setattr(
        contract_module,
        "_verify_scoring_code",
        lambda _protocol: [],
    )
    with pytest.raises(RuntimeError, match="model scoring is blocked"):
        preflight(draft, require_frozen=True)


def test_liquidity_boundaries_are_inclusive():
    document = {
        "matches": [
            {"polymarket_volume_usd": 49_999.99},
            {"polymarket_volume_usd": 50_000},
            {"polymarket_volume_usd": 99_999.99},
            {"polymarket_volume_usd": 100_000},
            {"polymarket_volume_usd": None},
        ]
    }
    assert liquidity_slice_counts(document) == {
        "all": 5,
        "min_volume_50000": 3,
        "min_volume_100000": 1,
    }


def test_protocol_paths_cannot_escape_repository():
    with pytest.raises(RuntimeError, match="escapes repository"):
        repo_path("../outside")


def test_tampered_fingerprint_is_rejected(tmp_path: Path):
    tampered = tmp_path / "protocol.yaml"
    text = PROTOCOL.read_text().replace(
        "82ccde16cf2b7e5f13a9236f2788f3c8be1582f312f5c028ec44a6ab76561028",
        "0" * 64,
        1,
    )
    tampered.write_text(text)
    with pytest.raises(RuntimeError, match="holdout fingerprint"):
        preflight(tampered)


def test_tampered_scoring_code_hash_is_rejected():
    protocol = load_protocol(PROTOCOL)
    artifacts = protocol["scoring_code"]["artifacts"]
    for artifact in artifacts:
        artifact["sha256"] = sha256_file(repo_path(artifact["path"]))
    artifacts[0]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="scoring-code hash mismatch"):
        _verify_scoring_code(protocol)
