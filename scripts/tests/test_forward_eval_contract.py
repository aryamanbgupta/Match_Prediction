"""Tests for the model-free forward-evaluation preflight contract."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from forward_eval_contract import (  # noqa: E402
    liquidity_slice_counts,
    preflight,
    repo_path,
)


PROTOCOL = (
    ROOT / "evaluation" / "forward_protocol_2026-06-01_2026-07-13.yaml"
)


def test_draft_preflight_verifies_everything_without_scoring():
    report = preflight(PROTOCOL)
    assert report["status"] == "PASS"
    assert report["protocol_status"] == "DRAFT"
    assert report["selected_matches"] == 137
    assert report["liquidity_slices"] == {
        "all": 137,
        "min_volume_50000": 61,
        "min_volume_100000": 30,
    }
    assert report["candidate_artifacts_verified"] == 14
    assert report["scoring_allowed"] is False
    assert report["model_imports_performed"] is False
    assert report["model_scoring_performed"] is False
    assert report["opening_condition_blockers"] == [
        "scorer_tests_complete",
        "ball_same_day_replay_complete",
        "scoring_code_hashes_recorded",
        "user_approved",
    ]


def test_require_frozen_fails_closed_on_draft():
    with pytest.raises(RuntimeError, match="model scoring is blocked"):
        preflight(PROTOCOL, require_frozen=True)


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
