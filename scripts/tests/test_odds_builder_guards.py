"""Regression tests for review finding ODDS2 (2026-08-14).

The odds builders' default output paths ARE the benchmarks of record
(betting_odds_polymarket_v2.json / betting_odds_golden_v2.json), so a bare
rerun of `write_outputs` must refuse to replace an existing manifest unless
the operator opts in with --overwrite-existing-odds (mirroring the women's
builder's --overwrite guard).
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

import build_polymarket_odds as base  # noqa: E402


def _point_outputs_at(tmp_path, monkeypatch):
    monkeypatch.setattr(base, "OUT_ODDS_PATH", tmp_path / "odds.json")
    monkeypatch.setattr(base, "OUT_TEST_DIR", tmp_path / "test_dir")
    monkeypatch.setattr(
        base, "OUT_UNMATCHED_PATH", tmp_path / "unmatched.json"
    )


def test_write_outputs_refuses_existing_manifest(tmp_path, monkeypatch):
    _point_outputs_at(tmp_path, monkeypatch)
    (tmp_path / "odds.json").write_text("{}")
    monkeypatch.setattr(base, "ALLOW_ODDS_OVERWRITE", False)
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        base.write_outputs([], [])
    assert not (tmp_path / "test_dir").exists(), \
        "the guard must fire before any side effect (test-dir copies)"
    assert (tmp_path / "odds.json").read_text() == "{}", \
        "the existing manifest must be untouched"


def test_write_outputs_overwrites_with_explicit_flag(tmp_path, monkeypatch):
    _point_outputs_at(tmp_path, monkeypatch)
    (tmp_path / "odds.json").write_text("{}")
    monkeypatch.setattr(base, "ALLOW_ODDS_OVERWRITE", True)
    base.write_outputs([], [])
    assert (tmp_path / "odds.json").read_text() != "{}"


def test_write_outputs_fresh_path_needs_no_flag(tmp_path, monkeypatch):
    _point_outputs_at(tmp_path, monkeypatch)
    monkeypatch.setattr(base, "ALLOW_ODDS_OVERWRITE", False)
    base.write_outputs([], [])
    assert (tmp_path / "odds.json").is_file()
