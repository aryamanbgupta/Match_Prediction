"""Pins the T1 snapshot-count guard (2026-08-14 implementation review).

The T1 parity audit certifies only the live same-day replay count path.
`OnlineT1OutcomeDists` used to fall back SILENTLY to date-snapshot SQLite
counts (which exclude earlier same-day matches) whenever the provider
lacked `get_t1_outcome_counts` — the path `run_sim_eval_t1.py`'s plain
StatsProvider selects. That fallback must now fail closed unless
explicitly overridden.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from sim_t1 import OnlineT1OutcomeDists  # noqa: E402


class _LiveProvider:
    """Duck-types the audited replay-provider surface."""

    def get_t1_outcome_counts(self, *a, **k):
        return [0] * 6

    def get_t1_outcome_prior(self, *a, **k):
        return (1 / 6,) * 6


class _StubBackend:
    _prior = (1 / 6,) * 6

    def _ensure_conn(self):
        pass


class _SnapshotOnlyProvider:
    """A plain provider without live same-day counts."""

    _backend = _StubBackend()


def test_snapshot_count_path_fails_closed(monkeypatch):
    monkeypatch.delenv("T1_ALLOW_SNAPSHOT_COUNTS", raising=False)
    with pytest.raises(RuntimeError, match="parity audit"):
        OnlineT1OutcomeDists(_SnapshotOnlyProvider(), metadata=None)


def test_snapshot_count_path_needs_explicit_override(monkeypatch):
    monkeypatch.setenv("T1_ALLOW_SNAPSHOT_COUNTS", "1")
    dists = OnlineT1OutcomeDists(_SnapshotOnlyProvider(), metadata=None)
    assert dists.live_counts is False


def test_live_count_path_constructs_without_override(monkeypatch):
    monkeypatch.delenv("T1_ALLOW_SNAPSHOT_COUNTS", raising=False)
    dists = OnlineT1OutcomeDists(_LiveProvider(), metadata=None)
    assert dists.live_counts is True
