"""Pins the T1 live-count requirement (2026-08-14 review; hardened 2026-08-22).

The T1 parity audit certifies only the live same-day replay count path.
`OnlineT1OutcomeDists` used to fall back SILENTLY to date-snapshot SQLite
counts (which exclude earlier same-day matches) whenever the provider lacked
`get_t1_outcome_counts` — the path `run_sim_eval_t1.py`'s plain StatsProvider
selected. That runner now builds a `SameDayReplayStatsProvider`, and the
snapshot fallback (with its `T1_ALLOW_SNAPSHOT_COUNTS` override) is deleted:
a provider without live counts is refused unconditionally.
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


class _SnapshotOnlyProvider:
    """A plain provider without live same-day counts."""


def test_snapshot_provider_is_refused_unconditionally(monkeypatch):
    # The old escape hatch must stay dead: the env var no longer exists.
    monkeypatch.setenv("T1_ALLOW_SNAPSHOT_COUNTS", "1")
    with pytest.raises(RuntimeError, match="parity audit"):
        OnlineT1OutcomeDists(_SnapshotOnlyProvider(), metadata=None)


def test_live_count_provider_constructs():
    dists = OnlineT1OutcomeDists(_LiveProvider(), metadata=None)
    assert dists.provider.get_t1_outcome_prior() == (1 / 6,) * 6
