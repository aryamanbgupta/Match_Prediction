"""Unit tests for EmpiricalBowlerSelector weight math.

Builds a tiny synthetic usage prior + fake match state, samples many times,
and asserts that empirical pick proportions match the analytic weights via
a chi-squared goodness-of-fit test. Catches EB-formulation regressions
without running the full sim.

Run:
    uv run python -m pytest scripts/tests/test_bowler_selector.py -v
"""
from __future__ import annotations

import json
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Importing sim_v1_2 loads heavy modules; import only the selector pieces.
from sim_v1_2 import EmpiricalBowlerSelector  # noqa: E402


@dataclass
class _FakePlayer:
    player_id: str


@dataclass
class _FakeLineup:
    players: List[_FakePlayer]


@dataclass
class _FakeState:
    balls: int
    bowling_lineup: _FakeLineup
    match_date: datetime


@pytest.fixture
def usage_file(tmp_path: Path) -> Path:
    """Three bowlers with sharply different phase profiles.

    A: death specialist  — 100 death balls only
    B: powerplay opener  — 100 pp balls only
    C: balanced workhorse — 50 pp, 50 mid, 50 death
    """
    payload = {
        "schema_version": 1,
        "built_at": "test",
        "by_player": {
            "A": {"2020": {"pp": 0, "mid": 0, "death": 100, "total": 100}},
            "B": {"2020": {"pp": 100, "mid": 0, "death": 0, "total": 100}},
            "C": {"2020": {"pp": 50, "mid": 50, "death": 50, "total": 150}},
        },
        "by_year_league": {
            "2020": {
                "pp_share": 0.32,
                "mid_share": 0.51,
                "death_share": 0.17,
                "total_balls": 1000,
            },
        },
        "global_league": {
            "pp_share": 0.32,
            "mid_share": 0.51,
            "death_share": 0.17,
            "total_balls": 1000,
        },
    }
    fp = tmp_path / "usage.json"
    fp.write_text(json.dumps(payload))
    return fp


def _expected_weights(phase: str, k: int = 30):
    """Hand-computed weights = phase_balls + k * league_share."""
    league = {"pp": 0.32, "mid": 0.51, "death": 0.17}
    alpha = k * league[phase]
    counts = {
        "A": {"pp": 0, "mid": 0, "death": 100},
        "B": {"pp": 100, "mid": 0, "death": 0},
        "C": {"pp": 50, "mid": 50, "death": 50},
    }
    w = {p: counts[p][phase] + alpha for p in ("A", "B", "C")}
    total = sum(w.values())
    return {p: w[p] / total for p in w}


def _sample_proportions(selector, state, n: int = 20_000):
    n_players = len(state.bowling_lineup.players)
    counts = {i: 0 for i in range(n_players)}
    available = list(range(n_players))
    for _ in range(n):
        idx = selector.select_bowler(state, available)
        counts[idx] += 1
    return {k: v / n for k, v in counts.items()}


def _chi_sq(observed: List[float], expected: List[float], n: int) -> float:
    """χ² statistic for goodness-of-fit (binned proportions × n)."""
    return sum(((o - e) ** 2) * n / e for o, e in zip(observed, expected) if e > 0)


def _make_state(balls: int) -> _FakeState:
    return _FakeState(
        balls=balls,
        bowling_lineup=_FakeLineup(players=[
            _FakePlayer("A"), _FakePlayer("B"), _FakePlayer("C"),
        ]),
        match_date=datetime(2025, 6, 1),  # so cumulative looks at year < 2025
    )


@pytest.mark.parametrize("balls,phase", [(0, "pp"), (60, "mid"), (110, "death")])
def test_selector_matches_analytic_weights(usage_file: Path, balls: int, phase: str):
    """Empirical proportions should match (phase_balls + α) / Σ analytically."""
    random.seed(20260512)
    selector = EmpiricalBowlerSelector(usage_path=str(usage_file), k=30)
    state = _make_state(balls)
    n = 20_000
    obs = _sample_proportions(selector, state, n=n)
    exp = _expected_weights(phase)

    # All three picks should be within ~0.02 of expected at n=20k.
    for letter, idx in (("A", 0), ("B", 1), ("C", 2)):
        assert abs(obs[idx] - exp[letter]) < 0.02, (
            f"phase={phase} bowler={letter}: observed {obs[idx]:.3f} "
            f"vs expected {exp[letter]:.3f}"
        )

    # χ² with df=2: critical value at p=0.001 is 13.82; we're checking a
    # MUCH looser bound to allow honest sampling noise.
    obs_vec = [obs[0], obs[1], obs[2]]
    exp_vec = [exp["A"], exp["B"], exp["C"]]
    chi = _chi_sq(obs_vec, exp_vec, n)
    assert chi < 13.82, f"χ² = {chi:.2f} too high for phase {phase}"


def test_unknown_bowler_falls_back_to_league_prior(usage_file: Path):
    """A bowler with no history should get weight = k * league_share."""
    random.seed(20260512)
    selector = EmpiricalBowlerSelector(usage_path=str(usage_file), k=30)
    state = _FakeState(
        balls=0,
        bowling_lineup=_FakeLineup(players=[
            _FakePlayer("unknown_1"),
            _FakePlayer("unknown_2"),
        ]),
        match_date=datetime(2025, 6, 1),
    )
    # Two unknown bowlers in PP — both get α = 30 * 0.32 = 9.6.
    # Weights tie ⇒ 50/50 sampling.
    obs = _sample_proportions(selector, state, n=20_000)
    assert abs(obs[0] - 0.5) < 0.02
    assert abs(obs[1] - 0.5) < 0.02


def test_known_specialist_dominates_unknown_in_their_phase(usage_file: Path):
    """A's 100 death balls (weight 100 + 5.1 ≈ 105.1) should crush an
    unknown bowler (weight 5.1) in death overs."""
    random.seed(20260512)
    selector = EmpiricalBowlerSelector(usage_path=str(usage_file), k=30)
    state = _FakeState(
        balls=110,  # death
        bowling_lineup=_FakeLineup(players=[
            _FakePlayer("A"), _FakePlayer("unknown_1"),
        ]),
        match_date=datetime(2025, 6, 1),
    )
    obs = _sample_proportions(selector, state, n=20_000)
    # Expected: A ≈ 105.1 / (105.1 + 5.1) ≈ 0.954
    assert obs[0] > 0.93, f"A's death dominance is too weak: {obs[0]:.3f}"


def test_as_of_year_excludes_future_data(tmp_path: Path):
    """Cumulative for year=2020 must NOT include 2020 or 2021 buckets."""
    payload = {
        "schema_version": 1,
        "by_player": {
            "P": {
                "2018": {"pp": 10, "mid": 20, "death": 5, "total": 35},
                "2019": {"pp": 30, "mid": 50, "death": 20, "total": 100},
                "2020": {"pp": 100, "mid": 100, "death": 100, "total": 300},
                "2021": {"pp": 999, "mid": 999, "death": 999, "total": 2997},
            },
        },
        "by_year_league": {
            "2019": {"pp_share": 0.32, "mid_share": 0.51,
                     "death_share": 0.17, "total_balls": 1000},
        },
        "global_league": {"pp_share": 0.32, "mid_share": 0.51,
                          "death_share": 0.17, "total_balls": 1000},
    }
    fp = tmp_path / "u.json"
    fp.write_text(json.dumps(payload))
    sel = EmpiricalBowlerSelector(usage_path=str(fp))
    as_of_2020 = sel._as_of(2020)
    # Should sum 2018 + 2019 only: pp=10+30=40, mid=20+50=70, death=5+20=25.
    assert as_of_2020["P"] == {"pp": 40, "mid": 70, "death": 25, "total": 135}
