"""Pre-implementation tests for the frozen I8 feature/config contract."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from feature_registry import I8_GROUPS, resolve_feature_list  # noqa: E402
from parsing_v2 import _classify_phase_pre_ball, _shrink_counts  # noqa: E402


I8_FEATURES = [
    *(f"batter_phase_p{suffix}" for suffix in ("0", "1", "2", "4", "6", "w")),
    *(f"bowler_phase_p{suffix}" for suffix in ("0", "1", "2", "4", "6", "w")),
    *(f"h2h_p{suffix}" for suffix in ("0", "1", "2", "4", "6", "w")),
]


def test_i8_adds_exactly_eighteen_features_to_i7_recipe():
    features = resolve_feature_list(I8_GROUPS)
    assert features[-18:] == I8_FEATURES
    assert len(features) == 132
    assert not any(name.startswith("phase_p") for name in features)


def test_i8_phase_boundaries_are_pre_ball_and_exhaustive():
    assert _classify_phase_pre_ball(0) == "powerplay"
    assert _classify_phase_pre_ball(35) == "powerplay"
    assert _classify_phase_pre_ball(36) == "middle"
    assert _classify_phase_pre_ball(95) == "middle"
    assert _classify_phase_pre_ball(96) == "death"
    assert _classify_phase_pre_ball(119) == "death"


def test_hierarchical_shrinkage_falls_back_to_parent_and_normalizes():
    parent = (0.4, 0.3, 0.1, 0.1, 0.05, 0.05)
    assert _shrink_counts((0, 0, 0, 0, 0, 0), parent, 60.0) == parent

    posterior = _shrink_counts((6, 2, 1, 1, 0, 0), parent, 60.0)
    assert abs(sum(posterior) - 1.0) < 1e-12
    assert posterior[0] > parent[0]
    assert posterior[4] < parent[4]


def test_h2h_parent_is_normalized_arithmetic_player_mean():
    batter = (0.3, 0.4, 0.1, 0.1, 0.05, 0.05)
    bowler = (0.5, 0.2, 0.1, 0.08, 0.07, 0.05)
    parent = tuple((a + b) / 2 for a, b in zip(batter, bowler))
    assert parent == pytest.approx((0.4, 0.3, 0.1, 0.09, 0.06, 0.05))
    assert abs(sum(parent) - 1.0) < 1e-12


def test_i8_config_freezes_schema_identity_and_shrinkage():
    config_path = ROOT / "experiments" / "configs" / "xgb_i8_phase_matchup.yaml"
    config = yaml.safe_load(config_path.read_text())

    assert config["data"]["version"] == "i8"
    assert config["data"]["cache_schema_version"] == 5
    assert config["data"]["delivery_semantics"] == "inclusive_total_runs_v1"
    assert config["outcome_dist"] == {
        "k_player": 30.0,
        "k_venue": 200.0,
        "k_phase": 30.0,
        "k_h2h": 60.0,
    }
    assert resolve_feature_list(config["features"]["groups"]) == (
        resolve_feature_list(I8_GROUPS)
    )
