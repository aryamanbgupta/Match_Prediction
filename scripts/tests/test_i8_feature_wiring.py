"""Train/materialize/simulation parity tests for the 18 I8 features."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from parsing_v2 import (  # noqa: E402
    PlayerStatsTracker,
    parse_match_data_v2,
)
from identity_maps import venue_alias_contract  # noqa: E402
from run_experiment import build_eval_cmd, build_training_cmd  # noqa: E402
from sim_i8 import (  # noqa: E402
    I8_FEATURE_COLUMNS,
    XGBoostModelI8,
    get_i8_outcome_dists,
)
from stats_provider import StatsProviderCache  # noqa: E402
from sim_v1_2 import XGBoostModelV2  # noqa: E402


PRIOR = (0.4, 0.3, 0.1, 0.1, 0.05, 0.05)


def _two_ball_match() -> dict:
    return {
        "info": {
            "venue": "Test Ground",
            "dates": ["2026-01-01"],
            "teams": ["A", "B"],
            "registry": {
                "people": {
                    "Alice": "bat",
                    "Beth": "non",
                    "Cara": "bowl",
                }
            },
            "players": {
                "A": ["Alice", "Beth"],
                "B": ["Cara"],
            },
            "toss": {"winner": "A", "decision": "bat"},
            "team_type": "club",
            "outcome": {"winner": "A"},
        },
        "innings": [
            {
                "team": "A",
                "overs": [
                    {
                        "over": 0,
                        "deliveries": [
                            {
                                "batter": "Alice",
                                "non_striker": "Beth",
                                "bowler": "Cara",
                                "runs": {
                                    "batter": 0,
                                    "extras": 0,
                                    "total": 0,
                                },
                            },
                            {
                                "batter": "Alice",
                                "non_striker": "Beth",
                                "bowler": "Cara",
                                "runs": {
                                    "batter": 0,
                                    "extras": 0,
                                    "total": 0,
                                },
                            },
                        ],
                    }
                ],
            }
        ],
    }


def test_parser_emits_pre_ball_i8_features_without_current_ball_leakage():
    tracker = PlayerStatsTracker(enable_i8=True)
    rows, *_ = parse_match_data_v2(
        json.dumps(_two_ball_match()),
        tracker,
        prior=PRIOR,
        k_player=30.0,
        k_phase=30.0,
        k_h2h=60.0,
    )

    assert len(rows) == 2
    first, second = rows
    assert first["batter_phase_p0"] == pytest.approx(PRIOR[0])
    assert first["bowler_phase_p0"] == pytest.approx(PRIOR[0])
    assert first["h2h_p0"] == pytest.approx(PRIOR[0])
    # The first dot enters state only after row 1, so row 2 moves upward.
    assert second["batter_phase_p0"] > first["batter_phase_p0"]
    assert second["bowler_phase_p0"] > first["bowler_phase_p0"]
    assert second["h2h_p0"] > first["h2h_p0"]

    for row in rows:
        for prefix in ("batter_phase", "bowler_phase", "h2h"):
            assert sum(
                row[f"{prefix}_p{suffix}"]
                for suffix in ("0", "1", "2", "4", "6", "w")
            ) == pytest.approx(1.0)


class _FakeProvider:
    def __init__(self):
        self.i8_calls = []
        self._backend = SimpleNamespace(schema_version=5)

    def get_batter_outcome_dist(self, *args, **kwargs):
        return {}

    def get_bowler_outcome_dist(self, *args, **kwargs):
        return {}

    def get_batter_vs_type_outcome_dist(self, *args, **kwargs):
        return {}

    def get_bowler_vs_hand_outcome_dist(self, *args, **kwargs):
        return {}

    def get_venue_outcome_dist(self, *args, **kwargs):
        return {}

    def get_phase_outcome_dist(self, *args, **kwargs):
        return {}

    def get_batter_phase_outcome_dist(self, *args, **kwargs):
        self.i8_calls.append(("batter", args, kwargs))
        return {
            f"batter_phase_p{suffix}": value
            for suffix, value in zip(
                ("0", "1", "2", "4", "6", "w"),
                (0.41, 0.30, 0.10, 0.09, 0.05, 0.05),
            )
        }

    def get_bowler_phase_outcome_dist(self, *args, **kwargs):
        self.i8_calls.append(("bowler", args, kwargs))
        return {
            f"bowler_phase_p{suffix}": value
            for suffix, value in zip(
                ("0", "1", "2", "4", "6", "w"),
                (0.42, 0.29, 0.10, 0.09, 0.05, 0.05),
            )
        }

    def get_h2h_outcome_dist(self, *args, **kwargs):
        self.i8_calls.append(("h2h", args, kwargs))
        return {
            f"h2h_p{suffix}": value
            for suffix, value in zip(
                ("0", "1", "2", "4", "6", "w"),
                (0.43, 0.28, 0.10, 0.09, 0.05, 0.05),
            )
        }


def test_i8_sim_adapter_queries_all_three_pre_ball_distributions():
    provider = _FakeProvider()
    i8 = get_i8_outcome_dists(
        provider,
        "bat",
        "bowl",
        "2026-01-01",
        36,
        k_player=30.0,
        k_phase=30.0,
        k_h2h=60.0,
    )
    assert i8["batter_phase_p0"] == 0.41
    assert i8["bowler_phase_p0"] == 0.42
    assert i8["h2h_p0"] == 0.43
    assert [call[0] for call in provider.i8_calls] == [
        "batter",
        "bowler",
        "h2h",
    ]
    assert provider.i8_calls[0][1][2] == 36
    assert provider.i8_calls[2][2]["k_h2h"] == 60.0


def test_training_command_carries_schema_and_all_shrinkage_values():
    config = yaml.safe_load(
        (
            ROOT
            / "experiments"
            / "configs"
            / "xgb_i8_phase_matchup.yaml"
        ).read_text()
    )
    cmd = build_training_cmd(config, [])
    payload = json.loads(cmd[cmd.index("--config-json") + 1])

    assert payload["data"]["version"] == "i8"
    assert payload["data"]["cache_schema_version"] == 5
    assert payload["outcome_dist"] == {
        "k_player": 30.0,
        "k_venue": 200.0,
        "k_phase": 30.0,
        "k_h2h": 60.0,
    }
    assert build_eval_cmd(config)[1] == (
        "scripts/sim_eval/run_sim_eval_i8.py"
    )


def _write_minimal_i8_model_artifacts(path: Path) -> dict:
    model_path = path / "xgboost_model_i8.pkl"
    batter_path = path / "batter_encoder_i8.pkl"
    bowler_path = path / "bowler_encoder_i8.pkl"
    features_path = path / "feature_columns_i8.txt"
    joblib.dump(
        SimpleNamespace(n_features_in_=len(I8_FEATURE_COLUMNS)),
        model_path,
    )
    encoder = SimpleNamespace(classes_=np.array(["known"]))
    joblib.dump(encoder, batter_path)
    joblib.dump(encoder, bowler_path)
    features_path.write_text("\n".join(I8_FEATURE_COLUMNS) + "\n")
    (path / "outcome_dist_config_i8.json").write_text(json.dumps({
        "k_player": 30.0,
        "k_venue": 200.0,
        "k_phase": 31.0,
        "k_h2h": 61.0,
    }))
    (path / "training_contract_i8.json").write_text(json.dumps({
        "data_version": "i8",
        "cache_schema_version": 5,
        "venue_identity": venue_alias_contract(),
    }))
    return {
        "model_path": str(model_path),
        "batter_encoder_path": str(batter_path),
        "bowler_encoder_path": str(bowler_path),
        "feature_columns_path": str(features_path),
    }


def test_i8_sim_model_rejects_v4_provider_and_loads_full_sidecar(
    tmp_path: Path,
):
    artifacts = _write_minimal_i8_model_artifacts(tmp_path)
    v4_provider = SimpleNamespace(
        _backend=SimpleNamespace(schema_version=4)
    )
    with pytest.raises(RuntimeError, match="requires SQLite schema 5"):
        XGBoostModelI8(**artifacts, stats_provider=v4_provider)

    v5_provider = SimpleNamespace(
        _backend=SimpleNamespace(schema_version=5)
    )
    model = XGBoostModelI8(**artifacts, stats_provider=v5_provider)
    assert model.k_phase == 31.0
    assert model.k_h2h == 61.0


def test_i8_sim_model_overwrites_all_eighteen_feature_slots(
    tmp_path: Path,
    monkeypatch,
):
    artifacts = _write_minimal_i8_model_artifacts(tmp_path)
    provider = _FakeProvider()
    model = XGBoostModelI8(**artifacts, stats_provider=provider)
    state = SimpleNamespace(
        current_striker=SimpleNamespace(player_id="bat"),
        current_bowler=SimpleNamespace(player_id="bowl"),
        match_date="2026-01-01",
        balls=36,
    )
    monkeypatch.setattr(
        XGBoostModelV2,
        "extract_features",
        lambda self, _state: self._feat_buf,
    )

    row = model.extract_features(state)

    assert row.shape == (18,)
    assert row[model.feature_columns.index("batter_phase_p0")] == 0.41
    assert row[model.feature_columns.index("bowler_phase_p0")] == 0.42
    assert row[model.feature_columns.index("h2h_p0")] == 0.43
    assert np.count_nonzero(row) == 18


def test_phase_memo_keys_by_three_phases_not_every_ball():
    calls = []

    class Provider:
        def get_batter_phase_outcome_dist(
            self,
            player_id,
            as_of_date,
            balls_bowled,
            **kwargs,
        ):
            calls.append(balls_bowled)
            return {"batter_phase_p0": float(balls_bowled)}

    cached = StatsProviderCache(Provider())
    assert cached.get_batter_phase_outcome_dist(
        "bat", "2026-01-01", 0) == {"batter_phase_p0": 0.0}
    # Same phase reuses the first result.
    assert cached.get_batter_phase_outcome_dist(
        "bat", "2026-01-01", 35) == {"batter_phase_p0": 0.0}
    assert cached.get_batter_phase_outcome_dist(
        "bat", "2026-01-01", 36) == {"batter_phase_p0": 36.0}
    assert cached.get_batter_phase_outcome_dist(
        "bat", "2026-01-01", 95) == {"batter_phase_p0": 36.0}
    assert cached.get_batter_phase_outcome_dist(
        "bat", "2026-01-01", 96) == {"batter_phase_p0": 96.0}
    assert calls == [0, 36, 96]
