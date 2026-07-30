"""Tests for the post-lock forward outcome and odds evaluator."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate_forward_predictions as evaluator_module  # noqa: E402
from evaluate_forward_predictions import (  # noqa: E402
    betting_summary,
    build_evaluation_report,
    evaluate,
    join_evaluation_rows,
    load_locked_prediction_artifact,
    probability_summary,
    write_locked_report,
)
from forward_eval_contract import load_protocol  # noqa: E402
from score_forward_match_m7 import write_locked_artifact  # noqa: E402


PROTOCOL_PATH = (
    ROOT / "evaluation" / "forward_protocol_2026-06-01_2026-07-13.yaml"
)


def _protocol():
    return {
        "protocol_id": "synthetic",
        "holdout": {
            "selected_matches": 4,
            "liquidity_slices": {
                "all": 4,
                "min_volume_50000": 3,
                "min_volume_100000": 2,
            },
        },
        "statistics": {
            "bootstrap_seed": 42,
            "bootstrap_resamples": 100,
        },
        "reporting": {
            "slices": [
                "all",
                "min_volume_50000",
                "min_volume_100000",
            ],
        },
        "betting_policies": {
            "flat": {
                "stake_units": 1.0,
                "minimum_edge": 0.0,
                "edge_comparison": "strictly_greater",
            },
            "a7": {
                "stake_units": 1.0,
                "elo_boundary": 5.0,
                "close_minimum_edge": 0.0,
                "mismatch_minimum_edge": 0.10,
                "edge_comparison": "strictly_greater",
            },
        },
        "decision": {
            "primary_slice": "min_volume_50000",
            "probability_market_log_loss_tolerance": 0.01,
            "probability_requires_better_than_ball_v7": True,
            "economic_policy": "a7",
            "economic_requires_positive_ci_lower_bound": True,
            "economic_requires_minimum_clusters": 10,
        },
    }


def _manifest():
    volumes = [49_999.99, 50_000.0, 100_000.0, 100_001.0]
    return [
        {
            "match_id": f"2026-01-0{i}_A_B_Ground",
            "cricsheet_id": f"00{i}",
            "date": f"2026-01-0{i}",
            "teams": ["A", "B"],
            "venue": "Ground",
            "competition": "Synthetic League",
            "volume_usd": volume,
        }
        for i, volume in enumerate(volumes, 1)
    ]


def _artifacts():
    m7_probs = [0.70, 0.55, 0.70, 0.60]
    ball_probs = [0.60, 0.45, 0.55, 0.55]
    elo = [1.0, 2.0, 10.0, 10.0]
    manifest = _manifest()

    def rows(probabilities, include_elo):
        result = []
        for row, probability, diff in zip(manifest, probabilities, elo):
            prediction = {
                "match_id": row["match_id"],
                "cricsheet_id": row["cricsheet_id"],
                "date": row["date"],
                "team1": "A",
                "team2": "B",
                "p_team1": probability,
                "p_team2": 1.0 - probability,
            }
            if include_elo:
                prediction["top6_batting_elo_diff"] = diff
            result.append(prediction)
        return result

    return {
        "predictions": rows(m7_probs, True),
    }, {
        "predictions": rows(ball_probs, False),
    }


def _odds(actuals=("A", "B", "A", "A")):
    rows = []
    for manifest_row, actual in zip(_manifest(), actuals):
        rows.append({
            "match_id": manifest_row["match_id"],
            "date": manifest_row["date"],
            "team1": "A",
            "team2": "B",
            "actual_winner": actual,
            "polymarket_volume_usd": manifest_row["volume_usd"],
            "odds": {"winner": {"A": 2.0, "B": 2.0}},
        })
    return {"matches": rows}


def _joined(actuals=("A", "B", "A", "A")):
    m7, ball = _artifacts()
    clusters = {
        row["match_id"]: f"cluster-{index // 2}"
        for index, row in enumerate(_manifest())
    }
    return join_evaluation_rows(
        _protocol(),
        m7,
        ball,
        _odds(actuals),
        clusters,
        manifest_rows=_manifest(),
    )


def test_exact_join_preserves_boundaries_and_unresolved_inventory():
    rows = _joined(("A", None, "A", "A"))
    assert len(rows) == 4
    assert rows[1]["result_status"] == "unresolved"
    assert rows[1]["actual_winner"] is None
    assert sum(row["polymarket_volume_usd"] >= 50_000 for row in rows) == 3
    assert sum(row["polymarket_volume_usd"] >= 100_000 for row in rows) == 2


def test_missing_alias_or_reversed_team_order_fails_closed():
    m7, ball = _artifacts()
    odds = _odds()
    odds["matches"][0]["team1"] = "Team A Alias"
    with pytest.raises(RuntimeError, match="odds identity/team order"):
        join_evaluation_rows(
            _protocol(),
            m7,
            ball,
            odds,
            manifest_rows=_manifest(),
        )

    odds = _odds()
    reversed_ball = copy.deepcopy(ball)
    reversed_ball["predictions"][0]["team1"] = "B"
    reversed_ball["predictions"][0]["team2"] = "A"
    with pytest.raises(RuntimeError, match="team order mismatch"):
        join_evaluation_rows(
            _protocol(),
            m7,
            reversed_ball,
            odds,
            manifest_rows=_manifest(),
        )

    with pytest.raises(RuntimeError, match="fixture set mismatch"):
        join_evaluation_rows(
            _protocol(),
            m7,
            ball,
            {"matches": _odds()["matches"][:-1]},
            manifest_rows=_manifest(),
        )


def test_probability_and_betting_math_including_strict_a7_boundary():
    rows = _joined()
    summary = probability_summary(
        rows,
        "predictions.match_m7",
        resamples=100,
        seed=42,
    )
    expected_ll = (
        -__import__("math").log(0.70)
        - __import__("math").log(0.45)
        - __import__("math").log(0.70)
        - __import__("math").log(0.60)
    ) / 4
    assert summary["binary_log_loss"] == pytest.approx(expected_ll)
    assert summary["brier_score"] == pytest.approx(
        (0.30**2 + 0.55**2 + 0.30**2 + 0.40**2) / 4
    )

    policies = _protocol()["betting_policies"]
    flat = betting_summary(
        rows,
        "match_m7",
        {"name": "flat", **policies["flat"]},
        resamples=100,
        seed=42,
    )
    assert flat["n_bets"] == 4
    assert flat["total_pnl"] == 2.0
    assert flat["win_rate"] == 0.75

    a7 = betting_summary(
        rows,
        "match_m7",
        {"name": "a7", **policies["a7"]},
        resamples=100,
        seed=42,
    )
    # Close rows 1/2 are kept; row 3 has 20-point mismatch edge and is kept;
    # row 4 has exactly 10 points and is dropped by strictly-greater.
    assert a7["n_bets"] == 3
    assert [bet["match_id"] for bet in a7["bets"]] == [
        row["match_id"] for row in rows[:3]
    ]


def test_zero_return_win_is_still_a_bet_and_a_win():
    row = _joined()[2]
    row["decimal_odds"]["A"] = 1.0
    row["market_probability"] = {"A": 0.6, "B": 0.4}
    row["predictions"]["match_m7"] = {"A": 0.9, "B": 0.1}
    policy = {
        "name": "flat",
        "stake_units": 1.0,
        "minimum_edge": 0.0,
        "edge_comparison": "strictly_greater",
    }
    result = betting_summary(
        [row],
        "match_m7",
        policy,
        resamples=20,
        seed=42,
    )
    assert result["n_bets"] == 1
    assert result["total_pnl"] == 0.0
    assert result["win_rate"] == 1.0


def test_report_contains_all_frozen_slices_policies_and_decision():
    protocol = _protocol()
    report = build_evaluation_report(
        protocol,
        {
            "protocol_sha256": "p",
            "holdout_fingerprint_sha256": "h",
        },
        _joined(),
        prediction_artifact_sha256={"match_m7": "m", "ball_v7": "b"},
    )
    assert list(report["slices"]) == [
        "all",
        "min_volume_50000",
        "min_volume_100000",
    ]
    assert report["slices"]["min_volume_50000"]["n_matches"] == 3
    assert set(
        report["slices"]["all"]["models"]["match_m7"]["betting"]
    ) == {"flat", "a7"}
    assert report["decision_assessment"][
        "economic_cluster_requirement_met"
    ] is False
    json.dumps(report, allow_nan=False)


def _locked_prediction(model_id, protocol_sha="protocol"):
    return {
        "schema_version": 1,
        "artifact_type": "locked_outcome_free_predictions",
        "model_id": model_id,
        "protocol_id": "synthetic",
        "protocol_sha256": protocol_sha,
        "holdout_fingerprint_sha256": "holdout",
        "state_fingerprint_sha256": "state",
        "prediction_count": 0,
        "outcomes_joined": False,
        "predictions": [],
    }


def test_locked_prediction_checksum_and_report_write_once(tmp_path):
    path = tmp_path / "m7.json"
    artifact = _locked_prediction("match_m7")
    write_locked_artifact(path, artifact)
    loaded = load_locked_prediction_artifact(
        path,
        expected_model_id="match_m7",
        protocol={"protocol_id": "synthetic"},
        preflight_report={
            "protocol_sha256": "protocol",
            "holdout_fingerprint_sha256": "holdout",
            "state_fingerprint_sha256": "state",
        },
    )
    assert loaded == artifact

    report_path = tmp_path / "report.json"
    write_locked_report(report_path, {"status": "ok"})
    with pytest.raises(FileExistsError):
        write_locked_report(report_path, {"status": "ok"})

    path.write_text(path.read_text() + " ")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        load_locked_prediction_artifact(
            path,
            expected_model_id="match_m7",
            protocol={"protocol_id": "synthetic"},
            preflight_report={
                "protocol_sha256": "protocol",
                "holdout_fingerprint_sha256": "holdout",
                "state_fingerprint_sha256": "state",
            },
        )


def test_preflight_gate_precedes_prediction_or_outcome_reads(
    tmp_path,
    monkeypatch,
):
    def _blocked_preflight(_path, *, require_frozen):
        assert require_frozen is True
        raise RuntimeError("synthetic preflight block")

    monkeypatch.setattr(
        evaluator_module,
        "preflight",
        _blocked_preflight,
    )
    output = tmp_path / "must_not_exist.json"
    with pytest.raises(RuntimeError, match="synthetic preflight block"):
        evaluate(
            PROTOCOL_PATH,
            tmp_path / "missing_m7.json",
            tmp_path / "missing_ball.json",
            output,
        )
    assert not output.exists()
    assert not output.with_suffix(".json.sha256").exists()


def test_consumed_protocol_remains_frozen():
    assert load_protocol(PROTOCOL_PATH)["status"] == "FROZEN"
