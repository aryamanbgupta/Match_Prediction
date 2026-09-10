from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sim_eval.market_math import CostModel
from sim_eval.reslice_eval_json import reslice
from sim_eval import blend_report


FIXTURES = Path(__file__).parent / "fixtures"
EVAL = FIXTURES / "reslice_market_math_synthetic.json"
ODDS = FIXTURES / "reslice_market_math_odds.json"


def test_synthetic_fixture_has_required_coverage():
    matches = json.loads(EVAL.read_text())["matches"]
    assert len(matches) >= 30
    assert any(max(m["edge"].values()) <= 0 for m in matches)
    assert any(min(m["market_odds"].values()) < 1 for m in matches)
    assert any(m["actual_winner"] is None for m in matches)


def test_reslice_zero_cost_parity_and_provenance():
    result = reslice(str(EVAL), str(ODDS), None, n_resamples=100)
    source = json.loads(EVAL.read_text())["matches"]
    source_by_id = {m["match_id"]: m for m in source}
    for row in result["matches"]:
        assert row["realized_pnl"] == source_by_id[row["match_id"]]["realized_pnl"]
    summary = result["summary"]
    assert summary["cost_model"] == CostModel.none().as_dict()
    assert summary["price_basis"] == "mid"
    assert summary["volume_basis"] == "event"


@pytest.mark.parametrize("corrupt_value", [999999.0, None])
def test_reslice_ignores_stored_pnl_and_nonzero_spread_changes_profit(
    tmp_path, corrupt_value
):
    payload = copy.deepcopy(json.loads(EVAL.read_text()))
    for match in payload["matches"]:
        if match.get("bet_placed") is True:
            assert match.get("bet_team") is not None
            match["realized_pnl"] = corrupt_value
    corrupted = tmp_path / "corrupted.json"
    corrupted.write_text(json.dumps(payload))
    baseline = reslice(str(EVAL), str(ODDS), None, n_resamples=100)
    corrupt = reslice(str(corrupted), str(ODDS), None, n_resamples=100)
    assert corrupt["matches"] == baseline["matches"]
    corrupt_summary = dict(corrupt["summary"])
    baseline_summary = dict(baseline["summary"])
    corrupt_summary.pop("reslice_source")
    baseline_summary.pop("reslice_source")
    assert corrupt_summary == baseline_summary
    spread = reslice(
        str(EVAL), str(ODDS), None, n_resamples=100,
        cost_model=CostModel(100, 0, "winnings"),
    )
    assert spread["summary"]["flat_betting_total_pnl"] != baseline["summary"]["flat_betting_total_pnl"]


def test_reslice_missing_market_odds_cost_policy_and_counter(tmp_path):
    payload = {
        "matches": [{
            "match_id": "new-format-missing-price",
            "teams": ["A", "B"],
            "actual_winner": "A",
            "edge": {"A": 0.1, "B": -0.1},
            "bet_placed": True,
            "bet_team": "A",
            "realized_pnl": 7.5,
            "log_loss": 0.5,
            "brier_score": 0.2,
        }]
    }
    eval_path = tmp_path / "missing-price.json"
    odds_path = tmp_path / "odds.json"
    eval_path.write_text(json.dumps(payload))
    odds_path.write_text('{"matches": []}')

    zero = reslice(str(eval_path), str(odds_path), None, n_resamples=100)
    assert zero["matches"][0]["realized_pnl"] == 7.5
    assert zero["summary"]["pnl_unrecomputable"] == 1

    cost = reslice(
        str(eval_path), str(odds_path), None, n_resamples=100,
        cost_model=CostModel(100, 0, "winnings"),
    )
    assert cost["matches"][0]["realized_pnl"] is None
    assert cost["summary"]["pnl_unrecomputable"] == 1
    assert cost["summary"]["flat_betting_bets_placed"] == 0


def test_legacy_placement_reconstruction_still_uses_settledness_sentinel(
    tmp_path,
):
    """Legacy P&L is only a placement sentinel here, never copied profit."""
    base = {
        "match_id": "legacy",
        "teams": ["A", "B"],
        "actual_winner": "A",
        "edge": {"A": 0.1, "B": -0.1},
        "market_odds": {"A": 2.0, "B": 2.0},
        "log_loss": 0.5,
        "brier_score": 0.2,
    }
    odds_path = tmp_path / "odds.json"
    odds_path.write_text('{"matches": []}')
    rows = []
    for sentinel in (123.0, None):
        eval_path = tmp_path / f"legacy-{sentinel}.json"
        eval_path.write_text(json.dumps({"matches": [{**base, "realized_pnl": sentinel}]}))
        rows.append(
            reslice(str(eval_path), str(odds_path), None, n_resamples=100)["matches"][0]
        )
    assert rows[0]["bet_placed"] is True
    assert rows[0]["realized_pnl"] == 1.0
    assert rows[1]["bet_placed"] is False


def test_event_and_market_volume_bases_select_different_rows():
    event = reslice(
        str(EVAL), str(ODDS), 50_000, n_resamples=100,
        volume_basis="event",
    )
    market = reslice(
        str(EVAL), str(ODDS), 50_000, n_resamples=100,
        volume_basis="market",
    )
    assert event["summary"]["n_matches_evaluated"] == 20
    assert market["summary"]["n_matches_evaluated"] == 15
    assert event["summary"]["volume_basis"] == "event"
    assert market["summary"]["volume_basis"] == "market"


def test_blend_report_scenarios_are_configurable_and_zero_cost_first(tmp_path):
    sliced = tmp_path / "sliced"
    sliced.mkdir()
    payload = reslice(str(EVAL), str(ODDS), None, n_resamples=100)
    (sliced / "synthetic_w0p00_all.json").write_text(json.dumps(payload))
    configured = blend_report._parse_cost_scenarios(
        "200:0:winnings,100:100:stake"
    )
    assert configured[0] == CostModel.none()
    markdown = blend_report.render_markdown(
        sliced, tmp_path / "unused-direct.json", configured
    )
    header = next(
        line for line in markdown.splitlines()
        if "gate safety check" in line and line.startswith("| w |")
    )
    assert header.index("0/0 bps winnings") < header.index("200/0 bps winnings")
    assert "100/100 bps stake" in header
