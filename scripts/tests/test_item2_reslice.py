from __future__ import annotations

import copy
import json
from pathlib import Path

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


def test_reslice_ignores_stored_pnl_and_nonzero_spread_changes_profit(tmp_path):
    payload = copy.deepcopy(json.loads(EVAL.read_text()))
    for match in payload["matches"]:
        match["realized_pnl"] = 999999.0
    corrupted = tmp_path / "corrupted.json"
    corrupted.write_text(json.dumps(payload))
    baseline = reslice(str(EVAL), str(ODDS), None, n_resamples=100)
    corrupt = reslice(str(corrupted), str(ODDS), None, n_resamples=100)
    assert corrupt["summary"]["flat_betting_total_pnl"] == baseline["summary"]["flat_betting_total_pnl"]
    spread = reslice(
        str(EVAL), str(ODDS), None, n_resamples=100,
        cost_model=CostModel(100, 0, "winnings"),
    )
    assert spread["summary"]["flat_betting_total_pnl"] != baseline["summary"]["flat_betting_total_pnl"]


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
