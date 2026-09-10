"""
Test BettingOddsLoader.load_odds(min_volume=...) drops the right matches.

Phase 1 of the outcome-dist follow-up plan.
"""

import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

from sim_eval.loaders import BettingOddsLoader


def _write(tmpdir: Path, payload: dict) -> str:
    p = tmpdir / "odds.json"
    p.write_text(json.dumps(payload))
    return str(p)


def _payload():
    return {
        "matches": [
            {"match_id": "m_low", "polymarket_volume_usd": 5_000,
             "market_selection": {"market_volume_usd": 250_000}},
            {"match_id": "m_mid", "polymarket_volume_usd": 75_000,
             "market_selection": {"market_volume_usd": 1}},
            {"match_id": "m_boundary", "polymarket_volume_usd": 50_000,
             "market_selection": {"market_volume_usd": 0}},
            {"match_id": "m_high", "polymarket_volume_usd": 250_000},
            {"match_id": "m_no_vol_field",
             "market_selection": {"market_volume_usd": 250_000}},
        ]
    }


def test_no_filter_keeps_everything():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(Path(tmp), _payload())
        result = BettingOddsLoader.load_odds(path, min_volume=None)
        assert set(result.keys()) == {
            "m_low", "m_mid", "m_boundary", "m_high", "m_no_vol_field"
        }, f"min_volume=None should keep all 5 matches, got: {sorted(result.keys())}"


def test_50k_filter_drops_low_and_missing():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(Path(tmp), _payload())
        result = BettingOddsLoader.load_odds(path, min_volume=50_000)
        # load_odds currently has no selectable volume basis: it filters on
        # event-level polymarket_volume_usd, not nested market volume. These
        # opposing event/market values pin that behavior. Exactly 50,000 is
        # inclusive, while a missing event volume is treated as zero.
        assert set(result.keys()) == {"m_mid", "m_boundary", "m_high"}, \
            f"min_volume=50k used the wrong basis/boundary: {sorted(result.keys())}"


def test_100k_filter_only_high():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(Path(tmp), _payload())
        result = BettingOddsLoader.load_odds(path, min_volume=100_000)
        assert set(result.keys()) == {"m_high"}, \
            f"min_volume=100k should keep only m_high, got: {sorted(result.keys())}"


def test_polymarket_real_file_slice_counts():
    """Regression pin: shipped v2 odds contain exactly 255 / 168 / 110 rows."""
    odds_path = PROJECT_ROOT / "betting_odds_polymarket_v2.json"
    all_matches = BettingOddsLoader.load_odds(str(odds_path))
    fifty_k    = BettingOddsLoader.load_odds(str(odds_path), min_volume=50_000)
    hundred_k  = BettingOddsLoader.load_odds(str(odds_path), min_volume=100_000)
    assert len(all_matches) == 255, f"all-slice expected 255, got {len(all_matches)}"
    assert len(fifty_k)    == 168, f">=$50k slice expected 168, got {len(fifty_k)}"
    assert len(hundred_k)  == 110, f">=$100k slice expected 110, got {len(hundred_k)}"


if __name__ == "__main__":
    test_no_filter_keeps_everything();         print("PASS test_no_filter_keeps_everything")
    test_50k_filter_drops_low_and_missing();   print("PASS test_50k_filter_drops_low_and_missing")
    test_100k_filter_only_high();              print("PASS test_100k_filter_only_high")
    test_polymarket_real_file_slice_counts();  print("PASS test_polymarket_real_file_slice_counts")
    print("\nAll min-volume filter tests passed.")
