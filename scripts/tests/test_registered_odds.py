import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]


def test_registered_odds_hashes_and_row_counts_match_files():
    registry = json.loads((REPO / "docs/registered_odds.json").read_text())
    assert {row["role"] for row in registry["registered_odds"]} == {
        "odds_iteration_v2", "odds_golden_v2", "odds_forward_sealed"
    }
    for row in registry["registered_odds"]:
        path = REPO / row["path"]
        payload = json.loads(path.read_text())
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
        assert len(payload["matches"]) == row["row_count"]
        assert row["registered"] == "2026-09-10"
