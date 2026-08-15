"""Regression test for review finding PROP1 (2026-08-14).

The `highest_individual_mae` target is the MATCH-level top score (max over
both innings), but the fair baseline predicted the shrunk mean of the
PER-INNINGS top score — structurally low by E[max of two] − E[one innings],
inflating the baseline's MAE. The corpus now logs a match-level top score
(`venue_match` row index 3) and the family spec reads it.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sim_eval.prop_fair_baselines import build_corpus_logs  # noqa: E402


def _delivery(batter, bowler, runs):
    return {
        "batter": batter,
        "bowler": bowler,
        "non_striker": "NS",
        "runs": {"batter": runs, "extras": 0, "total": runs},
    }


def _match_json(top1, top2):
    """Two innings whose top scorers make (innings-1 top, innings-2 top)."""
    def _innings(team, star, star_runs, bowler):
        deliveries = [_delivery(star, bowler, 6)] * (star_runs // 6)
        deliveries += [_delivery(f"{team} support", bowler, 1)] * 3
        return {"team": team,
                "overs": [{"over": 0, "deliveries": deliveries}]}

    return {
        "info": {
            "gender": "male",
            "match_type": "T20",
            "dates": ["2026-03-01"],
            "teams": ["A", "B"],
            "players": {"A": ["A star", "A support"],
                        "B": ["B star", "B support"]},
            "venue": "Test Oval",
        },
        "innings": [
            _innings("A", "A star", top1, "B bowler"),
            _innings("B", "B star", top2, "A bowler"),
        ],
    }


def test_venue_match_rows_carry_match_level_top_score(tmp_path):
    (tmp_path / "m1.json").write_text(json.dumps(_match_json(30, 72)))
    logs = build_corpus_logs(tmp_path)
    rows = logs["venue_match"]["Test Oval"]
    assert len(rows) == 1
    date, match_sixes, match_max_over, match_top = rows[0]
    assert date == "2026-03-01"
    assert match_top == 72, \
        "venue_match must log the max over BOTH innings' top scores"
    # And the per-innings log still carries each innings' own top score.
    inn_tops = sorted(row[3] for row in logs["venue_inn"]["Test Oval"])
    assert inn_tops == [30, 72]
