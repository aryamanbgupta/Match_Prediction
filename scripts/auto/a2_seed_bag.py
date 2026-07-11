"""A2 — Seed bagging: average p_team1 across the 5 A1 seed retrains.

Bagging = variance reduction with no new information. We take the arithmetic
mean of the predicted p_team1 across the five A1 seed models (same M7 config,
different seeds) and re-emit one prediction set in the same schema that
`blend_eval_json.py` consumes.

The averaged file feeds recipe A steps 2-3 exactly like any single-seed
test_predictions.json.
"""
from __future__ import annotations

import json
from pathlib import Path

SEEDS = [29, 7, 13, 42, 101]
IN_FILES = [Path(f"models/auto/a1_seed{s}/test_predictions.json") for s in SEEDS]
OUT_FILE = Path("models/auto/a2/test_predictions.json")


def main() -> None:
    preds = [json.load(open(f)) for f in IN_FILES]
    keys = set(preds[0].keys())
    for p in preds[1:]:
        assert set(p.keys()) == keys, "seed prediction files disagree on match keys"

    out = {}
    for k in preds[0].keys():
        base = preds[0][k]
        p1 = sum(p[k]["p_team1"] for p in preds) / len(preds)
        # Sanity: all seeds must agree on match-level metadata.
        for p in preds[1:]:
            assert p[k]["team1"] == base["team1"]
            assert p[k]["team2"] == base["team2"]
            assert p[k]["team1_wins"] == base["team1_wins"]
        out[k] = {
            "team1": base["team1"],
            "team2": base["team2"],
            "p_team1": p1,
            "p_team2": 1.0 - p1,
            "team1_wins": base["team1_wins"],
            "match_date": base["match_date"],
        }

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT_FILE, "w"), indent=2)
    print(f"wrote {OUT_FILE} with {len(out)} matches (mean of {len(preds)} seeds)")


if __name__ == "__main__":
    main()
