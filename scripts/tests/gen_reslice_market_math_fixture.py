"""Generate the fixed 30-row reslice/cost/volume acceptance corpus."""
from __future__ import annotations

import json
from pathlib import Path


OUT = Path(__file__).parent / "fixtures"


def main() -> None:
    matches = []
    odds_rows = []
    for index in range(30):
        kind = index % 5
        winner = "A" if index % 2 == 0 else "B"
        odds = {"A": 2.0, "B": 2.0}
        edge = {"A": 0.1, "B": -0.1}
        placed = True
        bet_team = "A"
        pnl = 1.0 if winner == "A" else -1.0
        if kind == 2:
            edge = {"A": 0.0, "B": 0.0}
            placed, bet_team, pnl = False, None, 0.0
        elif kind == 3:
            odds = {"A": 0.95, "B": 4.0}
            placed, bet_team, pnl = False, None, 0.0
        elif kind == 4:
            winner = None
            placed, bet_team, pnl = False, None, None
        match_id = f"synthetic-{index:03d}"
        matches.append({
            "match_id": match_id,
            "teams": ["A", "B"],
            "actual_winner": winner,
            "simulated_prob": {"A": 0.6, "B": 0.4},
            "market_prob": {"A": 0.5, "B": 0.5},
            "market_odds": odds,
            "edge": edge,
            "log_loss": 0.5 + index / 1000.0,
            "brier_score": 0.2,
            "realized_pnl": pnl,
            "bet_placed": placed,
            "bet_team": bet_team,
            "competition_cluster_id": f"event:{index // 2:02d}",
        })
        odds_rows.append({
            "match_id": match_id,
            "polymarket_volume_usd": 100000 if index < 20 else 10000,
            "market_selection": {
                "market_volume_usd": 100000 if index % 2 == 0 else 10000
            },
        })
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "reslice_market_math_synthetic.json").write_text(
        json.dumps({"matches": matches}, indent=2, sort_keys=True) + "\n"
    )
    (OUT / "reslice_market_math_odds.json").write_text(
        json.dumps({"matches": odds_rows}, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
