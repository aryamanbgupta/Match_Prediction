"""Build a sim-shaped eval JSON envelope for the golden set so that
blend_eval_json.py can be reused at w=0 (sim contribution mathematically
dropped).

For each fixture in betting_odds_golden.json we emit a per-match record
matching the schema MatchEvaluationResult.to_dict() produces — but with
`simulated_prob` set to a 50/50 placeholder. At w=0 the blender computes
logit(P_final) = 0*logit(P_sim) + 1*logit(P_direct), so the placeholder
is irrelevant to every downstream metric.

Usage:
    uv run python scripts/synthesize_golden_envelope.py
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


def implied_probs(odds: dict, source: str) -> dict:
    """Convert decimal odds {team: dec_odds} to fair probabilities.
    Polymarket prices are already margin-free (sum to 1.0); bookmaker
    odds carry a margin and need to be normalized. Mirrors
    BettingOddsLoader exactly so the envelope matches what
    run_sim_eval.py would have produced.
    """
    raw = {t: 1.0 / o for t, o in odds.items() if isinstance(o, (int, float))}
    if source == "polymarket":
        # Already margin-free; keep as-is.
        return raw
    total = sum(raw.values())
    return {t: p / total for t, p in raw.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--odds", type=Path,
                    default=Path("data/golden/betting_odds_golden.json"))
    ap.add_argument("--out", type=Path,
                    default=Path("data/golden/golden_sim_envelope.json"))
    args = ap.parse_args()

    with open(args.odds) as f:
        odds_data = json.load(f)

    matches_out = []
    for m in odds_data["matches"]:
        team1, team2 = m["team1"], m["team2"]
        market_odds = {team1: m["odds"]["winner"][team1],
                       team2: m["odds"]["winner"][team2]}
        market_prob = implied_probs(market_odds, source=odds_data["source"])
        # 50/50 placeholder — will be ignored at w=0.
        sim_prob = {team1: 0.5, team2: 0.5}
        # Edge / log_loss / realized_pnl will be recomputed by the blender,
        # but populating them with non-None values keeps the schema sane
        # for any other downstream tool that inspects the envelope.
        actual_winner = m["actual_winner"]
        edge = {t: sim_prob[t] - market_prob[t] for t in (team1, team2)}
        matches_out.append({
            "match_id": m["match_id"],
            "teams": [team1, team2],
            "match_date": m["date"],
            "venue": m["venue"],
            "simulated_prob": sim_prob,
            "market_prob": market_prob,
            "market_odds": market_odds,
            "actual_winner": actual_winner,
            "edge": edge,
            "log_loss": None,
            "brier_score": None,
            "realized_pnl": None,
            "expected_value": 0.0,
            "full_kelly_fraction": 0.0,
            "full_kelly_pnl": None,
            "fractional_kelly_pnl": None,
            "n_simulations": 0,
            "simulation_time": 0.0,
            # Carry the polymarket volume so reslice_eval_json.py can
            # build the >=$50k / >=$100k slices.
            "polymarket_volume_usd": m.get("polymarket_volume_usd"),
            "tournament": m.get("tournament"),
        })

    output = {
        "summary": {
            "envelope_for": "golden direct-only eval (w=0 blend)",
            "n_matches": len(matches_out),
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "source_odds": str(args.odds),
        },
        "matches": matches_out,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  wrote {len(matches_out)}-match envelope -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
