"""Post-hoc blend a v7 sim eval JSON with direct-model predictions
and re-emit a JSON in the same schema, ready to feed into
`reslice_eval_json.py` for sliced bootstrap CIs.

Phase A1 of the match-level direct + sim ensemble plan
(~/.claude/plans/okay-let-s-go-ahead-reflective-sunrise.md).

Per-match blend in logit space:
    logit(P_final) = w * logit(P_sim) + (1 - w) * logit(P_direct)

The output JSON preserves the input schema (per-match keys: match_id,
teams, actual_winner, simulated_prob, market_prob, market_odds, edge,
log_loss, brier_score, realized_pnl, ...) — `simulated_prob` is replaced
by the blended distribution, and downstream scalar metrics are
recomputed.

Usage:
    uv run python scripts/sim_eval/blend_eval_json.py \\
        --sim-json eval_out/phase5_hier/hier_all_20260425_165622.json \\
        --direct-json models/xgb_match_v1/test_predictions.json \\
        --w 0.0 0.2 0.35 0.5 0.65 0.8 1.0 \\
        --out-dir eval_out/blend_a1
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from sim_eval.eval_statistics import (  # noqa: E402
    cluster_id_for_record,
    flat_bet_team,
)
from match_identity import build_compatibility_alias_lookup  # noqa: E402

# Match the existing eval pipeline's edge threshold so realized_pnl is
# computed identically.
BET_EDGE_THRESHOLD = 0.0


def _logit(p: float) -> float:
    eps = 1e-9
    p = max(min(p, 1 - eps), eps)
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    if x > 0:
        ex = math.exp(-x)
        return 1.0 / (1.0 + ex)
    ex = math.exp(x)
    return ex / (1.0 + ex)


def _clip_logit(x: float, lo: float = -10.0, hi: float = 10.0) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def _recompute_realized_pnl(edge: Dict[str, float],
                            market_odds: Dict[str, float],
                            actual_winner: Optional[str]) -> Optional[float]:
    """Mirror match_evaluator._calculate_realized_pnl exactly."""
    if not actual_winner or not edge or not market_odds:
        return None
    best_team = None
    best_edge = 0.0
    for team, e in edge.items():
        if e > best_edge:
            best_edge = e
            best_team = team
    if not best_team or best_edge <= BET_EDGE_THRESHOLD:
        return 0.0
    try:
        odds = float(market_odds[best_team])
    except (KeyError, TypeError, ValueError):
        return 0.0
    if not math.isfinite(odds) or odds < 1.0:
        return 0.0
    if best_team == actual_winner:
        return odds - 1.0
    return -1.0


def _persist_bet_contract(record: dict, *, recompute: bool) -> dict:
    """Persist explicit placement and fallback cluster metadata."""
    out = dict(record)
    if recompute:
        out.pop("bet_placed", None)
        out.pop("bet_team", None)
    bet_team = flat_bet_team(out, BET_EDGE_THRESHOLD)
    out["bet_placed"] = bet_team is not None
    out["bet_team"] = bet_team
    out["competition_cluster_id"] = cluster_id_for_record(out)
    return out


def _blend_match(match: dict, p_direct_team1: Optional[float], w: float) -> dict:
    """Return a copy of `match` with simulated_prob, edge, log_loss,
    brier_score, realized_pnl, expected_value, full_kelly_*, and
    fractional_kelly_pnl recomputed from the logit-blended probabilities.
    If p_direct_team1 is None (match not in direct predictions), keep the
    sim-only outputs (treat as w=1.0) so this match still appears in the
    blended JSON for slice consistency.
    """
    out = dict(match)  # shallow copy
    teams = match.get("teams") or list(match.get("simulated_prob", {}).keys())
    if len(teams) != 2:
        return _persist_bet_contract(out, recompute=False)
    team1, team2 = teams[0], teams[1]
    sim_prob = match.get("simulated_prob", {})
    p_sim_t1 = sim_prob.get(team1)
    if p_sim_t1 is None:
        return _persist_bet_contract(out, recompute=False)

    if p_direct_team1 is None:
        # No direct prediction for this match — pass sim through.
        return _persist_bet_contract(out, recompute=False)

    # Blend in logit space.
    logit_blend = w * _logit(p_sim_t1) + (1 - w) * _logit(p_direct_team1)
    p_blend_t1 = _sigmoid(_clip_logit(logit_blend))
    new_sim = {team1: p_blend_t1, team2: 1.0 - p_blend_t1}
    out["simulated_prob"] = new_sim

    market_prob = match.get("market_prob", {})
    market_odds = match.get("market_odds", {})
    actual_winner = match.get("actual_winner")

    # Recompute edge.
    new_edge = {}
    for team in (team1, team2):
        mp = market_prob.get(team)
        if mp is None:
            new_edge[team] = 0.0
        else:
            new_edge[team] = new_sim[team] - mp
    out["edge"] = new_edge

    # log loss / brier on the actual winner.
    if actual_winner in (team1, team2):
        p_winner = new_sim[actual_winner]
        eps = 1e-9
        p_winner_c = max(min(p_winner, 1 - eps), eps)
        out["log_loss"] = -math.log(p_winner_c)
        # Brier: mean squared error against one-hot
        bs = 0.0
        for t in (team1, team2):
            target = 1.0 if t == actual_winner else 0.0
            bs += (new_sim[t] - target) ** 2
        out["brier_score"] = bs / 2.0
    else:
        out["log_loss"] = None
        out["brier_score"] = None

    # Bet decision + PnL.
    out["realized_pnl"] = _recompute_realized_pnl(
        new_edge, market_odds, actual_winner)
    out = _persist_bet_contract(out, recompute=True)

    # Expected value + Kelly on the chosen-bet side, mirroring
    # match_evaluator._calculate_expected_value / _calculate_kelly_fraction.
    best_team = None
    best_edge = 0.0
    for team, e in new_edge.items():
        if e > best_edge:
            best_edge = e
            best_team = team
    if best_team and best_edge > BET_EDGE_THRESHOLD and best_team in market_odds:
        odds = float(market_odds[best_team])
        win_prob = new_sim[best_team]
        if odds > 1.0:
            out["expected_value"] = win_prob * (odds - 1.0) - (1.0 - win_prob) * 1.0
            b = odds - 1.0
            kelly = (b * win_prob - (1 - win_prob)) / b if 0 < win_prob < 1 else 0.0
            kelly = max(kelly, 0.0)
            out["full_kelly_fraction"] = kelly
            if actual_winner == best_team:
                out["full_kelly_pnl"] = kelly * (odds - 1.0)
                out["fractional_kelly_pnl"] = (kelly * 0.25) * (odds - 1.0)
            elif actual_winner is not None:
                out["full_kelly_pnl"] = -kelly
                out["fractional_kelly_pnl"] = -(kelly * 0.25)
            else:
                out["full_kelly_pnl"] = None
                out["fractional_kelly_pnl"] = None
        else:
            out["expected_value"] = 0.0
            out["full_kelly_fraction"] = 0.0
            out["full_kelly_pnl"] = 0.0
            out["fractional_kelly_pnl"] = 0.0
    else:
        out["expected_value"] = 0.0
        out["full_kelly_fraction"] = 0.0
        out["full_kelly_pnl"] = 0.0
        out["fractional_kelly_pnl"] = 0.0

    return out


def blend(sim_json: dict, direct_preds: dict, w: float) -> dict:
    """Return a new eval JSON with each match blended at weight `w`."""
    direct_rows = []
    for key, value in direct_preds.items():
        row = dict(value)
        row.setdefault("match_id", str(key))
        direct_rows.append(row)
    direct_lookup = build_compatibility_alias_lookup(
        direct_rows,
        context="direct prediction artifact",
    )
    matches = sim_json.get("matches", [])
    out_matches = []
    n_blended = 0
    n_passthrough = 0
    # id(direct entry) -> sim match_id. Two sim rows resolving the same
    # direct row means a doubleheader is sharing one legacy display alias;
    # blending both against a single prediction must fail closed.
    consumed_entries: dict[int, str] = {}
    for m in matches:
        mid = m["match_id"]
        teams = m.get("teams", [])
        team1 = teams[0] if teams else None
        # Join by primary ID first, then by the sim row's own aliases so a
        # cricsheet-keyed sim JSON still joins a frozen synthetic-keyed
        # direct artifact (and vice versa).
        entry = None
        for join_key in (mid, m.get("cricsheet_id"), m.get("display_match_id")):
            if join_key and join_key in direct_lookup:
                entry = direct_lookup[join_key]
                break
        if entry is not None and team1:
            prior = consumed_entries.get(id(entry))
            if prior is not None and prior != str(mid):
                raise RuntimeError(
                    f"direct prediction row already blended into match "
                    f"{prior}; match {mid} resolves the same row via a "
                    "shared legacy display alias"
                )
            consumed_entries[id(entry)] = str(mid)
            # Direct predictions key team1/team2 per their own roster ordering
            # — align to the eval JSON's team1.
            if entry["team1"] == team1:
                p_direct_t1 = entry["p_team1"]
            elif entry["team2"] == team1:
                p_direct_t1 = entry["p_team2"]
            else:
                p_direct_t1 = None  # unaligned — fall through
            out_matches.append(_blend_match(m, p_direct_t1, w))
            if p_direct_t1 is not None:
                n_blended += 1
            else:
                n_passthrough += 1
        else:
            out_matches.append(_blend_match(m, None, w))
            n_passthrough += 1

    # Fresh summary stub — caller is expected to feed the result through
    # reslice_eval_json.py for the real bootstrap CIs.
    return {
        "summary": {
            "blend_w": w,
            "n_matches_total": len(matches),
            "n_matches_blended": n_blended,
            "n_matches_passthrough": n_passthrough,
            "source_summary": sim_json.get("summary", {}),
        },
        "matches": out_matches,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-json", required=True, type=Path,
                    help="Existing v7 sim eval JSON (e.g. eval_out/phase5_hier/hier_all_*.json)")
    ap.add_argument("--direct-json", required=True, type=Path,
                    help="Direct-model predictions (models/xgb_match_v1/test_predictions.json)")
    ap.add_argument("--w", type=float, nargs="+", required=True,
                    help="Blend weights to sweep (e.g. 0.0 0.2 0.35 0.5 0.65 0.8 1.0). "
                         "P_final = w*logit(P_sim) + (1-w)*logit(P_direct).")
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="Output directory; one JSON per w value.")
    args = ap.parse_args()

    with open(args.sim_json) as f:
        sim_json = json.load(f)
    with open(args.direct_json) as f:
        direct_preds = json.load(f)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    src_stem = args.sim_json.stem

    for w in args.w:
        result = blend(sim_json, direct_preds, w)
        # Encode w in filename — replace dot to keep extension parsing easy.
        w_tag = f"w{w:.2f}".replace(".", "p")
        out_path = args.out_dir / f"{src_stem}_{w_tag}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        s = result["summary"]
        print(f"  w={w:.2f}: {s['n_matches_blended']} blended, "
              f"{s['n_matches_passthrough']} passthrough → {out_path}")


if __name__ == "__main__":
    main()
