#!/usr/bin/env python3
"""Phase 1 prop-calibration check.

Runs the v7 ball-by-ball sim on a sample of the Polymarket test matches,
keeps per-batter / per-bowler / per-team distributions across N sims (instead
of letting ResultAggregator throw them away), and scores the sim's predicted
prop quantities against the cricsheet ground truth in the same JSON files.

Outputs:
  reports/prop_calibration_detail.json  -- per-match raw observations
  reports/prop_calibration_report.md    -- summary tables per prop family
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Allow running as `python scripts/sim_eval/prop_backtest.py`
sys.path.append(str(Path(__file__).resolve().parents[1]))

from sim_v1_2 import (  # noqa: E402
    Outcome,
    SimulationConfig,
    SimulationEngine,
    T20Rules,
    XGBoostModelV2,
)
from sim_eval.loaders import TestMatchLoader  # noqa: E402
from stats_provider import StatsProvider  # noqa: E402
from player_metadata import PlayerMetadataProvider  # noqa: E402


# ---------------------------------------------------------------------------
# Sim aggregation: keep what ResultAggregator throws away.
# ---------------------------------------------------------------------------


def aggregate_per_player(match_state, sim_results):
    """Collect per-batter / per-bowler / per-team distributions across N sims.

    Returns a dict ready for prop-probability calculations.
    """
    team1, team2 = match_state.team1, match_state.team2
    lineup = {
        team1: [p.name for p in match_state.team1_lineup.players],
        team2: [p.name for p in match_state.team2_lineup.players],
    }

    # Per-batter, keyed by (team, player_idx)
    batter_runs: Dict[Tuple[str, int], List[int]] = defaultdict(list)
    batter_balls: Dict[Tuple[str, int], List[int]] = defaultdict(list)
    batter_fours: Dict[Tuple[str, int], List[int]] = defaultdict(list)
    batter_sixes: Dict[Tuple[str, int], List[int]] = defaultdict(list)

    # Per-bowler, keyed by (team, player_idx)
    bowler_wkts: Dict[Tuple[str, int], List[int]] = defaultdict(list)
    bowler_runs: Dict[Tuple[str, int], List[int]] = defaultdict(list)
    bowler_balls: Dict[Tuple[str, int], List[int]] = defaultdict(list)

    # Per-team
    team_runs: Dict[str, List[int]] = defaultdict(list)
    team_fours: Dict[str, List[int]] = defaultdict(list)
    team_sixes: Dict[str, List[int]] = defaultdict(list)
    team_first_over_runs: Dict[str, List[int]] = defaultdict(list)

    # Match-level
    highest_individual: List[int] = []

    # Top-batter / top-bowler indicator across sims, keyed by (team, idx) for
    # batters in their batting innings and (team, idx) for bowlers in their
    # bowling innings.
    top_batter_hits: Dict[Tuple[str, int], int] = defaultdict(int)
    top_bowler_hits: Dict[Tuple[str, int], int] = defaultdict(int)
    top_batter_count_per_team_match: Dict[str, int] = defaultdict(int)
    top_bowler_count_per_team_match: Dict[str, int] = defaultdict(int)

    for r in sim_results:
        sim_top_runs = 0

        # For top-batter: per innings, record argmax(runs) over batters.
        # For top-bowler: per innings, record argmax(wkts) over bowlers (break
        # ties by fewer runs conceded -- standard DK rule).
        for inn in r.innings:
            bt = inn.batting_team
            bw = inn.bowling_team

            t_runs = 0
            t_fours = 0
            t_sixes = 0
            best_bat_runs = -1
            best_bat_idx = None
            for idx, (runs, balls, fours, sixes) in inn.batting_card.items():
                key = (bt, idx)
                batter_runs[key].append(runs)
                batter_balls[key].append(balls)
                batter_fours[key].append(fours)
                batter_sixes[key].append(sixes)
                t_runs += runs
                t_fours += fours
                t_sixes += sixes
                if runs > sim_top_runs:
                    sim_top_runs = runs
                if runs > best_bat_runs:
                    best_bat_runs = runs
                    best_bat_idx = idx
            if best_bat_idx is not None:
                top_batter_hits[(bt, best_bat_idx)] += 1
                top_batter_count_per_team_match[bt] += 1

            team_runs[bt].append(inn.total_runs)
            team_fours[bt].append(t_fours)
            team_sixes[bt].append(t_sixes)

            # First over runs (over index == 0). Sum every ball.runs (includes
            # wides/no-balls -- DK lines treat first-over runs inclusively).
            fo = sum(b.runs for b in inn.balls if b.over == 0)
            team_first_over_runs[bt].append(fo)

            best_bowl_wkts = -1
            best_bowl_runs_conceded = float("inf")
            best_bowl_idx = None
            for idx, (balls, runs, wkts) in inn.bowling_card.items():
                key = (bw, idx)
                bowler_wkts[key].append(wkts)
                bowler_runs[key].append(runs)
                bowler_balls[key].append(balls)
                if (wkts, -runs) > (best_bowl_wkts, -best_bowl_runs_conceded):
                    best_bowl_wkts = wkts
                    best_bowl_runs_conceded = runs
                    best_bowl_idx = idx
            if best_bowl_idx is not None:
                top_bowler_hits[(bw, best_bowl_idx)] += 1
                top_bowler_count_per_team_match[bw] += 1

        highest_individual.append(sim_top_runs)

    n_sims = len(sim_results)

    # Convert hits → probabilities. Denominator is the number of sims in which
    # this team batted/bowled (always n_sims for both teams in T20).
    top_batter_prob: Dict[Tuple[str, int], float] = {}
    for (team, idx), hits in top_batter_hits.items():
        denom = top_batter_count_per_team_match.get(team, n_sims)
        top_batter_prob[(team, idx)] = hits / max(1, denom)
    top_bowler_prob: Dict[Tuple[str, int], float] = {}
    for (team, idx), hits in top_bowler_hits.items():
        denom = top_bowler_count_per_team_match.get(team, n_sims)
        top_bowler_prob[(team, idx)] = hits / max(1, denom)

    return {
        "n_sims": n_sims,
        "lineup": lineup,
        "batter_runs": batter_runs,
        "batter_balls": batter_balls,
        "batter_fours": batter_fours,
        "batter_sixes": batter_sixes,
        "bowler_wkts": bowler_wkts,
        "bowler_runs": bowler_runs,
        "bowler_balls": bowler_balls,
        "team_runs": team_runs,
        "team_fours": team_fours,
        "team_sixes": team_sixes,
        "team_first_over_runs": team_first_over_runs,
        "highest_individual": highest_individual,
        "top_batter_prob": top_batter_prob,
        "top_bowler_prob": top_bowler_prob,
    }


# ---------------------------------------------------------------------------
# Cricsheet ground truth.
# ---------------------------------------------------------------------------


def compute_actuals(data: dict) -> dict:
    """Return per-batter / per-bowler / per-team actuals from a cricsheet JSON."""
    team_runs: Dict[str, int] = {}
    team_fours: Dict[str, int] = {}
    team_sixes: Dict[str, int] = {}
    team_first_over_runs: Dict[str, int] = {}
    batter_runs: Dict[str, int] = {}
    batter_fours: Dict[str, int] = {}
    batter_sixes: Dict[str, int] = {}
    batter_balls: Dict[str, int] = {}
    bowler_wkts: Dict[str, int] = {}
    bowler_runs_conceded: Dict[str, int] = {}
    # team -> top batter name; team -> top bowler name (DK tiebreaker: most
    # wickets, then fewest runs conceded; if no wickets, top batter undefined --
    # we still record one for ranking.)
    team_to_batters: Dict[str, set] = defaultdict(set)
    team_to_bowlers: Dict[str, set] = defaultdict(set)

    teams_seen = []
    for inn in data.get("innings", []):
        bt = inn["team"]
        if bt not in teams_seen:
            teams_seen.append(bt)
        team_runs.setdefault(bt, 0)
        team_fours.setdefault(bt, 0)
        team_sixes.setdefault(bt, 0)
        first_over = 0

        for over in inn.get("overs", []):
            over_idx = over["over"]
            for d in over["deliveries"]:
                runs_total = d["runs"]["total"]
                runs_batter = d["runs"]["batter"]
                team_runs[bt] += runs_total
                if over_idx == 0:
                    first_over += runs_total
                if runs_batter == 4:
                    team_fours[bt] += 1
                elif runs_batter == 6:
                    team_sixes[bt] += 1

                batter = d["batter"]
                team_to_batters[bt].add(batter)
                batter_runs[batter] = batter_runs.get(batter, 0) + runs_batter
                batter_fours.setdefault(batter, 0)
                batter_sixes.setdefault(batter, 0)
                batter_balls.setdefault(batter, 0)
                if runs_batter == 4:
                    batter_fours[batter] += 1
                elif runs_batter == 6:
                    batter_sixes[batter] += 1
                # Balls faced excludes wides (DK convention -- and matches
                # cricsheet semantics).
                if "wides" not in d.get("extras", {}):
                    batter_balls[batter] += 1

                bowler = d["bowler"]
                team_to_bowlers[bt].add(bowler)  # we'll fix: bowlers belong to
                # the OTHER team; we'll re-derive below.
                bowler_runs_conceded.setdefault(bowler, 0)
                bowler_wkts.setdefault(bowler, 0)
                # Conceded runs = batter runs + wides + no-balls (charged to bowler);
                # byes / leg-byes are NOT charged. Use simple: runs_total minus
                # bye/legbye extras.
                extras = d.get("extras", {}) or {}
                non_bowler_extras = (extras.get("byes", 0) or 0) + (extras.get("legbyes", 0) or 0)
                bowler_runs_conceded[bowler] += runs_total - non_bowler_extras
                if "wickets" in d:
                    for w in d["wickets"]:
                        kind = (w.get("kind") or "").lower()
                        if kind != "run out":
                            bowler_wkts[bowler] += 1

        team_first_over_runs[bt] = first_over

    # Fix team_to_bowlers: bowlers actually belong to the opposing team in each
    # innings. team_to_bowlers above accumulated bowlers under the BATTING team
    # name; flip it.
    fixed_team_to_bowlers: Dict[str, set] = defaultdict(set)
    for batting_team, bowlers in team_to_bowlers.items():
        # The opposing team is the other one in teams_seen.
        opposing = next((t for t in teams_seen if t != batting_team), batting_team)
        fixed_team_to_bowlers[opposing] |= bowlers
    team_to_bowlers = fixed_team_to_bowlers

    # Identify per-team top batter / top bowler.
    top_batter_per_team: Dict[str, str] = {}
    for team, batters in team_to_batters.items():
        if not batters:
            continue
        top_batter_per_team[team] = max(
            batters, key=lambda b: (batter_runs.get(b, 0), batter_balls.get(b, 0))
        )
    top_bowler_per_team: Dict[str, str] = {}
    for team, bowlers in team_to_bowlers.items():
        if not bowlers:
            continue
        # DK rule: most wickets, tiebreak fewest runs conceded.
        top_bowler_per_team[team] = max(
            bowlers,
            key=lambda b: (bowler_wkts.get(b, 0), -bowler_runs_conceded.get(b, 0)),
        )

    highest_individual = max(batter_runs.values()) if batter_runs else 0

    return {
        "teams": teams_seen,
        "team_runs": team_runs,
        "team_fours": team_fours,
        "team_sixes": team_sixes,
        "team_first_over_runs": team_first_over_runs,
        "batter_runs": batter_runs,
        "batter_fours": batter_fours,
        "batter_sixes": batter_sixes,
        "batter_balls": batter_balls,
        "bowler_wkts": bowler_wkts,
        "bowler_runs_conceded": bowler_runs_conceded,
        "top_batter_per_team": top_batter_per_team,
        "top_bowler_per_team": top_bowler_per_team,
        "highest_individual": highest_individual,
    }


# ---------------------------------------------------------------------------
# Per-match observation extraction.
# ---------------------------------------------------------------------------


def build_observations(match_id: str, sim_agg: dict, actuals: dict) -> dict:
    """Convert sim distributions + actuals into (predicted, observed) pairs
    grouped by prop family.
    """
    obs = {
        "top_batter": [],   # (predicted_prob, observed_y)
        "top_bowler": [],
        "batter_50plus": [],
        "batter_runs_mae": [],   # (sim_mean, actual_runs, [sim sample for quantile])
        "team_total_fours_mae": [],
        "team_total_sixes_mae": [],
        "team_first_over_mae": [],
        "highest_individual_mae": [],
        "batter_6plus_six": [],  # P(batter sixes >= 1) -> binary y
    }

    lineup = sim_agg["lineup"]

    # ---- Top batter (per team) ----
    for team, names in lineup.items():
        actual_top = actuals["top_batter_per_team"].get(team)
        for idx, pname in enumerate(names):
            prob = sim_agg["top_batter_prob"].get((team, idx), 0.0)
            y = 1 if (actual_top is not None and pname == actual_top) else 0
            obs["top_batter"].append({"team": team, "name": pname, "p": prob, "y": y})

    # ---- Top bowler (per team) ----
    for team, names in lineup.items():
        actual_top = actuals["top_bowler_per_team"].get(team)
        for idx, pname in enumerate(names):
            prob = sim_agg["top_bowler_prob"].get((team, idx), 0.0)
            y = 1 if (actual_top is not None and pname == actual_top) else 0
            obs["top_bowler"].append({"team": team, "name": pname, "p": prob, "y": y})

    # ---- Batter 50+ ----
    for (team, idx), runs_list in sim_agg["batter_runs"].items():
        if not runs_list:
            continue
        names = lineup[team]
        if idx >= len(names):
            continue
        pname = names[idx]
        # Only score batters who actually batted in the real match.
        if pname not in actuals["batter_runs"]:
            continue
        p_50 = sum(1 for r in runs_list if r >= 50) / len(runs_list)
        actual_runs = actuals["batter_runs"][pname]
        actual_balls = actuals["batter_balls"].get(pname, 0)
        # Skip "did not bat" batters (no balls faced).
        if actual_balls == 0:
            continue
        obs["batter_50plus"].append({
            "team": team, "name": pname,
            "p": p_50, "y": 1 if actual_runs >= 50 else 0,
        })

        # Continuous: runs MAE
        sim_mean = float(np.mean(runs_list))
        sim_p10 = float(np.percentile(runs_list, 10))
        sim_p90 = float(np.percentile(runs_list, 90))
        obs["batter_runs_mae"].append({
            "team": team, "name": pname,
            "sim_mean": sim_mean, "sim_p10": sim_p10, "sim_p90": sim_p90,
            "actual": actual_runs,
        })

        # P(>= 1 six)
        sixes_list = sim_agg["batter_sixes"].get((team, idx), [])
        if sixes_list:
            p_six = sum(1 for s in sixes_list if s >= 1) / len(sixes_list)
            actual_sixes = actuals["batter_sixes"].get(pname, 0)
            obs["batter_6plus_six"].append({
                "team": team, "name": pname,
                "p": p_six, "y": 1 if actual_sixes >= 1 else 0,
            })

    # ---- Team total fours / sixes / first over ----
    for team, fours_list in sim_agg["team_fours"].items():
        if not fours_list:
            continue
        actual = actuals["team_fours"].get(team, 0)
        sim_mean = float(np.mean(fours_list))
        sim_p10 = float(np.percentile(fours_list, 10))
        sim_p90 = float(np.percentile(fours_list, 90))
        obs["team_total_fours_mae"].append({
            "team": team,
            "sim_mean": sim_mean, "sim_p10": sim_p10, "sim_p90": sim_p90,
            "actual": actual,
        })
    for team, sixes_list in sim_agg["team_sixes"].items():
        if not sixes_list:
            continue
        actual = actuals["team_sixes"].get(team, 0)
        sim_mean = float(np.mean(sixes_list))
        sim_p10 = float(np.percentile(sixes_list, 10))
        sim_p90 = float(np.percentile(sixes_list, 90))
        obs["team_total_sixes_mae"].append({
            "team": team,
            "sim_mean": sim_mean, "sim_p10": sim_p10, "sim_p90": sim_p90,
            "actual": actual,
        })
    for team, fo_list in sim_agg["team_first_over_runs"].items():
        if not fo_list:
            continue
        actual = actuals["team_first_over_runs"].get(team, 0)
        sim_mean = float(np.mean(fo_list))
        sim_p10 = float(np.percentile(fo_list, 10))
        sim_p90 = float(np.percentile(fo_list, 90))
        obs["team_first_over_mae"].append({
            "team": team,
            "sim_mean": sim_mean, "sim_p10": sim_p10, "sim_p90": sim_p90,
            "actual": actual,
        })

    # ---- Highest individual score ----
    sim_mean = float(np.mean(sim_agg["highest_individual"]))
    sim_p10 = float(np.percentile(sim_agg["highest_individual"], 10))
    sim_p90 = float(np.percentile(sim_agg["highest_individual"], 90))
    obs["highest_individual_mae"].append({
        "sim_mean": sim_mean, "sim_p10": sim_p10, "sim_p90": sim_p90,
        "actual": actuals["highest_individual"],
    })

    return {"match_id": match_id, "obs": obs}


# ---------------------------------------------------------------------------
# Metrics.
# ---------------------------------------------------------------------------


def brier_score(rows):
    if not rows:
        return None
    return float(np.mean([(r["p"] - r["y"]) ** 2 for r in rows]))


def log_loss_binary(rows, eps=1e-6):
    if not rows:
        return None
    ll = []
    for r in rows:
        p = min(max(r["p"], eps), 1 - eps)
        ll.append(-(r["y"] * math.log(p) + (1 - r["y"]) * math.log(1 - p)))
    return float(np.mean(ll))


def base_rate(rows):
    if not rows:
        return None
    return float(np.mean([r["y"] for r in rows]))


def baseline_brier(rows):
    """Brier of always-predict-base-rate model (gives a calibration ceiling)."""
    if not rows:
        return None
    p = base_rate(rows)
    return float(np.mean([(p - r["y"]) ** 2 for r in rows]))


def calibration_table(rows, n_bins=10):
    """Return list of (bin_center, n, mean_p, mean_y) for reliability diag."""
    if not rows:
        return []
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        sub = [r for r in rows if (r["p"] > lo or (i == 0 and r["p"] == 0)) and r["p"] <= hi]
        if not sub:
            continue
        out.append({
            "bin": f"({lo:.1f}, {hi:.1f}]",
            "n": len(sub),
            "mean_p": float(np.mean([r["p"] for r in sub])),
            "mean_y": float(np.mean([r["y"] for r in sub])),
        })
    return out


def mae_continuous(rows):
    if not rows:
        return None, None, None
    diffs = [abs(r["sim_mean"] - r["actual"]) for r in rows]
    coverage = [1 if r["sim_p10"] <= r["actual"] <= r["sim_p90"] else 0 for r in rows]
    bias = [r["sim_mean"] - r["actual"] for r in rows]
    return float(np.mean(diffs)), float(np.mean(coverage)), float(np.mean(bias))


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", default="data/polymarket_test")
    ap.add_argument("--n-matches", type=int, default=30)
    ap.add_argument("--n-sims", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model-path", default="models/xgb_v3/xgboost_model_v3.pkl")
    ap.add_argument("--batter-encoder", default="models/xgb_v3/batter_encoder_v3.pkl")
    ap.add_argument("--bowler-encoder", default="models/xgb_v3/bowler_encoder_v3.pkl")
    ap.add_argument("--feature-columns", default="models/xgb_v3/feature_columns_v3.txt")
    ap.add_argument("--detail-out", default="reports/prop_calibration_detail.json")
    ap.add_argument("--report-out", default="reports/prop_calibration_report.md")
    args = ap.parse_args()

    np.random.seed(args.seed)

    # Load model + providers.
    print("Loading stats provider + player metadata + model ...")
    stats_provider = StatsProvider("models", version="v3")
    player_metadata = PlayerMetadataProvider("data/all_players_enriched.csv")
    model = XGBoostModelV2(
        model_path=args.model_path,
        batter_encoder_path=args.batter_encoder,
        bowler_encoder_path=args.bowler_encoder,
        feature_columns_path=args.feature_columns,
        stats_provider=stats_provider,
        player_metadata=player_metadata,
        ball_calibrator=None,
    )
    engine = SimulationEngine(model, T20Rules())

    # Load test matches.
    loader = TestMatchLoader()
    files = sorted(Path(args.test_dir).glob("*.json"))[: args.n_matches]
    print(f"Running prop backtest on {len(files)} matches × {args.n_sims} sims")

    detail = []
    overall_start = time.time()
    for i, fp in enumerate(files):
        with open(fp) as f:
            data = json.load(f)
        match_id, state = loader._create_match_state(data)
        if state is None:
            print(f"  [{i+1}/{len(files)}] SKIP (could not build state): {fp.name}")
            continue

        t0 = time.time()
        cfg = SimulationConfig(
            n_simulations=args.n_sims,
            parallel=False,
            random_seed=args.seed,
            verbose=False,
        )
        sims = engine.simulate_multiple(state, cfg)
        sim_agg = aggregate_per_player(state, sims)
        actuals = compute_actuals(data)
        obs = build_observations(match_id, sim_agg, actuals)
        detail.append(obs)

        elapsed = time.time() - t0
        avg_team_runs = {
            team: float(np.mean(r)) for team, r in sim_agg["team_runs"].items()
        }
        print(
            f"  [{i+1}/{len(files)}] {match_id[:60]:60s}  "
            f"sim_runs={avg_team_runs}  actual_runs={actuals['team_runs']}  "
            f"({elapsed:.1f}s)"
        )

    print(f"\nDone in {time.time() - overall_start:.1f}s")

    # Persist detail.
    Path(args.detail_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.detail_out, "w") as f:
        json.dump(detail, f, indent=2, default=float)
    print(f"Detail written to {args.detail_out}")

    # Build aggregate report.
    families = [
        "top_batter", "top_bowler", "batter_50plus", "batter_6plus_six",
    ]
    cont_families = [
        "batter_runs_mae", "team_total_fours_mae", "team_total_sixes_mae",
        "team_first_over_mae", "highest_individual_mae",
    ]

    flat = {fam: [] for fam in families + cont_families}
    for d in detail:
        for fam in families + cont_families:
            flat[fam].extend(d["obs"].get(fam, []))

    lines = []
    lines.append("# v7 sim — prop calibration backtest (Phase 1)")
    lines.append("")
    lines.append(
        f"Matches: {len(detail)} | Sims/match: {args.n_sims} | "
        f"Test set: `{args.test_dir}` | Model: `{args.model_path}`"
    )
    lines.append("")

    # Binary props.
    lines.append("## Binary props")
    lines.append("")
    lines.append("| family | n | base rate | sim Brier | base Brier | sim log loss |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for fam in families:
        rows = flat[fam]
        n = len(rows)
        if n == 0:
            lines.append(f"| {fam} | 0 | – | – | – | – |")
            continue
        br = brier_score(rows)
        bbr = baseline_brier(rows)
        ll = log_loss_binary(rows)
        bp = base_rate(rows)
        lines.append(f"| {fam} | {n} | {bp:.3f} | {br:.4f} | {bbr:.4f} | {ll:.4f} |")
    lines.append("")
    lines.append("Notes:")
    lines.append(
        "- Sim Brier < base Brier ⇒ sim has signal beyond the base rate "
        "(prop-level edge over a flat predictor)."
    )
    lines.append(
        "- Base Brier is `var(y)` -- the score from always predicting the "
        "marginal hit rate."
    )
    lines.append("")

    # Reliability diagrams (top batter / top bowler / 50+).
    for fam in ["top_batter", "top_bowler", "batter_50plus", "batter_6plus_six"]:
        rows = flat[fam]
        if not rows:
            continue
        lines.append(f"### Reliability — {fam}")
        lines.append("")
        lines.append("| bin | n | mean p | actual hit rate |")
        lines.append("|---|---:|---:|---:|")
        for c in calibration_table(rows, n_bins=10):
            lines.append(
                f"| {c['bin']} | {c['n']} | {c['mean_p']:.3f} | {c['mean_y']:.3f} |"
            )
        lines.append("")

    # Continuous props.
    lines.append("## Continuous props")
    lines.append("")
    lines.append("| family | n | MAE | mean bias (sim − actual) | P10–P90 coverage |")
    lines.append("|---|---:|---:|---:|---:|")
    for fam in cont_families:
        rows = flat[fam]
        n = len(rows)
        if n == 0:
            lines.append(f"| {fam} | 0 | – | – | – |")
            continue
        mae, cov, bias = mae_continuous(rows)
        lines.append(f"| {fam} | {n} | {mae:.2f} | {bias:+.2f} | {cov:.2%} |")
    lines.append("")
    lines.append(
        "Note: P10–P90 ideal coverage is 80%. Lower ⇒ sim under-disperses "
        "(over-confident); higher ⇒ over-disperses."
    )
    lines.append("")

    Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report_out, "w") as f:
        f.write("\n".join(lines))
    print(f"Report written to {args.report_out}")


if __name__ == "__main__":
    main()
