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
from typing import Dict, List, Optional, Tuple

import numpy as np

# Allow running as `python scripts/sim_eval/prop_backtest.py`
sys.path.append(str(Path(__file__).resolve().parents[1]))

from sim_v1_2 import (  # noqa: E402
    EmpiricalBowlerSelector,
    Outcome,
    RandomBowlerSelector,
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
    team_pp_runs: Dict[str, List[int]] = defaultdict(list)
    team_highest_individual: Dict[str, List[int]] = defaultdict(list)
    team_first_wicket_runs: Dict[str, List[int]] = defaultdict(list)

    # Match-level
    highest_individual: List[int] = []
    match_total_sixes: List[int] = []
    match_tie: List[int] = []
    highest_over_runs: List[int] = []

    # Top-batter / top-bowler indicator across sims, keyed by (team, idx) for
    # batters in their batting innings and (team, idx) for bowlers in their
    # bowling innings.
    top_batter_hits: Dict[Tuple[str, int], int] = defaultdict(int)
    top_bowler_hits: Dict[Tuple[str, int], int] = defaultdict(int)
    top_batter_count_per_team_match: Dict[str, int] = defaultdict(int)
    top_bowler_count_per_team_match: Dict[str, int] = defaultdict(int)

    for r in sim_results:
        sim_top_runs = 0
        sim_total_sixes = 0
        sim_max_over_runs = 0
        innings_totals: List[int] = []

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
            sim_total_sixes += t_sixes
            innings_totals.append(inn.total_runs)

            # First over runs (over index == 0). Sum every ball.runs (includes
            # wides/no-balls -- DK lines treat first-over runs inclusively).
            fo = sum(b.runs for b in inn.balls if b.over == 0)
            team_first_over_runs[bt].append(fo)

            # Powerplay total: first 6 overs (over < 6).
            pp = sum(b.runs for b in inn.balls if b.over < 6)
            team_pp_runs[bt].append(pp)

            # Per-team highest individual score in this innings.
            top_in_innings = max(
                (card[0] for card in inn.batting_card.values()), default=0
            )
            team_highest_individual[bt].append(top_in_innings)

            # Runs scored before the first wicket fell. If no wickets,
            # fall back to total innings runs ("partnership lasted the
            # full innings"). team_runs on the wicket ball is cumulative
            # POST-ball but wickets carry 0 runs (sim_v1_2.py:452-453).
            first_wkt_runs = inn.total_runs
            for b in inn.balls:
                if b.outcome == Outcome.WICKET:
                    first_wkt_runs = b.team_runs
                    break
            team_first_wicket_runs[bt].append(first_wkt_runs)

            # Track max single-over runs across the innings.
            over_runs_acc: Dict[int, int] = defaultdict(int)
            for b in inn.balls:
                over_runs_acc[b.over] += b.runs
            if over_runs_acc:
                sim_max_over_runs = max(sim_max_over_runs, max(over_runs_acc.values()))

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
        match_total_sixes.append(sim_total_sixes)
        highest_over_runs.append(sim_max_over_runs)
        match_tie.append(
            1 if len(innings_totals) == 2 and innings_totals[0] == innings_totals[1]
            else 0
        )

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
        "team_pp_runs": team_pp_runs,
        "team_highest_individual": team_highest_individual,
        "team_first_wicket_runs": team_first_wicket_runs,
        "highest_individual": highest_individual,
        "match_total_sixes": match_total_sixes,
        "match_tie": match_tie,
        "highest_over_runs": highest_over_runs,
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
    team_pp_runs: Dict[str, int] = {}
    team_first_wicket_runs: Dict[str, int] = {}
    team_max_over_runs: Dict[str, int] = {}
    batter_runs: Dict[str, int] = {}
    batter_fours: Dict[str, int] = {}
    batter_sixes: Dict[str, int] = {}
    batter_balls: Dict[str, int] = {}
    bowler_wkts: Dict[str, int] = {}
    bowler_runs_conceded: Dict[str, int] = {}
    bowler_legal_balls: Dict[str, int] = {}
    # team -> batter -> runs scored, used to derive per-team top scorer + per-team
    # highest individual.
    team_to_batter_runs: Dict[str, Dict[str, int]] = defaultdict(dict)
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
        pp_runs = 0
        first_wkt_runs: Optional[int] = None
        over_running_totals: Dict[int, int] = defaultdict(int)

        for over in inn.get("overs", []):
            over_idx = over["over"]
            for d in over["deliveries"]:
                runs_total = d["runs"]["total"]
                runs_batter = d["runs"]["batter"]
                team_runs[bt] += runs_total
                over_running_totals[over_idx] += runs_total
                if over_idx == 0:
                    first_over += runs_total
                if over_idx < 6:
                    pp_runs += runs_total
                if runs_batter == 4:
                    team_fours[bt] += 1
                elif runs_batter == 6:
                    team_sixes[bt] += 1

                batter = d["batter"]
                team_to_batters[bt].add(batter)
                batter_runs[batter] = batter_runs.get(batter, 0) + runs_batter
                team_to_batter_runs[bt][batter] = (
                    team_to_batter_runs[bt].get(batter, 0) + runs_batter
                )
                batter_fours.setdefault(batter, 0)
                batter_sixes.setdefault(batter, 0)
                batter_balls.setdefault(batter, 0)
                if runs_batter == 4:
                    batter_fours[batter] += 1
                elif runs_batter == 6:
                    batter_sixes[batter] += 1
                # Balls faced excludes wides (DK convention -- and matches
                # cricsheet semantics).
                extras = d.get("extras", {}) or {}
                if "wides" not in extras:
                    batter_balls[batter] += 1

                bowler = d["bowler"]
                team_to_bowlers[bt].add(bowler)  # we'll fix: bowlers belong to
                # the OTHER team; we'll re-derive below.
                bowler_runs_conceded.setdefault(bowler, 0)
                bowler_wkts.setdefault(bowler, 0)
                bowler_legal_balls.setdefault(bowler, 0)
                # Conceded runs = batter runs + wides + no-balls (charged to bowler);
                # byes / leg-byes are NOT charged. Use simple: runs_total minus
                # bye/legbye extras.
                non_bowler_extras = (extras.get("byes", 0) or 0) + (extras.get("legbyes", 0) or 0)
                bowler_runs_conceded[bowler] += runs_total - non_bowler_extras
                # Legal balls (excludes wides/no-balls) for bowler economy.
                if "wides" not in extras and "noballs" not in extras:
                    bowler_legal_balls[bowler] += 1
                if "wickets" in d:
                    for w in d["wickets"]:
                        kind = (w.get("kind") or "").lower()
                        if kind != "run out":
                            bowler_wkts[bowler] += 1
                            if first_wkt_runs is None:
                                first_wkt_runs = team_runs[bt]
                        elif first_wkt_runs is None:
                            # Even run-out counts toward "first wicket"
                            # for runs-before-first-wicket prop.
                            first_wkt_runs = team_runs[bt]

        team_first_over_runs[bt] = first_over
        team_pp_runs[bt] = pp_runs
        # No wicket fell ⇒ partnership lasted the full innings; record
        # full innings total. Same convention as the sim aggregator.
        team_first_wicket_runs[bt] = (
            first_wkt_runs if first_wkt_runs is not None else team_runs[bt]
        )
        team_max_over_runs[bt] = (
            max(over_running_totals.values()) if over_running_totals else 0
        )

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

    # Per-team highest individual = max batter runs scored under that team.
    team_highest_individual: Dict[str, int] = {
        team: (max(runs.values()) if runs else 0)
        for team, runs in team_to_batter_runs.items()
    }

    match_total_sixes = sum(team_sixes.values())
    match_total_fours = sum(team_fours.values())
    highest_over_runs_match = (
        max(team_max_over_runs.values()) if team_max_over_runs else 0
    )
    is_tie = (
        1 if len(teams_seen) == 2
        and team_runs.get(teams_seen[0]) == team_runs.get(teams_seen[1])
        else 0
    )

    return {
        "teams": teams_seen,
        "team_runs": team_runs,
        "team_fours": team_fours,
        "team_sixes": team_sixes,
        "team_first_over_runs": team_first_over_runs,
        "team_pp_runs": team_pp_runs,
        "team_first_wicket_runs": team_first_wicket_runs,
        "team_highest_individual": team_highest_individual,
        "team_max_over_runs": team_max_over_runs,
        "batter_runs": batter_runs,
        "batter_fours": batter_fours,
        "batter_sixes": batter_sixes,
        "batter_balls": batter_balls,
        "bowler_wkts": bowler_wkts,
        "bowler_runs_conceded": bowler_runs_conceded,
        "bowler_legal_balls": bowler_legal_balls,
        "top_batter_per_team": top_batter_per_team,
        "top_bowler_per_team": top_bowler_per_team,
        "highest_individual": highest_individual,
        "match_total_sixes": match_total_sixes,
        "match_total_fours": match_total_fours,
        "highest_over_runs": highest_over_runs_match,
        "is_tie": is_tie,
    }


# ---------------------------------------------------------------------------
# Per-match observation extraction.
# ---------------------------------------------------------------------------


def build_observations(match_id: str, sim_agg: dict, actuals: dict) -> dict:
    """Convert sim distributions + actuals into (predicted, observed) pairs
    grouped by prop family.
    """
    obs = {
        # Existing families
        "top_batter": [],   # (predicted_prob, observed_y)
        "top_bowler": [],
        "batter_50plus": [],
        "batter_runs_mae": [],   # (sim_mean, actual_runs, [sim sample for quantile])
        "team_total_fours_mae": [],
        "team_total_sixes_mae": [],
        "team_first_over_mae": [],
        "highest_individual_mae": [],
        "batter_6plus_six": [],  # P(batter sixes >= 1) -> binary y
        # User-named additions
        "innings_runs_ou_160_5": [],   # per-team innings runs > 160.5
        "innings_runs_ou_170_5": [],
        "innings_runs_ou_180_5": [],
        "batter_fours_1plus": [],
        "batter_fours_2plus": [],
        "batter_fours_3plus": [],
        "batter_fours_mae": [],
        "bowler_wkts_1plus": [],
        "bowler_wkts_2plus": [],
        "bowler_wkts_3plus": [],
        "team_highest_individual_ou_29_5": [],
        "team_highest_individual_ou_34_5": [],
        "team_highest_individual_ou_39_5": [],
        # Creative additions
        "pp_total_ou_45_5": [],   # team powerplay (0-5.6 overs) runs > 45.5
        "pp_total_ou_50_5": [],
        "pp_total_ou_55_5": [],
        "match_total_sixes_ou_15_5": [],
        "match_total_sixes_ou_20_5": [],
        "first_wicket_runs_ou_30_5": [],
        "bowler_economy_ou_8_5": [],
        "bowler_economy_ou_10_5": [],
        "p_tie": [],
        "highest_over_runs_ou_18_5": [],
        "highest_over_runs_ou_24_5": [],
    }

    def _ou(sim_values, actual, line):
        if not sim_values:
            return None
        p = sum(1 for v in sim_values if v > line) / len(sim_values)
        y = 1 if actual > line else 0
        return {"p": float(p), "y": y, "line": line, "sim_mean": float(np.mean(sim_values)),
                "actual": actual}

    def _at_least(sim_values, actual, threshold):
        if not sim_values:
            return None
        p = sum(1 for v in sim_values if v >= threshold) / len(sim_values)
        y = 1 if actual >= threshold else 0
        return {"p": float(p), "y": y, "threshold": threshold,
                "actual": actual}

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

    # ---- Innings runs O/U (per-team innings, multiple lines) ----
    for team, runs_list in sim_agg["team_runs"].items():
        actual = actuals["team_runs"].get(team, 0)
        for line, fam in (
            (160.5, "innings_runs_ou_160_5"),
            (170.5, "innings_runs_ou_170_5"),
            (180.5, "innings_runs_ou_180_5"),
        ):
            row = _ou(runs_list, actual, line)
            if row is not None:
                row["team"] = team
                obs[fam].append(row)

    # ---- Batter fours: P(>=1), P(>=2), P(>=3) + MAE ----
    for (team, idx), fours_list in sim_agg["batter_fours"].items():
        if not fours_list:
            continue
        names = lineup[team]
        if idx >= len(names):
            continue
        pname = names[idx]
        if pname not in actuals["batter_runs"]:
            continue
        if actuals["batter_balls"].get(pname, 0) == 0:
            continue
        actual_fours = actuals["batter_fours"].get(pname, 0)
        for thr, fam in ((1, "batter_fours_1plus"),
                         (2, "batter_fours_2plus"),
                         (3, "batter_fours_3plus")):
            row = _at_least(fours_list, actual_fours, thr)
            if row is not None:
                row["team"] = team; row["name"] = pname
                obs[fam].append(row)
        # Continuous MAE
        sim_mean = float(np.mean(fours_list))
        obs["batter_fours_mae"].append({
            "team": team, "name": pname,
            "sim_mean": sim_mean,
            "sim_p10": float(np.percentile(fours_list, 10)),
            "sim_p90": float(np.percentile(fours_list, 90)),
            "actual": actual_fours,
        })

    # ---- Bowler wickets: P(>=1), P(>=2), P(>=3) ----
    # Need bowler-name → team mapping from actuals.
    bowler_team: Dict[str, str] = {}
    for team in actuals.get("teams", []):
        # team_to_bowlers was flipped in compute_actuals; we don't have it
        # in actuals dict, so reverse via bowler_wkts which is only populated
        # for bowlers (i.e. the opposing-team players).
        pass
    # Easier: enumerate sim bowlers; we have (team, idx) → name via lineup.
    for (team, idx), wkts_list in sim_agg["bowler_wkts"].items():
        if not wkts_list:
            continue
        names = lineup[team]
        if idx >= len(names):
            continue
        pname = names[idx]
        # The "team" here is the bowling team (i.e. the team whose lineup
        # this player is in). Only score bowlers who actually bowled.
        if pname not in actuals["bowler_wkts"]:
            continue
        if actuals.get("bowler_legal_balls", {}).get(pname, 0) == 0:
            continue
        actual_wkts = actuals["bowler_wkts"].get(pname, 0)
        for thr, fam in ((1, "bowler_wkts_1plus"),
                         (2, "bowler_wkts_2plus"),
                         (3, "bowler_wkts_3plus")):
            row = _at_least(wkts_list, actual_wkts, thr)
            if row is not None:
                row["team"] = team; row["name"] = pname
                obs[fam].append(row)

    # ---- Team highest individual O/U ----
    for team, hi_list in sim_agg["team_highest_individual"].items():
        actual = actuals["team_highest_individual"].get(team, 0)
        for line, fam in (
            (29.5, "team_highest_individual_ou_29_5"),
            (34.5, "team_highest_individual_ou_34_5"),
            (39.5, "team_highest_individual_ou_39_5"),
        ):
            row = _ou(hi_list, actual, line)
            if row is not None:
                row["team"] = team
                obs[fam].append(row)

    # ---- Powerplay total O/U ----
    for team, pp_list in sim_agg["team_pp_runs"].items():
        actual = actuals["team_pp_runs"].get(team, 0)
        for line, fam in (
            (45.5, "pp_total_ou_45_5"),
            (50.5, "pp_total_ou_50_5"),
            (55.5, "pp_total_ou_55_5"),
        ):
            row = _ou(pp_list, actual, line)
            if row is not None:
                row["team"] = team
                obs[fam].append(row)

    # ---- Match total sixes O/U ----
    sixes_list = sim_agg["match_total_sixes"]
    actual_total_sixes = actuals.get("match_total_sixes", 0)
    for line, fam in (
        (15.5, "match_total_sixes_ou_15_5"),
        (20.5, "match_total_sixes_ou_20_5"),
    ):
        row = _ou(sixes_list, actual_total_sixes, line)
        if row is not None:
            obs[fam].append(row)

    # ---- First-wicket runs O/U (per innings) ----
    for team, fw_list in sim_agg["team_first_wicket_runs"].items():
        actual = actuals["team_first_wicket_runs"].get(team, 0)
        row = _ou(fw_list, actual, 30.5)
        if row is not None:
            row["team"] = team
            obs["first_wicket_runs_ou_30_5"].append(row)

    # ---- Bowler economy O/U (per bowler who actually bowled) ----
    for (team, idx), runs_list in sim_agg["bowler_runs"].items():
        balls_list = sim_agg["bowler_balls"].get((team, idx), [])
        if not runs_list or not balls_list:
            continue
        # economy per sim = runs * 6 / balls (skip sims where the bowler
        # didn't bowl at all in that sim → balls==0).
        eco_list = [
            (r * 6.0 / b) for r, b in zip(runs_list, balls_list) if b > 0
        ]
        if not eco_list:
            continue
        names = lineup[team]
        if idx >= len(names):
            continue
        pname = names[idx]
        if pname not in actuals["bowler_wkts"]:
            continue
        actual_balls = actuals.get("bowler_legal_balls", {}).get(pname, 0)
        if actual_balls == 0:
            continue
        actual_runs = actuals["bowler_runs_conceded"].get(pname, 0)
        actual_eco = actual_runs * 6.0 / actual_balls
        for line, fam in ((8.5, "bowler_economy_ou_8_5"),
                          (10.5, "bowler_economy_ou_10_5")):
            row = _ou(eco_list, actual_eco, line)
            if row is not None:
                row["team"] = team; row["name"] = pname
                obs[fam].append(row)

    # ---- P(tie) ----
    tie_list = sim_agg["match_tie"]
    if tie_list:
        p_tie = sum(tie_list) / len(tie_list)
        obs["p_tie"].append({
            "p": float(p_tie),
            "y": int(actuals.get("is_tie", 0)),
        })

    # ---- Highest single-over runs O/U (across both innings) ----
    hor_list = sim_agg["highest_over_runs"]
    actual_hor = actuals.get("highest_over_runs", 0)
    for line, fam in ((18.5, "highest_over_runs_ou_18_5"),
                      (24.5, "highest_over_runs_ou_24_5")):
        row = _ou(hor_list, actual_hor, line)
        if row is not None:
            obs[fam].append(row)

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


def bootstrap_ci(rows, metric_fn, n_reps=1000, alpha=0.05, seed=0):
    """Bootstrap CI for `metric_fn(rows)` by resampling rows with replacement.

    For paired CI across match boundaries, resample at the match level
    upstream and pass the flattened rows here.
    """
    if not rows:
        return None, None
    rng = np.random.default_rng(seed)
    samples = []
    n = len(rows)
    for _ in range(n_reps):
        idxs = rng.integers(0, n, size=n)
        sample = [rows[i] for i in idxs]
        v = metric_fn(sample)
        if v is not None:
            samples.append(v)
    if not samples:
        return None, None
    lo = float(np.percentile(samples, 100 * alpha / 2))
    hi = float(np.percentile(samples, 100 * (1 - alpha / 2)))
    return lo, hi


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", default="data/polymarket_test")
    ap.add_argument("--n-matches", default="30",
                    help="Number of matches to score, or 'all' for full test set.")
    ap.add_argument("--n-sims", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model-path", default="models/xgb_v3/xgboost_model_v3.pkl")
    ap.add_argument("--batter-encoder", default="models/xgb_v3/batter_encoder_v3.pkl")
    ap.add_argument("--bowler-encoder", default="models/xgb_v3/bowler_encoder_v3.pkl")
    ap.add_argument("--feature-columns", default="models/xgb_v3/feature_columns_v3.txt")
    ap.add_argument(
        "--stats-version",
        default="v3",
        help="StatsProvider artifact version (for example v3 or i5).",
    )
    ap.add_argument("--bowler-selector", choices=["empirical", "random"],
                    default="empirical",
                    help="Bowler selection strategy. Default = empirical (phase-aware).")
    ap.add_argument("--bowler-usage-path",
                    default="models/bowler_phase_usage.json",
                    help="Usage prior JSON for EmpiricalBowlerSelector.")
    ap.add_argument("--detail-out", default="reports/prop_calibration_detail.json")
    ap.add_argument("--report-out", default="reports/prop_calibration_report.md")
    ap.add_argument("--ball-calibrator", choices=["none", "vector"],
                    default="none",
                    help="'vector' = val-fit VectorScalingCalibrator that "
                         "undoes the balanced-class-weight tilt in the "
                         "booster's raw probabilities (E5, 2026-06-09).")
    ap.add_argument("--ball-calibrator-path",
                    default="models/xgb_v3/vector_scaling_calibrator_v1.pkl")
    args = ap.parse_args()

    np.random.seed(args.seed)

    ball_calibrator = None
    if args.ball_calibrator == "vector":
        import joblib
        ball_calibrator = joblib.load(args.ball_calibrator_path)
        print(f"Ball calibrator: vector scaling ({args.ball_calibrator_path})")

    # Load model + providers.
    print("Loading stats provider + player metadata + model ...")
    stats_provider = StatsProvider("models", version=args.stats_version)
    player_metadata = PlayerMetadataProvider("data/all_players_enriched.csv")
    model = XGBoostModelV2(
        model_path=args.model_path,
        batter_encoder_path=args.batter_encoder,
        bowler_encoder_path=args.bowler_encoder,
        feature_columns_path=args.feature_columns,
        stats_provider=stats_provider,
        player_metadata=player_metadata,
        ball_calibrator=ball_calibrator,
    )
    if args.bowler_selector == "empirical":
        selector = EmpiricalBowlerSelector(usage_path=args.bowler_usage_path)
    else:
        selector = RandomBowlerSelector()
    print(f"Bowler selector: {args.bowler_selector}")
    engine = SimulationEngine(model, T20Rules(selector))

    # Load test matches.
    loader = TestMatchLoader()
    all_files = sorted(Path(args.test_dir).glob("*.json"))
    if args.n_matches == "all":
        files = all_files
    else:
        files = all_files[: int(args.n_matches)]
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
        # Original Phase-1 families
        "top_batter", "top_bowler", "batter_50plus", "batter_6plus_six",
        # Innings runs O/U
        "innings_runs_ou_160_5", "innings_runs_ou_170_5", "innings_runs_ou_180_5",
        # Batter fours thresholds
        "batter_fours_1plus", "batter_fours_2plus", "batter_fours_3plus",
        # Bowler wickets thresholds
        "bowler_wkts_1plus", "bowler_wkts_2plus", "bowler_wkts_3plus",
        # Per-team highest individual O/U
        "team_highest_individual_ou_29_5", "team_highest_individual_ou_34_5",
        "team_highest_individual_ou_39_5",
        # Powerplay totals
        "pp_total_ou_45_5", "pp_total_ou_50_5", "pp_total_ou_55_5",
        # Match-level
        "match_total_sixes_ou_15_5", "match_total_sixes_ou_20_5",
        "first_wicket_runs_ou_30_5",
        "bowler_economy_ou_8_5", "bowler_economy_ou_10_5",
        "p_tie",
        "highest_over_runs_ou_18_5", "highest_over_runs_ou_24_5",
    ]
    cont_families = [
        "batter_runs_mae", "team_total_fours_mae", "team_total_sixes_mae",
        "team_first_over_mae", "highest_individual_mae",
        "batter_fours_mae",
    ]

    flat = {fam: [] for fam in families + cont_families}
    for d in detail:
        for fam in families + cont_families:
            flat[fam].extend(d["obs"].get(fam, []))

    lines = []
    model_label = Path(args.model_path).stem
    lines.append(f"# {model_label} — prop calibration backtest")
    lines.append("")
    lines.append(
        f"Matches: {len(detail)} | Sims/match: {args.n_sims} | "
        f"Test set: `{args.test_dir}` | Model: `{args.model_path}`"
    )
    lines.append("")

    # Binary props.
    lines.append("## Binary props")
    lines.append("")
    lines.append(
        "| family | n | base rate | sim Brier [95% CI] | base Brier | "
        "sim log loss | skill |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for fam in families:
        rows = flat[fam]
        n = len(rows)
        if n == 0:
            lines.append(f"| {fam} | 0 | – | – | – | – | – |")
            continue
        br = brier_score(rows)
        bbr = baseline_brier(rows)
        ll = log_loss_binary(rows)
        bp = base_rate(rows)
        ci_lo, ci_hi = bootstrap_ci(rows, brier_score, n_reps=1000, seed=args.seed)
        # Brier skill score: 1 = perfect, 0 = no skill vs base, <0 = worse than base.
        bss = 1 - br / bbr if bbr and bbr > 0 else None
        bss_str = f"{bss:+.3f}" if bss is not None else "–"
        ci_str = (
            f"{br:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]"
            if ci_lo is not None else f"{br:.4f}"
        )
        lines.append(
            f"| {fam} | {n} | {bp:.3f} | {ci_str} | {bbr:.4f} | {ll:.4f} | {bss_str} |"
        )
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
    lines.append(
        "- Skill = `1 − Brier/base_Brier`. Positive ⇒ sim beats base rate."
    )
    lines.append(
        "- Bootstrap CIs: 1000 resamples at the row level (n.b. not paired "
        "by match — match-level pairing would tighten CIs further)."
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
    lines.append(
        "| family | n | MAE [95% CI] | mean bias (sim − actual) | "
        "P10–P90 coverage |"
    )
    lines.append("|---|---:|---:|---:|---:|")
    for fam in cont_families:
        rows = flat[fam]
        n = len(rows)
        if n == 0:
            lines.append(f"| {fam} | 0 | – | – | – |")
            continue
        mae, cov, bias = mae_continuous(rows)
        ci_lo, ci_hi = bootstrap_ci(
            rows, lambda rs: mae_continuous(rs)[0],
            n_reps=1000, seed=args.seed,
        )
        ci_str = (
            f"{mae:.2f} [{ci_lo:.2f}, {ci_hi:.2f}]"
            if ci_lo is not None else f"{mae:.2f}"
        )
        lines.append(f"| {fam} | {n} | {ci_str} | {bias:+.2f} | {cov:.2%} |")
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
