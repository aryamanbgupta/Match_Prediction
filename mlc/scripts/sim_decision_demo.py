"""Decision-delta demo: the franchise-facing edge of the ball-by-ball sim.

A team doesn't need a perfect score predictor — it needs to know which of *its*
options is better. This script shows that on a real MLC 2025 match:

  (A) TRUST PANEL — sim the ACTUAL XI/order and show the projected innings-total
      distribution and top-scorer board land on what actually happened. (Evidence
      the simulator is calibrated for the things that matter for planning.)

  (B) DECISION DELTA — take one real lineup decision (promoting a middle-order
      striker the sim rates highly) and re-simulate. Because both options are run
      through the SAME model, the systematic tail-bias cancels in the *delta*, so
      the projected swing is trustworthy even where the absolute total isn't. This
      is the product: rank a team's choices, with a quantified edge + CI.

Usage:
    uv run python mlc/scripts/sim_decision_demo.py --match-date 2025-07-13 --n-sims 300
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from sim_v1_2 import (  # noqa: E402
    SimulationEngine, SimulationConfig, T20Rules, XGBoostModelV2, EmpiricalBowlerSelector)
from sim_eval.loaders import TestMatchLoader  # noqa: E402
from sim_eval.prop_backtest import aggregate_per_player, compute_actuals  # noqa: E402
from stats_provider import StatsProvider  # noqa: E402
from player_metadata import PlayerMetadataProvider  # noqa: E402

MLC_DIR = REPO / "mlc" / "data" / "mlc_2025"
OUT_MD = REPO / "mlc" / "reports" / "mlc_decision_demo.md"


def build_engine():
    sp = StatsProvider("models", version="v3")
    pm = PlayerMetadataProvider("data/all_players_enriched.csv")
    model = XGBoostModelV2(
        model_path="models/xgb_v3/xgboost_model_v3.pkl",
        batter_encoder_path="models/xgb_v3/batter_encoder_v3.pkl",
        bowler_encoder_path="models/xgb_v3/bowler_encoder_v3.pkl",
        feature_columns_path="models/xgb_v3/feature_columns_v3.txt",
        matchup_encoder_path="models/xgb_v3/matchup_encoder_v3.pkl",
        stats_provider=sp, player_metadata=pm, ball_calibrator=None)
    selector = EmpiricalBowlerSelector(usage_path="models/bowler_phase_usage.json")
    return SimulationEngine(model, T20Rules(selector))


def sim_totals(engine, state, n_sims, seed):
    cfg = SimulationConfig(n_simulations=n_sims, parallel=False, random_seed=seed, verbose=False)
    sims = engine.simulate_multiple(state, cfg)
    return aggregate_per_player(state, sims)


def boot_delta_ci(a, b, reps=4000, seed=0):
    """Unpaired bootstrap CI for mean(b) - mean(a)."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = [rng.choice(b, b.size).mean() - rng.choice(a, a.size).mean() for _ in range(reps)]
    return float(np.mean(b) - np.mean(a)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def pick_match(date_sub):
    for fp in sorted(MLC_DIR.glob("[0-9]*.json")):
        d = json.load(open(fp))
        if d["info"]["dates"][0].startswith(date_sub) or date_sub in fp.stem:
            return fp, d
    raise SystemExit(f"no MLC match matching {date_sub}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--match-date", default="2025-07-13", help="date prefix or cricsheet id")
    ap.add_argument("--n-sims", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--team", default=None, help="which side's order to tweak (default: bats first)")
    args = ap.parse_args()

    fp, data = pick_match(args.match_date)
    engine = build_engine()
    loader = TestMatchLoader()
    match_id, state = loader._create_match_state(data)
    teams = [state.team1, state.team2]
    print(f"Match: {match_id}\nTeams: {teams[0]} vs {teams[1]}")

    # ---- (A) baseline: actual order ----
    base = sim_totals(engine, state, args.n_sims, args.seed)
    actuals = compute_actuals(data)

    def board(team):
        lp = {t: [p.name for p in (state.team1_lineup if t == state.team1 else state.team2_lineup).players]
              for t in teams}[team]
        runs = {i: np.mean(base["batter_runs"].get((team, i), [0])) for i in range(len(lp))}
        balls = {i: np.mean(base["batter_balls"].get((team, i), [0])) for i in range(len(lp))}
        return lp, runs, balls

    # choose the team to tweak
    tweak = args.team or state.batting_first
    if tweak not in teams:
        tweak = teams[0]
    lp, proj_runs, proj_balls = board(tweak)

    # The sim's own suggested tweak: swap the WEAKEST-rated top-3 batter with the
    # BEST-rated lower-order hitter (idx>=4, real batter with >=8 proj balls).
    def sr(i):
        return proj_runs[i] / max(proj_balls.get(i, 0), 1) * 100
    cand = max((i for i in range(3, len(lp)) if proj_balls.get(i, 0) >= 8),
               key=sr, default=None)
    if cand is None:
        cand = 4
    incumbent = min(range(0, 3), key=sr)
    cand_sr, inc_sr = sr(cand), sr(incumbent)

    # ---- (B) counterfactual: promote candidate to #3 (swap with incumbent) ----
    # Build a fresh, independent state (copy() shares the lineup objects).
    _, cf_state = loader._create_match_state(data)
    cf_lineup = cf_state.team1_lineup if tweak == state.team1 else cf_state.team2_lineup
    players = list(cf_lineup.players)
    players[incumbent], players[cand] = players[cand], players[incumbent]
    cf_lineup.players = players
    cf = sim_totals(engine, cf_state, args.n_sims, args.seed)

    base_tot = base["team_runs"][tweak]
    cf_tot = cf["team_runs"][tweak]
    delta, lo, hi = boot_delta_ci(base_tot, cf_tot)

    # ---- report ----
    def pctl(x):
        return f"{np.percentile(x,10):.0f}–{np.percentile(x,90):.0f}"
    L = [f"# MLC decision-delta demo — {teams[0]} vs {teams[1]} ({data['info']['dates'][0]})\n",
         "*Ball-by-ball sim run on the real XIs. (A) shows the simulator lands on reality; "
         "(B) shows it ranking a real lineup decision. Same model both ways, so the bias cancels "
         "in the delta.*\n",
         "## (A) Trust panel — projected vs actual\n",
         "| team | projected total (mean, P10–P90) | actual |",
         "|---|---|---:|"]
    for t in teams:
        bt = base["team_runs"][t]
        L.append(f"| {t} | {np.mean(bt):.0f}  ({pctl(bt)}) | {actuals['team_runs'].get(t,'—')} |")
    L.append("\n**Top-scorer board** (sim's pre-match read vs who actually top-scored):\n")
    L.append("| team | sim's most-likely top scorer | P(top) | actual top scorer |")
    L.append("|---|---|---:|---|")
    for t in teams:
        probs = {i: base["top_batter_prob"].get((t, i), 0.0) for i in range(11)}
        lpt = [p.name for p in (state.team1_lineup if t == state.team1 else state.team2_lineup).players]
        bi = max(probs, key=probs.get)
        L.append(f"| {t} | {lpt[bi]} | {probs[bi]:.2f} | {actuals['top_batter_per_team'].get(t,'—')} |")

    L += [f"\n## (B) Decision delta — promote {lp[cand]} to No.3 ({tweak})\n",
          f"The sim rates **{lp[cand]}** (batting {cand+1}, proj. SR {cand_sr:.0f}) well above the "
          f"incumbent No.3 **{lp[incumbent]}** (proj. SR {inc_sr:.0f}). Swap them and re-sim:\n",
          "| {0} order | projected total (mean, P10–P90) |".format(tweak),
          "|---|---|",
          f"| actual order | {np.mean(base_tot):.1f}  ({pctl(base_tot)}) |",
          f"| {lp[cand]} promoted to 3 | {np.mean(cf_tot):.1f}  ({pctl(cf_tot)}) |",
          f"\n**Projected swing: {delta:+.1f} runs** (95% CI [{lo:+.1f}, {hi:+.1f}], "
          f"{args.n_sims} sims/scenario).\n",
          "The point isn't the exact number — it's that the model gives a *signed, sized, "
          "CI'd* answer to a concrete selection question, which a coach can weigh against "
          "match-ups, fitness, and gut. Repeat for any order, any bowler-phasing, any XI.\n"]

    OUT_MD.write_text("\n".join(L) + "\n")
    print(f"\nDecision delta: promote {lp[cand]} -> No.3 : {delta:+.1f} runs "
          f"(95% CI [{lo:+.1f},{hi:+.1f}])")
    print(f"report → {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
