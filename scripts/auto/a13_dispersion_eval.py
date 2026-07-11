#!/usr/bin/env python3
"""A13 — sim dispersion (variance) calibration on sampled score totals.

A8 showed a *marginal-rate* correction (vector scaling) cannot move the
tail-overshoot O/U props. This idea targets the orthogonal defect the vec
baseline report flagged: the sim **under-disperses** score totals (P10-P90
coverage 64-74% vs the ideal 80%). Under-dispersion is a *variance* defect, so
we widen the predictive spread of each family's per-sim total distribution to
nominal coverage and check whether the tail O/U Brier improves.

Mechanism (the idea's "multiplicative fan-out on centered per-sim totals"):
for a family's per-sim total list ``v`` with mean ``m``, replace each value with
``m + k*(v - m)``. This inflates the spread by factor k while leaving the mean
(hence every MAE point forecast) EXACTLY unchanged. Percentiles scale linearly
about m, so P10-P90 coverage is exact from (m, p10, p90, actual). k is fit
per-family on the ball-model VALIDATION window (2024-12-31..2025-06-29 T20Is,
held out from training AND disjoint from the polymarket test set) to hit 80%
coverage, then applied unchanged to the test set.

No eval-framework edit: this harness IMPORTS prop_backtest's own
``aggregate_per_player`` / ``build_observations`` / ``compute_actuals`` and only
inserts the fan-out on ``sim_agg`` between aggregate and observe. The k=1 path
reproduces prop_backtest exactly (sanity-checked vs detail_vec_n261.json).

Subcommands:
  capture-val  run sim on val-window T20Is, dump per-family (m,p10,p90,actual)
  fit          fit per-family k to 80% coverage from the capture-val dump
  score        run sim on test, emit baseline (k=1) + dispersion (k) detail JSONs
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

# Import the eval framework's own functions UNCHANGED (no edit).
from sim_eval.prop_backtest import (  # noqa: E402
    aggregate_per_player,
    build_observations,
    compute_actuals,
)
from sim_eval.loaders import TestMatchLoader  # noqa: E402
from sim_v1_2 import (  # noqa: E402
    EmpiricalBowlerSelector,
    SimulationConfig,
    SimulationEngine,
    T20Rules,
    XGBoostModelV2,
)
from stats_provider import StatsProvider  # noqa: E402
from player_metadata import PlayerMetadataProvider  # noqa: E402

# ---------------------------------------------------------------------------
# Family -> extraction from sim_agg / actuals, replicating build_observations'
# per-row construction & filters exactly so coverage matches the eval report.
# Each yields (list_of_per_sim_values, actual_scalar).
# ---------------------------------------------------------------------------

# Gate O/U families (Metric 2) + coverage families (Metric 1 + guard).
COVERAGE_FAMILIES = [
    "batter_runs", "team_total_fours", "team_total_sixes",
    "team_first_over", "highest_individual", "batter_fours",
]
OU_GATE_FAMILIES = ["pp_total", "first_wicket", "highest_over"]
ALL_FAMILIES = COVERAGE_FAMILIES + OU_GATE_FAMILIES

# sim_agg key each family scales (the per-sim total list to fan out).
FAMILY_TO_AGG_KEY = {
    "batter_runs": "batter_runs",
    "team_total_fours": "team_fours",
    "team_total_sixes": "team_sixes",
    "team_first_over": "team_first_over_runs",
    "highest_individual": "highest_individual",
    "batter_fours": "batter_fours",
    "pp_total": "team_pp_runs",
    "first_wicket": "team_first_wicket_runs",
    "highest_over": "highest_over_runs",
}


def extract_family_lists(sim_agg, actuals):
    """Yield {family: [(vals, actual), ...]} replicating build_observations."""
    lineup = sim_agg["lineup"]
    out = {f: [] for f in ALL_FAMILIES}

    # batter_runs / batter_fours (same filter as build_observations)
    for (team, idx), runs_list in sim_agg["batter_runs"].items():
        if not runs_list:
            continue
        names = lineup[team]
        if idx >= len(names):
            continue
        pname = names[idx]
        if pname not in actuals["batter_runs"]:
            continue
        if actuals["batter_balls"].get(pname, 0) == 0:
            continue
        out["batter_runs"].append((list(runs_list), actuals["batter_runs"][pname]))
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
        out["batter_fours"].append((list(fours_list), actuals["batter_fours"].get(pname, 0)))

    # team totals
    for team, l in sim_agg["team_fours"].items():
        if l:
            out["team_total_fours"].append((list(l), actuals["team_fours"].get(team, 0)))
    for team, l in sim_agg["team_sixes"].items():
        if l:
            out["team_total_sixes"].append((list(l), actuals["team_sixes"].get(team, 0)))
    for team, l in sim_agg["team_first_over_runs"].items():
        if l:
            out["team_first_over"].append((list(l), actuals["team_first_over_runs"].get(team, 0)))

    # highest individual (single per match)
    hi = sim_agg["highest_individual"]
    if hi:
        out["highest_individual"].append((list(hi), actuals["highest_individual"]))

    # O/U gate families
    for team, l in sim_agg["team_pp_runs"].items():
        if l:
            out["pp_total"].append((list(l), actuals["team_pp_runs"].get(team, 0)))
    for team, l in sim_agg["team_first_wicket_runs"].items():
        if l:
            out["first_wicket"].append((list(l), actuals["team_first_wicket_runs"].get(team, 0)))
    hor = sim_agg["highest_over_runs"]
    if hor:
        out["highest_over"].append((list(hor), actuals.get("highest_over_runs", 0)))
    return out


def fan_out(values, k):
    """Multiplicative spread inflation about the list mean; mean-preserving."""
    m = float(np.mean(values))
    return [m + k * (v - m) for v in values]


# ---------------------------------------------------------------------------
# Sim setup (replicates prop_backtest.main lines 882-908 exactly).
# ---------------------------------------------------------------------------

def build_engine(seed):
    np.random.seed(seed)
    import joblib
    ball_calibrator = joblib.load(REPO / "models/xgb_v3/vector_scaling_calibrator_v1.pkl")
    print(f"Ball calibrator: vector scaling (v1)")
    stats_provider = StatsProvider("models", version="v3")
    player_metadata = PlayerMetadataProvider(str(REPO / "data/all_players_enriched.csv"))
    model = XGBoostModelV2(
        model_path=str(REPO / "models/xgb_v3/xgboost_model_v3.pkl"),
        batter_encoder_path=str(REPO / "models/xgb_v3/batter_encoder_v3.pkl"),
        bowler_encoder_path=str(REPO / "models/xgb_v3/bowler_encoder_v3.pkl"),
        feature_columns_path=str(REPO / "models/xgb_v3/feature_columns_v3.txt"),
        stats_provider=stats_provider,
        player_metadata=player_metadata,
        ball_calibrator=ball_calibrator,
    )
    selector = EmpiricalBowlerSelector(usage_path=str(REPO / "models/bowler_phase_usage.json"))
    engine = SimulationEngine(model, T20Rules(selector))
    return engine


def iter_sims(engine, files, n_sims, seed):
    """Yield (match_id, sim_agg, actuals, data) per buildable match."""
    import time
    loader = TestMatchLoader()
    for i, fp in enumerate(files):
        with open(fp) as f:
            data = json.load(f)
        match_id, state = loader._create_match_state(data)
        if state is None:
            continue
        t0 = time.time()
        cfg = SimulationConfig(n_simulations=n_sims, parallel=False,
                               random_seed=seed, verbose=False)
        sims = engine.simulate_multiple(state, cfg)
        sim_agg = aggregate_per_player(state, sims)
        actuals = compute_actuals(data)
        yield i, match_id, sim_agg, actuals
        print(f"  [{i+1}/{len(files)}] {match_id[:55]:55s} ({time.time()-t0:.1f}s)", flush=True)


# ---------------------------------------------------------------------------
# Subcommands.
# ---------------------------------------------------------------------------

def cmd_capture_val(args):
    lo = date(*map(int, args.date_lo.split("-")))
    hi = date(*map(int, args.date_hi.split("-")))
    all_files = sorted(Path(args.src_dir).glob("*.json"))
    val = []
    for fp in all_files:
        try:
            d = json.load(open(fp))
            dt = d.get("info", {}).get("dates", [None])[0]
            if not dt:
                continue
            y, m, dd = map(int, dt.split("-"))
            if lo <= date(y, m, dd) <= hi:
                val.append(fp)
        except Exception:
            continue
    # deterministic even-spaced subsample
    if args.n_matches != "all" and len(val) > int(args.n_matches):
        step = len(val) / int(args.n_matches)
        val = [val[int(j * step)] for j in range(int(args.n_matches))]
    print(f"Val-window buildable candidates: {len(val)} (seed {args.seed})")

    engine = build_engine(args.seed)
    agg = {f: [] for f in ALL_FAMILIES}
    n_ok = 0
    for _, mid, sim_agg, actuals in iter_sims(engine, val, args.n_sims, args.seed):
        fam_lists = extract_family_lists(sim_agg, actuals)
        for f, entries in fam_lists.items():
            for vals, actual in entries:
                agg[f].append([float(np.mean(vals)),
                               float(np.percentile(vals, 10)),
                               float(np.percentile(vals, 90)),
                               float(actual)])
        n_ok += 1
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"n_matches": n_ok, "fams": agg}, open(args.out, "w"))
    print(f"\nCaptured {n_ok} matches -> {args.out}")
    for f in ALL_FAMILIES:
        c = coverage_from_stats(agg[f], 1.0) if agg[f] else float("nan")
        print(f"  {f:20s} n={len(agg[f]):5d}  cov@k=1 {c*100:5.1f}%")


def coverage_from_stats(rows, k):
    """rows = [[m,p10,p90,actual]]; exact coverage under fan-out k."""
    hit = n = 0
    for m, p10, p90, actual in rows:
        lo = m + k * (p10 - m)
        hi = m + k * (p90 - m)
        hit += 1 if (lo <= actual <= hi) else 0
        n += 1
    return hit / n if n else float("nan")


def fit_k(rows, target=0.80, klo=0.5, khi=6.0, tol=1e-3):
    """Bisection for k s.t. coverage == target (monotone increasing in k)."""
    if not rows:
        return 1.0
    c_lo, c_hi = coverage_from_stats(rows, klo), coverage_from_stats(rows, khi)
    if c_lo >= target:
        return klo
    if c_hi <= target:
        return khi
    for _ in range(60):
        km = 0.5 * (klo + khi)
        cm = coverage_from_stats(rows, km)
        if abs(cm - target) < tol:
            return km
        if cm < target:
            klo = km
        else:
            khi = km
    return 0.5 * (klo + khi)


def cmd_fit(args):
    d = json.load(open(args.val))
    fams = d["fams"]
    ks = {}
    print(f"Fitting per-family k to {args.target:.0%} coverage on "
          f"{d['n_matches']} val matches:\n")
    print(f"{'family':20s}{'n':>7}{'cov@1':>9}{'k*':>8}{'cov@k*':>9}")
    for f in ALL_FAMILIES:
        rows = fams.get(f, [])
        if not rows:
            ks[f] = 1.0
            continue
        c1 = coverage_from_stats(rows, 1.0)
        k = fit_k(rows, target=args.target)
        ck = coverage_from_stats(rows, k)
        ks[f] = k
        print(f"{f:20s}{len(rows):>7}{c1*100:>8.1f}{k:>8.3f}{ck*100:>8.1f}")
    json.dump(ks, open(args.out, "w"), indent=2)
    print(f"\nSaved k -> {args.out}")


def cmd_score(args):
    ks = json.load(open(args.k))
    all_files = sorted(Path(args.test_dir).glob("*.json"))
    files = all_files if args.n_matches == "all" else all_files[: int(args.n_matches)]
    print(f"Scoring {len(files)} test matches x {args.n_sims} sims (seed {args.seed})")
    print(f"k per family: {json.dumps({k: round(v,3) for k,v in ks.items()})}")

    engine = build_engine(args.seed)
    detail_base, detail_disp = [], []
    for _, mid, sim_agg, actuals in iter_sims(engine, files, args.n_sims, args.seed):
        # Baseline (k=1) BEFORE mutation.
        detail_base.append(build_observations(mid, sim_agg, actuals))
        # Fan out each family's per-sim list in sim_agg, then re-observe.
        for fam, key in FAMILY_TO_AGG_KEY.items():
            k = ks.get(fam, 1.0)
            if k == 1.0:
                continue
            obj = sim_agg[key]
            if isinstance(obj, dict):
                for kk in list(obj.keys()):
                    if obj[kk]:
                        obj[kk] = fan_out(obj[kk], k)
            elif isinstance(obj, list):
                if obj:
                    sim_agg[key] = fan_out(obj, k)
        detail_disp.append(build_observations(mid, sim_agg, actuals))

    Path(args.out_base).parent.mkdir(parents=True, exist_ok=True)
    json.dump(detail_base, open(args.out_base, "w"), indent=2, default=float)
    json.dump(detail_disp, open(args.out_disp, "w"), indent=2, default=float)
    print(f"\nBaseline detail -> {args.out_base}")
    print(f"Dispersion detail -> {args.out_disp}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("capture-val")
    a.add_argument("--src-dir", default="data/t20s_json")
    a.add_argument("--date-lo", default="2024-12-31")
    a.add_argument("--date-hi", default="2025-06-29")
    a.add_argument("--n-matches", default="180")
    a.add_argument("--n-sims", type=int, default=100)
    a.add_argument("--seed", type=int, default=42)
    a.add_argument("--out", default="models/auto/a13/val_capture.json")
    a.set_defaults(func=cmd_capture_val)

    b = sub.add_parser("fit")
    b.add_argument("--val", default="models/auto/a13/val_capture.json")
    b.add_argument("--target", type=float, default=0.80)
    b.add_argument("--out", default="models/auto/a13/k.json")
    b.set_defaults(func=cmd_fit)

    c = sub.add_parser("score")
    c.add_argument("--test-dir", default="data/polymarket_test")
    c.add_argument("--n-matches", default="all")
    c.add_argument("--n-sims", type=int, default=100)
    c.add_argument("--seed", type=int, default=42)
    c.add_argument("--k", default="models/auto/a13/k.json")
    c.add_argument("--out-base", default="models/auto/a13/detail_base_n261.json")
    c.add_argument("--out-disp", default="models/auto/a13/detail_disp_n261.json")
    c.set_defaults(func=cmd_score)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
