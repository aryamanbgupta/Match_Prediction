"""Honest baseline check: does the ball model add anything over trivial baselines?

Two questions, both raised as fair objections to the first MLC writeup:

1. TOP SCORER. "27% vs a 9% base rate" is a strawman — only the top ~6 batters can
   realistically top-score. So compare the sim's #1 pick against baselines a person
   would actually use: random-in-XI, always-the-opener, best fixed batting position,
   and "pick the highest career-strike-rate batter". If the sim ~= the best simple
   baseline, it has no edge.

2. STRIKE-RATE CORRELATION. The +0.52 rank-corr (predicted vs actual SR) — is that
   skill, or is the model just regurgitating each batter's career strike rate? Test:
   compute the SAME correlation for a pure career-reputation proxy (the batter's
   EB-shrunk career outcome distribution, batter_p*, which is the model's main batter
   feature). If career-proxy correlates with actual as well as the model does, the
   model is a lookup table, not an edge.

Teacher-forced over actual MLC 2025 deliveries (data/xgb_data_v3), same model as the
sim. Output: mlc/reports/mlc_baseline_check.md

Usage: uv run python mlc/scripts/baseline_check.py
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]
MODEL = REPO / "models" / "xgb_v3"
BALL_DIR = REPO / "data" / "xgb_data_v3"
PEOPLE = REPO / "data" / "cricsheet_people.csv"
OUT = REPO / "mlc" / "reports" / "mlc_baseline_check.md"

MLC_VENUES = {"Grand Prairie Stadium, Dallas",
              "Central Broward Regional Park Stadium Turf Ground, Lauderhill",
              "Oakland Coliseum,Oakland"}
DATE_LO, DATE_HI = "2025-06-12", "2025-07-13"
RUNS_BY_CLASS = {0: 0, 1: 1, 2: 2, 3: 4, 4: 6, 5: 0}


def load_balls():
    model = joblib.load(MODEL / "xgboost_model_v3.pkl")
    bat = joblib.load(MODEL / "batter_encoder_v3.pkl")
    bowl = joblib.load(MODEL / "bowler_encoder_v3.pkl")
    mu = joblib.load(MODEL / "matchup_encoder_v3.pkl")
    feat = [l.strip() for l in open(MODEL / "feature_columns_v3.txt") if l.strip()]
    bc = {str(c): i for i, c in enumerate(bat.classes_)}
    wc = {str(c): i for i, c in enumerate(bowl.classes_)}
    mc = {str(c): i for i, c in enumerate(mu.classes_)}
    frames = []
    for sp in ("validation", "test"):
        df = pd.read_parquet(BALL_DIR / f"cricket_data_v3_{sp}.parquet")
        df = df[df.venue.isin(MLC_VENUES) & df.match_date.astype(str).between(DATE_LO, DATE_HI)]
        frames.append(df)
    b = pd.concat(frames, ignore_index=True)
    b["batter_encoded"] = b.batter_id.astype(str).map(bc).fillna(-1).astype(int)
    b["bowler_encoded"] = b.bowler_id.astype(str).map(wc).fillna(-1).astype(int)
    b["matchup_type_encoded"] = b.matchup_type.astype(str).map(mc).fillna(-1).astype(int)
    b["venue_encoded"] = 0
    proba = model.predict_proba(b[feat].astype(np.float64))
    classes = list(model.classes_)
    runs_vec = np.array([RUNS_BY_CLASS[c] for c in classes], float)
    b["pred_runs"] = proba @ runs_vec
    b["act_runs"] = b.ball_outcome.clip(lower=0)
    # Career strike-rate proxy from EB-shrunk career outcome dist (model's batter feature)
    b["career_runs_pb"] = b.batter_p1 + 2 * b.batter_p2 + 4 * b.batter_p4 + 6 * b.batter_p6
    names = pd.read_csv(PEOPLE, usecols=["identifier", "name"])
    nm = dict(zip(names.identifier.astype(str), names.name.astype(str)))
    b["batter"] = b.batter_id.astype(str).map(nm).fillna(b.batter_id)
    return b


def main() -> int:
    b = load_balls()
    L = ["# MLC 2025 — honest baseline check\n",
         f"Teacher-forced over {len(b)} actual deliveries. Two questions: is the "
         "top-scorer call better than a sensible no-model baseline, and is the "
         "strike-rate correlation anything more than career reputation?\n"]

    # ---------- Q1: TOP SCORER vs fair baselines ----------
    # Per innings: batters with position (first-appearance order), balls, actual runs,
    # career proxy SR, and the model's predicted runs (teacher-forced on actual balls).
    first = b.groupby(["innings_id", "batter_id"]).agg(
        first_over=("over_idx", "min"),
        first_ball=("ball_idx", "min"),
        balls=("act_runs", "size"),
        act_runs=("act_runs", "sum"),
        pred_runs=("pred_runs", "sum"),
        career_pb=("career_runs_pb", "mean"),
        batter=("batter", "first")).reset_index()

    rows = []
    for inn, g in first.groupby("innings_id"):
        g = g.sort_values(["first_over", "first_ball"]).reset_index(drop=True)
        g["pos"] = np.arange(1, len(g) + 1)
        top_idx = g.act_runs.values.argmax()
        actual_top_pos = int(g.pos.iloc[top_idx])
        # reputation pick (pre-match, no peeking): highest career SR among the top-4
        # positions — "the team's best top-order bat".
        pool = g[g.pos <= 4]
        rep_pick_pos = int(pool.loc[pool.career_pb.idxmax(), "pos"]) if len(pool) else 1
        rows.append({"n_bat": len(g), "actual_top_pos": actual_top_pos,
                     "rep_hit": int(rep_pick_pos == actual_top_pos)})
    R = pd.DataFrame(rows)
    n = len(R)
    pos_rate = R.actual_top_pos.value_counts(normalize=True).sort_index()
    best_pos = int(pos_rate.idxmax())
    best_pos_rate = float(pos_rate.max())
    rand_in_xi = float((1 / R.n_bat).mean())

    # The model's legitimate top-scorer pick = the Monte-Carlo sim's #1 (it simulates
    # balls faced too — no peeking). Read it from the prop backtest detail.
    prop = json.loads((REPO / "mlc" / "reports" / "mlc_2025_prop_detail.json").read_text())
    from collections import defaultdict
    hit = tot = 0
    for m in prop:
        byteam = defaultdict(list)
        for o in m["obs"].get("top_batter", []):
            byteam[o["team"]].append(o)
        for _, rs in byteam.items():
            if rs:
                hit += int(max(rs, key=lambda r: r["p"])["y"] == 1); tot += 1
    sim_rate = hit / max(tot, 1)

    L += ["## Q1 — Top scorer: sim vs baselines a human would use\n",
          f"{n} team-innings. 'Hit' = the pick was the actual top scorer. The earlier "
          "'9% base rate' was a strawman (only the top order can realistically top-score). "
          "Fair comparison:\n",
          "| picker | hit rate | note |",
          "|---|---:|---|",
          f"| random among XI's batters | {rand_in_xi*100:.0f}% | the strawman I used before |",
          f"| always the opener (pos 1) | {float((R.actual_top_pos==1).mean())*100:.0f}% | zero-model, a-priori |",
          f"| always best position in hindsight (pos {best_pos}) | {best_pos_rate*100:.0f}% | zero-model, optimistic |",
          f"| best top-order bat by career SR | {float(R.rep_hit.mean())*100:.0f}% | reputation lookup, no sim |",
          f"| **our Monte-Carlo sim (#1 pick)** | **{sim_rate*100:.0f}%** | the model ({hit}/{tot}) |",
          "\n(Note: a *teacher-forced* 'pick highest predicted total over the actual balls "
          "faced' scores ~80% — but that's a leak: it conditions on who actually batted "
          "longest, which you don't know pre-match. The honest model number is the sim's "
          f"{sim_rate*100:.0f}%.)\n",
          "Top scorer by actual batting position:\n",
          "| position | " + " | ".join(str(int(p)) for p in pos_rate.index) + " |",
          "|---|" + "---|" * len(pos_rate),
          "| share | " + " | ".join(f"{v*100:.0f}%" for v in pos_rate.values) + " |\n",
          f"**Verdict:** the sim ({sim_rate*100:.0f}%) is level with just always backing a "
          f"top-order position ({best_pos_rate*100:.0f}%). No demonstrable edge on top scorer.\n"]

    # ---------- Q2: SR correlation — model vs career reputation ----------
    perb = b.groupby("batter").agg(
        balls=("act_runs", "size"), act_runs=("act_runs", "sum"),
        pred_runs=("pred_runs", "sum"), career_pb=("career_runs_pb", "mean")).reset_index()
    perb = perb[perb.balls >= 15].copy()
    perb["act_sr"] = perb.act_runs / perb.balls * 100
    perb["model_sr"] = perb.pred_runs / perb.balls * 100
    perb["career_sr"] = perb.career_pb * 100

    r_model = spearmanr(perb.model_sr, perb.act_sr).correlation
    r_career = spearmanr(perb.career_sr, perb.act_sr).correlation
    r_model_vs_career = spearmanr(perb.model_sr, perb.career_sr).correlation
    # does the model add anything beyond career? residual correlation
    # actual residual after removing career (linear), vs model residual after removing career
    def resid(y, x):
        x = np.c_[np.ones(len(x)), x]
        beta = np.linalg.lstsq(x, y, rcond=None)[0]
        return y - x @ beta
    act_res = resid(perb.act_sr.values, perb.career_sr.values)
    mod_res = resid(perb.model_sr.values, perb.career_sr.values)
    r_residual = spearmanr(mod_res, act_res).correlation

    L += ["## Q2 — Strike-rate correlation: skill or career lookup?\n",
          f"{len(perb)} batters with ≥15 balls. Spearman rank-corr with ACTUAL strike rate:\n",
          "| predictor | rank-corr vs actual SR |",
          "|---|---:|",
          f"| **career reputation only** (EB career outcome dist) | **{r_career:+.2f}** |",
          f"| full ball model (teacher-forced) | {r_model:+.2f} |",
          f"\n- Model vs career-proxy agreement: {r_model_vs_career:+.2f} (they rank batters "
          "almost identically).",
          f"\n- **Does the model beat the lookup?** After removing career reputation, the "
          f"model's *residual* correlation with actual SR is **{r_residual:+.2f}**. And even "
          "that is an optimistic ceiling: the model is teacher-forced on the realised innings, "
          "so a batter who actually survived into the high-strike-rate death overs gets fed "
          "those states — partly baking the outcome into the 'prediction'. A clean pre-match "
          "test would be lower. So the +0.5 headline is almost entirely 'good batters bat "
          "well', which you already know.\n",
          "Per-batter detail (top 18 by balls) — career vs model vs actual SR:\n",
          "| batter | balls | career SR | model SR | actual SR |",
          "|---|---:|---:|---:|---:|"]
    for _, r in perb.sort_values("balls", ascending=False).head(18).iterrows():
        L.append(f"| {r.batter} | {int(r.balls)} | {r.career_sr:.0f} | {r.model_sr:.0f} | {r.act_sr:.0f} |")

    L += ["\n## Bottom line\n",
          f"- **Top scorer:** sim {sim_rate*100:.0f}% ≈ best positional rule {best_pos_rate*100:.0f}%. "
          "No edge over 'back a top-order bat'.",
          f"- **Strike rate:** the {r_model:+.2f} correlation is the same as career reputation "
          f"alone ({r_career:+.2f}); residual over career is {r_residual:+.2f} and even that is "
          "inflated by teacher-forcing. The model is largely re-deriving career stats.\n"]

    OUT.write_text("\n".join(L) + "\n")
    print("TOP SCORER hit rates:")
    print(f"  random-in-XI {rand_in_xi*100:.0f}% | opener {(R.actual_top_pos==1).mean()*100:.0f}%"
          f" | best-pos(p{best_pos}) {best_pos_rate*100:.0f}% | career-SR-top-order {R.rep_hit.mean()*100:.0f}%"
          f" | SIM {sim_rate*100:.0f}%")
    print(f"SR rank-corr: career {r_career:+.2f} | model {r_model:+.2f} | "
          f"model⟂career residual {r_residual:+.2f} (teacher-forced, optimistic)")
    print(f"report → {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
