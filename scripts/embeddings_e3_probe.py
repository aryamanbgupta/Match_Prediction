"""E3 scoreboard: does a (player, season) vector predict NEXT-season
performance better than career vectors or raw rates?

Season offsets cannot move val/test LL (future seasons are pinned to the
zero UNK row by construction), so the rung is judged inside the train era:
for batters with >=200 balls in season y and >=100 balls in season y+1
(y+1 <= 2024), predict next-season mean runs-per-ball with:

  a) raw:    this-season mean runs/ball          (naive baseline)
  b) career: career-to-date mean runs/ball        (shrinkage-flavored)
  c) ridge on the E3 season vector (career offset + season offset +
     proj of that season's mean EB anchor)
  d) ridge on the career-only vector (same minus season offset)

Metric: 5-fold CV R^2 and Spearman, folds grouped by player so no player
appears in both train and test folds.

Usage: uv run python scripts/embeddings_e3_probe.py --dir models/embeddings/e3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from embeddings_e1 import EB_BAT_COLS, E1Model  # noqa: E402

from scipy.stats import spearmanr  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402


def cv_scores(X, y, groups, name):
    r2s, sps = [], []
    for tr_idx, te_idx in GroupKFold(n_splits=5).split(X, y, groups):
        m = Ridge(alpha=1.0).fit(X[tr_idx], y[tr_idx])
        pred = m.predict(X[te_idx])
        ss_res = ((y[te_idx] - pred) ** 2).sum()
        ss_tot = ((y[te_idx] - y[te_idx].mean()) ** 2).sum()
        r2s.append(1 - ss_res / ss_tot)
        sps.append(spearmanr(pred, y[te_idx]).correlation)
    print(f"  {name:<28} R2 {np.mean(r2s):+.4f}  Spearman {np.mean(sps):.4f}",
          flush=True)
    return float(np.mean(r2s)), float(np.mean(sps))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("models/embeddings/e3"))
    args = ap.parse_args()

    emb = np.load(args.dir / "embeddings.npz", allow_pickle=True)
    batters = list(emb["batter_ids"])
    bat_pos = {p: i for i, p in enumerate(batters)}
    season_vecs = {tuple(k.split("|")): emb["bat_season_vecs"][i]
                   for i, k in enumerate(emb["bat_season_keys"])}

    cfg = json.loads((args.dir / "metrics.json").read_text())["config"]
    model = E1Model(len(batters), len(emb["bowler_ids"]), cfg["dim"],
                    cfg["hidden"],
                    ctx_dim=(4 if cfg["context"] else 0)
                    + (6 if cfg.get("venue_ctx") else 0),
                    eb_anchor=True)
    sd = torch.load(args.dir / "model.pt", map_location="cpu")
    proj_w = sd["batter_proj.weight"].numpy()
    proj_b = sd["batter_proj.bias"].numpy()
    career_off = sd["batter_emb.weight"].numpy()[:len(batters)]

    print("building per-(batter, season) aggregates from train...", flush=True)
    tr = pd.read_parquet("data/xgb_data_v3/cricket_data_v3_train.parquet",
                         columns=["batter_id", "batter_runs", "match_date"]
                         + EB_BAT_COLS)
    tr["year"] = pd.to_datetime(tr["match_date"]).dt.year
    g = tr.groupby(["batter_id", "year"])
    agg = g.agg(balls=("batter_runs", "size"),
                rpb=("batter_runs", "mean")).reset_index()
    eb_mean = g[EB_BAT_COLS].mean().reset_index()
    agg = agg.merge(eb_mean, on=["batter_id", "year"])

    # career-to-date rpb (excluding current season would be stricter; we
    # use trailing-inclusive career mean up to season y)
    agg = agg.sort_values(["batter_id", "year"])
    agg["cum_runs"] = agg.groupby("batter_id").apply(
        lambda d: (d["rpb"] * d["balls"]).cumsum()).reset_index(drop=True)
    agg["cum_balls"] = agg.groupby("batter_id")["balls"].cumsum()
    agg["career_rpb"] = agg["cum_runs"] / agg["cum_balls"]

    nxt = agg[["batter_id", "year", "balls", "rpb"]].copy()
    nxt["year"] -= 1
    merged = agg.merge(nxt, on=["batter_id", "year"],
                       suffixes=("", "_next"))
    merged = merged[(merged["balls"] >= 200) & (merged["balls_next"] >= 100)]
    print(f"{len(merged)} batter-season pairs "
          f"({merged['batter_id'].nunique()} batters, "
          f"{merged['year'].min()}-{merged['year'].max()})", flush=True)

    y = merged["rpb_next"].to_numpy()
    groups = merged["batter_id"].to_numpy()

    anch = merged[EB_BAT_COLS].to_numpy(np.float32) @ proj_w.T + proj_b
    co = np.stack([career_off[bat_pos[p]] for p in merged["batter_id"]])
    so = np.stack([season_vecs.get((p, str(yr)), np.zeros(cfg["dim"],
                                                          dtype=np.float32))
                   for p, yr in zip(merged["batter_id"], merged["year"])])

    results = {}
    print("\n=== Next-season runs-per-ball prediction ===", flush=True)
    results["raw_this_season"] = cv_scores(
        merged[["rpb"]].to_numpy(), y, groups, "raw this-season rpb")
    results["career_rpb"] = cv_scores(
        merged[["career_rpb"]].to_numpy(), y, groups, "career-to-date rpb")
    results["career_vector"] = cv_scores(
        np.hstack([anch + co]), y, groups, "career vector (anchor+off)")
    results["season_vector"] = cv_scores(
        np.hstack([anch + co + so]), y, groups, "E3 season vector")
    results["season_vector_plus_raw"] = cv_scores(
        np.hstack([anch + co + so, merged[["rpb", "career_rpb"]].to_numpy()]),
        y, groups, "season vector + raw rates")

    (args.dir / "e3_probe_results.json").write_text(
        json.dumps(results, indent=2))
    print(f"\nsaved {args.dir}/e3_probe_results.json", flush=True)


if __name__ == "__main__":
    main()
