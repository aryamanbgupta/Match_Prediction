# manifest-exempt: embeddings-ladder diagnostic artifact namespace
"""E1 diagnosis: where does the pure-ID model win/lose vs the baselines?

Slices val/test log-loss by player familiarity, runs linear probes on the
learned embeddings, and prints the neighbor panel. Appends nothing to the
kit — read-only over models/embeddings/{eval_kit,e1}.

Usage:  uv run python scripts/embeddings_e1_analysis.py [--dir models/embeddings/e1]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from embeddings_e1 import CTX_COLS, E1Model, load_split, make_ctx, to_idx  # noqa: E402

KIT = Path("models/embeddings/eval_kit")
_ap = argparse.ArgumentParser()
_ap.add_argument("--dir", type=Path, default=Path("models/embeddings/e1"))
E1 = _ap.parse_args().dir

EB_COLS = json.loads((KIT / "manifest.json").read_text())["eb_cols"]
PANEL = json.loads((KIT / "manifest.json").read_text())["neighbor_panel"]


def ll_vec(y, probs):
    return -np.log(np.clip(probs[np.arange(len(y)), y], 1e-15, 1.0))


def run_probes_and_neighbors(emb, batters, bowlers):
    probes = pd.read_parquet(KIT / "probe_labels.parquet")

    print("\n=== Linear probes (players with >=200 balls in role) ===", flush=True)
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    bat_vecs, bowl_vecs = emb["batter_vecs"], emb["bowler_vecs"]
    for role, vecs, ids, count_col, targets in [
        ("batter", bat_vecs, batters, "train_balls_batting", ["bat_hand"]),
        ("bowler", bowl_vecs, bowlers, "train_balls_bowling",
         ["bowl_kind", "bowl_arm"]),
    ]:
        sub = probes.reindex(ids)
        keep = (sub[count_col] >= 200).to_numpy()
        for t in targets:
            yy = sub[t].to_numpy()[keep]
            ok = pd.notna(yy)
            if ok.sum() < 50:
                continue
            X, yy2 = vecs[keep][ok], yy[ok]
            acc = cross_val_score(LogisticRegression(max_iter=1000), X, yy2,
                                  cv=5).mean()
            base = pd.Series(yy2).value_counts(normalize=True).iloc[0]
            print(f"  {role} -> {t}: probe acc {acc:.3f} "
                  f"(majority {base:.3f}, n={ok.sum()})", flush=True)

    print("\n=== Neighbor panel (cosine, >=500 balls in role) ===", flush=True)
    enriched = pd.read_csv("data/all_players_enriched.csv")
    id2name = {i: (n if isinstance(n, str) else str(i))
               for i, n in zip(enriched["cricsheet_id"], enriched["unique_name"])}
    for role, vecs, ids, count_col in [
        ("batting", emb["batter_vecs"], batters, "train_balls_batting"),
        ("bowling", emb["bowler_vecs"], bowlers, "train_balls_bowling"),
    ]:
        sub = probes.reindex(ids)
        keep_idx = np.where((sub[count_col] >= 500).to_numpy())[0]
        V = vecs[keep_idx]
        V = V / np.linalg.norm(V, axis=1, keepdims=True)
        kept_ids = [ids[i] for i in keep_idx]
        pos = {p: i for i, p in enumerate(kept_ids)}
        for name, pid in PANEL.items():
            if pid not in pos:
                continue
            sims = V @ V[pos[pid]]
            top = np.argsort(-sims)[1:7]
            nbrs = ", ".join(id2name.get(kept_ids[i], kept_ids[i]) for i in top)
            print(f"  [{role}] {name}: {nbrs}", flush=True)


def main() -> None:
    emb = np.load(E1 / "embeddings.npz", allow_pickle=True)
    batters, bowlers = list(emb["batter_ids"]), list(emb["bowler_ids"])
    bat_vocab = {p: i for i, p in enumerate(batters)}
    bowl_vocab = {p: i for i, p in enumerate(bowlers)}
    cfg = json.loads((E1 / "metrics.json").read_text())["config"]
    use_ctx = bool(cfg.get("context", False))
    ctx_dim = (4 if use_ctx else 0) + (6 if cfg.get("venue_ctx") else 0)
    model = E1Model(len(batters), len(bowlers), cfg["dim"], cfg["hidden"],
                    ctx_dim=ctx_dim, eb_anchor=bool(cfg.get("eb_anchor")))
    model.load_state_dict(torch.load(E1 / "model.pt", map_location="cpu"))
    model.eval()
    if cfg.get("eb_anchor"):
        # Per-ball anchored predictions can't be reproduced from static
        # vectors; the trainer's metrics.json carries the sliced LLs.
        print("(eb-anchor run: sliced LL lives in metrics.json — "
              "skipping to probes/neighbors)", flush=True)
        run_probes_and_neighbors(emb, batters, bowlers)
        return

    probes = pd.read_parquet(KIT / "probe_labels.parquet")
    clf_b0b = joblib.load(KIT / "b0b_logistic.joblib")
    manifest = json.loads((KIT / "manifest.json").read_text())
    prior = np.array(manifest["train_prior"])

    print("=== Sliced log-loss (validation) ===", flush=True)
    val_full = pd.read_parquet("data/xgb_data_v3/cricket_data_v3_validation.parquet",
                               columns=["batter_id", "bowler_id", "ball_outcome"]
                               + EB_COLS + (CTX_COLS if use_ctx else []))
    val_full["y"] = val_full["ball_outcome"].map(
        {0: 0, 1: 1, 2: 2, 4: 3, 6: 4, -1: 5}).astype(np.int64)
    y = val_full["y"].to_numpy()

    # model probs (mean-emb UNK, same as training eval)
    bi = to_idx(val_full["batter_id"], bat_vocab)
    wi = to_idx(val_full["bowler_id"], bowl_vocab)
    with torch.no_grad():
        # UNK = last table row. For E1 checkpoints (id_dropout 0) it was set
        # to the mean at eval time; replicate that if the saved row is ~init.
        if cfg.get("id_dropout", 0.0) == 0.0:
            model.batter_emb.weight[len(batters)] = \
                model.batter_emb.weight[:len(batters)].mean(0)
            model.bowler_emb.weight[len(bowlers)] = \
                model.bowler_emb.weight[:len(bowlers)].mean(0)
        parts = [
            model.batter_emb.weight[np.where(bi < 0, len(batters), bi)],
            model.bowler_emb.weight[np.where(wi < 0, len(bowlers), wi)],
        ]
        if use_ctx:
            parts.append(torch.tensor(make_ctx(val_full)))
        p_e1 = torch.softmax(model.mlp(torch.cat(parts, dim=-1)), -1).numpy()
    p_b0b = clf_b0b.predict_proba(val_full[EB_COLS].to_numpy(np.float32))
    p_b0a = np.tile(prior, (len(y), 1))

    ll_e1, ll_b0b, ll_b0a = ll_vec(y, p_e1), ll_vec(y, p_b0b), ll_vec(y, p_b0a)

    bat_balls = probes["train_balls_batting"]
    bowl_balls = probes["train_balls_bowling"]
    bcount = val_full["batter_id"].map(bat_balls).fillna(0).to_numpy()
    wcount = val_full["bowler_id"].map(bowl_balls).fillna(0).to_numpy()
    min_count = np.minimum(bcount, wcount)

    buckets = [(-1, 0, "unknown (0 balls)"), (1, 200, "1-199"),
               (200, 1000, "200-999"), (1000, 5000, "1000-4999"),
               (5000, 10**9, "5000+")]
    print(f"{'bucket':>18} {'n':>8} {'E1':>8} {'B0b':>8} {'B0a':>8}")
    for lo, hi, label in buckets:
        m = (min_count >= max(lo, 0)) & (min_count < hi) if lo >= 0 \
            else (min_count == 0)
        if m.sum() == 0:
            continue
        print(f"{label:>18} {m.sum():>8,} {ll_e1[m].mean():>8.4f} "
              f"{ll_b0b[m].mean():>8.4f} {ll_b0a[m].mean():>8.4f}", flush=True)

    run_probes_and_neighbors(emb, batters, bowlers)


if __name__ == "__main__":
    main()
