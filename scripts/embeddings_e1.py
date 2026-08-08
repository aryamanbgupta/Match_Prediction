"""E1 — pure-ID (batter|bowler)2vec, the base rung of the embedding ladder.

Alcorn-faithful: predict the 6-class ball outcome from (batter, bowler)
embedding lookups ONLY. No context features. See
~/Projects/sloan-sports-analytics/EMBEDDINGS_TRANSFORMER_DESIGN.md (rung E1)
and models/embeddings/eval_kit/ for the frozen scoreboard it is judged on.

Question answered: how much outcome signal lives in player identity alone,
and does lateral generalization (unseen batter-bowler pairs) already beat
the EB-logistic baseline there?

Explicit E1 choices (revisited at later rungs):
  * Players absent from train map to a mean-embedding UNK at eval time
    (E4's EB anchoring replaces this properly).
  * Static career vectors (E3 adds time).
  * All train rows used, including wides/no-balls — matches production target.

Artifacts → models/embeddings/e1/. Golden is never read.

Usage:
    uv run python scripts/embeddings_e1.py [--dim 32] [--seed 42]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

KIT = Path("models/embeddings/eval_kit")
DATA = Path("data/xgb_data_v3")
OUT = Path("models/embeddings/e1")

CLASS_MAPPING = {0: 0, 1: 1, 2: 2, 4: 3, 6: 4, -1: 5}


class E1Model(nn.Module):
    def __init__(self, n_batters: int, n_bowlers: int, dim: int, hidden: int):
        super().__init__()
        self.batter_emb = nn.Embedding(n_batters, dim)
        self.bowler_emb = nn.Embedding(n_bowlers, dim)
        self.mlp = nn.Sequential(
            nn.Linear(2 * dim, hidden), nn.ReLU(), nn.Linear(hidden, 6)
        )

    def forward(self, batter_idx, bowler_idx):
        x = torch.cat([self.batter_emb(batter_idx),
                       self.bowler_emb(bowler_idx)], dim=-1)
        return self.mlp(x)


def load_split(name: str) -> pd.DataFrame:
    df = pd.read_parquet(DATA / f"cricket_data_v3_{name}.parquet",
                         columns=["batter_id", "bowler_id", "ball_outcome"])
    df["y"] = df["ball_outcome"].map(CLASS_MAPPING).astype(np.int64)
    return df


def to_idx(series: pd.Series, vocab: dict[str, int]) -> np.ndarray:
    """Map IDs to indices; unknown players -> -1 (resolved to UNK later)."""
    return series.map(vocab).fillna(-1).to_numpy(dtype=np.int64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)

    print("loading splits...", flush=True)
    train, val, test = load_split("train"), load_split("validation"), load_split("test")
    masks = np.load(KIT / "unseen_pair_masks.npz")

    batters = sorted(train["batter_id"].unique())
    bowlers = sorted(train["bowler_id"].unique())
    bat_vocab = {p: i for i, p in enumerate(batters)}
    bowl_vocab = {p: i for i, p in enumerate(bowlers)}
    print(f"vocab: {len(batters)} batters, {len(bowlers)} bowlers", flush=True)

    Xb_tr = torch.tensor(to_idx(train["batter_id"], bat_vocab))
    Xw_tr = torch.tensor(to_idx(train["bowler_id"], bowl_vocab))
    y_tr = torch.tensor(train["y"].to_numpy())

    model = E1Model(len(batters), len(bowlers), args.dim, args.hidden).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params:,}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    def eval_probs(df: pd.DataFrame) -> np.ndarray:
        """Predict probs; unknown players get the mean embedding (UNK)."""
        bi = to_idx(df["batter_id"], bat_vocab)
        wi = to_idx(df["bowler_id"], bowl_vocab)
        with torch.no_grad():
            # temporarily append mean-embedding rows as UNK index
            bat_mean = model.batter_emb.weight.mean(0, keepdim=True)
            bowl_mean = model.bowler_emb.weight.mean(0, keepdim=True)
            bat_w = torch.cat([model.batter_emb.weight, bat_mean])
            bowl_w = torch.cat([model.bowler_emb.weight, bowl_mean])
            bi = torch.tensor(np.where(bi < 0, len(batters), bi)).to(device)
            wi = torch.tensor(np.where(wi < 0, len(bowlers), wi)).to(device)
            out = []
            for s in range(0, len(bi), 65536):
                x = torch.cat([bat_w[bi[s:s+65536]], bowl_w[wi[s:s+65536]]], dim=-1)
                out.append(torch.softmax(model.mlp(x), dim=-1).cpu().numpy())
        return np.concatenate(out)

    def ll(y: np.ndarray, probs: np.ndarray) -> float:
        p = np.clip(probs[np.arange(len(y)), y], 1e-15, 1.0)
        return float(-np.mean(np.log(p)))

    n = len(y_tr)
    best_val, best_state, bad = np.inf, None, 0
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n)
        t0, tot = time.time(), 0.0
        for s in range(0, n, args.batch):
            idx = perm[s:s + args.batch]
            opt.zero_grad()
            logits = model(Xb_tr[idx].to(device), Xw_tr[idx].to(device))
            loss = loss_fn(logits, y_tr[idx].to(device))
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        model.eval()
        val_probs = eval_probs(val)
        vll = ll(val["y"].to_numpy(), val_probs)
        print(f"epoch {epoch}: train_ll={tot/n:.4f} val_ll={vll:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        if vll < best_val - 1e-5:
            best_val, bad = vll, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print("early stop", flush=True)
                break
    model.load_state_dict(best_state)
    model.eval()

    # --- Scoreboard ---------------------------------------------------------
    metrics: dict = {"config": vars(args), "n_params": n_params,
                     "device": device, "best_epoch_val_ll": round(best_val, 4)}
    for name, df in [("validation", val), ("test", test)]:
        probs = eval_probs(df)
        y = df["y"].to_numpy()
        m = masks[name]
        unk = ((to_idx(df["batter_id"], bat_vocab) < 0)
               | (to_idx(df["bowler_id"], bowl_vocab) < 0))
        metrics[f"{name}_ll"] = round(ll(y, probs), 4)
        metrics[f"{name}_ll_unseen_pairs"] = round(ll(y[m], probs[m]), 4)
        metrics[f"{name}_unknown_player_ball_frac"] = round(float(unk.mean()), 4)
    print(json.dumps({k: v for k, v in metrics.items() if "ll" in str(k)},
                     indent=2), flush=True)

    # --- Save embeddings + model -------------------------------------------
    np.savez_compressed(
        OUT / "embeddings.npz",
        batter_ids=np.array(batters), bowler_ids=np.array(bowlers),
        batter_vecs=model.batter_emb.weight.detach().cpu().numpy(),
        bowler_vecs=model.bowler_emb.weight.detach().cpu().numpy(),
    )
    torch.save(model.state_dict(), OUT / "model.pt")
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"saved to {OUT}/", flush=True)


if __name__ == "__main__":
    main()
