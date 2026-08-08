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

E1.5 (learned backoff — the one-ingredient upgrade after E1's diagnosis):
    --id-dropout p   During training, each ball's batter/bowler index is
                     independently replaced by a trainable UNK index with
                     probability p. The model is thereby forced to learn a
                     genuine "generic player" representation, which is also
                     used for players unseen in train at eval time.
    --emb-wd w       Weight decay applied to the embedding tables only
                     (MLP keeps 0.01). Pulls rare-player vectors toward
                     zero — the origin becomes "no information", which the
                     UNK/backoff makes meaningful.
    With --id-dropout 0 (default) eval falls back to E1's mean-embedding
    UNK, reproducing the original rung exactly.

Usage:
    uv run python scripts/embeddings_e1.py [--dim 32] [--seed 42]
    uv run python scripts/embeddings_e1.py --id-dropout 0.05 --emb-wd 0.1 \
        --out models/embeddings/e15
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

CLASS_MAPPING = {0: 0, 1: 1, 2: 2, 4: 3, 6: 4, -1: 5}


CTX_COLS = ["is_middle_overs", "is_death_overs", "wickets_in_hand",
            "chase_target"]


class E1Model(nn.Module):
    """Tables hold one extra row: the UNK index (last row of each table).

    ctx_dim > 0 (rung E2+): match-state features are concatenated to the
    player vectors BEFORE the MLP, so situational variance is absorbed by
    the context pathway and the embeddings are pushed toward pure identity.
    """

    def __init__(self, n_batters: int, n_bowlers: int, dim: int, hidden: int,
                 ctx_dim: int = 0):
        super().__init__()
        self.batter_emb = nn.Embedding(n_batters + 1, dim)
        self.bowler_emb = nn.Embedding(n_bowlers + 1, dim)
        self.ctx_dim = ctx_dim
        self.mlp = nn.Sequential(
            nn.Linear(2 * dim + ctx_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 6)
        )

    def forward(self, batter_idx, bowler_idx, ctx=None):
        parts = [self.batter_emb(batter_idx), self.bowler_emb(bowler_idx)]
        if self.ctx_dim:
            parts.append(ctx)
        return self.mlp(torch.cat(parts, dim=-1))


def make_ctx(df: pd.DataFrame) -> np.ndarray:
    """4 context features: phase (2 flags; PP = 00), wickets in hand /10,
    second-innings flag (chase_target > 0)."""
    return np.stack([
        df["is_middle_overs"].to_numpy(np.float32),
        df["is_death_overs"].to_numpy(np.float32),
        df["wickets_in_hand"].to_numpy(np.float32) / 10.0,
        (df["chase_target"].to_numpy(np.float32) > 0).astype(np.float32),
    ], axis=1)


def load_split(name: str, context: bool = False) -> pd.DataFrame:
    cols = ["batter_id", "bowler_id", "ball_outcome"]
    if context:
        cols += CTX_COLS
    df = pd.read_parquet(DATA / f"cricket_data_v3_{name}.parquet", columns=cols)
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
    ap.add_argument("--id-dropout", type=float, default=0.0,
                    help="prob of replacing each player index with UNK "
                         "during training (0 = E1 behavior)")
    ap.add_argument("--emb-wd", type=float, default=0.01,
                    help="weight decay on embedding tables (MLP stays 0.01)")
    ap.add_argument("--context", action="store_true",
                    help="rung E2: add phase/wickets/innings state inputs")
    ap.add_argument("--out", type=Path, default=Path("models/embeddings/e1"))
    args = ap.parse_args()
    OUT = args.out

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)

    print("loading splits...", flush=True)
    train = load_split("train", args.context)
    val = load_split("validation", args.context)
    test = load_split("test", args.context)
    masks = np.load(KIT / "unseen_pair_masks.npz")

    batters = sorted(train["batter_id"].unique())
    bowlers = sorted(train["bowler_id"].unique())
    bat_vocab = {p: i for i, p in enumerate(batters)}
    bowl_vocab = {p: i for i, p in enumerate(bowlers)}
    print(f"vocab: {len(batters)} batters, {len(bowlers)} bowlers", flush=True)

    Xb_tr = torch.tensor(to_idx(train["batter_id"], bat_vocab))
    Xw_tr = torch.tensor(to_idx(train["bowler_id"], bowl_vocab))
    y_tr = torch.tensor(train["y"].to_numpy())
    ctx_dim = 4 if args.context else 0
    Xc_tr = torch.tensor(make_ctx(train)) if args.context else None

    model = E1Model(len(batters), len(bowlers), args.dim, args.hidden,
                    ctx_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params:,}", flush=True)
    opt = torch.optim.AdamW([
        {"params": [model.batter_emb.weight, model.bowler_emb.weight],
         "weight_decay": args.emb_wd},
        {"params": model.mlp.parameters(), "weight_decay": 0.01},
    ], lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    UNK_B, UNK_W = len(batters), len(bowlers)

    def eval_probs(df: pd.DataFrame) -> np.ndarray:
        """Predict probs; unknown players map to the UNK row.

        With id-dropout the UNK row is trained; without it (E1), the UNK
        row is set to the mean embedding, reproducing E1's crutch.
        """
        bi = to_idx(df["batter_id"], bat_vocab)
        wi = to_idx(df["bowler_id"], bowl_vocab)
        with torch.no_grad():
            if args.id_dropout == 0.0:
                model.batter_emb.weight[UNK_B] = \
                    model.batter_emb.weight[:UNK_B].mean(0)
                model.bowler_emb.weight[UNK_W] = \
                    model.bowler_emb.weight[:UNK_W].mean(0)
            bi = torch.tensor(np.where(bi < 0, UNK_B, bi)).to(device)
            wi = torch.tensor(np.where(wi < 0, UNK_W, wi)).to(device)
            ctx = (torch.tensor(make_ctx(df)).to(device)
                   if args.context else None)
            out = []
            for s in range(0, len(bi), 65536):
                c = ctx[s:s+65536] if ctx is not None else None
                logits = model(bi[s:s+65536], wi[s:s+65536], c)
                out.append(torch.softmax(logits, dim=-1).cpu().numpy())
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
            bi, wi = Xb_tr[idx], Xw_tr[idx]
            if args.id_dropout > 0:
                bi = torch.where(torch.rand(len(bi)) < args.id_dropout,
                                 torch.tensor(UNK_B), bi)
                wi = torch.where(torch.rand(len(wi)) < args.id_dropout,
                                 torch.tensor(UNK_W), wi)
            c = Xc_tr[idx].to(device) if args.context else None
            logits = model(bi.to(device), wi.to(device), c)
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
    metrics: dict = {"config": {k: (str(v) if isinstance(v, Path) else v)
                                for k, v in vars(args).items()},
                     "n_params": n_params,
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
        batter_vecs=model.batter_emb.weight[:UNK_B].detach().cpu().numpy(),
        bowler_vecs=model.bowler_emb.weight[:UNK_W].detach().cpu().numpy(),
        batter_unk=model.batter_emb.weight[UNK_B].detach().cpu().numpy(),
        bowler_unk=model.bowler_emb.weight[UNK_W].detach().cpu().numpy(),
    )
    torch.save(model.state_dict(), OUT / "model.pt")
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"saved to {OUT}/", flush=True)


if __name__ == "__main__":
    main()
