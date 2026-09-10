# manifest-exempt: embeddings-ladder experimental artifact namespace
"""T1 — minimal innings-scope transformer over ball sequences (Phase 2).

Hypothesis (design doc + Phase-1 exit): within-innings history carries
information no EB marginal or state snapshot captures. Phase 1 answered the
identity question (EB-saturated), so tokens carry the EB features directly
(hybrid) — the learned-ID arm is intentionally absent at T1.

Token t = Linear(pre-ball features_t) + Embedding(outcome_{t-1})
  pre-ball: EB batter 18 + EB bowler 18 + venue 6 + state (phase flags,
  wickets, chasing, balls_remaining, score, RR, RRR) — all known before
  the ball. Outcomes enter only as history (BOS at t=0): the same causal
  discipline as the tracker pipeline, enforced by the shift + causal mask.

Scoreboard: per-ball LL on the frozen eval kit (the registered, converged
exact-feature logistic is 1.4433 validation / 1.4340 test), plus unseen-pair,
frequency-bucket and calibration diagnostics. Match simulation remains a
separate downstream gate.

Usage:
    uv run python scripts/transformer_t1.py [--dmodel 128] [--layers 2]
        [--out models/embeddings/t1]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from embeddings_e1 import (CLASS_MAPPING, CTX_COLS, EB_BAT_COLS,  # noqa: E402
                           EB_BOWL_COLS, VENUE_COLS, make_ctx)

KIT = Path("models/embeddings/eval_kit")
DATA = Path("data/xgb_data_v3")

STATE_COLS = ["balls_remaining", "score", "run_rate", "run_rate_required"]
BOS = 6  # outcome-history vocab: 6 classes + BOS
AUX_DIR = Path("models/embeddings/deepcrease_labels")
AUX_TASKS = ["shot", "line", "length", "control"]


def load_aux(split: str, n_rows: int, vocabs: dict | None,
             aux_dir: Path = AUX_DIR):
    """Per-row aux targets from the DeepCrease join; -1 = unlabeled.
    Vocabs are built from the train file and reused for val/test."""
    lab = pd.read_parquet(aux_dir / f"{split}.parquet")
    # Missing labels must stay missing until AFTER vocab construction:
    # the old fillna(-1)-then-vocab order made "-1" (and the shot
    # placeholder "-") real trainable classes that passed the tgt >= 0
    # mask, so the control/shot heads trained on a fabricated "unlabeled"
    # class. Mirrors run_xr_same_cohort._normalise_labels semantics.
    lab["control"] = lab["control"].map(
        lambda v: str(int(v)) if pd.notna(v) else np.nan)
    lab["shot"] = lab["shot"].replace("-", np.nan)
    if vocabs is None:
        vocabs = {t: {v: i for i, v in enumerate(sorted(
            lab[t].dropna().unique()))} for t in AUX_TASKS}
    out = {}
    for t in AUX_TASKS:
        arr = np.full(n_rows, -1, dtype=np.int64)
        vals = lab[t].map(vocabs[t]).fillna(-1).astype(np.int64)
        arr[lab["row_idx"].to_numpy()] = vals
        out[t] = arr
    return out, vocabs


def load_split(name: str, data_dir: Path = DATA) -> pd.DataFrame:
    cols = (["innings_id", "batter_id", "bowler_id", "ball_outcome"]
            + EB_BAT_COLS + EB_BOWL_COLS + VENUE_COLS + CTX_COLS + STATE_COLS)
    df = pd.read_parquet(
        data_dir / f"cricket_data_v3_{name}.parquet", columns=cols)
    df["y"] = df["ball_outcome"].map(CLASS_MAPPING).astype(np.int64)
    return df


def build_features(df: pd.DataFrame) -> np.ndarray:
    state = np.stack([
        df["balls_remaining"].to_numpy(np.float32) / 120.0,
        df["score"].to_numpy(np.float32) / 200.0,
        df["run_rate"].to_numpy(np.float32) / 12.0,
        np.clip(df["run_rate_required"].to_numpy(np.float32), 0, 36) / 12.0,
    ], axis=1)
    return np.hstack([
        df[EB_BAT_COLS + EB_BOWL_COLS + VENUE_COLS].to_numpy(np.float32),
        make_ctx(df, venue=False), state,
    ])


def build_innings(df: pd.DataFrame):
    """Per-innings row-index lists, preserving parquet (ball) order."""
    return list(df.groupby("innings_id", sort=False).indices.values())


class TokenMLPBlock(nn.Module):
    """Residual token-local feed-forward block with no sequence inputs."""

    def __init__(self, dmodel: int):
        super().__init__()
        self.norm = nn.LayerNorm(dmodel)
        self.linear1 = nn.Linear(dmodel, 2 * dmodel)
        self.linear2 = nn.Linear(2 * dmodel, dmodel)
        self.dropout = nn.Dropout(0.1)

    def forward(self, value):
        hidden = nn.functional.gelu(self.linear1(self.norm(value)))
        return value + self.dropout(self.linear2(self.dropout(hidden)))


class T1Model(nn.Module):
    def __init__(self, n_feats: int, dmodel: int, layers: int, heads: int,
                 aux_sizes: dict | None = None, arm: str = "full"):
        super().__init__()
        if arm not in {"full", "mlp", "no_attention", "no_history"}:
            raise ValueError(f"unknown T1 ablation arm: {arm}")
        self.arm = arm
        self.feat_proj = nn.Linear(n_feats, dmodel)
        if arm == "mlp":
            # Two token-local FF blocks per transformer layer approximately
            # match the full arm's parameter budget without sequence access.
            self.token_mlp = nn.Sequential(*[
                TokenMLPBlock(dmodel) for _ in range(2 * layers)])
        else:
            self.out_emb = nn.Embedding(7, dmodel)  # 6 classes + BOS
            self.pos_emb = nn.Embedding(200, dmodel)
            layer = nn.TransformerEncoderLayer(
                d_model=dmodel, nhead=heads, dim_feedforward=2 * dmodel,
                dropout=0.1, batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = nn.Linear(dmodel, 6)
        self.aux_heads = nn.ModuleDict(
            {t: nn.Linear(dmodel, n) for t, n in (aux_sizes or {}).items()})

    def forward(self, feats, prev_y, pad_mask):
        L = feats.shape[1]
        if self.arm == "mlp":
            h = self.token_mlp(self.feat_proj(feats))
            aux = {t: hd(h) for t, hd in self.aux_heads.items()}
            return self.head(h), aux

        pos = torch.arange(L, device=feats.device)
        if self.arm == "no_history":
            prev_y = torch.full_like(prev_y, BOS)
        x = self.feat_proj(feats) + self.out_emb(prev_y) + self.pos_emb(pos)
        if self.arm == "no_attention":
            # Preserve the transformer's parameters and token pathway while
            # preventing all cross-token mixing. Each token can still use its
            # immediately preceding outcome via the causally shifted input.
            attn_mask = ~torch.eye(L, dtype=torch.bool, device=feats.device)
        else:
            attn_mask = torch.triu(
                torch.ones((L, L), dtype=torch.bool, device=feats.device),
                diagonal=1)
        # In the diagonal arm, padding cannot mix into real tokens. Passing a
        # key-padding mask would leave each padded query with no legal key and
        # produce NaNs, so padded outputs are simply computed and discarded.
        key_padding = None if self.arm == "no_attention" else pad_mask
        h = self.encoder(x, mask=attn_mask,
                         src_key_padding_mask=key_padding)
        aux = {t: hd(h) for t, hd in self.aux_heads.items()}
        return self.head(h), aux


def collate(idx_lists, feats, y, device, aux=None):
    B = len(idx_lists)
    L = max(len(ix) for ix in idx_lists)
    f = np.zeros((B, L, feats.shape[1]), dtype=np.float32)
    py = np.full((B, L), BOS, dtype=np.int64)
    ty = np.zeros((B, L), dtype=np.int64)
    pad = np.ones((B, L), dtype=bool)
    ax = {t: np.full((B, L), -1, dtype=np.int64) for t in (aux or {})}
    for b, ix in enumerate(idx_lists):
        n = len(ix)
        f[b, :n] = feats[ix]
        ty[b, :n] = y[ix]
        py[b, 1:n] = y[ix][:-1]  # outcome enters as NEXT ball's history
        pad[b, :n] = False
        for t in ax:
            ax[t][b, :n] = aux[t][ix]
    return (torch.tensor(f).to(device), torch.tensor(py).to(device),
            torch.tensor(ty).to(device), torch.tensor(pad).to(device),
            {t: torch.tensor(v).to(device) for t, v in ax.items()})


def calibration_metrics(probs: np.ndarray, y: np.ndarray,
                        n_bins: int = 15) -> dict:
    """Multiclass Brier and confidence ECE with fixed-width bins."""
    onehot = np.eye(probs.shape[1], dtype=np.float32)[y]
    brier = np.square(probs - onehot).sum(axis=1).mean()
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == y
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        selected = (confidence >= lo) & (
            confidence < hi if hi < 1.0 else confidence <= hi)
        if selected.any():
            ece += selected.mean() * abs(
                confidence[selected].mean() - correct[selected].mean())
    return {"brier": round(float(brier), 6),
            "confidence_ece_15": round(float(ece), 6)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dmodel", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"],
                    default="auto")
    ap.add_argument("--arm", choices=["full", "mlp", "no_attention",
                                      "no_history"], default="full")
    ap.add_argument("--aux", action="store_true",
                    help="T1.5: multi-task heads on DeepCrease "
                         "shot/line/length/control labels")
    ap.add_argument("--aux-weight", type=float, default=0.2)
    ap.add_argument("--data-dir", type=Path, default=DATA)
    ap.add_argument("--kit-dir", type=Path, default=KIT)
    ap.add_argument("--aux-dir", type=Path, default=AUX_DIR)
    ap.add_argument("--save-predictions", action="store_true",
                    help="write row-aligned probabilities for paired audits")
    ap.add_argument("--out", type=Path, default=Path("models/embeddings/t1"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device == "auto":
        device = ("mps" if torch.backends.mps.is_available()
                  else "cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = args.device
        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("--device mps requested but MPS is unavailable")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
    print(f"device={device}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)

    print("loading splits...", flush=True)
    train, val, test = (
        load_split("train", args.data_dir),
        load_split("validation", args.data_dir),
        load_split("test", args.data_dir),
    )
    F_tr, F_va, F_te = (build_features(d) for d in (train, val, test))
    y_tr, y_va, y_te = (d["y"].to_numpy() for d in (train, val, test))
    inn_tr, inn_va, inn_te = (build_innings(d) for d in (train, val, test))
    print(f"innings: train {len(inn_tr)}, val {len(inn_va)}, "
          f"test {len(inn_te)}; feats {F_tr.shape[1]}", flush=True)

    aux_tr = aux_va = aux_te = None
    aux_sizes = None
    vocabs = None
    if args.aux:
        aux_tr, vocabs = load_aux("train", len(train), None, args.aux_dir)
        aux_va, _ = load_aux("validation", len(val), vocabs, args.aux_dir)
        aux_te, _ = load_aux("test", len(test), vocabs, args.aux_dir)
        aux_sizes = {t: len(v) for t, v in vocabs.items()}
        print(f"aux tasks: {aux_sizes}", flush=True)

    model = T1Model(F_tr.shape[1], args.dmodel, args.layers, args.heads,
                    aux_sizes, arm=args.arm).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss(reduction="none")

    def eval_split(feats, y, innings, df, aux=None):
        model.eval()
        probs = np.zeros((len(y), 6), dtype=np.float32)
        aux_hits = {t: [0, 0] for t in (aux or {})}
        with torch.no_grad():
            for s in range(0, len(innings), args.batch):
                chunk = innings[s:s + args.batch]
                f, py, ty, pad, ax = collate(chunk, feats, y, device, aux)
                logits, aux_out = model(f, py, pad)
                p = torch.softmax(logits, dim=-1).cpu().numpy()
                for b, ix in enumerate(chunk):
                    probs[ix] = p[b, :len(ix)]
                for t in aux_hits:
                    tgt = ax[t]
                    sel = tgt >= 0
                    if sel.any():
                        pred = aux_out[t].argmax(-1)
                        aux_hits[t][0] += int((pred[sel] == tgt[sel]).sum())
                        aux_hits[t][1] += int(sel.sum())
        ll = -np.log(np.clip(probs[np.arange(len(y)), y], 1e-15, 1.0))
        aux_acc = {t: round(h / max(n, 1), 4) for t, (h, n) in aux_hits.items()}
        return probs, float(ll.mean()), ll, aux_acc

    best_val, best_state, bad = np.inf, None, 0
    order = np.arange(len(inn_tr))
    for epoch in range(args.epochs):
        model.train()
        np.random.shuffle(order)
        t0, tot, cnt = time.time(), 0.0, 0
        for s in range(0, len(order), args.batch):
            chunk = [inn_tr[i] for i in order[s:s + args.batch]]
            f, py, ty, pad, ax = collate(chunk, F_tr, y_tr, device, aux_tr)
            opt.zero_grad()
            logits, aux_out = model(f, py, pad)
            raw = loss_fn(logits.reshape(-1, 6), ty.reshape(-1))
            keep = (~pad).reshape(-1).float()
            loss = (raw * keep).sum() / keep.sum()
            for t, tgt in ax.items():
                sel = (tgt >= 0).reshape(-1)
                if sel.any():
                    a = nn.functional.cross_entropy(
                        aux_out[t].reshape(-1, aux_out[t].shape[-1])[sel],
                        tgt.reshape(-1)[sel])
                    loss = loss + args.aux_weight * a
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * keep.sum().item()
            cnt += keep.sum().item()
        _, vll, _, _ = eval_split(F_va, y_va, inn_va, val)
        print(f"epoch {epoch}: train_ll={tot/cnt:.4f} val_ll={vll:.4f} "
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

    # --- Scoreboard ---------------------------------------------------------
    masks = np.load(args.kit_dir / "unseen_pair_masks.npz")
    probes_df = pd.read_parquet(args.kit_dir / "probe_labels.parquet")
    metrics = {"config": {k: (str(v) if isinstance(v, Path) else v)
                          for k, v in vars(args).items()},
               "n_params": sum(p.numel() for p in model.parameters()),
               "device": device}
    for name, feats, y, innings, df, aux in [
        ("validation", F_va, y_va, inn_va, val, aux_va),
        ("test", F_te, y_te, inn_te, test, aux_te),
    ]:
        probs, mean_ll, ll_vec, aux_acc = eval_split(feats, y, innings, df, aux)
        if aux_acc:
            metrics[f"{name}_aux_acc"] = aux_acc
        m = masks[name]
        metrics[f"{name}_ll"] = round(mean_ll, 4)
        metrics[f"{name}_ll_unseen_pairs"] = round(float(ll_vec[m].mean()), 4)
        metrics[f"{name}_calibration"] = calibration_metrics(probs, y)
        if args.save_predictions:
            np.savez_compressed(
                args.out / f"predictions_{name}.npz",
                probs=probs, y=y,
                innings_id=np.asarray(
                    df["innings_id"].astype(str).tolist(), dtype=str))
        if name == "validation":
            bc = df["batter_id"].map(probes_df["train_balls_batting"]).fillna(0)
            wc = df["bowler_id"].map(probes_df["train_balls_bowling"]).fillna(0)
            mc = np.minimum(bc.to_numpy(), wc.to_numpy())
            buckets = {}
            for lo, hi, label in [(0, 1, "unknown"), (1, 200, "1-199"),
                                  (200, 1000, "200-999"),
                                  (1000, 5000, "1000-4999"),
                                  (5000, 10**9, "5000+")]:
                sel = (mc >= lo) & (mc < hi)
                if sel.sum():
                    buckets[label] = {"n": int(sel.sum()),
                                      "ll": round(float(ll_vec[sel].mean()), 4)}
            metrics["validation_ll_by_min_train_balls"] = buckets
    print(json.dumps({k: v for k, v in metrics.items() if "ll" in str(k)},
                     indent=2), flush=True)

    torch.save(model.state_dict(), args.out / "model.pt")
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"saved to {args.out}/", flush=True)


if __name__ == "__main__":
    main()
