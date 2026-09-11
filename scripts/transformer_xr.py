# manifest-exempt: embeddings-ladder experimental artifact namespace
"""Two-stage xR prototype: intent -> delivery -> reaction, on the T1 encoder.

T1.5's lesson: predicting shot/control from PRE-ball context alone collapses
to base rates — shot selection is a reaction to the delivery. This script
stages the conditioning properly and turns the aux heads into MEASUREMENT
(all aux heads read a DETACHED hidden state, so the main next-ball objective
is untouched — expect main LL ~= T1's 1.4372/1.4288):

  h_t = T1 encoder hidden state (pre-ball context + innings history)
  Stage 1 "intent"   : line(5), length(6)        <- h_t
  Stage 2 "reaction" : shot(24), control(3)      <- h_t + emb(line, length)
  Stage 2 "xR"       : outcome(6)                <- h_t + emb(line, length)

The xR head vs the main head on the SAME labeled balls measures what
delivery placement is worth per ball in LL — the stage-1/stage-2
information split of the expected-runs decomposition.

Labels: models/embeddings/deepcrease_labels/ (scripts/deepcrease_join.py).
Positions without labels are excluded from aux losses; teacher forcing uses
REALIZED line/length (known only post-delivery — never fed to the main head).

Usage: uv run python scripts/transformer_xr.py [--out models/embeddings/xr1]
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
from transformer_t1 import (BOS, T1Model, build_features, build_innings,  # noqa: E402
                            collate, load_aux, load_split)

KIT = Path("models/embeddings/eval_kit")


class XRModel(nn.Module):
    """T1 encoder + staged measurement heads on detached hidden states."""

    def __init__(self, n_feats: int, dmodel: int, layers: int, heads: int,
                 sizes: dict, demb: int = 16):
        super().__init__()
        self.core = T1Model(n_feats, dmodel, layers, heads)
        self.line_emb = nn.Embedding(sizes["line"] + 1, demb)
        self.len_emb = nn.Embedding(sizes["length"] + 1, demb)
        self.intent_line = nn.Linear(dmodel, sizes["line"])
        self.intent_length = nn.Linear(dmodel, sizes["length"])
        self.react_shot = nn.Linear(dmodel + 2 * demb, sizes["shot"])
        self.react_control = nn.Linear(dmodel + 2 * demb, sizes["control"])
        self.xr_outcome = nn.Linear(dmodel + 2 * demb, 6)

    def forward(self, feats, prev_y, pad_mask, line_idx, len_idx):
        L = feats.shape[1]
        pos = torch.arange(L, device=feats.device)
        x = (self.core.feat_proj(feats) + self.core.out_emb(prev_y)
             + self.core.pos_emb(pos))
        causal = torch.triu(
            torch.full((L, L), float("-inf"), device=feats.device), diagonal=1)
        h = self.core.encoder(x, mask=causal, src_key_padding_mask=pad_mask)
        main = self.core.head(h)
        hd = h.detach()
        deliv = torch.cat([hd, self.line_emb(line_idx),
                           self.len_emb(len_idx)], dim=-1)
        return main, {
            "line": self.intent_line(hd), "length": self.intent_length(hd),
            "shot": self.react_shot(deliv),
            "control": self.react_control(deliv),
            "xr": self.xr_outcome(deliv),
        }


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
    ap.add_argument("--out", type=Path, default=Path("models/embeddings/xr1"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device={device}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)

    print("loading splits + labels...", flush=True)
    train, val, test = load_split("train"), load_split("validation"), load_split("test")
    F_tr, F_va, F_te = (build_features(d) for d in (train, val, test))
    y_tr, y_va, y_te = (d["y"].to_numpy() for d in (train, val, test))
    inn_tr, inn_va, inn_te = (build_innings(d) for d in (train, val, test))
    aux_tr, vocabs = load_aux("train", len(train), None)
    aux_va, _ = load_aux("validation", len(val), vocabs)
    aux_te, _ = load_aux("test", len(test), vocabs)
    sizes = {t: len(v) for t, v in vocabs.items()}
    print(f"aux sizes: {sizes}", flush=True)

    model = XRModel(F_tr.shape[1], args.dmodel, args.layers, args.heads,
                    sizes).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    ce_none = nn.CrossEntropyLoss(reduction="none")

    def run_batch(chunk, feats, y, aux):
        # `collate` returns a Batch namedtuple since stage 2; its first five
        # fields are the stage 1 tuple verbatim.
        f, py, ty, pad, ax = collate(chunk, feats, y, device, aux)[:5]
        # teacher-forced delivery indices; -1 (unlabeled) -> UNK row
        li = ax["line"].clamp(min=-1)
        ni = ax["length"].clamp(min=-1)
        li_in = torch.where(li < 0, torch.full_like(li, sizes["line"]), li)
        ni_in = torch.where(ni < 0, torch.full_like(ni, sizes["length"]), ni)
        main, heads = model(f, py, pad, li_in, ni_in)
        return main, heads, ty, pad, ax

    def losses(main, heads, ty, pad, ax):
        raw = ce_none(main.reshape(-1, 6), ty.reshape(-1))
        keep = (~pad).reshape(-1).float()
        out = {"main": (raw * keep).sum() / keep.sum()}
        for t in ("line", "length"):  # intent: needs own label
            sel = (ax[t] >= 0).reshape(-1)
            if sel.any():
                out[t] = nn.functional.cross_entropy(
                    heads[t].reshape(-1, heads[t].shape[-1])[sel],
                    ax[t].reshape(-1)[sel])
        deliv_ok = ((ax["line"] >= 0) & (ax["length"] >= 0)).reshape(-1)
        for t in ("shot", "control"):  # reaction: needs delivery + own label
            sel = deliv_ok & (ax[t] >= 0).reshape(-1)
            if sel.any():
                out[t] = nn.functional.cross_entropy(
                    heads[t].reshape(-1, heads[t].shape[-1])[sel],
                    ax[t].reshape(-1)[sel])
        if deliv_ok.any():  # xR: outcome given delivery
            out["xr"] = nn.functional.cross_entropy(
                heads["xr"].reshape(-1, 6)[deliv_ok],
                ty.reshape(-1)[deliv_ok])
        return out

    def eval_split(feats, y, innings, aux, name):
        model.eval()
        agg = {}
        n_agg = {}
        main_ll_lab, xr_ll_lab, n_lab = 0.0, 0.0, 0
        probs = np.zeros((len(y), 6), dtype=np.float32)
        with torch.no_grad():
            for s in range(0, len(innings), args.batch):
                chunk = innings[s:s + args.batch]
                main, heads, ty, pad, ax = run_batch(chunk, feats, y, aux)
                p = torch.softmax(main, dim=-1).cpu().numpy()
                for b, ix in enumerate(chunk):
                    probs[ix] = p[b, :len(ix)]
                for t in ("line", "length", "shot", "control"):
                    need_deliv = t in ("shot", "control")
                    sel = (ax[t] >= 0).reshape(-1)
                    if need_deliv:
                        sel &= ((ax["line"] >= 0) & (ax["length"] >= 0)).reshape(-1)
                    if sel.any():
                        pred = heads[t].reshape(-1, heads[t].shape[-1]).argmax(-1)
                        agg[t] = agg.get(t, 0) + int(
                            (pred[sel] == ax[t].reshape(-1)[sel]).sum())
                        n_agg[t] = n_agg.get(t, 0) + int(sel.sum())
                deliv_ok = ((ax["line"] >= 0) & (ax["length"] >= 0)).reshape(-1)
                if deliv_ok.any():
                    tyf = ty.reshape(-1)[deliv_ok]
                    m_ll = ce_none(main.reshape(-1, 6)[deliv_ok], tyf)
                    x_ll = ce_none(heads["xr"].reshape(-1, 6)[deliv_ok], tyf)
                    main_ll_lab += float(m_ll.sum())
                    xr_ll_lab += float(x_ll.sum())
                    n_lab += int(deliv_ok.sum())
        ll = -np.log(np.clip(probs[np.arange(len(y)), y], 1e-15, 1.0))
        return {
            f"{name}_main_ll": round(float(ll.mean()), 4),
            f"{name}_aux_acc": {t: round(agg[t] / n_agg[t], 4) for t in agg},
            f"{name}_labeled_main_ll": round(main_ll_lab / max(n_lab, 1), 4),
            f"{name}_labeled_xr_ll": round(xr_ll_lab / max(n_lab, 1), 4),
            f"{name}_labeled_n": n_lab,
        }

    best_val, best_state, bad = np.inf, None, 0
    order = np.arange(len(inn_tr))
    for epoch in range(args.epochs):
        model.train()
        np.random.shuffle(order)
        t0 = time.time()
        for s in range(0, len(order), args.batch):
            chunk = [inn_tr[i] for i in order[s:s + args.batch]]
            main, heads, ty, pad, ax = run_batch(chunk, F_tr, y_tr, aux_tr)
            opt.zero_grad()
            ls = losses(main, heads, ty, pad, ax)
            total = sum(ls.values())
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        ev = eval_split(F_va, y_va, inn_va, aux_va, "validation")
        vll = ev["validation_main_ll"]
        print(f"epoch {epoch}: val_main={vll:.4f} "
              f"xr={ev['validation_labeled_xr_ll']:.4f} "
              f"vs main-on-labeled={ev['validation_labeled_main_ll']:.4f} "
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

    metrics = {"config": {k: (str(v) if isinstance(v, Path) else v)
                          for k, v in vars(args).items()},
               "aux_sizes": sizes}
    metrics.update(eval_split(F_va, y_va, inn_va, aux_va, "validation"))
    metrics.update(eval_split(F_te, y_te, inn_te, aux_te, "test"))
    print(json.dumps(metrics, indent=2, default=str), flush=True)
    torch.save(model.state_dict(), args.out / "model.pt")
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"saved to {args.out}/", flush=True)


if __name__ == "__main__":
    main()
