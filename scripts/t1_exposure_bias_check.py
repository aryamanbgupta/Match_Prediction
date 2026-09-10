# manifest-exempt: embeddings-ladder diagnostic artifact namespace
"""Diagnostic: does T1 degrade under self-conditioned (free-running)
decoding vs. teacher-forced (ground-truth-history) decoding?

Motivation: T1's per-ball metrics (1.4372 val / 1.4288 test) were computed
with teacher forcing — the previous-ball OUTCOME TOKEN fed to the model is
always the REAL observed outcome (see transformer_t1.collate: `py[b,1:n] =
y[ix][:-1]`). But scripts/sim_t1.py, used for Monte Carlo rollouts, feeds
back the model's OWN SAMPLED outcome at each step, because there is no
ground truth during simulation. This is the classic exposure-bias /
train-test mismatch for autoregressive sequence models: a model can score
well under teacher forcing yet drift once it must condition on its own
(imperfect) samples, since sampled trajectories can wander into token
histories the model never saw in training.

This script isolates that ONE mechanism (the outcome-token history path)
on real held-out innings, holding pre-ball state/EB features fixed to
their true values (i.e. NOT re-simulating the whole match — a narrower,
cheaper test than reproducing the full engine). For a sample of validation
innings:
  (a) teacher-forced log-loss (sanity check against T1's reported 1.4372)
  (b) self-conditioned log-loss: at each ball, sample prev outcome from
      the model's own predicted distribution (single stochastic rollout,
      matching the sim's random.choices), and log-loss the true next-ball
      label under THAT rollout's history.
Both averaged over the same balls for a fair paired comparison. Also
reports how the self-conditioned gap grows with ball index within the
innings (early vs late in cricket sequences) — exposure bias predicts a
growing gap as sampled errors compound.

Usage: uv run python scripts/t1_exposure_bias_check.py [--n-innings 300]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from transformer_t1 import BOS, T1Model, build_features, build_innings, load_split  # noqa: E402

KIT = Path("models/embeddings/eval_kit")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-innings", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model-dir", type=Path, default=Path("models/embeddings/t1"))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    print("loading validation split + T1 checkpoint...", flush=True)
    val = load_split("validation")
    F_va = build_features(val)
    y_va = val["y"].to_numpy()
    innings = build_innings(val)
    cfg = json.loads((args.model_dir / "metrics.json").read_text())["config"]
    model = T1Model(F_va.shape[1], cfg["dmodel"], cfg["layers"], cfg["heads"])
    model.load_state_dict(torch.load(args.model_dir / "model.pt", map_location="cpu"))
    model.eval()

    sample = rng.choice(len(innings), size=min(args.n_innings, len(innings)),
                        replace=False)
    tf_lls, sc_lls = [], []
    by_pos_tf, by_pos_sc = {}, {}

    with torch.no_grad():
        for idx in sample:
            ix = innings[idx]
            L = len(ix)
            feats = torch.tensor(F_va[ix]).unsqueeze(0)  # (1, L, F)
            true_y = y_va[ix]

            # (a) teacher-forced: single batched forward, real history
            prev_tf = np.concatenate([[BOS], true_y[:-1]]).astype(np.int64)
            pad = torch.zeros((1, L), dtype=torch.bool)
            out_tf = model(feats, torch.tensor(prev_tf).unsqueeze(0), pad)
            logits_tf = out_tf[0] if isinstance(out_tf, tuple) else out_tf
            logp_tf = torch.log_softmax(logits_tf[0], dim=-1).numpy()
            ll_tf = -logp_tf[np.arange(L), true_y]

            # (b) self-conditioned: step-by-step, feed the model's OWN
            # sampled previous outcome (matches sim_v1_2's random.choices
            # sampling), state/EB features held at their TRUE values.
            prev_sc = np.full(L, BOS, dtype=np.int64)
            ll_sc = np.zeros(L)
            cur_prev = BOS
            for t in range(L):
                prev_sc[t] = cur_prev
                ft = feats[:, :t + 1, :]
                pt = torch.tensor(prev_sc[:t + 1]).unsqueeze(0)
                pd_ = torch.zeros((1, t + 1), dtype=torch.bool)
                out = model(ft, pt, pd_)
                logits = out[0] if isinstance(out, tuple) else out
                logp = torch.log_softmax(logits[0, -1], dim=-1).numpy()
                ll_sc[t] = -logp[true_y[t]]
                probs = np.exp(logp)
                probs = probs / probs.sum()
                cur_prev = int(rng.choice(6, p=probs))  # model's own sample

            tf_lls.append(ll_tf)
            sc_lls.append(ll_sc)
            for t in range(L):
                by_pos_tf.setdefault(t, []).append(ll_tf[t])
                by_pos_sc.setdefault(t, []).append(ll_sc[t])

    tf_all = np.concatenate(tf_lls)
    sc_all = np.concatenate(sc_lls)
    print(f"\nn_innings={len(sample)}, n_balls={len(tf_all)}")
    print(f"teacher-forced   mean LL: {tf_all.mean():.4f}  "
          f"(T1 reported val: {1.4372})")
    print(f"self-conditioned mean LL: {sc_all.mean():.4f}")
    print(f"gap (self-conditioned - teacher-forced): "
          f"{sc_all.mean() - tf_all.mean():+.4f}")

    print("\nmean LL by ball position within innings (bucketed):")
    edges = [0, 10, 30, 60, 90, 120]
    for lo, hi in zip(edges[:-1], edges[1:]):
        tf_b = np.concatenate([by_pos_tf[t] for t in range(lo, hi) if t in by_pos_tf])
        sc_b = np.concatenate([by_pos_sc[t] for t in range(lo, hi) if t in by_pos_sc])
        if len(tf_b):
            print(f"  balls [{lo:3d},{hi:3d}): "
                  f"teacher-forced {tf_b.mean():.4f}  "
                  f"self-conditioned {sc_b.mean():.4f}  "
                  f"gap {sc_b.mean()-tf_b.mean():+.4f}  (n={len(tf_b)})")

    out = {
        "n_innings": int(len(sample)), "n_balls": int(len(tf_all)),
        "teacher_forced_ll": round(float(tf_all.mean()), 4),
        "self_conditioned_ll": round(float(sc_all.mean()), 4),
        "gap": round(float(sc_all.mean() - tf_all.mean()), 4),
    }
    Path("models/embeddings/t1_exposure_bias_check.json").write_text(
        json.dumps(out, indent=2))
    print(f"\nsaved models/embeddings/t1_exposure_bias_check.json")


if __name__ == "__main__":
    main()
