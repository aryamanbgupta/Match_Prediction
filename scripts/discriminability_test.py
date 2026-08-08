"""Identifiability test: can players be told apart from outcome sequences AT ALL?

Phase-1 found that learned vectors carry no style structure (probes at
chance). Two readings: (a) the embedding method failed, or (b) ball
outcomes simply do not identify players beyond their rates. This test
separates them.

Protocol (per role):
  * Players with >= 2*HALF balls in train. Balls split RANDOMLY into two
    halves (random split removes era/form drift; identity is the only
    shared factor).
  * Each half -> a small outcome signature: 6-class distribution overall
    and per phase (PP/mid/death), plus wide rate and boundary-vs-dot
    ratios. ~26 dims.
  * Pair feature = |sig_A - sig_B|.
  * Positives: (half1, half2) of the same player. Negatives: halves of two
    DIFFERENT players — both unmatched and TYPE-MATCHED (pace-pace or
    spin-spin for bowlers; same batting hand for batters).
  * Logistic regression, 5-fold CV, report AUC.

Reading: AUC >> 0.5 on type-matched negatives => outcome sequences do carry
individual style, and Phase-1's failure is a method problem. AUC ~ 0.5 =>
outcomes don't identify players beyond rate + type — an information bound
that makes the Phase-1 negative method-independent.

Usage: uv run python scripts/discriminability_test.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from embeddings_e1 import CLASS_MAPPING  # noqa: E402

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import cross_val_predict  # noqa: E402

KIT = Path("models/embeddings/eval_kit")
HALF = 600  # balls per half => players need >= 1200 balls
RNG = np.random.default_rng(42)


def signature(y: np.ndarray, phase: np.ndarray, is_wide: np.ndarray) -> np.ndarray:
    """26-dim outcome signature of a set of balls."""
    parts = [np.bincount(y, minlength=6) / len(y)]
    for ph in (0, 1, 2):
        sel = phase == ph
        parts.append(np.bincount(y[sel], minlength=6) / max(sel.sum(), 1))
    dots = (y == 0).mean()
    bounds = ((y == 3) | (y == 4)).mean()
    parts.append([is_wide.mean(), bounds / max(dots, 1e-3)])
    return np.concatenate(parts)


def run_role(df: pd.DataFrame, id_col: str, probes: pd.DataFrame,
             type_col: str, role: str) -> dict:
    counts = df[id_col].value_counts()
    players = counts[counts >= 2 * HALF].index.to_numpy()
    print(f"[{role}] {len(players)} players with >= {2*HALF} balls", flush=True)

    sigs = {}
    for pid in players:
        sub = df[df[id_col] == pid]
        idx = RNG.permutation(len(sub))[: 2 * HALF]
        a, b = idx[:HALF], idx[HALF:]
        y, ph, w = (sub["y"].to_numpy(), sub["phase"].to_numpy(),
                    sub["is_wide"].to_numpy())
        sigs[pid] = (signature(y[a], ph[a], w[a]), signature(y[b], ph[b], w[b]))

    ptype = probes[type_col].reindex(players)
    results = {}
    for mode in ("unmatched", "type_matched"):
        X, yy = [], []
        for pid in players:
            X.append(np.abs(sigs[pid][0] - sigs[pid][1]))
            yy.append(1)
        for pid in players:
            if mode == "type_matched":
                pool = players[(ptype.values == ptype[pid])
                               & (players != pid)]
            else:
                pool = players[players != pid]
            if len(pool) == 0:
                continue
            other = RNG.choice(pool)
            X.append(np.abs(sigs[pid][0] - sigs[other][1]))
            yy.append(0)
        X, yy = np.asarray(X), np.asarray(yy)
        scores = cross_val_predict(
            LogisticRegression(max_iter=2000), X, yy, cv=5,
            method="predict_proba")[:, 1]
        auc = roc_auc_score(yy, scores)
        results[mode] = round(float(auc), 4)
        print(f"  {mode}: AUC {auc:.4f} (n={len(yy)})", flush=True)
    return results


def main() -> None:
    print("loading train balls...", flush=True)
    df = pd.read_parquet(
        "data/xgb_data_v3/cricket_data_v3_train.parquet",
        columns=["batter_id", "bowler_id", "ball_outcome", "is_wide",
                 "is_middle_overs", "is_death_overs"])
    df["y"] = df["ball_outcome"].map(CLASS_MAPPING).astype(np.int64)
    df["phase"] = (df["is_middle_overs"].astype(int)
                   + 2 * df["is_death_overs"].astype(int))
    probes = pd.read_parquet(KIT / "probe_labels.parquet")

    out = {
        "bowler": run_role(df, "bowler_id", probes, "bowl_kind", "bowler"),
        "batter": run_role(df, "batter_id", probes, "bat_hand", "batter"),
        "half_balls": HALF,
    }
    Path("models/embeddings/discriminability.json").write_text(
        json.dumps(out, indent=2))
    print("\nsaved models/embeddings/discriminability.json", flush=True)


if __name__ == "__main__":
    main()
