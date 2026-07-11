"""A4 — Regularized logistic regression on the match-level feature set.

A "different bias class" candidate for the winner-market model: a heavily
L2-regularized logistic regression on the same signed-diff numeric features
the XGBoost direct model uses. Emits an eval-compatible test_predictions.json
(recipe A) plus a 50/50 logit-average with the M7 production predictions.

Feature set: the 43 signed-difference numeric features from the shared parquet
(same source as the XGBoost trainer). The two label-encoded categoricals
(venue_id_encoded, competition_tier_encoded) are EXCLUDED: they are arbitrary
integer codes, and a linear model reads them as a meaningless monotone ordering
— invalid as linear terms. This is documented in research/reports/auto/A4.md.

Regularization strength C is selected on the validation split (lowest val log
loss); no test peeking. Features standardized with a train-fit StandardScaler.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent

METADATA_COLS = {
    "match_id", "cricsheet_id", "match_date",
    "team1", "team2", "venue", "competition_tier",
    "team1_wins",
}
# Label-encoded categoricals are arbitrary integer codes — invalid linear terms.
CATEGORICAL_STR = ["venue", "competition_tier"]


def _numeric_features(df: pd.DataFrame) -> list:
    return [c for c in df.columns
            if c not in METADATA_COLS and c not in CATEGORICAL_STR]


def _load(data_dir: Path, name: str) -> pd.DataFrame:
    return pd.read_parquet(data_dir / f"{name}.parquet")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/xgb_match_data_v2_clean")
    ap.add_argument("--model-dir", default="models/auto/a4")
    ap.add_argument("--m7-json",
                    default="models/xgb_match_v3_m7_production/test_predictions.json",
                    help="Production preds for the optional 50/50 logit-average.")
    ap.add_argument("--c-grid", default="0.003,0.01,0.03,0.1,0.3,1.0")
    ap.add_argument("--seed", type=int, default=29)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    train = _load(data_dir, "train")
    val = _load(data_dir, "validation")
    test = _load(data_dir, "test")

    feat_cols = _numeric_features(train)
    print(f"  features ({len(feat_cols)}): {feat_cols}")
    print(f"  train {len(train):,}  val {len(val):,}  test {len(test):,}")

    Xtr, ytr = train[feat_cols].values, train["team1_wins"].values
    Xva, yva = val[feat_cols].values, val["team1_wins"].values
    Xte, yte = test[feat_cols].values, test["team1_wins"].values

    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xva_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xva), scaler.transform(Xte)

    # --- Select C on validation log loss (no test peeking) ---
    c_grid = [float(x) for x in args.c_grid.split(",")]
    best_c, best_val_ll, best_model = None, np.inf, None
    print("\n  C-sweep (validation LL):")
    for C in c_grid:
        m = LogisticRegression(C=C, penalty="l2", solver="lbfgs",
                               max_iter=5000, random_state=args.seed)
        m.fit(Xtr_s, ytr)
        vll = log_loss(yva, m.predict_proba(Xva_s)[:, 1], labels=[0, 1])
        flag = ""
        if vll < best_val_ll:
            best_val_ll, best_c, best_model, flag = vll, C, m, "  <== best"
        print(f"    C={C:<7} val_LL={vll:.4f}{flag}")

    model = best_model
    print(f"\n  selected C={best_c}  (val LL={best_val_ll:.4f})")

    va_p = model.predict_proba(Xva_s)[:, 1]
    te_p = model.predict_proba(Xte_s)[:, 1]
    print(f"  val  LL={log_loss(yva, va_p, labels=[0,1]):.4f}  "
          f"Brier={brier_score_loss(yva, va_p):.4f}")
    print(f"  test LL={log_loss(yte, te_p, labels=[0,1]):.4f}  "
          f"Brier={brier_score_loss(yte, te_p):.4f}")

    # --- Write standalone logistic predictions (eval-compatible) ---
    def _write_preds(proba, path):
        preds = {}
        for (_, row), p in zip(test.iterrows(), proba):
            preds[row["match_id"]] = {
                "team1": row["team1"], "team2": row["team2"],
                "p_team1": float(p), "p_team2": float(1.0 - p),
                "team1_wins": int(row["team1_wins"]),
                "match_date": row["match_date"],
            }
        with open(path, "w") as f:
            json.dump(preds, f, indent=2)
        return preds

    logit_preds = _write_preds(te_p, model_dir / "test_predictions.json")
    print(f"\n  standalone logistic preds -> {model_dir/'test_predictions.json'} "
          f"({len(logit_preds)} matches)")

    # --- Optional: 50/50 logit-average with M7 production ---
    m7 = json.load(open(args.m7_json))
    blend = {}
    n_missing = 0
    for mid, lp in logit_preds.items():
        if mid not in m7:
            n_missing += 1
            p = lp["p_team1"]
        else:
            # average in logit space (per the idea: "logit-average")
            def _logit(x):
                x = min(max(x, 1e-6), 1 - 1e-6)
                return np.log(x / (1 - x))
            z = 0.5 * _logit(lp["p_team1"]) + 0.5 * _logit(m7[mid]["p_team1"])
            p = float(1.0 / (1.0 + np.exp(-z)))
        blend[mid] = {**lp, "p_team1": p, "p_team2": 1.0 - p}
    with open(model_dir / "blend5050_predictions.json", "w") as f:
        json.dump(blend, f, indent=2)
    bll = log_loss([blend[m]["team1_wins"] for m in blend],
                   [blend[m]["p_team1"] for m in blend], labels=[0, 1])
    print(f"  50/50 logit-avg blend -> blend5050_predictions.json  "
          f"(full-test LL={bll:.4f}, {n_missing} m7-missing)")

    joblib.dump({"model": model, "scaler": scaler, "feat_cols": feat_cols,
                 "C": best_c}, model_dir / "model.pkl")
    with open(model_dir / "train_metrics.json", "w") as f:
        json.dump({
            "selected_C": best_c,
            "val_log_loss": float(log_loss(yva, va_p, labels=[0, 1])),
            "test_log_loss_fulltest": float(log_loss(yte, te_p, labels=[0, 1])),
            "blend5050_log_loss_fulltest": float(bll),
            "n_features": len(feat_cols),
        }, f, indent=2)
    print(f"  saved -> {model_dir}")


if __name__ == "__main__":
    main()
