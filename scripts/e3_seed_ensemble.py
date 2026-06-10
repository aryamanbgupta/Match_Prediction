"""E3 — Seed-ensemble for the match-level direct model.

Hypothesis: a single XGBoost at depth 4 / lr 0.05 / subsample 0.8 /
colsample 0.9 on 7,912 rows carries nontrivial seed variance; averaging
K seeds is pure variance reduction (no new capacity, no leakage surface)
and typically buys 0.003-0.008 LL. Never tried on this project.

Protocol:
  - Exact M7 production config (hyperparams from reports/m7_architecture_eval.md),
    production encoders.pkl + feature_columns.txt reused verbatim.
  - Seed 29 must reproduce models/xgb_match_v3_m7_production train_metrics
    (val LL 0.6459 / test LL 0.5924) before the ensemble is trusted.
  - K=10 seeds; combine by prob-mean and logit-mean; **choose combiner on
    val LL only**, then read out on the iteration sliced eval.
  - Keep iff iteration >=$50k LL improves AND ROI CI doesn't materially
    regress vs raw single-model baseline (0.6299 / +21.90 [+2.28, +43.83]).

Usage:
    uv run python scripts/e3_seed_ensemble.py
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss
from xgboost import XGBClassifier

from e1_temperature_sharpen import (_logit, _sigmoid, collect_summary,
                                    run_blend_and_reslice)
from xgboost_match_v1 import _apply_encoders, _load_split

REPO = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO / "models/xgb_match_v3_m7_production"
DATA_DIR = REPO / "data/xgb_match_data_v3_m3_unfrozen"
OUT_ROOT = REPO / "eval_out_e3"
SEEDS = [29, 7, 42, 101, 271, 314, 555, 1337, 2026, 90210]

M7_PARAMS = dict(
    objective="binary:logistic", eval_metric="logloss",
    n_estimators=1000, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.9,
    reg_alpha=0.1, reg_lambda=1.0, early_stopping_rounds=30,
)


def main():
    train = _load_split(DATA_DIR, "train")
    val = _load_split(DATA_DIR, "validation")
    test = _load_split(DATA_DIR, "test")
    encoders = joblib.load(MODEL_DIR / "encoders.pkl")
    feat_cols = [l.strip() for l in open(MODEL_DIR / "feature_columns.txt")
                 if l.strip()]
    train_e = _apply_encoders(train, encoders)
    val_e = _apply_encoders(val, encoders)
    test_e = _apply_encoders(test, encoders)
    X_tr, y_tr = train_e[feat_cols], train_e["team1_wins"]
    X_va, y_va = val_e[feat_cols], val_e["team1_wins"]
    X_te, y_te = test_e[feat_cols], test_e["team1_wins"]
    print(f"train {len(X_tr)}  val {len(X_va)}  test {len(X_te)}  "
          f"features {len(feat_cols)}")

    val_probs, test_probs = [], []
    for seed in SEEDS:
        m = XGBClassifier(random_state=seed, **M7_PARAMS)
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        vp = m.predict_proba(X_va)[:, 1]
        tp = m.predict_proba(X_te)[:, 1]
        val_probs.append(vp)
        test_probs.append(tp)
        print(f"seed {seed:>6d}  best_iter {m.best_iteration:>4d}  "
              f"val LL {log_loss(y_va, vp):.4f}  test LL {log_loss(y_te, tp):.4f}")

    ref = json.load(open(MODEL_DIR / "train_metrics.json"))
    print(f"\nseed-29 reproduction check: val LL {log_loss(y_va, val_probs[0]):.4f} "
          f"(prod {ref['val_log_loss']:.4f})  test LL "
          f"{log_loss(y_te, test_probs[0]):.4f} (prod {ref['test_log_loss']:.4f})")

    V, T = np.array(val_probs), np.array(test_probs)
    combos = {
        "prob_mean": (V.mean(0), T.mean(0)),
        "logit_mean": (_sigmoid(_logit(V).mean(0)), _sigmoid(_logit(T).mean(0))),
    }
    print("\n-- combiner selection on val ONLY --")
    best_name, best_vll = None, np.inf
    for name, (vp, _) in combos.items():
        vll = log_loss(y_va, vp)
        vbr = brier_score_loss(y_va, vp)
        print(f"{name:11s} val LL {vll:.4f}  Brier {vbr:.4f}")
        if vll < best_vll:
            best_name, best_vll = name, vll
    single_vll = log_loss(y_va, V[0])
    print(f"single(29)  val LL {single_vll:.4f}")
    print(f"chosen: {best_name} (Δ val LL {best_vll - single_vll:+.4f} vs single)")

    # write ensemble test predictions in the standard JSON schema
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    ens_test = combos[best_name][1]
    preds = {}
    for (_, row), p in zip(test.iterrows(), ens_test):
        preds[row["match_id"]] = {
            "team1": row["team1"], "team2": row["team2"],
            "p_team1": float(p), "p_team2": float(1.0 - p),
            "team1_wins": int(row["team1_wins"]),
            "match_date": row["match_date"],
        }
    pred_path = OUT_ROOT / "preds" / f"ensemble_{best_name}_predictions.json"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(preds, open(pred_path, "w"), indent=1)

    print("\n-- iteration sliced eval (raw single-model baseline vs ensemble) --")
    summaries = {"raw_single": collect_summary(
        run_blend_and_reslice(MODEL_DIR / "test_predictions.json",
                              "raw_single", OUT_ROOT))}
    summaries[f"ens_{best_name}"] = collect_summary(
        run_blend_and_reslice(pred_path, f"ens_{best_name}", OUT_ROOT))

    hdr = f"{'variant':16s} {'slice':>8s} {'n':>4s} {'LL':>8s} {'ROI%':>8s} {'ROI CI':>20s}"
    print(hdr)
    print("-" * len(hdr))
    for name, rows in summaries.items():
        for vol in sorted(rows):
            r = rows[vol]
            tag = {0: "all", 50000: ">=50k", 100000: ">=100k"}[vol]
            ci = (f"[{r['roi_ci'][0]:+.2f}, {r['roi_ci'][1]:+.2f}]"
                  if r['roi_ci'][0] is not None else "—")
            print(f"{name:16s} {tag:>8s} {r['n']:>4d} {r['ll']:>8.4f} "
                  f"{r['roi']:>+8.2f} {ci:>20s}")

    json.dump(summaries, open(OUT_ROOT / "e3_summaries.json", "w"), indent=2)
    print(f"\nsummaries -> {OUT_ROOT}/e3_summaries.json")


if __name__ == "__main__":
    main()
