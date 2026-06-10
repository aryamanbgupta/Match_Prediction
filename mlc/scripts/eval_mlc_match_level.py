"""Score the production match-level model (xgb_match_v3_m7_production) on the
2025-season MLC matches.

MLC is NOT out-of-sample the way the Blast golden pool is: all 33 MLC 2025
matches are in data/t20s_json and were materialized into the standard
splits. June matches (<= val_end 2025-06-30) are in *validation* (seen at
training/early-stop time); July matches are in the held-out *test* split.
So we report three slices:
  * test (n=12)  -- the clean held-out subset; the honest headline
  * validation (n=21) -- in-sample, reported for completeness only
  * all (n=33)

Features come from the unfrozen materialization (data/xgb_match_data_v3_m3
_unfrozen) which carries the production feature set; encoders are applied
exactly as predict_golden.py does.

Usage:
    uv run python scripts/eval_mlc_match_level.py
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

REPO = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO / "models" / "xgb_match_v3_m7_production"
DATA_DIR = REPO / "data" / "xgb_match_data_v3_m3_unfrozen"
OUT_JSON = REPO / "mlc" / "data" / "mlc_2025_predictions.json"

MLC = {"San Francisco Unicorns", "Washington Freedom", "Texas Super Kings",
       "MI New York", "Los Angeles Knight Riders", "Seattle Orcas"}


def apply_encoders(df: pd.DataFrame, encoders) -> pd.DataFrame:
    df = df.copy()
    for col, le in encoders.items():
        encoded_col = f"{col}_id_encoded" if col == "venue" else f"{col}_encoded"
        known = set(le.classes_)
        df[col] = df[col].astype(str).apply(lambda v: v if v in known else le.classes_[0])
        df[encoded_col] = le.transform(df[col].astype(str))
    return df


def metrics(truth, p):
    truth = np.asarray(truth)
    p = np.asarray(p)
    ll = log_loss(truth, p, labels=[0, 1])
    brier = brier_score_loss(truth, p)
    pred_team1 = p >= 0.5
    acc = float(np.mean(pred_team1 == (truth == 1)))
    return ll, brier, acc


def main() -> int:
    model = joblib.load(MODEL_DIR / "model.pkl")
    encoders = joblib.load(MODEL_DIR / "encoders.pkl")
    feat_cols = [l.strip() for l in open(MODEL_DIR / "feature_columns.txt") if l.strip()]

    frames = []
    for split in ("validation", "test"):
        df = pd.read_parquet(DATA_DIR / f"{split}.parquet")
        m = df[df["team1"].isin(MLC) & df["team2"].isin(MLC)].copy()
        m["split"] = split
        frames.append(m)
    mlc = pd.concat(frames, ignore_index=True).sort_values("match_date")

    enc = apply_encoders(mlc, encoders)
    p_team1 = model.predict_proba(enc[feat_cols])[:, 1]
    mlc = mlc.assign(p_team1=p_team1)

    # Per-match records + predictions JSON.
    preds = {}
    rows = []
    for (_, r), p in zip(mlc.iterrows(), p_team1):
        winner = r["team1"] if r["team1_wins"] == 1 else r["team2"]
        pick = r["team1"] if p >= 0.5 else r["team2"]
        rows.append({
            "date": r["match_date"], "split": r["split"],
            "team1": r["team1"], "team2": r["team2"],
            "p_team1": float(p), "team1_wins": int(r["team1_wins"]),
            "winner": winner, "pick": pick, "correct": bool(pick == winner),
            "conf": float(max(p, 1 - p)),
        })
        preds[r["match_id"]] = {
            "team1": r["team1"], "team2": r["team2"], "p_team1": float(p),
            "p_team2": float(1 - p), "team1_wins": int(r["team1_wins"]),
            "match_date": r["match_date"], "split": r["split"],
        }
    OUT_JSON.write_text(json.dumps(preds, indent=2))

    # Slice metrics.
    print(f"\n{'slice':12} {'n':>3} {'LL':>8} {'Brier':>8} {'acc':>9}   coinflip LL=0.6931\n" + "-" * 55)
    for name, sub in (("test", mlc[mlc.split == "test"]),
                      ("validation", mlc[mlc.split == "validation"]),
                      ("all", mlc)):
        ll, brier, acc = metrics(sub["team1_wins"], sub["p_team1"])
        ncorrect = int(((sub["p_team1"] >= 0.5) == (sub["team1_wins"] == 1)).sum())
        tag = " (in-sample)" if name == "validation" else (" (HELD-OUT)" if name == "test" else "")
        print(f"{name:12} {len(sub):>3} {ll:>8.4f} {brier:>8.4f} {ncorrect:>2}/{len(sub)}={acc*100:4.1f}%{tag}")

    # Per-match table.
    print(f"\n{'date':11} {'split':4} {'match':46} {'P(t1)':>6} {'pick':>26} {'winner':>26} {'ok'}")
    for r in rows:
        sp = "TEST" if r["split"] == "test" else "val"
        m = f"{r['team1']} v {r['team2']}"
        print(f"{r['date']:11} {sp:4} {m:46.46} {r['p_team1']*100:5.1f}% "
              f"{r['pick']:>26.26} {r['winner']:>26.26} {'OK' if r['correct'] else 'X'}")

    print(f"\npredictions → {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
