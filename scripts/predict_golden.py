"""Score the existing xgb_match_v2_frozen model against an arbitrary
match-level parquet (default: data/xgb_match_data_v2_golden/golden_test.parquet)
and write a predictions JSON in the same shape as test_predictions.json.

Standalone — does not modify xgboost_match_v1.py. Loads the saved
model.pkl + encoders.pkl + feature_columns.txt and applies them directly.

Usage:
    uv run python scripts/predict_golden.py
    uv run python scripts/predict_golden.py --parquet <path> --out-json <path>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path,
                    default=Path("models/xgb_match_v2_frozen"))
    ap.add_argument("--parquet", type=Path,
                    default=Path("data/xgb_match_data_v2_golden/golden_test.parquet"))
    ap.add_argument("--out-json", type=Path,
                    default=Path("models/xgb_match_v2_frozen/golden_predictions.json"))
    args = ap.parse_args()

    if not args.parquet.exists():
        print(f"ERROR: parquet not found: {args.parquet}")
        return 1
    for f in ("model.pkl", "encoders.pkl", "feature_columns.txt"):
        if not (args.model_dir / f).exists():
            print(f"ERROR: missing {args.model_dir / f}")
            return 1

    model = joblib.load(args.model_dir / "model.pkl")
    encoders = joblib.load(args.model_dir / "encoders.pkl")
    with open(args.model_dir / "feature_columns.txt") as f:
        feat_cols = [line.strip() for line in f if line.strip()]

    df = pd.read_parquet(args.parquet)
    print(f"  loaded {len(df)} rows from {args.parquet}")

    # Apply categorical encoders. Mirror xgboost_match_v1._apply_encoders.
    df = df.copy()
    unseen_warnings = {}
    for col, le in encoders.items():
        encoded_col = f"{col}_id_encoded" if col == "venue" else f"{col}_encoded"
        known = set(le.classes_)
        seen_in_df = set(df[col].astype(str))
        unseen = seen_in_df - known
        if unseen:
            unseen_warnings[col] = sorted(unseen)
            # Map unseen to the most common class (the encoder's first class
            # by alphabetical sort) — XGBoost will then make a reasonable
            # default prediction. Without this, transform() raises.
            fallback = le.classes_[0]
            df[col] = df[col].astype(str).apply(
                lambda v: v if v in known else fallback)
        df[encoded_col] = le.transform(df[col].astype(str))

    if unseen_warnings:
        print("\n  WARN: unseen categorical values mapped to fallback class:")
        for c, vals in unseen_warnings.items():
            print(f"    {c}: {len(vals)} unseen → {vals[:5]}{'...' if len(vals)>5 else ''}")

    proba = model.predict_proba(df[feat_cols])[:, 1]

    predictions = {}
    for (_, row), p in zip(df.iterrows(), proba):
        predictions[row["match_id"]] = {
            "team1": row["team1"],
            "team2": row["team2"],
            "p_team1": float(p),
            "p_team2": float(1.0 - p),
            "team1_wins": int(row["team1_wins"]),
            "match_date": row["match_date"],
        }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(predictions, f, indent=2)

    truth = df["team1_wins"].values
    standalone_ll = log_loss(truth, proba, labels=[0, 1])
    standalone_brier = brier_score_loss(truth, proba)
    print(f"\n  standalone metrics on {len(df)} matches "
          f"(no liquidity slice, no market join):")
    print(f"    LL    = {standalone_ll:.4f}")
    print(f"    Brier = {standalone_brier:.4f}")
    print(f"    coinflip ref = 0.6931")
    print(f"\n  predictions written → {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
