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

from sim_eval.market_math import CostModel
from artifacts import artifact_path


def main() -> int:
    ap = argparse.ArgumentParser()
    # Defaults follow the production of record (post 2026-07-31 promotion);
    # the pre-2026-08-14 defaults scored the long-retired v2_frozen model
    # against a legacy-identity frame with no contract check at all.
    ap.add_argument("--role", default="match_model_prod",
                    help="Manifest role for the match model")
    ap.add_argument("--model-dir", type=Path, default=None)
    ap.add_argument("--frame-role", default="match_frame_i7_v2",
                    help="Manifest role for the match frame")
    ap.add_argument("--parquet", type=Path, default=None)
    ap.add_argument("--out-json", type=Path, default=None,
                    help="Default: <model-dir>/golden_predictions.json")
    ap.add_argument('--spread-bps', type=float, default=0.0)
    ap.add_argument('--fee-bps', type=float, default=0.0)
    ap.add_argument('--fee-basis', choices=('winnings', 'stake'),
                    default='winnings')
    args = ap.parse_args()
    args.model_dir = artifact_path(args.role, args.model_dir)
    frame_dir = artifact_path(args.frame_role)
    args.parquet = artifact_path(
        args.frame_role, args.parquet
    )
    if args.parquet == frame_dir:
        args.parquet = args.parquet / "golden_test.parquet"
    CostModel(args.spread_bps, args.fee_bps, args.fee_basis)
    if args.out_json is None:
        args.out_json = args.model_dir / "golden_predictions.json"

    if not args.parquet.exists():
        print(f"ERROR: parquet not found: {args.parquet}")
        return 1
    for f in ("model.pkl", "encoders.pkl", "feature_columns.txt"):
        if not (args.model_dir / f).exists():
            print(f"ERROR: missing {args.model_dir / f}")
            return 1

    # Venue-identity contract: scoring an i7 model against a legacy frame
    # (or vice versa) produces a plausible-looking golden LL on mismatched
    # state. Both artifacts stamp venue_identity.json when built under the
    # I7 contract; compare when either side declares one.
    model_identity_path = args.model_dir / "venue_identity.json"
    frame_identity_path = args.parquet.parent / "venue_identity.json"
    model_identity = (
        json.loads(model_identity_path.read_text())
        if model_identity_path.exists() else None
    )
    frame_identity = (
        json.loads(frame_identity_path.read_text())
        if frame_identity_path.exists() else None
    )
    if model_identity != frame_identity:
        print("ERROR: venue-identity contract mismatch between model and "
              f"frame:\n  model ({model_identity_path}): {model_identity}\n"
              f"  frame ({frame_identity_path}): {frame_identity}\n"
              "Score the model against a frame built under the same "
              "identity contract.")
        return 1
    if model_identity is None:
        print("  WARN: neither model nor frame declares a venue-identity "
              "contract (legacy pair) — proceeding unchecked.")

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

    # I15 identity contract: key by the unique Cricsheet ID, never the
    # synthetic display string — same-day doubleheaders share the synthetic
    # key and dict insertion would silently drop a match (last-write-wins).
    use_cricsheet = "cricsheet_id" in df.columns
    if not use_cricsheet and df["match_id"].duplicated().any():
        dupes = sorted(df["match_id"][df["match_id"].duplicated()].unique())
        print(f"ERROR: duplicate synthetic match_id values {dupes[:5]} and no "
              "cricsheet_id column to disambiguate them.")
        print("  Re-materialize the parquet with the I15 identity contract "
              "before predicting.")
        return 1

    predictions = {}
    for (_, row), p in zip(df.iterrows(), proba):
        key = str(row["cricsheet_id"]) if use_cricsheet else str(row["match_id"])
        if key in predictions:
            print(f"ERROR: duplicate primary match key {key!r} in {args.parquet}")
            return 1
        record = {
            "team1": row["team1"],
            "team2": row["team2"],
            "p_team1": float(p),
            "p_team2": float(1.0 - p),
            "team1_wins": int(row["team1_wins"]),
            "match_date": row["match_date"],
            "display_match_id": str(row["match_id"]),
        }
        if use_cricsheet:
            record["cricsheet_id"] = key
        predictions[key] = record

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
