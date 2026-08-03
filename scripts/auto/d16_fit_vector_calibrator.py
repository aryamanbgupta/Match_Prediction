"""D16 — fit the fresh vector-scaling ball calibrator on the CONTROL arm.

WRITTEN AND COMMITTED BEFORE EITHER D16 MODEL EXISTED.

Copy-adapted from `scripts/fit_i5_ball_calibrator.py`, which is NOT edited by
D16. Two differences, both forced:

  1. that script hard-refuses any training contract whose
     `delivery_semantics` is not `legal_off_bat_v1`; the i7 frame is
     `inclusive_total_runs_v1`, so it cannot be used unmodified;
  2. it delegates encoding to `calibration._apply_encoders_to_df`, which
     silently FITS a fresh LabelEncoder when the saved encoder is missing.
     Here encoding is explicit (shared with `d16_marginal_audit.encode_frame`),
     so a missing encoder raises instead of inventing codes. The
     `_apply_encoders_to_df` result is still computed and cross-checked, which
     is the on-the-record proof that it handles the venue column.

Fits `calibration.VectorScalingCalibrator` on the CONTROL (balanced-weights)
arm's teacher-forced predictions over the i7 validation parquet, using
training-time encoder codes INCLUDING the i7 venue encoder — i.e. the served
distribution. The D16 twin design deliberately pairs
CONTROL + this fresh calibrator against NO-WEIGHTS raw; the legacy
`models/xgb_v3/vector_scaling_calibrator_v1.pkl` is meaningless on this frame
(fit on the legacy booster's venue_zero distribution) and is never used here.

Run:
    uv run python scripts/auto/d16_fit_vector_calibrator.py \
        --version i7 --data-dir data/xgb_data_i7 \
        --model-dir models/auto/d16/control \
        --out models/auto/d16/vector_scaling_calibrator_d16.pkl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "auto"))

from calibration import VectorScalingCalibrator, _apply_encoders_to_df  # noqa: E402
from d16_marginal_audit import encode_frame  # noqa: E402

TARGET_MAP = {-1: 5, 0: 0, 1: 1, 2: 2, 4: 3, 6: 4}
CLASS_NAMES = ["dot", "one", "two", "four", "six", "wicket"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="i7")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if (Path(args.version).name != args.version
            or args.version in {"", ".", ".."}):
        parser.error(f"unsafe version {args.version!r}")

    data_dir = args.data_dir or Path(f"data/xgb_data_{args.version}")
    model_dir = args.model_dir or Path(f"models/xgb_{args.version}")
    validation_path = (
        data_dir / f"cricket_data_{args.version}_validation.parquet")
    model_path = model_dir / f"xgboost_model_{args.version}.pkl"
    feature_path = model_dir / f"feature_columns_{args.version}.txt"
    contract_path = model_dir / f"training_contract_{args.version}.json"
    out_path = args.out or (
        model_dir / f"vector_scaling_calibrator_{args.version}.pkl")

    if not contract_path.is_file():
        raise FileNotFoundError(f"missing training contract {contract_path}")
    with contract_path.open() as handle:
        contract = json.load(handle)
    print(f"training contract: delivery_semantics="
          f"{contract.get('delivery_semantics')!r}  "
          f"data_version={contract.get('data_version')!r}  "
          f"venue_identity="
          f"{(contract.get('venue_identity') or {}).get('venue_alias_version')!r}")
    if contract.get("data_version") != args.version:
        raise RuntimeError(
            f"training contract data_version {contract.get('data_version')!r} "
            f"!= --version {args.version!r}")

    features = [
        line.strip() for line in feature_path.read_text().splitlines()
        if line.strip()
    ]
    frame_raw = pd.read_parquet(validation_path)
    print(f"validation parquet: {validation_path}  rows={len(frame_raw):,}  "
          f"features={len(features)}")

    # --- encoding path (explicit, training-time codes incl. venue) ---------
    print("encoding path: explicit training-time encoders from "
          f"{model_dir} (batter/bowler/matchup/venue)")
    frame = encode_frame(frame_raw, model_dir, args.version, features)
    venue_enc = joblib.load(model_dir / f"venue_encoder_{args.version}.pkl")
    print(f"  venue encoder ACTIVE ({len(venue_enc.classes_)} venues); "
          f"distinct venue codes in val: {frame['venue_encoded'].nunique()}")

    # --- cross-check calibration._apply_encoders_to_df --------------------
    check = frame_raw.copy()
    _apply_encoders_to_df(check, features, str(model_dir))
    enc_cols = [c for c in ("batter_encoded", "bowler_encoded",
                            "venue_encoded", "matchup_type_encoded")
                if c in features]
    for col in enc_cols:
        same = bool((check[col].to_numpy() == frame[col].to_numpy()).all())
        print(f"  cross-check _apply_encoders_to_df[{col}]: "
              f"{'identical' if same else 'DIFFERS'}")
        if not same:
            raise RuntimeError(
                f"encoding mismatch on {col} between explicit encoders and "
                f"calibration._apply_encoders_to_df")

    target = frame["ball_outcome"].map(TARGET_MAP)
    valid = target.notna()
    dropped = int((~valid).sum())
    if dropped:
        print(f"  dropped {dropped:,} rows with unmapped ball_outcome")
    frame = frame.loc[valid]
    labels = target.loc[valid].astype(int).to_numpy()

    model = joblib.load(model_path)
    raw = model.predict_proba(frame[features])
    calibrator = VectorScalingCalibrator().fit(raw, labels)
    calibrated = calibrator.calibrate_probs(raw)

    actual = np.bincount(labels, minlength=raw.shape[1]) / len(labels)
    raw_marg = raw.mean(axis=0)
    cal_marg = calibrated.mean(axis=0)
    print(f"\nvalidation balls: {len(labels):,}")
    print(f"raw log loss:        {log_loss(labels, raw):.6f}")
    print(f"calibrated log loss: {log_loss(labels, calibrated):.6f}")
    print(f"actual marginals:    {np.round(actual, 6)}")
    print(f"raw marginals:       {np.round(raw_marg, 6)}")
    print(f"calibrated marginal: {np.round(cal_marg, 6)}")
    print(f"weights (fitted 6-vector): {np.round(calibrator._v, 8)}")

    print("\nfit residuals (calibrated marginal - actual val frequency):")
    print(f"| {'class':<8}| {'actual':>10} | {'raw':>10} | {'calibrated':>11} | "
          f"{'resid':>11} |")
    for c in range(6):
        print(f"| {CLASS_NAMES[c]:<8}| {actual[c]:>10.6f} | "
              f"{raw_marg[c]:>10.6f} | {cal_marg[c]:>11.6f} | "
              f"{cal_marg[c] - actual[c]:>+11.2e} |")
    print(f"max |residual| = {np.max(np.abs(cal_marg - actual)):.3e}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrator, out_path)
    print(f"saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
