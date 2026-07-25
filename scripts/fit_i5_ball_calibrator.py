"""Fit the I5 global vector-scaling calibrator on validation balls only."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from calibration import VectorScalingCalibrator, _apply_encoders_to_df


TARGET_MAP = {-1: 5, 0: 0, 1: 1, 2: 2, 4: 3, 6: 4}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="i5")
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
    out_path = (
        args.out
        or model_dir / f"vector_scaling_calibrator_{args.version}.pkl"
    )

    if not contract_path.is_file():
        raise FileNotFoundError(f"missing training contract {contract_path}")
    import json
    with contract_path.open() as handle:
        contract = json.load(handle)
    if contract.get("delivery_semantics") != "legal_off_bat_v1":
        raise RuntimeError(
            "refusing to fit I5 calibrator for non-I5 training contract")

    features = [
        line.strip() for line in feature_path.read_text().splitlines()
        if line.strip()
    ]
    frame = pd.read_parquet(validation_path)
    _apply_encoders_to_df(frame, features, str(model_dir))
    target = frame["ball_outcome"].map(TARGET_MAP)
    valid = target.notna()
    frame = frame.loc[valid]
    labels = target.loc[valid].astype(int).to_numpy()

    model = joblib.load(model_path)
    raw = model.predict_proba(frame[features].to_numpy())
    calibrator = VectorScalingCalibrator().fit(raw, labels)
    calibrated = calibrator.calibrate_probs(raw)

    actual = np.bincount(labels, minlength=raw.shape[1]) / len(labels)
    print(f"validation balls: {len(labels):,}")
    print(f"raw log loss:        {log_loss(labels, raw):.6f}")
    print(f"calibrated log loss: {log_loss(labels, calibrated):.6f}")
    print(f"actual marginals:    {np.round(actual, 6)}")
    print(f"raw marginals:       {np.round(raw.mean(axis=0), 6)}")
    print(f"calibrated marginal: {np.round(calibrated.mean(axis=0), 6)}")
    print(f"weights:             {np.round(calibrator._v, 6)}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrator, out_path)
    print(f"saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
