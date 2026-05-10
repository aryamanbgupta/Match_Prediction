"""Post-hoc calibration for the match-level direct model.

M1 (2026-05-10) — sizing-only layer. Fits a calibrator (Platt by default
at val n=525; isotonic available for larger val sets) on val predictions
vs val truth via LOOCV, then applies the fitted calibrator to
test_predictions.json and golden_predictions.json, writing
*_calibrated.json adjacent to the originals.

Default is Platt rather than isotonic: empirically on val n=525,
isotonic LOOCV regresses test LL by ~+0.018 (overfitting noisy bins),
while Platt 2-param fits cleanly and gives test Δ ~−0.004. Switch to
isotonic via --method isotonic once val grows beyond ~1000.

The raw model output is left untouched. Downstream consumers
(blend_eval_json.py, reslice_eval_json.py) accept either the raw or
calibrated JSON via --direct-json. Calibration is for honest Kelly
sizing only; the headline LL gate stays raw — see IMPROVEMENTS.md
§ "Calibration vs. resolution" for why naive calibration anti-correlates
with flat ROI on this project.

Usage:
    uv run python scripts/calibrate_match_predictions.py \\
        --model-dir models/xgb_match_v3_baseline \\
        --data-dir data/xgb_match_data_v2_clean
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from calibration import IsotonicCalibrator, PlattCalibrator
from xgboost_match_v1 import _apply_encoders


_METHOD_REGISTRY = {
    "isotonic": IsotonicCalibrator,
    "platt": PlattCalibrator,
}


def _val_predictions(model_dir: Path, data_dir: Path):
    """Score val.parquet with the saved booster + encoders.

    Returns (probs, truth) ndarrays for use as the calibration training
    set. We score val fresh rather than persisting it at train time so
    this script works against any saved model directory that has the
    standard {model.pkl, encoders.pkl, feature_columns.txt} layout.
    """
    val = pd.read_parquet(data_dir / "validation.parquet")
    model = joblib.load(model_dir / "model.pkl")
    encoders = joblib.load(model_dir / "encoders.pkl")
    with open(model_dir / "feature_columns.txt") as f:
        feat_cols = [l.strip() for l in f if l.strip()]
    val = _apply_encoders(val, encoders)
    probs = model.predict_proba(val[feat_cols])[:, 1]
    truth = val["team1_wins"].values.astype(float)
    return probs, truth


def _calibrate_predictions_json(in_path: Path, out_path: Path,
                                 calibrator) -> tuple:
    """Apply calibrator to a predictions JSON in the synth format
    written by xgboost_match_v1.predict_test. Returns (raw_ll, cal_ll)
    on whatever truth is present in the JSON.
    """
    preds = json.load(open(in_path))
    mids = list(preds.keys())
    raw = np.array([preds[m]["p_team1"] for m in mids])
    truth = np.array([preds[m]["team1_wins"] for m in mids], dtype=float)
    cal = calibrator.predict(raw)

    out = {}
    for m, p_cal in zip(mids, cal):
        rec = dict(preds[m])
        rec["p_team1"] = float(p_cal)
        rec["p_team2"] = float(1.0 - p_cal)
        rec["p_team1_raw"] = float(preds[m]["p_team1"])
        out[m] = rec
    out_path.write_text(json.dumps(out, indent=2))

    raw_ll = log_loss(truth, raw, labels=[0, 1]) if len(set(truth)) > 1 else float("nan")
    cal_ll = log_loss(truth, cal, labels=[0, 1]) if len(set(truth)) > 1 else float("nan")
    return raw_ll, cal_ll, len(mids)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--predictions",
                    nargs="+",
                    default=["test_predictions.json", "golden_predictions.json"],
                    help="Predictions JSONs (relative to --model-dir) to "
                    "calibrate. Default: test + golden.")
    ap.add_argument("--method", choices=list(_METHOD_REGISTRY.keys()),
                    default="platt",
                    help="Calibration method. Platt is 2-parameter and stable "
                    "on small samples (val n=525 in 2026-05-10 baseline); "
                    "isotonic is non-parametric and needs ~1000+ samples to "
                    "avoid noise-driven LL regressions. Default: platt.")
    args = ap.parse_args()

    cls = _METHOD_REGISTRY[args.method]
    print(f"Fitting {args.method} calibrator on val from {args.data_dir}...")
    val_probs, val_truth = _val_predictions(args.model_dir, args.data_dir)
    print(f"  val n = {len(val_probs)}")
    print(f"  val raw LL    = {log_loss(val_truth, val_probs):.4f}")
    print(f"  val raw Brier = {brier_score_loss(val_truth, val_probs):.4f}")

    cal = cls()
    val_cal_loocv = cal.fit_loocv(val_probs, val_truth)
    print(f"  val LOOCV cal LL    = {log_loss(val_truth, val_cal_loocv):.4f}")
    print(f"  val LOOCV cal Brier = {brier_score_loss(val_truth, val_cal_loocv):.4f}")

    cal_path = args.model_dir / f"{args.method}_calibrator.{'pkl' if args.method == 'isotonic' else 'json'}"
    cal.save(str(cal_path))
    print(f"  fitted calibrator saved → {cal_path}")

    print("\nApplying calibrator to predictions JSONs:")
    for name in args.predictions:
        in_path = args.model_dir / name
        if not in_path.exists():
            print(f"  skip {name} (not found)")
            continue
        out_name = name.replace(".json", "_calibrated.json")
        out_path = args.model_dir / out_name
        raw_ll, cal_ll, n = _calibrate_predictions_json(in_path, out_path, cal)
        print(f"  {name}: n={n}  raw LL={raw_ll:.4f}  cal LL={cal_ll:.4f}  "
              f"Δ={cal_ll - raw_ll:+.4f}  → {out_name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
