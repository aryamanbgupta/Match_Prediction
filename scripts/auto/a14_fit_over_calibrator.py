"""A14 — fit a per-over vector-scaling ball calibrator.

Follow-up to E5 (`reports/e5_class_weight_fix.md`) and A8 (phase-conditional
vector scaling, which netted to null on multi-phase aggregate props). The
landed global `VectorScalingCalibrator`
(`models/xgb_v3/vector_scaling_calibrator_v1.pkl`) fits one 6-vector on all
validation balls; A8's 3-phase variant washed out because balls span phases.
Here we fit TWENTY 6-vectors — one per over (balls_bowled // 6, 0-19) — on the
same validation balls, plus a global fallback vector (which must reproduce v1
exactly — the fitting pipeline is validated by that identity). The hypothesis:
single-over props (`team_first_over_mae`, `highest_over_runs_*`) live inside one
over, so a per-over scaler has resolution A8's PP bucket (over 1 lumped with
2-6) lacked precisely where it can't wash out.

Fits under the sim's input distribution (venue_encoded = 0), exactly as v1 was.
Over index follows the sim ball counter: over = balls_bowled // 6, clamped 0-19.
Overs with fewer than MIN_BALLS validation balls fall back to the global vector.

Output: models/auto/a14/over_vector_calibrator.pkl
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from calibration import (OverVectorScalingCalibrator,  # noqa: E402
                         _apply_encoders_to_df)

VAL = REPO / "data/xgb_data_v3/cricket_data_v3_validation.parquet"
MODEL = REPO / "models/xgb_v3/xgboost_model_v3.pkl"
FEATS = REPO / "models/xgb_v3/feature_columns_v3.txt"
ENC_DIR = REPO / "models/xgb_v3"
V1 = REPO / "models/xgb_v3/vector_scaling_calibrator_v1.pkl"
OUT = REPO / "models/auto/a14/over_vector_calibrator.pkl"

CLASS_NAMES = ["dot", "one", "two", "four", "six", "wicket"]
MIN_BALLS = 1000  # overs sparser than this fall back to the global vector


def over_of(balls):
    return np.minimum((np.asarray(balls) // 6).astype(int), 19)


def main():
    feats = [l.strip() for l in open(FEATS)]
    model = joblib.load(MODEL)
    df = pd.read_parquet(VAL)

    _apply_encoders_to_df(df, feats, str(ENC_DIR))
    df["venue_encoded"] = 0  # sim input distribution (E5)

    tgt = df["ball_outcome"].replace({-1: 7}).map({0: 0, 1: 1, 2: 2, 4: 3, 6: 4, 7: 5})
    mask = tgt.notna()
    df = df[mask].copy()
    y = tgt[mask].astype(int).values
    X = df[feats].values
    print(f"Val balls (valid outcomes): {len(y):,}")

    raw = model.predict_proba(X)
    overs = over_of(df["balls_bowled"].values)

    cal = OverVectorScalingCalibrator()

    # global fallback (must == v1)
    g = cal.set_global(raw, y)
    v1 = joblib.load(V1)._v
    print(f"\nglobal (fallback) weights: {np.round(g, 6)}")
    print(f"v1 weights:                {np.round(v1, 6)}")
    print(f"  max abs diff vs v1: {float(np.max(np.abs(g - v1))):.2e}  "
          f"(0 => fitting pipeline validated)\n")

    print("Per-over fits (over: n_balls  -> weights; * = fell back to global):")
    n_fit, n_fallback = 0, 0
    for ov in range(OverVectorScalingCalibrator.N_OVERS):
        sel = overs == ov
        n = int(sel.sum())
        if n < MIN_BALLS:
            n_fallback += 1
            print(f"  over {ov:>2}: n={n:>7,}  * fallback to global (< {MIN_BALLS})")
            continue
        v = cal.fit_over(ov, raw[sel], y[sel])
        n_fit += 1
        print(f"  over {ov:>2}: n={n:>7,}  weights={np.round(v, 5)}")

    print(f"\nFit {n_fit} overs; {n_fallback} fell back to global.")

    # How much each over vector diverges from the global one (max class ratio).
    print("\nPer-over vector / global vector ratio (max |ratio-1| across classes):")
    for ov in range(OverVectorScalingCalibrator.N_OVERS):
        if ov in cal._v:
            ratio = cal._v[ov] / g
            print(f"  over {ov:>2}: max|ratio-1| = {float(np.max(np.abs(ratio - 1))):.3f}  "
                  f"ratios={np.round(ratio, 3)}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(cal, OUT)
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
