"""A8 — fit a phase-conditional vector-scaling ball calibrator.

Follow-up to E5 (`reports/e5_class_weight_fix.md`). The landed global
`VectorScalingCalibrator` (`models/xgb_v3/vector_scaling_calibrator_v1.pkl`)
fits one 6-vector on all validation balls; it corrects marginal class rates
but under-corrects boundary-heavy contexts because the `balanced`-weight tilt
is not uniform across match phases. Here we fit THREE 6-vectors — powerplay /
middle / death — on the same validation balls, bucketed by over, plus a global
fallback vector (which must reproduce v1 exactly — the fitting pipeline is
validated by that identity).

Fits under the sim's input distribution (venue_encoded = 0), exactly as v1 was.
Phase buckets follow the sim: pp = balls<36, mid = 36<=balls<96, death>=96.

Output: models/auto/a8/phase_vector_calibrator.pkl
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from calibration import (PhaseVectorScalingCalibrator, VectorScalingCalibrator,  # noqa: E402
                         _apply_encoders_to_df)

VAL = REPO / "data/xgb_data_v3/cricket_data_v3_validation.parquet"
MODEL = REPO / "models/xgb_v3/xgboost_model_v3.pkl"
FEATS = REPO / "models/xgb_v3/feature_columns_v3.txt"
ENC_DIR = REPO / "models/xgb_v3"
V1 = REPO / "models/xgb_v3/vector_scaling_calibrator_v1.pkl"
OUT = REPO / "models/auto/a8/phase_vector_calibrator.pkl"

CLASS_NAMES = ["dot", "one", "two", "four", "six", "wicket"]


def phase_of(balls):
    # matches XGBoostModelV2.extract_features / EmpiricalBowlerSelector
    return np.where(balls < 36, "pp", np.where(balls < 96, "mid", "death"))


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
    phases = phase_of(df["balls_bowled"].values)

    cal = PhaseVectorScalingCalibrator()

    # global fallback (must == v1)
    g = cal.set_global(raw, y)
    v1 = joblib.load(V1)._v
    print(f"\nglobal (fallback) weights: {np.round(g, 6)}")
    print(f"v1 weights:                {np.round(v1, 6)}")
    print(f"  max abs diff vs v1: {float(np.max(np.abs(g - v1))):.2e}  "
          f"(0 => fitting pipeline validated)")

    print("\nPer-phase fits:")
    for ph in PhaseVectorScalingCalibrator.PHASES:
        sel = phases == ph
        n = int(sel.sum())
        actual = np.bincount(y[sel], minlength=6) / n
        v = cal.fit_phase(ph, raw[sel], y[sel])
        print(f"  [{ph:>5}] n={n:>7,}  actual_dist="
              f"{{{', '.join(f'{c}:{actual[i]:.3f}' for i, c in enumerate(CLASS_NAMES))}}}")
        print(f"          weights={np.round(v, 6)}")

    # Table: how much each phase vector diverges from the global one.
    print("\nPer-phase vector / global vector ratio (per class):")
    print("  " + "  ".join(f"{c:>7}" for c in CLASS_NAMES))
    for ph in PhaseVectorScalingCalibrator.PHASES:
        ratio = cal._v[ph] / g
        print(f"  {ph:>5} " + "  ".join(f"{r:7.3f}" for r in ratio))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(cal, OUT)
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
