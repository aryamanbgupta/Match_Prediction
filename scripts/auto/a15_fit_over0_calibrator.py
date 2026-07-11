"""A15 — fit a minimal OVER-0-ONLY vector-scaling ball calibrator.

Parsimony follow-up to A14 (`scripts/auto/a14_fit_over_calibrator.py`, LANDED),
which fit TWENTY per-over 6-vectors + a global fallback and improved
`team_first_over_mae` (dMAE -0.022, CI excludes 0). A14's own diagnostic showed
the entire first-over win is concentrated in **over 0** (its vector diverges
most from global: six/wicket x1.23), while the other 19 per-over vectors buy
nothing observable and are pure overfitting surface.

This fits ONLY over 0 plus the global fallback (== v1). Overs 1-19 have no entry
in `_v`, so `OverVectorScalingCalibrator._vector_for` falls back to `_global`
for them -> the sim behaves EXACTLY like the E5 v1 single-vector calibrator
everywhere except over 0, where it uses the over-0 vector. That over-0 vector is
byte-identical to A14's (same `_fit_scaling_vector` on the same val balls), so
this cleanly isolates "does over 0 alone carry the full team_first_over_mae win?"

2-vector calibrator (over-0 + global) vs A14's 20-vector. Same fitting pipeline
validated by the global == v1 identity check.

Output: models/auto/a15/over0_calibrator.pkl
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
A14 = REPO / "models/auto/a14/over_vector_calibrator.pkl"
OUT = REPO / "models/auto/a15/over0_calibrator.pkl"

CLASS_NAMES = ["dot", "one", "two", "four", "six", "wicket"]
MIN_BALLS = 1000  # same floor A14 used (over 0 has ~26k val balls, well above)


def over_of(balls):
    return np.minimum((np.asarray(balls) // 6).astype(int), 19)


def main():
    feats = [l.strip() for l in open(FEATS)]
    model = joblib.load(MODEL)
    df = pd.read_parquet(VAL)

    _apply_encoders_to_df(df, feats, str(ENC_DIR))
    df["venue_encoded"] = 0  # sim input distribution (E5), identical to A14/v1

    tgt = df["ball_outcome"].replace({-1: 7}).map({0: 0, 1: 1, 2: 2, 4: 3, 6: 4, 7: 5})
    mask = tgt.notna()
    df = df[mask].copy()
    y = tgt[mask].astype(int).values
    X = df[feats].values
    print(f"Val balls (valid outcomes): {len(y):,}")

    raw = model.predict_proba(X)
    overs = over_of(df["balls_bowled"].values)

    cal = OverVectorScalingCalibrator()

    # global fallback (must == v1) -- used for EVERY over except 0.
    g = cal.set_global(raw, y)
    v1 = joblib.load(V1)._v
    print(f"\nglobal (fallback, used for overs 1-19) weights: {np.round(g, 6)}")
    print(f"v1 weights:                                     {np.round(v1, 6)}")
    print(f"  max abs diff vs v1: {float(np.max(np.abs(g - v1))):.2e}  "
          f"(0 => fitting pipeline validated)\n")

    # over-0 vector only.
    sel = overs == 0
    n0 = int(sel.sum())
    assert n0 >= MIN_BALLS, f"over 0 has only {n0} balls (< {MIN_BALLS})"
    v0 = cal.fit_over(0, raw[sel], y[sel])
    print(f"over 0: n={n0:,}  weights={np.round(v0, 5)}")
    print(f"over-0 / global ratio: {np.round(v0 / g, 4)}  "
          f"(max|ratio-1| = {float(np.max(np.abs(v0 / g - 1))):.3f})")

    assert sorted(cal._v.keys()) == [0], f"expected only over 0 in _v, got {sorted(cal._v.keys())}"

    # Identity check vs A14: this over-0 vector must equal A14's over-0 vector
    # AND this global must equal A14's global (same fit on same val balls).
    if A14.exists():
        a14 = joblib.load(A14)
        print(f"\nvs A14 20-vector calibrator:")
        print(f"  max|over0 - A14.over0| = {float(np.max(np.abs(v0 - a14._v[0]))):.2e}")
        print(f"  max|global - A14.global| = {float(np.max(np.abs(g - a14._global))):.2e}")
        print(f"  (both 0 => over-0 correction is byte-identical to A14; A15 only "
              f"drops overs 1-19 -> global)")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(cal, OUT)
    print(f"\nSaved -> {OUT}")
    print(f"Effective distinct vectors: 2 (over-0 + global) vs A14's 20.")


if __name__ == "__main__":
    main()
