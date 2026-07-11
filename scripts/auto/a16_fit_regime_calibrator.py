"""A16 — fit a SPARSE regime-change-over vector-scaling ball calibrator.

Extension of A15 (`scripts/auto/a15_fit_over0_calibrator.py`, LANDED), which fit
ONLY over 0 + the global fallback and captured the full `team_first_over_mae`
win with 2 effective vectors (vs A14's 20). A15's finding: the sim's single
largest per-over calibration defect is **over 0** (fixing it alone earns the
first-over MAE win AND `pp_total_ou_45_5` -0.0037), while A14's blanket 19 other
per-over vectors were inert/overfit.

A16 tests whether the OTHER "regime-change" overs carry a similar,
mechanically-distinct miscalibration a sparse calibrator could fix:
  - over 0  : first ball a fresh batter + new bowler face (A15's win)
  - over 6  : start of the middle overs (field restriction lifts, bowling change)
  - over 15 : start of the death (specialist death bowlers enter)
The ~17 non-boundary overs stay on the global vector (A15 discipline: add a
vector only where it's mechanically distinct and observable, not a blanket grid).

Fits vectors for overs {0, 6, 15} only; all other overs fall back to `_global`
(== E5 v1). Over-0 vector and global are byte-identical to A15/A14 (same
`_fit_scaling_vector` on the same val balls), so A16 differs from A15 ONLY by
ALSO carrying over-6 and over-15 vectors — cleanly isolating "do overs 6 and 15
add anything observable beyond over 0?".

Output: models/auto/a16/regime_calibrator.pkl
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
A15 = REPO / "models/auto/a15/over0_calibrator.pkl"
OUT = REPO / "models/auto/a16/regime_calibrator.pkl"

CLASS_NAMES = ["dot", "one", "two", "four", "six", "wicket"]
REGIME_OVERS = [0, 6, 15]
MIN_BALLS = 1000  # same floor A14/A15 used


def over_of(balls):
    return np.minimum((np.asarray(balls) // 6).astype(int), 19)


def main():
    feats = [l.strip() for l in open(FEATS)]
    model = joblib.load(MODEL)
    df = pd.read_parquet(VAL)

    _apply_encoders_to_df(df, feats, str(ENC_DIR))
    df["venue_encoded"] = 0  # sim input distribution (E5), identical to A14/A15/v1

    tgt = df["ball_outcome"].replace({-1: 7}).map({0: 0, 1: 1, 2: 2, 4: 3, 6: 4, 7: 5})
    mask = tgt.notna()
    df = df[mask].copy()
    y = tgt[mask].astype(int).values
    X = df[feats].values
    print(f"Val balls (valid outcomes): {len(y):,}")

    raw = model.predict_proba(X)
    overs = over_of(df["balls_bowled"].values)

    cal = OverVectorScalingCalibrator()

    # global fallback (must == v1) -- used for EVERY over except {0,6,15}.
    g = cal.set_global(raw, y)
    v1 = joblib.load(V1)._v
    print(f"\nglobal (fallback, used for all non-regime overs): {np.round(g, 6)}")
    print(f"v1 weights:                                        {np.round(v1, 6)}")
    print(f"  max abs diff vs v1: {float(np.max(np.abs(g - v1))):.2e}  "
          f"(0 => fitting pipeline validated)\n")

    # regime-change over vectors {0, 6, 15}.
    print("=== PRE-RUN per-over divergences (A16 method: report BEFORE sim) ===")
    print("threshold context: A8/A12 washed out below ~0.05 max|ratio-1|\n")
    for ov in REGIME_OVERS:
        sel = overs == ov
        n = int(sel.sum())
        assert n >= MIN_BALLS, f"over {ov} has only {n} balls (< {MIN_BALLS})"
        v = cal.fit_over(ov, raw[sel], y[sel])
        ratio = v / g
        j = int(np.argmax(np.abs(ratio - 1)))
        print(f"over {ov:2d}: n={n:6,}  weights={np.round(v, 5)}")
        print(f"         ratio/global={np.round(ratio, 4)}")
        print(f"         max|ratio-1|={float(np.max(np.abs(ratio - 1))):.3f}  "
              f"biggest: {CLASS_NAMES[j]} x{ratio[j]:.3f}\n")

    assert sorted(cal._v.keys()) == REGIME_OVERS, \
        f"expected {REGIME_OVERS} in _v, got {sorted(cal._v.keys())}"

    # Identity check vs A15/A14: over-0 vector + global must be byte-identical
    # (same fit on same val balls). This proves A16 differs from A15 ONLY by
    # carrying over-6 and over-15 vectors.
    if A15.exists():
        a15 = joblib.load(A15)
        d0 = float(np.max(np.abs(cal._v[0] - a15._v[0])))
        dg = float(np.max(np.abs(g - a15._global)))
        print(f"vs A15 (over-0-only): max|over0 - A15.over0|={d0:.2e}  "
              f"max|global - A15.global|={dg:.2e}")
    if A14.exists():
        a14 = joblib.load(A14)
        for ov in REGIME_OVERS:
            d = float(np.max(np.abs(cal._v[ov] - a14._v[ov])))
            print(f"vs A14 (20-vector): max|over{ov} - A14.over{ov}|={d:.2e}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(cal, OUT)
    print(f"\nSaved -> {OUT}")
    print(f"Effective distinct vectors: 4 (overs 0/6/15 + global) vs A15's 2, A14's 20.")


if __name__ == "__main__":
    main()
