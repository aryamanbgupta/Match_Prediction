"""B7 — refit the ball calibrators on the venue-ON sim input distribution.

B6 shipped the training-time venue encoder into the default sim path
(`models/xgb_v3/venue_encoder_v3.pkl` sidecar): every simulated ball now
scores with its real venue code instead of venue_encoded=0. But both
production ball calibrators were fit on *venue-blind* val predictions:

  - `models/xgb_v3/vector_scaling_calibrator_v1.pkl` (E5): global 6-vector,
    fit with `df["venue_encoded"] = 0` "so the correction matches deployment"
    — which was true pre-B6 and is false now.
  - `models/auto/a15/over0_calibrator.pkl` (A15): over-0 vector + the same
    global fallback, fit the same way (a15_fit_over0_calibrator.py line 57).

This script refits BOTH under the venue-ON input distribution: identical
val-ball construction to A14/A15 (same parquet, same encoder application,
same target remap, same `_fit_scaling_vector` fixed point), with exactly one
change — `venue_encoded` carries the real training codes from the shipped
encoder instead of being zeroed.

Pre-run washout call (per IDEAS.md B7): if the refit vectors are close to the
stale ones (max |ratio-1| < the ~0.05 threshold below which A8/A12 per-ball
tilts washed out on aggregate props), expect a null result and say so BEFORE
the sim run.

Outputs (models/auto/b7/):
  - vector_global_venueon.pkl  VectorScalingCalibrator, refit global (v1's
                               drop-in shape; candidate v1 replacement)
  - over0_calibrator_venueon.pkl  OverVectorScalingCalibrator, refit global
                               fallback + refit over-0 vector (the full
                               recommended stack; the eval challenger)
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from calibration import (OverVectorScalingCalibrator,  # noqa: E402
                         VectorScalingCalibrator, _apply_encoders_to_df)

VAL = REPO / "data/xgb_data_v3/cricket_data_v3_validation.parquet"
MODEL = REPO / "models/xgb_v3/xgboost_model_v3.pkl"
FEATS = REPO / "models/xgb_v3/feature_columns_v3.txt"
ENC_DIR = REPO / "models/xgb_v3"
VENUE_ENC = REPO / "models/xgb_v3/venue_encoder_v3.pkl"
V1 = REPO / "models/xgb_v3/vector_scaling_calibrator_v1.pkl"
A15 = REPO / "models/auto/a15/over0_calibrator.pkl"
OUT_DIR = REPO / "models/auto/b7"

CLASS_NAMES = ["dot", "one", "two", "four", "six", "wicket"]
MIN_BALLS = 1000  # same floor as A14/A15
WASHOUT = 0.05    # A8/A12 threshold: per-ball tilts below this netted 0


def over_of(balls):
    return np.minimum((np.asarray(balls) // 6).astype(int), 19)


def main():
    assert VENUE_ENC.exists(), (
        "venue encoder sidecar missing from models/xgb_v3/ — B6's ship step "
        "is a precondition for B7")

    feats = [l.strip() for l in open(FEATS)]
    model = joblib.load(MODEL)
    df = pd.read_parquet(VAL)

    # Identical to A14/A15 EXCEPT: no `df["venue_encoded"] = 0` afterwards.
    # _apply_encoders_to_df now finds venue_encoder_v3.pkl in models/xgb_v3
    # and maps the real venue strings to training codes (unknown -> -1).
    _apply_encoders_to_df(df, feats, str(ENC_DIR))

    n_unk = int((df["venue_encoded"] == -1).sum())
    n_codes = df["venue_encoded"].nunique()
    print(f"venue-ON encoding: {n_codes} distinct val venue codes, "
          f"{n_unk} balls with unknown venue (-1) of {len(df):,} "
          f"({100.0 * n_unk / len(df):.2f}%)")

    tgt = df["ball_outcome"].replace({-1: 7}).map({0: 0, 1: 1, 2: 2, 4: 3, 6: 4, 7: 5})
    mask = tgt.notna()
    df = df[mask].copy()
    y = tgt[mask].astype(int).values
    X = df[feats].values
    print(f"Val balls (valid outcomes): {len(y):,}")

    raw = model.predict_proba(X)
    overs = over_of(df["balls_bowled"].values)

    v1 = joblib.load(V1)._v
    a15 = joblib.load(A15)

    # --- refit GLOBAL vector under venue-ON ---
    cal = OverVectorScalingCalibrator()
    g = cal.set_global(raw, y)
    r_g = g / v1
    print(f"\nclasses:                  {CLASS_NAMES}")
    print(f"refit global (venue-ON):  {np.round(g, 6)}")
    print(f"stale v1 (venue-blind):   {np.round(v1, 6)}")
    print(f"  refit/v1 ratio: {np.round(r_g, 4)}")
    div_g = float(np.max(np.abs(r_g - 1)))
    print(f"  max|ratio-1| = {div_g:.4f}")

    # --- refit OVER-0 vector under venue-ON ---
    sel = overs == 0
    n0 = int(sel.sum())
    assert n0 >= MIN_BALLS, f"over 0 has only {n0} balls (< {MIN_BALLS})"
    v0 = cal.fit_over(0, raw[sel], y[sel])
    r_0 = v0 / a15._v[0]
    print(f"\nrefit over-0 (venue-ON):  {np.round(v0, 6)}  (n={n0:,})")
    print(f"stale A15 over-0:         {np.round(a15._v[0], 6)}")
    print(f"  refit/A15 ratio: {np.round(r_0, 4)}")
    div_0 = float(np.max(np.abs(r_0 - 1)))
    print(f"  max|ratio-1| = {div_0:.4f}")
    print(f"\nover-0 / refit-global ratio (the A15 first-over correction under "
          f"venue-ON): {np.round(v0 / g, 4)}  "
          f"(max|ratio-1| = {float(np.max(np.abs(v0 / g - 1))):.3f})")

    assert sorted(cal._v.keys()) == [0], f"expected only over 0 in _v, got {sorted(cal._v.keys())}"

    # --- pre-run washout call (pre-committed in IDEAS.md B7) ---
    print("\n--- PRE-RUN WASHOUT CALL ---")
    print(f"staleness of global vector: max|refit/v1 - 1| = {div_g:.4f} "
          f"(threshold {WASHOUT})")
    print(f"staleness of over-0 vector: max|refit/A15 - 1| = {div_0:.4f} "
          f"(threshold {WASHOUT})")
    if div_g < WASHOUT and div_0 < WASHOUT:
        print("BOTH below the ~0.05 washout threshold -> the refit-vs-stale "
              "part of B7 is expected NULL; any gate movement would then come "
              "from the over-0 vector (absent from the b6 baseline run), i.e. "
              "an A15-survives-re-baseline test rather than a staleness fix.")
    else:
        print("At least one refit diverges above the washout threshold -> a "
              "real staleness correction is in play; the sim run is a live test.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_over0 = OUT_DIR / "over0_calibrator_venueon.pkl"
    joblib.dump(cal, out_over0)
    out_global = OUT_DIR / "vector_global_venueon.pkl"
    joblib.dump(VectorScalingCalibrator(weights=g), out_global)
    print(f"\nSaved -> {out_over0}  (eval challenger: refit global + refit over-0)")
    print(f"Saved -> {out_global}  (drop-in v1 replacement shape)")


if __name__ == "__main__":
    main()
