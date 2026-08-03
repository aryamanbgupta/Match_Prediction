"""B8 — compose the HYBRID ball calibrator (stale v1 global + B7 venue-ON over-0).

NO fitting happens here. This script is pure composition of two artifacts that
already exist:

  * `models/xgb_v3/vector_scaling_calibrator_v1.pkl` — the E5 single global
    vector (`VectorScalingCalibrator._v`). B7 showed that REFITTING this global
    on venue-ON validation predictions actively HURTS (pooled 6-line tail
    dBrier +0.0079, CI excluding 0), i.e. the stale v1 global is *not* stale
    under venue-ON. So we keep it bit-exactly.
  * `models/auto/b7/over0_calibrator_venueon.pkl` — B7's venue-ON refit
    `OverVectorScalingCalibrator`. Its over-0 vector (`._v[0]`) carried the
    A14/A15 first-over win under venue-ON (`team_first_over_mae` dMAE -0.024
    [-0.040, -0.007]). Its `._global`, however, is B7's HARMFUL refit global
    and must NOT leak into the hybrid.

Result: `OverVectorScalingCalibrator(weights={0: b7._v[0]},
global_weights=v1._v)` — behaves EXACTLY like v1 on every ball outside over 0
(overs 1-19 and over=None fall through `_vector_for` to `_global`), and applies
B7's venue-ON over-0 correction inside over 0.

Output: models/auto/b8/hybrid_calibrator.pkl
"""
import sys
from pathlib import Path

import joblib
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from calibration import (OverVectorScalingCalibrator,  # noqa: E402
                         VectorScalingCalibrator)  # noqa: E402

V1 = REPO / "models/xgb_v3/vector_scaling_calibrator_v1.pkl"
B7 = REPO / "models/auto/b7/over0_calibrator_venueon.pkl"
OUT = REPO / "models/auto/b8/hybrid_calibrator.pkl"

CLASS_NAMES = ["dot", "one", "two", "four", "six", "wicket"]


def check(label, ok):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    assert ok, f"ASSERTION FAILED: {label}"


def main():
    v1 = joblib.load(V1)
    b7 = joblib.load(B7)
    assert isinstance(v1, VectorScalingCalibrator), type(v1)
    assert isinstance(b7, OverVectorScalingCalibrator), type(b7)

    print(f"v1  : {V1}")
    print(f"      weights = {np.round(v1._v, 6)}")
    print(f"b7  : {B7}")
    print(f"      _v keys = {sorted(b7._v.keys())}")
    print(f"      _v[0]   = {np.round(b7._v[0], 6)}")
    print(f"      _global = {np.round(b7._global, 6)}   <-- REFIT global, DROPPED")

    hybrid = OverVectorScalingCalibrator(weights={0: b7._v[0]},
                                         global_weights=v1._v)

    print("\nStructural assertions:")
    check("sorted(b7._v.keys()) == [0]", sorted(b7._v.keys()) == [0])
    d_glob = float(np.max(np.abs(hybrid._global - v1._v)))
    check(f"max|hybrid._global - v1._v| == 0.0  (got {d_glob:.3e})", d_glob == 0.0)
    d_v0 = float(np.max(np.abs(hybrid._v[0] - b7._v[0])))
    check(f"max|hybrid._v[0] - b7._v[0]| == 0.0  (got {d_v0:.3e})", d_v0 == 0.0)
    # PLAN DEVIATION (documented in research/handoff/B8/result.md): the plan
    # wrote this check as `max|b7._global - v1._v| > 0.05` while stating the
    # expected divergence is "~0.17". 0.17 is the RELATIVE divergence
    # (max|ratio-1|); the absolute diff is only 0.0255 because both vectors are
    # sum-normalised to 1, so the absolute-form threshold is unsatisfiable by
    # construction. The check below is the relative form the plan describes.
    # Both numbers are printed either way; the hybrid itself is unaffected.
    d_refit_abs = float(np.max(np.abs(b7._global - v1._v)))
    d_refit_rel = float(np.max(np.abs(b7._global / v1._v - 1)))
    print(f"  (dropped refit global vs v1: max abs diff {d_refit_abs:.6f}, "
          f"max|ratio-1| {d_refit_rel:.6f})")
    check(f"max|b7._global/v1._v - 1| > 0.05  (got {d_refit_rel:.6f}) "
          f"-> confirms the dropped refit global diverges", d_refit_rel > 0.05)
    check("sorted(hybrid._v.keys()) == [0]", sorted(hybrid._v.keys()) == [0])

    # Functional equivalence on a random probability batch.
    rng = np.random.default_rng(0)
    p = rng.random((5000, 6))
    p = p / p.sum(axis=1, keepdims=True)

    print("\nFunctional assertions (bit-exact, 5000 random prob rows):")
    ref_v1 = v1.calibrate_probs(p)
    for k in (1, 7, 19):
        d = float(np.max(np.abs(hybrid.calibrate_probs(p, over=k) - ref_v1)))
        check(f"hybrid.calibrate_probs(p, over={k}) == v1.calibrate_probs(p) "
              f"(max abs diff {d:.3e})", d == 0.0)
    d = float(np.max(np.abs(hybrid.calibrate_probs(p, over=None) - ref_v1)))
    check(f"hybrid.calibrate_probs(p, over=None) == v1.calibrate_probs(p) "
          f"(max abs diff {d:.3e})", d == 0.0)
    d0 = float(np.max(np.abs(hybrid.calibrate_probs(p, over=0)
                             - b7.calibrate_probs(p, over=0))))
    check(f"hybrid.calibrate_probs(p, over=0) == b7.calibrate_probs(p, over=0) "
          f"(max abs diff {d0:.3e})", d0 == 0.0)
    # Sanity: over 0 must actually DIFFER from v1 (otherwise the hybrid is a
    # no-op relative to the baseline run).
    d0v1 = float(np.max(np.abs(hybrid.calibrate_probs(p, over=0) - ref_v1)))
    check(f"hybrid over-0 output differs from v1 output "
          f"(max abs diff {d0v1:.3e} > 0)", d0v1 > 0.0)

    ratio = hybrid._v[0] / hybrid._global
    print("\nOver-0 vs global divergence (this is the only behavioural change):")
    for name, r in zip(CLASS_NAMES, ratio):
        print(f"    {name:<8} ratio = {r:.4f}")
    print(f"  max|_v[0]/_global - 1| = {float(np.max(np.abs(ratio - 1))):.4f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(hybrid, OUT)
    print(f"\nSaved -> {OUT}")
    print("ALL ASSERTIONS PASSED")


if __name__ == "__main__":
    main()
