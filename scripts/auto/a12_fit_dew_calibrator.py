"""A12 — fit a dew-conditional 2nd-innings vector-scaling ball calibrator.

Follow-up to A6 (dew has no *match-level* winner signal) and A8/A14 (vector /
per-over calibrators). Hypothesis: the wet-ball dew effect is a **ball-level,
second-innings** phenomenon that match aggregation washes out. We fit two
marginal-matching vectors on validation **innings-2** balls — one for low-dew,
one for high-dew (split at the median evening RH) — and store their ratio as a
*centered* tilt (sqrt(v_high/v_low)) so that at mean dew the correction equals
the E5 v1 global vector exactly. See `calibration.DewVectorScalingCalibrator`.

Evening-RH lookup is rebuilt OFFLINE from the cached open-meteo archive
(`data/external/weather/`, populated by A6) — no network. It replicates A6's
exact per-venue date-range fetch so every archive-cache key hits. Venues
without geocode/cache or dates without evening coverage fall back to the global
vector (== vec baseline), same as A6's 89.9% coverage.

Output: models/auto/a12/dew_calibrator.pkl
"""
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from calibration import (DewVectorScalingCalibrator,  # noqa: E402
                         VectorScalingCalibrator, _apply_encoders_to_df,
                         _fit_scaling_vector)

MATCH_DIR = REPO / "data/xgb_match_data_v2_clean"          # for the RH lookup
VAL = REPO / "data/xgb_data_v3/cricket_data_v3_validation.parquet"
MODEL = REPO / "models/xgb_v3/xgboost_model_v3.pkl"
FEATS = REPO / "models/xgb_v3/feature_columns_v3.txt"
ENC_DIR = REPO / "models/xgb_v3"
V1 = REPO / "models/xgb_v3/vector_scaling_calibrator_v1.pkl"
POLY_DIR = REPO / "data/polymarket_test"
WX_DIR = REPO / "data/external/weather"
GEO_CACHE = WX_DIR / "geocode.json"
ARCH_DIR = WX_DIR / "archive"
OUT = REPO / "models/auto/a12/dew_calibrator.pkl"

CLASS_NAMES = ["dot", "one", "two", "four", "six", "wicket"]
EVENING_HOURS = [18, 19, 20, 21, 22]


# ---- offline weather lookup (replicates A6, cache-only, no network) ----------
def _sanitize(s):
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")[:80]


def _archive_cached(lat, lon, start, end):
    key = _sanitize(f"{lat:.3f}_{lon:.3f}_{start}_{end}")
    fp = ARCH_DIR / (key + ".json")
    if fp.exists():
        return json.load(open(fp))
    return None


def _evening_by_date(arch):
    h = arch.get("hourly", {}) if isinstance(arch, dict) else {}
    times, rh = h.get("time", []), h.get("relative_humidity_2m", [])
    acc = defaultdict(list)
    for i, t in enumerate(times):
        if int(t[11:13]) in EVENING_HOURS and i < len(rh) and rh[i] is not None:
            acc[t[:10]].append(rh[i])
    return {d: float(np.mean(v)) for d, v in acc.items() if v}


def build_rh_lookup():
    """{(venue, 'YYYY-MM-DD'): evening_rh} over the match-level parquet's
    venues/date-spans (== A6's fetch ranges, so archive cache hits)."""
    geo = json.load(open(GEO_CACHE)) if GEO_CACHE.exists() else {}
    frames = [pd.read_parquet(MATCH_DIR / f"{s}.parquet",
                              columns=["venue", "match_date"])
              for s in ["train", "validation", "test"]]
    allm = pd.concat(frames).dropna()
    venues = sorted(allm["venue"].unique())
    spans = allm.groupby("venue")["match_date"].agg(["min", "max"])
    lookup = {}
    n_geo = n_arch = 0
    for v in venues:
        g = geo.get(v)
        if not g:
            continue
        n_geo += 1
        start, end = str(spans.loc[v, "min"])[:10], str(spans.loc[v, "max"])[:10]
        arch = _archive_cached(g["lat"], g["lon"], start, end)
        if arch is None:
            continue
        n_arch += 1
        for d, r in _evening_by_date(arch).items():
            lookup[(v, d)] = r
    print(f"[rh] venues={len(venues)} geocoded={n_geo} archive-cached={n_arch} "
          f"-> {len(lookup):,} (venue,date) RH rows")
    return lookup


def poly_coverage(lookup):
    """Report how many of the 261 sim matches have an RH entry (eval coverage)."""
    hit = tot = 0
    for fp in sorted(POLY_DIR.glob("*.json")):
        j = json.load(open(fp))
        info = j.get("info", {})
        ven = info.get("venue")
        dates = info.get("dates") or []
        if not ven or not dates:
            continue
        tot += 1
        if (ven, str(dates[0])[:10]) in lookup:
            hit += 1
    print(f"[rh] polymarket_test eval coverage: {hit}/{tot} matches "
          f"({hit/max(tot,1):.1%})")


# ---- fit ---------------------------------------------------------------------
def main():
    lookup = build_rh_lookup()
    poly_coverage(lookup)

    feats = [l.strip() for l in open(FEATS)]
    model = joblib.load(MODEL)
    df = pd.read_parquet(VAL)
    _apply_encoders_to_df(df, feats, str(ENC_DIR))
    df["venue_encoded"] = 0  # sim input distribution (E5/A14)

    tgt = df["ball_outcome"].replace({-1: 7}).map(
        {0: 0, 1: 1, 2: 2, 4: 3, 6: 4, 7: 5})
    keep = tgt.notna()
    df = df[keep].copy()
    y = tgt[keep].astype(int).values
    raw = model.predict_proba(df[feats].values)
    print(f"[fit] val balls (valid outcomes): {len(y):,}")

    # global fallback vector — must reproduce v1 exactly.
    g = _fit_scaling_vector(raw, y)
    v1 = joblib.load(V1)._v
    print(f"[fit] global weights: {np.round(g, 6)}")
    print(f"[fit] v1 weights:     {np.round(v1, 6)}")
    print(f"[fit] max|global - v1| = {float(np.max(np.abs(g - v1))):.2e} "
          f"(0 => pipeline validated)")

    # innings-2 balls with RH coverage.
    inn2 = df["inning_idx"].values == 2
    rh = np.array([lookup.get((v, str(d)[:10]), np.nan)
                   for v, d in zip(df["venue"].values, df["match_date"].values)])
    have = inn2 & ~np.isnan(rh)
    n_inn2 = int(inn2.sum())
    n_have = int(have.sum())
    thr = float(np.median(rh[have]))
    print(f"[fit] innings-2 balls: {n_inn2:,}; with RH: {n_have:,} "
          f"({n_have/max(n_inn2,1):.1%}); median evening RH = {thr:.2f}%")

    hi = have & (rh >= thr)
    lo = have & (rh < thr)
    print(f"[fit] high-dew (rh>={thr:.1f}) balls: {int(hi.sum()):,}; "
          f"low-dew: {int(lo.sum()):,}")

    v_hi = _fit_scaling_vector(raw[hi], y[hi])
    v_lo = _fit_scaling_vector(raw[lo], y[lo])
    tilt = v_hi / v_lo
    sqrt_tilt = np.sqrt(tilt)
    print(f"\n[fit] {'class':<8} {'v_low':>9} {'v_high':>9} {'tilt=hi/lo':>11} "
          f"{'sqrt_tilt':>10}")
    for i, c in enumerate(CLASS_NAMES):
        print(f"      {c:<8} {v_lo[i]:>9.5f} {v_hi[i]:>9.5f} "
              f"{tilt[i]:>11.4f} {sqrt_tilt[i]:>10.4f}")
    print(f"[fit] max|tilt-1| across classes = {float(np.max(np.abs(tilt-1))):.4f} "
          f"(dew signal strength; ~0 => dew null at ball level)")

    cal = DewVectorScalingCalibrator(global_weights=g, sqrt_tilt=sqrt_tilt,
                                     rh_lookup=lookup, rh_threshold=thr)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(cal, OUT)
    print(f"\n[done] saved -> {OUT}")

    # sanity: innings-1 / no-RH path == baseline vec exactly
    test_probs = raw[:5]
    base = VectorScalingCalibrator(weights=g).calibrate_probs(test_probs)
    dew_inn1 = cal.calibrate_probs(test_probs, innings=1, venue="X",
                                   match_date="2026-01-01")
    print(f"[chk] innings-1 path vs vec baseline max abs diff: "
          f"{float(np.max(np.abs(base - dew_inn1))):.2e} (0 => byte-identical)")


if __name__ == "__main__":
    main()
