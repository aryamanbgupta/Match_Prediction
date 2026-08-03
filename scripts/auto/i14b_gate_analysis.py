"""I14b gate — do physical venue features help the ball model where the
venue embedding is weakest?

WRITTEN AND COMMITTED BEFORE THE I14B MODEL WAS TRAINED OR SCORED.

The I14 idea text (research/IDEAS.md) declares the gate: "Test whether this
improves unseen/low-history venue performance … Gate on grouped-by-venue
holdout performance so repeated matches at major grounds cannot hide
regressions at sparse venues." This script implements that as a paired
teacher-forced comparison on the held-out test split of the i14b frame
(identical rows to the i7 test split; the frame only adds vphys_* columns):

  baseline  models/xgb_i7_noweights_production  (the promoted D16 arm,
            114 features, suffix i7 — its own encoders)
  i14b      models/auto/i14b/venuephys          (same no-weights config,
            124 features = 114 + 10 vphys_*, suffix i14b — its own encoders)

PRE-COMMITTED GATE:

  PRIMARY  paired per-ball NLL delta (i14b − baseline) on the LOW-HISTORY
           slice — balls at venues with ≤ 20 distinct TRAIN matches
           (including venues absent from train) — must be CI-clean
           negative (95% cluster-bootstrap CI by match, 2000 resamples,
           seed 29, hi < 0).
  GUARD    overall test NLL delta must NOT be CI-clean positive
           (CI lo <= 0).

  Mapping: PRIMARY met + GUARD held -> BALL GATE PASS (worth wiring the
  vphys features into the sim path and running the recipe-B prop gate).
  PRIMARY not CI-clean negative -> FAILED (the learned venue embedding +
  venue_p* features already carry the physical signal at ball level;
  registry stays a data asset, no model integration). GUARD broken ->
  FAILED with a regression flag regardless of PRIMARY.

  This script PRINTS the mapping; the orchestrator (interactive session)
  issues the verdict.

Context (cannot flip the verdict): per-bucket deltas by venue train-match
count {0, 1–5, 6–20, 21–100, >100}, registry-covered vs not, boundary-known
vs not, and validation LL for both arms.

Run:
  uv run python scripts/auto/i14b_gate_analysis.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

FRAME = REPO / "data" / "xgb_data_i14b"
BASELINE_DIR = REPO / "models" / "xgb_i7_noweights_production"
I14B_DIR = REPO / "models" / "auto" / "i14b" / "venuephys"

N_BOOT = 2000
BOOT_SEED = 29
LOW_HISTORY_MAX_TRAIN_MATCHES = 20
BUCKETS = [(0, 0), (1, 5), (6, 20), (21, 100), (101, 10**9)]

CLS_MAP = {0: 0, 1: 1, 2: 2, 4: 3, 6: 4, -1: 5}


def feature_list(model_dir: Path, suffix: str) -> list[str]:
    p = model_dir / f"feature_columns_{suffix}.txt"
    return [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]


def encode_frame(df: pd.DataFrame, model_dir: Path, suffix: str,
                 feat: list[str]) -> pd.DataFrame:
    out = df.copy()
    enc_specs = [
        ("batter_id", "batter_encoded", f"batter_encoder_{suffix}.pkl"),
        ("bowler_id", "bowler_encoded", f"bowler_encoder_{suffix}.pkl"),
        ("matchup_type", "matchup_type_encoded", f"matchup_encoder_{suffix}.pkl"),
        ("venue", "venue_encoded", f"venue_encoder_{suffix}.pkl"),
    ]
    for raw, enc_name, enc_file in enc_specs:
        if enc_name not in feat:
            continue
        le = joblib.load(model_dir / enc_file)
        lut = {str(c): i for i, c in enumerate(le.classes_)}
        out[enc_name] = out[raw].astype(str).map(lut).fillna(-1).astype(int)
    return out


def per_ball_nll(model_dir: Path, suffix: str, df: pd.DataFrame,
                 y: np.ndarray) -> np.ndarray:
    feat = feature_list(model_dir, suffix)
    enc = encode_frame(df, model_dir, suffix, feat)
    missing = [f for f in feat if f not in enc.columns]
    if missing:
        raise SystemExit(f"{model_dir.name}: missing feature columns {missing}")
    model = joblib.load(
        model_dir / f"xgboost_model_{suffix}.pkl")
    proba = model.predict_proba(enc[feat])
    p = np.clip(proba[np.arange(len(y)), y], 1e-12, 1.0)
    return -np.log(p)


def cluster_boot_ci(delta: np.ndarray, groups: np.ndarray,
                    n_boot: int = N_BOOT, seed: int = BOOT_SEED):
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    idx_by_g = [np.where(inv == g)[0] for g in range(len(uniq))]
    means = np.empty(n_boot)
    for b in range(n_boot):
        picks = rng.integers(0, len(idx_by_g), size=len(idx_by_g))
        sel = np.concatenate([idx_by_g[i] for i in picks])
        means[b] = delta[sel].mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def flag(lo: float, hi: float) -> str:
    if hi < 0:
        return "DOWN(better) CI-clean"
    if lo > 0:
        return "UP(worse) CI-clean"
    return "~noise"


def slice_report(name: str, mask: np.ndarray, d: np.ndarray,
                 groups: np.ndarray, nll_a: np.ndarray, nll_b: np.ndarray):
    if mask.sum() == 0:
        print(f"{name:<42} (no balls)")
        return None
    lo, hi = cluster_boot_ci(d[mask], groups[mask])
    print(f"{name:<42}{mask.sum():>9,}{nll_a[mask].mean():>10.4f}"
          f"{nll_b[mask].mean():>10.4f}{d[mask].mean():>+10.4f}   "
          f"[{lo:+.4f},{hi:+.4f}]  {flag(lo, hi)}")
    return d[mask].mean(), lo, hi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-dir", type=Path, default=BASELINE_DIR)
    ap.add_argument("--i14b-dir", type=Path, default=I14B_DIR)
    args = ap.parse_args()

    test = pd.read_parquet(FRAME / "cricket_data_i14b_test.parquet")
    y = np.array([CLS_MAP[v] for v in test["ball_outcome"].astype(int).values])
    match_key = test["innings_id"].str.split("_", n=1).str[1].values

    train_venues = pd.read_parquet(
        FRAME / "cricket_data_i14b_train.parquet",
        columns=["venue", "innings_id"])
    train_counts = (
        train_venues.assign(
            mk=train_venues["innings_id"].str.split("_", n=1).str[1])
        .groupby("venue")["mk"].nunique())
    venue_train_matches = test["venue"].map(train_counts).fillna(0).astype(int)

    print(f"test balls: {len(test):,} | matches: {len(set(match_key)):,} | "
          f"venues: {test['venue'].nunique()}")
    print(f"baseline: {args.baseline_dir}")
    print(f"i14b arm: {args.i14b_dir}\n")

    print("scoring baseline (suffix i7) ...")
    nll_a = per_ball_nll(args.baseline_dir, "i7", test, y)
    print(f"  overall test LL {nll_a.mean():.4f}")
    print("scoring i14b arm (suffix i14b) ...")
    nll_b = per_ball_nll(args.i14b_dir, "i14b", test, y)
    print(f"  overall test LL {nll_b.mean():.4f}\n")

    d = nll_b - nll_a
    groups = match_key

    hdr = (f"{'slice':<42}{'balls':>9}{'base':>10}{'i14b':>10}"
           f"{'delta':>10}   95% CI (cluster by match)")
    print(hdr)
    print("-" * len(hdr))

    low_mask = (venue_train_matches <= LOW_HISTORY_MAX_TRAIN_MATCHES).values
    primary = slice_report(
        f"PRIMARY: venues <= {LOW_HISTORY_MAX_TRAIN_MATCHES} train matches",
        low_mask, d, groups, nll_a, nll_b)
    overall = slice_report("GUARD: all test balls",
                           np.ones(len(d), bool), d, groups, nll_a, nll_b)

    print("\ncontext buckets (venue train-match count):")
    for lo_b, hi_b in BUCKETS:
        m = ((venue_train_matches >= lo_b) & (venue_train_matches <= hi_b)).values
        label = f"  {lo_b}" if lo_b == hi_b else f"  {lo_b}-{hi_b if hi_b < 10**9 else '+'}"
        slice_report(label, m, d, groups, nll_a, nll_b)
    print("\ncontext coverage slices:")
    slice_report("  boundary known", (test["vphys_boundary_known"] == 1).values,
                 d, groups, nll_a, nll_b)
    slice_report("  boundary unknown", (test["vphys_boundary_known"] == 0).values,
                 d, groups, nll_a, nll_b)
    slice_report("  climate known", (test["vphys_climate_known"] == 1).values,
                 d, groups, nll_a, nll_b)

    gate1 = primary is not None and primary[2] < 0
    gate2 = overall is not None and not (overall[1] > 0)
    print(f"\nPRIMARY (low-history CI-clean better): {'PASS' if gate1 else 'FAIL'}")
    print(f"GUARD   (overall not CI-clean worse):   {'HELD' if gate2 else 'BROKEN'}")
    mapping = ("BALL GATE PASS -> wire vphys into the sim path, run recipe-B"
               if gate1 and gate2 else "FAILED")
    print("\nPre-committed verdict MAPPING (orchestrator decides): " + mapping)


if __name__ == "__main__":
    main()
