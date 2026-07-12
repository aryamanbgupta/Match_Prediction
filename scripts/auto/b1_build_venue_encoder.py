"""B1 — rebuild the training-time venue LabelEncoder for the v7 sim model.

xgboost_v2.py fits le_venue on the unique venues of train+val+test at training
time but never dumps it (only batter/bowler/matchup encoders are saved), so
XGBoostModelV2 has no way to set venue_encoded at sim time and every simulated
ball scores as venue code 0 (TODO.md "Pipeline bugs", filed at E5).

LabelEncoder.fit sorts classes (np.unique), so refitting on the same parquet
reproduces the training mapping exactly. To prove today's parquet IS the one
that trained the deployed model, we refit the batter and bowler encoders the
same way and require their classes_ to byte-match the artifacts saved at
training time (models/xgb_v3/{batter,bowler}_encoder_v3.pkl). If those match,
the venue rebuild is training-exact by the same construction.

Outputs models/auto/b1/venue_encoder_v3.pkl + coverage stats over the
polymarket_test venues.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

DATA_DIR = Path("data/xgb_data_v3")
OUT_DIR = Path("models/auto/b1")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def uniques_across_splits(col: str) -> np.ndarray:
    """Reproduce xgboost_v2.py's pd.concat([...astype(str)]).unique()."""
    parts = []
    for split in ("train", "validation", "test"):
        df = pd.read_parquet(DATA_DIR / f"cricket_data_v3_{split}.parquet",
                             columns=[col])
        parts.append(df[col].astype(str))
    return pd.concat(parts).unique()


def main():
    # --- 1. Prove the parquet is the training parquet (batter/bowler refit) ---
    for col, artifact in (("batter_id", "models/xgb_v3/batter_encoder_v3.pkl"),
                          ("bowler_id", "models/xgb_v3/bowler_encoder_v3.pkl")):
        le = LabelEncoder().fit(uniques_across_splits(col))
        saved = joblib.load(artifact)
        same = (len(le.classes_) == len(saved.classes_)
                and bool(np.all(le.classes_ == saved.classes_)))
        print(f"{col}: refit {len(le.classes_)} classes vs saved "
              f"{len(saved.classes_)} — {'MATCH' if same else 'MISMATCH'}")
        if not same:
            raise SystemExit(
                f"ABORT: refit {col} encoder does not match the saved training "
                f"artifact — parquet has drifted since training; venue rebuild "
                f"would not be training-exact.")

    # --- 2. Rebuild the venue encoder exactly as xgboost_v2.py did ---
    le_venue = LabelEncoder().fit(uniques_across_splits("venue"))
    out_path = OUT_DIR / "venue_encoder_v3.pkl"
    joblib.dump(le_venue, out_path)
    print(f"venue: {len(le_venue.classes_)} classes -> {out_path}")

    # --- 3. Coverage over the sim eval set (raw cricsheet info.venue,
    #        the exact string TestMatchLoader puts on state.venue) ---
    classes = set(str(c) for c in le_venue.classes_)
    test_files = sorted(Path("data/polymarket_test").glob("*.json"))
    venues, missing = [], []
    for fp in test_files:
        with open(fp) as f:
            v = json.load(f)["info"].get("venue", "Unknown")
        venues.append(v)
        if str(v) not in classes:
            missing.append((fp.name, v))
    n = len(venues)
    print(f"polymarket_test coverage: {n - len(missing)}/{n} matches "
          f"({100.0 * (n - len(missing)) / n:.1f}%) have their venue in the "
          f"encoder")
    if missing:
        print("missing venues:")
        for name, v in missing:
            print(f"  {name}: {v!r}")

    # --- 4. Context: venue_encoded importance in the deployed model ---
    imp = json.load(open("models/xgb_v3/feature_importance.json"))
    if isinstance(imp, dict) and "venue_encoded" in imp:
        ranked = sorted(imp.items(), key=lambda kv: -kv[1])
        rank = [k for k, _ in ranked].index("venue_encoded") + 1
        print(f"venue_encoded importance: {imp['venue_encoded']:.6f} "
              f"(rank {rank}/{len(ranked)})")


if __name__ == "__main__":
    main()
