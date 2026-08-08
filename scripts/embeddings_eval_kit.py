"""Frozen evaluation kit for the player-embedding ladder (E1..E4).

Builds, once, the scoreboard every embedding rung is judged against
(see ~/Projects/sloan-sports-analytics/EMBEDDINGS_TRANSFORMER_DESIGN.md):

  1. Target construction identical to production: parsing_v2's
     normalize_ball_outcome output (already in the parquets as
     `ball_outcome`) remapped with xgboost_v2's class_mapping.
  2. Unseen-pairs slices: val/test rows whose (batter_id, bowler_id)
     pair never co-occurs in train — the lateral-generalization eval.
  3. Probe labels per player: batting hand, pace/spin, bowling arm
     (from data/all_players_enriched.csv via PlayerMetadataProvider's
     style classifications).
  4. Neighbor panel: fixed list of well-known players (resolved to
     cricsheet IDs) whose nearest-neighbor lists are eyeballed per rung.
  5. Baselines: B0a = train-prior distribution; B0b = multinomial
     logistic regression on the 42 V6 EB-shrunk outcome-dist features.
     (B0c = production ball model, recorded externally.)

Artifacts under models/embeddings/eval_kit/ (never overwrites production
paths). data/golden/ is not read. Re-running overwrites the kit ONLY if
--force is passed; the kit is meant to be built once and frozen.

Usage:
    uv run python scripts/embeddings_eval_kit.py [--force] [--skip-b0b]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from feature_registry import FEATURE_GROUPS  # noqa: E402
from player_metadata import PlayerMetadataProvider  # noqa: E402

DATA_DIR = Path("data/xgb_data_v3")
OUT_DIR = Path("models/embeddings/eval_kit")

# xgboost_v2.py:106-109,259 — ball_outcome -> 6 contiguous classes.
CLASS_MAPPING = {0: 0, 1: 1, 2: 2, 4: 3, 6: 4, -1: 5}
CLASS_NAMES = ["dot", "one", "two", "four", "six", "wicket"]

EB_COLS = (
    FEATURE_GROUPS["batter_outcome_dist"]
    + FEATURE_GROUPS["bowler_outcome_dist"]
    + FEATURE_GROUPS["batter_vs_type_dist"]
    + FEATURE_GROUPS["bowler_vs_hand_dist"]
    + FEATURE_GROUPS["venue_outcome_dist"]
)

ID_COLS = ["batter_id", "bowler_id", "ball_outcome", "match_date", "innings_id"]

# Neighbor panel: household-name players across roles/eras. Resolved
# against all_players_enriched.csv `name` (cricsheet registry names);
# unresolved names are reported and dropped, not fatal.
NEIGHBOR_PANEL_NAMES = [
    "V Kohli", "RG Sharma", "MS Dhoni", "AB de Villiers", "CH Gayle",
    "DA Warner", "JC Buttler", "GJ Maxwell", "AD Russell", "KA Pollard",
    "SP Narine", "Rashid Khan", "JJ Bumrah", "SL Malinga", "DJ Bravo",
    "Shakib Al Hasan", "Babar Azam", "SA Yadav", "HH Pandya", "YS Chahal",
    "TG Southee", "Wanindu Hasaranga",
]


def load_split(name: str, columns: list[str]) -> pd.DataFrame:
    path = DATA_DIR / f"cricket_data_v3_{name}.parquet"
    df = pd.read_parquet(path, columns=columns)
    print(f"  {name}: {len(df):,} balls", flush=True)
    return df


def log_loss_from_probs(y: np.ndarray, probs: np.ndarray) -> float:
    p = np.clip(probs[np.arange(len(y)), y], 1e-15, 1.0)
    return float(-np.mean(np.log(p)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-b0b", action="store_true",
                    help="Skip the logistic-regression baseline (slowest step)")
    args = ap.parse_args()

    if OUT_DIR.exists() and not args.force:
        sys.exit(f"{OUT_DIR} exists — the kit is frozen. Pass --force to rebuild.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest: dict = {
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "class_mapping": {str(k): v for k, v in CLASS_MAPPING.items()},
        "class_names": CLASS_NAMES,
        "eb_cols": EB_COLS,
        "feature_hash": (DATA_DIR / ".feature_hash").read_text().strip(),
    }

    print("Loading splits (train/validation/test — golden untouched)...", flush=True)
    cols = ID_COLS + EB_COLS
    train = load_split("train", cols)
    val = load_split("validation", cols)
    test = load_split("test", cols)

    for name, df in [("train", train), ("validation", val), ("test", test)]:
        df["y"] = df["ball_outcome"].map(CLASS_MAPPING).astype(np.int8)
        manifest[f"{name}_rows"] = len(df)
        manifest[f"{name}_class_counts"] = (
            df["y"].value_counts().sort_index().tolist()
        )

    # --- Unseen-pairs slices -------------------------------------------------
    print("Building unseen-pair masks...", flush=True)
    train_pairs = set(zip(train["batter_id"], train["bowler_id"]))
    masks = {}
    for name, df in [("validation", val), ("test", test)]:
        pairs = list(zip(df["batter_id"], df["bowler_id"]))
        mask = np.fromiter((p not in train_pairs for p in pairs),
                           dtype=bool, count=len(pairs))
        masks[name] = mask
        manifest[f"{name}_unseen_pair_balls"] = int(mask.sum())
        manifest[f"{name}_unseen_pair_frac"] = round(float(mask.mean()), 4)
        print(f"  {name}: {mask.sum():,} balls from unseen pairs "
              f"({mask.mean():.1%})", flush=True)
    np.savez_compressed(OUT_DIR / "unseen_pair_masks.npz", **masks)

    # --- Probe labels --------------------------------------------------------
    print("Building probe labels...", flush=True)
    meta = PlayerMetadataProvider("data/all_players_enriched.csv")
    all_ids = pd.unique(
        pd.concat([train["batter_id"], train["bowler_id"],
                   val["batter_id"], val["bowler_id"],
                   test["batter_id"], test["bowler_id"]])
    )
    rows = []
    for pid in all_ids:
        rows.append({
            "player_id": pid,
            "bat_hand": meta.get_batter_hand(pid),
            "bowl_kind": meta.get_bowling_type(pid),
            "bowl_arm": meta.get_bowler_arm(pid),
            "train_balls_batting": 0, "train_balls_bowling": 0,
        })
    probes = pd.DataFrame(rows).set_index("player_id")
    bat_counts = train["batter_id"].value_counts()
    bowl_counts = train["bowler_id"].value_counts()
    probes["train_balls_batting"] = bat_counts.reindex(probes.index).fillna(0).astype(int)
    probes["train_balls_bowling"] = bowl_counts.reindex(probes.index).fillna(0).astype(int)
    probes.to_parquet(OUT_DIR / "probe_labels.parquet")
    manifest["n_players"] = len(probes)
    manifest["probe_coverage"] = {
        c: int(probes[c].notna().sum()) for c in ["bat_hand", "bowl_kind", "bowl_arm"]
    }
    print(f"  {len(probes):,} players; coverage {manifest['probe_coverage']}", flush=True)

    # --- Neighbor panel ------------------------------------------------------
    enriched = pd.read_csv("data/all_players_enriched.csv")
    # Registry short forms ("V Kohli") live in unique_name; `name` holds the
    # full name ("Virat Kohli"). Panel names are short forms — try both.
    name_to_id = dict(zip(enriched["name"], enriched["cricsheet_id"]))
    name_to_id.update(zip(enriched["unique_name"], enriched["cricsheet_id"]))
    panel = {n: name_to_id[n] for n in NEIGHBOR_PANEL_NAMES if n in name_to_id}
    missing = [n for n in NEIGHBOR_PANEL_NAMES if n not in name_to_id]
    if missing:
        print(f"  neighbor panel: unresolved names dropped: {missing}", flush=True)
    manifest["neighbor_panel"] = panel
    print(f"  neighbor panel: {len(panel)} players resolved", flush=True)

    # --- B0a: train-prior baseline ------------------------------------------
    print("B0a: train-prior baseline...", flush=True)
    prior = np.bincount(train["y"], minlength=6).astype(np.float64)
    prior /= prior.sum()
    manifest["train_prior"] = [round(p, 6) for p in prior]
    b0a = {}
    for name, df in [("validation", val), ("test", test)]:
        y = df["y"].to_numpy()
        probs = np.tile(prior, (len(y), 1))
        b0a[name] = log_loss_from_probs(y, probs)
        b0a[f"{name}_unseen"] = log_loss_from_probs(
            y[masks[name]], probs[masks[name]])
    manifest["B0a_log_loss"] = {k: round(v, 4) for k, v in b0a.items()}
    print(f"  {manifest['B0a_log_loss']}", flush=True)

    # --- B0b: logistic on 42 EB features ------------------------------------
    if not args.skip_b0b:
        print("B0b: multinomial logistic on 42 EB features "
              "(slowest step, several minutes)...", flush=True)
        from sklearn.linear_model import LogisticRegression
        X_tr = train[EB_COLS].to_numpy(dtype=np.float32)
        clf = LogisticRegression(max_iter=200, n_jobs=-1)
        t0 = time.time()
        clf.fit(X_tr, train["y"])
        print(f"  fit in {time.time()-t0:.0f}s (converged: "
              f"{clf.n_iter_.max() < 200})", flush=True)
        b0b = {}
        for name, df in [("validation", val), ("test", test)]:
            probs = clf.predict_proba(df[EB_COLS].to_numpy(dtype=np.float32))
            y = df["y"].to_numpy()
            b0b[name] = log_loss_from_probs(y, probs)
            b0b[f"{name}_unseen"] = log_loss_from_probs(
                y[masks[name]], probs[masks[name]])
        manifest["B0b_log_loss"] = {k: round(v, 4) for k, v in b0b.items()}
        manifest["B0b_n_iter"] = int(clf.n_iter_.max())
        print(f"  {manifest['B0b_log_loss']}", flush=True)
        import joblib
        joblib.dump(clf, OUT_DIR / "b0b_logistic.joblib")

    manifest["B0c_reference"] = {
        "note": "production ball model xgb_i7_noweights (D16), per-ball LL "
                "recorded in research/reports/auto/D16.md — not recomputed here",
        "log_loss": 1.4253,
    }

    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nEval kit frozen at {OUT_DIR}/ — manifest.json written.", flush=True)


if __name__ == "__main__":
    main()
