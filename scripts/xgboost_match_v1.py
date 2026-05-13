"""Train a match-level XGBoost binary classifier on the cheap-subset
feature set materialized by `scripts/materialize_match_features.py`.

Output: `models/xgb_match_v1/{model.pkl, feature_columns.txt,
encoders.pkl, test_predictions.json}`. Phase A1 of the match-level
direct + sim ensemble plan
(`~/.claude/plans/okay-let-s-go-ahead-reflective-sunrise.md`).

Subcommands:
    train         — fit on data/xgb_match_data_v1/{train,validation}.parquet,
                    save model + encoders + feature_columns.txt, then
                    score the test split for sanity (no JSON written).
    predict-test  — score test.parquet, write
                    models/xgb_match_v1/test_predictions.json keyed by
                    match_id (synth format compatible with eval JSONs).
    both          — run train then predict-test in one shot (default).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent

# Columns excluded from the feature set. Includes metadata (match_id,
# team names, dates, target) and the categorical raw strings that are
# label-encoded separately.
METADATA_COLS = {
    "match_id", "cricsheet_id", "match_date",
    "team1", "team2", "venue", "competition_tier",
    "team1_wins",
}
CATEGORICAL_FEATURES = ["venue", "competition_tier"]

# Per-feature monotone constraints (M1, 2026-05-10). Only directional features
# whose sign w.r.t. P(team1_wins) is physically unambiguous get constrained;
# style/composition features (lineup mix counts, venue character, toss flags)
# are left unconstrained (0) so the model can still pick up interactions.
# bowling_econ_diff is sign-flipped because LOWER team1 economy = BETTER team1.
_MONOTONE_SIGNS = {
    "top6_batting_elo_diff": 1,
    "bottom5_bowling_elo_diff": 1,
    "elo_diff_batting": 1,
    "elo_diff_bowling": 1,
    "win_rate_diff": 1,
    "batting_avg_diff": 1,
    "bowling_econ_diff": -1,
    "h2h_team1_win_rate_shrunk": 1,
    "is_team1_home": 1,
    "is_team2_home": -1,
    # M2 (2026-05-10) outcome-dist diffs.
    # batting diffs: team1 batters' expected boundary rate − team2's.
    # Higher team1 boundary rate = team1 scores more = wins more.
    "p4_batting_diff": 1,
    "p6_batting_diff": 1,
    # pw_batting_diff: HIGHER team1 wicket-as-batter rate = team1 batters
    # get out more often = team1 wins LESS.
    "pw_batting_diff": -1,
    # bowling diffs: team1 bowlers' expected boundary rate CONCEDED − team2's.
    # Higher team1 conceded = team1 leaks more boundaries = team1 wins LESS.
    "p4_bowling_diff": -1,
    "p6_bowling_diff": -1,
    # pw_bowling_diff: HIGHER team1 wicket-taking rate = team1 wins MORE.
    "pw_bowling_diff": 1,
    # M3 (2026-05-10) rolling-form diffs.
    "batting_avg_recent_diff": 1,    # higher team1 recent avg = better
    "batting_sr_recent_diff": 1,     # higher team1 recent SR = better
    "bowling_avg_recent_diff": -1,   # higher team1 recent bowling avg = WORSE
    "bowling_econ_recent_diff": -1,  # higher team1 recent econ = leaks more = worse
    "inform_batters_diff": 1,        # more in-form team1 batters = better
    "outofform_batters_diff": -1,    # more out-of-form team1 batters = worse
    # M4 (2026-05-10) within-tournament / scheduling diffs.
    "win_rate_last_60d_diff": 1,     # team1 winning recently = good
    "competition_win_rate_diff": 1,  # team1 winning IN-COMPETITION recently = good
    # days_since_diff and back-to-back diffs are NOT monotone in the
    # P(team1_wins) direction — fatigue (negative for too short gap) and
    # rust (negative for too long gap) effects can co-exist; leave
    # unconstrained so the tree can learn the U-shape if it exists.
}


def _build_monotone_constraints(feat_cols: list) -> tuple:
    """Return a tuple of {-1, 0, 1} aligned to feat_cols index order, ready
    to pass into XGBClassifier(monotone_constraints=...).
    """
    return tuple(_MONOTONE_SIGNS.get(c, 0) for c in feat_cols)


def _load_split(data_dir: Path, name: str) -> pd.DataFrame:
    path = data_dir / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run scripts/materialize_match_features.py first")
    return pd.read_parquet(path)


def _auto_numeric_features(df: pd.DataFrame) -> list:
    """Every column not in METADATA_COLS and not a CATEGORICAL string.
    Order is parquet column order, deterministic across train/val/test
    by virtue of materializer using the same record schema for all.
    """
    return [c for c in df.columns
            if c not in METADATA_COLS and c not in CATEGORICAL_FEATURES]


def _fit_encoders(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame):
    encoders = {}
    for col in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        all_vals = pd.concat([train[col].astype(str),
                              val[col].astype(str),
                              test[col].astype(str)]).unique()
        le.fit(all_vals)
        encoders[col] = le
    return encoders


def _apply_encoders(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    df = df.copy()
    for col, le in encoders.items():
        encoded_col = f"{col}_id_encoded" if col == "venue" else f"{col}_encoded"
        df[encoded_col] = le.transform(df[col].astype(str))
    return df


def _feature_columns(numeric: list, encoders: dict) -> list:
    cols = list(numeric)
    if "venue" in encoders:
        cols.append("venue_id_encoded")
    if "competition_tier" in encoders:
        cols.append("competition_tier_encoded")
    return cols


def train_model(args) -> tuple:
    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    train = _load_split(data_dir, "train")
    val = _load_split(data_dir, "validation")
    test = _load_split(data_dir, "test")

    print(f"  train: {len(train):,}   val: {len(val):,}   test: {len(test):,}")

    # Detect numeric features BEFORE encoders run, so the encoded columns
    # don't double-count.
    numeric = _auto_numeric_features(train)

    encoders = _fit_encoders(train, val, test)
    train = _apply_encoders(train, encoders)
    val = _apply_encoders(val, encoders)
    test = _apply_encoders(test, encoders)

    feat_cols = _feature_columns(numeric, encoders)

    if args.drop_features:
        drop_subs = [s.strip() for s in args.drop_features.split(",") if s.strip()]
        before = len(feat_cols)
        feat_cols = [c for c in feat_cols if not any(s in c for s in drop_subs)]
        print(f"  drop_features: {drop_subs}  →  {before} → {len(feat_cols)} features")

    print(f"  features ({len(feat_cols)}): {feat_cols}")

    monotone = _build_monotone_constraints(feat_cols) if args.monotone else None
    if monotone is not None:
        n_constrained = sum(1 for s in monotone if s != 0)
        print(f"  monotone constraints: {n_constrained}/{len(feat_cols)} features"
              f" constrained ({sum(1 for s in monotone if s == 1)} +1, "
              f"{sum(1 for s in monotone if s == -1)} -1)")

    X_train, y_train = train[feat_cols], train["team1_wins"]
    X_val, y_val = val[feat_cols], val["team1_wins"]
    X_test, y_test = test[feat_cols], test["team1_wins"]

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        early_stopping_rounds=args.early_stopping_rounds,
        monotone_constraints=monotone,
        random_state=29,
    )

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)

    val_proba = model.predict_proba(X_val)[:, 1]
    test_proba = model.predict_proba(X_test)[:, 1]

    val_ll = log_loss(y_val, val_proba)
    test_ll = log_loss(y_test, test_proba)
    val_brier = brier_score_loss(y_val, val_proba)
    test_brier = brier_score_loss(y_test, test_proba)

    print(f"\n  val LL  = {val_ll:.4f}   val Brier  = {val_brier:.4f}")
    print(f"  test LL = {test_ll:.4f}   test Brier = {test_brier:.4f}")
    print(f"  baseline (predict 0.5 always): LL={np.log(2):.4f}, Brier=0.25")

    print("\n  feature importances:")
    fi = pd.DataFrame({
        "feature": feat_cols,
        "gain": model.feature_importances_,
    }).sort_values("gain", ascending=False)
    for _, row in fi.iterrows():
        print(f"    {row['feature']:30s} {row['gain']:.4f}")

    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "model.pkl")
    joblib.dump(encoders, model_dir / "encoders.pkl")
    with open(model_dir / "feature_columns.txt", "w") as f:
        for c in feat_cols:
            f.write(c + "\n")
    with open(model_dir / "train_metrics.json", "w") as f:
        json.dump({
            "val_log_loss": float(val_ll),
            "val_brier": float(val_brier),
            "test_log_loss": float(test_ll),
            "test_brier": float(test_brier),
            "n_train": int(len(X_train)),
            "n_val": int(len(X_val)),
            "n_test": int(len(X_test)),
            "feature_importances": {
                feat: float(imp)
                for feat, imp in zip(feat_cols, model.feature_importances_)
            },
        }, f, indent=2)
    print(f"\n  saved → {model_dir}")
    return model, encoders, feat_cols


def predict_test(args, model=None, encoders=None, feat_cols=None) -> Path:
    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    if model is None:
        model = joblib.load(model_dir / "model.pkl")
    if encoders is None:
        encoders = joblib.load(model_dir / "encoders.pkl")
    if feat_cols is None:
        with open(model_dir / "feature_columns.txt") as f:
            feat_cols = [l.strip() for l in f if l.strip()]

    test = _load_split(data_dir, "test")
    test_enc = _apply_encoders(test, encoders)
    proba = model.predict_proba(test_enc[feat_cols])[:, 1]

    predictions = {}
    for (_, row), p in zip(test.iterrows(), proba):
        predictions[row["match_id"]] = {
            "team1": row["team1"],
            "team2": row["team2"],
            "p_team1": float(p),
            "p_team2": float(1.0 - p),
            "team1_wins": int(row["team1_wins"]),
            "match_date": row["match_date"],
        }

    out_path = model_dir / "test_predictions.json"
    with open(out_path, "w") as f:
        json.dump(predictions, f, indent=2)

    # Standalone test LL on the FULL test slice (all 791 matches).
    truth = test["team1_wins"].values
    test_ll = log_loss(truth, proba, labels=[0, 1])
    test_brier = brier_score_loss(truth, proba)
    print(f"\n  standalone test ({len(test)} matches): LL={test_ll:.4f}  "
          f"Brier={test_brier:.4f}")
    print(f"  predictions → {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cmd", choices=["train", "predict-test", "both"],
                    default="both")
    ap.add_argument("--data-dir", type=str,
                    default="data/xgb_match_data_v1",
                    help="Parquet directory; pass data/xgb_match_data_v2 for Phase A2.")
    ap.add_argument("--model-dir", type=str,
                    default="models/xgb_match_v1",
                    help="Output artifact dir.")
    ap.add_argument("--n-estimators", type=int, default=1000)
    ap.add_argument("--max-depth", type=int, default=4)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--subsample", type=float, default=0.8)
    ap.add_argument("--colsample-bytree", type=float, default=0.9)
    ap.add_argument("--reg-alpha", type=float, default=0.1)
    ap.add_argument("--reg-lambda", type=float, default=1.0)
    ap.add_argument("--early-stopping-rounds", type=int, default=30)
    ap.add_argument("--monotone", action="store_true",
                    help="Apply per-feature monotone constraints from "
                    "_MONOTONE_SIGNS. Off by default for back-compat with "
                    "xgb_match_v2_clean baseline; on for xgb_match_v3_baseline+.")
    ap.add_argument("--drop-features", type=str, default=None,
                    help="Comma-separated substrings; any feature whose name "
                    "contains ANY listed substring is excluded from training. "
                    "Used for drop-one ablation. Examples: "
                    "'top6_p,top6_pw' drops the M2 batter outcome-dist set; "
                    "'bowlers_p,pw_bowling_diff,p4_bowling_diff,p6_bowling_diff' "
                    "drops the M2 bowler set; 'venue_p4,venue_p6,venue_pw' "
                    "drops M2 venue.")
    ap.add_argument("--config-json", type=str, default=None,
                    help="JSON config from run_experiment (overrides CLI hyperparameters)")
    args = ap.parse_args()

    if args.config_json:
        cfg = json.loads(args.config_json)
        hp = cfg.get("model", {}).get("hyperparameters", {})
        for k, v in hp.items():
            attr = k.replace("-", "_")
            if hasattr(args, attr):
                setattr(args, attr, v)

    if args.cmd in ("train", "both"):
        result = train_model(args)
    else:
        result = (None, None, None)

    if args.cmd in ("predict-test", "both"):
        predict_test(args, *result)


if __name__ == "__main__":
    main()
