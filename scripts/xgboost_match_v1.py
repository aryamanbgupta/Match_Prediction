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
                    models/xgb_match_v1/test_predictions.json keyed by the
                    stable Cricsheet primary match ID.
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

from identity_maps import assert_venue_alias_contract, venue_alias_contract
from elo_update import (
    BASELINE_ELO_UPDATE_VERSION,
    assert_elo_update_version,
    elo_update_contract,
    resolve_elo_update_version,
)
from match_identity import (
    MATCH_IDENTITY_VERSION,
    identity_contract,
    legacy_identity_contract,
    resolve_match_identity,
)

ROOT = Path(__file__).resolve().parent.parent

# Columns excluded from the feature set. Includes metadata (match_id,
# team names, dates, target) and the categorical raw strings that are
# label-encoded separately.
METADATA_COLS = {
    "match_id", "cricsheet_id", "display_match_id",
    "match_identity_version", "elo_update_version", "match_date",
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


# D7 (2026-07-17): team-swap symmetry augmentation. Each train row gets a
# mirrored copy with the two teams exchanged and the label flipped, enforcing
# P(team1 wins | A, B) = 1 - P(team1 wins | B, A) by construction. The
# mapping is exhaustive over the v2_clean schema; _swap_frame refuses to run
# on any column it cannot classify. h2h uses a Beta(1,1) prior (k=2 -> 0.5),
# so 1-x is its exact mirror; no-result/tie matches are dropped at
# materialization, so the label flip is always valid.
_SWAP_PAIRS = [
    ("team1_batting_elo", "team2_batting_elo"),
    ("team1_bowling_elo", "team2_bowling_elo"),
    ("team1_batting_avg", "team2_batting_avg"),
    ("team1_batting_sr", "team2_batting_sr"),
    ("team1_bowling_avg", "team2_bowling_avg"),
    ("team1_bowling_econ", "team2_bowling_econ"),
    ("team1_win_rate_last_10", "team2_win_rate_last_10"),
    ("team1_lhb_count", "team2_lhb_count"),
    ("team1_pace_count", "team2_pace_count"),
    ("team1_spinner_count", "team2_spinner_count"),
    ("is_team1_home", "is_team2_home"),
    ("team1_top6_batting_elo_avg", "team2_top6_batting_elo_avg"),
    ("team1_bottom5_bowling_elo_avg", "team2_bottom5_bowling_elo_avg"),
    ("team1", "team2"),
    # D12: M2 outcome-dist expected columns (verified exact t1/t2 mirrors on
    # the m3_unfrozen parquet; absent from v2_clean, so no-ops there).
    ("team1_top6_p4_expected", "team2_top6_p4_expected"),
    ("team1_top6_p6_expected", "team2_top6_p6_expected"),
    ("team1_top6_pw_expected", "team2_top6_pw_expected"),
    ("team1_bowlers_p4_expected", "team2_bowlers_p4_expected"),
    ("team1_bowlers_p6_expected", "team2_bowlers_p6_expected"),
    ("team1_bowlers_pw_expected", "team2_bowlers_pw_expected"),
    # I12: full-superset frame columns (M3/M4/M5 feature groups that the
    # trimmed i7 frame omits). All diffs below verified exactly t1 - t2 on
    # the w1 train parquet (max |err| = 0).
    ("team1_top6_bat_elo_max", "team2_top6_bat_elo_max"),
    ("team1_top6_bat_elo_spread", "team2_top6_bat_elo_spread"),
    ("team1_bowl_elo_max", "team2_bowl_elo_max"),
    ("team1_bowl_elo_top2", "team2_bowl_elo_top2"),
    ("team1_win_rate_last_60d", "team2_win_rate_last_60d"),
    ("team1_n_matches_last_60d", "team2_n_matches_last_60d"),
    ("team1_competition_win_rate", "team2_competition_win_rate"),
    ("team1_competition_n_matches", "team2_competition_n_matches"),
    ("days_since_team1_last_match", "days_since_team2_last_match"),
    ("is_team1_back_to_back", "is_team2_back_to_back"),
    ("team1_top6_batting_avg_recent", "team2_top6_batting_avg_recent"),
    ("team1_top6_batting_sr_recent", "team2_top6_batting_sr_recent"),
    ("team1_bowlers_avg_recent", "team2_bowlers_avg_recent"),
    ("team1_bowlers_econ_recent", "team2_bowlers_econ_recent"),
    ("team1_n_inform_batters", "team2_n_inform_batters"),
    ("team1_n_outofform_batters", "team2_n_outofform_batters"),
    ("team1_top6_avg_vs_opp_shrunk", "team2_top6_avg_vs_opp_shrunk"),
    ("team1_top6_sr_vs_opp_shrunk", "team2_top6_sr_vs_opp_shrunk"),
    ("team1_h2h_balls_total", "team2_h2h_balls_total"),
]
_SWAP_NEGATE = [
    "elo_diff_batting", "elo_diff_bowling", "batting_avg_diff",
    "bowling_econ_diff", "win_rate_diff", "top6_batting_elo_diff",
    "bottom5_bowling_elo_diff",
    # D12: M2 diffs, all verified exactly t1 - t2 on the m3_unfrozen parquet.
    "p4_batting_diff", "p6_batting_diff", "pw_batting_diff",
    "p4_bowling_diff", "p6_bowling_diff", "pw_bowling_diff",
    # I12: superset diffs, verified exactly t1 - t2 on the w1 train parquet.
    "top6_bat_elo_max_diff", "top6_bat_elo_spread_diff",
    "bowl_elo_max_diff", "bowl_elo_top2_diff",
    "win_rate_last_60d_diff", "competition_win_rate_diff",
    "days_since_diff", "batting_avg_recent_diff", "batting_sr_recent_diff",
    "bowling_avg_recent_diff", "bowling_econ_recent_diff",
    "inform_batters_diff", "outofform_batters_diff",
    "avg_vs_opp_diff", "sr_vs_opp_diff",
]
_SWAP_ONE_MINUS = [
    "h2h_team1_win_rate_shrunk", "toss_winner_is_team1",
    "team1_batting_first", "team1_wins",
]
_SWAP_INVARIANT = [
    "match_id", "cricsheet_id", "display_match_id",
    "match_identity_version", "elo_update_version",
    "match_date", "venue", "competition_tier",
    "venue_avg_score", "venue_chase_win_pct", "venue_dot_pct",
    "venue_boundary_pct", "is_international", "toss_decision_bat",
    "h2h_n_meetings",
    # D12: venue outcome-dist rates (M2 venue-only; properties of the venue,
    # team-order invariant).
    "venue_p4", "venue_p6", "venue_pw",
    # I12: calendar/conditions columns (fixture properties, team-order
    # invariant).
    "month_of_year", "day_of_week", "is_dew_prone_month",
]


def _swap_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return the team-swapped mirror of df (same column order/dtypes)."""
    covered = set(_SWAP_INVARIANT) | set(_SWAP_NEGATE) | set(_SWAP_ONE_MINUS)
    for a, b in _SWAP_PAIRS:
        covered |= {a, b}
    unclassified = [c for c in df.columns if c not in covered]
    if unclassified:
        raise ValueError(
            f"swap-augment: unclassified columns {unclassified} — extend the "
            "_SWAP_* mapping before augmenting this schema")
    sw = df.copy()
    for a, b in _SWAP_PAIRS:
        if a in df.columns and b in df.columns:
            sw[a] = df[b].to_numpy()
            sw[b] = df[a].to_numpy()
    for c in _SWAP_NEGATE:
        if c in df.columns:
            sw[c] = -df[c]
    for c in _SWAP_ONE_MINUS:
        if c in df.columns:
            sw[c] = 1 - df[c]
    return sw


def _swap_augment_train(train: pd.DataFrame) -> pd.DataFrame:
    """Append the mirrored copy of every train row (label flipped)."""
    sw = _swap_frame(train)
    back = _swap_frame(sw)
    num_cols = [c for c in train.columns
                if pd.api.types.is_numeric_dtype(train[c])]
    obj_cols = [c for c in train.columns if c not in num_cols]
    if not np.allclose(back[num_cols].to_numpy(dtype=float),
                       train[num_cols].to_numpy(dtype=float),
                       rtol=0, atol=1e-12):
        raise AssertionError("swap involution failed on numeric columns")
    if not back[obj_cols].equals(train[obj_cols]):
        raise AssertionError("swap involution failed on non-numeric columns")
    return pd.concat([train, sw], ignore_index=True)


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
    identity_path = data_dir / "venue_identity.json"
    if not identity_path.exists():
        raise RuntimeError(
            f"{identity_path} is missing; rematerialize match features"
        )
    assert_venue_alias_contract(
        json.loads(identity_path.read_text()),
        context="match training parquet",
    )
    train = _load_split(data_dir, "train")
    val = _load_split(data_dir, "validation")
    test = _load_split(data_dir, "test")
    match_identity_path = data_dir / "match_identity.json"
    if match_identity_path.exists():
        match_identity = json.loads(match_identity_path.read_text())
        if match_identity != identity_contract():
            raise RuntimeError(
                f"{match_identity_path} has an unsupported match identity "
                "contract"
            )
        for split_name, frame in (
            ("train", train),
            ("validation", val),
            ("test", test),
        ):
            for row in frame[
                [
                    "match_id",
                    "cricsheet_id",
                    "display_match_id",
                    "match_identity_version",
                ]
            ].to_dict("records"):
                resolved = resolve_match_identity(row)
                if resolved.version != MATCH_IDENTITY_VERSION:
                    raise RuntimeError(
                        f"{split_name} contains a legacy match identity "
                        "under a Cricsheet-primary sidecar"
                    )
    else:
        # Existing frozen materializations remain trainable for replay, but
        # their model artifact is explicitly marked legacy.
        match_identity = legacy_identity_contract()
    elo_update_path = data_dir / "elo_update.json"
    if elo_update_path.exists():
        elo_update_metadata = json.loads(elo_update_path.read_text())
    else:
        elo_update_metadata = elo_update_contract(
            BASELINE_ELO_UPDATE_VERSION
        )
    elo_update_version = resolve_elo_update_version(elo_update_metadata)
    assert_elo_update_version(
        elo_update_metadata,
        expected=elo_update_version,
        context="match training parquet",
    )
    if "elo_update_version" in train.columns:
        for split_name, frame in (
            ("train", train),
            ("validation", val),
            ("test", test),
        ):
            versions = set(frame["elo_update_version"].astype(str))
            if versions != {elo_update_version}:
                raise RuntimeError(
                    f"{split_name} ELO update versions {sorted(versions)} "
                    f"do not match sidecar {elo_update_version!r}"
                )
    elif elo_update_version != BASELINE_ELO_UPDATE_VERSION:
        raise RuntimeError(
            "provisional-ELO match parquet is missing row provenance"
        )

    print(f"  train: {len(train):,}   val: {len(val):,}   test: {len(test):,}")

    if args.swap_augment:
        train = _swap_augment_train(train)
        print(f"  swap-augment: train doubled → {len(train):,} rows "
              f"(base rate {train['team1_wins'].mean():.4f}); "
              "val/test untouched")

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
        random_state=args.seed,
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
            "seed": int(args.seed),
            "feature_importances": {
                feat: float(imp)
                for feat, imp in zip(feat_cols, model.feature_importances_)
            },
            "venue_identity": venue_alias_contract(),
            "match_identity": match_identity,
            "elo_update": elo_update_contract(elo_update_version),
        }, f, indent=2)
    with open(model_dir / "venue_identity.json", "w") as f:
        json.dump(venue_alias_contract(), f, indent=2)
    with open(model_dir / "match_identity.json", "w") as f:
        json.dump(match_identity, f, indent=2)
    with open(model_dir / "elo_update.json", "w") as f:
        json.dump(elo_update_contract(elo_update_version), f, indent=2)
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
        identity = resolve_match_identity(row.to_dict())
        if not identity.cricsheet_id:
            raise ValueError(
                "test prediction requires a Cricsheet ID; rematerialize "
                "before writing a primary-keyed artifact"
            )
        if identity.primary_id in predictions:
            raise ValueError(
                "duplicate test prediction primary match ID: "
                f"{identity.primary_id}"
            )
        predictions[identity.primary_id] = {
            "match_id": identity.primary_id,
            "cricsheet_id": identity.cricsheet_id,
            "display_match_id": identity.display_id,
            "match_identity_version": MATCH_IDENTITY_VERSION,
            "elo_update_version": str(
                row.get(
                    "elo_update_version",
                    BASELINE_ELO_UPDATE_VERSION,
                )
            ),
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
    ap.add_argument("--seed", type=int, default=29)
    ap.add_argument("--swap-augment", action="store_true",
                    help="D7: append a team-swapped mirror of every TRAIN row "
                    "(label flipped) to enforce antisymmetry. Val/test are "
                    "never augmented.")
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
