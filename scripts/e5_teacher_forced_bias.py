"""E5 — Decompose the sim's PP-total / bowler-wicket overshoot.

The prop backtest (and E2's fair-baseline audit) shows the v7 sim
systematically over-states powerplay totals and per-bowler wicket
counts. Two candidate mechanisms:

  (A) per-ball MODEL bias — the XGBoost itself over-predicts
      boundaries/wickets in those contexts. Visible under TEACHER
      FORCING: score the model on real test deliveries and compare
      predicted class rates to actual outcomes, per phase.
  (B) SIM dynamics divergence — the per-ball model is calibrated on
      real states, but Monte-Carlo rollouts drift into unrealistic
      states (wrong bowlers in the powerplay, unrealistic batter/
      non-striker pairs, compounding survivorship of hitters).

v7's feature list already contains the chase/pressure/momentum groups
(run_rate_required, pressure_cooker_index, last-N-ball momentum), so
"add intent features" is NOT automatically the fix — this diagnostic
decides whether feature work (A) or sim-engine work (B) is the right
next move.

Method: teacher-forced scoring of the v7 booster on the ball-level test
split (real deliveries, real features — the exact distribution the model
was trained on), sliced by phase × innings, plus per-decile bowler_pw
calibration for the wicket channel.

Usage:
    uv run python scripts/e5_teacher_forced_bias.py
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
MODEL = REPO / "models" / "xgb_v3"
BALL_DIR = REPO / "data" / "xgb_data_v3"
OUT = REPO / "reports" / "e5_teacher_forced_bias.md"

RUNS_BY_CLASS = np.array([0, 1, 2, 4, 6, 0], dtype=float)
CLASS_NAMES = ["dot", "one", "two", "four", "six", "wicket"]


def load_scored_test():
    """Score the test split twice: with training-time venue codes and with
    the sim's actual input (venue_encoded = 0 for every ball — the XGB sim
    wrapper never sets it; `_feat_buf` defaults missing keys to 0).
    """
    model = joblib.load(MODEL / "xgboost_model_v3.pkl")
    feat = [l.strip() for l in open(MODEL / "feature_columns_v3.txt")
            if l.strip()]
    encs = {
        "batter_id": joblib.load(MODEL / "batter_encoder_v3.pkl"),
        "bowler_id": joblib.load(MODEL / "bowler_encoder_v3.pkl"),
        "matchup_type": joblib.load(MODEL / "matchup_encoder_v3.pkl"),
    }
    df = pd.read_parquet(BALL_DIR / "cricket_data_v3_test.parquet")
    for raw, enc_name in (("batter_id", "batter_encoded"),
                          ("bowler_id", "bowler_encoded"),
                          ("matchup_type", "matchup_type_encoded")):
        if enc_name in feat:
            lut = {str(c): i for i, c in enumerate(encs[raw].classes_)}
            df[enc_name] = df[raw].astype(str).map(lut).fillna(-1).astype(int)
    # training-time venue codes: xgboost_v2.py fits LabelEncoder over the
    # union of all three splits' venue strings -> codes = alphabetical rank.
    venues = set()
    for sp in ("train", "validation", "test"):
        venues |= set(pd.read_parquet(BALL_DIR / f"cricket_data_v3_{sp}.parquet",
                                      columns=["venue"])["venue"].astype(str).unique())
    vlut = {v: i for i, v in enumerate(sorted(venues))}
    df["venue_encoded"] = df["venue"].astype(str).map(vlut).astype(int)

    proba_real = model.predict_proba(df[feat])
    df0 = df.copy()
    df0["venue_encoded"] = 0
    proba_sim = model.predict_proba(df0[feat])

    from calibration import PriorCorrectionCalibrator
    corr = PriorCorrectionCalibrator()
    proba_corr = corr.calibrate_probs(proba_sim)  # sim input + correction
    return df, proba_real, proba_sim, proba_corr


def main():
    df, proba_real, proba_sim, proba_corr = load_scored_test()
    # ball_outcome: runs {0,1,2,4,6}, -1 = wicket (trainer maps -1 -> class
    # 7; predict_proba column order is sorted([0,1,2,4,6,7]) so col 5 = wkt)
    raw = df["ball_outcome"].astype(int).values
    cls_map = {0: 0, 1: 1, 2: 2, 4: 3, 6: 4, -1: 5}
    y = np.array([cls_map[v] for v in raw])
    print(f"test balls scored: {len(df):,}")

    lines = ["# E5 — Teacher-forced per-ball bias audit of v7 (test split)",
             "",
             f"n = {len(df):,} real deliveries (2025-07-01 → 2026-04-16), "
             "scored with the production v7 booster on REAL features — any "
             "bias here is model bias (mechanism A); calibrated rows point "
             "to sim-dynamics divergence (mechanism B) as the overshoot "
             "source. Each table is computed twice: `real` = training-time "
             "venue codes; `sim` = venue_encoded=0 for all balls, which is "
             "what the sim wrapper actually feeds (bug: XGBoostModelV2 "
             "never sets venue_encoded; missing keys default to 0).", ""]

    def slice_table(title, masks):
        rows = [f"## {title}", "",
                "| slice | input | n | " +
                " | ".join(f"Δp({c})" for c in CLASS_NAMES) +
                " | pred runs/ball | actual | Δ |", "|---|---|" + "---:|" * 9]
        for name, m in masks:
            if m.sum() == 0:
                continue
            for label, proba in (("real", proba_real), ("sim", proba_sim),
                                 ("sim+corr", proba_corr)):
                p = proba[m]
                a = np.bincount(y[m], minlength=6) / m.sum()
                dp = p.mean(0) - a
                pr = float((p @ RUNS_BY_CLASS).mean())
                ar = float(RUNS_BY_CLASS[y[m]].mean())
                rows.append(
                    f"| {name} | {label} | {m.sum():,} | " +
                    " | ".join(f"{d:+.4f}" for d in dp) +
                    f" | {pr:.4f} | {ar:.4f} | {pr - ar:+.4f} |")
        return rows + [""]

    inn = df["inning_idx"].astype(int).values
    pp = df["is_powerplay"].astype(bool).values
    mid = df["is_middle_overs"].astype(bool).values
    death = df["is_death_overs"].astype(bool).values

    lines += slice_table("Phase × innings", [
        ("PP inn1", pp & (inn == 1)), ("PP inn2", pp & (inn == 2)),
        ("mid inn1", mid & (inn == 1)), ("mid inn2", mid & (inn == 2)),
        ("death inn1", death & (inn == 1)), ("death inn2", death & (inn == 2)),
        ("ALL", np.ones(len(df), bool)),
    ])

    # wicket channel by bowler quality decile
    lines += ["## Wicket probability by bowler_pw decile", "",
              "| bowler_pw decile | n | pred real | pred sim | pred sim+corr "
              "| actual | Δ real | Δ sim+corr |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    bpw = df["bowler_pw"].values
    deciles = pd.qcut(bpw, 10, labels=False, duplicates="drop")
    for d in range(int(np.nanmax(deciles)) + 1):
        m = deciles == d
        pr = proba_real[m, 5].mean()
        ps = proba_sim[m, 5].mean()
        pc = proba_corr[m, 5].mean()
        pa = (y[m] == 5).mean()
        lines.append(f"| {d} | {m.sum():,} | {pr:.4f} | {ps:.4f} | {pc:.4f} "
                     f"| {pa:.4f} | {pr - pa:+.4f} | {pc - pa:+.4f} |")
    lines.append("")

    OUT.write_text("\n".join(lines))
    print(f"report -> {OUT}")
    # console summary
    for ln in lines:
        if ln.startswith("|"):
            print(ln)


if __name__ == "__main__":
    main()
