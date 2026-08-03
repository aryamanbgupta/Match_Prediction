"""D16 — teacher-forced marginal audit of an i7-frame ball model (GATE 1(a)).

WRITTEN AND COMMITTED BEFORE EITHER D16 MODEL EXISTED.

Copy-adapted from `scripts/auto/d6_marginal_audit.py` (recovered from 5fb16bb),
which was itself copy-adapted from `scripts/e5_teacher_forced_bias.py` (that
file hardcodes `models/xgb_v3` and is NOT edited by D16). Every legacy `_v3`
path in the D6 original is re-pointed at the I7 identity frame here:

    BALL_DIR   data/xgb_data_v3  ->  data/xgb_data_i7
    PROD_DIR   models/xgb_v3     ->  models/xgb_i7   (context arm only)
    suffix     v3                ->  i7

The question this answers is narrower than E5's: does a booster trained WITHOUT
`balanced` class weights reproduce the real per-ball outcome marginals on
held-out data, i.e. does the structural fix do what the deployed vector
calibrator does by construction?

Pre-committed tolerance (research/handoff/D16/plan.md, from
`reports/e5_class_weight_fix.md`):

    PASS iff  |P_hat(wicket) - actual| <= 0.005  AND
              |runs/ball_hat - runs/ball_actual| <= 0.05

Legacy context from that report (production v7 on the legacy frame, sim input
distribution): raw balanced booster +0.0647 / +0.3829; deployed v1 vector
calibrator -0.0016 / +0.0237 on its venue_zero fit path. Those are cross-frame
and are NOT the bar — the bar is the absolute tolerance above.

Two input distributions are scored because they differ in what the sim feeds:

  venue_on   real training-time venue codes from the arm's own
             `venue_encoder_i7.pkl`. This is the i7 sim path
             (`prop_backtest.py` auto-detects the venue encoder next to the
             model, and `TestMatchLoader` canonicalizes venue at state build).
             PRIMARY — the plan's `"pass"` flag is this one.
  venue_zero venue_encoded = 0 for every ball. Pre-B1 sim input; context only.

The reference/context arm is `models/xgb_i7` — the archived balanced-weights
booster on the same frame — scored with ITS OWN encoders (never the audited
arm's), so a determinism drift in encoders cannot silently corrupt it. It is
READ-ONLY: this script never writes into `models/xgb_i7`.

Optionally `--calibrator <pkl>` scores the audited model's probabilities after
a `calibrate_probs` transform (used to record the control + d16-vector
marginals as context).

Run:
    uv run python scripts/auto/d16_marginal_audit.py \
        --model-dir models/auto/d16/noweights --suffix i7
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

BALL_DIR = REPO / "data" / "xgb_data_i7"
PROD_DIR = REPO / "models" / "xgb_i7"

RUNS_BY_CLASS = np.array([0, 1, 2, 4, 6, 0], dtype=float)
CLASS_NAMES = ["dot", "one", "two", "four", "six", "wicket"]
WICKET_COL = 5

TOL_WICKET = 0.005
TOL_RUNS = 0.05


def _feature_list(model_dir: Path, suffix: str) -> list:
    return [ln.strip() for ln
            in (model_dir / f"feature_columns_{suffix}.txt").read_text().splitlines()
            if ln.strip()]


def encode_frame(df: pd.DataFrame, model_dir: Path, suffix: str,
                 feat: list) -> pd.DataFrame:
    """Add training-time encoded columns using `model_dir`'s own encoders."""
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
        enc_path = model_dir / enc_file
        if not enc_path.exists():
            raise FileNotFoundError(f"missing encoder {enc_path}")
        le = joblib.load(enc_path)
        lut = {str(c): i for i, c in enumerate(le.classes_)}
        out[enc_name] = out[raw].astype(str).map(lut).fillna(-1).astype(int)
        n_unk = int((out[enc_name] == -1).sum())
        if n_unk:
            print(f"    {enc_name}: {n_unk:,} rows unseen by encoder -> -1")
    return out


def load_test_frame(suffix: str):
    """Raw test parquet + integer labels (encoding is per-arm, applied later)."""
    df = pd.read_parquet(BALL_DIR / f"cricket_data_{suffix}_test.parquet")
    cls_map = {0: 0, 1: 1, 2: 2, 4: 3, 6: 4, -1: 5}
    outcomes = df["ball_outcome"].astype(int)
    bad = sorted(set(outcomes.unique()) - set(cls_map))
    if bad:
        raise ValueError(f"unexpected ball_outcome values in {suffix} test: {bad}")
    y = np.array([cls_map[v] for v in outcomes.values])
    return df, y


def multiclass_ll(proba: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(proba[np.arange(len(y)), y], 1e-15, 1.0)
    return float(-np.log(p).mean())


def score_arm(proba: np.ndarray, y: np.ndarray, label: str) -> dict:
    pred_marg = proba.mean(axis=0)
    actual_marg = np.bincount(y, minlength=6) / len(y)
    pred_runs = float((proba @ RUNS_BY_CLASS).mean())
    actual_runs = float(RUNS_BY_CLASS[y].mean())
    d_wkt = float(pred_marg[WICKET_COL] - actual_marg[WICKET_COL])
    d_runs = pred_runs - actual_runs
    passed = abs(d_wkt) <= TOL_WICKET and abs(d_runs) <= TOL_RUNS
    return {
        "label": label,
        "n_balls": int(len(y)),
        "per_class": [
            {"class": CLASS_NAMES[c],
             "pred": float(pred_marg[c]),
             "actual": float(actual_marg[c]),
             "delta": float(pred_marg[c] - actual_marg[c])}
            for c in range(6)
        ],
        "pred_wicket": float(pred_marg[WICKET_COL]),
        "actual_wicket": float(actual_marg[WICKET_COL]),
        "delta_wicket": d_wkt,
        "pred_runs_per_ball": pred_runs,
        "actual_runs_per_ball": actual_runs,
        "delta_runs_per_ball": d_runs,
        "test_multiclass_logloss": multiclass_ll(proba, y),
        "abs_delta_wicket_le_tol": bool(abs(d_wkt) <= TOL_WICKET),
        "abs_delta_runs_le_tol": bool(abs(d_runs) <= TOL_RUNS),
        "pass": bool(passed),
    }


def print_arm(a: dict) -> None:
    print(f"\n--- {a['label']}  (n = {a['n_balls']:,} balls) ---")
    print(f"| {'class':<8}| {'pred':>9} | {'actual':>9} | {'delta':>9} |")
    print(f"|{'-' * 9}|{'-' * 11}|{'-' * 11}|{'-' * 11}|")
    for row in a["per_class"]:
        print(f"| {row['class']:<8}| {row['pred']:>9.5f} | "
              f"{row['actual']:>9.5f} | {row['delta']:>+9.5f} |")
    print(f"  runs/ball   pred {a['pred_runs_per_ball']:.4f}  "
          f"actual {a['actual_runs_per_ball']:.4f}  "
          f"delta {a['delta_runs_per_ball']:+.4f}  "
          f"(tol {TOL_RUNS}) -> "
          f"{'ok' if a['abs_delta_runs_le_tol'] else 'FAIL'}")
    print(f"  P(wicket)   pred {a['pred_wicket']:.5f}  "
          f"actual {a['actual_wicket']:.5f}  "
          f"delta {a['delta_wicket']:+.5f}  "
          f"(tol {TOL_WICKET}) -> "
          f"{'ok' if a['abs_delta_wicket_le_tol'] else 'FAIL'}")
    print(f"  test multiclass LL {a['test_multiclass_logloss']:.4f}")
    print(f"  ARM VERDICT: {'PASS' if a['pass'] else 'FAIL'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(REPO / "models/auto/d16/noweights"))
    ap.add_argument("--suffix", default="i7")
    ap.add_argument("--out", default=None,
                    help="default: <model-dir>/marginal_audit.json")
    ap.add_argument("--calibrator", default=None,
                    help="optional calibrator pkl with .calibrate_probs; "
                         "scored as extra context arms")
    ap.add_argument("--skip-reference", action="store_true")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    out_path = Path(args.out) if args.out else model_dir / "marginal_audit.json"

    df_raw, y = load_test_frame(args.suffix)
    feat = _feature_list(model_dir, args.suffix)
    print(f"model dir: {model_dir}")
    print(f"test parquet: "
          f"{BALL_DIR / f'cricket_data_{args.suffix}_test.parquet'}")
    print(f"test balls scored: {len(df_raw):,}  features: {len(feat)}")

    print("  encoding with the audited arm's own encoders ...")
    df = encode_frame(df_raw, model_dir, args.suffix, feat)
    X_on = df[feat]
    df0 = df.copy()
    df0["venue_encoded"] = 0
    X_zero = df0[feat]

    model = joblib.load(model_dir / f"xgboost_model_{args.suffix}.pkl")
    p_on = model.predict_proba(X_on)
    p_zero = model.predict_proba(X_zero)
    arms = {
        "venue_on": score_arm(p_on, y,
                              f"{model_dir.name} raw / venue_on (PRIMARY)"),
        "venue_zero": score_arm(p_zero, y, f"{model_dir.name} raw / venue_zero"),
    }
    if args.calibrator:
        calib = joblib.load(args.calibrator)
        cal_name = Path(args.calibrator).name
        arms["venue_on_calibrated"] = score_arm(
            calib.calibrate_probs(p_on), y,
            f"{model_dir.name} + {cal_name} / venue_on (CONTEXT)")
        arms["venue_zero_calibrated"] = score_arm(
            calib.calibrate_probs(p_zero), y,
            f"{model_dir.name} + {cal_name} / venue_zero (CONTEXT)")
    for a in arms.values():
        print_arm(a)

    reference = {}
    if not args.skip_reference:
        print("\n" + "=" * 72)
        print(f"CONTEXT — archived {PROD_DIR} (balanced weights, same i7 "
              f"frame), raw; scored with ITS OWN encoders")
        print("=" * 72)
        try:
            pfeat = _feature_list(PROD_DIR, args.suffix)
            pdf = encode_frame(df_raw, PROD_DIR, args.suffix, pfeat)
            pdf0 = pdf.copy()
            pdf0["venue_encoded"] = 0
            prod = joblib.load(PROD_DIR / f"xgboost_model_{args.suffix}.pkl")
            reference["prod_raw_venue_on"] = score_arm(
                prod.predict_proba(pdf[pfeat]), y,
                f"{PROD_DIR.name} raw / venue_on")
            reference["prod_raw_venue_zero"] = score_arm(
                prod.predict_proba(pdf0[pfeat]), y,
                f"{PROD_DIR.name} raw / venue_zero")
            for a in reference.values():
                print_arm(a)
        except Exception as exc:  # pragma: no cover
            print(f"  reference arms failed: {type(exc).__name__}: {exc}")
            reference = {"error": f"{type(exc).__name__}: {exc}"}

    payload = {
        "model_dir": str(model_dir),
        "test_parquet": str(BALL_DIR / f"cricket_data_{args.suffix}_test.parquet"),
        "n_balls": int(len(df_raw)),
        "calibrator": args.calibrator,
        "tolerance": {"abs_delta_wicket": TOL_WICKET,
                      "abs_delta_runs_per_ball": TOL_RUNS},
        "primary_arm": "venue_on",
        "arms": arms,
        "reference": reference,
        "pass": bool(arms["venue_on"]["pass"]),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out_path}")
    print(f"GATE 1(a) marginal audit (primary = venue_on): "
          f"{'PASS' if payload['pass'] else 'FAIL'}")


if __name__ == "__main__":
    main()
