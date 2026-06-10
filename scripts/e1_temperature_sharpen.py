"""E1 — Temperature sharpening for the match-level direct model.

Motivation (reports/reliability_diagnostic.png, 2026-06-07): the
production model `xgb_match_v3_m7_production` is *under-confident* on the
iteration set (calibration slope 1.34-1.75 depending on slice) while its
Brier resolution matches the market's. Platt (slope+intercept) was
rejected at M7 because it over-corrects; but a slope>1 diagnostic implies
the LL-optimal transform is *expansive* — push probabilities away from
0.5, the opposite of what killed ROI in the 2026-03/2026-04 calibration
attempts.

Protocol (strict val-only fitting):
  1. Score validation.parquet (n=525) with the production booster.
  2. Fit each candidate transform on val ONLY:
       - temp:     p' = sigmoid(T * logit(p)), T free (LL-optimal slope)
       - temp_ge1: same, T clamped to >= 1 (pure sharpening, no shrink)
       - platt:    slope + intercept (M1 reference, expected to lose)
       - beta:     3-param beta calibration (asymmetric generalisation)
  3. Apply each to test_predictions.json, run the standard
     blend(w=0) -> reslice pipeline against polymarket odds.
  4. Decision on iteration slices: keep iff >=\$50k LL improves vs raw
     0.6299 (target: < market 0.6267) AND flat-ROI CI does not
     materially regress vs raw +21.90% [+2.28, +43.83].

Variant selection happens on val LL; iteration numbers are the readout.

Usage:
    uv run python scripts/e1_temperature_sharpen.py \
        --model-dir models/xgb_match_v3_m7_production \
        --data-dir data/xgb_match_data_v3_m3_unfrozen \
        --out-root eval_out_e1
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from calibrate_match_predictions import _val_predictions

REPO = Path(__file__).resolve().parent.parent
SIM_ENVELOPE = REPO / "eval_out_phase5_hier" / "hier_all_20260425_165622.json"
ODDS = REPO / "betting_odds_polymarket.json"
EPS = 1e-9


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# ---------------------------------------------------------------- calibrators
class TempScaler:
    """p' = sigmoid(T * logit(p)). T fit by LL-optimal logistic slope."""

    def __init__(self, min_T: float | None = None):
        self.min_T = min_T
        self.T = 1.0

    def fit(self, probs: np.ndarray, truth: np.ndarray):
        lr = LogisticRegression(fit_intercept=False, C=1e6, max_iter=10000)
        lr.fit(_logit(probs).reshape(-1, 1), truth)
        self.T = float(lr.coef_[0][0])
        if self.min_T is not None:
            self.T = max(self.T, self.min_T)
        return self

    def predict(self, probs: np.ndarray) -> np.ndarray:
        return _sigmoid(self.T * _logit(probs))

    def describe(self) -> str:
        return f"T={self.T:.4f}"


class PlattScaler:
    """slope + intercept on logit (M1-style reference)."""

    def __init__(self):
        self.a = 1.0
        self.b = 0.0

    def fit(self, probs: np.ndarray, truth: np.ndarray):
        lr = LogisticRegression(C=1e6, max_iter=10000)
        lr.fit(_logit(probs).reshape(-1, 1), truth)
        self.a = float(lr.coef_[0][0])
        self.b = float(lr.intercept_[0])
        return self

    def predict(self, probs: np.ndarray) -> np.ndarray:
        return _sigmoid(self.a * _logit(probs) + self.b)

    def describe(self) -> str:
        return f"a={self.a:.4f} b={self.b:.4f}"


class BetaScaler:
    """Beta calibration (Kull et al. 2017): logistic on [ln p, -ln(1-p)]."""

    def __init__(self):
        self.lr = None

    def fit(self, probs: np.ndarray, truth: np.ndarray):
        X = self._features(probs)
        self.lr = LogisticRegression(C=1e6, max_iter=10000)
        self.lr.fit(X, truth)
        return self

    @staticmethod
    def _features(probs: np.ndarray) -> np.ndarray:
        p = np.clip(probs, EPS, 1 - EPS)
        return np.column_stack([np.log(p), -np.log(1 - p)])

    def predict(self, probs: np.ndarray) -> np.ndarray:
        return self.lr.predict_proba(self._features(probs))[:, 1]

    def describe(self) -> str:
        a, b = self.lr.coef_[0]
        return f"a={a:.4f} b={b:.4f} c={self.lr.intercept_[0]:.4f}"


# ------------------------------------------------------------------ pipeline
def apply_to_predictions_json(in_path: Path, out_path: Path, calibrator):
    preds = json.load(open(in_path))
    mids = list(preds.keys())
    raw = np.array([preds[m]["p_team1"] for m in mids])
    cal = calibrator.predict(raw)
    out = {}
    for m, p in zip(mids, cal):
        rec = dict(preds[m])
        rec["p_team1_raw"] = rec["p_team1"]
        rec["p_team1"] = float(p)
        out[m] = rec
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_path, "w"), indent=1)


def run_blend_and_reslice(pred_json: Path, tag: str, out_root: Path) -> Path:
    blend_dir = out_root / f"blend_{tag}"
    sliced_dir = out_root / f"sliced_{tag}"
    subprocess.run(
        [sys.executable, str(REPO / "scripts/sim_eval/blend_eval_json.py"),
         "--sim-json", str(SIM_ENVELOPE),
         "--direct-json", str(pred_json),
         "--w", "0.0", "--out-dir", str(blend_dir)],
        check=True, capture_output=True, text=True)
    blended = blend_dir / f"{SIM_ENVELOPE.stem}_w0p00.json"
    subprocess.run(
        [sys.executable, str(REPO / "scripts/sim_eval/reslice_eval_json.py"),
         "--in", str(blended), "--odds", str(ODDS),
         "--out-dir", str(sliced_dir),
         "--min-volume", "0", "--min-volume", "50000",
         "--min-volume", "100000"],
        check=True, capture_output=True, text=True)
    return sliced_dir


def collect_summary(sliced_dir: Path) -> dict:
    rows = {}
    for f in sorted(sliced_dir.glob("*_min_volume_*.json")):
        s = json.load(open(f))["summary"]
        vol = int(s["min_volume"])
        rows[vol] = {
            "n": s["n_matches_evaluated"],
            "ll": s["avg_log_loss"],
            "ll_ci": (s.get("avg_log_loss_ci_low"), s.get("avg_log_loss_ci_high")),
            "roi": s.get("flat_betting_roi_pct"),
            "roi_ci": (s.get("flat_betting_roi_ci_low"), s.get("flat_betting_roi_ci_high")),
        }
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path,
                    default=REPO / "models/xgb_match_v3_m7_production")
    ap.add_argument("--data-dir", type=Path,
                    default=REPO / "data/xgb_match_data_v3_m3_unfrozen")
    ap.add_argument("--out-root", type=Path, default=REPO / "eval_out_e1")
    args = ap.parse_args()

    print("== E1 temperature sharpening ==")
    print(f"model: {args.model_dir.name}  val source: {args.data_dir.name}")
    val_probs, val_truth = _val_predictions(args.model_dir, args.data_dir)
    print(f"val n={len(val_probs)}  raw val LL={log_loss(val_truth, val_probs):.4f}")

    variants = {
        "temp": TempScaler(),
        "temp_ge1": TempScaler(min_T=1.0),
        "platt": PlattScaler(),
        "beta": BetaScaler(),
    }
    fitted = {}
    print("\n-- val fits (fit on val ONLY) --")
    for name, cal in variants.items():
        cal.fit(val_probs, val_truth)
        vll = log_loss(val_truth, np.clip(cal.predict(val_probs), EPS, 1 - EPS))
        fitted[name] = (cal, vll)
        print(f"{name:10s} {cal.describe():30s} val LL {vll:.4f}")

    test_json = args.model_dir / "test_predictions.json"
    args.out_root.mkdir(parents=True, exist_ok=True)

    all_summaries = {}
    # raw baseline through the identical pipeline (same bootstrap conditions)
    sliced = run_blend_and_reslice(test_json, "raw", args.out_root)
    all_summaries["raw"] = collect_summary(sliced)

    for name, (cal, _) in fitted.items():
        pred_out = args.out_root / "preds" / f"{name}_predictions.json"
        apply_to_predictions_json(test_json, pred_out, cal)
        sliced = run_blend_and_reslice(pred_out, name, args.out_root)
        all_summaries[name] = collect_summary(sliced)

    print("\n-- iteration-test sliced results (market LL ref @>=50k: 0.6267) --")
    hdr = f"{'variant':10s} {'slice':>8s} {'n':>4s} {'LL':>8s} {'ROI%':>8s} {'ROI CI':>20s}"
    print(hdr)
    print("-" * len(hdr))
    for name, rows in all_summaries.items():
        for vol in sorted(rows):
            r = rows[vol]
            tag = {0: "all", 50000: ">=50k", 100000: ">=100k"}[vol]
            ci = f"[{r['roi_ci'][0]:+.2f}, {r['roi_ci'][1]:+.2f}]" \
                if r['roi_ci'][0] is not None else "—"
            print(f"{name:10s} {tag:>8s} {r['n']:>4d} {r['ll']:>8.4f} "
                  f"{r['roi']:>+8.2f} {ci:>20s}")

    json.dump(all_summaries, open(args.out_root / "e1_summaries.json", "w"),
              indent=2)
    print(f"\nsummaries -> {args.out_root}/e1_summaries.json")


if __name__ == "__main__":
    main()
