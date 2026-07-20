"""D11 — inference-time symmetrization on the D7 swap-augmented models.

D7's augmentation makes the match model *approximately* antisymmetric but not
exactly (trees on augmented data still fit residual orientation noise).
Averaging the two orientations at predict time,

    p_sym = (p(A,B) + 1 - p(B,A)) / 2,

enforces exact antisymmetry and is a free eval-only variance cut on top of D7
(flagged by D7's own gate note). No retraining — reuses the saved 5-seed
models.

Arms (each = 5 seeds {29,7,13,42,101}, paired sym-vs-raw per seed):
  d7_swap  PRIMARY (gate)  models/auto/d7/swap_seed*  on v2_clean
  d7_base  second arm      models/auto/d7/base_seed*  on v2_clean
           (does symmetrization alone recover part of D7's gain?)
  d12_swap context arm     models/auto/d12/swap_seed* on data/auto/d12
           (production-frame transfer; free per D12's note)

Hard-fail controls before any eval number is read:
  - re-scoring the raw test frame must reproduce the arm's saved
    test_predictions.json p_team1 exactly (max|dp| <= 1e-12) — ties this
    scoring path to the logged D7/D12 numbers (env-drift control);
  - the swapped frame's venue/tier encoded columns must equal the raw
    frame's (team columns don't touch the encoders — verified, not assumed);
  - _swap_frame coverage + involution are hard-checked inside the trainer's
    own _swap_frame/_swap_augment_train (imported, not reimplemented).

PRE-COMMITTED GATE (program.md eval-only floors, written before any result):
  On the PRIMARY arm (d7_swap), mean over the 5 seeds of paired
  (sym - raw) on the >=$50k slice:
    LL improved  iff dLL <= -0.002
    ROI improved iff dROI >= +2.0pp
  BOTH -> LANDED; exactly one -> TABLED; none -> FAILED.
  d7_base / d12_swap arms are report context only and cannot move the verdict.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from xgboost_match_v1 import _apply_encoders, _swap_frame  # noqa: E402

SEEDS = [29, 7, 13, 42, 101]
SIM_ENVELOPE = ROOT / "eval_out/phase5_hier/hier_all_20260425_165622.json"
ODDS = ROOT / "betting_odds_polymarket.json"
OUT = ROOT / "models/auto/d11"

ARMS = {
    "d7_swap": {
        "models": ROOT / "models/auto/d7", "variant": "swap",
        "data": ROOT / "data/xgb_match_data_v2_clean",
        "raw_results": ROOT / "models/auto/d7/d7_results.json",
    },
    "d7_base": {
        "models": ROOT / "models/auto/d7", "variant": "base",
        "data": ROOT / "data/xgb_match_data_v2_clean",
        "raw_results": ROOT / "models/auto/d7/d7_results.json",
    },
    "d12_swap": {
        "models": ROOT / "models/auto/d12", "variant": "swap",
        "data": ROOT / "data/auto/d12",
        "raw_results": ROOT / "models/auto/d12/d12_results.json",
    },
}

LL_FLOOR = -0.002   # mean paired dLL must be <= this to count as improved
ROI_FLOOR = 2.0     # mean paired dROI must be >= this to count as improved


def run(cmd):
    print("  $", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)


def score_seed(arm: str, seed: int) -> dict:
    cfg = ARMS[arm]
    mdir = cfg["models"] / f"{cfg['variant']}_seed{seed}"
    model = joblib.load(mdir / "model.pkl")
    encoders = joblib.load(mdir / "encoders.pkl")
    feat_cols = [l.strip() for l in (mdir / "feature_columns.txt")
                 .read_text().splitlines() if l.strip()]

    test = pd.read_parquet(cfg["data"] / "test.parquet")
    enc_raw = _apply_encoders(test, encoders)
    p_raw = model.predict_proba(enc_raw[feat_cols])[:, 1]

    # Control 1: this scoring path must reproduce the saved predictions.
    # The JSON is keyed by match_id with last-write-wins (test has 791 rows /
    # 782 unique ids), so compare on the last occurrence per id — the rows
    # the JSON actually stores and the eval actually consumes.
    saved = json.loads((mdir / "test_predictions.json").read_text())
    p_saved = np.array([saved[m]["p_team1"] for m in test["match_id"]])
    last = ~test["match_id"].duplicated(keep="last").to_numpy()
    dev = np.max(np.abs(p_raw[last] - p_saved[last]))
    print(f"  [{arm} s{seed}] raw-reproduction max|dp| = {dev:.3e} "
          f"({int(last.sum())} unique-id rows)")
    assert dev <= 1e-12, "scoring path does not reproduce saved predictions"

    # Swapped orientation. _swap_frame hard-fails on unclassified columns.
    sw = _swap_frame(test)
    enc_sw = _apply_encoders(sw, encoders)
    # Control 2: encoded categoricals are orientation-invariant.
    for col in ("venue_id_encoded", "competition_tier_encoded"):
        if col in feat_cols:
            assert (enc_sw[col].to_numpy() == enc_raw[col].to_numpy()).all(), \
                f"{col} changed under swap — encoding not orientation-invariant"
    p_sw = model.predict_proba(enc_sw[feat_cols])[:, 1]

    asym = p_raw - (1.0 - p_sw)          # residual orientation noise
    p_sym = 0.5 * (p_raw + 1.0 - p_sw)   # exact antisymmetry by construction

    odir = OUT / arm / f"seed{seed}"
    odir.mkdir(parents=True, exist_ok=True)
    preds = {}
    for (_, row), p in zip(test.iterrows(), p_sym):
        preds[row["match_id"]] = {
            "team1": row["team1"], "team2": row["team2"],
            "p_team1": float(p), "p_team2": float(1.0 - p),
            "team1_wins": int(row["team1_wins"]),
            "match_date": row["match_date"],
        }
    (odir / "test_predictions.json").write_text(json.dumps(preds, indent=2))
    return {
        "pred_json": odir / "test_predictions.json", "out_dir": odir,
        "mean_abs_asym": float(np.mean(np.abs(asym))),
        "max_abs_asym": float(np.max(np.abs(asym))),
    }


def eval_seed(pred_json: Path, odir: Path) -> dict:
    evd = odir / "eval"
    run(["uv", "run", "python", "scripts/sim_eval/blend_eval_json.py",
         "--sim-json", str(SIM_ENVELOPE), "--direct-json", str(pred_json),
         "--w", "0.0", "--out-dir", str(evd)])
    blended = evd / "hier_all_20260425_165622_w0p00.json"
    run(["uv", "run", "python", "scripts/sim_eval/reslice_eval_json.py",
         "--in", str(blended), "--odds", str(ODDS),
         "--out-dir", str(evd / "sliced"),
         "--min-volume", "50000", "--min-volume", "100000"])
    out = {}
    for tag in ("50000", "100000"):
        sliced = (evd / "sliced"
                  / f"hier_all_20260425_165622_w0p00_min_volume_{tag}.json")
        s = json.loads(sliced.read_text())["summary"]
        out[tag] = {
            "ll": s["avg_log_loss"], "market_ll": s.get("market_avg_log_loss"),
            "roi": s["flat_betting_roi_pct"],
            "roi_lo": s["flat_betting_roi_ci_low"],
            "roi_hi": s["flat_betting_roi_ci_high"],
            "n_bets": s["flat_betting_bets_placed"],
            "win": s["flat_betting_win_rate"],
            "n_matches": s["n_matches_evaluated"],
        }
    return out


def main():
    results = {}
    for arm, cfg in ARMS.items():
        raw_all = json.loads(cfg["raw_results"].read_text())[cfg["variant"]]
        results[arm] = {"raw": raw_all, "sym": {}, "asym": {}}
        for seed in SEEDS:
            print(f"\n=== {arm} seed {seed} ===")
            sc = score_seed(arm, seed)
            results[arm]["asym"][str(seed)] = {
                "mean_abs": sc["mean_abs_asym"], "max_abs": sc["max_abs_asym"]}
            results[arm]["sym"][str(seed)] = eval_seed(
                sc["pred_json"], sc["out_dir"])

    OUT.mkdir(parents=True, exist_ok=True)
    serializable = {
        arm: {"raw": r["raw"], "sym": r["sym"], "asym": r["asym"]}
        for arm, r in results.items()}
    (OUT / "d11_results.json").write_text(json.dumps(serializable, indent=2))

    verdict_parts = {}
    for arm, r in results.items():
        print(f"\n\n===== D11 PAIRED SUMMARY — {arm} (>=$50k) =====")
        print(f"{'seed':>5} | {'raw LL':>8} {'sym LL':>8} {'dLL':>8} | "
              f"{'raw ROI':>9} {'sym ROI':>9} {'dROI':>8} | mean|asym|")
        dlls, drois = [], []
        for s in SEEDS:
            raw = r["raw"][str(s)]["50000"]
            sym = r["sym"][str(s)]["50000"]
            dll = sym["ll"] - raw["ll"]
            droi = sym["roi"] - raw["roi"]
            dlls.append(dll)
            drois.append(droi)
            print(f"{s:>5} | {raw['ll']:>8.4f} {sym['ll']:>8.4f} {dll:>+8.4f} | "
                  f"{raw['roi']:>+9.2f} {sym['roi']:>+9.2f} {droi:>+8.2f} | "
                  f"{r['asym'][str(s)]['mean_abs']:.4f}")
        mdll, mdroi = float(np.mean(dlls)), float(np.mean(drois))
        n_ll_better = sum(1 for d in dlls if d < 0)
        n_roi_better = sum(1 for d in drois if d > 0)
        print(f"{'MEAN':>5} | {'':>8} {'':>8} {mdll:>+8.4f} | "
              f"{'':>9} {'':>9} {mdroi:>+8.2f}")
        print(f"  LL better {n_ll_better}/5 seeds, ROI better {n_roi_better}/5")
        r100_raw = np.mean([r["raw"][str(s)]["100000"]["roi"] for s in SEEDS])
        r100_sym = np.mean([r["sym"][str(s)]["100000"]["roi"] for s in SEEDS])
        l100 = np.mean([r["sym"][str(s)]["100000"]["ll"]
                        - r["raw"][str(s)]["100000"]["ll"] for s in SEEDS])
        print(f"  >=$100k context: dLL {l100:+.4f}, ROI {r100_raw:+.2f} -> "
              f"{r100_sym:+.2f} ({r100_sym - r100_raw:+.2f}pp)")
        verdict_parts[arm] = (mdll, mdroi)

    mdll, mdroi = verdict_parts["d7_swap"]
    ll_up = mdll <= LL_FLOOR
    roi_up = mdroi >= ROI_FLOOR
    verdict = ("LANDED" if ll_up and roi_up
               else "TABLED" if ll_up or roi_up else "FAILED")
    print("\n===== PRE-COMMITTED VERDICT (PRIMARY = d7_swap, >=$50k mean) =====")
    print(f"  dLL {mdll:+.4f} (improved iff <= {LL_FLOOR}) -> "
          f"{'IMPROVED' if ll_up else 'not improved'}")
    print(f"  dROI {mdroi:+.2f}pp (improved iff >= +{ROI_FLOOR}) -> "
          f"{'IMPROVED' if roi_up else 'not improved'}")
    print(f"  VERDICT: {verdict}")


if __name__ == "__main__":
    main()
