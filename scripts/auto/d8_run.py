"""D8 — recency-weighted training (match model): val-selected half-life,
then paired 5-seed iteration eval.

Train spans 2005-2024 uniformly weighted (`model.fit` passed no
sample_weight); T20 is non-stationary. D8 adds exponential time-decay
`w = 0.5 ** (age_years / HL)` on the TRAIN loss only (normalized to mean 1;
val stays raw for early stopping) via the new
`--decay-half-life-years` flag in xgboost_match_v1.py.

PRE-COMMITTED PROCEDURE (written before any sweep number was seen):
  1. Sweep HL in {3, 6, 10} x A1 seeds {29,7,13,42,101} on
     data/xgb_match_data_v2_clean, trainer defaults otherwise. The HL=inf
     arm (= uniform weights = trainer defaults) is D7's base arm — val LLs
     read from models/auto/d7/base_seed*/train_metrics.json (same data-dir,
     same seeds, same defaults, trained same-day; base seed29 reproduced
     A1/production behavior exactly in D7/D12).
  2. Select HL* = argmin over {3, 6, 10, inf} of MEAN VAL LL across the 5
     seeds. VAL ONLY — the iteration set plays no part in selection.
  3. Cheap exit: if HL* == inf, no decay improves val LL -> the idea's
     chosen configuration is the existing baseline -> FAILED (M5-style
     pre-training exit; iteration eval never touched).
  4. Else paired 5-seed recipe-A eval of the HL* models vs D7's logged
     per-seed base >=$50k results (models/auto/d7/d7_results.json).
     Env-drift guard: D7's base was verified to reproduce A1's logged
     numbers exactly earlier today; the parquet is unchanged (git-clean).
  5. Verdict (program.md retrain floors, >1 seed-std): mean dLL <= -0.007
     AND mean dROI >= +2.3pp -> LANDED; exactly one -> TABLED; none ->
     FAILED.

Artifacts: models/auto/d8/ (gitignored scratch).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

SEEDS = [29, 7, 13, 42, 101]
HLS = [3.0, 6.0, 10.0]  # inf arm = D7 base (uniform weights)
DATA = ROOT / "data/xgb_match_data_v2_clean"
MODELS = ROOT / "models/auto/d8"
D7_BASE = ROOT / "models/auto/d7"
SIM_ENVELOPE = ROOT / "eval_out/phase5_hier/hier_all_20260425_165622.json"
ODDS = ROOT / "betting_odds_polymarket.json"

LL_FLOOR = 0.007   # A1 seed-std on >=$50k LL
ROI_FLOOR = 2.3    # A1 seed-std on >=$50k flat ROI (pp)


def run(cmd):
    print("  $", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)


def train_arm(hl: float, seed: int, cmd: str = "train") -> Path:
    tag = f"hl{hl:g}_seed{seed}"
    mdir = MODELS / tag
    args = ["uv", "run", "python", "scripts/xgboost_match_v1.py",
            "--cmd", cmd, "--data-dir", str(DATA), "--model-dir", str(mdir),
            "--seed", str(seed), "--decay-half-life-years", str(hl)]
    run(args)
    return mdir


def val_ll(mdir: Path) -> float:
    return json.loads((mdir / "train_metrics.json").read_text())["val_log_loss"]


def eval_arm(mdir: Path) -> dict:
    evd = mdir / "eval"
    run(["uv", "run", "python", "scripts/sim_eval/blend_eval_json.py",
         "--sim-json", str(SIM_ENVELOPE),
         "--direct-json", str(mdir / "test_predictions.json"),
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
            "ll": s["avg_log_loss"],
            "roi": s["flat_betting_roi_pct"],
            "roi_lo": s["flat_betting_roi_ci_low"],
            "roi_hi": s["flat_betting_roi_ci_high"],
            "n_bets": s["flat_betting_bets_placed"],
            "win": s["flat_betting_win_rate"],
            "n_matches": s["n_matches_evaluated"],
        }
    return out


def main():
    MODELS.mkdir(parents=True, exist_ok=True)

    # ---- Stage 1: val-LL sweep --------------------------------------------
    sweep = {}  # hl -> {seed: val_ll}
    for hl in HLS:
        sweep[hl] = {}
        for seed in SEEDS:
            print(f"\n=== sweep HL={hl:g} seed {seed} ===")
            mdir = train_arm(hl, seed, cmd="both")  # both: test preds saved
            sweep[hl][seed] = val_ll(mdir)
    sweep["inf"] = {}
    for seed in SEEDS:
        sweep["inf"][seed] = val_ll(D7_BASE / f"base_seed{seed}")

    print("\n\n============ D8 VAL-LL SWEEP (mean over 5 seeds) ============")
    means = {}
    for hl in HLS + ["inf"]:
        vals = [sweep[hl][s] for s in SEEDS]
        means[hl] = sum(vals) / len(vals)
        per_seed = "  ".join(f"{v:.4f}" for v in vals)
        print(f"  HL={hl!s:>4}: mean val LL {means[hl]:.4f}   [{per_seed}]")
    hl_star = min(means, key=means.get)
    print(f"\n  SELECTED (argmin mean val LL): HL* = {hl_star}")
    (MODELS / "d8_sweep.json").write_text(json.dumps(
        {"sweep": {str(k): v for k, v in sweep.items()},
         "means": {str(k): v for k, v in means.items()},
         "hl_star": str(hl_star)}, indent=2))

    if hl_star == "inf":
        print("\n  HL* = inf (uniform weights) -> no decay half-life improves "
              "val LL -> D8 FAILED (pre-committed cheap exit; iteration set "
              "untouched).")
        return

    # ---- Stage 2: paired 5-seed iteration eval at HL* ---------------------
    d7 = json.loads((D7_BASE / "d7_results.json").read_text())["base"]
    results = {}
    for seed in SEEDS:
        print(f"\n=== eval HL*={hl_star:g} seed {seed} ===")
        mdir = MODELS / f"hl{hl_star:g}_seed{seed}"
        results[seed] = eval_arm(mdir)
    (MODELS / "d8_results.json").write_text(json.dumps(
        {"hl_star": hl_star, "decay": results}, indent=2))

    def mean(d, k, tag="50000"):
        return sum(d[str(s)][tag][k] if str(s) in d else d[s][tag][k]
                   for s in SEEDS) / len(SEEDS)

    print(f"\n\n============ D8 PAIRED SUMMARY (>=$50k, HL*={hl_star:g}) "
          "============")
    print(f"{'seed':>5} | {'base LL':>8} {'d8 LL':>8} {'dLL':>8} | "
          f"{'base ROI':>9} {'d8 ROI':>9} {'dROI':>8}")
    for s in SEEDS:
        b = d7[str(s)]["50000"]
        a = results[s]["50000"]
        print(f"{s:>5} | {b['ll']:>8.4f} {a['ll']:>8.4f} "
              f"{a['ll'] - b['ll']:>+8.4f} | "
              f"{b['roi']:>+9.2f} {a['roi']:>+9.2f} "
              f"{a['roi'] - b['roi']:>+8.2f}")
    bLL, aLL = mean(d7, "ll"), mean(results, "ll")
    bR, aR = mean(d7, "roi"), mean(results, "roi")
    print(f"{'MEAN':>5} | {bLL:>8.4f} {aLL:>8.4f} {aLL - bLL:>+8.4f} | "
          f"{bR:>+9.2f} {aR:>+9.2f} {aR - bR:>+8.2f}")
    dLL, dR = aLL - bLL, aR - bR
    print(f"\npaired dLL {dLL:+.4f} (improve if <= -{LL_FLOOR})   "
          f"dROI {dR:+.2f}pp (improve if >= +{ROI_FLOOR})")
    print("market LL reference: 0.6267")
    b100, a100 = mean(d7, "roi", "100000"), mean(results, "roi", "100000")
    print(f">=$100k context: base ROI {b100:+.2f}%  d8 ROI {a100:+.2f}%  "
          f"dLL {mean(results, 'll', '100000') - mean(d7, 'll', '100000'):+.4f}")

    ll_up = dLL <= -LL_FLOOR
    roi_up = dR >= ROI_FLOOR
    verdict = "LANDED" if (ll_up and roi_up) else (
        "TABLED" if (ll_up or roi_up) else "FAILED")
    print(f"\nVERDICT (pre-committed rule): LL improved={ll_up}, "
          f"ROI improved={roi_up} -> {verdict}")


if __name__ == "__main__":
    main()
