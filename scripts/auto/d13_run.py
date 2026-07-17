"""D13 — swap augmentation + recency decay combined (D7 x D8), paired 5-seed.

D7 (LANDED): --swap-augment, dLL -0.0121 / dROI +3.01pp vs base.
D8 (TABLED): --decay-half-life-years 6, dLL -0.0093 / dROI -1.46pp vs base.
Disjoint mechanisms on the same training procedure (data augmentation vs loss
weighting). D13 runs BOTH flags together on data/xgb_match_data_v2_clean at
the A1 seeds, trainer defaults otherwise. HL is FIXED at 6 from D8's
pre-committed val-only sweep — no new sweep, no extra selection.

PRE-COMMITTED PROCEDURE (written before any combo model was trained):
  0. Mirrored-weight verification: decay weights are computed AFTER
     _swap_augment_train inside train_model (code order verified:
     xgboost_match_v1.py :246 swap, :288 weights; match_date is
     _SWAP_INVARIANT). Empirically assert on the augmented frame at HL=6:
     w[:n] == w[n:] exactly (mirror pairs share match_date -> same weight),
     mean(w) == 1, and w[:n] == the base-frame normalized weights (the mirror
     doubles every weight exactly once, so mean-1 normalization is unchanged).
     Hard-fail if any check breaks.
  1. Train 5 seeds {29,7,13,42,101} with
     --swap-augment --decay-half-life-years 6 -> models/auto/d13/combo_seed*.
  2. Recipe-A eval each (blend w0.0 + reslice, min-volume 50k/100k).
  3. Paired comparison vs the D7 SWAP arm (the stronger parent is the
     control, per the idea text): models/auto/d7/d7_results.json ["swap"].
     Env-drift guard: D7's arms were trained same-branch on the unchanged
     v2_clean parquet, and D7's base arm reproduced A1's logged numbers
     exactly on all 5 seeds.
  4. Verdict (pre-committed, program.md retrain floors — conservative given
     D7 measured swap-arm LL seed-std 0.0027):
     mean dLL <= -0.007 AND mean dROI >= +2.3pp -> LANDED;
     exactly one -> TABLED; none -> FAILED.
     Per-seed direction counts and swap-arm seed-std reported either way.
     Context only (not gating): comparison vs D7 base arm.

Artifacts: models/auto/d13/ (gitignored scratch).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

SEEDS = [29, 7, 13, 42, 101]
HL = 6.0  # fixed by D8's val-only sweep — no new selection
DATA = ROOT / "data/xgb_match_data_v2_clean"
MODELS = ROOT / "models/auto/d13"
D7_RESULTS = ROOT / "models/auto/d7/d7_results.json"
SIM_ENVELOPE = ROOT / "eval_out/phase5_hier/hier_all_20260425_165622.json"
ODDS = ROOT / "betting_odds_polymarket.json"

LL_FLOOR = 0.007
ROI_FLOOR = 2.3


def run(cmd):
    print("  $", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)


def verify_mirrored_weights():
    import xgboost_match_v1 as xm

    train = xm._load_split(DATA, "train")
    n = len(train)
    aug = xm._swap_augment_train(train)
    assert len(aug) == 2 * n, "augmented frame is not exactly doubled"

    def decay_weights(df):
        dates = pd.to_datetime(df["match_date"])
        age_years = (dates.max() - dates).dt.days / 365.25
        w = np.power(0.5, age_years / HL)
        return (w / w.mean()).to_numpy()

    w_aug = decay_weights(aug)
    w_base = decay_weights(train)
    assert np.array_equal(w_aug[:n], w_aug[n:]), \
        "mirrored rows do NOT get identical weights"
    assert abs(w_aug.mean() - 1.0) < 1e-12, "mean-1 normalization broken"
    assert np.allclose(w_aug[:n], w_base, rtol=0, atol=1e-12), \
        "augmented-frame weights differ from base-frame weights"
    eff_n = w_aug.sum() ** 2 / np.square(w_aug).sum()
    print(f"  weight verification PASSED: n={n} -> {2*n} rows, "
          f"w[:n]==w[n:] exact, mean {w_aug.mean():.12f}, "
          f"w[:n]==base exact, min {w_aug.min():.4f} max {w_aug.max():.4f}, "
          f"effective n = {eff_n:.0f} / {2*n}")


def train_arm(seed: int) -> Path:
    mdir = MODELS / f"combo_seed{seed}"
    run(["uv", "run", "python", "scripts/xgboost_match_v1.py",
         "--cmd", "both", "--data-dir", str(DATA), "--model-dir", str(mdir),
         "--seed", str(seed), "--swap-augment",
         "--decay-half-life-years", str(HL)])
    return mdir


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


def seed_get(d, s):
    return d[str(s)] if str(s) in d else d[s]


def mean_of(d, k, tag="50000"):
    return sum(seed_get(d, s)[tag][k] for s in SEEDS) / len(SEEDS)


def std_of(d, k, tag="50000"):
    vals = [seed_get(d, s)[tag][k] for s in SEEDS]
    return float(np.std(vals))


def main():
    MODELS.mkdir(parents=True, exist_ok=True)

    print("=== Stage 0: mirrored-weight verification (HL=6, augmented frame) ===")
    verify_mirrored_weights()

    results = {}
    for seed in SEEDS:
        print(f"\n=== train+eval combo seed {seed} "
              f"(--swap-augment --decay-half-life-years {HL:g}) ===")
        mdir = train_arm(seed)
        results[seed] = eval_arm(mdir)
    (MODELS / "d13_results.json").write_text(json.dumps(
        {"hl": HL, "combo": results}, indent=2))

    d7 = json.loads(D7_RESULTS.read_text())
    swap, base = d7["swap"], d7["base"]

    print("\n\n============ D13 PAIRED SUMMARY vs D7 SWAP arm (>=$50k) ============")
    print(f"{'seed':>5} | {'swap LL':>8} {'d13 LL':>8} {'dLL':>8} | "
          f"{'swap ROI':>9} {'d13 ROI':>9} {'dROI':>8}")
    ll_better = roi_better = 0
    for s in SEEDS:
        b = seed_get(swap, s)["50000"]
        a = results[s]["50000"]
        dll, droi = a["ll"] - b["ll"], a["roi"] - b["roi"]
        ll_better += dll < 0
        roi_better += droi > 0
        print(f"{s:>5} | {b['ll']:>8.4f} {a['ll']:>8.4f} {dll:>+8.4f} | "
              f"{b['roi']:>+9.2f} {a['roi']:>+9.2f} {droi:>+8.2f}")
    bLL, aLL = mean_of(swap, "ll"), mean_of(results, "ll")
    bR, aR = mean_of(swap, "roi"), mean_of(results, "roi")
    print(f"{'MEAN':>5} | {bLL:>8.4f} {aLL:>8.4f} {aLL - bLL:>+8.4f} | "
          f"{bR:>+9.2f} {aR:>+9.2f} {aR - bR:>+8.2f}")
    dLL, dR = aLL - bLL, aR - bR
    print(f"\nper-seed directions: LL better {ll_better}/5, ROI up {roi_better}/5")
    print(f"seed-std: swap LL {std_of(swap, 'll'):.4f} -> d13 LL "
          f"{std_of(results, 'll'):.4f}; swap ROI {std_of(swap, 'roi'):.2f} -> "
          f"d13 ROI {std_of(results, 'roi'):.2f}")
    print(f"paired dLL {dLL:+.4f} (improve if <= -{LL_FLOOR})   "
          f"dROI {dR:+.2f}pp (improve if >= +{ROI_FLOOR})")
    print("market LL reference: 0.6267")
    print(f">=$100k: swap LL {mean_of(swap, 'll', '100000'):.4f} -> d13 "
          f"{mean_of(results, 'll', '100000'):.4f} "
          f"(d {mean_of(results, 'll', '100000') - mean_of(swap, 'll', '100000'):+.4f}); "
          f"swap ROI {mean_of(swap, 'roi', '100000'):+.2f} -> d13 "
          f"{mean_of(results, 'roi', '100000'):+.2f}")
    print(f"\ncontext (not gating) vs D7 BASE arm: base LL "
          f"{mean_of(base, 'll'):.4f} ROI {mean_of(base, 'roi'):+.2f} -> d13 "
          f"dLL {aLL - mean_of(base, 'll'):+.4f}, dROI "
          f"{aR - mean_of(base, 'roi'):+.2f}pp")

    ll_up = dLL <= -LL_FLOOR
    roi_up = dR >= ROI_FLOOR
    verdict = "LANDED" if (ll_up and roi_up) else (
        "TABLED" if (ll_up or roi_up) else "FAILED")
    print(f"\nVERDICT (pre-committed rule, vs D7 swap arm): "
          f"LL improved={ll_up}, ROI improved={roi_up} -> {verdict}")


if __name__ == "__main__":
    main()
