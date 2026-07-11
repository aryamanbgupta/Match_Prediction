"""A9 — E4 quantile pooling forward test.

E4 (reports/e4_quantile_pooling.md) added 8 quantile-pooled ELO "survivor"
features (best-2 bowling ELO, batting-ELO spread diff, per-team bowling maxima)
on top of the M2-venue-only base. E4 discarded them on its pre-registered val
rule (val LL got *worse*), but on the iteration readout the +8 variant ("all8")
was directionally better on every slice, and E4 filed it as a forward-test
candidate. A9 forward-tests it under this loop's dual-metric gate.

Design (paired, seed-controlled):
  - `data/auto/a9/` is built from the existing E4 unfrozen parquet
    (`data/xgb_match_data_v3_e4_unfrozen`) subset to exactly
    [metadata + E4 base-46 numeric + 8 survivors]. Auto-selected feature set:
    all8 = 56, base = 48 (drop the 8 survivors via --drop-features).
  - Train BOTH base (48) and all8 (56) on the SAME a9 parquet at A1's 5 seeds
    {29,7,13,42,101}, trainer defaults (matches recipe A / A1). This isolates
    the 8-feature effect; the E4-unfrozen base is NOT A1's v2_clean base
    (v2_clean is M1/frozen, 45 feats), so a same-parquet base is the only
    clean control.
  - Recipe A eval each: blend --w 0.0 -> reslice vs betting_odds_polymarket.json
    -> read the >=$50k slice (LL, flat ROI, CI, bets).

Verdict basis: paired all8 - base on >=$50k, mean over 5 seeds, vs the A1 seed
floor (LL 0.007 / ROI 2.3pp). BOTH up -> LANDED; one -> TABLED; none -> FAILED.
"""
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
SEEDS = [29, 7, 13, 42, 101]
SURV = ["team2_top6_bat_elo_max", "team1_bowl_elo_max", "team1_bowl_elo_top2",
        "team2_bowl_elo_max", "team2_bowl_elo_top2", "top6_bat_elo_spread_diff",
        "bowl_elo_max_diff", "bowl_elo_top2_diff"]
SIM_ENVELOPE = ROOT / "eval_out_phase5_hier/hier_all_20260425_165622.json"
ODDS = ROOT / "betting_odds_polymarket.json"
A9_DATA = ROOT / "data/auto/a9"
A9_MODELS = ROOT / "models/auto/a9"


def build_parquet():
    """Subset the E4 unfrozen parquet to metadata + the 54 non-encoded all8
    features, IN E4's EXACT all8 column order.

    Column order matters: colsample_bytree=0.9 samples columns by position, so a
    different order yields a different model even at the same seed. Ordering the
    a9 feature columns exactly as E4's all8 (survivors interspersed BEFORE
    venue_p4/p6/pw) makes both variants byte-reproduce E4 seed 29:
    base 0.6312/+15.38, all8 0.6288/+24.35. Dropping the 8 survivors from this
    order recovers E4's base order exactly.
    """
    A9_DATA.mkdir(parents=True, exist_ok=True)
    all8 = [l.strip() for l in
            open(ROOT / "models/xgb_match_e4_all8/feature_columns.txt") if l.strip()]
    enc = {"venue_id_encoded", "competition_tier_encoded"}
    all8_54 = [c for c in all8 if c not in enc]      # E4 all8 order, 54 numeric
    assert len(all8_54) == 54, len(all8_54)
    assert all(s in all8_54 for s in SURV)
    meta = ["match_id", "cricsheet_id", "match_date", "team1", "team2",
            "venue", "competition_tier", "team1_wins"]
    e4dir = ROOT / "data/xgb_match_data_v3_e4_unfrozen"
    cols0 = set(pd.read_parquet(e4dir / "train.parquet").columns)
    meta_present = [c for c in meta if c in cols0]
    keep = meta_present + all8_54
    assert all(c in cols0 for c in all8_54)
    for split in ["train", "validation", "test"]:
        df = pd.read_parquet(e4dir / f"{split}.parquet").reset_index(drop=True)
        df[keep].to_parquet(A9_DATA / f"{split}.parquet")
    print(f"[build] a9 parquet: {len(keep)} cols "
          f"(meta {len(meta_present)} + all8_54 in E4 order)")


def run(cmd):
    print("  $", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)


def train_and_eval(variant: str, seed: int) -> dict:
    mdir = A9_MODELS / f"{variant}_seed{seed}"
    cmd = ["uv", "run", "python", "scripts/xgboost_match_v1.py",
           "--cmd", "both", "--data-dir", str(A9_DATA),
           "--model-dir", str(mdir), "--seed", str(seed)]
    if variant == "base":
        cmd += ["--drop-features", ",".join(SURV)]
    run(cmd)
    # recipe A step 2: blend --w 0.0
    evd = mdir / "eval"
    run(["uv", "run", "python", "scripts/sim_eval/blend_eval_json.py",
         "--sim-json", str(SIM_ENVELOPE),
         "--direct-json", str(mdir / "test_predictions.json"),
         "--w", "0.0", "--out-dir", str(evd)])
    blended = evd / "hier_all_20260425_165622_w0p00.json"
    # recipe A step 3: reslice -> >=$50k
    run(["uv", "run", "python", "scripts/sim_eval/reslice_eval_json.py",
         "--in", str(blended), "--odds", str(ODDS),
         "--out-dir", str(evd / "sliced"), "--min-volume", "50000"])
    sliced = evd / "sliced" / "hier_all_20260425_165622_w0p00_min_volume_50000.json"
    s = json.loads(sliced.read_text())["summary"]
    return {"ll": s["avg_log_loss"], "roi": s["flat_betting_roi_pct"],
            "roi_lo": s["flat_betting_roi_ci_low"], "roi_hi": s["flat_betting_roi_ci_high"],
            "n_bets": s["flat_betting_bets_placed"], "win": s["flat_betting_win_rate"],
            "n_matches": s["n_matches_evaluated"]}


def main():
    build_parquet()
    results = {"base": {}, "all8": {}}
    for seed in SEEDS:
        for variant in ("base", "all8"):
            print(f"\n=== {variant} seed {seed} ===")
            results[variant][seed] = train_and_eval(variant, seed)
    out = A9_MODELS / "a9_results.json"
    out.write_text(json.dumps(results, indent=2))

    def mean(v, k):
        return sum(results[v][s][k] for s in SEEDS) / len(SEEDS)

    print("\n\n================ A9 PAIRED SUMMARY (>=$50k) ================")
    print(f"{'seed':>5} | {'base LL':>8} {'all8 LL':>8} {'dLL':>8} | "
          f"{'base ROI':>9} {'all8 ROI':>9} {'dROI':>8}")
    for s in SEEDS:
        b, a = results["base"][s], results["all8"][s]
        print(f"{s:>5} | {b['ll']:>8.4f} {a['ll']:>8.4f} {a['ll']-b['ll']:>+8.4f} | "
              f"{b['roi']:>+9.2f} {a['roi']:>+9.2f} {a['roi']-b['roi']:>+8.2f}")
    bLL, aLL = mean("base", "ll"), mean("all8", "ll")
    bR, aR = mean("base", "roi"), mean("all8", "roi")
    print(f"{'MEAN':>5} | {bLL:>8.4f} {aLL:>8.4f} {aLL-bLL:>+8.4f} | "
          f"{bR:>+9.2f} {aR:>+9.2f} {aR-bR:>+8.2f}")
    print(f"\nbase mean: LL {bLL:.4f}  ROI {bR:+.2f}%   (n_matches "
          f"{results['base'][29]['n_matches']}, n_bets {results['base'][29]['n_bets']})")
    print(f"all8 mean: LL {aLL:.4f}  ROI {aR:+.2f}%")
    print(f"paired dLL {aLL-bLL:+.4f} (floor 0.007)   dROI {aR-bR:+.2f}pp (floor 2.3)")
    print("market LL reference: 0.6267")


if __name__ == "__main__":
    main()
