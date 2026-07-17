"""D12 — swap augmentation on the PRODUCTION config (D7 transfer test).

D7 landed team-swap symmetry augmentation on the loop's recipe-A baseline
(v2_clean frozen parquet, 45 features, trainer defaults, no monotone). This
tests whether the gain transfers to the production configuration of record
(`models/xgb_match_v3_m7_production`):
  - unfrozen parquet, production 48-feature set (46 numeric + 2 encoded),
    exact production column order (data/auto/d12, built by
    d12_build_parquet.py from the m3_unfrozen superset),
  - trainer defaults (lr 0.05, cs 0.9 = M7) PLUS --monotone (verified ON in
    the production model.pkl: 12 non-zero constraints).

Design mirrors d7_run.py (paired 5-seed base-vs-swap, A1 seeds). Extra
verification: base seed 29 == production's own training seed, so its
test_predictions.json is compared against the production artifact
(reproduction check; reported, not gated — env drift is possible).

Verdict basis (pre-committed, program.md retraining floors): paired
swap - base MEAN over 5 seeds on >=$50k; LL improves if dLL < -0.007,
ROI improves if dROI > +2.3pp. BOTH -> LANDED; one -> TABLED; none -> FAILED.
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

from xgboost_match_v1 import (  # noqa: E402
    METADATA_COLS, CATEGORICAL_FEATURES, _swap_augment_train, _SWAP_NEGATE)

SEEDS = [29, 7, 13, 42, 101]
DATA = ROOT / "data/auto/d12"
MODELS = ROOT / "models/auto/d12"
SIM_ENVELOPE = ROOT / "eval_out/phase5_hier/hier_all_20260425_165622.json"
ODDS = ROOT / "betting_odds_polymarket.json"
PROD = ROOT / "models/xgb_match_v3_m7_production"

# diff column -> (team1 col, team2 col); negation is only valid if the
# parquet's diff is exactly t1 - t2. M1 diffs only — the production feature
# set carries no M2 team-level diffs (M2 landed venue-only).
DIFF_IDENTITIES = {
    "elo_diff_batting": ("team1_batting_elo", "team2_batting_elo"),
    "elo_diff_bowling": ("team1_bowling_elo", "team2_bowling_elo"),
    "batting_avg_diff": ("team1_batting_avg", "team2_batting_avg"),
    "bowling_econ_diff": ("team1_bowling_econ", "team2_bowling_econ"),
    "win_rate_diff": ("team1_win_rate_last_10", "team2_win_rate_last_10"),
    "top6_batting_elo_diff": ("team1_top6_batting_elo_avg",
                              "team2_top6_batting_elo_avg"),
    "bottom5_bowling_elo_diff": ("team1_bottom5_bowling_elo_avg",
                                 "team2_bottom5_bowling_elo_avg"),
}


def verify() -> None:
    train = pd.read_parquet(DATA / "train.parquet")

    # 1. feature order: auto-detected features on this parquet must equal the
    # production feature_columns.txt exactly, order included.
    prod_cols = [c for c in
                 (PROD / "feature_columns.txt").read_text().split() if c]
    numeric = [c for c in train.columns
               if c not in METADATA_COLS and c not in CATEGORICAL_FEATURES]
    detected = numeric + ["venue_id_encoded", "competition_tier_encoded"]
    assert detected == prod_cols, (
        "feature order mismatch vs production:\n"
        f"  detected: {detected}\n  production: {prod_cols}")
    print(f"  [verify] feature order == production feature_columns.txt "
          f"({len(detected)} cols, exact order)")

    # 2. every negated diff is exactly t1 - t2 on THIS parquet.
    present_negate = [c for c in _SWAP_NEGATE if c in train.columns]
    assert set(present_negate) == set(DIFF_IDENTITIES), \
        f"unexpected negate set on d12 parquet: {present_negate}"
    for diff, (a, b) in DIFF_IDENTITIES.items():
        dev = np.max(np.abs(train[diff].to_numpy()
                            - (train[a].to_numpy() - train[b].to_numpy())))
        print(f"  [verify] {diff}: max|diff - (t1-t2)| = {dev:.3e}")
        assert dev == 0.0, f"{diff} is not exactly t1 - t2; negation invalid"

    # 3. h2h prior: 1-x mirror is exact only if the n=0 prior is 0.5.
    h2h0 = train.loc[train["h2h_n_meetings"] == 0, "h2h_team1_win_rate_shrunk"]
    assert (h2h0 == 0.5).all(), "h2h prior is not 0.5 at n=0"
    print(f"  [verify] h2h prior at n=0 == 0.5 on {len(h2h0):,} rows")

    # 4. coverage + involution hard-checks live inside _swap_augment_train.
    aug = _swap_augment_train(train)
    assert len(aug) == 2 * len(train)
    rate = aug["team1_wins"].mean()
    print(f"  [verify] involution + coverage OK; augmented rows "
          f"{len(aug):,}, base rate {rate:.6f}")
    assert abs(rate - 0.5) < 1e-12


def run(cmd):
    print("  $", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)


def train_and_eval(variant: str, seed: int) -> dict:
    mdir = MODELS / f"{variant}_seed{seed}"
    cmd = ["uv", "run", "python", "scripts/xgboost_match_v1.py",
           "--cmd", "both", "--data-dir", str(DATA),
           "--model-dir", str(mdir), "--seed", str(seed), "--monotone"]
    if variant == "swap":
        cmd += ["--swap-augment"]
    run(cmd)
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
            "ll": s["avg_log_loss"], "market_ll": s.get("market_avg_log_loss"),
            "roi": s["flat_betting_roi_pct"],
            "roi_lo": s["flat_betting_roi_ci_low"],
            "roi_hi": s["flat_betting_roi_ci_high"],
            "n_bets": s["flat_betting_bets_placed"],
            "win": s["flat_betting_win_rate"],
            "n_matches": s["n_matches_evaluated"],
        }
    return out


def reproduction_check() -> None:
    """base seed29 vs the production artifact (same config, same seed)."""
    mine = json.loads((MODELS / "base_seed29/test_predictions.json").read_text())
    prod = json.loads((PROD / "test_predictions.json").read_text())
    m = {k: v["p_team1"] for k, v in mine.items()}
    p = {k: v["p_team1"] for k, v in prod.items()}
    common = sorted(set(m) & set(p))
    diffs = np.array([abs(m[k] - p[k]) for k in common])
    print(f"  [repro] base_seed29 vs production: {len(common)}/{len(p)} "
          f"matches common, max|dp| = {diffs.max():.3e}, "
          f"mean|dp| = {diffs.mean():.3e}")


def main():
    print("=== D12 pre-run verification (production frame) ===")
    verify()

    results = {"base": {}, "swap": {}}
    for seed in SEEDS:
        for variant in ("base", "swap"):
            print(f"\n=== {variant} seed {seed} ===")
            results[variant][seed] = train_and_eval(variant, seed)
    MODELS.mkdir(parents=True, exist_ok=True)
    (MODELS / "d12_results.json").write_text(json.dumps(results, indent=2))

    try:
        reproduction_check()
    except Exception as e:  # reported, not gated
        print(f"  [repro] check failed to run: {e}")

    def mean(v, k, tag="50000"):
        return sum(results[v][s][tag][k] for s in SEEDS) / len(SEEDS)

    print("\n\n============ D12 PAIRED SUMMARY (>=$50k) ============")
    print(f"{'seed':>5} | {'base LL':>8} {'swap LL':>8} {'dLL':>8} | "
          f"{'base ROI':>9} {'swap ROI':>9} {'dROI':>8}")
    for s in SEEDS:
        b = results["base"][s]["50000"]
        a = results["swap"][s]["50000"]
        print(f"{s:>5} | {b['ll']:>8.4f} {a['ll']:>8.4f} "
              f"{a['ll'] - b['ll']:>+8.4f} | "
              f"{b['roi']:>+9.2f} {a['roi']:>+9.2f} "
              f"{a['roi'] - b['roi']:>+8.2f}")
    bLL, aLL = mean("base", "ll"), mean("swap", "ll")
    bR, aR = mean("base", "roi"), mean("swap", "roi")
    print(f"{'MEAN':>5} | {bLL:>8.4f} {aLL:>8.4f} {aLL - bLL:>+8.4f} | "
          f"{bR:>+9.2f} {aR:>+9.2f} {aR - bR:>+8.2f}")
    print(f"\npaired dLL {aLL - bLL:+.4f} (improve if < -0.007)   "
          f"dROI {aR - bR:+.2f}pp (improve if > +2.3)")
    print("market LL reference: 0.6267")
    b100, a100 = mean("base", "roi", "100000"), mean("swap", "roi", "100000")
    print(f">=$100k context: base ROI {b100:+.2f}%  swap ROI {a100:+.2f}%  "
          f"dLL {mean('swap', 'll', '100000') - mean('base', 'll', '100000'):+.4f}")


if __name__ == "__main__":
    main()
