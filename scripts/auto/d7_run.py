"""D7 — team-swap symmetry augmentation (match model), paired 5-seed eval.

The match model consumes absolute team1_/team2_ features and team assignment
is arbitrary cricsheet order, so it is not antisymmetric (train/val/test base
rates 0.488/0.476/0.472; asymmetric importances). D7 augments TRAIN ONLY with
the mirrored copy of every row (teams exchanged, signed diffs negated, h2h ->
1-x, toss/bat-first/label flipped), doubling effective n to ~15.8k and
enforcing P(t1|A,B) = 1 - P(t1|B,A) at training time.

Design (paired, seed-controlled, same shape as a9_run.py):
  - data: data/xgb_match_data_v2_clean (recipe A canonical; same parquet A1
    used for the fresh-baseline row).
  - Train BOTH base and swap-augmented variants at A1's 5 seeds
    {29,7,13,42,101}, trainer defaults (= M7 config). Same-session base is
    the paired control (A1's logged per-seed numbers are printed alongside
    as an env-drift cross-check).
  - Recipe A eval each: blend --w 0.0 -> reslice vs betting_odds_polymarket
    -> read the >=$50k slice (gate) + >=$100k (context only).

Pre-run verification (all hard-fail):
  - swap mapping covers every parquet column (raises in _swap_frame),
  - swap(swap(df)) == df (involution; checked inside _swap_augment_train),
  - every _SWAP_NEGATE diff identity holds in the parquet
    (diff == t1_col - t2_col exactly), so negation == recomputation,
  - h2h prior sanity: rows with h2h_n_meetings == 0 sit at exactly 0.5.

Verdict basis (program.md): paired swap - base on >=$50k, mean over 5 seeds,
vs the A1 seed floor (LL 0.007 / ROI 2.3pp). BOTH up -> LANDED; one ->
TABLED; none -> FAILED.
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

from xgboost_match_v1 import _swap_augment_train, _SWAP_NEGATE  # noqa: E402

SEEDS = [29, 7, 13, 42, 101]
DATA = ROOT / "data/xgb_match_data_v2_clean"
MODELS = ROOT / "models/auto/d7"
SIM_ENVELOPE = ROOT / "eval_out/phase5_hier/hier_all_20260425_165622.json"
ODDS = ROOT / "betting_odds_polymarket.json"

# diff column -> (team1 col, team2 col); negation is only valid if the
# parquet's diff is exactly t1 - t2.
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

# A1 logged per-seed >=$50k results (research/reports/auto/A1.md) — printed
# as an env-drift cross-check next to the same-session base.
A1_LOGGED = {29: (0.6231, 23.96), 7: (0.6373, 18.22), 13: (0.6293, 21.30),
             42: (0.6292, 18.73), 101: (0.6399, 20.61)}


def verify_mapping() -> None:
    train = pd.read_parquet(DATA / "train.parquet")
    assert set(DIFF_IDENTITIES) == set(_SWAP_NEGATE)
    for diff, (a, b) in DIFF_IDENTITIES.items():
        dev = np.max(np.abs(train[diff].to_numpy()
                            - (train[a].to_numpy() - train[b].to_numpy())))
        print(f"  [verify] {diff}: max|diff - (t1-t2)| = {dev:.3e}")
        assert dev == 0.0, f"{diff} is not exactly t1 - t2; negation invalid"
    h2h0 = train.loc[train["h2h_n_meetings"] == 0, "h2h_team1_win_rate_shrunk"]
    assert (h2h0 == 0.5).all(), "h2h prior is not 0.5 at n=0"
    print(f"  [verify] h2h prior at n=0 == 0.5 on {len(h2h0):,} rows")
    aug = _swap_augment_train(train)  # coverage + involution hard-checks
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
           "--model-dir", str(mdir), "--seed", str(seed)]
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


def main():
    print("=== D7 pre-run mapping verification ===")
    verify_mapping()

    results = {"base": {}, "swap": {}}
    for seed in SEEDS:
        for variant in ("base", "swap"):
            print(f"\n=== {variant} seed {seed} ===")
            results[variant][seed] = train_and_eval(variant, seed)
    MODELS.mkdir(parents=True, exist_ok=True)
    (MODELS / "d7_results.json").write_text(json.dumps(results, indent=2))

    def mean(v, k, tag="50000"):
        return sum(results[v][s][tag][k] for s in SEEDS) / len(SEEDS)

    print("\n\n============ D7 PAIRED SUMMARY (>=$50k) ============")
    print(f"{'seed':>5} | {'base LL':>8} {'swap LL':>8} {'dLL':>8} | "
          f"{'base ROI':>9} {'swap ROI':>9} {'dROI':>8} | A1-logged base")
    for s in SEEDS:
        b = results["base"][s]["50000"]
        a = results["swap"][s]["50000"]
        a1 = A1_LOGGED[s]
        print(f"{s:>5} | {b['ll']:>8.4f} {a['ll']:>8.4f} "
              f"{a['ll'] - b['ll']:>+8.4f} | "
              f"{b['roi']:>+9.2f} {a['roi']:>+9.2f} "
              f"{a['roi'] - b['roi']:>+8.2f} | {a1[0]:.4f}/{a1[1]:+.2f}")
    bLL, aLL = mean("base", "ll"), mean("swap", "ll")
    bR, aR = mean("base", "roi"), mean("swap", "roi")
    print(f"{'MEAN':>5} | {bLL:>8.4f} {aLL:>8.4f} {aLL - bLL:>+8.4f} | "
          f"{bR:>+9.2f} {aR:>+9.2f} {aR - bR:>+8.2f}")
    print(f"\npaired dLL {aLL - bLL:+.4f} (floor -0.007 to improve)   "
          f"dROI {aR - bR:+.2f}pp (floor +2.3 to improve)")
    print("market LL reference: 0.6267")
    b100, a100 = mean("base", "roi", "100000"), mean("swap", "roi", "100000")
    print(f">=$100k context: base ROI {b100:+.2f}%  swap ROI {a100:+.2f}%  "
          f"dLL {mean('swap', 'll', '100000') - mean('base', 'll', '100000'):+.4f}")


if __name__ == "__main__":
    main()
