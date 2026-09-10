# manifest-exempt: closed-idea script, archived by item 8
"""A8 gate analysis — phase-vector vs single-vector ball calibrator.

Reads two prop_backtest detail JSONs (same matches, same seed; only the ball
calibrator differs) and computes, per prop family, a PAIRED run-vs-run change
in the sim's Brier score, cluster-bootstrapped by match. Rows align
positionally within (match_id, family) because both runs score identical
lineups / lines — only the simulated probability differs.

Gate pair (E5 follow-up):
  1. tail overshoot  = pp_total_ou_* + bowler_wkts_*  (want Brier_sim DOWN)
  2. top_bowler skill = must NOT regress               (want Brier_sim not up)

Also reports the E5 boundary-clustering regressors (highest_over_runs_ou_*,
team_first_over_mae) which the phase hypothesis predicts it should fix, and
folds in the fair-baseline summary JSONs (sim vs fair baseline) when given.
"""
import argparse
import sys
from pathlib import Path
import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS / "sim_eval"))

from eval_statistics import (  # noqa: E402,F401
    brier_pair,
    cluster_boot,
    load,
    mae_pair,
    metric_rows,
    paired_rows,
)

TAIL = ["pp_total_ou_45_5", "pp_total_ou_50_5", "pp_total_ou_55_5",
        "bowler_wkts_1plus", "bowler_wkts_2plus", "bowler_wkts_3plus"]
WATCH = ["top_bowler", "highest_over_runs_ou_18_5", "highest_over_runs_ou_24_5"]
MAE_WATCH = ["team_first_over_mae", "batter_runs_mae"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vec", required=True, help="single-vector detail JSON (baseline)")
    ap.add_argument("--phase", required=True, help="phase-vector detail JSON")
    ap.add_argument("--fairbase-vec", default=None)
    ap.add_argument("--fairbase-phase", default=None)
    args = ap.parse_args()

    dv = load(args.vec)
    dp = load(args.phase)
    print(f"vec detail: {len(dv)} matches | phase detail: {len(dp)} matches\n")

    def line(fam):
        rows = paired_rows(dv, dp, fam)
        if not rows:
            return None
        ba, bb, d, (lo, hi) = brier_pair(rows)
        flag = "DOWN(better)" if hi < 0 else "UP(worse)" if lo > 0 else "~noise"
        return (fam, len(rows), ba, bb, d, lo, hi, flag)

    hdr = f"{'family':<26}{'n':>7}{'Brier_vec':>11}{'Brier_phase':>13}{'dBrier':>10}{'  95% CI (phase-vec)':>22}  flag"
    print("=== BINARY: paired Brier_sim (phase − vec), cluster-boot by match ===")
    print(hdr)
    tail_d = []
    for fam in TAIL:
        r = line(fam)
        if r:
            print(f"{r[0]:<26}{r[1]:>7}{r[2]:>11.4f}{r[3]:>13.4f}{r[4]:>+10.4f}   [{r[5]:+.4f},{r[6]:+.4f}]  {r[7]}")
            tail_d.append(r[4])
    print("  --- GATE METRIC 1 (tail overshoot): mean dBrier over %d tail families = %+.4f (want < 0)"
          % (len(tail_d), np.mean(tail_d) if tail_d else float('nan')))
    # aggregate paired CI over all tail rows pooled
    pooled = []
    for fam in TAIL:
        pooled += paired_rows(dv, dp, fam)
    if pooled:
        lo, hi = cluster_boot(pooled, lambda r: (r[3] - r[1]) ** 2 - (r[2] - r[1]) ** 2)
        d = float(np.mean([(r[3]-r[1])**2 - (r[2]-r[1])**2 for r in pooled]))
        print(f"      pooled tail dBrier = {d:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
              f"({'improved' if hi<0 else 'worse' if lo>0 else 'noise'})")

    print("\n=== WATCH binary families (top_bowler = GATE METRIC 2: must not regress) ===")
    print(hdr)
    for fam in WATCH:
        r = line(fam)
        if r:
            print(f"{r[0]:<26}{r[1]:>7}{r[2]:>11.4f}{r[3]:>13.4f}{r[4]:>+10.4f}   [{r[5]:+.4f},{r[6]:+.4f}]  {r[7]}")

    print("\n=== MAE watch families (E5 regressors; phase should help) ===")
    for fam in MAE_WATCH:
        res = mae_pair(dv, dp, fam)
        if res:
            ma, mb, d, (lo, hi), n = res
            flag = "DOWN(better)" if hi < 0 else "UP(worse)" if lo > 0 else "~noise"
            print(f"{fam:<26}{n:>7}  MAE_vec={ma:.3f}  MAE_phase={mb:.3f}  dMAE={d:+.3f}  [{lo:+.3f},{hi:+.3f}]  {flag}")

    if args.fairbase_vec and args.fairbase_phase:
        fv = load(args.fairbase_vec)
        fp = load(args.fairbase_phase)
        print("\n=== Fair-baseline margins (sim − fair base); negative = sim skill ===")
        print(f"{'family':<26}{'dB_vec':>10}{'dB_phase':>10}   CI_phase")
        for fam in TAIL + ["top_bowler"]:
            if fam in fv and fam in fp:
                dv_ = fv[fam]["brier_sim"] - fv[fam]["brier_base"]
                dp_ = fp[fam]["brier_sim"] - fp[fam]["brier_base"]
                ci = fp[fam]["delta_ci"]
                print(f"{fam:<26}{dv_:>+10.4f}{dp_:>+10.4f}   [{ci[0]:+.4f},{ci[1]:+.4f}]")


if __name__ == "__main__":
    main()
