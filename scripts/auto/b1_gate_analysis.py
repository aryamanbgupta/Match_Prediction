"""B1 gate analysis — venue-encoder fix vs venue-blind sim (venue_encoded=0).

Two prop_backtest detail JSONs (same 261 matches, same seed 42, same vector
ball calibrator; ONLY difference = the B1 run loads the rebuilt training-time
venue encoder so venue_encoded carries real codes instead of 0). Paired
run-vs-run change in Brier/MAE, cluster-bootstrapped by match, reusing the
A8 tooling (this file lives in scripts/auto/, NOT the frozen eval framework).

PRE-COMMITTED GATE (written before results were seen; per IDEAS.md B1):

  GATE 1 (venue-sensitive improvement, decisive):
    PRIMARY = pooled paired dBrier across the 8 venue-sensitive BINARY lines
        pp_total_ou_{45_5,50_5,55_5}         (PP totals)
        innings_runs_ou_{160_5,170_5,180_5}  (team totals)
        match_total_sixes_ou_{15_5,20_5}     (sixes)
    must be < 0 with 95% CI excluding 0,
    AND neither venue-sensitive MAE family (team_total_fours_mae,
    team_total_sixes_mae) may regress with CI excluding 0.
    A single pooled statistic is the primary on purpose — the A16 lesson is
    that per-family scans across ~10 lines produce chance 95% crossings.

  GATE 2 (guards, must NOT regress = no CI-excludes-0 increase):
    top_bowler, bowler_wkts_{1,2,3}plus (the only fair-baseline-beating
    binary family + E5's overshoot family), team_first_over_mae and
    batter_runs_mae (established continuous-skill metrics).

  Verdict mapping: GATE1 met + GATE2 held -> LANDED; exactly one -> TABLED;
  neither -> FAILED.

Everything else is reported for context only and cannot flip the verdict.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "auto"))

from a8_gate_analysis import (brier_pair, cluster_boot, load, mae_pair,  # noqa
                              paired_rows)

VENUE_BINARY = [
    "pp_total_ou_45_5", "pp_total_ou_50_5", "pp_total_ou_55_5",
    "innings_runs_ou_160_5", "innings_runs_ou_170_5", "innings_runs_ou_180_5",
    "match_total_sixes_ou_15_5", "match_total_sixes_ou_20_5",
]
VENUE_MAE = ["team_total_fours_mae", "team_total_sixes_mae"]

GUARD_BINARY = ["top_bowler", "bowler_wkts_1plus", "bowler_wkts_2plus",
                "bowler_wkts_3plus"]
GUARD_MAE = ["team_first_over_mae", "batter_runs_mae"]

CONTEXT_BINARY = [
    "first_wicket_runs_ou_30_5",
    "team_highest_individual_ou_29_5", "team_highest_individual_ou_34_5",
    "team_highest_individual_ou_39_5",
    "highest_over_runs_ou_18_5", "highest_over_runs_ou_24_5",
    "bowler_economy_ou_8_5", "bowler_economy_ou_10_5",
    "batter_50plus", "batter_6plus_six",
    "batter_fours_1plus", "batter_fours_2plus", "batter_fours_3plus",
    "top_batter", "p_tie",
]
CONTEXT_MAE = ["highest_individual_mae", "batter_fours_mae"]


def report_binary(dv, db, fams, la, lb):
    hdr = (f"{'family':<32}{'n':>7}{'Brier_' + la:>12}{'Brier_' + lb:>12}"
           f"{'dBrier':>10}   95% CI ({lb}-{la})   flag")
    print(hdr)
    for fam in fams:
        rows = paired_rows(dv, db, fam)
        if not rows:
            continue
        ba, bb, d, (lo, hi) = brier_pair(rows)
        flag = "DOWN(better)" if hi < 0 else "UP(worse)" if lo > 0 else "~noise"
        print(f"{fam:<32}{len(rows):>7}{ba:>12.4f}{bb:>12.4f}{d:>+10.4f}"
              f"   [{lo:+.4f},{hi:+.4f}]  {flag}")


def report_mae(dv, db, fams, la, lb):
    for fam in fams:
        res = mae_pair(dv, db, fam)
        if res:
            ma, mb, d, (lo, hi), n = res
            flag = "DOWN(better)" if hi < 0 else "UP(worse)" if lo > 0 else "~noise"
            print(f"{fam:<32}{n:>7}  MAE_{la}={ma:.3f}  MAE_{lb}={mb:.3f}  "
                  f"dMAE={d:+.3f}  [{lo:+.3f},{hi:+.3f}]  {flag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vec", default=str(REPO / "models/auto/a8/detail_vec_n261.json"),
                    help="venue-blind single-vector baseline detail JSON")
    ap.add_argument("--b1", default=str(REPO / "models/auto/b1/detail_venue_n261.json"),
                    help="venue-encoder-fixed detail JSON")
    args = ap.parse_args()

    dv, db = load(args.vec), load(args.b1)
    print(f"vec (venue-blind): {len(dv)} matches | b1 (venue-fixed): {len(db)} matches\n")

    print("=" * 84)
    print("GATE 1 — venue-sensitive families (PRIMARY = pooled binary dBrier, CI must be <0)")
    print("=" * 84)
    report_binary(dv, db, VENUE_BINARY, "vec", "b1")
    pooled = []
    for fam in VENUE_BINARY:
        pooled += paired_rows(dv, db, fam)
    if pooled:
        d = float(np.mean([(r[3] - r[1]) ** 2 - (r[2] - r[1]) ** 2 for r in pooled]))
        lo, hi = cluster_boot(pooled, lambda r: (r[3] - r[1]) ** 2 - (r[2] - r[1]) ** 2)
        verdict = "IMPROVED (CI<0)" if hi < 0 else "WORSE (CI>0)" if lo > 0 else "NOISE (CI straddles 0)"
        print(f"\n  >>> PRIMARY pooled dBrier over {len(VENUE_BINARY)} binary lines "
              f"({len(pooled)} obs) = {d:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  ->  {verdict}")
    print("\n  venue-sensitive MAE (must not regress with CI>0; improvement supports):")
    report_mae(dv, db, VENUE_MAE, "vec", "b1")

    print("\n" + "=" * 84)
    print("GATE 2 — guards (must NOT regress: no CI-excludes-0 increase)")
    print("=" * 84)
    report_binary(dv, db, GUARD_BINARY, "vec", "b1")
    report_mae(dv, db, GUARD_MAE, "vec", "b1")

    print("\n" + "=" * 84)
    print("CONTEXT (reported for completeness; cannot flip the verdict)")
    print("=" * 84)
    report_binary(dv, db, CONTEXT_BINARY, "vec", "b1")
    report_mae(dv, db, CONTEXT_MAE, "vec", "b1")


if __name__ == "__main__":
    main()
