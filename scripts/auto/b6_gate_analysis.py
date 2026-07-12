"""B6 gate analysis — venue-encoder fix re-gated on a batter-level continuous
primary at a FRESH seed (B1 follow-up).

Two prop_backtest detail JSONs, BOTH new at seed 43 (venue-on vs venue-blind;
same 261 matches, same vector ball calibrator; ONLY difference = the venue-on
run loads the rebuilt training-time venue encoder so venue_encoded carries
real codes instead of 0). B1's seed-42 run identified batter_runs_mae as the
CI-clean winner *post hoc*, so B6 re-tests it on fresh Monte Carlo draws where
that selection cannot leak: both sides of the pairing are new draws never seen
during the B1 family scan. Paired run-vs-run change in Brier/MAE,
cluster-bootstrapped by match, reusing the A8 tooling (this file lives in
scripts/auto/, NOT the frozen eval framework).

PRE-COMMITTED GATE (written before any seed-43 results were seen; per
IDEAS.md B6):

  GATE 1 (PRIMARY, decisive):
    batter_runs_mae improves paired (dMAE < 0 with 95% CI excluding 0)
    at seed 43. ONE pre-named family, no scan — the A16 lesson.

  GATE 2 (guards, must NOT regress = no CI-excludes-0 increase):
    top_bowler, bowler_wkts_{1,2,3}plus, team_first_over_mae,
    team_total_fours_mae, team_total_sixes_mae.

  Verdict mapping: GATE1 met + GATE2 held -> LANDED (ship the encoder sidecar
  into models/xgb_v3/ as the default sim path; RE-BASELINE all future sim
  comparisons); exactly one -> TABLED; neither -> FAILED (B1's batter-level
  signal was seed-42 selection noise).

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

PRIMARY_MAE = "batter_runs_mae"

GUARD_BINARY = ["top_bowler", "bowler_wkts_1plus", "bowler_wkts_2plus",
                "bowler_wkts_3plus"]
GUARD_MAE = ["team_first_over_mae", "team_total_fours_mae",
             "team_total_sixes_mae"]

# Context: B1's seed-42 batter-level co-movers (do they reproduce at 43?),
# plus B1's original venue-binary primary families and the usual scan.
CONTEXT_BINARY = [
    "batter_50plus", "batter_6plus_six",
    "batter_fours_1plus", "batter_fours_2plus", "batter_fours_3plus",
    "team_highest_individual_ou_29_5", "team_highest_individual_ou_34_5",
    "team_highest_individual_ou_39_5",
    "highest_over_runs_ou_18_5", "highest_over_runs_ou_24_5",
    "pp_total_ou_45_5", "pp_total_ou_50_5", "pp_total_ou_55_5",
    "innings_runs_ou_160_5", "innings_runs_ou_170_5", "innings_runs_ou_180_5",
    "match_total_sixes_ou_15_5", "match_total_sixes_ou_20_5",
    "first_wicket_runs_ou_30_5",
    "bowler_economy_ou_8_5", "bowler_economy_ou_10_5",
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
    ap.add_argument("--blind",
                    default=str(REPO / "models/auto/b6/detail_blind_s43_n261.json"),
                    help="venue-blind seed-43 detail JSON (fresh baseline)")
    ap.add_argument("--venue",
                    default=str(REPO / "models/auto/b6/detail_venue_s43_n261.json"),
                    help="venue-encoder-fixed seed-43 detail JSON")
    args = ap.parse_args()

    dv, db = load(args.blind), load(args.venue)
    print(f"blind (venue_encoded=0, s43): {len(dv)} matches | "
          f"venue (encoder ACTIVE, s43): {len(db)} matches\n")

    print("=" * 84)
    print("GATE 1 — PRIMARY: batter_runs_mae (dMAE must be <0 with CI excluding 0)")
    print("=" * 84)
    res = mae_pair(dv, db, PRIMARY_MAE)
    gate1 = False
    if res:
        ma, mb, d, (lo, hi), n = res
        gate1 = hi < 0
        verdict = ("IMPROVED (CI<0)" if hi < 0
                   else "WORSE (CI>0)" if lo > 0 else "NOISE (CI straddles 0)")
        print(f"  >>> PRIMARY {PRIMARY_MAE} ({n} obs): MAE_blind={ma:.3f}  "
              f"MAE_venue={mb:.3f}  dMAE={d:+.3f}  95% CI [{lo:+.3f},{hi:+.3f}]"
              f"  ->  {verdict}")
    else:
        print("  >>> PRIMARY family missing from detail JSONs — cannot gate")

    print("\n" + "=" * 84)
    print("GATE 2 — guards (must NOT regress: no CI-excludes-0 increase)")
    print("=" * 84)
    report_binary(dv, db, GUARD_BINARY, "blind", "venue")
    report_mae(dv, db, GUARD_MAE, "blind", "venue")

    print("\n" + "=" * 84)
    print("CONTEXT (reported for completeness; cannot flip the verdict)")
    print("=" * 84)
    report_binary(dv, db, CONTEXT_BINARY, "blind", "venue")
    report_mae(dv, db, CONTEXT_MAE, "blind", "venue")

    print("\nGATE 1 (primary):", "MET" if gate1 else "NOT MET",
          "| GATE 2: read guard flags above (any UP(worse) = regressed)")


if __name__ == "__main__":
    main()
