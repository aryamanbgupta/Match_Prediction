"""D1 gate analysis — sim-side run_rate scale fix vs the venue-ON baseline.

Two prop_backtest detail JSONs, both venue-ON at seed 43, stale v1 global
vector calibrator on BOTH sides (same 261 matches, same model + venue
encoder + calibrator; ONLY difference = the run_rate feature formula):

  baseline   models/auto/b6/detail_venue_s43_n261.json
             run_rate computed runs-per-BALL (score/(balls+1)) — the
             pre-D1 OOD input (~6.24x below training scale).
  challenger models/auto/d1/detail_d1_s43_n261.json
             run_rate aligned to the training formula
             score / max(balls/6, 0.1) (parsing_v2.calculate_basic_features).

Paired run-vs-run change in Brier/MAE, cluster-bootstrapped by match,
reusing the A8 tooling (this file lives in scripts/auto/, NOT the frozen
eval framework).

PRE-COMMITTED GATE (written before any D1 sim result existed; per IDEAS.md
D1, seeded by the 2026-07-16 supervisor review):

  GATE 1 (teacher-forced training parity) — the aligned formula must
    reproduce the training parquet's run_rate exactly. Evaluated BEFORE the
    sim run; result recorded here for the log:
      MET 2026-07-17: max|parquet - formula| = 0.0 across test (186,667
      rows) + validation (124,292 rows); live XGBoostModelV2.extract_features
      spot check exact on 6 states incl. the balls=0/score>0 guard case.

  GATE 2 (guards, must NOT regress = no CI-excludes-0 increase, paired
    vs the venue-ON baseline):
      batter_runs_mae, team_first_over_mae,
      top_bowler, bowler_wkts_1plus,
      team_total_fours_mae, team_total_sixes_mae.
    CI-clean improvements are a bonus, not required.

  Verdict mapping: GATE1 met + GATE2 held -> LANDED (correctness fix kept;
  re-baseline warning applies — this changes the sim's input distribution);
  exactly one -> TABLED; neither -> FAILED.

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

GUARD_MAE = ["batter_runs_mae", "team_first_over_mae",
             "team_total_fours_mae", "team_total_sixes_mae"]
GUARD_BINARY = ["top_bowler", "bowler_wkts_1plus"]

CONTEXT_BINARY = [
    "bowler_wkts_2plus", "bowler_wkts_3plus",
    "batter_50plus", "batter_6plus_six",
    "batter_fours_1plus", "batter_fours_2plus", "batter_fours_3plus",
    "team_highest_individual_ou_29_5", "team_highest_individual_ou_34_5",
    "team_highest_individual_ou_39_5",
    "pp_total_ou_45_5", "pp_total_ou_50_5", "pp_total_ou_55_5",
    "first_wicket_runs_ou_30_5",
    "highest_over_runs_ou_18_5", "highest_over_runs_ou_24_5",
    "innings_runs_ou_160_5", "innings_runs_ou_170_5", "innings_runs_ou_180_5",
    "match_total_sixes_ou_15_5", "match_total_sixes_ou_20_5",
    "bowler_economy_ou_8_5", "bowler_economy_ou_10_5",
    "top_batter", "p_tie",
]
CONTEXT_MAE = ["highest_individual_mae", "batter_fours_mae"]


def report_binary(da, db, fams, la, lb):
    hdr = (f"{'family':<32}{'n':>7}{'Brier_' + la:>12}{'Brier_' + lb:>12}"
           f"{'dBrier':>10}   95% CI ({lb}-{la})   flag")
    print(hdr)
    flags = {}
    for fam in fams:
        rows = paired_rows(da, db, fam)
        if not rows:
            continue
        ba, bb, d, (lo, hi) = brier_pair(rows)
        flag = "DOWN(better)" if hi < 0 else "UP(worse)" if lo > 0 else "~noise"
        flags[fam] = (d, lo, hi, flag)
        print(f"{fam:<32}{len(rows):>7}{ba:>12.4f}{bb:>12.4f}{d:>+10.4f}"
              f"   [{lo:+.4f},{hi:+.4f}]  {flag}")
    return flags


def report_mae(da, db, fams, la, lb):
    flags = {}
    for fam in fams:
        res = mae_pair(da, db, fam)
        if res:
            ma, mb, d, (lo, hi), n = res
            flag = "DOWN(better)" if hi < 0 else "UP(worse)" if lo > 0 else "~noise"
            flags[fam] = (d, lo, hi, flag)
            print(f"{fam:<32}{n:>7}  MAE_{la}={ma:.3f}  MAE_{lb}={mb:.3f}  "
                  f"dMAE={d:+.3f}  [{lo:+.3f},{hi:+.3f}]  {flag}")
    return flags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline",
                    default=str(REPO / "models/auto/b6/detail_venue_s43_n261.json"),
                    help="venue-ON baseline, pre-D1 runs-per-ball run_rate (s43)")
    ap.add_argument("--fixed",
                    default=str(REPO / "models/auto/d1/detail_d1_s43_n261.json"),
                    help="venue-ON run with training-aligned run_rate (s43)")
    args = ap.parse_args()

    da, db = load(args.baseline), load(args.fixed)
    print(f"baseline (runs-per-ball run_rate, venue-ON s43): {len(da)} matches | "
          f"fixed (training-scale run_rate, venue-ON s43): {len(db)} matches\n")

    gate1 = True  # teacher-forced parity, established pre-run (see docstring)
    print("GATE 1 (teacher-forced training parity): MET pre-run — "
          "max|parquet-formula|=0.0 on 310,959 rows; live wrapper exact.\n")

    print("=" * 84)
    print("GATE 2 — guards (must NOT regress: no CI-excludes-0 increase)")
    print("=" * 84)
    fg_m = report_mae(da, db, GUARD_MAE, "base", "d1")
    fg_b = report_binary(da, db, GUARD_BINARY, "base", "d1")
    allf = {**fg_m, **fg_b}
    gate2 = (all(not (v[1] > 0) for v in allf.values())
             and len(allf) == len(GUARD_MAE) + len(GUARD_BINARY))

    print("\n" + "=" * 84)
    print("CONTEXT (reported for completeness; cannot flip the verdict)")
    print("=" * 84)
    report_binary(da, db, CONTEXT_BINARY, "base", "d1")
    report_mae(da, db, CONTEXT_MAE, "base", "d1")

    print("\n" + "=" * 84)
    print(f"GATE 1 (training parity): {'MET' if gate1 else 'NOT MET'}")
    print(f"GATE 2 (guards): {'HELD' if gate2 else 'REGRESSED'}")
    bonus = [f for f, v in allf.items() if v[2] < 0]
    if bonus:
        print(f"Bonus CI-clean guard improvements: {', '.join(bonus)}")
    v = ("LANDED" if gate1 and gate2
         else "TABLED" if gate1 or gate2 else "FAILED")
    print(f"VERDICT per pre-committed mapping: {v}")


if __name__ == "__main__":
    main()
