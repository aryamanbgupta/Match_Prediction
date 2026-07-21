"""D15 gate analysis — full attribution unit (D2 extras semantics + D14
pre-ball snapshot + D4 run-out dismissal channel) vs the D1 baseline.

Two prop_backtest detail JSONs, both venue-ON at seed 43, training-aligned
run_rate (post-D1 default path), stale v1 global vector calibrator on BOTH
sides (same 261 matches, same model + venue encoder + calibrator):

  baseline   models/auto/d1/detail_d1_s43_n261.json
             canonical venue-ON post-D1 baseline; pre-D2 extras semantics,
             pre-D14 attribution, no run-out channel (100% bowler credit).
  challenger models/auto/d15/detail_d15_s43_n261.json
             D2 extras semantics + D14 attribution snapshot (re-applied via
             revert of 87d9133) + D15 run-out dismissal channel
             (p_runout=0.075077 / nonstriker_share=0.468470, empirical
             as-of < 2025-07-01; run-outs carry NO bowling-card credit,
             total wicket rate untouched).

ONLY delta between the runs = the D2+D14+D15 sim semantics, applied as one
unit per the IDEAS.md D15 entry.

MECHANISM NOTE (pre-run): the eval's actuals side counts a bowler wicket
only for kind != "run out" (prop_backtest.py:326) while the pre-D15 sim
credits 100% of sampled wickets to the bowler — D14's correct over-final
keying UNMASKED that overshoot (bowler_wkts_1plus +0.0027 CI-clean); the
run-out channel removes ~7.5% of bowler-credited wickets, the right sign
and rough magnitude.

Paired run-vs-run change in Brier/MAE, cluster-bootstrapped by match,
reusing the A8 tooling (this file lives in scripts/auto/, NOT the frozen
eval framework).

PRE-COMMITTED GATE (written before any D15 sim result existed; families
exactly as pre-registered in the IDEAS.md D15 entry):

  Precondition — the extended unit check (scripts/auto/d15_unit_check.py:
  D2's 26 assertions + D14 card/label contract + run-out attribution +
  live-path draw frequencies) must pass, established BEFORE the sim run.
  If not met, the verdict is FAILED regardless of the numbers.

  PRIMARY (both sub-conditions required):
    P1: bowler_wkts_1plus does NOT regress CI-clean (dBrier CI lower
        bound <= 0) — D14's unmasked regression must disappear.
    P2: team_first_over_mae RETAINS a CI-clean improvement (dMAE CI upper
        bound < 0) — D14's extraction-window win must survive.

  GUARDS (no CI-excludes-0 regression on any):
    top_bowler (D4's warning: credit redistribution), batter_runs_mae,
    bowler_wkts_2plus.

  Verdict mapping: PRIMARY met + GUARDS held -> LANDED (D2+D14+D4 ship as
  the attribution unit; re-baseline warning applies — striker sequencing,
  card attribution, over labels and wicket crediting all change);
  exactly one -> TABLED; neither -> FAILED.

Everything else is reported for context only and cannot flip the verdict.
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "auto"))

from a8_gate_analysis import (brier_pair, cluster_boot, load, mae_pair,  # noqa
                              paired_rows)

PRIMARY_BINARY = ["bowler_wkts_1plus"]
PRIMARY_MAE = ["team_first_over_mae"]
GUARD_BINARY = ["top_bowler", "bowler_wkts_2plus"]
GUARD_MAE = ["batter_runs_mae"]

CONTEXT_BINARY = [
    "bowler_wkts_3plus",
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
CONTEXT_MAE = ["highest_individual_mae", "batter_fours_mae",
               "team_total_fours_mae", "team_total_sixes_mae"]


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
                    default=str(REPO / "models/auto/d1/detail_d1_s43_n261.json"),
                    help="canonical venue-ON post-D1 baseline (s43)")
    ap.add_argument("--fixed",
                    default=str(REPO / "models/auto/d15/detail_d15_s43_n261.json"),
                    help="run with the D2+D14+D15 attribution unit (s43)")
    ap.add_argument("--unit-check", choices=["met", "notmet"], required=True,
                    help="result of scripts/auto/d15_unit_check.py, "
                         "established before the sim run")
    args = ap.parse_args()

    da, db = load(args.baseline), load(args.fixed)
    print(f"baseline (pre-unit, venue-ON s43): {len(da)} matches | "
          f"fixed (D2+D14+D15 unit, venue-ON s43): {len(db)} matches\n")

    unit_ok = args.unit_check == "met"
    print(f"Precondition (extended unit check incl. run-out attribution): "
          f"{'MET' if unit_ok else 'NOT MET'} (established pre-run)\n")

    print("=" * 84)
    print("PRIMARY — P1: bowler_wkts_1plus must NOT regress CI-clean; "
          "P2: team_first_over_mae must improve CI-clean")
    print("=" * 84)
    fp_b = report_binary(da, db, PRIMARY_BINARY, "base", "d15")
    fp_m = report_mae(da, db, PRIMARY_MAE, "base", "d15")
    p1 = ("bowler_wkts_1plus" in fp_b
          and not (fp_b["bowler_wkts_1plus"][1] > 0))
    p2 = ("team_first_over_mae" in fp_m
          and fp_m["team_first_over_mae"][2] < 0)
    primary = p1 and p2
    print(f"\n  P1 (bowler_wkts_1plus no CI-clean regression): "
          f"{'MET' if p1 else 'NOT MET'}")
    print(f"  P2 (team_first_over_mae CI-clean improvement): "
          f"{'MET' if p2 else 'NOT MET'}")

    print("\n" + "=" * 84)
    print("GUARDS — no CI-excludes-0 regression")
    print("=" * 84)
    fg_m = report_mae(da, db, GUARD_MAE, "base", "d15")
    fg_b = report_binary(da, db, GUARD_BINARY, "base", "d15")
    allg = {**fg_m, **fg_b}
    guards = (all(not (v[1] > 0) for v in allg.values())
              and len(allg) == len(GUARD_MAE) + len(GUARD_BINARY))

    print("\n" + "=" * 84)
    print("CONTEXT (reported for completeness; cannot flip the verdict)")
    print("=" * 84)
    report_binary(da, db, CONTEXT_BINARY, "base", "d15")
    report_mae(da, db, CONTEXT_MAE, "base", "d15")

    print("\n" + "=" * 84)
    print(f"Precondition (unit check): {'MET' if unit_ok else 'NOT MET'}")
    print(f"PRIMARY: {'MET' if primary else 'NOT MET'} "
          f"(P1 {'ok' if p1 else 'FAIL'}, P2 {'ok' if p2 else 'FAIL'})")
    print(f"GUARDS: {'HELD' if guards else 'REGRESSED'}")
    bonus = [f for f, v in allg.items() if v[2] < 0]
    if bonus:
        print(f"Bonus CI-clean guard improvements: {', '.join(bonus)}")
    if not unit_ok:
        v = "FAILED"
    else:
        v = ("LANDED" if primary and guards
             else "TABLED" if primary or guards else "FAILED")
    print(f"VERDICT per pre-committed mapping: {v}")


if __name__ == "__main__":
    main()
