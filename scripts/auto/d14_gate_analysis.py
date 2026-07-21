"""D14 gate analysis — card-attribution fix + re-applied D2 extras semantics
(one unit) vs the D1 baseline.

Two prop_backtest detail JSONs, both venue-ON at seed 43, training-aligned
run_rate (post-D1 default path), stale v1 global vector calibrator on BOTH
sides (same 261 matches, same model + venue encoder + calibrator):

  baseline   models/auto/d1/detail_d1_s43_n261.json
             canonical venue-ON post-D1 baseline; pre-D2 extras semantics
             AND pre-D14 attribution (cards/BallResult keyed on state read
             AFTER update()).
  challenger models/auto/d14/detail_d14_s43_n261.json
             D2 extras semantics re-applied (revert of bc44473) + the D14
             pre-ball attribution snapshot in `_simulate_innings`.

ONLY delta between the runs = `MatchState.update` extras semantics (D2) +
`_simulate_innings` attribution keying (D14), applied as one unit per the
IDEAS.md D14 entry.

SCOPE STATEMENT (declared pre-run): the attribution fix keys EVERYTHING the
per-ball read-after-update defect touches in that one statement block —
batting card (striker), bowling card (bowler: `simulate_ball` reassigns
`state.bowler_idx` to the next over's bowler at over end BEFORE the card
read, so every over-final ball was credited to the wrong bowler), and
BallResult striker/bowler/over/ball labels (`state.balls` post-increment
rolled each over's 6th legal delivery into the next over — the extractor's
`b.over == 0` first-over sum was systematically missing its final
delivery). `team_runs`/`team_wickets` stay cumulative POST-ball (the
first-wicket extractor documents and depends on that). This is the same
defect class as the IDEAS entry names, fixed completely rather than
half-per-field; the gate families below are exactly the four named in
IDEAS.md D14 and were not altered after seeing any result.

Paired run-vs-run change in Brier/MAE, cluster-bootstrapped by match,
reusing the A8 tooling (this file lives in scripts/auto/, NOT the frozen
eval framework).

PRE-COMMITTED GATE (written before any D14 sim result existed):

  GATE 1 (correctness) — the extended unit check
    (`scripts/auto/d14_unit_check.py`) must pass: D2's 26 scripted-delivery
    assertions PLUS card-vs-stats equality, over-final bowler attribution,
    over/ball label, and conservation assertions on a scripted innings
    through the REAL `_simulate_innings`. Evaluated BEFORE the sim run;
    result recorded in the report.

  GATE 2 (guards, must NOT regress = no CI-excludes-0 increase, paired
    vs the D1 baseline) — exactly the four families named in IDEAS.md D14:
      batter_runs_mae, top_bowler, bowler_wkts_1plus, team_first_over_mae.
    (i.e. D2's first-over regression must disappear.) Batter-family
    improvements are a bonus, not required.

  Verdict mapping: GATE1 met + GATE2 held -> LANDED (both fixes ship as one
  unit; re-baseline warning applies — striker sequencing, card attribution
  and over labels all change); exactly one -> TABLED; neither -> FAILED.

Everything else is reported for context only and cannot flip the verdict.
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "auto"))

from a8_gate_analysis import (brier_pair, cluster_boot, load, mae_pair,  # noqa
                              paired_rows)

GUARD_MAE = ["batter_runs_mae", "team_first_over_mae"]
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
                    help="canonical venue-ON post-D1 baseline (s43), pre-D2/D14")
    ap.add_argument("--fixed",
                    default=str(REPO / "models/auto/d14/detail_d14_s43_n261.json"),
                    help="run with D2 extras semantics + D14 attribution fix (s43)")
    ap.add_argument("--gate1", choices=["met", "notmet"], required=True,
                    help="result of scripts/auto/d14_unit_check.py, "
                         "established before the sim run")
    args = ap.parse_args()

    da, db = load(args.baseline), load(args.fixed)
    print(f"baseline (pre-D2/D14, venue-ON s43): {len(da)} matches | "
          f"fixed (D2+D14 unit, venue-ON s43): {len(db)} matches\n")

    gate1 = args.gate1 == "met"
    print(f"GATE 1 (extended unit check incl. card-vs-stats): "
          f"{'MET' if gate1 else 'NOT MET'} (established pre-run)\n")

    print("=" * 84)
    print("GATE 2 — guards (must NOT regress: no CI-excludes-0 increase)")
    print("=" * 84)
    fg_m = report_mae(da, db, GUARD_MAE, "base", "d14")
    fg_b = report_binary(da, db, GUARD_BINARY, "base", "d14")
    allf = {**fg_m, **fg_b}
    gate2 = (all(not (v[1] > 0) for v in allf.values())
             and len(allf) == len(GUARD_MAE) + len(GUARD_BINARY))

    print("\n" + "=" * 84)
    print("CONTEXT (reported for completeness; cannot flip the verdict)")
    print("=" * 84)
    report_binary(da, db, CONTEXT_BINARY, "base", "d14")
    report_mae(da, db, CONTEXT_MAE, "base", "d14")

    print("\n" + "=" * 84)
    print(f"GATE 1 (extended unit check): {'MET' if gate1 else 'NOT MET'}")
    print(f"GATE 2 (guards): {'HELD' if gate2 else 'REGRESSED'}")
    bonus = [f for f, v in allf.items() if v[2] < 0]
    if bonus:
        print(f"Bonus CI-clean guard improvements: {', '.join(bonus)}")
    v = ("LANDED" if gate1 and gate2
         else "TABLED" if gate1 or gate2 else "FAILED")
    print(f"VERDICT per pre-committed mapping: {v}")


if __name__ == "__main__":
    main()
