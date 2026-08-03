"""D16 gate analysis — no-class-weights retrain vs its control+vector twin (i7).

WRITTEN AND COMMITTED BEFORE EITHER D16 EVAL RESULT EXISTED.

E5's root cause was `balanced` class weights sampled raw. Every fix since has
been a post-hoc calibration patch (E5 global vector -> A8 -> A14/A15 -> A16 ->
B7 -> B8, chain CLOSED). D16 tests the structural alternative: retrain the ball
model with sample weights OFF so the booster estimates P(outcome|state)
directly, and run it with NO ball calibrator at all.

D6 tried this on the legacy `data/xgb_data_v3` frame and CRASHED — that frame
predates the I7 venue-alias contract and the trainer fail-closes on it. D16 is
the same question run on the trainable I7 frame as a PAIRED TWIN, so there is
zero frame confound: both arms are trained this session from
`experiments/configs/xgb_i7_venue_identity.yaml` on `data/xgb_data_i7`, and the
only delta is the weights∘calibrator stack.

Two prop_backtest detail JSONs, seed 46, n=261 x 100 sims, i7 stats identity,
venue-ON default path (D1 run_rate + D15 attribution + B12-shipped B10 usage
selector):

  baseline  models/auto/d16/detail_control_vec_s46_n261.json
            control booster (balanced weights) + the FRESH d16 vector
            calibrator fit on its own i7 validation predictions — the
            deployed-stack design transplanted to i7.
  d16       models/auto/d16/detail_noweights_raw_s46_n261.json
            no-class-weights retrain, `--ball-calibrator none`.

The calibrator asymmetry is deliberate and is the point of the idea: the claim
under test is "structural retrain beats the calibrated stack design".

There is NO pre-existing i7 sim detail anywhere, and the legacy B12 detail
(`models/auto/b12/detail_b10_s44_n261.json`) is a different model, encoder and
stats identity — it is context only and can never be the comparator here.

PRE-COMMITTED GATE (per research/handoff/D16/plan.md):

  GATE 1 (primary), ALL THREE conditions:
    (a) the NO-WEIGHTS arm's teacher-forced marginal audit PASSES its
        tolerance (|dP(wicket)| <= 0.005 and |d runs/ball| <= 0.05), read from
        models/auto/d16/noweights/marginal_audit.json — recorded BEFORE the
        sim evals;
    (b) pooled tail dBrier (noweights - control_vec) over the ROW POOL
        {pp_total_ou_45_5, pp_total_ou_50_5, pp_total_ou_55_5,
         bowler_wkts_1plus} improves CI-clean (95% CI hi < 0);
    (c) `batter_runs_mae` does NOT regress CI-clean (paired dMAE CI lo <= 0).
  GATE 2 (guards): no CI-clean regression on `top_bowler` dBrier OR
    `team_first_over_mae` dMAE (CI lo <= 0 on each).

  Both -> LANDED; exactly one -> TABLED; neither -> FAILED.
  This script PRINTS the mapping; the ORCHESTRATOR issues the verdict.

Pairing machinery is imported verbatim from `b12_gate_analysis` (paired
per-row delta, cluster bootstrap BY MATCH, 2000 resamples, seed 29, rows
matched by (team, name) identity with a positional cross-check) so D16 is
measured with exactly the tooling that produced the b12 gate.

Run:
  uv run python scripts/auto/d16_gate_analysis.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "auto"))

from b12_gate_analysis import (  # noqa: E402
    N_BOOT,
    BOOT_SEED,
    flag_of,
    keyed_rows,
    load,
    paired_stat,
    positional_rows,
    report,
)
from a8_gate_analysis import cluster_boot  # noqa: E402

TAIL_POOL = ["pp_total_ou_45_5", "pp_total_ou_50_5", "pp_total_ou_55_5",
             "bowler_wkts_1plus"]
GATE1_MAE = "batter_runs_mae"
GUARD_BINARY = ["top_bowler"]
GUARD_MAE = ["team_first_over_mae"]


def pooled_tail(det_a, det_b, fams):
    """Concatenate the tail families' paired rows and score one Brier delta.

    Rows carry the match id in slot 0, so `cluster_boot` resamples whole
    matches across families jointly — the correct clustering for a pooled
    statistic. Equal weight PER ROW (the plan's 'row-pool'); an
    equal-weight-per-family variant is printed separately as context.
    """
    rows, dropped, per_fam = [], 0, {}
    for fam in fams:
        frows, fdrop = keyed_rows(det_a, det_b, fam)
        per_fam[fam] = frows
        rows.extend(frows)
        dropped += fdrop
    if not rows:
        return None
    ba = float(np.mean([(a - y) ** 2 for _, y, a, _ in rows]))
    bb = float(np.mean([(b - y) ** 2 for _, y, _, b in rows]))
    fn = lambda r: (r[3] - r[1]) ** 2 - (r[2] - r[1]) ** 2  # noqa: E731
    lo, hi = cluster_boot(rows, fn, n_boot=N_BOOT, seed=BOOT_SEED)
    return {"brier_a": ba, "brier_b": bb, "delta": bb - ba,
            "lo": lo, "hi": hi, "n": len(rows), "dropped": dropped,
            "per_fam": per_fam}


def equal_weight_family_delta(per_fam):
    """Context only: mean of the per-family dBrier, one vote per family."""
    deltas = []
    for fam, rows in per_fam.items():
        if not rows:
            continue
        da = float(np.mean([(a - y) ** 2 for _, y, a, _ in rows]))
        db = float(np.mean([(b - y) ** 2 for _, y, _, b in rows]))
        deltas.append(db - da)
    return float(np.mean(deltas)) if deltas else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--baseline",
        default=str(REPO / "models/auto/d16/detail_control_vec_s46_n261.json"))
    ap.add_argument(
        "--d16",
        default=str(REPO / "models/auto/d16/detail_noweights_raw_s46_n261.json"))
    ap.add_argument(
        "--audit",
        default=str(REPO / "models/auto/d16/noweights/marginal_audit.json"))
    args = ap.parse_args()

    da, db = load(args.baseline), load(args.d16)
    print(f"baseline: {len(da)} matches ({Path(args.baseline).name})")
    print(f"d16:      {len(db)} matches ({Path(args.d16).name})")
    print(f"pairing: cluster bootstrap by match, n_boot={N_BOOT}, "
          f"seed={BOOT_SEED}; delta = d16 - baseline (negative = d16 better)\n")

    # ---------------------------------------------------------- GATE 1 (a)
    print("=" * 96)
    print("GATE 1(a) — teacher-forced marginal audit of the NO-WEIGHTS arm "
          "(frozen BEFORE the sim evals)")
    print("=" * 96)
    gate1a = False
    try:
        audit = json.load(open(args.audit))
        prim = audit["arms"][audit["primary_arm"]]
        gate1a = bool(audit["pass"])
        print(f"  audit file: {args.audit}")
        print(f"  primary arm: {prim['label']}  (n = {prim['n_balls']:,})")
        print(f"  dP(wicket)   {prim['delta_wicket']:+.5f}   "
              f"(tol +/-{audit['tolerance']['abs_delta_wicket']})")
        print(f"  d runs/ball  {prim['delta_runs_per_ball']:+.5f}   "
              f"(tol +/-{audit['tolerance']['abs_delta_runs_per_ball']})")
        print(f"  test multiclass LL {prim['test_multiclass_logloss']:.4f}")
        print(f"  GATE 1(a): {'PASS' if gate1a else 'FAIL'}")
    except Exception as exc:
        print(f"  audit unreadable: {type(exc).__name__}: {exc}")
        print("  GATE 1(a): FAIL (no recorded audit)")

    # ---------------------------------------------------------- GATE 1 (b)
    print("\n" + "=" * 96)
    print("GATE 1(b) — POOLED TAIL dBrier over the row-pool "
          + ", ".join(TAIL_POOL) + " must be CI-clean negative")
    print("=" * 96)
    pool = pooled_tail(da, db, TAIL_POOL)
    gate1b = False
    if pool is None:
        print("  no rows — GATE 1(b): FAIL")
    else:
        gate1b = pool["hi"] < 0
        print(f"{'pooled tail':<34}{pool['n']:>6}{pool['dropped']:>6}"
              f"{pool['brier_a']:>12.4f}{pool['brier_b']:>12.4f}"
              f"{pool['delta']:>+10.4f}   "
              f"[{pool['lo']:+.4f},{pool['hi']:+.4f}]  "
              f"{flag_of(pool['lo'], pool['hi'])}")
        print(f"  (context, equal weight per family: "
              f"{equal_weight_family_delta(pool['per_fam']):+.4f})")
        print("\n  per-family breakdown (context; the pool is the gate):")
        report(da, db, TAIL_POOL, "control_vec", "noweights",
               show_positional=True)
        print(f"\n  GATE 1(b): {'PASS' if gate1b else 'FAIL'}")

    # ---------------------------------------------------------- GATE 1 (c)
    print("\n" + "=" * 96)
    print(f"GATE 1(c) — {GATE1_MAE} must NOT regress CI-clean (CI lo <= 0)")
    print("=" * 96)
    g1c = report(da, db, [GATE1_MAE], "control_vec", "noweights",
                 show_positional=True)
    v = g1c.get(GATE1_MAE)
    gate1c = bool(v and not (v[1] > 0))
    print(f"  GATE 1(c): {'PASS' if gate1c else 'FAIL'}")

    gate1 = gate1a and gate1b and gate1c
    print(f"\n  GATE 1 (a AND b AND c): {'MET' if gate1 else 'NOT MET'}   "
          f"[a={gate1a} b={gate1b} c={gate1c}]")

    # ------------------------------------------------------------- GATE 2
    print("\n" + "=" * 96)
    print("GATE 2 — guards: no CI-clean regression on "
          + ", ".join(GUARD_BINARY + GUARD_MAE))
    print("=" * 96)
    g2 = report(da, db, GUARD_BINARY + GUARD_MAE, "control_vec", "noweights",
                show_positional=True)
    gate2 = (len(g2) == len(GUARD_BINARY) + len(GUARD_MAE)
             and all(not (val[1] > 0) for val in g2.values()))
    for fam, val in g2.items():
        print(f"  {fam:<28} {'REGRESSED CI-clean' if val[1] > 0 else 'ok'}")
    print(f"  GATE 2: {'MET' if gate2 else 'NOT MET'}")

    # ------------------------------------------------------------ CONTEXT
    print("\n" + "=" * 96)
    print("CONTEXT — full family scan (cannot flip the verdict)")
    print("=" * 96)
    all_fams = sorted(set(da[0]["obs"]) - {"cricsheet_id", "display_match_id",
                                           "match_identity_version"})
    scan = report(da, db, all_fams, "control_vec", "noweights")
    clean = [(f, val) for f, val in scan.items()
             if val[1] > 0 or val[2] < 0]
    print("\n  CI-excludes-0 families (either direction):")
    if not clean:
        print("    (none)")
    for f, val in sorted(clean, key=lambda x: x[1][0]):
        print(f"    {f:<34}{val[0]:>+10.4f}  "
              f"[{val[1]:+.4f},{val[2]:+.4f}]  {val[3]}")

    print("\n" + "=" * 96)
    print(f"GATE 1: {'MET' if gate1 else 'NOT MET'} | "
          f"GATE 2: {'MET' if gate2 else 'NOT MET'}")
    mapping = ("LANDED" if gate1 and gate2
               else "TABLED" if gate1 or gate2 else "FAILED")
    print(f"Pre-committed verdict MAPPING (orchestrator decides): {mapping}")


if __name__ == "__main__":
    main()
