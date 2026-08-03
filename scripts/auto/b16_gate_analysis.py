"""B16 gate analysis — quote-layer coverage re-check on the PROMOTED i7 stack.

The B15 scale-only quote calibrator (record
`models/auto/b15/quote_calibrator_scale_only.json`, scales 1.19/1.09/1.26,
shifts 0) was fit against the LEGACY balanced-weights ball engine's
under-dispersion. A13's STEP 0 showed the promoted i7 no-weights stack
disperses materially wider at the prop layer, so on the migrated quote path
the B15 scales may over-widen the bands and the RAW quotes may already be in
band with NO quote calibrator (D17's null, one layer up).

Everything here is a deterministic transform of ONE fresh i7 test quote run;
no arm re-draws and no arm refits anything on test.

Correction transform (imported verbatim from b15_gate_analysis.evaluate_arm,
itself copied from b14 — the code is the spec, not the prose):

    c50 = p50 - shift        (shift == 0 on every B16 arm)
    c10 = c50 - scale * (p50 - p10)
    c90 = c50 + scale * (p90 - p50)

RAW is that same transform at scale = 1.0, shift = 0 (identity). Coverage is
inclusive (c10 <= actual <= c90). The naive run-rate baseline is NEVER
corrected — it is the fixed comparator.

PRE-COMMITTED B16 GATE PAIR
---------------------------
GATE 1 (skill retained on the i7 path): pooled paired dMAE
  (|raw sim P50 err| - |naive err|), cluster-bootstrapped by match
  (2000 reps, seed 29 — the exact B14/B15 contract), CI hi < 0.

GATE 2 (coverage in band): at least one PRE-DECLARED arm keeps inclusive
  P10-P90 coverage in [0.70, 0.90] POINT ESTIMATE at ALL THREE checkpoints
  (6/10/15). Arm preference is PRE-COMMITTED (parsimony order, D17 mirror):
    1. RAW (no quote calibrator)   -- if in band, RAW is the certified arm
                                      even if a scaled arm looks more
                                      centered (preferring one post hoc is
                                      slice-shopping).
    2. B15 scales (existing record applied unchanged, shift 0)
    3. Refit scale-only (i7 val fit) -- only ever run if 1 and 2 both fail.

Verdict mapping (applied by the ORCHESTRATOR, not by this script):
  BOTH gates met -> LANDED (record the winning arm; if RAW wins, the outcome
  is "calibrator retired for the i7 quote path"; B15 scales remain the record
  for the legacy path regardless). Exactly one -> TABLED. Neither -> FAILED.
Per-cp coverage CIs are context only; the gate is on point estimates.

Run:
  # mandatory self-test: reproduce B15's LANDED numbers on its frozen quotes
  uv run python scripts/auto/b16_gate_analysis.py --self-test
  # the i7 arms
  uv run python scripts/auto/b16_gate_analysis.py \
      --quotes models/auto/b16/quotes_i7_s48_n261.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from b15_gate_analysis import (  # noqa: E402
    BOOT_REPS,
    BOOT_SEED,
    CHECKPOINTS,
    COV_LO,
    COV_HI,
    evaluate_arm,
    print_arm,
)

B15_CALIBRATOR = "models/auto/b15/quote_calibrator_scale_only.json"
B15_FROZEN_QUOTES = "models/auto/b15/quotes_s45_n261.json"

# The B15 record's scales, hard-asserted (never refit here).
EXPECTED_B15_FIT = {
    "6": {"shift": 0.0, "scale": 1.19},
    "10": {"shift": 0.0, "scale": 1.09},
    "15": {"shift": 0.0, "scale": 1.26},
}

# B15's LANDED numbers as logged in research/reports/auto/B15.md and
# research/handoff/B15/raw/gate_output.txt. Checked at the printed precision
# (3 dp), which is the precision at which B15's result was recorded.
B15_SELFTEST = {
    "scale_only_coverage": "0.818/0.834/0.768",
    "raw_coverage": "0.755/0.791/0.660",
    "pooled_raw_dmae": "-3.131 [-4.909, -1.356]",
    "scale_only_mae": "20.773/16.990/12.338",
    "naive_mae": "25.897/20.000/13.575",
}

RAW_FIT = {str(cp): {"shift": 0.0, "scale": 1.0} for cp in CHECKPOINTS}


def load_fit(path, expect=None, label=""):
    with open(path) as f:
        cal = json.load(f)
    fit = cal["per_checkpoint"]
    if expect is not None:
        for cp, exp in expect.items():
            got_shift = float(fit[cp]["shift"])
            got_scale = float(fit[cp]["scale"])
            assert got_shift == exp["shift"], (
                f"{label} cp{cp} shift {got_shift!r} != {exp['shift']!r}")
            assert got_scale == exp["scale"], (
                f"{label} cp{cp} scale {got_scale!r} != {exp['scale']!r}")
    return cal, fit


def cov_string(res):
    return "/".join(f"{res['per_cp'][c]['cov_corr']:.3f}"
                    for c in CHECKPOINTS if c in res["per_cp"])


def raw_cov_string(res):
    return "/".join(f"{res['per_cp'][c]['cov_raw']:.3f}"
                    for c in CHECKPOINTS if c in res["per_cp"])


def in_band(res):
    return all(res["per_cp"].get(cp, {}).get("cov_ok", False)
               for cp in CHECKPOINTS)


def run(quotes_path, b15_cal_path, refit_cal_path, self_test):
    _, b15_fit = load_fit(b15_cal_path, EXPECTED_B15_FIT, "B15")

    refit_fit = None
    if refit_cal_path:
        # A refit arm may carry any scale, but the shift MUST be 0 (B15 rule:
        # never fit a location term — val->test bias sign mismatch).
        _, refit_fit = load_fit(refit_cal_path)
        for cp in CHECKPOINTS:
            assert float(refit_fit[str(cp)]["shift"]) == 0.0, (
                f"refit cp{cp} shift must be 0.0 (scale-only rule)")

    with open(quotes_path) as f:
        payload = json.load(f)
    rows = payload["rows"]
    cfg = payload.get("config", {})

    print("B16 gate analysis — quote-layer coverage on the promoted i7 stack")
    print(f"  quotes: {quotes_path}")
    print(f"  config: model={cfg.get('model')} "
          f"stats_version={cfg.get('stats_version')} "
          f"ball_calibrator={cfg.get('ball_calibrator')}")
    print(f"          n_sims={cfg.get('n_sims')} seed={cfg.get('seed')} "
          f"quote_center={cfg.get('quote_center')} "
          f"usage_json={cfg.get('usage_json')} "
          f"elapsed={cfg.get('elapsed_s', 0):.1f}s")
    print(f"  rows: {len(rows)} from "
          f"{len({r['match_id'] for r in rows})} matches "
          f"({len(payload.get('skips', []))} matches skipped)")
    print(f"  B15 calibrator: {b15_cal_path} — assertion PASS "
          "(scales 1.19/1.09/1.26, shifts 0; NO refit in this script)")
    if refit_fit is not None:
        print(f"  refit calibrator: {refit_cal_path} — scales " + "/".join(
            f"{float(refit_fit[str(c)]['scale']):.2f}" for c in CHECKPOINTS))
    print(f"  bootstrap: {BOOT_REPS} reps, seed {BOOT_SEED}, "
          "cluster by match\n")

    arms = []
    raw_arm = evaluate_arm(rows, RAW_FIT, use_shift=False,
                           label="1. RAW (no quote calibrator)")
    arms.append(("RAW", raw_arm))
    b15_arm = evaluate_arm(rows, b15_fit, use_shift=False,
                           label="2. B15 scales (1.19/1.09/1.26, shift 0)")
    arms.append(("B15", b15_arm))
    if refit_fit is not None:
        refit_arm = evaluate_arm(rows, refit_fit, use_shift=False,
                                 label="3. refit scale-only (i7 val fit)")
        arms.append(("REFIT", refit_arm))

    print_arm(raw_arm, show_raw=False)
    print_arm(b15_arm, show_raw=True)
    if refit_fit is not None:
        print_arm(refit_arm, show_raw=False)

    # ---------------- GATE 1 (skill retained on the i7 path) -------------
    # RAW P50 == every arm's P50 (scale-only leaves P50 untouched), so the
    # raw dMAE rows are identical across arms by construction.
    pooled_raw = raw_arm["pooled_raw_dmae"]
    pooled_raw_ci = raw_arm["pooled_raw_ci"]
    gate1 = pooled_raw_ci[1] < 0
    print("=== GATE 1: raw sim P50 vs naive (skill retained on i7) ===")
    for cp in CHECKPOINTS:
        d = raw_arm["per_cp"].get(cp)
        if d is None:
            continue
        print(f"  cp{cp:>2} (n={d['n']}): raw MAE {d['mae_raw']:7.3f}  "
              f"naive {d['mae_naive']:7.3f}  "
              f"dMAE {d['raw_dmae']:+7.3f} "
              f"[{d['raw_dmae_ci'][0]:+.3f}, {d['raw_dmae_ci'][1]:+.3f}]")
    print(f"  pooled paired dMAE (raw - naive, {raw_arm['pooled_n']} rows): "
          f"{pooled_raw:+.3f} [{pooled_raw_ci[0]:+.3f}, "
          f"{pooled_raw_ci[1]:+.3f}]")
    print(f"  GATE 1 (CI hi < 0): {'MET' if gate1 else 'NOT MET'}\n")

    # ---------------- GATE 2 (coverage in band) --------------------------
    print("=== GATE 2: inclusive P10-P90 coverage in "
          f"[{COV_LO}, {COV_HI}] at ALL THREE cps ===")
    for name, res in arms:
        flags = [res["per_cp"].get(cp, {}).get("cov_ok") for cp in CHECKPOINTS]
        print(f"  {name:>5}: coverage {cov_string(res)}  in-band {flags}  "
              f"-> {'IN BAND (all 3)' if in_band(res) else 'FAILS'}")
    winner = None
    for name, res in arms:  # arms are already in pre-committed preference order
        if in_band(res):
            winner = (name, res)
            break
    gate2 = winner is not None
    print(f"  pre-committed preference order: "
          f"{' > '.join(n for n, _ in arms)}")
    if gate2:
        print(f"  GATE 2: MET — certified arm = {winner[0]} "
              f"({winner[1]['label']})")
        if winner[0] == "RAW":
            print("  => outcome if LANDED: quote calibrator RETIRED for the "
                  "i7 quote path (B15 scales remain the record for the "
                  "legacy path).")
    else:
        print("  GATE 2: NOT MET — no pre-declared arm is in band at all "
              "three checkpoints.")
    print()

    print("=== PRE-COMMITTED B16 GATE PAIR ===")
    print(f"GATE 1 (pooled raw dMAE CI hi < 0): "
          f"{'MET' if gate1 else 'NOT MET'} "
          f"(pooled {pooled_raw:+.3f} hi {pooled_raw_ci[1]:+.3f})")
    print(f"GATE 2 (an arm in band at all 3 cps): "
          f"{'MET' if gate2 else 'NOT MET'}"
          + (f" (arm {winner[0]}, coverage {cov_string(winner[1])})"
             if gate2 else ""))
    print("verdict mapping (orchestrator applies it, not this script): "
          "BOTH -> LANDED, exactly one -> TABLED, neither -> FAILED")
    return arms, raw_arm, b15_arm


def self_test():
    print("B16 SELF-TEST — reproduce B15's LANDED numbers on its frozen "
          "quotes\n")
    _, b15_fit = load_fit(B15_CALIBRATOR, EXPECTED_B15_FIT, "B15")
    with open(B15_FROZEN_QUOTES) as f:
        payload = json.load(f)
    rows = payload["rows"]
    print(f"  quotes: {B15_FROZEN_QUOTES}  "
          f"({len(rows)} rows from "
          f"{len({r['match_id'] for r in rows})} matches, "
          f"{len(payload.get('skips', []))} skipped)")
    print(f"  calibrator: {B15_CALIBRATOR} (scales 1.19/1.09/1.26, shift 0)\n")

    arm = evaluate_arm(rows, b15_fit, use_shift=False,
                       label="B15 scale-only (self-test)")
    print_arm(arm, show_raw=True)

    got = {
        "scale_only_coverage": cov_string(arm),
        "raw_coverage": raw_cov_string(arm),
        "pooled_raw_dmae": (f"{arm['pooled_raw_dmae']:.3f} "
                            f"[{arm['pooled_raw_ci'][0]:.3f}, "
                            f"{arm['pooled_raw_ci'][1]:.3f}]"),
        "scale_only_mae": "/".join(
            f"{arm['per_cp'][c]['mae_corr']:.3f}" for c in CHECKPOINTS),
        "naive_mae": "/".join(
            f"{arm['per_cp'][c]['mae_naive']:.3f}" for c in CHECKPOINTS),
    }
    print("=== SELF-TEST vs B15's logged LANDED numbers "
          "(research/reports/auto/B15.md, 3 dp = the recorded precision) ===")
    ok = True
    for k, exp in B15_SELFTEST.items():
        hit = got[k] == exp
        ok = ok and hit
        print(f"  {k:>20}  expected {exp:<28}  got {got[k]:<28}  "
              f"{'PASS' if hit else 'FAIL'}")

    # The RAW arm must be the identity transform: identical to the raw rows.
    raw_arm = evaluate_arm(rows, RAW_FIT, use_shift=False, label="raw check")
    raw_ok = (cov_string(raw_arm) == got["raw_coverage"]
              and f"{raw_arm['pooled_dmae']:.3f}"
              == f"{arm['pooled_raw_dmae']:.3f}")
    ok = ok and raw_ok
    print(f"  {'RAW arm == raw rows':>20}  expected "
          f"{got['raw_coverage']:<28}  got {cov_string(raw_arm):<28}  "
          f"{'PASS' if raw_ok else 'FAIL'}")

    print(f"\n  SELF-TEST: {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(
            "SELF-TEST FAILED — the transform does not reproduce B15; do NOT "
            "use this script on the i7 draws.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quotes", default=None,
                    help="i7 test quote run JSON (required unless --self-test)")
    ap.add_argument("--b15-calibrator", default=B15_CALIBRATOR)
    ap.add_argument("--refit-calibrator", default=None,
                    help="optional i7 val-fit scale-only calibrator (arm 3); "
                         "only ever produced if arms 1 and 2 both fail GATE 2")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return
    if not args.quotes:
        raise SystemExit("--quotes is required (or pass --self-test)")
    run(args.quotes, args.b15_calibrator, args.refit_calibrator, False)


if __name__ == "__main__":
    main()
