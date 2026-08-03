"""B18 gate analysis — empirical extras graft on the promoted i7 stack.

WRITTEN AND COMMITTED BEFORE ANY B18 EVAL OUTPUT EXISTED. The mandatory
self-test reproduces B16's logged numbers from its FROZEN quote run, so the
transform is pinned against an already-published result before a single
fresh B18 number exists (B16 precedent).

B17 attributed 92.7% of the promoted i7 stack's in-play P50 under-prediction
(-4.781/-3.026/-1.946 remaining runs at cps 6/10/15) to carried run mass:
g_i7 = -0.052785 runs per legal ball, of which -0.039559 is the flat 1%+1%
extras graft under-carrying explicit extras. B18 grafts empirical rates AND
credits empirical integer runs per extras event. The falsifiable claim: the
bias shrinks at the quote layer WITHOUT the innings-total regression that
killed D3 on the legacy stack.

Everything here is a deterministic transform of the committed run artifacts;
nothing is fitted, and no arm re-draws anything.

Quote correction transform (imported verbatim from b15_gate_analysis.
evaluate_arm — the code is the spec, not the prose):

    c50 = p50 - shift        (shift == 0 on every B18 arm)
    c10 = c50 - scale * (p50 - p10)
    c90 = c50 + scale * (p90 - p50)

RAW is that transform at scale = 1.0, shift = 0 (identity). Coverage is
inclusive. The naive run-rate baseline is NEVER corrected.

PRE-COMMITTED B18 GATE (research/handoff/B18/plan.md)
-----------------------------------------------------
PRIMARY (quote layer, the SAME-SEED s49 twins):
  P-A  |P50 bias| shrinks at ALL THREE checkpoints — point test,
       |bias_b18| < |bias_raw| per cp on the same-seed twins.
  P-B  pooled paired dMAE (b18 P50 vs naive run-rate extrapolation,
       cluster-boot by match, 2000 resamples, boot seed 29) has CI hi < 0.
  PRIMARY MET = P-A AND P-B.

GUARDS:
  G-1  recipe-B, paired cluster-boot 2000/seed29 vs
       models/auto/d16/detail_noweights_raw_s46_n261.json: dBrier on
       innings_runs_ou_{160_5,170_5,180_5} and pp_total_ou_{45_5,50_5,55_5}
       — NO line CI-clean worse (D3's exact legacy failure mode).
  G-2  batter_runs_mae: no CI-clean regression (same pairing).
  G-3  coverage on the certified arm: apply the B15 scales (1.19/1.09/1.26,
       shift 0) to the b18 s49 quotes — inclusive P10-P90 coverage at cps
       6/10/15 ALL in [0.70, 0.90].
  GUARDS MET = G-1 AND G-2 AND G-3.

Verdict mapping is applied by the ORCHESTRATOR, not by this script. The
full 33-family scan is printed as context and CANNOT flip the gate.

Run:
  # mandatory self-test (no B18 artifact required)
  uv run python scripts/auto/b18_gate_analysis.py --self-test
  # the B18 arms
  uv run python scripts/auto/b18_gate_analysis.py \
      --quotes-raw models/auto/b18/quotes_raw_s49_n261.json \
      --quotes-b18 models/auto/b18/quotes_b18_s49_n261.json \
      --detail-b18 models/auto/b18/detail_b18_s46_n261.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "auto"))

from b15_gate_analysis import (  # noqa: E402
    BOOT_REPS,
    BOOT_SEED,
    CHECKPOINTS,
    COV_LO,
    COV_HI,
    evaluate_arm,
    print_arm,
)
from b12_gate_analysis import (  # noqa: E402
    N_BOOT,
    flag_of,
    load,
    report,
)

B15_CALIBRATOR = REPO / "models/auto/b15/quote_calibrator_scale_only.json"
B16_FROZEN_QUOTES = REPO / "models/auto/b16/quotes_i7_s48_n261.json"
D16_BASELINE = REPO / "models/auto/d16/detail_noweights_raw_s46_n261.json"

# The B15 record's scales, hard-asserted (never refit here).
EXPECTED_B15_FIT = {
    "6": {"shift": 0.0, "scale": 1.19},
    "10": {"shift": 0.0, "scale": 1.09},
    "15": {"shift": 0.0, "scale": 1.26},
}
RAW_FIT = {str(cp): {"shift": 0.0, "scale": 1.0} for cp in CHECKPOINTS}

# B16's logged numbers (research/handoff/B16/result.md +
# research/handoff/B16/raw/gate_output.txt), checked at the printed 3-dp
# precision — the precision at which B16's result was recorded.
B16_SELFTEST = {
    "raw_bias": "-4.781/-3.026/-1.946",
    "raw_coverage": "0.787/0.798/0.684",
    "b15_coverage": "0.822/0.838/0.792",
    "pooled_raw_dmae": "-3.417 [-4.878, -2.066]",
    "raw_mae": "20.678/16.579/11.986",
    "naive_mae": "25.897/20.000/13.575",
}

GUARD_LINES = ["innings_runs_ou_160_5", "innings_runs_ou_170_5",
               "innings_runs_ou_180_5", "pp_total_ou_45_5",
               "pp_total_ou_50_5", "pp_total_ou_55_5"]
GUARD_MAE = "batter_runs_mae"

LABEL_A = "d16_raw"
LABEL_B = "b18_graft"


# --------------------------------------------------------------- helpers
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


def joined(res, key, fmt="{:.3f}"):
    return "/".join(fmt.format(res["per_cp"][c][key])
                    for c in CHECKPOINTS if c in res["per_cp"])


def load_rows(path):
    with open(path) as f:
        payload = json.load(f)
    return payload["rows"], payload


def describe(payload, path):
    cfg = payload.get("config", {})
    rows = payload["rows"]
    print(f"  {path}")
    print(f"    config: model={cfg.get('model')} "
          f"stats_version={cfg.get('stats_version')} "
          f"ball_calibrator={cfg.get('ball_calibrator')} "
          f"n_sims={cfg.get('n_sims')} seed={cfg.get('seed')}")
    print(f"    usage_json={cfg.get('usage_json')} "
          f"quote_center={cfg.get('quote_center')} "
          f"elapsed={cfg.get('elapsed_s', 0):.1f}s")
    print(f"    rows: {len(rows)} from "
          f"{len({r['match_id'] for r in rows})} matches "
          f"({len(payload.get('skips', []))} matches skipped)")


# ------------------------------------------------------------- self-test
def self_test():
    print("B18 SELF-TEST — reproduce B16's logged numbers on its FROZEN "
          "i7 quotes\n")
    _, b15_fit = load_fit(B15_CALIBRATOR, EXPECTED_B15_FIT, "B15")
    rows, payload = load_rows(B16_FROZEN_QUOTES)
    describe(payload, str(B16_FROZEN_QUOTES))
    print(f"  calibrator: {B15_CALIBRATOR} "
          "(scales 1.19/1.09/1.26, shift 0 — asserted, never refit)")
    print(f"  bootstrap: {BOOT_REPS} reps, seed {BOOT_SEED}, "
          "cluster by match\n")

    raw_arm = evaluate_arm(rows, RAW_FIT, use_shift=False,
                           label="RAW (self-test)")
    b15_arm = evaluate_arm(rows, b15_fit, use_shift=False,
                           label="B15 scales (self-test)")
    print_arm(b15_arm, show_raw=True)

    got = {
        "raw_bias": joined(raw_arm, "bias_raw"),
        "raw_coverage": joined(raw_arm, "cov_raw"),
        "b15_coverage": joined(b15_arm, "cov_corr"),
        "pooled_raw_dmae": (f"{raw_arm['pooled_raw_dmae']:.3f} "
                            f"[{raw_arm['pooled_raw_ci'][0]:.3f}, "
                            f"{raw_arm['pooled_raw_ci'][1]:.3f}]"),
        "raw_mae": joined(raw_arm, "mae_raw"),
        "naive_mae": joined(raw_arm, "mae_naive"),
    }
    print("=== SELF-TEST vs B16's logged numbers (3 dp = recorded precision) "
          "===")
    ok = True
    for k, exp in B16_SELFTEST.items():
        hit = got[k] == exp
        ok = ok and hit
        print(f"  {k:>18}  expected {exp:<28}  got {got[k]:<28}  "
              f"{'PASS' if hit else 'FAIL'}")
    print(f"\n  SELF-TEST: {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(
            "SELF-TEST FAILED — the transform does not reproduce B16; do NOT "
            "use this script on the B18 draws.")


# ------------------------------------------------------------- main gate
def run(quotes_raw, quotes_b18, detail_baseline, detail_b18):
    print("B18 gate analysis — empirical extras graft on the promoted i7 "
          "stack\n")
    _, b15_fit = load_fit(B15_CALIBRATOR, EXPECTED_B15_FIT, "B15")

    # ---------------------------------------------- PRIMARY (quote layer)
    print("=" * 96)
    print("PRIMARY — quote layer, SAME-SEED (s49) twins")
    print("=" * 96)
    rows_raw, pay_raw = load_rows(quotes_raw)
    rows_b18, pay_b18 = load_rows(quotes_b18)
    print("  RAW twin (production stack, no sidecar):")
    describe(pay_raw, quotes_raw)
    print("  B18 twin (same booster + extras_graft_v1 sidecar):")
    describe(pay_b18, quotes_b18)

    seed_raw = pay_raw.get("config", {}).get("seed")
    seed_b18 = pay_b18.get("config", {}).get("seed")
    if seed_raw != seed_b18:
        raise SystemExit(
            f"twins are NOT same-seed: raw {seed_raw} vs b18 {seed_b18}")
    print(f"  same-seed check: seed {seed_raw} on both twins — OK")
    n_raw = len({r["match_id"] for r in rows_raw})
    n_b18 = len({r["match_id"] for r in rows_b18})
    print(f"  row/match parity: raw {len(rows_raw)}/{n_raw}  "
          f"b18 {len(rows_b18)}/{n_b18}  "
          f"{'OK' if (len(rows_raw), n_raw) == (len(rows_b18), n_b18) else 'MISMATCH'}")
    print()

    raw_arm = evaluate_arm(rows_raw, RAW_FIT, use_shift=False,
                           label="RAW twin (no sidecar), uncorrected")
    b18_raw_arm = evaluate_arm(rows_b18, RAW_FIT, use_shift=False,
                               label="B18 twin (graft), uncorrected")
    b18_b15_arm = evaluate_arm(rows_b18, b15_fit, use_shift=False,
                               label="B18 twin (graft) + B15 scales")
    print_arm(raw_arm, show_raw=False)
    print_arm(b18_raw_arm, show_raw=False)
    print_arm(b18_b15_arm, show_raw=True)

    # --- P-A: |bias| shrinks at all three checkpoints
    print("=== P-A: |P50 bias| shrinks at ALL THREE checkpoints "
          "(point test, same-seed twins) ===")
    pa_flags = []
    for cp in CHECKPOINTS:
        if cp not in raw_arm["per_cp"] or cp not in b18_raw_arm["per_cp"]:
            continue
        br = raw_arm["per_cp"][cp]["bias_raw"]
        bb = b18_raw_arm["per_cp"][cp]["bias_raw"]
        ok = abs(bb) < abs(br)
        pa_flags.append(ok)
        print(f"  cp{cp:>2}: bias raw {br:+7.3f}  b18 {bb:+7.3f}   "
              f"|raw| {abs(br):6.3f} -> |b18| {abs(bb):6.3f}   "
              f"shrink {abs(br) - abs(bb):+7.3f}   "
              f"{'SHRANK' if ok else 'DID NOT SHRINK'}")
    pa = bool(pa_flags) and all(pa_flags)
    print(f"  P-A: {'MET' if pa else 'NOT MET'}\n")

    # --- P-B: pooled paired dMAE of the B18 P50 vs naive
    print("=== P-B: pooled paired dMAE (B18 P50 - naive), cluster-boot by "
          f"match, {BOOT_REPS} reps seed {BOOT_SEED}, CI hi < 0 ===")
    for cp in CHECKPOINTS:
        d = b18_raw_arm["per_cp"].get(cp)
        if d is None:
            continue
        print(f"  cp{cp:>2} (n={d['n']}): b18 MAE {d['mae_raw']:7.3f}  "
              f"naive {d['mae_naive']:7.3f}  "
              f"dMAE {d['raw_dmae']:+7.3f} "
              f"[{d['raw_dmae_ci'][0]:+.3f}, {d['raw_dmae_ci'][1]:+.3f}]")
    pooled = b18_raw_arm["pooled_raw_dmae"]
    pooled_ci = b18_raw_arm["pooled_raw_ci"]
    pb = pooled_ci[1] < 0
    print(f"  pooled paired dMAE ({b18_raw_arm['pooled_n']} rows): "
          f"{pooled:+.3f} [{pooled_ci[0]:+.3f}, {pooled_ci[1]:+.3f}]")
    print(f"  P-B: {'MET' if pb else 'NOT MET'}")
    print(f"  (context — RAW twin pooled dMAE: "
          f"{raw_arm['pooled_raw_dmae']:+.3f} "
          f"[{raw_arm['pooled_raw_ci'][0]:+.3f}, "
          f"{raw_arm['pooled_raw_ci'][1]:+.3f}])")
    primary = pa and pb
    print(f"\n  PRIMARY (P-A AND P-B): {'MET' if primary else 'NOT MET'}   "
          f"[P-A={pa} P-B={pb}]\n")

    # ------------------------------------------------------------ G-3
    print("=" * 96)
    print(f"G-3 — B15-scaled coverage on the B18 arm in "
          f"[{COV_LO}, {COV_HI}] at ALL THREE cps")
    print("=" * 96)
    g3_flags = []
    for cp in CHECKPOINTS:
        d = b18_b15_arm["per_cp"].get(cp)
        if d is None:
            continue
        g3_flags.append(d["cov_ok"])
        print(f"  cp{cp:>2}: scale {d['scale']:.2f}  coverage "
              f"{d['cov_corr']:.3f} [{d['cov_corr_ci'][0]:.3f}, "
              f"{d['cov_corr_ci'][1]:.3f}]  "
              f"{'IN BAND' if d['cov_ok'] else 'OUT OF BAND'}   "
              f"(raw context {d['cov_raw']:.3f})")
    g3 = bool(g3_flags) and all(g3_flags)
    print(f"  B18+B15 coverage: {joined(b18_b15_arm, 'cov_corr')}   "
          f"RAW-twin context: {joined(raw_arm, 'cov_raw')}")
    print(f"  G-3: {'MET' if g3 else 'NOT MET'}\n")

    # ---------------------------------------------- G-1 / G-2 (prop layer)
    print("=" * 96)
    print("G-1 / G-2 — recipe-B paired vs the canonical D16 baseline")
    print("=" * 96)
    da, db = load(detail_baseline), load(detail_b18)
    print(f"  baseline ({LABEL_A}): {len(da)} matches "
          f"({Path(detail_baseline).name})")
    print(f"  b18      ({LABEL_B}): {len(db)} matches "
          f"({Path(detail_b18).name})")
    print(f"  pairing: cluster bootstrap by match, n_boot={N_BOOT}, "
          f"seed={BOOT_SEED}; delta = {LABEL_B} - {LABEL_A} "
          "(positive = B18 worse)\n")

    print(f"G-1 — no CI-clean regression on any of: {', '.join(GUARD_LINES)}")
    g1_res = report(da, db, GUARD_LINES, LABEL_A, LABEL_B,
                    show_positional=True)
    g1 = (len(g1_res) == len(GUARD_LINES)
          and all(not (v[1] > 0) for v in g1_res.values()))
    for fam, v in g1_res.items():
        print(f"  {fam:<28} {'REGRESSED CI-clean' if v[1] > 0 else 'ok'}")
    if len(g1_res) != len(GUARD_LINES):
        missing = [f for f in GUARD_LINES if f not in g1_res]
        print(f"  MISSING FAMILIES (gate cannot be met): {missing}")
    print(f"  G-1: {'MET' if g1 else 'NOT MET'}\n")

    print(f"G-2 — {GUARD_MAE} must NOT regress CI-clean (CI lo <= 0)")
    g2_res = report(da, db, [GUARD_MAE], LABEL_A, LABEL_B,
                    show_positional=True)
    v = g2_res.get(GUARD_MAE)
    g2 = bool(v and not (v[1] > 0))
    print(f"  G-2: {'MET' if g2 else 'NOT MET'}\n")

    guards = g1 and g2 and g3
    print(f"  GUARDS (G-1 AND G-2 AND G-3): "
          f"{'MET' if guards else 'NOT MET'}   "
          f"[G-1={g1} G-2={g2} G-3={g3}]\n")

    # -------------------------------------------------------- 33-fam scan
    print("=" * 96)
    print("CONTEXT — full family scan (cannot flip the gate)")
    print("=" * 96)
    all_fams = sorted(set(da[0]["obs"]) - {"cricsheet_id", "display_match_id",
                                           "match_identity_version"})
    print(f"  {len(all_fams)} families scanned\n")
    scan = report(da, db, all_fams, LABEL_A, LABEL_B)
    movers = [(f, v) for f, v in scan.items() if v[1] > 0 or v[2] < 0]
    better = [(f, v) for f, v in movers if v[2] < 0]
    worse = [(f, v) for f, v in movers if v[1] > 0]
    print(f"\n  CI-excludes-0 families: {len(movers)} "
          f"({len(better)} favorable, {len(worse)} regressions)")
    for f, v in sorted(movers, key=lambda x: x[1][0]):
        print(f"    {f:<34}{v[0]:>+10.4f}  [{v[1]:+.4f},{v[2]:+.4f}]  {v[3]}")
    if worse:
        print(f"\n  !! CI-clean REGRESSIONS anywhere: "
              f"{[f for f, _ in worse]}")

    print("\n" + "=" * 96)
    print(f"PRIMARY: {'MET' if primary else 'NOT MET'} "
          f"(P-A={pa}, P-B={pb}) | "
          f"GUARDS: {'MET' if guards else 'NOT MET'} "
          f"(G-1={g1}, G-2={g2}, G-3={g3})")
    print("verdict mapping (orchestrator applies it, not this script): "
          "BOTH -> LANDED, exactly one -> TABLED, neither -> FAILED")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--quotes-raw",
                    default=str(REPO / "models/auto/b18/quotes_raw_s49_n261.json"))
    ap.add_argument("--quotes-b18",
                    default=str(REPO / "models/auto/b18/quotes_b18_s49_n261.json"))
    ap.add_argument("--detail-baseline", default=str(D16_BASELINE))
    ap.add_argument("--detail-b18",
                    default=str(REPO / "models/auto/b18/detail_b18_s46_n261.json"))
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return
    run(args.quotes_raw, args.quotes_b18, args.detail_baseline,
        args.detail_b18)


if __name__ == "__main__":
    main()
