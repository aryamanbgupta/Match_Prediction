"""B15 gate analysis — scale-only vs B14-full quote calibrator, fresh draws.

B14 LANDED a val-fit per-checkpoint quote calibrator (shift + band scale)
but its shift term was mis-signed val -> test: fitted P50 shifts are all
negative (sim UNDER-predicts on the 2024-12 -> 2025-06 val pool) while
frozen-test raw bias is positive (+4.670/+3.204/+0.514 at cps 6/10/15).
The shift therefore moved test P50 the WRONG way; essentially all of
B14's win came from the band-widening scale term.

B15 tests a SCALE-ONLY arm (shift := 0, same scales, NO refit) on FRESH
test quote draws (seed 45), because re-scoring the same frozen s43 quotes
after seeing B14's decomposition would be post-hoc selection.

Correction transform (copied verbatim from b14_gate_analysis.py — the
code is the spec, not the prose):

    c50 = p50 - shift
    c10 = c50 - scale * (p50 - p10)
    c90 = c50 + scale * (p90 - p50)

with the scale-only arm using shift = 0.0 and the B14-full arm using the
val-fit shift. Coverage is inclusive (c10 <= actual <= c90). The naive
run-rate baseline is NEVER corrected — it is the fixed comparator.

Pre-committed B15 gate (evaluated on the SCALE-ONLY arm only):

  PRIMARY-A (coverage): corrected inclusive P10-P90 coverage point
    estimate in [0.70, 0.90] at ALL THREE checkpoints (6, 10, 15).

  PRIMARY-B (MAE): pooled paired dMAE (|corrected P50 err| - |naive err|)
    cluster-bootstrapped by match (2000 reps, seed 29 — the exact
    b5/b14_gate_analysis.py construction) has CI hi < 0, AND corrected
    P50 MAE beats naive MAE at EACH checkpoint in point estimate.

SECONDARY (recommendation only, no gate weight): scale-only vs B14-full
pooled dMAE point estimate on the same fresh draws.

Verdict mapping (LANDED/TABLED/FAILED) is applied by the orchestrator,
not by this script.

Run:
  # self-test: reproduce B14's logged frozen-quote numbers
  uv run python scripts/auto/b15_gate_analysis.py --self-test \
      --quotes models/auto/b5/quotes_s43_n261.json
  # fresh draws
  uv run python scripts/auto/b15_gate_analysis.py \
      --quotes models/auto/b15/quotes_s45_n261.json
"""
import argparse
import json
from collections import defaultdict

import numpy as np

BOOT_REPS = 2000
BOOT_SEED = 29
COV_LO, COV_HI = 0.70, 0.90
CHECKPOINTS = (6, 10, 15)

# Exact values that MUST be present in models/auto/b14/quote_calibrator.json.
# No refit ever happens in this script; these are asserted, not fitted.
EXPECTED_FIT = {
    "6": {"shift": -1.4482421875, "scale": 1.19},
    "10": {"shift": -1.78125, "scale": 1.09},
    "15": {"shift": -2.9714566929133857, "scale": 1.26},
}

# B5 frozen-test (seed 43) raw P50 bias, for the diagnostic table.
B5_FROZEN_RAW_BIAS = {6: 4.670, 10: 3.204, 15: 0.514}

# B14's logged frozen-quote numbers, checked by --self-test.
B14_SELFTEST = {
    "pooled": "-2.774 [-4.631, -0.864]",
    "coverage": "0.802/0.802/0.756",
}


def cluster_boot_ci(values_by_match, stat_fn, reps=BOOT_REPS, seed=BOOT_SEED):
    """Percentile 95% CI of stat_fn over match-level resamples.

    Verbatim from b14_gate_analysis.py / b5_gate_analysis.py.
    """
    match_ids = sorted(values_by_match)
    rng = np.random.default_rng(seed)
    n = len(match_ids)
    stats = []
    for _ in range(reps):
        idxs = rng.integers(0, n, size=n)
        sample = []
        for i in idxs:
            sample.extend(values_by_match[match_ids[i]])
        stats.append(stat_fn(np.asarray(sample)))
    return (float(np.percentile(stats, 2.5)),
            float(np.percentile(stats, 97.5)))


def evaluate_arm(rows, fit, use_shift, label):
    """Evaluate one correction arm. Returns a dict of results (no printing)."""
    cps = sorted({r["checkpoint"] for r in rows})
    out = {"label": label, "per_cp": {}, "cps": cps}
    pooled_delta_by_match = defaultdict(list)
    pooled_raw_delta_by_match = defaultdict(list)

    for cp in cps:
        sub = [r for r in rows if r["checkpoint"] == cp]
        a = np.array([r["actual_remaining"] for r in sub], dtype=float)
        p50 = np.array([r["sim_p50"] for r in sub], dtype=float)
        nv = np.array([r["naive_remaining"] for r in sub], dtype=float)
        p10 = np.array([r["sim_p10"] for r in sub], dtype=float)
        p90 = np.array([r["sim_p90"] for r in sub], dtype=float)

        shift = float(fit[str(cp)]["shift"]) if use_shift else 0.0
        scale = float(fit[str(cp)]["scale"])
        # --- correction transform (b14_gate_analysis.py, verbatim) ---
        c50 = p50 - shift
        c10 = c50 - scale * (p50 - p10)
        c90 = c50 + scale * (p90 - p50)

        corr_err = np.abs(c50 - a)
        raw_err = np.abs(p50 - a)
        nv_err = np.abs(nv - a)
        delta = corr_err - nv_err
        raw_delta = raw_err - nv_err
        cov_corr = ((c10 <= a) & (a <= c90)).astype(float)
        cov_raw = ((p10 <= a) & (a <= p90)).astype(float)

        by_match_delta = defaultdict(list)
        by_match_raw_delta = defaultdict(list)
        by_match_cov = defaultdict(list)
        by_match_cov_raw = defaultdict(list)
        for r, dv, rdv, cv, rcv in zip(sub, delta, raw_delta,
                                       cov_corr, cov_raw):
            by_match_delta[r["match_id"]].append(float(dv))
            by_match_raw_delta[r["match_id"]].append(float(rdv))
            by_match_cov[r["match_id"]].append(float(cv))
            by_match_cov_raw[r["match_id"]].append(float(rcv))
            pooled_delta_by_match[r["match_id"]].append(float(dv))
            pooled_raw_delta_by_match[r["match_id"]].append(float(rdv))

        d_lo, d_hi = cluster_boot_ci(by_match_delta, np.mean)
        rd_lo, rd_hi = cluster_boot_ci(by_match_raw_delta, np.mean)
        c_lo, c_hi = cluster_boot_ci(by_match_cov, np.mean)
        cr_lo, cr_hi = cluster_boot_ci(by_match_cov_raw, np.mean)

        out["per_cp"][cp] = {
            "n": len(sub),
            "shift": shift,
            "scale": scale,
            "mae_corr": float(corr_err.mean()),
            "mae_raw": float(raw_err.mean()),
            "mae_naive": float(nv_err.mean()),
            "dmae": float(corr_err.mean() - nv_err.mean()),
            "dmae_ci": (d_lo, d_hi),
            "raw_dmae": float(raw_err.mean() - nv_err.mean()),
            "raw_dmae_ci": (rd_lo, rd_hi),
            "cov_corr": float(cov_corr.mean()),
            "cov_corr_ci": (c_lo, c_hi),
            "cov_raw": float(cov_raw.mean()),
            "cov_raw_ci": (cr_lo, cr_hi),
            "bias_corr": float((c50 - a).mean()),
            "bias_raw": float((p50 - a).mean()),
            "band_corr": float((c90 - c10).mean()),
            "band_raw": float((p90 - p10).mean()),
            "actual_sd": float(a.std()),
            "mae_ok": bool(corr_err.mean() < nv_err.mean()),
            "cov_ok": bool(COV_LO <= float(cov_corr.mean()) <= COV_HI),
        }

    pooled = np.concatenate(
        [np.asarray(v) for v in pooled_delta_by_match.values()])
    p_lo, p_hi = cluster_boot_ci(pooled_delta_by_match, np.mean)
    out["pooled_n"] = int(pooled.size)
    out["pooled_dmae"] = float(pooled.mean())
    out["pooled_ci"] = (p_lo, p_hi)

    pooled_raw = np.concatenate(
        [np.asarray(v) for v in pooled_raw_delta_by_match.values()])
    rp_lo, rp_hi = cluster_boot_ci(pooled_raw_delta_by_match, np.mean)
    out["pooled_raw_dmae"] = float(pooled_raw.mean())
    out["pooled_raw_ci"] = (rp_lo, rp_hi)
    return out


def print_arm(res, show_raw=False):
    print(f"=== ARM: {res['label']} ===")
    for cp in res["cps"]:
        d = res["per_cp"][cp]
        print(f"checkpoint {cp:>2} (n={d['n']}):  "
              f"applied shift {d['shift']:+.4f} scale {d['scale']:.2f}")
        print(f"  MAE  corr(P50) {d['mae_corr']:7.3f}  "
              f"naive {d['mae_naive']:7.3f}  "
              f"dMAE {d['dmae']:+7.3f} "
              f"[{d['dmae_ci'][0]:+.3f}, {d['dmae_ci'][1]:+.3f}]  "
              f"{'CORRECTED BETTER' if d['mae_ok'] else 'NAIVE BETTER'}")
        if show_raw:
            print(f"  MAE  raw (P50) {d['mae_raw']:7.3f}  "
                  f"naive {d['mae_naive']:7.3f}  "
                  f"dMAE {d['raw_dmae']:+7.3f} "
                  f"[{d['raw_dmae_ci'][0]:+.3f}, {d['raw_dmae_ci'][1]:+.3f}]"
                  f"   (uncorrected context)")
        print(f"  P10-P90 coverage corr {d['cov_corr']:6.3f} "
              f"[{d['cov_corr_ci'][0]:.3f}, {d['cov_corr_ci'][1]:.3f}]  "
              f"target [{COV_LO}, {COV_HI}]  "
              f"{'IN BAND' if d['cov_ok'] else 'OUT OF BAND'}")
        if show_raw:
            print(f"  P10-P90 coverage raw  {d['cov_raw']:6.3f} "
                  f"[{d['cov_raw_ci'][0]:.3f}, {d['cov_raw_ci'][1]:.3f}]"
                  f"   (uncorrected context)")
        print(f"  context: bias corr P50 {d['bias_corr']:+.3f}  "
              f"bias raw P50 {d['bias_raw']:+.3f}  "
              f"band width corr {d['band_corr']:.1f}  "
              f"raw {d['band_raw']:.1f}  "
              f"actual sd {d['actual_sd']:.1f}")
    print(f"\npooled paired dMAE (corrected - naive, {res['pooled_n']} rows, "
          f"cluster-boot by match): {res['pooled_dmae']:+.3f} "
          f"[{res['pooled_ci'][0]:+.3f}, {res['pooled_ci'][1]:+.3f}]")
    if show_raw:
        print(f"pooled paired dMAE (RAW - naive, uncorrected context): "
              f"{res['pooled_raw_dmae']:+.3f} "
              f"[{res['pooled_raw_ci'][0]:+.3f}, "
              f"{res['pooled_raw_ci'][1]:+.3f}]")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrator",
                    default="models/auto/b14/quote_calibrator.json")
    ap.add_argument("--quotes", default="models/auto/b15/quotes_s45_n261.json")
    ap.add_argument("--self-test", action="store_true",
                    help="check the B14-full arm reproduces B14's logged "
                         "frozen-quote numbers")
    args = ap.parse_args()

    with open(args.calibrator) as f:
        cal = json.load(f)
    fit = cal["per_checkpoint"]

    # --- hard assertion: the calibrator is exactly B14's, no refit ---
    for cp, exp in EXPECTED_FIT.items():
        got_shift = float(fit[cp]["shift"])
        got_scale = float(fit[cp]["scale"])
        assert got_shift == exp["shift"], (
            f"cp{cp} shift {got_shift!r} != expected {exp['shift']!r}")
        assert got_scale == exp["scale"], (
            f"cp{cp} scale {got_scale!r} != expected {exp['scale']!r}")

    with open(args.quotes) as f:
        payload = json.load(f)
    rows = payload["rows"]
    cfg = payload.get("config", {})

    print("B15 gate analysis — scale-only vs B14-full quote calibrator")
    print(f"  calibrator: {args.calibrator} "
          f"(fit on {cal.get('source_quotes')}, "
          f"val target coverage {cal.get('target_val_coverage')})")
    print("  calibrator assertion: PASS — all 6 val-fit values match "
          "B14 exactly (NO refit anywhere in this script)")
    print(f"  TEST quotes: {args.quotes}")
    print(f"  config: n_sims={cfg.get('n_sims')} seed={cfg.get('seed')} "
          f"quote_center={cfg.get('quote_center')} "
          f"usage_json={cfg.get('usage_json')} "
          f"elapsed={cfg.get('elapsed_s', 0):.1f}s")
    print(f"  rows: {len(rows)} from "
          f"{len({r['match_id'] for r in rows})} matches "
          f"({len(payload.get('skips', []))} matches skipped)")
    print("  val-fit correction: " + "  ".join(
        f"cp{c}: shift {fit[c]['shift']:+.4f} scale {fit[c]['scale']:.2f}"
        for c in sorted(fit, key=int)))
    print(f"  bootstrap: {BOOT_REPS} reps, seed {BOOT_SEED}, "
          f"cluster by match\n")

    scale_only = evaluate_arm(rows, fit, use_shift=False,
                              label="scale-only (shift := 0)")
    b14_full = evaluate_arm(rows, fit, use_shift=True,
                            label="B14-full (val-fit shift + scale)")

    # scale-only printed with the raw/uncorrected context rows, since its
    # P50 IS the raw P50 (only the band changes).
    print_arm(scale_only, show_raw=True)
    print_arm(b14_full, show_raw=False)

    # --- diagnostic bias table (no gate weight) ---
    print("=== DIAGNOSTIC: per-checkpoint bias (no gate weight) ===")
    hdr = (f"{'cp':>3} {'val shift (fit)':>16} "
           f"{'this-quotes raw bias':>21} {'B5 frozen raw bias':>20}")
    print(hdr)
    print("-" * len(hdr))
    for cp in CHECKPOINTS:
        d = scale_only["per_cp"].get(cp)
        if d is None:
            continue
        print(f"{cp:>3} {float(fit[str(cp)]['shift']):>+16.4f} "
              f"{d['bias_raw']:>+21.3f} "
              f"{B5_FROZEN_RAW_BIAS[cp]:>+20.3f}")
    print("(val shift = mean(sim_p50 - actual) on val; raw bias = same "
          "quantity on these quotes.\n"
          " Same sign => the shift term helps; opposite sign => it hurts.)\n")

    # --- SECONDARY (recommendation only) ---
    print("=== SECONDARY (recommendation only, no gate weight) ===")
    print(f"  pooled dMAE scale-only {scale_only['pooled_dmae']:+.3f} "
          f"[{scale_only['pooled_ci'][0]:+.3f}, "
          f"{scale_only['pooled_ci'][1]:+.3f}]")
    print(f"  pooled dMAE B14-full   {b14_full['pooled_dmae']:+.3f} "
          f"[{b14_full['pooled_ci'][0]:+.3f}, "
          f"{b14_full['pooled_ci'][1]:+.3f}]")
    diff = scale_only["pooled_dmae"] - b14_full["pooled_dmae"]
    better = "scale-only" if diff < 0 else "B14-full"
    print(f"  scale-only - B14-full = {diff:+.3f}  "
          f"(more negative dMAE is better => {better} better on point "
          f"estimate)\n")

    # --- PRE-COMMITTED GATE, scale-only arm only ---
    primary_a = all(scale_only["per_cp"].get(cp, {}).get("cov_ok", False)
                    for cp in CHECKPOINTS)
    per_cp_mae = [scale_only["per_cp"].get(cp, {}).get("mae_ok", False)
                  for cp in CHECKPOINTS]
    pooled_hi = scale_only["pooled_ci"][1]
    primary_b = all(per_cp_mae) and pooled_hi < 0

    print("=== PRE-COMMITTED B15 GATE (scale-only arm) ===")
    print(f"PRIMARY-A (corrected coverage in [{COV_LO}, {COV_HI}] at all 3 "
          f"cps): {'MET' if primary_a else 'NOT MET'} "
          f"(per-cp coverage "
          f"{[round(scale_only['per_cp'][c]['cov_corr'], 3) for c in CHECKPOINTS if c in scale_only['per_cp']]}, "
          f"in-band {[scale_only['per_cp'].get(c, {}).get('cov_ok') for c in CHECKPOINTS]})")
    print(f"PRIMARY-B (corrected MAE beats naive at all 3 cps + pooled "
          f"cluster-boot CI hi < 0): {'MET' if primary_b else 'NOT MET'} "
          f"(per-cp {per_cp_mae}, pooled hi {pooled_hi:+.3f})")
    print("\n(verdict mapping is applied by the orchestrator, not here)")

    if args.self_test:
        got_pooled = (f"{b14_full['pooled_dmae']:.3f} "
                      f"[{b14_full['pooled_ci'][0]:.3f}, "
                      f"{b14_full['pooled_ci'][1]:.3f}]")
        got_cov = "/".join(
            f"{b14_full['per_cp'][c]['cov_corr']:.3f}" for c in CHECKPOINTS)
        ok_p = got_pooled == B14_SELFTEST["pooled"]
        ok_c = got_cov == B14_SELFTEST["coverage"]
        print("\n=== SELF-TEST vs B14's logged frozen-quote numbers ===")
        print(f"  B14-full pooled dMAE  expected {B14_SELFTEST['pooled']}  "
              f"got {got_pooled}  {'PASS' if ok_p else 'FAIL'}")
        print(f"  B14-full coverage     expected "
              f"{B14_SELFTEST['coverage']}  got {got_cov}  "
              f"{'PASS' if ok_c else 'FAIL'}")
        print(f"  SELF-TEST: {'PASS' if (ok_p and ok_c) else 'FAIL'}")
        if not (ok_p and ok_c):
            raise SystemExit(
                "SELF-TEST FAILED — correction transform does not reproduce "
                "B14; do NOT use this script on fresh draws.")


if __name__ == "__main__":
    main()
