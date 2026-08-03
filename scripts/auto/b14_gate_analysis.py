"""B14 gate analysis — PRE-COMMITTED before any corrected TEST number exists.

Applies the VAL-fit per-checkpoint quote calibrator
(models/auto/b14/quote_calibrator.json, produced by
b14_fit_quote_calibrator.py from val quotes only) to the FROZEN B5 TEST
quotes (models/auto/b5/quotes_s43_n261.json — never regenerated), then
evaluates the pre-committed B14 gate pair:

  GATE 1' (no-regress): the CORRECTED P50 quote still beats the naive
    run-rate-extrapolation baseline on remaining-runs MAE at ALL THREE
    checkpoints (point estimate), AND the pooled paired per-row delta
    |corr_err| - |naive_err| has a cluster-bootstrap CI (by match, 2000
    reps, seed 29 — the exact b5_gate_analysis.py construction) with
    hi < 0.

  GATE 2' (calibration): the CORRECTED inclusive P10-P90 coverage lies
    within [0.70, 0.90] at ALL THREE checkpoints (point estimates; CIs
    reported as context).

The naive baseline is NOT corrected — it is the fixed comparator. The
0.80 coverage target used when fitting is a VAL-only target; the TEST bar
is [0.70, 0.90].

Verdict mapping is applied by the orchestrator, not by this script.

Run: uv run python scripts/auto/b14_gate_analysis.py
"""
import argparse
import json
from collections import defaultdict

import numpy as np

BOOT_REPS = 2000
BOOT_SEED = 29
COV_LO, COV_HI = 0.70, 0.90


def cluster_boot_ci(values_by_match, stat_fn, reps=BOOT_REPS, seed=BOOT_SEED):
    """Percentile 95% CI of stat_fn over match-level resamples."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrator",
                    default="models/auto/b14/quote_calibrator.json")
    ap.add_argument("--quotes", default="models/auto/b5/quotes_s43_n261.json")
    args = ap.parse_args()

    with open(args.calibrator) as f:
        cal = json.load(f)
    with open(args.quotes) as f:
        payload = json.load(f)
    rows = payload["rows"]
    cfg = payload.get("config", {})

    print(f"B14 gate analysis")
    print(f"  calibrator: {args.calibrator} "
          f"(fit on {cal.get('source_quotes')}, "
          f"val target coverage {cal.get('target_val_coverage')})")
    print(f"  TEST quotes: {args.quotes}")
    print(f"  config: n_sims={cfg.get('n_sims')} seed={cfg.get('seed')} "
          f"quote_center={cfg.get('quote_center')} "
          f"elapsed={cfg.get('elapsed_s', 0):.0f}s")
    print(f"  rows: {len(rows)} from "
          f"{len({r['match_id'] for r in rows})} matches "
          f"({len(payload.get('skips', []))} matches skipped)")
    fit = cal["per_checkpoint"]
    print("  correction applied (val-fit): " + "  ".join(
        f"cp{c}: shift {fit[c]['shift']:+.4f} scale {fit[c]['scale']:.2f}"
        for c in sorted(fit, key=int)) + "\n")

    cps = sorted({r["checkpoint"] for r in rows})
    per_cp_ok_mae, per_cp_ok_cov = {}, {}
    pooled_delta_by_match = defaultdict(list)

    for cp in cps:
        sub = [r for r in rows if r["checkpoint"] == cp]
        a = np.array([r["actual_remaining"] for r in sub], dtype=float)
        p50 = np.array([r["sim_p50"] for r in sub])
        nv = np.array([r["naive_remaining"] for r in sub])
        p10 = np.array([r["sim_p10"] for r in sub])
        p90 = np.array([r["sim_p90"] for r in sub])

        shift = float(fit[str(cp)]["shift"])
        scale = float(fit[str(cp)]["scale"])
        c50 = p50 - shift
        c10 = c50 - scale * (p50 - p10)
        c90 = c50 + scale * (p90 - p50)

        corr_err = np.abs(c50 - a)
        raw_err = np.abs(p50 - a)
        nv_err = np.abs(nv - a)
        delta = corr_err - nv_err          # GATE 1' statistic
        raw_delta = raw_err - nv_err       # B5 context
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

        d_lo, d_hi = cluster_boot_ci(by_match_delta, np.mean)
        rd_lo, rd_hi = cluster_boot_ci(by_match_raw_delta, np.mean)
        c_lo, c_hi = cluster_boot_ci(by_match_cov, np.mean)
        cr_lo, cr_hi = cluster_boot_ci(by_match_cov_raw, np.mean)

        mae_corr = float(corr_err.mean())
        mae_raw = float(raw_err.mean())
        mae_nv = float(nv_err.mean())
        coverage_corr = float(cov_corr.mean())
        coverage_raw = float(cov_raw.mean())
        per_cp_ok_mae[cp] = mae_corr < mae_nv
        per_cp_ok_cov[cp] = COV_LO <= coverage_corr <= COV_HI

        print(f"checkpoint {cp:>2} (n={len(sub)}):")
        print(f"  MAE  corr(P50) {mae_corr:7.3f}  naive {mae_nv:7.3f}  "
              f"dMAE {mae_corr - mae_nv:+7.3f} "
              f"[{d_lo:+.3f}, {d_hi:+.3f}]  "
              f"{'CORRECTED BETTER' if mae_corr < mae_nv else 'NAIVE BETTER'}")
        print(f"  MAE  raw (P50) {mae_raw:7.3f}  naive {mae_nv:7.3f}  "
              f"dMAE {mae_raw - mae_nv:+7.3f} "
              f"[{rd_lo:+.3f}, {rd_hi:+.3f}]   (uncorrected context)")
        print(f"  P10-P90 coverage corr {coverage_corr:6.3f} "
              f"[{c_lo:.3f}, {c_hi:.3f}]  target [{COV_LO}, {COV_HI}]  "
              f"{'IN BAND' if per_cp_ok_cov[cp] else 'OUT OF BAND'}")
        print(f"  P10-P90 coverage raw  {coverage_raw:6.3f} "
              f"[{cr_lo:.3f}, {cr_hi:.3f}]   (uncorrected context)")
        print(f"  context: bias corr P50 {float((c50 - a).mean()):+.3f}  "
              f"bias raw P50 {float((p50 - a).mean()):+.3f}  "
              f"band width corr {float((c90 - c10).mean()):.1f}  "
              f"raw {float((p90 - p10).mean()):.1f}  "
              f"actual sd {float(a.std()):.1f}")

    pooled = np.concatenate(
        [np.asarray(v) for v in pooled_delta_by_match.values()])
    p_lo, p_hi = cluster_boot_ci(pooled_delta_by_match, np.mean)
    pooled_mean = float(pooled.mean())
    print(f"\npooled paired dMAE (corrected - naive, {len(pooled)} rows, "
          f"cluster-boot by match): {pooled_mean:+.3f} "
          f"[{p_lo:+.3f}, {p_hi:+.3f}]")

    gate1 = all(per_cp_ok_mae.get(cp, False) for cp in (6, 10, 15)) \
        and p_hi < 0
    gate2 = all(per_cp_ok_cov.get(cp, False) for cp in (6, 10, 15))

    print(f"\nGATE 1' (corrected MAE beats naive at all 3 cps + pooled "
          f"cluster-boot CI hi < 0): {'MET' if gate1 else 'NOT MET'} "
          f"(per-cp {[per_cp_ok_mae.get(c) for c in (6, 10, 15)]}, "
          f"pooled hi {p_hi:+.3f})")
    print(f"GATE 2' (corrected coverage in [{COV_LO},{COV_HI}] at all 3 "
          f"cps): {'MET' if gate2 else 'NOT MET'} "
          f"(per-cp {[per_cp_ok_cov.get(c) for c in (6, 10, 15)]})")
    print("\n(verdict mapping is applied by the orchestrator, not here)")


if __name__ == "__main__":
    main()
