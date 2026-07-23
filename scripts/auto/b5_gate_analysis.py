"""B5 gate analysis — PRE-COMMITTED before any eval result exists.

Gate (per the B5 entry in research/IDEAS.md, translated to the loop's
noise discipline BEFORE the eval ran):

  GATE 1 (MAE vs naive): the sim P50 quote beats the naive
    run-rate-extrapolation baseline on remaining-runs MAE at ALL THREE
    checkpoints (point estimate), AND the pooled paired per-row
    delta |sim_err| - |naive_err| has a cluster-bootstrap CI (by match,
    2000 reps, seed 29) that excludes 0 on the favorable side (hi < 0).

  GATE 2 (calibration band): empirical P10-P90 coverage (inclusive) lies
    within [0.70, 0.90] at ALL THREE checkpoints (point estimate — the
    entry states the band directly; CIs reported as context).

Verdict mapping (program.md): both gates -> LANDED; exactly one ->
TABLED; none -> FAILED.

Run: uv run python scripts/auto/b5_gate_analysis.py \
         --quotes models/auto/b5/quotes_s43_n261.json
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
    ap.add_argument("--quotes", default="models/auto/b5/quotes_s43_n261.json")
    args = ap.parse_args()

    with open(args.quotes) as f:
        payload = json.load(f)
    rows = payload["rows"]
    cfg = payload.get("config", {})
    print(f"B5 gate analysis on {args.quotes}")
    print(f"  config: n_sims={cfg.get('n_sims')} seed={cfg.get('seed')} "
          f"quote_center={cfg.get('quote_center')} "
          f"elapsed={cfg.get('elapsed_s', 0):.0f}s")
    print(f"  rows: {len(rows)} from "
          f"{len({r['match_id'] for r in rows})} matches "
          f"({len(payload.get('skips', []))} matches skipped)\n")

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
        mean = np.array([r["sim_mean"] for r in sub])

        sim_err = np.abs(p50 - a)
        nv_err = np.abs(nv - a)
        delta = sim_err - nv_err
        cov = ((p10 <= a) & (a <= p90)).astype(float)

        by_match_delta = defaultdict(list)
        by_match_cov = defaultdict(list)
        for r, dv, cv in zip(sub, delta, cov):
            by_match_delta[r["match_id"]].append(float(dv))
            by_match_cov[r["match_id"]].append(float(cv))
            pooled_delta_by_match[r["match_id"]].append(float(dv))

        d_lo, d_hi = cluster_boot_ci(by_match_delta, np.mean)
        c_lo, c_hi = cluster_boot_ci(by_match_cov, np.mean)

        mae_sim = float(sim_err.mean())
        mae_nv = float(nv_err.mean())
        coverage = float(cov.mean())
        per_cp_ok_mae[cp] = mae_sim < mae_nv
        per_cp_ok_cov[cp] = COV_LO <= coverage <= COV_HI

        print(f"checkpoint {cp:>2} (n={len(sub)}):")
        print(f"  MAE  sim(P50) {mae_sim:7.3f}  naive {mae_nv:7.3f}  "
              f"dMAE {mae_sim - mae_nv:+7.3f} "
              f"[{d_lo:+.3f}, {d_hi:+.3f}]  "
              f"{'SIM BETTER' if mae_sim < mae_nv else 'NAIVE BETTER'}")
        print(f"  P10-P90 coverage {coverage:6.3f} "
              f"[{c_lo:.3f}, {c_hi:.3f}]  target [{COV_LO}, {COV_HI}]  "
              f"{'IN BAND' if per_cp_ok_cov[cp] else 'OUT OF BAND'}")
        print(f"  context: bias P50 {float((p50 - a).mean()):+.3f}  "
              f"mean {float((mean - a).mean()):+.3f}  "
              f"band width P90-P10 {float((p90 - p10).mean()):.1f}  "
              f"actual sd {float(a.std()):.1f}")

    pooled = np.concatenate(
        [np.asarray(v) for v in pooled_delta_by_match.values()])
    p_lo, p_hi = cluster_boot_ci(pooled_delta_by_match, np.mean)
    pooled_mean = float(pooled.mean())
    print(f"\npooled paired dMAE (sim - naive, {len(pooled)} rows, "
          f"cluster-boot by match): {pooled_mean:+.3f} "
          f"[{p_lo:+.3f}, {p_hi:+.3f}]")

    gate1 = all(per_cp_ok_mae.get(cp, False) for cp in (6, 10, 15)) \
        and p_hi < 0
    gate2 = all(per_cp_ok_cov.get(cp, False) for cp in (6, 10, 15))

    print(f"\nGATE 1 (MAE beats naive, all 3 cps + pooled CI<0): "
          f"{'MET' if gate1 else 'NOT MET'} "
          f"(per-cp {[per_cp_ok_mae.get(c) for c in (6, 10, 15)]}, "
          f"pooled hi {p_hi:+.3f})")
    print(f"GATE 2 (coverage in [{COV_LO},{COV_HI}] at all 3 cps): "
          f"{'MET' if gate2 else 'NOT MET'} "
          f"(per-cp {[per_cp_ok_cov.get(c) for c in (6, 10, 15)]})")

    verdict = ("LANDED" if (gate1 and gate2)
               else "TABLED" if (gate1 or gate2) else "FAILED")
    print(f"\nVERDICT: {verdict}")


if __name__ == "__main__":
    main()
