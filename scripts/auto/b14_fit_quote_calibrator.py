"""B14 — fit a per-checkpoint quote-layer calibrator on VAL quotes only.

B5 (TABLED) showed the sim's in-play remaining-runs P50 beats naive
run-rate extrapolation CI-clean at every checkpoint but its P10-P90 band
is too narrow late (cp15 coverage 0.664 < 0.70 floor). This script fits a
post-hoc, per-checkpoint correction on VALIDATION quotes only:

  corrP50 = sim_p50 - shift[cp]
  corrP10 = corrP50 - s * (sim_p50 - sim_p10)
  corrP90 = corrP50 + s * (sim_p90 - sim_p50)

with
  shift[cp] = mean(sim_p50 - actual_remaining) over val rows at cp,
  scale[cp] = s in grid 0.50..3.00 step 0.01 minimizing
              |val inclusive coverage - 0.80|; ties -> smaller s.

The band half-widths are asymmetric and are widened about the CORRECTED
P50. Coverage is inclusive (corrP10 <= actual <= corrP90), matching
b5_gate_analysis.py.

The 0.80 target is the VAL fitting target only. The TEST bar (applied by
b14_gate_analysis.py against the FROZEN B5 test quotes) is [0.70, 0.90].

Run:
  uv run python scripts/auto/b14_fit_quote_calibrator.py \
      --quotes models/auto/b14/quotes_val_s47_n545.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

CHECKPOINTS = (6, 10, 15)
TARGET_COVERAGE = 0.80
SCALE_GRID = np.round(np.arange(0.50, 3.0001, 0.01), 2)


def corrected(p10, p50, p90, shift, scale):
    """Apply the B14 correction; returns (corrP10, corrP50, corrP90)."""
    c50 = p50 - shift
    c10 = c50 - scale * (p50 - p10)
    c90 = c50 + scale * (p90 - p50)
    return c10, c50, c90


def coverage(c10, actual, c90):
    return float(((c10 <= actual) & (actual <= c90)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quotes", default="models/auto/b14/quotes_val_s47_n545.json")
    ap.add_argument("--out", default="models/auto/b14/quote_calibrator.json")
    args = ap.parse_args()

    with open(args.quotes) as f:
        payload = json.load(f)
    rows = payload["rows"]
    cfg = payload.get("config", {})
    print(f"B14 quote-calibrator fit on {args.quotes}")
    print(f"  config: n_sims={cfg.get('n_sims')} seed={cfg.get('seed')} "
          f"quote_center={cfg.get('quote_center')} "
          f"usage_json={cfg.get('usage_json')} "
          f"elapsed={cfg.get('elapsed_s', 0):.1f}s")
    n_matches = len({r["match_id"] for r in rows})
    print(f"  rows: {len(rows)} from {n_matches} matches "
          f"({len(payload.get('skips', []))} matches skipped)\n")

    out = {
        "source_quotes": args.quotes,
        "target_val_coverage": TARGET_COVERAGE,
        "scale_grid": {"lo": 0.50, "hi": 3.00, "step": 0.01},
        "n_rows_total": len(rows),
        "n_matches": n_matches,
        "per_checkpoint": {},
    }

    hdr = (f"{'cp':>3} {'n':>5} {'shift':>8} {'scale':>6} "
           f"{'cov_raw':>8} {'cov_corr':>9} {'mae_raw':>8} {'mae_corr':>9}")
    print(hdr)
    print("-" * len(hdr))

    for cp in CHECKPOINTS:
        sub = [r for r in rows if r["checkpoint"] == cp]
        if not sub:
            raise SystemExit(f"no val rows at checkpoint {cp}")
        a = np.array([r["actual_remaining"] for r in sub], dtype=float)
        p10 = np.array([r["sim_p10"] for r in sub], dtype=float)
        p50 = np.array([r["sim_p50"] for r in sub], dtype=float)
        p90 = np.array([r["sim_p90"] for r in sub], dtype=float)

        shift = float((p50 - a).mean())

        best_s, best_obj = None, None
        for s in SCALE_GRID:
            c10, _, c90 = corrected(p10, p50, p90, shift, float(s))
            obj = abs(coverage(c10, a, c90) - TARGET_COVERAGE)
            # ties -> smaller s (grid is ascending; strict < keeps first)
            if best_obj is None or obj < best_obj - 1e-12:
                best_obj, best_s = obj, float(s)
        scale = best_s

        cov_raw = coverage(p10, a, p90)
        c10, c50, c90 = corrected(p10, p50, p90, shift, scale)
        cov_corr = coverage(c10, a, c90)
        mae_raw = float(np.abs(p50 - a).mean())
        mae_corr = float(np.abs(c50 - a).mean())

        out["per_checkpoint"][str(cp)] = {
            "shift": shift,
            "scale": scale,
            "n_rows": len(sub),
            "val_coverage_raw": cov_raw,
            "val_coverage_corrected": cov_corr,
            "val_mae_raw": mae_raw,
            "val_mae_corrected": mae_corr,
            "val_coverage_objective": float(best_obj),
        }
        print(f"{cp:>3} {len(sub):>5} {shift:>+8.4f} {scale:>6.2f} "
              f"{cov_raw:>8.4f} {cov_corr:>9.4f} {mae_raw:>8.4f} "
              f"{mae_corr:>9.4f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
