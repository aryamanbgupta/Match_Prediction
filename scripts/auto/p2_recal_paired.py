"""P2 recalibration re-look — paired calibrated-vs-raw statistics.

Pairs two sliced eval JSONs (same slice, same fixtures, raw vs calibrated
predictions) on match_id and reports cluster-bootstrapped paired deltas
under the I3 contract:

  - dLL  = ll_cal - ll_raw          (per match; want < 0)
  - dPNL = pnl_cal - pnl_raw        (per match; no-bet contributes 0)

Clusters come from each row's own `competition_cluster_id`. CI machinery is
the framework's `_bootstrap_ci` (seed 42, 10,000 whole-cluster resamples),
identical to `a7_conditional_threshold.py`.

Usage:
    uv run python scripts/auto/p2_recal_paired.py \\
        --raw <sliced.json> --cal <sliced.json> [--label platt-50k]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "sim_eval"))

from reslice_eval_json import _bootstrap_ci  # noqa: E402


def _rows(path: Path) -> dict:
    with open(path) as f:
        return {m["match_id"]: m for m in json.load(f)["matches"]}


def _pnl(row: dict) -> float:
    return float(row["realized_pnl"]) if row.get("bet_placed") else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--cal", type=Path, required=True)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    raw, cal = _rows(args.raw), _rows(args.cal)
    ids = sorted(set(raw) & set(cal))
    if set(raw) != set(cal):
        print(f"  WARNING: match sets differ (raw {len(raw)} / cal {len(cal)}"
              f" / paired {len(ids)})")

    d_ll, d_pnl, clusters = [], [], []
    flips = bets_raw = bets_cal = skipped = 0
    for mid in ids:
        r, c = raw[mid], cal[mid]
        if r.get("log_loss") is None or c.get("log_loss") is None:
            # Unpriced fixture: no market row, no bet on either arm.
            skipped += 1
            continue
        d_ll.append(float(c["log_loss"]) - float(r["log_loss"]))
        d_pnl.append(_pnl(c) - _pnl(r))
        clusters.append(str(r.get("competition_cluster_id") or mid))
        bets_raw += bool(r.get("bet_placed"))
        bets_cal += bool(c.get("bet_placed"))
        if (r.get("bet_team") or c.get("bet_team")) and \
                r.get("bet_team") != c.get("bet_team"):
            flips += 1

    ll_lo, ll_hi = _bootstrap_ci(d_ll, clusters=clusters)
    pnl_lo, pnl_hi = _bootstrap_ci(d_pnl, clusters=clusters)
    n = len(d_ll)
    print(f"[{args.label or 'pair'}] n={n} (skipped {skipped} unpriced) "
          f"clusters={len(set(clusters))} "
          f"bets raw/cal={bets_raw}/{bets_cal} side-flips={flips}")
    print(f"  dLL  mean {np.mean(d_ll):+.4f}  CI [{ll_lo:+.4f}, {ll_hi:+.4f}]"
          f"  ({'CI-clean' if ll_hi < 0 or ll_lo > 0 else 'straddles 0'})")
    print(f"  dPNL mean {np.mean(d_pnl):+.4f}u/match "
          f"(={np.mean(d_pnl)*100:+.2f}pp ROI-equivalent)  "
          f"CI [{pnl_lo:+.4f}, {pnl_hi:+.4f}]"
          f"  ({'CI-clean' if pnl_hi < 0 or pnl_lo > 0 else 'straddles 0'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
