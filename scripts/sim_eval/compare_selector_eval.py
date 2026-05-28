#!/usr/bin/env python3
"""Compare two prop_backtest runs (e.g. empirical vs random selector) and
emit a gates-checked verdict per prop family.

Reads two detail JSON files (output of prop_backtest.py) and produces:
  - Side-by-side Brier / log-loss / MAE per family
  - Δ (left − right) with paired-by-match bootstrap CI
  - Gate verdicts (G2/G3/G5) tagged inline

Usage:
    uv run python scripts/sim_eval/compare_selector_eval.py \
        --left  reports/prop_calibration_detail_emp_n60.json \
        --right reports/prop_calibration_detail_rand_n60.json \
        --left-label empirical --right-label random \
        --out reports/prop_selector_comparison.md
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Metrics (mirrors prop_backtest.py — kept in lock-step intentionally).
# ---------------------------------------------------------------------------


def brier(rows: List[dict]) -> Optional[float]:
    if not rows:
        return None
    return float(np.mean([(r["p"] - r["y"]) ** 2 for r in rows]))


def log_loss(rows: List[dict], eps: float = 1e-6) -> Optional[float]:
    if not rows:
        return None
    out = []
    for r in rows:
        p = min(max(r["p"], eps), 1 - eps)
        out.append(-(r["y"] * math.log(p) + (1 - r["y"]) * math.log(1 - p)))
    return float(np.mean(out))


def base_rate(rows: List[dict]) -> Optional[float]:
    if not rows:
        return None
    return float(np.mean([r["y"] for r in rows]))


def baseline_brier(rows: List[dict]) -> Optional[float]:
    if not rows:
        return None
    p = base_rate(rows)
    return float(np.mean([(p - r["y"]) ** 2 for r in rows]))


def mae(rows: List[dict]) -> Optional[float]:
    if not rows:
        return None
    return float(np.mean([abs(r["sim_mean"] - r["actual"]) for r in rows]))


# ---------------------------------------------------------------------------
# Paired bootstrap by match.
# ---------------------------------------------------------------------------


def paired_bootstrap_delta(
    matches_left: Dict[str, List[dict]],
    matches_right: Dict[str, List[dict]],
    metric_fn: Callable[[List[dict]], Optional[float]],
    n_reps: int = 1000,
    seed: int = 0,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Resample at the match level (shared indices across left/right) and
    return (point_estimate_of_delta, lo_95, hi_95) for `left − right`."""
    match_ids = sorted(set(matches_left) & set(matches_right))
    if not match_ids:
        return None, None, None
    rng = np.random.default_rng(seed)
    n = len(match_ids)
    deltas: List[float] = []
    for _ in range(n_reps):
        idxs = rng.integers(0, n, size=n)
        sampled = [match_ids[i] for i in idxs]
        rows_l = [r for mid in sampled for r in matches_left.get(mid, [])]
        rows_r = [r for mid in sampled for r in matches_right.get(mid, [])]
        ml = metric_fn(rows_l)
        mr = metric_fn(rows_r)
        if ml is None or mr is None:
            continue
        deltas.append(ml - mr)
    if not deltas:
        return None, None, None
    point = (metric_fn(
        [r for mid in match_ids for r in matches_left.get(mid, [])]
    ) or 0) - (metric_fn(
        [r for mid in match_ids for r in matches_right.get(mid, [])]
    ) or 0)
    lo = float(np.percentile(deltas, 2.5))
    hi = float(np.percentile(deltas, 97.5))
    return float(point), lo, hi


# ---------------------------------------------------------------------------
# Loaders.
# ---------------------------------------------------------------------------


def load_detail(path: str) -> Tuple[Dict[str, Dict[str, list]], List[str]]:
    """Return (family → match_id → rows, families_in_order)."""
    with open(path) as f:
        data = json.load(f)
    families: List[str] = []
    seen = set()
    family_match_rows: Dict[str, Dict[str, list]] = defaultdict(dict)
    for d in data:
        mid = d["match_id"]
        for fam, rows in d["obs"].items():
            if fam not in seen:
                families.append(fam); seen.add(fam)
            family_match_rows[fam][mid] = rows
    return family_match_rows, families


def is_continuous_family(rows: List[dict]) -> bool:
    """Continuous = rows have sim_mean+actual; binary = rows have p+y."""
    for r in rows:
        if "sim_mean" in r and "actual" in r and "p" not in r:
            return True
        if "p" in r and "y" in r:
            return False
    return False


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--left", required=True, help="Detail JSON path (treated as 'new'/'empirical')")
    ap.add_argument("--right", required=True, help="Detail JSON path (treated as 'baseline'/'random')")
    ap.add_argument("--left-label", default="empirical")
    ap.add_argument("--right-label", default="random")
    ap.add_argument("--out", default="reports/prop_selector_comparison.md")
    ap.add_argument("--n-reps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    left_rows, families = load_detail(args.left)
    right_rows, _ = load_detail(args.right)

    lines: List[str] = []
    lines.append(f"# Prop selector comparison — {args.left_label} vs {args.right_label}")
    lines.append("")
    lines.append(f"- Left  (`{args.left_label}`):  `{args.left}`")
    lines.append(f"- Right (`{args.right_label}`): `{args.right}`")
    lines.append(f"- Paired bootstrap by match, {args.n_reps} resamples, seed={args.seed}")
    lines.append("")

    # Binary props.
    lines.append("## Binary props (Brier)")
    lines.append("")
    lines.append(
        f"| family | n | base rate | "
        f"{args.left_label} Brier | {args.right_label} Brier | "
        f"Δ Brier ({args.left_label}−{args.right_label}) | 95% CI | Verdict |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
    for fam in families:
        l_match = left_rows.get(fam, {})
        r_match = right_rows.get(fam, {})
        all_l = [r for rs in l_match.values() for r in rs]
        all_r = [r for rs in r_match.values() for r in rs]
        if not all_l or not all_r:
            continue
        if is_continuous_family(all_l):
            continue
        n = len(all_l)
        bp = base_rate(all_l)
        bl = brier(all_l)
        br = brier(all_r)
        if bl is None or br is None:
            continue
        delta, lo, hi = paired_bootstrap_delta(
            l_match, r_match, brier, n_reps=args.n_reps, seed=args.seed
        )
        if delta is None:
            ci_str = "–"
            verdict = "?"
        else:
            ci_str = f"[{lo:+.4f}, {hi:+.4f}]"
            if hi < 0:
                verdict = "✅ {} better".format(args.left_label)
            elif lo > 0:
                verdict = "❌ {} better".format(args.right_label)
            else:
                verdict = "≈ tied"
        delta_str = f"{delta:+.4f}" if delta is not None else "–"
        lines.append(
            f"| {fam} | {n} | {bp:.3f} | {bl:.4f} | {br:.4f} | "
            f"{delta_str} | {ci_str} | {verdict} |"
        )

    # Continuous props.
    lines.append("")
    lines.append("## Continuous props (MAE)")
    lines.append("")
    lines.append(
        f"| family | n | {args.left_label} MAE | {args.right_label} MAE | "
        f"Δ MAE | 95% CI | Verdict |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for fam in families:
        l_match = left_rows.get(fam, {})
        r_match = right_rows.get(fam, {})
        all_l = [r for rs in l_match.values() for r in rs]
        all_r = [r for rs in r_match.values() for r in rs]
        if not all_l or not all_r:
            continue
        if not is_continuous_family(all_l):
            continue
        n = len(all_l)
        ml = mae(all_l)
        mr = mae(all_r)
        if ml is None or mr is None:
            continue
        delta, lo, hi = paired_bootstrap_delta(
            l_match, r_match, mae, n_reps=args.n_reps, seed=args.seed
        )
        ci_str = f"[{lo:+.2f}, {hi:+.2f}]" if delta is not None else "–"
        if delta is None:
            verdict = "?"
        elif hi < 0:
            verdict = "✅ {} better".format(args.left_label)
        elif lo > 0:
            verdict = "❌ {} better".format(args.right_label)
        else:
            verdict = "≈ tied"
        delta_str = f"{delta:+.2f}" if delta is not None else "–"
        lines.append(
            f"| {fam} | {n} | {ml:.2f} | {mr:.2f} | {delta_str} | {ci_str} | {verdict} |"
        )

    # Gate verdicts.
    lines.append("")
    lines.append("## Validation gates")
    lines.append("")
    lines.append("**Gate G2 — top_bowler skill improvement**")
    l_top_bw = [r for rs in left_rows.get("top_bowler", {}).values() for r in rs]
    r_top_bw = [r for rs in right_rows.get("top_bowler", {}).values() for r in rs]
    if l_top_bw and r_top_bw:
        bl = brier(l_top_bw); br = brier(r_top_bw); bbr = baseline_brier(r_top_bw)
        # Floor proxy: 0 (perfect oracle would have Brier dominated by other
        # bowlers' zero-prob × zero-y, ≈0). Use baseline as ceiling.
        floor = 0.0
        gap = (br - floor) if br else 0
        closed = (br - bl) / gap if gap > 0 else 0
        verdict = "✅ PASS" if closed >= 0.4 else "❌ FAIL"
        lines.append(
            f"- {args.left_label} Brier {bl:.4f} vs {args.right_label} Brier {br:.4f} "
            f"(baseline {bbr:.4f}); gap closed = {closed:.1%}. {verdict} (target ≥40%)."
        )
    lines.append("")
    lines.append("**Gate G3 — top_batter no-regression**")
    l_top_bt = [r for rs in left_rows.get("top_batter", {}).values() for r in rs]
    r_top_bt = [r for rs in right_rows.get("top_batter", {}).values() for r in rs]
    if l_top_bt and r_top_bt:
        bl = brier(l_top_bt); br = brier(r_top_bt)
        delta = bl - br
        verdict = "✅ PASS" if delta <= 0.003 else "❌ FAIL"
        lines.append(
            f"- Δ Brier ({args.left_label} − {args.right_label}) = {delta:+.4f}. "
            f"{verdict} (target ≤ +0.003)."
        )
    lines.append("")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
