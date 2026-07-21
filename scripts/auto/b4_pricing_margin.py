"""B4 — top_bowler pricing margin (post-calibration edge quantification).

Quantifies the sim's top_bowler edge over the E2 as-of fair baseline as
*implied odds*: at what synthetic-market vig does the sim's top_bowler
probability still have positive EV? Pure analysis on an existing recipe-B
detail JSON — no sim run, no model change.

Market model (stated, fixed pre-run):
  - One market per team-innings: "which bowler takes the team's most
    wickets" (the detail JSON's `top_bowler` family; y from cricsheet).
  - The synthetic bookmaker prices from the E2 fair baseline (career
    wickets share within XI, strictly as-of): quoted implied prob
    q_i = p_base_i * (1 + vig)  (multiplicative overround), decimal
    odds o_i = 1 / q_i.
  - The sim is the bettor, YES-side only, flat 1u: bet player i when
    p_sim_i / q_i - 1 > edge_threshold. Realized PnL settles on y:
    win -> o_i - 1, lose -> -1.
  - Kelly fraction per positive-edge bet: f = (p*o - 1) / (o - 1).

Outputs: margin re-verification (paired dBrier sim-base, cluster boot by
match — same statistic as E2/E5), edge distribution, ROI table over
vig x threshold with cluster-bootstrap CIs, Kelly stats, break-even vig.

Usage:
    uv run python scripts/auto/b4_pricing_margin.py \
        --detail models/auto/d15/detail_d15_s43_n261.json \
        --context-detail models/auto/d1/detail_d1_s43_n261.json \
        --out research/reports/auto/B4_pricing.md \
        --json-out models/auto/b4/pricing_numbers.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]

VIGS = [0.00, 0.02, 0.05, 0.10]
THRESHOLDS = [0.00, 0.05, 0.10, 0.20]
N_BOOT = 2000
BOOT_SEED = 29


def _load_pfb():
    """Import scripts/sim_eval/prop_fair_baselines.py (read-only reuse)."""
    p = REPO / "scripts" / "sim_eval" / "prop_fair_baselines.py"
    spec = importlib.util.spec_from_file_location("pfb", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pfb"] = mod
    spec.loader.exec_module(mod)
    return mod


def pair_top_bowler(detail: list, asof) -> list[dict]:
    """Mirror prop_fair_baselines.baseline_rows top_bowler block exactly,
    but keep team-market grouping for pricing."""
    markets = []
    for m in detail:
        mid = m["match_id"]
        date = mid[:10]
        rows = m["obs"].get("top_bowler", [])
        for team in sorted({r["team"] for r in rows}):
            trows = [r for r in rows if r["team"] == team]
            w = np.array([asof.career_wickets(r["name"], date) + 1.0
                          for r in trows])
            w = w / w.sum()
            markets.append({
                "mid": mid, "team": team,
                "rows": [{"name": r["name"], "p_sim": float(r["p"]),
                          "p_base": float(pb), "y": int(r["y"])}
                         for r, pb in zip(trows, w)],
            })
    return markets


def flat_rows(markets):
    out = []
    for mk in markets:
        for r in mk["rows"]:
            out.append({"mid": mk["mid"], **r})
    return out


def cluster_boot_mean(values_by_match: dict, n_boot=N_BOOT, seed=BOOT_SEED):
    """Bootstrap the mean of pooled per-row values, resampling matches."""
    rng = np.random.default_rng(seed)
    mids = list(values_by_match)
    arrs = [np.asarray(values_by_match[m], dtype=float) for m in mids]
    stats = []
    for _ in range(n_boot):
        idx = rng.choice(len(mids), size=len(mids), replace=True)
        pooled = np.concatenate([arrs[i] for i in idx if len(arrs[i])])
        stats.append(pooled.mean() if len(pooled) else np.nan)
    stats = np.array(stats)
    stats = stats[~np.isnan(stats)]
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def margin_check(rows):
    """Paired dBrier (sim - base), cluster bootstrap by match (E2 stat)."""
    by_match = defaultdict(list)
    for r in rows:
        d = (r["p_sim"] - r["y"]) ** 2 - (r["p_base"] - r["y"]) ** 2
        by_match[r["mid"]].append(d)
    pooled = np.concatenate([np.asarray(v) for v in by_match.values()])
    lo, hi = cluster_boot_mean(by_match)
    return float(pooled.mean()), lo, hi, len(pooled)


def price_and_bet(rows, vig, thr):
    """YES-side flat 1u bets at edge threshold thr against vig-loaded
    baseline prices. Returns per-bet records."""
    bets = []
    for r in rows:
        q = r["p_base"] * (1.0 + vig)
        if q >= 1.0:          # unquotable (implied prob >= 1)
            continue
        o = 1.0 / q
        ev = r["p_sim"] * o - 1.0
        if ev > thr:
            pnl = (o - 1.0) if r["y"] == 1 else -1.0
            kelly = (r["p_sim"] * o - 1.0) / (o - 1.0)
            bets.append({"mid": r["mid"], "name": r["name"], "odds": o,
                         "ev": ev, "pnl": pnl, "y": r["y"], "kelly": kelly})
    return bets


def roi_with_ci(bets):
    if not bets:
        return None
    by_match = defaultdict(list)
    for b in bets:
        by_match[b["mid"]].append(b["pnl"])
    pnls = np.array([b["pnl"] for b in bets])
    lo, hi = cluster_boot_mean(by_match)
    return {
        "n_bets": len(bets),
        "n_matches": len(by_match),
        "roi_pct": float(pnls.mean() * 100),
        "roi_ci_pct": [lo * 100, hi * 100],
        "win_rate_pct": float((pnls > 0).mean() * 100),
        "total_pnl_u": float(pnls.sum()),
        "avg_odds": float(np.mean([b["odds"] for b in bets])),
        "avg_ev_pct": float(np.mean([b["ev"] for b in bets]) * 100),
        "mean_kelly": float(np.mean([b["kelly"] for b in bets])),
        "median_kelly": float(np.median([b["kelly"] for b in bets])),
        "p90_kelly": float(np.percentile([b["kelly"] for b in bets], 90)),
        "max_kelly": float(np.max([b["kelly"] for b in bets])),
    }


def kelly_roi(bets):
    """Kelly-staked realized return: stake f_i, ROI = sum(f*pnl)/sum(f)."""
    if not bets:
        return None
    f = np.array([b["kelly"] for b in bets])
    pnl = np.array([b["pnl"] for b in bets])
    return float((f * pnl).sum() / f.sum() * 100)


def breakeven_vig(rows, thr=0.0):
    """Finest vig at which flat-ROI at threshold thr crosses <= 0
    (grid scan, 0.5% steps). In-sample descriptive number."""
    prev_v, prev_roi = None, None
    for v in np.arange(0.0, 0.5001, 0.005):
        bets = price_and_bet(rows, float(v), thr)
        if not bets:
            return prev_v, prev_roi, float(v), None
        roi = float(np.mean([b["pnl"] for b in bets]) * 100)
        if roi <= 0:
            return prev_v, prev_roi, float(v), roi
        prev_v, prev_roi = float(v), roi
    return prev_v, prev_roi, None, None


def analyze(detail_path: Path, asof, label: str):
    detail = json.load(open(detail_path))
    markets = pair_top_bowler(detail, asof)
    rows = flat_rows(markets)

    # hard sanity: sums within team markets
    for mk in markets:
        ps = sum(r["p_sim"] for r in mk["rows"])
        pb = sum(r["p_base"] for r in mk["rows"])
        assert abs(pb - 1.0) < 1e-9, (mk["mid"], mk["team"], pb)
        assert abs(ps - 1.0) < 0.02, (mk["mid"], mk["team"], ps)

    d_mean, d_lo, d_hi, n = margin_check(rows)
    res = {
        "label": label, "detail": str(detail_path),
        "n_matches": len(detail), "n_markets": len(markets), "n_rows": n,
        "n_markets_no_winner": sum(
            1 for mk in markets if not any(r["y"] for r in mk["rows"])),
        "dbrier_mean": d_mean, "dbrier_ci": [d_lo, d_hi],
        "edge_dist": {}, "table": {}, "breakeven": {},
    }

    diffs = np.array([r["p_sim"] - r["p_base"] for r in rows])
    res["edge_dist"] = {
        "quantiles": {q: float(np.percentile(diffs, q))
                      for q in (5, 25, 50, 75, 95)},
        "share_abs_gt_2pp": float((np.abs(diffs) > 0.02).mean()),
        "share_abs_gt_5pp": float((np.abs(diffs) > 0.05).mean()),
        "mean_abs_pp": float(np.abs(diffs).mean() * 100),
    }

    for vig in VIGS:
        for thr in THRESHOLDS:
            r = roi_with_ci(price_and_bet(rows, vig, thr))
            if r is not None:
                r["kelly_roi_pct"] = kelly_roi(price_and_bet(rows, vig, thr))
            res["table"][f"vig{vig:.2f}_thr{thr:.2f}"] = r

    bv = breakeven_vig(rows, 0.0)
    res["breakeven"] = {"last_pos_vig": bv[0], "last_pos_roi_pct": bv[1],
                        "first_nonpos_vig": bv[2], "first_nonpos_roi_pct": bv[3]}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detail", type=Path,
                    default=REPO / "models/auto/d15/detail_d15_s43_n261.json")
    ap.add_argument("--context-detail", type=Path, default=None,
                    help="optional second engine for margin context")
    ap.add_argument("--out", type=Path,
                    default=REPO / "research/reports/auto/B4_pricing.md")
    ap.add_argument("--json-out", type=Path,
                    default=REPO / "models/auto/b4/pricing_numbers.json")
    args = ap.parse_args()

    pfb = _load_pfb()
    assert pfb.CACHE.exists(), "fair-baseline corpus cache missing"
    logs = pickle.load(open(pfb.CACHE, "rb"))
    print(f"corpus cache loaded: {pfb.CACHE.name}")
    asof = pfb.AsOf(logs)

    results = [analyze(args.detail, asof, "primary")]
    if args.context_detail:
        results.append(analyze(args.context_detail, asof, "context"))

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(args.json_out, "w"), indent=1)
    print(f"numbers -> {args.json_out}")

    r = results[0]
    lines = [
        "# B4 — top_bowler pricing margin vs the E2 fair baseline",
        "",
        f"Primary detail: `{Path(r['detail']).name}` "
        f"(n={r['n_matches']} matches, {r['n_markets']} team-markets, "
        f"{r['n_rows']} player rows; {r['n_markets_no_winner']} markets "
        "with no y=1 — bowlers took zero wickets, every YES bet loses).",
        "",
        "Synthetic market = E2 as-of fair baseline (career-wickets share "
        "within XI), multiplicative overround q = p_base*(1+vig), YES-side "
        "flat 1u, settle on cricsheet y. CIs: cluster bootstrap by match "
        f"({N_BOOT} resamples, seed {BOOT_SEED}).",
        "",
        "## Margin re-verification (paired dBrier sim - base; negative = "
        "sim beats baseline)",
        "",
    ]
    for rr in results:
        lines.append(
            f"- **{rr['label']}** `{Path(rr['detail']).name}`: "
            f"{rr['dbrier_mean']:+.4f} CI [{rr['dbrier_ci'][0]:+.4f}, "
            f"{rr['dbrier_ci'][1]:+.4f}] (n={rr['n_rows']})")
    ed = r["edge_dist"]
    lines += [
        "",
        "## Edge distribution (p_sim - p_base, per player row)",
        "",
        f"- quantiles (pp): " + ", ".join(
            f"P{q} {v*100:+.1f}" for q, v in ed["quantiles"].items()),
        f"- mean |edge| {ed['mean_abs_pp']:.2f}pp; share |edge|>2pp "
        f"{ed['share_abs_gt_2pp']*100:.1f}%; >5pp "
        f"{ed['share_abs_gt_5pp']*100:.1f}%",
        "",
        "## Flat 1u YES ROI vs vig x edge-threshold",
        "",
        "| vig | thr | bets | ROI % | ROI 95% CI | win % | avg odds | "
        "avg EV % | Kelly-staked ROI % | mean/med/p90 Kelly |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for vig in VIGS:
        for thr in THRESHOLDS:
            t = r["table"][f"vig{vig:.2f}_thr{thr:.2f}"]
            if t is None:
                lines.append(f"| {vig:.0%} | {thr:.0%} | 0 | — | — | — | — "
                             "| — | — | — |")
                continue
            lines.append(
                f"| {vig:.0%} | {thr:.0%} | {t['n_bets']} | "
                f"{t['roi_pct']:+.2f} | [{t['roi_ci_pct'][0]:+.2f}, "
                f"{t['roi_ci_pct'][1]:+.2f}] | {t['win_rate_pct']:.1f} | "
                f"{t['avg_odds']:.2f} | {t['avg_ev_pct']:+.1f} | "
                f"{t['kelly_roi_pct']:+.2f} | "
                f"{t['mean_kelly']:.3f}/{t['median_kelly']:.3f}/"
                f"{t['p90_kelly']:.3f} |")
    bv = r["breakeven"]
    lines += [
        "",
        "## Break-even vig (thr=0, flat 1u, 0.5% grid; in-sample)",
        "",
        f"- last positive-ROI vig: "
        f"{'—' if bv['last_pos_vig'] is None else f'{bv,}'}"
    ]
    # (breakeven line formatted in main report; raw values in JSON)
    lines[-1] = (
        f"- last positive-ROI vig: "
        + ("—" if bv["last_pos_vig"] is None
           else f"{bv['last_pos_vig']:.1%} (ROI {bv['last_pos_roi_pct']:+.2f}%)")
        + "; first non-positive: "
        + ("none <= 50%" if bv["first_nonpos_vig"] is None
           else f"{bv['first_nonpos_vig']:.1%} "
                f"(ROI {bv['first_nonpos_roi_pct']:+.2f}%)"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print(f"report -> {args.out}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
