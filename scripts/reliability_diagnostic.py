"""Reliability diagram + Brier resolution decomposition: model vs market.

Answers the conceptual question "is the model's near-50% behaviour a
recoverable scaling artifact (under-dispersion, fixable by sharpening) or a
genuine resolution deficit (no information to spread on)?"

Three lenses, model vs de-vigged polymarket market, on the iteration set:
  1. Reliability diagram (binned predicted prob vs empirical win rate).
  2. Brier decomposition  BS = Reliability − Resolution + Uncertainty (Murphy 1973).
     Resolution is the skill term; we compare model's to the market's.
  3. Calibration slope: logistic regression of outcome on logit(prediction).
     slope ≈ 1  → calibrated; slope > 1 → under-confident (sharpenable);
     slope < 1  → over-confident. This is the Case-1-vs-Case-2 tiebreaker.

Usage:
    uv run python scripts/reliability_diagnostic.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn.linear_model import LogisticRegression

REPO = Path(__file__).resolve().parent.parent
MODEL = REPO / "models" / "xgb_match_v3_m7_production"
PRED = MODEL / "test_predictions.json"
ODDS = REPO / "betting_odds_polymarket.json"
PRED_BLAST = MODEL / "blast_golden_predictions.json"
ODDS_BLAST = REPO / "data" / "golden_blast" / "betting_odds_blast.json"
OUT = REPO / "reports" / "reliability_diagnostic.png"
NBINS = 10


def load_joined(pred_path=PRED, odds_path=ODDS):
    preds = json.load(open(pred_path))
    odds = {m["match_id"]: m for m in json.load(open(odds_path))["matches"]}
    rows = []
    for mid, p in preds.items():
        oe = odds.get(mid)
        if not oe:
            continue
        t1, t2 = p["team1"], p["team2"]
        o = oe["odds"]["winner"]
        if not (o.get(t1) and o.get(t2)):
            continue
        m1, m2 = 1.0 / o[t1], 1.0 / o[t2]
        mkt_p1 = m1 / (m1 + m2)  # de-vig (multiplicative)
        rows.append({
            "y": int(p["team1_wins"]),
            "model_p": float(p["p_team1"]),
            "mkt_p": float(mkt_p1),
            "vol": float(oe.get("polymarket_volume_usd") or 0.0),
        })
    return rows


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def logloss(p, y, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def decomposition(p, y, nbins=NBINS):
    """Murphy 1973: BS = REL - RES + UNC. Returns dict + per-bin curve."""
    N = len(y)
    obar = float(np.mean(y))
    edges = np.linspace(0.0, 1.0, nbins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, nbins - 1)
    rel = res = 0.0
    curve = []
    for k in range(nbins):
        m = idx == k
        nk = int(m.sum())
        if nk == 0:
            continue
        pbar_k = float(np.mean(p[m]))
        obar_k = float(np.mean(y[m]))
        rel += nk * (pbar_k - obar_k) ** 2
        res += nk * (obar_k - obar) ** 2
        curve.append((pbar_k, obar_k, nk))
    rel /= N
    res /= N
    unc = obar * (1 - obar)
    return {"REL": rel, "RES": res, "UNC": unc, "BS": brier(p, y),
            "LL": logloss(p, y), "std": float(np.std(p)), "obar": obar,
            "curve": curve, "n": N}


def calibration_slope(p, y, eps=1e-6):
    """Logistic regression of y on logit(p); coefficient is the cal slope."""
    z = np.log(np.clip(p, eps, 1 - eps) / np.clip(1 - p, eps, 1 - eps)).reshape(-1, 1)
    if len(np.unique(y)) < 2:
        return float("nan")
    lr = LogisticRegression(C=1e9, solver="lbfgs", max_iter=1000)
    lr.fit(z, y)
    return float(lr.coef_[0][0])


def summarize(rows, label):
    y = np.array([r["y"] for r in rows])
    mp = np.array([r["model_p"] for r in rows])
    kp = np.array([r["mkt_p"] for r in rows])
    md = decomposition(mp, y)
    kd = decomposition(kp, y)
    md["slope"] = calibration_slope(mp, y)
    kd["slope"] = calibration_slope(kp, y)
    return {"label": label, "model": md, "market": kd}


def fmt_table(summaries):
    lines = []
    hdr = (f"{'slice':<14}{'who':<8}{'n':>4}{'base':>7}{'Brier':>8}{'LogL':>8}"
           f"{'REL↓':>8}{'RES↑':>8}{'UNC':>7}{'std':>7}{'calSlope':>9}")
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for s in summaries:
        for who in ("model", "market"):
            d = s[who]
            lines.append(
                f"{s['label']:<14}{who:<8}{d['n']:>4}{d['obar']:>7.3f}"
                f"{d['BS']:>8.4f}{d['LL']:>8.4f}{d['REL']:>8.4f}{d['RES']:>8.4f}"
                f"{d['UNC']:>7.4f}{d['std']:>7.3f}{d['slope']:>9.2f}")
        lines.append("")
    return "\n".join(lines)


def _reliability_ax(ax, s, title):
    ax.plot([0, 1], [0, 1], "--", color="#888", lw=1)
    for who, color in (("model", "#0366d6"), ("market", "#e8590c")):
        curve = s[who]["curve"]
        xs = [c[0] for c in curve]; ys = [c[1] for c in curve]; ns = [c[2] for c in curve]
        ax.plot(xs, ys, "-", color=color, lw=1.3, alpha=0.55)
        ax.scatter(xs, ys, s=[max(18, n * 6) for n in ns], color=color,
                   alpha=0.75, edgecolor="white", linewidth=0.6,
                   label=f"{who}: slope {s[who]['slope']:.2f}, RES {s[who]['RES']:.3f}")
    ax.axvline(0.5, color="#ddd", lw=0.8); ax.axhline(0.5, color="#ddd", lw=0.8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_xlabel("mean predicted P(team1 wins)")
    ax.set_ylabel("empirical win rate")
    ax.set_title(title, fontsize=11)
    ax.legend(loc="upper left", fontsize=8.5)


def _dist_ax(ax, rows, title):
    mp = np.array([r["model_p"] for r in rows]); kp = np.array([r["mkt_p"] for r in rows])
    bins = np.linspace(0, 1, 26)
    ax.hist(mp, bins=bins, alpha=0.55, color="#0366d6", label=f"model (std {np.std(mp):.3f})")
    ax.hist(kp, bins=bins, alpha=0.55, color="#e8590c", label=f"market (std {np.std(kp):.3f})")
    ax.axvline(0.5, color="#888", ls="--", lw=1)
    ax.set_xlabel("predicted P(team1 wins)"); ax.set_ylabel("# matches")
    ax.set_title(title, fontsize=11); ax.legend(loc="upper right", fontsize=8.5)


def main():
    rows = load_joined()
    rows50 = [r for r in rows if r["vol"] >= 50000]
    blast = load_joined(PRED_BLAST, ODDS_BLAST)

    s_iter = summarize(rows, "iter-all")
    s_iter50 = summarize(rows50, "iter≥$50k")
    s_blast = summarize(blast, "blast")
    table = fmt_table([s_iter, s_iter50, s_blast])
    print(table)

    fig = plt.figure(figsize=(14, 12))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1.25, 0.95, 0.7], hspace=0.33, wspace=0.22)
    _reliability_ax(fig.add_subplot(gs[0, 0]), s_iter,
                    "Iteration set (IPL/intl, n=255) — model HAS resolution")
    _reliability_ax(fig.add_subplot(gs[0, 1]), s_blast,
                    "T20 Blast (n=34) — model has ~NO resolution")
    _dist_ax(fig.add_subplot(gs[1, 0]), rows,
             "Iteration forecast spread — model timid but rightly-ordered")
    _dist_ax(fig.add_subplot(gs[1, 1]), blast,
             "Blast forecast spread — model hugs 0.5, market spreads out")

    ax3 = fig.add_subplot(gs[2, :]); ax3.axis("off")
    note = (
        f"production model xgb_match_v3_m7_production vs de-vigged polymarket\n\n{table}\n"
        f"READING:  RES↑ = resolution/skill term (higher better).  REL↓ = calibration error (lower better).  BS = REL − RES + UNC.\n"
        f"cal-slope:  ≈1 calibrated · >1 under-confident (predictions too timid, SHARPENABLE) · <1 over-confident.\n"
        f"Iteration: model under-confident (slope ~1.7) but RES ≥ market → real signal, just timid.   "
        f"Blast: model RES≈{s_blast['model']['RES']:.3f} (vs market {s_blast['market']['RES']:.3f}) → nothing to be confident about."
    )
    ax3.text(0.0, 1.0, note, va="top", ha="left", family="monospace", fontsize=9.8)

    fig.suptitle("Reliability + resolution decomposition: model vs market — two regimes",
                 fontsize=15, fontweight="bold", y=0.965)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
